#!/usr/bin/env python3
"""Reproducible external benchmark on AmazonScience/MASSIVE Russian intent data.

Downloads the public parquet files at a pinned Hub commit. Only the official
`train` partition is passed into AMORPH; the official `test` partition remains
unseen until after the internal selector and acceptance gate have completed.

Install the core package plus benchmark readers first:
  python -m pip install -e '.[test]'
  python -m pip install pandas pyarrow huggingface_hub
For the optional E5 route, also run:
  python -m pip install -e '.[embeddings]'

Run:
  python scripts/benchmark_massive_ru.py --output-dir benchmark_runs/massive_ru
  python scripts/benchmark_massive_ru.py --with-e5 --output-dir benchmark_runs/massive_ru_e5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATASET_ID = "AmazonScience/massive"
# Pinned revision containing ru-RU/massive-{train,validation,test}.parquet.
DATASET_REVISION = "c4aa61e89895160619e18a94de72f19b99e72ca2"
DATASET_LICENSE = "CC BY 4.0"
DATASET_CARD = "https://huggingface.co/datasets/AmazonScience/massive"
E5_MODEL_ID = "intfloat/multilingual-e5-small"
E5_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def download_parquet(filename: str, cache_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Missing benchmark dependency. Install `huggingface_hub`.") from exc
    path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=filename,
        revision=DATASET_REVISION,
        cache_dir=str(cache_dir),
    )
    return Path(path)


def load_frame(path: Path):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Missing benchmark dependency. Install `pandas` and `pyarrow`.") from exc
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="benchmark_runs/massive_ru")
    parser.add_argument("--with-e5", action="store_true", help="Also train/evaluate multilingual-e5-small embeddings + logistic regression; downloads model weights.")
    parser.add_argument("--max-runtime-seconds", type=float, default=7200.0)
    parser.add_argument("--expected-queries", type=int, default=1000)
    parser.add_argument("--compute-eur-per-hour", type=float, default=0.0)
    parser.add_argument("--human-hours", type=float, default=0.0)
    parser.add_argument("--human-eur-per-hour", type=float, default=0.0)
    parser.add_argument("--e5-classifier-c", type=float, default=1.0,
                        help="E5 LogisticRegression C (default is current C=1.0)")
    parser.add_argument("--e5-class-weight", choices=("none", "balanced"), default="none",
                        help="E5 LogisticRegression class weight for a controlled A/B experiment")
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "hf_cache"

    train_source = download_parquet("ru-RU/massive-train.parquet", cache_dir)
    test_source = download_parquet("ru-RU/massive-test.parquet", cache_dir)
    train_frame = load_frame(train_source)
    test_frame = load_frame(test_source)

    needed = {"utt", "intent", "worker_id"}
    for label, frame in (("train", train_frame), ("test", test_frame)):
        missing = sorted(needed - set(frame.columns))
        if missing:
            raise RuntimeError(f"MASSIVE {label} file is missing required columns: {missing}")

    def clean(frame):
        frame = frame[["utt", "intent", "worker_id"]].copy()
        for name in ("utt", "intent", "worker_id"):
            frame[name] = frame[name].fillna("").astype(str).str.strip()
        frame = frame[(frame["utt"] != "") & (frame["intent"] != "") & (frame["worker_id"] != "")]
        return frame.reset_index(drop=True)

    train_frame = clean(train_frame)
    test_frame = clean(test_frame)
    if train_frame["intent"].nunique() < 2 or test_frame["intent"].nunique() < 2:
        raise RuntimeError("The downloaded dataset does not contain enough intent classes.")

    # MASSIVE can contain exactly normalized utterances assigned to different intents.
    # Do not weaken AMORPH's fail-closed duplicate validation or arbitrarily relabel data:
    # remove every training row belonging to a conflicting normalized-text group and record
    # counts plus hashes. Keep the official test partition untouched for report-only evaluation.
    train_rows_after_required_field_filter = len(train_frame)
    normalized_train_text = (
        train_frame["utt"].str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )
    conflict_check = train_frame.assign(_normalized_text=normalized_train_text)
    labels_per_normalized_text = conflict_check.groupby("_normalized_text")["intent"].nunique()
    conflicting_text_keys = sorted(
        labels_per_normalized_text[labels_per_normalized_text > 1].index.tolist()
    )
    conflicting_key_set = set(conflicting_text_keys)
    conflicting_mask = normalized_train_text.isin(conflicting_key_set)
    removed_conflict_rows = train_frame.loc[conflicting_mask, ["utt", "intent", "worker_id"]]
    conflict_keys_sha256 = hashlib.sha256(
        "\n".join(conflicting_text_keys).encode("utf-8")
    ).hexdigest()
    conflict_rows_canonical = "\n".join(
        f"{normalized}\t{label}\t{worker_id}"
        for normalized, label, worker_id in zip(
            normalized_train_text[conflicting_mask].tolist(),
            removed_conflict_rows["intent"].tolist(),
            removed_conflict_rows["worker_id"].tolist(),
        )
    )
    training_conflicting_text_filter = {
        "policy": "drop_all_training_rows_for_conflicting_normalized_text_groups",
        "normalization": "collapse_whitespace_then_strip_then_casefold",
        "ambiguous_normalized_text_groups_removed": len(conflicting_text_keys),
        "training_rows_removed": int(conflicting_mask.sum()),
        "training_rows_retained": int((~conflicting_mask).sum()),
        "rows_after_required_field_filter_before_conflict_removal": train_rows_after_required_field_filter,
        "conflicting_text_keys_sha256": conflict_keys_sha256,
        "removed_training_rows_sha256": hashlib.sha256(
            conflict_rows_canonical.encode("utf-8")
        ).hexdigest(),
    }
    train_frame = train_frame.loc[~conflicting_mask].reset_index(drop=True)
    if train_frame["intent"].nunique() < 2:
        raise RuntimeError(
            "Conflicting-text removal left fewer than two training intent classes."
        )

    # Gate protocol 1+3: the configured acceptance gate needs five rows/class.
    # Requiring >=4x that many unique texts and four worker groups is a support
    # heuristic, not proof that the grouped MILP is feasible or statistically powered.
    # The MILP later enforces actual per-split row floors.
    min_gate_examples_per_class = 5
    minimum_train_examples_per_class = max(5, 4 * min_gate_examples_per_class)
    minimum_worker_groups_per_class = 4
    support_normalized_text = (
        train_frame["utt"].str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )
    unique_support_rows = train_frame.assign(
        _normalized_text=support_normalized_text
    ).drop_duplicates(subset=["_normalized_text"], keep="first")
    counts_after_conflict_filter = unique_support_rows["intent"].value_counts()
    worker_groups_after_conflict_filter = (
        unique_support_rows.groupby("intent")["worker_id"].nunique()
    )
    low_count_labels = set(
        counts_after_conflict_filter[
            counts_after_conflict_filter < minimum_train_examples_per_class
        ].index.tolist()
    )
    low_group_labels = set(
        worker_groups_after_conflict_filter[
            worker_groups_after_conflict_filter < minimum_worker_groups_per_class
        ].index.tolist()
    )
    low_support_labels = sorted(low_count_labels | low_group_labels)
    low_support_mask = train_frame["intent"].isin(low_support_labels)
    low_support_rows = train_frame.loc[low_support_mask, ["utt", "intent", "worker_id"]]
    low_support_canonical = "\n".join(
        f"{text}\t{label}\t{worker_id}"
        for text, label, worker_id in low_support_rows.itertuples(index=False, name=None)
    )
    training_low_support_class_filter = {
        "policy": "exclude_classes_below_gate_support_floor_4x_min_gate_texts_and_4_worker_groups",
        "normalization": "collapse_whitespace_then_strip_then_casefold",
        "duplicate_policy_for_support_counts": "first occurrence retained, matching AMORPH load_records",
        "minimum_unique_text_examples_per_class": minimum_train_examples_per_class,
        "minimum_distinct_worker_groups_per_class": minimum_worker_groups_per_class,
        "excluded_labels": low_support_labels,
        "excluded_label_unique_text_counts": {
            label: int(counts_after_conflict_filter.get(label, 0))
            for label in low_support_labels
        },
        "excluded_label_worker_group_counts": {
            label: int(worker_groups_after_conflict_filter.get(label, 0))
            for label in low_support_labels
        },
        "labels_below_unique_text_floor": sorted(low_count_labels),
        "labels_below_worker_group_floor": sorted(low_group_labels),
        "training_rows_removed": int(low_support_mask.sum()),
        "training_rows_retained": int((~low_support_mask).sum()),
        "removed_training_rows_sha256": hashlib.sha256(
            low_support_canonical.encode("utf-8")
        ).hexdigest(),
    }
    train_frame = train_frame.loc[~low_support_mask].reset_index(drop=True)
    if train_frame["intent"].nunique() < 2:
        raise RuntimeError(
            "Filtering under-supported training classes left fewer than two classes."
        )

    train_label_set = set(train_frame["intent"])
    test_label_set = set(test_frame["intent"])
    test_labels_without_training_support = sorted(test_label_set - train_label_set)
    train_labels_absent_from_test = sorted(train_label_set - test_label_set)

    train_csv = data_dir / "massive_ru_train.csv"
    test_csv = data_dir / "massive_ru_official_test.csv"
    train_frame.rename(columns={"utt": "text", "intent": "label", "worker_id": "group_id"})[["text", "label", "group_id"]].to_csv(train_csv, index=False, encoding="utf-8")
    test_frame.rename(columns={"utt": "text", "intent": "label"})[["text", "label"]].to_csv(test_csv, index=False, encoding="utf-8")

    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "locale": "ru-RU",
        "license": DATASET_LICENSE,
        "dataset_card_url": DATASET_CARD,
        "attribution": "Amazon Science / AmazonScience/MASSIVE; see the dataset card for required citation and attribution.",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "train_parquet": {"path": str(train_source), "sha256": sha256_file(train_source), "rows_after_required-field_filter": train_rows_after_required_field_filter},
            "test_parquet": {"path": str(test_source), "sha256": sha256_file(test_source), "rows_after_required-field_filter": len(test_frame)},
        },
        "prepared_files": {
            "train_csv": {"path": str(train_csv), "sha256": sha256_file(train_csv), "rows": len(train_frame)},
            "official_test_csv": {"path": str(test_csv), "sha256": sha256_file(test_csv), "rows": len(test_frame)},
        },
        "training_conflicting_text_filter": training_conflicting_text_filter,
        "training_low_support_class_filter": training_low_support_class_filter,
        "train_label_count": int(train_frame["intent"].nunique()),
        "test_label_count": int(test_frame["intent"].nunique()),
        "train_labels_absent_from_official_test": train_labels_absent_from_test,
        "official_test_labels_without_training_support": test_labels_without_training_support,
        "official_test_label_coverage_fraction": (
            len(test_label_set & train_label_set) / len(train_label_set)
        ),
        "official_test_labels_supported_by_training_fraction": (
            len(test_label_set & train_label_set) / len(test_label_set) if test_label_set else 1.0
        ),
        "training_labels_covered_by_official_test_fraction": (
            len(test_label_set & train_label_set) / len(train_label_set) if train_label_set else 1.0
        ),
        "train_group_count_worker_id": int(train_frame["worker_id"].nunique()),
        "split_policy": "Official MASSIVE train partition only enters AMORPH. Rows sharing worker_id stay in one internal split. Official MASSIVE test is evaluated only after strategy selection and is never used by the selector or acceptance gate.",
        "limitations": [
            "This is a public benchmark, not proof of production performance on a business-specific distribution.",
            "MASSIVE intent classification is a 60-class voice-assistant task, not retrieval/RAG or generative-model fine-tuning.",
            "AMORPH still makes its internal train/validation/gate/test split within the official train partition, so only its train subset fits the models.",
            "The official test partition may include annotation-worker identities also represented in the official training partition; report worker-disjoint internal metrics separately.",
            "The official test can omit a label present in training; missing test-label support is recorded explicitly and must be considered when interpreting macro metrics.",
            "All official-training rows whose normalized utterance has multiple intent labels are excluded wholesale before AMORPH validation; counts and hashes are recorded. This changes the effective training sample and is a benchmark limitation.",
            "Training labels with fewer than five unique normalized texts or fewer than five distinct worker groups after ambiguous-text removal are excluded wholesale to satisfy AMORPH's grouped-split constraints. Official test rows are never removed; labels without training support are reported explicitly.",
        ],
    }
    if train_labels_absent_from_test:
        provenance["limitations"].append(
            "Training labels absent from the official test partition: "
            + ", ".join(train_labels_absent_from_test)
        )
    if len(train_frame) > 5000:
        provenance["limitations"].append(
            "AMORPH internal near-duplicate audit is skipped above 5,000 training rows; worker_id grouping prevents shared-worker cross-split leakage but does not rule out near-duplicates across different workers."
        )
    provenance["limitations"].append(
        "The legacy official_test_label_coverage_fraction uses the training-label universe as its denominator; use official_test_labels_supported_by_training_fraction for the share of official-test labels with training support."
    )
    write_json(out / "dataset_provenance.json", provenance)

    # Imports are delayed so data provenance is retained even if AMORPH fails.
    from amorph_autotrainer.benchmark_metrics import per_class_classification_metrics
    from amorph_autotrainer.cost import CostConfig
    from amorph_autotrainer.metrics import classification_metrics
    from amorph_autotrainer.pipeline import run_training
    from amorph_autotrainer.strategies import E5Candidate
    import joblib
    import numpy as np

    strategies = ["tfidf", "e5"] if args.with_e5 else ["tfidf"]
    cost_config = CostConfig(
        max_runtime_seconds=args.max_runtime_seconds,
        max_candidates=len(strategies),
        expected_queries=args.expected_queries,
        compute_eur_per_hour=args.compute_eur_per_hour,
        human_hours=args.human_hours,
        human_eur_per_hour=args.human_eur_per_hour,
        bootstrap_samples=500,
        min_gate_examples_per_class=5,
        random_seed=42,
    )
    result = run_training(
        csv_path=train_csv,
        output_dir=out / "runs",
        text_column="text",
        label_column="label",
        group_column="group_id",
        strategies=strategies,
        config=cost_config,
        e5_model_id=E5_MODEL_ID,
        e5_model_revision=E5_MODEL_REVISION,
        e5_classifier_c=args.e5_classifier_c,
        e5_class_weight=None if args.e5_class_weight == "none" else args.e5_class_weight,
    )

    test_texts = test_frame["utt"].tolist()
    test_labels = np.asarray(test_frame["intent"].tolist())
    expected_labels = sorted(train_label_set)
    eligible_training_support_by_label = {
        label: int(count) for label, count in train_frame["intent"].value_counts().items()
    }

    def external_classification_metrics(y_true, y_pred):
        metrics = classification_metrics(y_true, y_pred)
        from sklearn.metrics import f1_score, recall_score

        metrics["macro_f1_fixed_training_label_universe"] = float(
            f1_score(
                y_true, y_pred, labels=expected_labels,
                average="macro", zero_division=0,
            )
        )
        metrics["balanced_accuracy_fixed_training_label_universe"] = float(
            recall_score(
                y_true, y_pred, labels=expected_labels,
                average="macro", zero_division=0,
            )
        )
        observed = set(y_true.tolist())
        metrics["training_label_count"] = len(expected_labels)
        metrics["official_test_labels_absent_from_support"] = sorted(
            set(expected_labels) - observed
        )
        metrics["official_test_labels_without_training_support"] = sorted(
            observed - set(expected_labels)
        )
        metrics["official_test_label_coverage_fraction"] = (
            len(observed & set(expected_labels)) / len(expected_labels)
        )
        metrics["official_test_labels_supported_by_training_fraction"] = (
            len(observed & set(expected_labels)) / len(observed) if observed else 1.0
        )
        metrics["training_labels_covered_by_official_test_fraction"] = (
            len(observed & set(expected_labels)) / len(expected_labels) if expected_labels else 1.0
        )
        metrics["test_support_by_training_label"] = {
            label: int(np.sum(y_true == label)) for label in expected_labels
        }
        metrics["per_class_metrics"] = per_class_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            eligible_training_support_by_label=eligible_training_support_by_label,
        )
        metrics["per_class_metrics_scope"] = "official_test_report_only"
        return metrics

    run_dir = Path(result["run_dir"])

    tfidf_model_path = run_dir / "candidates" / "tfidf" / "model.joblib"
    tfidf_model = joblib.load(tfidf_model_path)
    tfidf_predictions = tfidf_model.predict(test_texts)
    external_metrics = {
        "tfidf": external_classification_metrics(test_labels, tfidf_predictions),
    }

    if args.with_e5 and (run_dir / "candidates" / "e5" / "classifier.joblib").is_file():
        e5_dir = run_dir / "candidates" / "e5"
        e5_metadata = json.loads((e5_dir / "strategy.json").read_text(encoding="utf-8"))
        encoder = E5Candidate._load_encoder(
            e5_metadata["model_id"], e5_metadata.get("model_revision", E5_MODEL_REVISION)
        )
        classifier = joblib.load(e5_dir / "classifier.joblib")
        e5_features = E5Candidate._encode(encoder, test_texts)
        external_metrics["e5"] = external_classification_metrics(
            test_labels, classifier.predict(e5_features)
        )

    benchmark_result = {
        "project": "AMORPH AutoTrainer",
        "project_version": "0.1.0 P0 verification extension",
        "environment": {"python_version": platform.python_version(), "platform": platform.platform()},
        "dataset_provenance_file": str(out / "dataset_provenance.json"),
        "internal_amorph_run": result,
        "selected_strategy": result["selected_strategy"],
        "e5_classifier_config": {"C": args.e5_classifier_c, "class_weight": None if args.e5_class_weight == "none" else args.e5_class_weight, "model_revision": E5_MODEL_REVISION},
        "per_class_metrics_protocol": {
            "scope": "official_test_report_only",
            "label_universe": "union of eligible official-training labels, official-test labels, and predicted labels",
            "eligible_training_support_definition": "row count in prepared official training partition after benchmark eligibility filters, before AMORPH internal splitting",
            "zero_division": 0,
            "used_for_selection_or_gate": False,
        },
        "external_test_metrics_by_strategy": external_metrics,
        "selected_strategy_official_test_metrics": external_metrics.get(result["selected_strategy"]),
        "conclusion": "Official-test metrics are report-only and are calculated after internal selection. Interpret this as a task-specific classification benchmark; do not infer competitiveness against RAG, LLM fine-tuning, or other task families.",
    }
    write_json(out / "benchmark_result.json", benchmark_result)
    print(json.dumps({
        "output_dir": str(out),
        "train_rows": len(train_frame),
        "official_test_rows": len(test_frame),
        "train_classes": int(train_frame["intent"].nunique()),
        "test_classes": int(test_frame["intent"].nunique()),
        "train_labels_absent_from_official_test": train_labels_absent_from_test,
        "official_test_labels_without_training_support": test_labels_without_training_support,
        "official_test_label_coverage_fraction": provenance["official_test_label_coverage_fraction"],
        "selected_strategy": result["selected_strategy"],
        "e5_classifier_config": {"C": args.e5_classifier_c, "class_weight": None if args.e5_class_weight == "none" else args.e5_class_weight},
        "external_test_metrics_by_strategy": external_metrics,
        "provenance_file": str(out / "dataset_provenance.json"),
        "result_file": str(out / "benchmark_result.json"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BENCHMARK ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Run bounded, reproducible AMORPH experiments on public customer-intent datasets.

Datasets:
- PolyAI/banking77: 77 online-banking customer-service intents (English).
- DeepPavlov/clinc150: broad intent-classification benchmark (English; includes
  out-of-scope examples where present in the published split).

Only the official train split fits AMORPH. The official test split is evaluated
after candidate selection and is report-only. Since the sources do not publish
stable customer/author IDs, this experiment cannot perform group-aware splitting.
A per-class cap keeps the input at <= 6000 rows so AMORPH's near-duplicate audit
can execute instead of being skipped by its configured row limit. Text is never
written into the uploaded evidence artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DATASETS: dict[str, dict[str, str | int]] = {
    "banking77": {
        "repo_id": "PolyAI/banking77",
        "text_candidates": "text,utterance,query",
        "label_candidates": "label,intent,category",
        "license": "CC BY 4.0 (dataset card)",
        "card": "https://huggingface.co/datasets/PolyAI/banking77",
        "max_per_class": 75,
        "domain": "online banking customer-service intent classification",
        "provenance_caveat": (
            "The dataset card describes online-banking queries with intent labels, "
            "but does not fully document the original utterance collection process. "
            "Do not treat it as verified raw CRM export."
        ),
    },
    "clinc150": {
        "repo_id": "DeepPavlov/clinc150",
        "text_candidates": "utterance,text,query",
        "label_candidates": "label,intent,category",
        "license": "License provenance requires reconciliation: original clinc/oos-eval repository states CC BY 3.0 Unported; UCI CLINC150 record may state CC BY 4.0. Do not assume the repackaged Hub dataset has identical redistribution terms without checking.",
        "card": "https://huggingface.co/datasets/DeepPavlov/clinc150",
        "max_per_class": 39,
        "domain": "crowdsourced multi-domain intent and out-of-scope classification",
        "provenance_caveat": (
            "CLINC150 is a crowdsourced intent benchmark, not a collection of "
            "verified production customer-support tickets."
        ),
    },
}
E5_MODEL_ID = "intfloat/multilingual-e5-small"
E5_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SEED = 42
MAX_AUDIT_ROWS = 6000


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def canonical_rows_sha256(rows: Iterable[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(
            {"text": row["text"], "label": row["label"]},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def artifact_inventory(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "artifact_inventory.json":
            continue
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "algorithm": "SHA-256",
        "files_count": len(files),
        "inventory_self_included": False,
        "files": files,
    }


def choose_column(columns: list[str], candidates: str, kind: str) -> str:
    for name in candidates.split(","):
        if name in columns:
            return name
    raise ValueError(f"Could not identify {kind} column. Available columns: {columns}")


def value_to_label(value: Any, feature: Any) -> str:
    names = getattr(feature, "names", None)
    if isinstance(value, int) and names and hasattr(feature, "int2str"):
        return str(feature.int2str(value)).strip()
    return str(value).strip()


def resolve_label(raw_label: Any, feature: Any, *, missing_label: str | None = None) -> str:
    if raw_label is None or str(raw_label).strip().casefold() in {"", "none", "null", "nan", "nat", "<na>"}:
        if missing_label is None:
            raise ValueError("Missing label in source dataset row.")
        return missing_label
    return value_to_label(raw_label, feature)


def materialize_rows(
    dataset_split: Any,
    text_column: str,
    label_column: str,
    *,
    missing_label: str | None = None,
) -> list[dict[str, str]]:
    label_feature = dataset_split.features.get(label_column)
    rows: list[dict[str, str]] = []
    for index, item in enumerate(dataset_split):
        text = str(item.get(text_column) or "").strip()
        if not text:
            raise ValueError(f"Empty text in source split at row {index}")
        try:
            label = resolve_label(item.get(label_column), label_feature, missing_label=missing_label)
        except ValueError as exc:
            raise ValueError(f"Missing/invalid label in source split at row {index}") from exc
        rows.append({"text": text, "label": label})
    return rows


def exclude_exact_test_text_overlaps(
    raw_rows: list[dict[str, str]],
    official_test_texts: Iterable[str],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Remove train rows whose normalized text occurs in official test inputs.

    Only held-out input text is read; no held-out labels or metrics are accessed.
    This leakage-hygiene pass is performed before model fitting and selection.
    """
    official_test_texts = list(official_test_texts)
    test_keys = {normalized_text(text) for text in official_test_texts if normalized_text(text)}
    overlap_keys = {
        normalized_text(row["text"])
        for row in raw_rows
        if normalized_text(row["text"]) in test_keys
    }
    retained = [
        {"text": row["text"], "label": row["label"]}
        for row in raw_rows
        if normalized_text(row["text"]) not in test_keys
    ]
    audit = {
        "official_test_input_rows_seen_without_labels": len(official_test_texts),
        "official_test_normalized_text_keys": len(test_keys),
        "training_rows_removed_for_exact_normalized_test_text_overlap": len(raw_rows) - len(retained),
        "overlapping_normalized_text_keys": len(overlap_keys),
        "official_test_normalized_text_keys_sha256": sha256_bytes("\n".join(sorted(test_keys)).encode("utf-8")),
        "overlap_key_set_sha256": sha256_bytes("\n".join(sorted(overlap_keys)).encode("utf-8")),
        "test_labels_read_during_hygiene": False,
    }
    return retained, audit


def prepare_training_rows(
    raw_rows: list[dict[str, str]],
    *,
    max_per_class: int,
    seed: int = SEED,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Remove ambiguous exact-text groups, deduplicate, then sample deterministically.

    The sample is drawn from the official training partition only. The official test
    partition is not passed to this function.
    """
    by_key_labels: dict[str, set[str]] = defaultdict(set)
    for row in raw_rows:
        by_key_labels[normalized_text(row["text"])].add(row["label"])
    conflict_keys = {key for key, labels in by_key_labels.items() if len(labels) > 1}

    retained: list[dict[str, str]] = []
    seen: set[str] = set()
    conflicting_rows_removed = 0
    same_label_duplicates_removed = 0
    for row in raw_rows:
        key = normalized_text(row["text"])
        if key in conflict_keys:
            conflicting_rows_removed += 1
            continue
        if key in seen:
            same_label_duplicates_removed += 1
            continue
        seen.add(key)
        retained.append({"text": row["text"], "label": row["label"]})

    class_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in retained:
        class_rows[row["label"]].append(row)
    low_support = {label: len(rows) for label, rows in class_rows.items() if len(rows) < 20}
    if low_support:
        raise ValueError(f"At least 20 unique training texts per class are required: {low_support}")
    if len(class_rows) < 2:
        raise ValueError("At least two intent labels are required.")

    # Stable, label-stratified sample at or below MAX_AUDIT_ROWS. This is done before
    # model fitting and never uses validation, gate, internal test, or official test scores.
    sampled: list[dict[str, str]] = []
    for ordinal, label in enumerate(sorted(class_rows)):
        candidates = sorted(class_rows[label], key=lambda r: normalized_text(r["text"]))
        rng = random.Random(seed + ordinal)
        rng.shuffle(candidates)
        sampled.extend(candidates[:max_per_class])
    sampled.sort(key=lambda r: (r["label"], normalized_text(r["text"])))

    if len(sampled) > MAX_AUDIT_ROWS:
        raise ValueError(
            f"Prepared training sample has {len(sampled)} rows; near-duplicate audit cap is {MAX_AUDIT_ROWS}."
        )
    sampled_counts = Counter(row["label"] for row in sampled)
    if min(sampled_counts.values()) < 20:
        raise ValueError(f"Sampling left insufficient class support: {dict(sampled_counts)}")

    audit = {
        "input_rows_after_official_test_text_hygiene": len(raw_rows),
        "unique_normalized_texts_before_conflict_removal": len(by_key_labels),
        "conflicting_normalized_text_keys_removed": len(conflict_keys),
        "training_rows_removed_for_conflicting_labels": conflicting_rows_removed,
        "same_label_exact_duplicates_removed": same_label_duplicates_removed,
        "unique_training_rows_before_cap": len(retained),
        "max_examples_per_class": max_per_class,
        "selected_training_rows": len(sampled),
        "selected_class_counts": dict(sorted(sampled_counts.items())),
        "sampling_seed": seed,
        "sampling_policy": "sort normalized text per label, seeded shuffle per label, cap each label, then stable sort",
        "conflicting_key_set_sha256": sha256_bytes("\n".join(sorted(conflict_keys)).encode("utf-8")),
        "selected_training_rows_canonical_sha256": canonical_rows_sha256(sampled),
        "raw_text_written_to_audit": False,
    }
    return sampled, audit


def run_dataset(dataset_key: str, output_dir: Path, max_runtime_seconds: int, bootstrap_samples: int) -> int:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    cfg = DATASETS[dataset_key]
    repo_id = str(cfg["repo_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "benchmark": dataset_key,
        "dataset_id": repo_id,
        "dataset_card": cfg["card"],
        "license": cfg["license"],
        "domain": cfg["domain"],
        "dataset_provenance_caveat": cfg["provenance_caveat"],
        "resolved_dataset_revision": None,
        "official_train_used_for_fit": True,
        "official_validation_used": False,
        "official_test_used_for_selection_or_gate": False,
        "clinc150_missing_label_mapping": ("A non-empty CLINC150 utterance with a missing ClassLabel is mapped to the explicit out_of_scope intent, consistent with this published dataset variant; other benchmarks fail closed on missing labels." if dataset_key == "clinc150" else None),
        "group_column": None,
        "group_split_available": False,
        "group_split_limitation": (
            "The public dataset exposes no stable customer/author/source group ID. "
            "No synthetic group IDs are invented; this run cannot establish group-disjoint generalization."
        ),
        "training_cap_policy": {
            "purpose": "keep input <= the AMORPH near-duplicate audit cap",
            "max_near_duplicate_audit_rows": MAX_AUDIT_ROWS,
            "max_examples_per_class": cfg["max_per_class"],
            "sample_seed": SEED,
        },
        "e5": {"model_id": E5_MODEL_ID, "model_revision": E5_MODEL_REVISION, "classifier_C": 0.1, "class_weight": "balanced"},
        "strategies": ["tfidf", "e5"],
        "cost_interpretation": (
            "Cost rates and labor costs are zero-valued defaults unless explicitly supplied; "
            "this is not measured monetary TCO or proof of monetary cost optimality."
        ),
        "status": "preparing",
    }
    write_json(output_dir / "dataset_manifest.json", manifest)
    run_exit_code: int | None = None

    try:
        resolved_revision = HfApi().dataset_info(repo_id, revision="main").sha
        manifest["resolved_dataset_revision"] = resolved_revision
        train_split = load_dataset(repo_id, split="train", revision=resolved_revision, trust_remote_code=True)
        test_split = load_dataset(repo_id, split="test", revision=resolved_revision, trust_remote_code=True)

        text_column = choose_column(list(train_split.column_names), str(cfg["text_candidates"]), "text")
        label_column = choose_column(list(train_split.column_names), str(cfg["label_candidates"]), "label")
        if text_column not in test_split.column_names or label_column not in test_split.column_names:
            raise ValueError(
                f"Official test columns don't match train columns: {test_split.column_names}"
            )
        missing_label = "out_of_scope" if dataset_key == "clinc150" else None
        raw_train = materialize_rows(train_split, text_column, label_column, missing_label=missing_label)
        # Leakage hygiene only: compare input text, without touching official test labels.
        # Index only the text column here. Do not request/iterate the test label column
        # until the selector and Proof Gate have finished.
        official_test_texts_for_hygiene = [
            str(value or "").strip() for value in test_split[text_column]
        ]
        if any(not text for text in official_test_texts_for_hygiene):
            raise ValueError("Official test contains empty text; refusing an unverifiable overlap audit.")
        overlap_clean_train, overlap_audit = exclude_exact_test_text_overlaps(
            raw_train, official_test_texts_for_hygiene
        )
        prepared, preparation_audit = prepare_training_rows(
            overlap_clean_train, max_per_class=int(cfg["max_per_class"]), seed=SEED
        )
        manifest.update({
            "source_text_column": text_column,
            "source_label_column": label_column,
            "official_train_canonical_sha256": canonical_rows_sha256(raw_train),
            "official_train_row_count": len(raw_train),
            "training_rows_after_official_test_text_hygiene": len(overlap_clean_train),
            "official_test_overlap_hygiene": overlap_audit,
            "official_test_label_access_policy": "Official test input texts are used only to remove exact normalized-text overlaps from training before fitting; official test labels are not loaded until after selector and Proof Gate finish. Official test metrics are report-only.",
            "training_preparation_audit": preparation_audit,
            "sampled_training_rows_canonical_sha256": canonical_rows_sha256(prepared),
            "sampled_training_rows": len(prepared),
            "sampled_training_classes": len({row["label"] for row in prepared}),
            "status": "prepared",
        })
        write_json(output_dir / "dataset_manifest.json", manifest)

        from amorph_autotrainer.cost import CostConfig
        from amorph_autotrainer.metrics import classification_metrics
        from amorph_autotrainer.pipeline import run_training
        from amorph_autotrainer.strategies import E5Candidate
        import joblib
        import numpy as np

        run_root = output_dir / "amorph_runs"
        with tempfile.TemporaryDirectory(prefix=f"amorph_{dataset_key}_") as tmp:
            csv_path = Path(tmp) / "prepared_train.csv"
            import csv
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["text", "label"], lineterminator="\n")
                writer.writeheader()
                writer.writerows(prepared)

            result = run_training(
                csv_path=csv_path,
                output_dir=run_root,
                text_column="text",
                label_column="label",
                group_column=None,
                strategies=["tfidf", "e5"],
                config=CostConfig(
                    max_runtime_seconds=max_runtime_seconds,
                    max_candidates=2,
                    expected_queries=1000,
                    bootstrap_samples=bootstrap_samples,
                    min_gate_examples_per_class=5,
                    noninferiority_margin=0.01,
                    random_seed=SEED,
                ),
                e5_model_id=E5_MODEL_ID,
                e5_model_revision=E5_MODEL_REVISION,
                e5_classifier_c=0.1,
                e5_class_weight="balanced",
                max_near_duplicate_audit_rows=MAX_AUDIT_ROWS,
            )
            report_path = Path(result["run_dir"]) / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            test_rows = materialize_rows(test_split, text_column, label_column, missing_label=missing_label)
            test_texts = [row["text"] for row in test_rows]
            test_labels = np.asarray([row["label"] for row in test_rows])

            # Official test is evaluated only after the full internal selector and Proof Gate
            # have completed. It is a report-only external metric, never used for selection.
            label_universe = sorted({row["label"] for row in prepared} | {row["label"] for row in test_rows})
            def external_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
                from sklearn.metrics import f1_score, recall_score
                metrics = classification_metrics(y_true, y_pred)
                metrics["macro_f1_fixed_train_test_label_universe"] = float(
                    f1_score(y_true, y_pred, labels=label_universe, average="macro", zero_division=0)
                )
                metrics["balanced_accuracy_fixed_train_test_label_universe"] = float(
                    recall_score(y_true, y_pred, labels=label_universe, average="macro", zero_division=0)
                )
                return metrics

            models_dir = Path(result["run_dir"]) / "candidates"
            model_metrics: dict[str, Any] = {}
            tfidf_model = joblib.load(models_dir / "tfidf" / "model.joblib")
            tfidf_predictions = tfidf_model.predict(test_texts)
            model_metrics["tfidf"] = external_metrics(test_labels, tfidf_predictions)

            e5_dir = models_dir / "e5"
            if (e5_dir / "classifier.joblib").is_file():
                metadata = json.loads((e5_dir / "strategy.json").read_text(encoding="utf-8"))
                encoder = E5Candidate._load_encoder(
                    metadata.get("model_id", E5_MODEL_ID),
                    metadata.get("model_revision", E5_MODEL_REVISION),
                )
                classifier = joblib.load(e5_dir / "classifier.joblib")
                features = E5Candidate._encode(encoder, test_texts)
                e5_predictions = classifier.predict(features)
                model_metrics["e5"] = external_metrics(test_labels, e5_predictions)

            sampled_train_keys = {normalized_text(row["text"]) for row in prepared}
            test_keys = [normalized_text(row["text"]) for row in test_rows]
            exact_text_overlap_count = sum(key in sampled_train_keys for key in test_keys)
            manifest.update({
                "status": "completed",
                "python_version": platform.python_version(),
                "environment_platform": platform.platform(),
                "amorph_run_id": result.get("run_id"),
                "amorph_report_sha256": sha256_file(report_path),
                "amorph_report_relative_path": report_path.relative_to(output_dir).as_posix(),
                "selected_strategy": result.get("selected_strategy"),
                "validation_metrics_all_candidates": report.get("validation_metrics_all_candidates"),
                "proofgate": report.get("proofgate"),
                "acceptance_gate_metrics": report.get("acceptance_gate_metrics"),
                "internal_final_test_metrics": report.get("final_test_metrics"),
                "internal_test_used_for_selection": report.get("test_split_used_for_selection"),
                "official_test_rows": len(test_rows),
                "official_test_canonical_sha256": canonical_rows_sha256(test_rows),
                "official_test_metrics_by_strategy_report_only": model_metrics,
                "official_test_exact_normalized_text_overlap_with_sampled_training": exact_text_overlap_count,
                "official_test_overlap_policy": "Exact normalized-text overlaps were removed from training before fitting using test input text only; test labels were not read during the hygiene pass, and official test metrics remain report-only.",
                "limits": [
                    "Public benchmark results are not owner-domain results.",
                    str(cfg["provenance_caveat"]),
                    "No stable source/customer group ID is exposed; random internal split cannot prove group-disjoint generalization.",
                    "Training was deterministically capped per class to keep the full near-duplicate audit under its 6000-row limit; this is not a full-training-set leaderboard run.",
                    "Official test input text is used only for exact-overlap hygiene; official test labels and metrics are not used for fitting, selector, or Proof Gate. Metrics are report-only; do not tune future parameters based on them.",
                    "TCO uses zero-valued cost inputs and is not actual monetary cost measurement.",
                ],
            })
        write_json(output_dir / "dataset_manifest.json", manifest)
        write_json(output_dir / "benchmark_result.json", {
            "dataset": dataset_key,
            "dataset_revision": resolved_revision,
            "selected_strategy": manifest.get("selected_strategy"),
            "validation_metrics_all_candidates": manifest.get("validation_metrics_all_candidates"),
            "proofgate": manifest.get("proofgate"),
            "official_test_metrics_report_only": manifest.get("official_test_metrics_by_strategy_report_only"),
            "official_test_used_for_selection": False,
            "group_disjointness_proven": False,
            "training_cap_per_class": cfg["max_per_class"],
            "source_revision": resolved_revision,
            "official_train_canonical_sha256": manifest.get("official_train_canonical_sha256"),
            "sampled_training_rows_canonical_sha256": manifest.get("sampled_training_rows_canonical_sha256"),
            "official_test_canonical_sha256": manifest.get("official_test_canonical_sha256"),
        })
        run_exit_code = 0
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        })
        write_json(output_dir / "dataset_manifest.json", manifest)
        print(f"{dataset_key} benchmark failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        run_exit_code = 1

    inventory = artifact_inventory(output_dir)
    write_json(output_dir / "artifact_inventory.json", inventory)
    print(json.dumps({
        "dataset": dataset_key,
        "status": manifest.get("status"),
        "dataset_revision": manifest.get("resolved_dataset_revision"),
        "selected_strategy": manifest.get("selected_strategy"),
        "proofgate": manifest.get("proofgate"),
        "official_test_metrics_report_only": manifest.get("official_test_metrics_by_strategy_report_only"),
        "artifact_files": inventory["files_count"],
        "raw_text_in_evidence": False,
    }, ensure_ascii=False, indent=2))
    return int(run_exit_code if run_exit_code is not None else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-runtime-seconds", type=int, default=4800)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    return run_dataset(args.dataset, args.output_dir.resolve(), args.max_runtime_seconds, args.bootstrap_samples)


if __name__ == "__main__":
    raise SystemExit(main())

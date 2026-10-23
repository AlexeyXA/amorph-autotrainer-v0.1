#!/usr/bin/env python3
"""Experimental AMORPH benchmark on Russian user reviews with rating-derived proxy labels."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "d0rj/geo-reviews-dataset-2023"
DATASET_FILE = "data/train-00000-of-00001-49261e4e5a35a5a0.parquet"
DATASET_CARD = "https://github.com/yandex/geo-reviews-dataset-2023"
DATASET_MIRROR = "https://huggingface.co/datasets/d0rj/geo-reviews-dataset-2023"
SEED = 42
MAX_ROWS = 4900
MAX_PER_ORGANIZATION = 8


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def rating_to_label(value: Any) -> str | None:
    """1-2 stars => negative; 3 => neutral; 4-5 => positive; rating 0 is excluded."""
    try:
        rating = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if rating in (1, 2):
        return "negative"
    if rating == 3:
        return "neutral"
    if rating in (4, 5):
        return "positive"
    return None


def organization_group_id(address: Any, name: Any) -> str | None:
    address_key, name_key = normalize(address), normalize(name)
    if not address_key or not name_key:
        return None
    return sha256_bytes((address_key + "\x1f" + name_key).encode("utf-8"))[:24]


def prepare_review_records(
    raw_rows: Iterable[dict[str, Any]], *, max_rows: int = MAX_ROWS,
    max_per_organization: int = MAX_PER_ORGANIZATION, seed: int = SEED,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Clean proxy labels, remove ambiguous exact-text duplicates, and sample deterministically."""
    if max_rows < 100 or max_per_organization < 1:
        raise ValueError("max_rows must be >= 100 and max_per_organization must be >= 1")
    converted: list[dict[str, str]] = []
    input_rows = dropped_missing = dropped_unrated = 0
    for item in raw_rows:
        input_rows += 1
        text = str(item.get("text") or "").strip()
        label = rating_to_label(item.get("rating"))
        group_id = organization_group_id(item.get("address"), item.get("name_ru"))
        if not text or not group_id:
            dropped_missing += 1
            continue
        if label is None:
            dropped_unrated += 1
            continue
        key = normalize(text)
        if not key:
            dropped_missing += 1
            continue
        converted.append({"text": text, "label": label, "group_id": group_id, "_key": key})

    labels_by_text: dict[str, set[str]] = defaultdict(set)
    for row in converted:
        labels_by_text[row["_key"]].add(row["label"])
    conflicting = {key for key, labels in labels_by_text.items() if len(labels) > 1}
    clean = [row for row in converted if row["_key"] not in conflicting]
    removed_conflict_rows = len(converted) - len(clean)

    # Global exact-text dedup prevents identical text from crossing organization groups.
    clean.sort(key=lambda row: (row["_key"], row["group_id"], row["text"], row["label"]))
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    duplicates_removed = 0
    for row in clean:
        if row["_key"] in seen:
            duplicates_removed += 1
            continue
        seen.add(row["_key"])
        unique.append(row)

    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in unique:
        by_group[row["group_id"]].append(row)
    group_order = sorted(
        by_group, key=lambda gid: sha256_bytes(f"{seed}|{gid}".encode("utf-8"))
    )
    selected: list[dict[str, str]] = []
    selected_groups: set[str] = set()
    for gid in group_order:
        group_rows = sorted(
            by_group[gid],
            key=lambda row: sha256_bytes(f"{seed}|{row['_key']}".encode("utf-8")),
        )[:max_per_organization]
        room = max_rows - len(selected)
        if room <= 0:
            break
        chosen = group_rows[:room]
        if chosen:
            selected.extend(chosen)
            selected_groups.add(gid)
        if len(chosen) < len(group_rows):
            break

    labels = Counter(row["label"] for row in selected)
    groups_per_label: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        groups_per_label[row["label"]].add(row["group_id"])
    for label in ("negative", "neutral", "positive"):
        if labels[label] < 20 or len(groups_per_label[label]) < 8:
            raise ValueError(
                f"Insufficient support for {label}: {labels[label]} rows, "
                f"{len(groups_per_label[label])} organization groups"
            )
    canonical = b"".join(
        json.dumps(
            {"text": row["text"], "label": row["label"], "group_id": row["group_id"]},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8") + b"\n" for row in selected
    )
    audit = {
        "input_rows": input_rows,
        "rows_dropped_missing_text_or_organization": dropped_missing,
        "rows_dropped_unrated_or_rating_zero": dropped_unrated,
        "rows_removed_for_conflicting_exact_text_labels": removed_conflict_rows,
        "conflicting_normalized_text_keys": len(conflicting),
        "same_label_exact_text_duplicates_removed": duplicates_removed,
        "unique_rows_before_sampling": len(unique),
        "selected_rows": len(selected),
        "selected_organization_groups": len(selected_groups),
        "max_rows": max_rows,
        "max_rows_per_organization": max_per_organization,
        "seed": seed,
        "rating_to_label_mapping": {"1-2": "negative", "3": "neutral", "4-5": "positive", "0": "excluded"},
        "selected_label_counts": dict(sorted(labels.items())),
        "selected_organization_groups_by_label": {
            label: len(groups_per_label[label]) for label in ("negative", "neutral", "positive")
        },
        "selected_rows_canonical_sha256": sha256_bytes(canonical),
        "raw_text_in_audit": False,
        "organization_names_or_addresses_in_audit": False,
    }
    return [
        {"text": row["text"], "label": row["label"], "group_id": row["group_id"]}
        for row in selected
    ], audit


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="benchmark_runs/yandex_geo_reviews")
    parser.add_argument("--max-runtime-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    started = time.time()
    output = Path(args.output_dir).resolve()
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        from huggingface_hub import HfApi, hf_hub_download

        revision = HfApi().dataset_info(DATASET_ID, revision="main").sha
        source_path = Path(hf_hub_download(
            repo_id=DATASET_ID, filename=DATASET_FILE,
            repo_type="dataset", revision=revision,
        ))
        source_hash = sha256_bytes(source_path.read_bytes())
        source_frame = pd.read_parquet(source_path)
        required = {"text", "rating", "address", "name_ru"}
        missing = sorted(required - set(source_frame.columns))
        if missing:
            raise RuntimeError(f"Dataset missing required columns: {missing}")
        records, audit = prepare_review_records(
            source_frame[["text", "rating", "address", "name_ru"]].to_dict("records")
        )
        prepared = data_dir / "prepared_reviews.csv"
        pd.DataFrame(records).to_csv(prepared, index=False, encoding="utf-8")
        prepared_hash = sha256_bytes(prepared.read_bytes())
        write_json(output / "dataset_provenance.json", {
            "dataset_id": DATASET_ID, "dataset_revision": revision,
            "official_repository": DATASET_CARD, "mirror": DATASET_MIRROR,
            "license": "MIT per the public dataset repository/card",
            "source_file": DATASET_FILE, "source_file_sha256": source_hash,
            "source_rows": int(len(source_frame)), "prepared_rows": len(records),
            "prepared_file_sha256": prepared_hash,
            "group_policy": "Opaque SHA-256 derived from normalized organization address + name; organization metadata is not used as model features.",
            "split_policy": "AMORPH four-way internal split, group-disjoint by organization; source has no official independent test split.",
            "target": "Rating-derived negative/neutral/positive proxy, not intent or manually annotated sentiment.",
            "rating_target_policy": "1-2=negative, 3=neutral, 4-5=positive, rating 0 excluded.",
            "sampling_and_cleaning": audit,
            "raw_text_in_provenance": False, "organization_names_or_addresses_in_provenance": False,
        })

        from amorph_autotrainer.cost import CostConfig
        from amorph_autotrainer.pipeline import run_training
        config = CostConfig(
            max_runtime_seconds=args.max_runtime_seconds, max_candidates=2,
            expected_queries=1000, compute_eur_per_hour=0.0,
            human_hours=0.0, human_eur_per_hour=0.0,
            bootstrap_samples=500, min_gate_examples_per_class=5,
            random_seed=SEED,
        )
        result = run_training(
            csv_path=prepared, output_dir=output / "runs",
            text_column="text", label_column="label", group_column="group_id",
            strategies=["tfidf", "e5"], config=config,
            e5_model_id="intfloat/multilingual-e5-small",
            e5_model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            e5_classifier_c=0.1, e5_class_weight="balanced",
            max_near_duplicate_audit_rows=5000,
        )
        run_dir = Path(result["run_dir"])
        report_path = run_dir / "report.json"
        summary = {
            "project": "AMORPH AutoTrainer",
            "benchmark": "Yandex Maps Russian review-rating buckets",
            "source_revision": revision, "source_file_sha256": source_hash,
            "prepared_file_sha256": prepared_hash, "prepared_rows": len(records),
            "label_counts": audit["selected_label_counts"],
            "organization_group_count": audit["selected_organization_groups"],
            "strategies": ["tfidf", "e5"],
            "e5_config": {"model_id": "intfloat/multilingual-e5-small",
                          "model_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
                          "classifier_C": 0.1, "class_weight": "balanced"},
            "selected_strategy": result.get("selected_strategy"),
            "validation_metrics": result.get("validation_metrics"),
            "acceptance_gate": result.get("acceptance_gate"),
            "proofgate": result.get("proofgate"),
            "candidate_errors": result.get("candidate_errors"),
            "internal_final_test_metrics": {
                "baseline": result.get("baseline_test_metrics"),
                "selected_candidate": result.get("selected_candidate_test_metrics"),
            },
            "internal_run_status": result.get("status"), "run_id": result.get("run_id"),
            "run_report_sha256": sha256_bytes(report_path.read_bytes()) if report_path.is_file() else None,
            "interpretation_limits": [
                "Actual user-written reviews and ratings, but the target is a rating-derived proxy, not a human-annotated sentiment class.",
                "Tests Russian review classification, not support-intent detection or the owner's metal-products domain.",
                "No official independent test split exists; internal final-test metrics become observed after this run.",
                "Zero compute/labor rates are assumptions and do not establish monetary TCO.",
            ],
        }
        write_json(output / "benchmark_result.json", summary)
        write_json(output / "execution_manifest.json", {
            "status": "result_written", "source_revision": revision,
            "dataset_id": DATASET_ID, "dataset_revision": revision,
            "source_file_sha256": source_hash, "prepared_file_sha256": prepared_hash,
            "elapsed_seconds": time.time() - started,
            "selected_strategy": result.get("selected_strategy"),
            "internal_run_id": result.get("run_id"),
            "run_report_sha256": summary["run_report_sha256"],
        })
        print(json.dumps({
            "status": "result_written", "selected_strategy": result.get("selected_strategy"),
            "prepared_rows": len(records), "label_counts": audit["selected_label_counts"],
            "validation_metrics": result.get("validation_metrics"),
            "proofgate": result.get("proofgate"),
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"BENCHMARK ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

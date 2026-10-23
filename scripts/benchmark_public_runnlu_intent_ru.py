#!/usr/bin/env python3
"""Run AMORPH on the public, pinned Russian RuNLU intent train split.

Only the official train parquet is downloaded. The official test split is never
downloaded or used. Raw public input data is held in a temporary directory and
is not included in the uploaded evidence artifact.

The source has a long-tail class distribution and no speaker/session/source ID.
Before the run, labels with fewer than 20 distinct normalized train utterances
are excluded solely on predeclared class-support grounds: a 25% acceptance gate
needs five examples per class to meet the configured per-class floor. Excluded
classes and row/text hashes are recorded; they are not relabeled or silently
treated as evaluated.

Requires this repository's `.[embeddings,test]` extra plus pandas, pyarrow and
huggingface_hub. E5 uses the pinned model revision declared below.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "mteb/RuNLUIntentClassification"
DATASET_REVISION = "424d0f767aaa5c411e3a529eec04658e5726a39e"
DATASET_FILE = "rus/train-00000-of-00001.parquet"
DATASET_LICENSE = "CC-BY-4.0"
DATASET_CARD = "https://huggingface.co/datasets/mteb/RuNLUIntentClassification"
DATASET_README = "https://huggingface.co/datasets/mteb/RuNLUIntentClassification/blob/main/README.md"
DATASET_PAPER = "https://arxiv.org/abs/1903.05566"
E5_MODEL_ID = "intfloat/multilingual-e5-small"
E5_MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MIN_UNIQUE_TEXTS_PER_LABEL = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def prepare_records(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Remove empty rows and *all* rows for conflicting normalized-text labels.

    Exact duplicate rows with the same label are left for AMORPH's own audited
    duplicate handling. Conflicting labels are never arbitrarily resolved.
    """
    valid: list[dict[str, str]] = []
    dropped_empty = 0
    normalized_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        raw_text = row.get("text")
        raw_label = row.get("label")
        # Hugging Face parquet columns may encode missing values as NaN/NA.
        text = "" if raw_text is None else str(raw_text).strip()
        label = "" if raw_label is None else str(raw_label).strip()
        if text.casefold() in {"nan", "nat", "<na>"}:
            text = ""
        if label.casefold() in {"nan", "nat", "<na>"}:
            label = ""
        if not text or not label:
            dropped_empty += 1
            continue
        valid.append({"text": text, "label": label})
        normalized_labels[normalize_text(text)].add(label)

    conflicting_keys = {key for key, labels in normalized_labels.items() if len(labels) > 1}
    kept: list[dict[str, str]] = []
    removed_conflicts: list[tuple[str, str]] = []
    for row in valid:
        normalized = normalize_text(row["text"])
        if normalized in conflicting_keys:
            removed_conflicts.append((normalized, row["label"]))
        else:
            kept.append(row)

    canonical_conflict_keys = "\n".join(sorted(conflicting_keys))
    canonical_conflict_rows = "\n".join(
        f"{text}\t{label}" for text, label in sorted(removed_conflicts)
    )
    counts = Counter(row["label"] for row in kept)
    audit = {
        "input_rows": len(valid) + dropped_empty,
        "rows_with_empty_text_or_label_removed": dropped_empty,
        "ambiguous_normalized_text_keys_removed": len(conflicting_keys),
        "rows_removed_for_conflicting_labels": len(removed_conflicts),
        "rows_retained_before_class_support_filter_and_AMORPH_deduplication": len(kept),
        "distinct_labels_before_conflict_removal": len(normalized_labels),
        "distinct_labels_after_conflict_removal": len(counts),
        "class_counts_after_conflict_removal": dict(sorted(counts.items())),
        "conflicting_text_keys_sha256": sha256_text(canonical_conflict_keys),
        "removed_conflict_rows_sha256": sha256_text(canonical_conflict_rows),
        "policy": "drop all normalized-text rows with conflicting labels; never relabel",
    }
    return kept, audit


def filter_low_support_classes(
    records: list[dict[str, str]],
    minimum_unique_texts_per_label: int = MIN_UNIQUE_TEXTS_PER_LABEL,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Filter labels by distinct normalized-text support, never by model metrics."""
    unique_texts_by_label: dict[str, set[str]] = defaultdict(set)
    for row in records:
        unique_texts_by_label[row["label"]].add(normalize_text(row["text"]))
    unique_counts = {
        label: len(texts) for label, texts in sorted(unique_texts_by_label.items())
    }
    excluded_labels = {
        label: count
        for label, count in unique_counts.items()
        if count < minimum_unique_texts_per_label
    }
    excluded_set = set(excluded_labels)
    retained = [row for row in records if row["label"] not in excluded_set]
    excluded_rows = [row for row in records if row["label"] in excluded_set]
    excluded_canonical = "\n".join(sorted(
        f"{normalize_text(row['text'])}\t{row['label']}" for row in excluded_rows
    ))
    retained_unique_counts = {
        label: count
        for label, count in unique_counts.items()
        if label not in excluded_set
    }
    audit = {
        "minimum_unique_texts_per_label": minimum_unique_texts_per_label,
        "rationale": (
            "The target acceptance-gate fraction is 25%; 20 unique rows/class supports "
            "the configured floor of five gate examples/class in the target split design."
        ),
        "filter_basis": "unique normalized text support only; no validation/gate/test metrics",
        "excluded_labels_and_unique_text_counts": dict(sorted(excluded_labels.items())),
        "excluded_label_count": len(excluded_labels),
        "rows_excluded_for_low_class_support": len(excluded_rows),
        "rows_retained_for_training_input": len(retained),
        "rows_retained_unique_texts_by_label": retained_unique_counts,
        "excluded_rows_sha256": sha256_text(excluded_canonical),
        "retained_labels": sorted(retained_unique_counts),
    }
    return retained, audit


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def build_artifact_inventory(root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "artifact_inventory.json":
            continue
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "algorithm": "SHA-256",
        "files": entries,
        "files_count": len(entries),
        "inventory_self_included": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-runtime-seconds", type=int, default=4800)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "amorph_runs"
    source_manifest: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_file": DATASET_FILE,
        "license": DATASET_LICENSE,
        "dataset_card": DATASET_CARD,
        "dataset_readme": DATASET_README,
        "source_paper": DATASET_PAPER,
        "source_split_loaded": "train",
        "official_test_split_downloaded": False,
        "official_test_split_used": False,
        "source_commit_sha": os.environ.get("GITHUB_SHA"),
        "python_version": sys.version.split()[0],
        "group_column": None,
        "grouping_caveat": (
            "The source provides no speaker/session/source group ID. AMORPH's bounded "
            "character-ngram near-duplicate clustering is used at the 5,000-row cap; it "
            "cannot establish independence by source or time."
        ),
        "strategies": ["tfidf", "e5"],
        "e5_model_id": E5_MODEL_ID,
        "e5_model_revision": E5_MODEL_REVISION,
        "acceptance_criteria": {
            "seed": 42,
            "noninferiority_margin": 0.01,
            "bootstrap_samples": args.bootstrap_samples,
            "min_gate_examples_per_class": 5,
            "max_near_duplicate_audit_rows": 5000,
            "min_macro_f1": None,
        },
        "status": "preparing",
    }
    write_json(output_dir / "public_dataset_manifest.json", source_manifest)

    run_exit_code: int | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="amorph_public_runnlu_") as scratch:
            scratch_dir = Path(scratch)
            try:
                from huggingface_hub import hf_hub_download
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError(
                    "Missing public-dataset dependencies: install pandas, pyarrow and huggingface_hub"
                ) from exc

            parquet_path = Path(hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=DATASET_FILE,
                revision=DATASET_REVISION,
                cache_dir=str(scratch_dir / "hf-cache"),
            ))
            frame = pd.read_parquet(parquet_path)
            missing = sorted({"text", "label"} - set(frame.columns))
            if missing:
                raise RuntimeError(f"Pinned public dataset file is missing required columns: {missing}")

            clean_records, prep_audit = prepare_records(
                frame[["text", "label"]].fillna("").to_dict("records")
            )
            supported_records, support_audit = filter_low_support_classes(clean_records)
            source_manifest["source_parquet_sha256"] = sha256_file(parquet_path)
            source_manifest["source_parquet_bytes"] = parquet_path.stat().st_size
            source_manifest["dataset_prep_audit"] = prep_audit
            source_manifest["class_support_filter"] = support_audit
            if not supported_records:
                raise RuntimeError("No records remain after the predeclared class-support filter.")

            post_filter_unique_counts = {
                label: len({normalize_text(row["text"]) for row in supported_records if row["label"] == label})
                for label in sorted({row["label"] for row in supported_records})
            }
            below_floor = {
                label: count for label, count in post_filter_unique_counts.items()
                if count < MIN_UNIQUE_TEXTS_PER_LABEL
            }
            if below_floor:
                raise RuntimeError(
                    "Class-support filter invariant failed: at least one retained class is below "
                    f"{MIN_UNIQUE_TEXTS_PER_LABEL} unique texts: {below_floor}"
                )

            input_csv = scratch_dir / "runnlu_train_clean.csv"
            write_csv(input_csv, supported_records)
            source_manifest["cleaned_training_csv_sha256"] = sha256_file(input_csv)
            source_manifest["cleaned_training_rows_before_AMORPH_deduplication"] = len(supported_records)
            source_manifest["num_training_classes_after_support_filter"] = len(post_filter_unique_counts)
            source_manifest["status"] = "prepared"
            write_json(output_dir / "public_dataset_manifest.json", source_manifest)

            command = [
                sys.executable, "-m", "amorph_autotrainer.cli", "train", str(input_csv),
                "--text-column", "text",
                "--label-column", "label",
                "--out", str(runs_dir),
                "--strategies", "tfidf,e5",
                "--max-candidates", "2",
                "--max-runtime-seconds", str(args.max_runtime_seconds),
                "--seed", "42",
                "--noninferiority-margin", "0.01",
                "--bootstrap-samples", str(args.bootstrap_samples),
                "--min-gate-examples-per-class", "5",
                "--max-near-duplicate-audit-rows", "5000",
                "--e5-model-revision", E5_MODEL_REVISION,
            ]
            run_result = subprocess.run(command, check=False, capture_output=True, text=True)
            run_exit_code = run_result.returncode
            (output_dir / "train_stdout.log").write_text(run_result.stdout, encoding="utf-8")
            (output_dir / "train_stderr.log").write_text(run_result.stderr, encoding="utf-8")

            run_dirs = sorted(
                (p for p in runs_dir.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ) if runs_dir.exists() else []
            report_path = run_dirs[0] / "report.json" if run_dirs else None
            if report_path is None or not report_path.is_file():
                raise RuntimeError(
                    f"AMORPH did not produce a run report; process exit code was {run_exit_code}."
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            source_manifest["amorph_run_id"] = report.get("run_id")
            source_manifest["amorph_run_report_sha256"] = sha256_file(report_path)
            source_manifest["amorph_report_status"] = report.get("status")
            source_manifest["selected_strategy"] = report.get("strategy")
            source_manifest["validation_metrics_all_candidates"] = report.get(
                "validation_metrics_all_candidates"
            )
            source_manifest["proofgate"] = report.get("proofgate")
            source_manifest["acceptance_gate_metrics"] = report.get("acceptance_gate_metrics")
            source_manifest["test_split_used_for_selection"] = report.get("test_split_used_for_selection")
            source_manifest["amorph_warnings"] = report.get("warnings")
            source_manifest["amorph_report_relative_path"] = report_path.relative_to(output_dir).as_posix()
            source_manifest["train_process_exit_code"] = run_exit_code
            source_manifest["status"] = "completed" if run_exit_code == 0 else "train_process_failed"
            write_json(output_dir / "public_dataset_manifest.json", source_manifest)
    except Exception as exc:
        source_manifest["status"] = "failed"
        source_manifest["error_type"] = type(exc).__name__
        source_manifest["error_message"] = str(exc)
        if run_exit_code is None or run_exit_code == 0:
            run_exit_code = 1
        source_manifest["train_process_exit_code"] = run_exit_code
        write_json(output_dir / "public_dataset_manifest.json", source_manifest)
        print(f"Public RuNLU run failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    inventory = build_artifact_inventory(output_dir)
    write_json(output_dir / "artifact_inventory.json", inventory)
    print(json.dumps({
        "status": source_manifest.get("status"),
        "dataset_revision": DATASET_REVISION,
        "source_split_loaded": "train",
        "official_test_used": False,
        "cleaned_training_rows_before_AMORPH_deduplication": source_manifest.get(
            "cleaned_training_rows_before_AMORPH_deduplication"
        ),
        "class_count_after_support_filter": source_manifest.get(
            "num_training_classes_after_support_filter"
        ),
        "excluded_label_count": source_manifest.get("class_support_filter", {}).get("excluded_label_count"),
        "amorph_run_id": source_manifest.get("amorph_run_id"),
        "amorph_report_status": source_manifest.get("amorph_report_status"),
        "selected_strategy": source_manifest.get("selected_strategy"),
        "proofgate": source_manifest.get("proofgate"),
        "artifact_inventory_files": inventory["files_count"],
    }, indent=2, ensure_ascii=False))
    return int(run_exit_code if run_exit_code is not None else 1)


if __name__ == "__main__":
    raise SystemExit(main())

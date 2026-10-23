#!/usr/bin/env python3
"""Run AMORPH on the public, pinned Russian RuNLU intent train split.

The benchmark reads only the official train parquet. It never downloads or uses
the official test split. Raw public input data is placed in a temporary directory
and is not included in the uploaded evidence artifact.

Install the repository's embedding extra plus pandas, pyarrow and huggingface_hub
before running this script. E5 uses the existing default model/configuration.
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
    """Drop empty rows and *all* rows for conflicting normalized text labels.

    Exact duplicate rows with the same label are left for AMORPH's own audited
    duplicate handling. Conflicting labels are never arbitrarily resolved.
    """
    valid: list[dict[str, str]] = []
    dropped_empty = 0
    normalized_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        text = str(row.get("text") or "").strip()
        label = str(row.get("label") or "").strip()
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
        "rows_retained_before_AMORPH_deduplication": len(kept),
        "distinct_labels_before_conflict_removal": len(normalized_labels),
        "distinct_labels_after_conflict_removal": len(counts),
        "class_counts_after_conflict_removal": dict(sorted(counts.items())),
        "conflicting_text_keys_sha256": sha256_text(canonical_conflict_keys),
        "removed_conflict_rows_sha256": sha256_text(canonical_conflict_rows),
        "policy": "drop all normalized-text rows with conflicting labels; never relabel",
    }
    return kept, audit


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

    run_exit_code = 1
    run_result: subprocess.CompletedProcess[str] | None = None
    with tempfile.TemporaryDirectory(prefix="amorph_public_runnlu_") as scratch:
        scratch_dir = Path(scratch)
        parquet_path: Path | None = None
        input_csv = scratch_dir / "runnlu_train_clean.csv"
        try:
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

            clean_records, audit = prepare_records(frame[["text", "label"]].to_dict("records"))
            source_manifest["source_parquet_sha256"] = sha256_file(parquet_path)
            source_manifest["source_parquet_bytes"] = parquet_path.stat().st_size
            source_manifest["dataset_prep_audit"] = audit
            if not clean_records:
                raise RuntimeError("No valid records remain after data validation.")
            insufficient = {
                label: count
                for label, count in audit["class_counts_after_conflict_removal"].items()
                if count < 5
            }
            source_manifest["classes_below_minimum_5"] = insufficient
            if insufficient:
                raise RuntimeError(
                    "At least one intent has fewer than five unambiguous examples; "
                    "refusing to silently drop classes or weaken the split constraints."
                )

            write_csv(input_csv, clean_records)
            source_manifest["cleaned_training_csv_sha256"] = sha256_file(input_csv)
            source_manifest["cleaned_training_rows"] = len(clean_records)
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
            if report_path and report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                source_manifest["amorph_run_id"] = report.get("run_id")
                source_manifest["amorph_run_report_sha256"] = sha256_file(report_path)
                source_manifest["amorph_report_status"] = report.get("status")
                source_manifest["selected_strategy"] = report.get("strategy")
                source_manifest["validation_metrics_all_candidates"] = report.get(
                    "validation_metrics_all_candidates"
                )
                source_manifest["proofgate"] = report.get("proofgate")
                source_manifest["acceptance_gate_metrics"] = report.get(
                    "acceptance_gate_metrics"
                )
                source_manifest["test_split_used_for_selection"] = report.get(
                    "test_split_used_for_selection"
                )
                source_manifest["amorph_warnings"] = report.get("warnings")
                source_manifest["amorph_report_relative_path"] = report_path.relative_to(output_dir).as_posix()
            else:
                source_manifest["amorph_run_report_missing"] = True
            source_manifest["train_process_exit_code"] = run_exit_code
            source_manifest["status"] = "completed" if run_exit_code == 0 else "train_process_failed"
            write_json(output_dir / "public_dataset_manifest.json", source_manifest)
        except Exception as exc:
            source_manifest["status"] = "failed"
            source_manifest["error_type"] = type(exc).__name__
            source_manifest["error_message"] = str(exc)
            source_manifest["train_process_exit_code"] = run_exit_code
            write_json(output_dir / "public_dataset_manifest.json", source_manifest)
            print(f"Public RuNLU run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            run_exit_code = run_exit_code or 1
        finally:
            # The downloaded parquet, temporary cleaned CSV, and Hub cache are not
            # copied to output_dir or the uploaded GitHub Actions artifact.

            pass

    inventory = build_artifact_inventory(output_dir)
    write_json(output_dir / "artifact_inventory.json", inventory)
    print(json.dumps({
        "status": source_manifest.get("status"),
        "dataset_revision": DATASET_REVISION,
        "source_split_loaded": "train",
        "official_test_used": False,
        "cleaned_training_rows": source_manifest.get("cleaned_training_rows"),
        "num_classes": (
            source_manifest.get("dataset_prep_audit", {}).get("distinct_labels_after_conflict_removal")
        ),
        "amorph_run_id": source_manifest.get("amorph_run_id"),
        "amorph_report_status": source_manifest.get("amorph_report_status"),
        "selected_strategy": source_manifest.get("selected_strategy"),
        "proofgate": source_manifest.get("proofgate"),
        "artifact_inventory_files": inventory["files_count"],
    }, indent=2, ensure_ascii=False))
    return int(run_exit_code)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Group-aware AMORPH benchmark on the public Russian author_profiling dataset.

Only the official train.jsonl is downloaded. The official validation/test files
are not downloaded or used. The target is the non-personal task label
`age_imitation` (younger / older / no_age_imitation); `author_id` is the real
anonymized group identifier. Rows from one author remain in one internal split.

Raw text and the cleaned CSV stay in a temporary directory. The uploaded artifact
contains only reports, hashes, model artifacts, metrics and aggregate audits.
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

DATASET_ID = "sagteam/author_profiling"
DATASET_REVISION_REQUESTED = "main"
DATASET_FILE = "data/train.jsonl"
DATASET_LICENSE = "Apache-2.0"
DATASET_CARD = "https://huggingface.co/datasets/sagteam/author_profiling"
DATASET_MIRROR_CARD = "https://huggingface.co/datasets/SPG29/author_profiling"
DATASET_PAPER = "https://doi.org/10.1007/s10994-021-06022-0"
TASK_COLUMN = "age_imitation"
TEXT_COLUMN = "text"
GROUP_COLUMN = "author_id"
TASK_LABELS = {"younger", "older", "no_age_imitation"}
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


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().casefold() in {"", "none", "null", "nan", "nat", "<na>"}


def prepare_task_records(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Filter the chosen task and fail closed on ambiguous/cross-author duplicate text.

    Selection is independent of any model result. A normalized text that has
    conflicting task labels is removed in full. An exact normalized text assigned
    to multiple authors is also removed in full to avoid cross-group text leakage.
    Same-author/same-label duplicates collapse to one row.
    """
    input_rows = 0
    not_applicable_rows = 0
    missing_text_rows = 0
    missing_group_rows = 0
    candidate_rows: list[dict[str, str]] = []

    for row in rows:
        input_rows += 1
        label_value = row.get(TASK_COLUMN)
        if is_missing(label_value) or str(label_value).strip() not in TASK_LABELS:
            not_applicable_rows += 1
            continue
        text_value = row.get(TEXT_COLUMN)
        if is_missing(text_value):
            missing_text_rows += 1
            continue
        group_value = row.get(GROUP_COLUMN)
        if is_missing(group_value):
            missing_group_rows += 1
            continue
        candidate_rows.append({
            "text": str(text_value).strip(),
            "label": str(label_value).strip(),
            "author_id": str(group_value).strip(),
        })

    if missing_group_rows:
        raise ValueError(
            f"{missing_group_rows} applicable rows have no {GROUP_COLUMN}; "
            "group-aware evaluation fails closed."
        )

    labels_by_text: dict[str, set[str]] = defaultdict(set)
    authors_by_text: dict[str, set[str]] = defaultdict(set)
    row_keys: list[str] = []
    for row in candidate_rows:
        key = normalize_text(row["text"])
        row_keys.append(key)
        labels_by_text[key].add(row["label"])
        authors_by_text[key].add(row["author_id"])

    conflicting_keys = {key for key, labels in labels_by_text.items() if len(labels) > 1}
    cross_author_keys = {
        key for key, authors in authors_by_text.items()
        if len(authors) > 1 and key not in conflicting_keys
    }

    retained: list[dict[str, str]] = []
    seen_same_author_text: set[tuple[str, str]] = set()
    removed_conflict_rows = 0
    removed_cross_author_duplicate_rows = 0
    removed_same_author_duplicate_rows = 0
    removed_texts: list[str] = []

    for row, key in zip(candidate_rows, row_keys):
        if key in conflicting_keys:
            removed_conflict_rows += 1
            removed_texts.append(f"{key}\tconflicting_label\t{row['label']}")
            continue
        if key in cross_author_keys:
            removed_cross_author_duplicate_rows += 1
            removed_texts.append(f"{key}\tcross_author_duplicate\t{row['label']}")
            continue
        identity = (key, row["author_id"])
        if identity in seen_same_author_text:
            removed_same_author_duplicate_rows += 1
            removed_texts.append(f"{key}\tsame_author_duplicate\t{row['label']}")
            continue
        seen_same_author_text.add(identity)
        retained.append(row)

    class_counts = Counter(row["label"] for row in retained)
    group_rows: dict[str, set[str]] = defaultdict(set)
    for row in retained:
        group_rows[row["author_id"]].add(row["label"])
    per_group_label_counts = Counter(tuple(sorted(labels)) for labels in group_rows.values())

    audit = {
        "source_rows": input_rows,
        "not_applicable_or_unrecognized_task_labels_removed": not_applicable_rows,
        "missing_text_rows_removed": missing_text_rows,
        "missing_group_rows": missing_group_rows,
        "applicable_rows_before_duplicate_audit": len(candidate_rows),
        "ambiguous_normalized_text_keys_removed": len(conflicting_keys),
        "rows_removed_for_conflicting_labels": removed_conflict_rows,
        "normalized_text_keys_repeated_across_authors_removed": len(cross_author_keys),
        "rows_removed_for_cross_author_duplicate_text": removed_cross_author_duplicate_rows,
        "rows_removed_for_same_author_same_text_duplicate": removed_same_author_duplicate_rows,
        "rows_retained_for_training_input": len(retained),
        "label_counts_retained": dict(sorted(class_counts.items())),
        "unique_author_groups_retained": len(group_rows),
        "per_group_label_presence_pattern_counts": {
            "|".join(key): value for key, value in sorted(per_group_label_counts.items())
        },
        "removed_row_keys_sha256": sha256_text("\n".join(sorted(removed_texts))),
        "duplicate_policy": (
            "drop all normalized-text rows with conflicting labels; drop all exact "
            "normalized-text repetitions across different authors; collapse repeated "
            "text within the same author; never relabel"
        ),
    }

    if set(class_counts) != TASK_LABELS:
        raise ValueError(
            f"Expected the three predeclared task labels {sorted(TASK_LABELS)}; "
            f"retained labels were {sorted(class_counts)}."
        )
    if min(class_counts.values()) < 20:
        raise ValueError(
            f"Each target class needs at least 20 retained rows; counts={dict(class_counts)}"
        )
    if len(group_rows) < 4:
        raise ValueError(
            f"Group-aware four-way split requires multiple author groups; got {len(group_rows)}"
        )
    return retained, audit


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["text", "label", "author_id"],
            lineterminator="\n",
        )
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


def resolved_hub_revision(path: Path) -> str | None:
    parts = path.parts
    try:
        idx = parts.index("snapshots")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION_REQUESTED)
    parser.add_argument("--max-runtime-seconds", type=int, default=4800)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "amorph_runs"
    manifest: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "dataset_revision_requested": args.dataset_revision,
        "dataset_revision_requested_default": DATASET_REVISION_REQUESTED,
        "dataset_file": DATASET_FILE,
        "dataset_license": DATASET_LICENSE,
        "dataset_card": DATASET_CARD,
        "dataset_mirror_card": DATASET_MIRROR_CARD,
        "dataset_paper": DATASET_PAPER,
        "official_split_downloaded": "train",
        "official_validation_downloaded": False,
        "official_test_downloaded": False,
        "official_test_used": False,
        "task_column": TASK_COLUMN,
        "text_column": TEXT_COLUMN,
        "group_column": GROUP_COLUMN,
        "target_labels": sorted(TASK_LABELS),
        "task_definition": (
            "Three-way classification of a crowdworker's task-labeled age imitation style: "
            "younger, older, or no_age_imitation. Not a prediction of the author's actual age."
        ),
        "grouping_method": (
            "AMORPH receives author_id as --group-column; rows sharing author_id stay in "
            "one internal train/validation/acceptance_gate/test split."
        ),
        "group_leakage_claim": (
            "The source card states official train/validation/test authors do not overlap. "
            "This run uses only official train and validates author-disjointness within its "
            "four internal splits using AMORPH's grouped splitter. Exact text repeated across "
            "authors is dropped before splitting."
        ),
        "privacy": {
            "raw_text_in_uploaded_artifact": False,
            "raw_author_ids_in_uploaded_artifact": False,
            "raw_dataset_in_uploaded_artifact": False,
            "temporary_csv_uploaded": False,
        },
        "e5_model_id": E5_MODEL_ID,
        "e5_model_revision": E5_MODEL_REVISION,
        "strategies": ["tfidf", "e5"],
        "acceptance_criteria": {
            "seed": 42,
            "noninferiority_margin": 0.01,
            "bootstrap_samples": args.bootstrap_samples,
            "min_gate_examples_per_class": 5,
            "max_near_duplicate_audit_rows": 6000,
            "min_macro_f1": None,
        },
        "status": "preparing",
    }
    write_json(output_dir / "public_grouped_dataset_manifest.json", manifest)
    run_exit_code: int | None = None

    try:
        with tempfile.TemporaryDirectory(prefix="amorph_author_profile_") as scratch:
            scratch_dir = Path(scratch)
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise RuntimeError("Install huggingface_hub to download the public JSONL.") from exc

            source_path = Path(hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                filename=DATASET_FILE,
                revision=args.dataset_revision,
                cache_dir=str(scratch_dir / "hf-cache"),
            ))
            manifest["dataset_revision_resolved"] = resolved_hub_revision(source_path)
            manifest["source_jsonl_sha256"] = sha256_file(source_path)
            manifest["source_jsonl_bytes"] = source_path.stat().st_size

            records = []
            with source_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at source line {line_number}") from exc
                    records.append(row)

            prepared, prep_audit = prepare_task_records(records)
            manifest["preparation_audit"] = prep_audit
            manifest["cleaned_training_csv_sha256"] = None
            input_csv = scratch_dir / "age_imitation_train.csv"
            write_csv(input_csv, prepared)
            manifest["cleaned_training_csv_sha256"] = sha256_file(input_csv)
            manifest["cleaned_training_rows_before_AMORPH_deduplication"] = len(prepared)
            manifest["retained_author_groups_before_AMORPH_duplicate_audit"] = prep_audit[
                "unique_author_groups_retained"
            ]
            manifest["status"] = "prepared"
            write_json(output_dir / "public_grouped_dataset_manifest.json", manifest)

            command = [
                sys.executable, "-m", "amorph_autotrainer.cli", "train", str(input_csv),
                "--text-column", TEXT_COLUMN,
                "--label-column", "label",
                "--group-column", GROUP_COLUMN,
                "--out", str(runs_dir),
                "--strategies", "tfidf,e5",
                "--max-candidates", "2",
                "--max-runtime-seconds", str(args.max_runtime_seconds),
                "--seed", "42",
                "--noninferiority-margin", "0.01",
                "--bootstrap-samples", str(args.bootstrap_samples),
                "--min-gate-examples-per-class", "5",
                "--max-near-duplicate-audit-rows", "6000",
                "--e5-model-revision", E5_MODEL_REVISION,
            ]
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            run_exit_code = result.returncode
            (output_dir / "train_stdout.log").write_text(result.stdout, encoding="utf-8")
            (output_dir / "train_stderr.log").write_text(result.stderr, encoding="utf-8")

            run_dirs = sorted(
                (p for p in runs_dir.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ) if runs_dir.exists() else []
            report_path = run_dirs[0] / "report.json" if run_dirs else None
            if report_path is None or not report_path.is_file():
                raise RuntimeError(
                    f"AMORPH did not produce report.json; CLI exit code={run_exit_code}."
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            manifest["amorph_run_id"] = report.get("run_id")
            manifest["amorph_run_report_sha256"] = sha256_file(report_path)
            manifest["amorph_report_relative_path"] = report_path.relative_to(output_dir).as_posix()
            manifest["amorph_status"] = report.get("status")
            manifest["selected_strategy"] = report.get("strategy")
            manifest["validation_metrics_all_candidates"] = report.get(
                "validation_metrics_all_candidates"
            )
            manifest["proofgate"] = report.get("proofgate")
            manifest["acceptance_gate_metrics"] = report.get("acceptance_gate_metrics")
            manifest["final_test_metrics"] = report.get("final_test_metrics")
            manifest["test_split_used_for_selection"] = report.get("test_split_used_for_selection")
            manifest["acceptance_gate_sufficient"] = report.get("acceptance_gate_sufficient")
            manifest["acceptance_gate_examples_per_class"] = report.get(
                "acceptance_gate_examples_per_class"
            )
            manifest["split_config"] = report.get("split_config")
            manifest["warnings"] = report.get("warnings")
            manifest["train_process_exit_code"] = run_exit_code
            manifest["status"] = "completed" if run_exit_code == 0 else "train_process_failed"
            write_json(output_dir / "public_grouped_dataset_manifest.json", manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error_message"] = str(exc)
        manifest["train_process_exit_code"] = (
            1 if run_exit_code is None or run_exit_code == 0 else run_exit_code
        )
        write_json(output_dir / "public_grouped_dataset_manifest.json", manifest)
        print(f"Grouped author-profiling run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if run_exit_code is None or run_exit_code == 0:
            run_exit_code = 1

    inventory = build_artifact_inventory(output_dir)
    write_json(output_dir / "artifact_inventory.json", inventory)
    print(json.dumps({
        "status": manifest.get("status"),
        "dataset_revision_resolved": manifest.get("dataset_revision_resolved"),
        "source_split_loaded": "train",
        "official_validation_test_used": False,
        "task": TASK_COLUMN,
        "group_column": GROUP_COLUMN,
        "rows": manifest.get("cleaned_training_rows_before_AMORPH_deduplication"),
        "groups": manifest.get("retained_author_groups_before_AMORPH_duplicate_audit"),
        "class_counts": manifest.get("preparation_audit", {}).get("label_counts_retained"),
        "amorph_run_id": manifest.get("amorph_run_id"),
        "selected_strategy": manifest.get("selected_strategy"),
        "proofgate": manifest.get("proofgate"),
        "artifact_inventory_files": inventory["files_count"],
    }, indent=2, ensure_ascii=False))
    return int(run_exit_code if run_exit_code is not None else 1)


if __name__ == "__main__":
    raise SystemExit(main())

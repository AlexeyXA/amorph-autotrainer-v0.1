#!/usr/bin/env python3
"""Prepare reproducible text/label/group_id CSVs from Promptfoo's public four-model evaluation.

This is a MODEL-SOURCE ATTRIBUTION benchmark (which model produced the response), not a
measure of answer correctness, usefulness, or economic value. The same question's four
responses share group_id so that a grouped split keeps them together.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_REVISION = "7a4f845d61262c4727911d1ec3abd9a974a64847"
SOURCE_URL = (
    "https://huggingface.co/datasets/promptfoo/political-questions/resolve/"
    + SOURCE_REVISION + "/political-bias-answers.csv"
)
DATASET_CARD = "https://huggingface.co/datasets/promptfoo/political-questions"
REPORT_URL = "https://www.promptfoo.dev/blog/grok-4-political-bias/"
LICENSE = "CC BY 4.0"
MODEL_COLUMNS = {
    "answer_gpt_4_1": "gpt-4.1",
    "answer_claude_opus_4": "claude-opus-4",
    "answer_gemini_2_5_pro": "gemini-2.5-pro",
    "answer_grok_4": "grok-4",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_pinned_source(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "political-bias-answers.csv"
    if not path.exists():
        req = urllib.request.Request(
            SOURCE_URL,
            headers={"User-Agent": "AMORPH-AutoTrainer-public-benchmark-preparer/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            data = response.read()
        if not data or b"question_id" not in data[:8192]:
            raise RuntimeError("Pinned Promptfoo source did not look like the expected CSV")
        path.write_bytes(data)
    if path.stat().st_size == 0:
        raise RuntimeError("Cached source CSV is empty")
    return path


def read_source(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"question_id", "question", "axis", *MODEL_COLUMNS.keys()}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(
                f"Promptfoo CSV schema changed; missing columns={missing}; "
                f"actual columns={reader.fieldnames}"
            )
        rows = list(reader)
    if len(rows) != 2500:
        raise RuntimeError(f"Expected 2500 question rows in pinned source, found {len(rows)}")
    ids = [row["question_id"].strip() for row in rows]
    if any(not x for x in ids):
        raise RuntimeError("Source contains an empty question_id")
    if len(set(ids)) != len(ids):
        raise RuntimeError("Source contains duplicate question_id values")
    return rows


def write_three_column_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "label", "group_id"], extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="prepared_promptfoo")
    parser.add_argument("--cache-dir", default=".cache/promptfoo")
    args = parser.parse_args()
    out = Path(args.output_dir)
    source = download_pinned_source(Path(args.cache_dir))
    source_rows = read_source(source)
    source_hash = sha256_file(source)

    response_rows: list[dict[str, str]] = []
    label_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    topic_rows: list[dict[str, str]] = []
    missing: list[str] = []

    for row in source_rows:
        qid = row["question_id"].strip()
        question = row["question"].strip()
        axis = row["axis"].strip()
        if not question or not axis:
            raise RuntimeError(f"Question {qid!r} has empty question or axis")
        topic_rows.append({"text": question, "label": axis, "group_id": qid})
        for column, model_label in MODEL_COLUMNS.items():
            answer = (row.get(column) or "").strip()
            if not answer:
                missing.append(f"{qid}:{column}")
                continue
            response_rows.append({"text": answer, "label": model_label, "group_id": qid})
            label_counts[model_label] += 1
            group_counts[qid] += 1

    if missing:
        raise RuntimeError(
            f"Refusing silent data loss: {len(missing)} model responses are empty; "
            f"first examples={missing[:10]}"
        )
    if len(response_rows) != len(source_rows) * len(MODEL_COLUMNS):
        raise RuntimeError("Unexpected output count; expected four model responses per question")
    if any(group_counts[qid] != 4 for qid in (r["question_id"].strip() for r in source_rows)):
        raise RuntimeError("Every group_id must contain exactly four model responses")

    response_path = out / "model_response_source_attribution.csv"
    topic_path = out / "political_question_topic.csv"
    write_three_column_csv(response_path, response_rows)
    write_three_column_csv(topic_path, topic_rows)

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Promptfoo Political Questions / Political Bias Answers",
        "source_revision": SOURCE_REVISION,
        "source_url": SOURCE_URL,
        "dataset_card_url": DATASET_CARD,
        "evaluation_report_url": REPORT_URL,
        "license": LICENSE,
        "source_file_bytes": source.stat().st_size,
        "source_file_sha256": source_hash,
        "source_question_rows": len(source_rows),
        "model_response_rows": len(response_rows),
        "unique_group_ids": len(group_counts),
        "model_response_label_counts": dict(sorted(label_counts.items())),
        "topic_label_counts": dict(sorted(Counter(r["label"] for r in topic_rows).items())),
        "outputs": {
            response_path.name: {
                "rows": len(response_rows),
                "sha256": sha256_file(response_path),
                "columns": ["text", "label", "group_id"],
                "task": "classify which of the four model families produced a response",
            },
            topic_path.name: {
                "rows": len(topic_rows),
                "sha256": sha256_file(topic_path),
                "columns": ["text", "label", "group_id"],
                "task": "classify a political question by its provided topic axis",
            },
        },
        "grouping_policy": (
            "group_id is question_id. All four model responses to the same question share one "
            "group so they must remain in the same train/validation/gate/test partition."
        ),
        "important_limitations": [
            "The source-attribution labels identify model family; they do not measure answer correctness, truthfulness, usefulness, safety or cost.",
            "The responses were generated in 2025 and represent GPT-4.1, Claude Opus 4, Gemini 2.5 Pro and Grok 4, not current model releases.",
            "The political-bias scores are model-judged, not independent human ground truth; they are not used as labels in these two CSVs.",
            "The public evaluation uses political questions, so it is a benchmark for source attribution/topic classification and grouped-split plumbing, not a substitute for the owner's production task.",
            "Source CSV remains in the local cache only; do not upload raw data to main or use official report examples as evidence of economic savings.",
        ],
    }
    manifest_path = out / "dataset_provenance.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "prepared",
        "source_revision": SOURCE_REVISION,
        "source_sha256": source_hash,
        "source_questions": len(source_rows),
        "response_rows": len(response_rows),
        "labels": dict(sorted(label_counts.items())),
        "group_ids": len(group_counts),
        "outputs": manifest["outputs"],
        "provenance": str(manifest_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

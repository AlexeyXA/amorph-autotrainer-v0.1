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
import math
import statistics
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


def write_dict_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def approximate_token_count(text: str) -> int:
    """Planning proxy only: rough English estimate, not any provider's billed tokenizer."""
    return math.ceil(len(text) / 4)


MODEL_RATE_ROWS = [
    {
        "provider": "OpenAI", "model_id": "gpt-6-luna", "display_model": "GPT-6 Luna",
        "input_usd_per_million": 0.10, "cached_input_usd_per_million": 0.01, "output_usd_per_million": 0.50,
        "mode": "standard_short_context", "price_window": "verified 2026-10-24; no explicit end date",
        "reference_profile_model": "gpt-4.1", "source_url": "https://developers.openai.com/api/docs/pricing",
        "source_note": "Current official Standard pricing; short-context rates. Cached input not applied in scenario cost."
    },
    {
        "provider": "OpenAI", "model_id": "gpt-6.1-sol", "display_model": "GPT-6.1 Sol",
        "input_usd_per_million": 2.00, "cached_input_usd_per_million": 0.10, "output_usd_per_million": 10.00,
        "mode": "standard_short_context", "price_window": "verified 2026-10-24; no explicit end date",
        "reference_profile_model": "gpt-4.1", "source_url": "https://developers.openai.com/api/docs/pricing",
        "source_note": "Current official Standard pricing; short-context rates. Cached input not applied in scenario cost."
    },
    {
        "provider": "OpenAI", "model_id": "gpt-6-astra", "display_model": "GPT-6 Astra",
        "input_usd_per_million": 10.00, "cached_input_usd_per_million": 1.00, "output_usd_per_million": 50.00,
        "mode": "standard_short_context", "price_window": "verified 2026-10-24; no explicit end date",
        "reference_profile_model": "gpt-4.1", "source_url": "https://developers.openai.com/api/docs/pricing",
        "source_note": "Current official Standard pricing; short-context rates. Cached input not applied in scenario cost."
    },
    {
        "provider": "Anthropic", "model_id": "claude-opus-5-5", "display_model": "Claude Opus 5.5",
        "input_usd_per_million": 4.00, "cached_input_usd_per_million": 0.20, "output_usd_per_million": 20.00,
        "mode": "standard", "price_window": "model page released 2026-09-22; verified 2026-10-24",
        "reference_profile_model": "claude-opus-4", "source_url": "https://platform.claude.com/docs/en/models/opus-5-5/overview",
        "source_note": "Official $4/$20 input/output; cache read $0.20; Batch API advertises 50% off input/output. Cache not applied."
    },
    {
        "provider": "xAI", "model_id": "grok-4.5", "display_model": "Grok 4.5",
        "input_usd_per_million": 2.00, "cached_input_usd_per_million": 0.30, "output_usd_per_million": 6.00,
        "mode": "standard_short_context", "price_window": "verified 2026-10-24; no explicit end date",
        "reference_profile_model": "grok-4", "source_url": "https://docs.x.ai/developers/pricing",
        "source_note": "Current official short-context input/cached/output $2/$0.30/$6; long-context rates apply from prompt >=200k tokens."
    },
    {
        "provider": "Google", "model_id": "gemini-3.8-flash", "display_model": "Gemini 3.8 Flash",
        "input_usd_per_million": 0.75, "cached_input_usd_per_million": 0.075, "output_usd_per_million": 3.75,
        "mode": "standard_intro", "price_window": "through 2026-12-31",
        "reference_profile_model": "gemini-2.5-pro", "source_url": "https://ai.google.dev/gemini-api/docs/pricing?hl=en",
        "source_note": "Intro standard price through 2026-12-31; standard input/output double on 2027-01-01. Cache storage billed separately."
    },
    {
        "provider": "Google", "model_id": "gemini-3.8-flash", "display_model": "Gemini 3.8 Flash",
        "input_usd_per_million": 1.50, "cached_input_usd_per_million": 0.15, "output_usd_per_million": 7.50,
        "mode": "standard_from_2027", "price_window": "from 2027-01-01",
        "reference_profile_model": "gemini-2.5-pro", "source_url": "https://ai.google.dev/gemini-api/docs/pricing?hl=en",
        "source_note": "Published standard price starting 2027-01-01; cache storage billed separately."
    },
]


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

    # Joined workload profile: actual Promptfoo question text + stored historical response.
    topic_by_group = {row["group_id"]: row for row in topic_rows}
    workload_rows: list[dict[str, Any]] = []
    profile_tokens: dict[str, dict[str, list[int]]] = {}
    for answer_row in response_rows:
        qid = answer_row["group_id"]
        question = topic_by_group[qid]["text"]
        historical_model = answer_row["label"]
        # Add 100 input tokens as a visible assumption for task/system instructions.
        input_tokens_est = approximate_token_count(question) + 100
        output_tokens_est = approximate_token_count(answer_row["text"])
        workload_rows.append({
            "text": answer_row["text"],
            "label": historical_model,
            "group_id": qid,
            "query_text": question,
            "query_axis": topic_by_group[qid]["label"],
            "input_chars": len(question),
            "response_chars": len(answer_row["text"]),
            "input_tokens_est": input_tokens_est,
            "output_tokens_est": output_tokens_est,
            "token_estimation_method": "ceil(characters/4) + 100 assumed instruction tokens; approximate, not billed usage",
            "historical_response_model": historical_model,
        })
        profile_tokens.setdefault(historical_model, {"input": [], "output": []})
        profile_tokens[historical_model]["input"].append(input_tokens_est)
        profile_tokens[historical_model]["output"].append(output_tokens_est)

    workload_path = out / "llm_tco_workload_profile_promptfoo.csv"
    workload_fields = [
        "text", "label", "group_id", "query_text", "query_axis", "input_chars", "response_chars",
        "input_tokens_est", "output_tokens_est", "token_estimation_method", "historical_response_model",
    ]
    write_dict_csv(workload_path, workload_fields, workload_rows)

    rate_card_path = out / "llm_tco_rate_card_2026-10-24.csv"
    rate_fields = list(MODEL_RATE_ROWS[0].keys())
    write_dict_csv(rate_card_path, rate_fields, MODEL_RATE_ROWS)

    tco_rows: list[dict[str, Any]] = []
    volume_scenarios = [2500, 10000, 100000, 1000000]
    historical_label_by_provider = {
        "OpenAI": "gpt-4.1",
        "Anthropic": "claude-opus-4",
        "xAI": "grok-4",
        "Google": "gemini-2.5-pro",
    }
    for rate in MODEL_RATE_ROWS:
        historical_label = historical_label_by_provider[rate["provider"]]
        token_profile = profile_tokens[historical_label]
        avg_input = statistics.mean(token_profile["input"])
        avg_output = statistics.mean(token_profile["output"])
        for nq in volume_scenarios:
            total_input = round(avg_input * nq)
            total_output = round(avg_output * nq)
            cost_usd = (
                total_input * float(rate["input_usd_per_million"])
                + total_output * float(rate["output_usd_per_million"])
            ) / 1_000_000
            tco_rows.append({
                "provider": rate["provider"],
                "model_id": rate["model_id"],
                "display_model": rate["display_model"],
                "pricing_mode": rate["mode"],
                "Nq": nq,
                "reference_dataset_queries": len(source_rows),
                "Nq_is_actual_owner_volume": False,
                "historical_response_profile": historical_label,
                "avg_input_tokens_est": round(avg_input, 2),
                "avg_output_tokens_est": round(avg_output, 2),
                "total_input_tokens_est": total_input,
                "total_output_tokens_est": total_output,
                "input_usd_per_million": rate["input_usd_per_million"],
                "output_usd_per_million": rate["output_usd_per_million"],
                "estimated_token_api_cost_usd": round(cost_usd, 6),
                "estimated_token_api_cost_usd_per_query": round(cost_usd / nq, 8),
                "estimate_method": "historical Promptfoo text-length profile; ceil(chars/4)+100 assumed instruction tokens; applies current rate to historical output length",
                "excluded_costs": "tools, retries, caching, cache storage/write, current-model reasoning tokens/output-length changes, human review, hosting, taxes, success/error costs",
                "price_source_url": rate["source_url"],
            })
    tco_path = out / "llm_tco_scenarios_promptfoo.csv"
    tco_fields = list(tco_rows[0].keys())
    write_dict_csv(tco_path, tco_fields, tco_rows)

    # Apples-to-apples scenario table: all providers see the same estimated
    # input length and one of a few fixed output-length assumptions.
    # This is distinct from provider-specific historical response profiles above.
    common_avg_input = statistics.mean(approximate_token_count(row["question"].strip()) + 100 for row in source_rows)
    pooled_avg_output = statistics.mean(
        approximate_token_count(answer_row["text"]) for answer_row in response_rows
    )
    common_output_profiles = [
        ("pooled_historical_mean_output", round(pooled_avg_output, 2)),
        ("fixed_50_output_tokens", 50.0),
        ("fixed_100_output_tokens", 100.0),
    ]
    common_rows: list[dict[str, Any]] = []
    for rate in MODEL_RATE_ROWS:
        for output_profile, output_tokens in common_output_profiles:
            for nq in volume_scenarios:
                total_input = round(common_avg_input * nq)
                total_output = round(output_tokens * nq)
                cost_usd = (
                    total_input * float(rate["input_usd_per_million"])
                    + total_output * float(rate["output_usd_per_million"])
                ) / 1_000_000
                common_rows.append({
                    "provider": rate["provider"],
                    "model_id": rate["model_id"],
                    "display_model": rate["display_model"],
                    "pricing_mode": rate["mode"],
                    "output_profile": output_profile,
                    "Nq": nq,
                    "reference_dataset_queries": len(source_rows),
                    "Nq_is_actual_owner_volume": False,
                    "avg_input_tokens_est": round(common_avg_input, 2),
                    "output_tokens_per_query_assumption": output_tokens,
                    "total_input_tokens_est": total_input,
                    "total_output_tokens_est": total_output,
                    "input_usd_per_million": rate["input_usd_per_million"],
                    "output_usd_per_million": rate["output_usd_per_million"],
                    "estimated_token_api_cost_usd": round(cost_usd, 6),
                    "estimated_token_api_cost_usd_per_query": round(cost_usd / nq, 8),
                    "estimate_method": (
                        "common input profile: mean ceil(question characters/4)+100 assumed instruction tokens; "
                        "same fixed output-token assumption for every provider; not billed usage"
                    ),
                    "excluded_costs": (
                        "tools, retries, caching, cache storage/write, reasoning/output-length variation, "
                        "human review, hosting, taxes, success/error costs"
                    ),
                    "price_source_url": rate["source_url"],
                })
    common_tco_path = out / "llm_tco_common_profile_scenarios_promptfoo.csv"
    common_tco_fields = list(common_rows[0].keys())
    write_dict_csv(common_tco_path, common_tco_fields, common_rows)

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
            workload_path.name: {
                "rows": len(workload_rows),
                "sha256": sha256_file(workload_path),
                "columns": workload_fields,
                "task": "join each historical response to its public benchmark prompt and estimate a token-length workload profile",
            },
            rate_card_path.name: {
                "rows": len(MODEL_RATE_ROWS),
                "sha256": sha256_file(rate_card_path),
                "columns": rate_fields,
                "task": "official current list price snapshot checked 2026-10-24",
            },
            tco_path.name: {
                "rows": len(tco_rows),
                "sha256": sha256_file(tco_path),
                "columns": tco_fields,
                "task": "API token-cost scenarios using provider-specific historical output-length profiles; scenarios are not actual owner traffic",
            },
            common_tco_path.name: {
                "rows": len(common_rows),
                "sha256": sha256_file(common_tco_path),
                "columns": common_tco_fields,
                "task": "apples-to-apples price comparison using the same estimated input and fixed output-token profiles for all current candidates",
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
        "tco_scenario_rows": len(tco_rows),
        "common_profile_tco_scenario_rows": len(common_rows),
        "common_profile_input_tokens_est": round(common_avg_input, 2),
        "common_profile_output_token_assumptions": common_output_profiles,
        "token_estimation": "ceil(chars/4) + 100 assumed instruction tokens; not provider billed usage",
        "provenance": str(manifest_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

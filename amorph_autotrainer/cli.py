from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .cost import CostConfig
from .data import load_records
from .demo import create_demo_csv
from .experiment_db import ExperimentDB
from .pipeline import RuntimeBudgetExceededError, run_training
from .strategies import E5Candidate, extract_loaded_features, load_strategy_model, predict_loaded


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amorph-autotrainer",
        description="Budget-aware AutoTrainer MVP for classification and feature extraction.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    demo = subs.add_parser("demo", help="Create synthetic CSV for smoke tests; not a benchmark dataset")
    demo.add_argument("--output", default="data/demo_classification.csv")
    demo.add_argument("--examples-per-class", type=int, default=50)

    train = subs.add_parser("train", help="Train/evaluate candidates from a CSV file")
    train.add_argument("csv_path")
    train.add_argument("--text-column", default="text")
    train.add_argument("--label-column", default="label")
    train.add_argument(
        "--group-column", default=None,
        help="Optional CSV column identifying a user/source/project; rows sharing a value stay in one split",
    )
    train.add_argument("--out", default="runs")
    train.add_argument("--strategies", default="tfidf", help="Comma-separated: tfidf,e5,hybrid_tfidf_e5; tfidf is always required")
    train.add_argument("--e5-model-id", default=E5Candidate.model_id)
    train.add_argument("--e5-model-revision", default=E5Candidate.default_model_revision,
                       help="Pinned Hugging Face model revision (commit SHA); pass 'main' only to opt into a mutable branch")
    train.add_argument("--max-runtime-seconds", type=float, default=7200)
    train.add_argument("--max-candidates", type=int, default=3)
    train.add_argument("--expected-queries", type=int, default=1000)
    train.add_argument("--compute-eur-per-hour", "--eur-per-hour-compute", dest="compute_eur_per_hour", type=float, default=0.0,
                       help="Compute cost rate; zero means assumed-free compute")
    train.add_argument("--fixed-data-cost-eur", type=float, default=0.0)
    train.add_argument("--maintenance-cost-eur", type=float, default=0.0)
    train.add_argument("--human-hours", type=float, default=0.0)
    train.add_argument("--human-eur-per-hour", "--eur-per-human-hour", dest="human_eur_per_hour", type=float, default=0.0,
                       help="Human labor cost rate; zero means assumed-free labor")
    train.add_argument("--e5-classifier-c", type=float, default=1.0,
                       help="E5 LogisticRegression C; default preserves current C=1.0")
    train.add_argument("--e5-class-weight", choices=("none", "balanced"), default="none",
                       help="E5 LogisticRegression class weight for a controlled A/B experiment")
    train.add_argument("--hybrid-classifier-c", type=float, default=1.0,
                       help="Hybrid TF-IDF+E5 LogisticRegression C (experimental strategy only)")
    train.add_argument("--hybrid-class-weight", choices=("none", "balanced"), default="none",
                       help="Hybrid TF-IDF+E5 LogisticRegression class weight")
    train.add_argument("--min-macro-f1", type=float, default=None)
    train.add_argument("--noninferiority-margin", type=float, default=0.01)
    train.add_argument("--bootstrap-samples", type=int, default=500)
    train.add_argument("--min-gate-examples-per-class", type=int, default=5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--near-duplicate-threshold", type=float, default=0.985)
    train.add_argument("--max-near-duplicate-audit-rows", type=int, default=5000)
    train.add_argument("--export-features", action="store_true", help="Export selected model's split features; vectors may be sensitive")

    predict = subs.add_parser("predict", help="Predict labels from an input CSV")
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--text-column", default="text")
    predict.add_argument("--output", default="predictions.csv")

    features = subs.add_parser("features", help="Extract features from a CSV using a saved model bundle")
    features.add_argument("--model-dir", required=True)
    features.add_argument("--input", required=True)
    features.add_argument("--text-column", default="text")
    features.add_argument("--output", default="features.npz")

    runs = subs.add_parser("runs", help="List recent experiments in SQLite database")
    runs.add_argument("--db", default="runs/experiments.sqlite3")
    runs.add_argument("--limit", type=int, default=20)

    return parser


def _read_text_csv(path: str | Path, text_column: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if text_column not in (reader.fieldnames or []):
            raise ValueError(f"Text column '{text_column}' not present in {path}")
        rows = []
        for row in reader:
            text = (row.get(text_column) or "").strip()
            if text:
                rows.append({"text": text, **row})
        if not rows:
            raise ValueError("Input CSV contains no non-empty text rows")
        return rows


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            if args.examples_per_class < 5:
                raise ValueError("examples-per-class must be >= 5")
            path = create_demo_csv(args.output, args.examples_per_class)
            print(f"Synthetic smoke-test data created: {path}")
            print("Warning: this data is artificial and must not be used to claim real-world model quality.")
            return 0

        if args.command == "train":
            strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
            config = CostConfig(
                max_runtime_seconds=args.max_runtime_seconds,
                max_candidates=args.max_candidates,
                expected_queries=args.expected_queries,
                compute_eur_per_hour=args.compute_eur_per_hour,
                fixed_data_cost_eur=args.fixed_data_cost_eur,
                maintenance_cost_eur=args.maintenance_cost_eur,
                human_hours=args.human_hours,
                human_eur_per_hour=args.human_eur_per_hour,
                min_macro_f1=args.min_macro_f1,
                noninferiority_margin=args.noninferiority_margin,
                bootstrap_samples=args.bootstrap_samples,
                min_gate_examples_per_class=args.min_gate_examples_per_class,
                random_seed=args.seed,
            )
            result = run_training(
                csv_path=args.csv_path,
                output_dir=args.out,
                text_column=args.text_column,
                label_column=args.label_column,
                group_column=args.group_column,
                strategies=strategies,
                config=config,
                export_features=args.export_features,
                e5_model_id=args.e5_model_id,
                e5_model_revision=args.e5_model_revision,
                e5_classifier_c=args.e5_classifier_c,
                e5_class_weight=None if args.e5_class_weight == "none" else args.e5_class_weight,
                hybrid_classifier_c=args.hybrid_classifier_c,
                hybrid_class_weight=None if args.hybrid_class_weight == "none" else args.hybrid_class_weight,
                near_duplicate_threshold=args.near_duplicate_threshold,
                max_near_duplicate_audit_rows=args.max_near_duplicate_audit_rows,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if result.get("runtime_budget_exceeded"):
                return 3
            if not result.get("acceptance_gate_sufficient", False):
                return 4
            return 2 if result["absolute_target_passed"] is False else 0

        if args.command == "predict":
            rows = _read_text_csv(args.input, args.text_column)
            loaded = load_strategy_model(args.model_dir)
            texts = [row["text"] for row in rows]
            predictions = predict_loaded(loaded, texts)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["text", "predicted_label"])
                writer.writeheader()
                writer.writerows({"text": text, "predicted_label": str(label)}
                                 for text, label in zip(texts, predictions))
            print(f"Predictions saved: {output}")
            return 0

        if args.command == "features":
            rows = _read_text_csv(args.input, args.text_column)
            loaded = load_strategy_model(args.model_dir)
            matrix = extract_loaded_features(loaded, [row["text"] for row in rows])
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(matrix, "tocsr"):
                from scipy.sparse import save_npz
                save_npz(output, matrix.tocsr())
                feature_type = "sparse"
            else:
                np.savez_compressed(output, features=matrix)
                feature_type = "dense"
            print(json.dumps({"output": str(output), "feature_type": feature_type,
                              "shape": list(matrix.shape)}, ensure_ascii=False))
            return 0

        if args.command == "runs":
            db = ExperimentDB(args.db)
            print(json.dumps(db.list_runs(args.limit), indent=2, ensure_ascii=False))
            return 0

        parser.error(f"Unhandled command: {args.command}")
    except RuntimeBudgetExceededError as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

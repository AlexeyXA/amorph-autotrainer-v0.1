#!/usr/bin/env python3
"""Run AMORPH on pinned/public intent datasets; never train on official test.

BANKING77: PolyAI, CC BY 4.0, fixed Hugging Face revision.
CLINC150: UCI official data_full.json archive, CC BY 4.0.
The official train split enters AMORPH's internal train/validation/gate/test split.
The upstream official test is evaluated only after internal selection and is report-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BANKING77_ID = "PolyAI-LDN/task-specific-datasets/banking_data"
BANKING77_REVISION = "9d081458ff52e53cf7e848f414e6e9344e4e6696"
BANKING77_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/9d081458ff52e53cf7e848f414e6e9344e4e6696/banking_data"
BANKING77_TRAIN_SHA256 = "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
BANKING77_TEST_SHA256 = "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"
CLINC150_URL = "https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_full.json"
CLINC150_CARD = "https://archive.ics.uci.edu/dataset/570/clinc150"
CLINC150_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
CLINC150_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_banking77(cache: Path):
    import csv
    import io
    import requests

    cache.mkdir(parents=True, exist_ok=True)

    def fetch(name, expected_hash):
        path = cache / name
        if not path.exists():
            url = f"{BANKING77_URL}/{name}"
            response = requests.get(url, timeout=(20, 120))
            response.raise_for_status()
            payload = response.content
            actual_hash = hashlib.sha256(payload).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(f"BANKING77 {name} checksum mismatch: expected {expected_hash}, got {actual_hash}")
            path.write_bytes(payload)
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Cached BANKING77 {name} checksum mismatch: expected {expected_hash}, got {actual_hash}")
        return path

    train_path = fetch("train.csv", BANKING77_TRAIN_SHA256)
    test_path = fetch("test.csv", BANKING77_TEST_SHA256)

    def rows(path):
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "text" not in reader.fieldnames or "category" not in reader.fieldnames:
                raise RuntimeError(f"Unexpected BANKING77 CSV header in {path}: {reader.fieldnames}")
            result = []
            for item in reader:
                text = str(item.get("text") or "").strip()
                label = str(item.get("category") or "").strip()
                if text and label:
                    result.append({"text": text, "label": label})
            return result

    train_rows, test_rows = rows(train_path), rows(test_path)
    if len(train_rows) != 10003 or len(test_rows) != 3080:
        raise RuntimeError(f"BANKING77 row count mismatch: train={len(train_rows)}, test={len(test_rows)}")
    return train_rows, test_rows, {
        "dataset_id": BANKING77_ID,
        "revision": BANKING77_REVISION,
        "source_url": BANKING77_URL,
        "license": "CC BY 4.0",
        "source_file_sha256": {
            "train.csv": BANKING77_TRAIN_SHA256,
            "test.csv": BANKING77_TEST_SHA256,
        },
        "train_split": "official train only",
        "validation_split": "not used; AMORPH creates its own internal split from official train",
        "test_split": "official test; report-only after selector and Proof Gate",
        "limitations": [
            "No reliable customer/session/source group ID is included; internal split is not group-disjoint.",
            "English banking intent benchmark; not the owner's production traffic.",
        ],
    }

def load_clinc150(cache: Path):
    import requests

    cache.mkdir(parents=True, exist_ok=True)
    source_file = cache / "data_full.json"
    if not source_file.exists():
        response = requests.get(CLINC150_URL, timeout=(20, 120))
        response.raise_for_status()
        payload = response.content
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != CLINC150_SHA256:
            raise RuntimeError(
                f"CLINC150 source checksum mismatch: expected {CLINC150_SHA256}, got {actual_hash}"
            )
        source_file.write_bytes(payload)
    source_hash = sha256_file(source_file)
    if source_hash != CLINC150_SHA256:
        raise RuntimeError(
            f"Cached CLINC150 checksum mismatch: expected {CLINC150_SHA256}, got {source_hash}"
        )
    raw = json.loads(source_file.read_text(encoding="utf-8"))

    def rows(items, split_name, force_label=None):
        out = []
        for item in items:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text, label = item[0], (force_label if force_label is not None else item[1])
            elif isinstance(item, dict):
                text = item.get("text", item.get("utterance"))
                label = force_label or item.get("intent", item.get("label"))
            elif isinstance(item, str):
                if force_label is None:
                    raise RuntimeError(
                        f"Unlabelled CLINC150 row found in {split_name}; expected split-specific label"
                    )
                text, label = item, force_label
            else:
                raise RuntimeError(f"Unexpected CLINC row in {split_name}: {type(item)}")
            text, label = str(text or "").strip(), str(label or "").strip()
            if text and label:
                out.append({"text": text, "label": label})
        return out

    if "train" not in raw or "test" not in raw:
        raise RuntimeError(f"Unexpected CLINC split keys: {list(raw)}")
    train_rows = rows(raw["train"], "train")
    test_rows = rows(raw["test"], "test")
    if "oos_train" in raw:
        train_rows.extend(rows(raw["oos_train"], "oos_train", force_label="oos"))
    if "oos_test" in raw:
        test_rows.extend(rows(raw["oos_test"], "oos_test", force_label="oos"))
    return train_rows, test_rows, {
        "dataset_id": "UCI CLINC150 data_full.json",
        "revision": CLINC150_REVISION,
        "source_url": CLINC150_URL,
        "dataset_card_url": CLINC150_CARD,
        "license": "CC BY 4.0 (UCI dataset card)",
        "source_file_sha256": source_hash,
        "source_file_bytes": source_file.stat().st_size,
        "source_file": "data/data_full.json",
        "train_split": "official in-domain train plus explicit oos_train",
        "official_validation_rows_not_used": len(raw.get("val", [])) + len(raw.get("oos_val", [])),
        "test_split": "official in-domain test plus explicit oos_test; report-only after selection and Proof Gate",
        "oos_training_rows_included": len(raw.get("oos_train", [])),
        "oos_test_rows_included": len(raw.get("oos_test", [])),
        "limitations": [
            "OOS examples are treated as the explicit benchmark class 'oos'; this is closed-set classification, not open-world OOD detection.",
            "No reliable customer/session/source group ID is included; internal split is not group-disjoint.",
            "Fetched file is pinned to a commit and required to match the published SHA-256 before parsing.",
        ],
    }

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["banking77", "clinc150"], required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--with-e5", action="store_true")
    p.add_argument("--max-runtime-seconds", type=float, default=7200)
    p.add_argument("--expected-queries", type=int, default=1000,
                   help="Scenario value for the current runtime proxy, not observed owner volume")
    args = p.parse_args()
    started = time.perf_counter()
    out = Path(args.output_dir).resolve()
    cache, data = out / "cache", out / "data"
    data.mkdir(parents=True, exist_ok=True)

    train_rows, test_rows, provenance = load_banking77(cache) if args.dataset == "banking77" else load_clinc150(cache)
    if not train_rows or not test_rows:
        raise RuntimeError("Dataset train and test must both be non-empty")
    train_labels = {r["label"] for r in train_rows}
    test_labels = {r["label"] for r in test_rows}
    if len(train_labels) < 2:
        raise RuntimeError("At least two train labels are required")

    train_csv, test_csv = data / "official_train.csv", data / "official_test.csv"
    write_csv(train_csv, train_rows)
    write_csv(test_csv, test_rows)
    provenance.update({
        "train_rows": len(train_rows),
        "official_test_rows": len(test_rows),
        "train_classes": len(train_labels),
        "official_test_classes": len(test_labels),
        "official_test_labels_without_training_support": sorted(test_labels - train_labels),
        "prepared_train_csv_sha256": sha256_file(train_csv),
        "prepared_official_test_csv_sha256": sha256_file(test_csv),
        "prepared_csv_files_are_public_dataset_derivatives": True,
    })
    write_json(out / "dataset_provenance.json", provenance)

    import joblib
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, recall_score
    from amorph_autotrainer.cost import CostConfig
    from amorph_autotrainer.pipeline import run_training
    from amorph_autotrainer.strategies import E5Candidate

    strategies = ["tfidf", "e5"] if args.with_e5 else ["tfidf"]
    cfg = CostConfig(
        max_runtime_seconds=args.max_runtime_seconds,
        max_candidates=len(strategies),
        expected_queries=args.expected_queries,
        compute_eur_per_hour=0.0,
        fixed_data_cost_eur=0.0,
        maintenance_cost_eur=0.0,
        human_hours=0.0,
        human_eur_per_hour=0.0,
        noninferiority_margin=0.01,
        bootstrap_samples=500,
        min_gate_examples_per_class=5,
        random_seed=42,
    )
    train_result = run_training(
        csv_path=train_csv,
        output_dir=out / "runs",
        text_column="text",
        label_column="label",
        group_column=None,
        strategies=strategies,
        config=cfg,
        e5_model_revision=E5Candidate.default_model_revision,
    )
    run_dir = Path(train_result["run_dir"])
    x_test = [r["text"] for r in test_rows]
    y_true = np.asarray([r["label"] for r in test_rows], dtype=object)
    fixed_labels = sorted(train_labels)
    metrics = {}

    train_label_counts = {label: sum(1 for row in train_rows if row["label"] == label) for label in train_labels}
    all_report_labels = sorted(train_labels | test_labels)

    def score(predictions):
        y_pred = np.asarray(predictions, dtype=object)
        precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=all_report_labels, average=None, zero_division=0
        )
        per_class_metrics = []
        for i, label in enumerate(all_report_labels):
            per_class_metrics.append({
                "label": label,
                "training_support": int(train_label_counts.get(label, 0)),
                "official_test_support": int(support[i]),
                "predicted_support": int(np.count_nonzero(y_pred == label)),
                "precision": float(precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
            })
        oos_metrics = next((item for item in per_class_metrics if item["label"] == "oos"), None)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1_fixed_training_label_universe": float(f1_score(
                y_true, y_pred, labels=fixed_labels, average="macro", zero_division=0
            )),
            "balanced_accuracy_fixed_training_label_universe": float(recall_score(
                y_true, y_pred, labels=fixed_labels, average="macro", zero_division=0
            )),
            "macro_f1_official_test_label_universe": float(f1_score(
                y_true, y_pred, labels=sorted(test_labels), average="macro", zero_division=0
            )),
            "balanced_accuracy_official_test_label_universe": float(recall_score(
                y_true, y_pred, labels=sorted(test_labels), average="macro", zero_division=0
            )),
            "official_test_rows": len(y_true),
            "training_label_count": len(fixed_labels),
            "official_test_labels_absent_from_training": sorted(test_labels - train_labels),
            "per_class_metrics": per_class_metrics,
            "oos_metrics": oos_metrics,
            "scope": "report_only_after_selection_and_gate",
        }

    tfidf_path = run_dir / "candidates" / "tfidf" / "model.joblib"
    metrics["tfidf"] = score(joblib.load(tfidf_path).predict(x_test))
    e5_path = run_dir / "candidates" / "e5" / "classifier.joblib"
    if args.with_e5 and e5_path.exists():
        meta = json.loads((run_dir / "candidates" / "e5" / "strategy.json").read_text(encoding="utf-8"))
        encoder = E5Candidate._load_encoder(
            meta.get("model_id", E5Candidate.model_id),
            meta.get("model_revision", E5Candidate.default_model_revision),
        )
        classifier = joblib.load(e5_path)
        metrics["e5"] = score(classifier.predict(E5Candidate._encode(encoder, x_test)))

    result = {
        "project": "AMORPH AutoTrainer",
        "benchmark_kind": "public labelled intent-classification task; not owner-domain evidence",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_provenance": provenance,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "requested_strategies": strategies,
        "selected_strategy": train_result.get("selected_strategy"),
        "internal_run": train_result,
        "external_official_test_metrics_by_strategy": metrics,
        "official_test_policy": "report-only; not used by selection or Proof Gate",
        "economic_claims": {
            "money_tco_optimized": False,
            "reason": "current selector minimizes runtime proxy; zero cost rates are assumed-free placeholders",
            "expected_queries_is_observed_owner_volume": False,
        },
        "limitations": [
            "Public benchmarks do not establish production performance on the owner's task.",
            "No source/customer group IDs; near-duplicate audit may be skipped above its row cap.",
            "Macro-F1/accuracy do not measure business error costs, hallucinations, safety or forgotten capabilities.",
            "No GPT/Grok/Gemini hosted API is called; candidates are local TF-IDF and optional pinned E5.",
        ],
    }
    write_json(out / "benchmark_result.json", result)
    elapsed = round(time.perf_counter() - started, 3)
    execution = {
        "status": "result_written",
        "dataset": args.dataset,
        "source_commit": None,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "train_csv_sha256": sha256_file(train_csv),
        "official_test_csv_sha256": sha256_file(test_csv),
        "benchmark_result_sha256": sha256_file(out / "benchmark_result.json"),
        "dataset_provenance_sha256": sha256_file(out / "dataset_provenance.json"),
        "selected_strategy": train_result.get("selected_strategy"),
        "official_test_metrics": metrics,
    }
    write_json(out / "execution_manifest.json", execution)
    print(json.dumps({
        "dataset": args.dataset,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "class_count": len(train_labels),
        "selected_strategy": train_result.get("selected_strategy"),
        "official_test_metrics": metrics,
        "elapsed_seconds": elapsed,
        "money_tco_optimized": False,
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

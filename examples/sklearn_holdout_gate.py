#!/usr/bin/env python3
"""Standalone scikit-learn example: compare two predeclared classifiers with Proof Gate.

This example trains outside AMORPH AutoTrainer. Its single holdout is an acceptance
gate, not a final test set and not a tuning set. Do not repeatedly tune against it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from amorph_autotrainer.proofgate import PROOFGATE_VERSION, evaluate, write_report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_indices(indices: Sequence[int]) -> str:
    canonical = json.dumps(sorted(int(i) for i in indices), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_strings(values: Sequence[str]) -> str:
    canonical = json.dumps(sorted(str(value) for value in values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_csv(
    path: Path,
    *,
    text_column: str,
    label_column: str,
    group_column: str | None,
) -> tuple[list[str], np.ndarray, list[str] | None]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns:
            raise ValueError("CSV must contain a header row")
        if len(set(columns)) != len(columns):
            raise ValueError("CSV contains duplicate column names; make the header unique")
        required = {text_column, label_column}
        if group_column:
            required.add(group_column)
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        texts: list[str] = []
        labels: list[str] = []
        groups: list[str] | None = [] if group_column else None
        for row_number, row in enumerate(reader, start=2):
            if row.get(None):
                raise ValueError(f"extra CSV fields in row {row_number}; make every row match the header")
            text = (row.get(text_column) or "").strip()
            label = (row.get(label_column) or "").strip()
            if not text:
                raise ValueError(f"empty text in CSV row {row_number}; fix the source data explicitly")
            if not label:
                raise ValueError(f"empty label in CSV row {row_number}; fix the source data explicitly")
            texts.append(text)
            labels.append(label)
            if groups is not None:
                group_value = (row.get(group_column) or "").strip()
                if not group_value:
                    raise ValueError(
                        f"empty group ID in CSV row {row_number}; group-safe split requires every row to have one"
                    )
                groups.append(group_value)

    if len(texts) < 4:
        raise ValueError(f"need at least 4 valid rows; received {len(texts)}")
    if len(set(labels)) < 2:
        raise ValueError("need at least two target classes")
    counts = Counter(labels)
    too_rare = {label: count for label, count in counts.items() if count < 2}
    if too_rare:
        raise ValueError(
            "every class needs at least two rows for a holdout split; "
            f"rare classes: {too_rare}"
        )
    return texts, np.asarray(labels, dtype=object), groups


def split_rows(
    labels: np.ndarray,
    *,
    gate_size: float,
    seed: int,
    groups: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    indices = np.arange(len(labels))
    if groups is None:
        train_indices, gate_indices = train_test_split(
            indices,
            test_size=gate_size,
            random_state=seed,
            stratify=labels,
        )
        method = "stratified_row_holdout"
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=gate_size, random_state=seed)
        train_indices, gate_indices = next(
            splitter.split(indices, labels, groups=np.asarray(groups, dtype=object))
        )
        train_groups = {groups[int(i)] for i in train_indices}
        gate_groups = {groups[int(i)] for i in gate_indices}
        overlap = sorted(train_groups & gate_groups)
        if overlap:
            raise RuntimeError(f"group leakage detected across train/gate: {overlap[:5]}")
        method = "group_shuffle_holdout"

    # Stable row order improves reproducibility and makes row-index hashes canonical.
    train_indices = np.sort(train_indices)
    gate_indices = np.sort(gate_indices)

    all_classes = set(labels.tolist())
    train_classes = set(labels[train_indices].tolist())
    gate_classes = set(labels[gate_indices].tolist())
    if train_classes != all_classes:
        missing = sorted(all_classes - train_classes)
        raise ValueError(
            "invalid holdout: some source classes are absent from training: "
            f"{missing}. Collect more examples or predeclare a different split."
        )
    if gate_classes != all_classes:
        missing = sorted(all_classes - gate_classes)
        raise ValueError(
            "invalid holdout: some source classes are absent from the acceptance gate: "
            f"{missing}. Increase the holdout or collect more examples; do not accept a partial-taxonomy gate."
        )
    return train_indices, gate_indices, method


def build_baseline() -> Pipeline:
    """Predeclared baseline: word TF-IDF + Logistic Regression."""
    return Pipeline(
        [
            (
                "features",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000)),
        ]
    )


def build_candidate() -> Pipeline:
    """Predeclared challenger: word + character TF-IDF, class-balanced linear head."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train two fixed scikit-learn classifiers on one train split and evaluate "
            "their paired predictions on a held-out acceptance gate with Proof Gate."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Input CSV; rows are not silently dropped")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--group-column",
        help="Optional group/user/source ID; when set, groups never cross the train/gate split",
    )
    parser.add_argument("--gate-size", type=float, default=0.25, help="Holdout fraction, in (0, 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-examples-per-class", type=int, default=5)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--min-macro-f1", type=float, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("proofgate-report.json"))
    args = parser.parse_args(argv)

    try:
        if not 0 < args.gate_size < 0.5:
            raise ValueError("--gate-size must be in (0, 0.5)")
        if args.min_examples_per_class < 1:
            raise ValueError("--min-examples-per-class must be >= 1")

        texts, labels, groups = load_csv(
            args.csv_path,
            text_column=args.text_column,
            label_column=args.label_column,
            group_column=args.group_column,
        )
        train_indices, gate_indices, split_method = split_rows(
            labels, gate_size=args.gate_size, seed=args.seed, groups=groups
        )
        x_train = [texts[int(i)] for i in train_indices]
        y_train = labels[train_indices]
        x_gate = [texts[int(i)] for i in gate_indices]
        y_gate = labels[gate_indices]

        baseline = build_baseline()
        candidate = build_candidate()
        baseline.fit(x_train, y_train)
        candidate.fit(x_train, y_train)
        pred_baseline = baseline.predict(x_gate)
        pred_candidate = candidate.predict(x_gate)

        dataset_sha256 = sha256_file(args.csv_path)
        gate_rows_sha256 = sha256_indices(gate_indices)
        train_rows_sha256 = sha256_indices(train_indices)
        metadata = {
            "dataset_sha256": dataset_sha256,
            "dataset_row_count": int(len(texts)),
            "train_row_count": int(len(train_indices)),
            "gate_row_count": int(len(gate_indices)),
            "train_row_indices_sha256": train_rows_sha256,
            "gate_row_indices_sha256": gate_rows_sha256,
            "split_method": split_method,
            "group_column": args.group_column,
            "train_group_ids_sha256": (
                sha256_strings([groups[int(i)] for i in train_indices]) if groups is not None else None
            ),
            "gate_group_ids_sha256": (
                sha256_strings([groups[int(i)] for i in gate_indices]) if groups is not None else None
            ),
            "seed": args.seed,
            "baseline_model": "word_tfidf_1_2 + LogisticRegression(C=1.0, class_weight=None)",
            "candidate_model": "word_char_tfidf + LogisticRegression(C=1.0, class_weight=balanced)",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "proofgate_version": PROOFGATE_VERSION,
            "holdout_role": "acceptance_gate_only_not_final_test",
        }
        report = evaluate(
            y_gate,
            pred_baseline,
            pred_candidate,
            margin=args.margin,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            min_examples_per_class=args.min_examples_per_class,
            min_macro_f1=args.min_macro_f1,
            metadata=metadata,
        )
        payload = write_report(report, args.output)
    except (OSError, csv.Error, ValueError, RuntimeError) as exc:
        print(f"sklearn_holdout_gate: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "subtype": payload["subtype"],
                "powered": payload["sample"]["powered"],
                "report_sha256": payload["report_sha256"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    # A valid INCONCLUSIVE report is still a successful evaluation run. Consumers
    # must inspect the reported verdict, not infer acceptance from exit status zero.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

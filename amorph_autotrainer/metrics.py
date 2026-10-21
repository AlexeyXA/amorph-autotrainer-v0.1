from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "num_examples": int(len(y_true)),
        "labels": labels,
    }


def paired_stratified_bootstrap_macro_f1(
    y_true: np.ndarray,
    baseline_pred: np.ndarray,
    candidate_pred: np.ndarray,
    *,
    n_bootstrap: int = 500,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Paired, class-stratified bootstrap of candidate-minus-baseline macro-F1.

    This estimates test-sample uncertainty conditional on the fitted models; it does
    not capture variability from retraining on different training samples.
    """
    if not (len(y_true) == len(baseline_pred) == len(candidate_pred)):
        raise ValueError("Bootstrap inputs must have equal lengths")
    if len(y_true) == 0:
        raise ValueError("Bootstrap requires at least one test example")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100 for this gate")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    labels = np.asarray(sorted(set(y_true.tolist())))
    groups = [np.flatnonzero(y_true == label) for label in labels]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        sampled = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in groups
        ])
        base_f1 = f1_score(y_true[sampled], baseline_pred[sampled], labels=labels,
                           average="macro", zero_division=0)
        candidate_f1 = f1_score(y_true[sampled], candidate_pred[sampled], labels=labels,
                                average="macro", zero_division=0)
        deltas[b] = candidate_f1 - base_f1

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(deltas, [alpha, 1.0 - alpha])
    return {
        "delta_macro_f1_mean": float(deltas.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
    }


def gate_candidate(
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    comparison: dict[str, Any],
    *,
    min_macro_f1: float | None = None,
    noninferiority_margin: float = 0.01,
) -> dict[str, Any]:
    reasons: list[str] = []
    if min_macro_f1 is not None and candidate_metrics["macro_f1"] < min_macro_f1:
        reasons.append("candidate_below_min_macro_f1")
    if comparison["ci_lower"] < -noninferiority_margin:
        reasons.append("noninferiority_not_established")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "noninferiority_margin": noninferiority_margin,
        "candidate_macro_f1": candidate_metrics["macro_f1"],
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "comparison": comparison,
        "interpretation": (
            "Pass means the predeclared acceptance gate passed for this fitted model and gate sample; "
            "it is not a guarantee of performance on future distributions."
        ),
    }

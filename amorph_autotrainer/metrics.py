from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "num_examples": int(len(y_true)),
        "labels": labels,
    }


def evaluate_task_requirements(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    min_recall_by_label: dict[str, float] | None = None,
    min_precision_by_label: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate predeclared per-label constraints separately from aggregate Proof Gate.

    A required label absent from y_true fails closed: recall cannot be demonstrated
    without positive examples in the evaluation sample. This checks the supplied
    sample only; it does not establish performance on shifted future distributions.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Task requirement inputs must have equal lengths")
    if len(y_true) == 0:
        raise ValueError("Task requirement evaluation requires at least one example")

    recall_requirements = dict(min_recall_by_label or {})
    precision_requirements = dict(min_precision_by_label or {})
    configured = bool(recall_requirements or precision_requirements)
    if not configured:
        return {
            "configured": False,
            "passed": None,
            "criteria": {"min_recall_by_label": {}, "min_precision_by_label": {}},
            "metrics_by_label": {},
            "reasons": [],
            "scope": "not_configured",
        }

    true_labels = np.asarray(y_true).astype(str)
    pred_labels = np.asarray(y_pred).astype(str)
    all_labels = sorted(set(true_labels.tolist()) | set(pred_labels.tolist())
                        | set(recall_requirements) | set(precision_requirements))
    precision, recall, _, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=all_labels, average=None, zero_division=0
    )
    metrics_by_label: dict[str, dict[str, Any]] = {}
    index_by_label = {label: i for i, label in enumerate(all_labels)}
    for label, i in index_by_label.items():
        metrics_by_label[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "true_support": int(support[i]),
            "predicted_support": int(np.count_nonzero(pred_labels == label)),
        }

    checks: list[dict[str, Any]] = []
    reasons: list[str] = []
    for label, threshold in sorted(recall_requirements.items()):
        metric = metrics_by_label[label]
        observed = metric["recall"] if metric["true_support"] > 0 else None
        passed = observed is not None and observed >= float(threshold)
        check = {
            "metric": "recall",
            "label": label,
            "threshold": float(threshold),
            "observed": observed,
            "support": metric["true_support"],
            "passed": bool(passed),
        }
        checks.append(check)
        if not passed:
            reasons.append(
                "required_label_missing_from_evaluation" if observed is None
                else "minimum_recall_not_met"
            )

    for label, threshold in sorted(precision_requirements.items()):
        metric = metrics_by_label[label]
        observed = metric["precision"]
        passed = observed >= float(threshold)
        check = {
            "metric": "precision",
            "label": label,
            "threshold": float(threshold),
            "observed": observed,
            "support": metric["true_support"],
            "predicted_support": metric["predicted_support"],
            "passed": bool(passed),
        }
        checks.append(check)
        if not passed:
            reasons.append("minimum_precision_not_met")

    return {
        "configured": True,
        "passed": all(item["passed"] for item in checks),
        "criteria": {
            "min_recall_by_label": recall_requirements,
            "min_precision_by_label": precision_requirements,
        },
        "checks": checks,
        "metrics_by_label": metrics_by_label,
        "reasons": sorted(set(reasons)),
        "scope": "only_the_supplied_evaluation_sample; no future_distribution_guarantee",
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
    # A candidate is accepted as non-inferior only if the lower paired-bootstrap
    # confidence bound is above -margin. This is not proof of universal equivalence.
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

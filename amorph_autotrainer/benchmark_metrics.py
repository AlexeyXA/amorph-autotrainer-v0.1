from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Hashable

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def per_class_classification_metrics(
    *,
    y_true: Sequence[Hashable] | np.ndarray,
    y_pred: Sequence[Hashable] | np.ndarray,
    eligible_training_support_by_label: Mapping[Hashable, int],
) -> list[dict[str, Any]]:
    """Return report-only per-class metrics with explicit training/test support.

    The label universe includes eligible training labels, official-test labels,
    and predicted labels. Thus unsupported official-test labels are kept in the
    report with eligible_training_support=0 rather than silently omitted.

    eligible_training_support_by_label counts rows in the prepared official
    training partition after benchmark filters and before AMORPH's internal split.
    This report is diagnostic only and must not feed model selection or the gate.
    """
    true_array = np.asarray(y_true)
    pred_array = np.asarray(y_pred)
    if true_array.ndim != 1 or pred_array.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if len(true_array) != len(pred_array):
        raise ValueError("y_true and y_pred must have equal lengths")
    if len(true_array) == 0:
        raise ValueError("per-class metrics require at least one example")

    for label, count in eligible_training_support_by_label.items():
        if not isinstance(count, (int, np.integer)) or int(count) < 0:
            raise ValueError(
                f"eligible training support for label {label!r} must be a non-negative integer"
            )

    label_universe = (
        set(true_array.tolist())
        | set(pred_array.tolist())
        | {label for label, count in eligible_training_support_by_label.items() if int(count) > 0}
    )
    labels = sorted(label_universe, key=lambda value: (type(value).__name__, str(value)))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_array,
        pred_array,
        labels=labels,
        average=None,
        zero_division=0,
    )
    predicted_counts = {label: int(np.count_nonzero(pred_array == label)) for label in labels}

    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        training_support = int(eligible_training_support_by_label.get(label, 0))
        rows.append({
            "label": str(label),
            "eligible_training_support": training_support,
            "has_training_support": training_support > 0,
            "official_test_support": int(support[index]),
            "predicted_support": predicted_counts[label],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        })
    return rows

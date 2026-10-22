import numpy as np
import pytest

from amorph_autotrainer.benchmark_metrics import per_class_classification_metrics


def test_per_class_report_keeps_official_test_labels_without_training_support():
    rows = per_class_classification_metrics(
        y_true=np.asarray([0, 0, 1, 2]),
        y_pred=np.asarray([0, 1, 1, 1]),
        eligible_training_support_by_label={0: 10, 1: 4, 3: 7},
    )
    by_label = {row["label"]: row for row in rows}

    unsupported = by_label["2"]
    assert unsupported["eligible_training_support"] == 0
    assert unsupported["has_training_support"] is False
    assert unsupported["official_test_support"] == 1
    assert unsupported["predicted_support"] == 0
    assert unsupported["precision"] == 0.0
    assert unsupported["recall"] == 0.0
    assert unsupported["f1"] == 0.0

    absent_test_label = by_label["3"]
    assert absent_test_label["eligible_training_support"] == 7
    assert absent_test_label["official_test_support"] == 0
    assert absent_test_label["predicted_support"] == 0

    supported = by_label["1"]
    assert supported["eligible_training_support"] == 4
    assert supported["official_test_support"] == 1
    assert supported["recall"] == 1.0
    assert np.isclose(supported["precision"], 1.0 / 3.0)


def test_per_class_metrics_reject_mismatched_and_empty_inputs():
    with pytest.raises(ValueError, match="equal lengths"):
        per_class_classification_metrics(
            y_true=[0, 1],
            y_pred=[0],
            eligible_training_support_by_label={0: 1, 1: 1},
        )
    with pytest.raises(ValueError, match="at least one"):
        per_class_classification_metrics(
            y_true=[],
            y_pred=[],
            eligible_training_support_by_label={},
        )


def test_per_class_metrics_reject_negative_or_non_integer_support():
    with pytest.raises(ValueError, match="non-negative integer"):
        per_class_classification_metrics(
            y_true=[0],
            y_pred=[0],
            eligible_training_support_by_label={0: -1},
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        per_class_classification_metrics(
            y_true=[0],
            y_pred=[0],
            eligible_training_support_by_label={0: 1.5},
        )

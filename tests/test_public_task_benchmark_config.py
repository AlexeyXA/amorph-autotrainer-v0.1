import pytest

from scripts.benchmark_public_tasks import parse_label_thresholds


def test_public_benchmark_label_thresholds_parse_repeated_options():
    parsed = parse_label_thresholds(
        ["oos=0.80", "refund=0.95"],
        option_name="--min-recall-by-label",
    )
    assert parsed == {"oos": 0.80, "refund": 0.95}


@pytest.mark.parametrize(
    "values",
    [
        ["oos"],
        ["=0.80"],
        ["oos="],
        ["oos=not-a-number"],
        ["oos=nan"],
        ["oos=1.01"],
        ["oos=-0.01"],
        ["oos=0.80", "oos=0.90"],
    ],
)
def test_public_benchmark_label_thresholds_reject_invalid_or_duplicate_values(values):
    with pytest.raises(ValueError):
        parse_label_thresholds(values, option_name="--min-recall-by-label")

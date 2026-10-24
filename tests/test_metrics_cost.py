import numpy as np
import pytest

from amorph_autotrainer.cost import CostConfig, estimate_tco_eur
from amorph_autotrainer.pipeline import _choose_candidate_by_objective
from amorph_autotrainer.metrics import (
    evaluate_task_requirements,
    paired_stratified_bootstrap_macro_f1,
    gate_candidate,
)


def test_identical_predictions_have_zero_bootstrap_delta():
    y = np.array(["a", "a", "b", "b", "c", "c"] * 4)
    pred = y.copy()
    comp = paired_stratified_bootstrap_macro_f1(y, pred, pred, n_bootstrap=100)
    assert comp["delta_macro_f1_mean"] == pytest.approx(0.0)
    assert comp["ci_lower"] == pytest.approx(0.0)
    assert comp["ci_upper"] == pytest.approx(0.0)


def test_gate_rejects_underperforming_confidence_interval():
    baseline = {"macro_f1": 0.8}
    candidate = {"macro_f1": 0.7}
    comparison = {"ci_lower": -0.2, "ci_upper": -0.05, "delta_macro_f1_mean": -0.1}
    result = gate_candidate(candidate, baseline, comparison, noninferiority_margin=0.01)
    assert result["passed"] is False
    assert "noninferiority_not_established" in result["reasons"]


def test_total_cost_includes_training_inference_human_and_fixed_costs():
    config = CostConfig(
        expected_queries=100,
        compute_eur_per_hour=2.0,
        fixed_data_cost_eur=3.0,
        maintenance_cost_eur=4.0,
        human_hours=2.0,
        human_eur_per_hour=10.0,
    )
    result = estimate_tco_eur(fit_seconds=3600, inference_seconds_per_query=0.036, config=config)
    # 1 training hour + 3.6 seconds inference; plus fixed 3 + maintenance 4 + labor 20.
    assert result["estimated_tco_eur"] == pytest.approx(29.002)
    assert result["runtime_proxy_seconds"] == pytest.approx(3603.6)


def test_cost_config_validates_hard_budgets():
    with pytest.raises(ValueError):
        CostConfig(max_runtime_seconds=0).validate()


@pytest.mark.parametrize("kwargs", [
    {"max_runtime_seconds": float("nan")},
    {"compute_eur_per_hour": float("inf")},
    {"min_macro_f1": float("nan")},
])
def test_cost_config_rejects_non_finite_values(kwargs):
    with pytest.raises(ValueError, match="finite number"):
        CostConfig(**kwargs).validate()



def test_task_requirement_fails_when_oos_recall_is_below_threshold_despite_good_other_class():
    y_true = np.asarray(["oos"] * 1000 + ["intent"] * 100, dtype=object)
    y_pred = np.asarray(["oos"] * 39 + ["other"] * 961 + ["intent"] * 100, dtype=object)

    result = evaluate_task_requirements(
        y_true,
        y_pred,
        min_recall_by_label={"oos": 0.80},
    )

    assert result["configured"] is True
    assert result["passed"] is False
    check = next(item for item in result["checks"] if item["label"] == "oos")
    assert check["observed"] == pytest.approx(0.039)
    assert check["passed"] is False
    assert "minimum_recall_not_met" in result["reasons"]


def test_task_requirement_fails_closed_when_required_label_is_absent_from_evaluation():
    result = evaluate_task_requirements(
        np.asarray(["sports", "sports", "economy", "economy"], dtype=object),
        np.asarray(["sports", "sports", "economy", "economy"], dtype=object),
        min_recall_by_label={"oos": 0.80},
    )

    assert result["passed"] is False
    check = next(item for item in result["checks"] if item["label"] == "oos")
    assert check["observed"] is None
    assert check["support"] == 0
    assert "required_label_missing_from_evaluation" in result["reasons"]


def test_task_requirement_can_require_minimum_precision():
    result = evaluate_task_requirements(
        np.asarray(["a", "a", "b", "b"], dtype=object),
        np.asarray(["a", "b", "b", "b"], dtype=object),
        min_precision_by_label={"b": 0.90},
    )

    assert result["passed"] is False
    check = next(item for item in result["checks"] if item["label"] == "b")
    assert check["observed"] == pytest.approx(2 / 3)
    assert check["passed"] is False


def test_task_requirement_is_explicitly_unconfigured_when_no_criteria_are_set():
    result = evaluate_task_requirements(
        np.asarray(["a", "b"], dtype=object),
        np.asarray(["a", "b"], dtype=object),
    )
    assert result["configured"] is False
    assert result["passed"] is None


def test_cost_config_validates_per_label_thresholds():
    config = CostConfig(
        min_recall_by_label={"oos": 0.80},
        min_precision_by_label={"refund": 0.95},
    )
    config.validate()

    with pytest.raises(ValueError, match="must be in"):
        CostConfig(min_recall_by_label={"oos": 1.01}).validate()
    with pytest.raises(ValueError, match="non-empty strings"):
        CostConfig(min_recall_by_label={" ": 0.80}).validate()



def test_tco_estimator_adds_strategy_specific_usage_and_fixed_costs():
    config = CostConfig(
        selection_objective="tco",
        expected_queries=1000,
        compute_eur_per_hour=0.0,
        strategy_usage_cost_eur_per_query={"e5": 0.002},
        strategy_fixed_cost_eur={"e5": 3.0},
        fixed_data_cost_eur=1.0,
        maintenance_cost_eur=2.0,
    )
    result = estimate_tco_eur(
        fit_seconds=100,
        inference_seconds_per_query=0.01,
        config=config,
        strategy_name="e5",
    )

    assert result["strategy_usage_cost_eur"] == pytest.approx(2.0)
    assert result["strategy_fixed_cost_eur"] == pytest.approx(3.0)
    assert result["estimated_tco_eur"] == pytest.approx(8.0)
    assert result["strategy_name"] == "e5"


def test_tco_objective_can_select_slower_candidate_with_lower_total_cost():
    fit = {"tfidf": 1.0, "e5": 3.0}
    latency = {"tfidf": 0.001, "e5": 0.002}
    costs = CostConfig(
        selection_objective="tco",
        expected_queries=100,
        compute_eur_per_hour=0.0,
        strategy_usage_cost_eur_per_query={"tfidf": 0.10, "e5": 0.001},
        strategy_fixed_cost_eur={"tfidf": 0.0, "e5": 0.0},
    )

    selected, report = _choose_candidate_by_objective(
        ["tfidf", "e5"],
        fit_seconds_by_strategy=fit,
        inference_seconds_per_query_by_strategy=latency,
        config=costs,
    )

    assert selected == "e5"
    assert report["candidate_costs"]["e5"]["estimated_tco_eur"] == pytest.approx(0.10)
    assert report["candidate_costs"]["tfidf"]["estimated_tco_eur"] == pytest.approx(10.0)
    assert report["selection_objective"] == "tco"
    assert report["tie_breaker"] == "none"
    assert report["candidate_tco_values_differ"] is True


def test_runtime_proxy_objective_does_not_use_monetary_override_for_ranking():
    fit = {"tfidf": 1.0, "e5": 3.0}
    latency = {"tfidf": 0.001, "e5": 0.002}
    costs = CostConfig(
        selection_objective="runtime_proxy",
        expected_queries=100,
        compute_eur_per_hour=0.0,
        strategy_usage_cost_eur_per_query={"tfidf": 0.10, "e5": 0.001},
        strategy_fixed_cost_eur={"tfidf": 0.0, "e5": 0.0},
    )

    selected, report = _choose_candidate_by_objective(
        ["tfidf", "e5"],
        fit_seconds_by_strategy=fit,
        inference_seconds_per_query_by_strategy=latency,
        config=costs,
    )

    assert selected == "tfidf"
    assert report["selection_objective"] == "runtime_proxy"


def test_tco_objective_breaks_equal_cost_ties_by_runtime_and_records_tie():
    config = CostConfig(
        selection_objective="tco",
        expected_queries=100,
        compute_eur_per_hour=0.0,
        strategy_usage_cost_eur_per_query={"tfidf": 0.0, "e5": 0.0},
        strategy_fixed_cost_eur={"tfidf": 0.0, "e5": 0.0},
    )
    selected, report = _choose_candidate_by_objective(
        ["tfidf", "e5"],
        fit_seconds_by_strategy={"tfidf": 1.0, "e5": 3.0},
        inference_seconds_per_query_by_strategy={"tfidf": 0.001, "e5": 0.002},
        config=config,
    )

    assert selected == "tfidf"
    assert set(report["tco_tied_candidates"]) == {"tfidf", "e5"}
    assert report["tie_breaker"] == "runtime_proxy_for_equal_tco"
    assert report["candidate_tco_values_differ"] is False


def test_cost_config_rejects_unknown_selection_objective_and_negative_strategy_rates():
    with pytest.raises(ValueError, match="selection_objective"):
        CostConfig(selection_objective="cheapest").validate()
    with pytest.raises(ValueError, match="must be >= 0"):
        CostConfig(strategy_usage_cost_eur_per_query={"e5": -0.01}).validate()
    with pytest.raises(ValueError, match="finite number"):
        CostConfig(strategy_fixed_cost_eur={"e5": float("inf")}).validate()

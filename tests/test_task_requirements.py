from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from amorph_autotrainer.cli import main as cli_main
from amorph_autotrainer.cost import CostConfig
from amorph_autotrainer.demo import create_demo_csv
from amorph_autotrainer.metrics import evaluate_task_requirements
from amorph_autotrainer.pipeline import _validation_candidate_eligibility


def test_unconfigured_task_requirements_are_unknown_not_passed():
    result = evaluate_task_requirements(
        np.asarray(["a", "b"]),
        np.asarray(["a", "b"]),
    )
    assert result["configured"] is False
    assert result["passed"] is None


def test_recall_floor_rejects_candidate_with_low_required_label_recall():
    result = evaluate_task_requirements(
        np.asarray(["pay", "pay", "bug", "bug"]),
        np.asarray(["pay", "bug", "bug", "bug"]),
        min_recall_by_label={"pay": 0.9},
    )
    assert result["configured"] is True
    assert result["passed"] is False
    assert result["metrics_by_label"]["pay"]["recall"] == pytest.approx(0.5)
    assert "minimum_recall_not_met" in result["reasons"]


def test_required_recall_label_missing_from_evaluation_fails_closed():
    result = evaluate_task_requirements(
        np.asarray(["pay", "pay", "bug", "bug"]),
        np.asarray(["pay", "pay", "bug", "bug"]),
        min_recall_by_label={"out_of_scope": 0.8},
    )
    assert result["passed"] is False
    assert {
        "metric": "recall",
        "label": "out_of_scope",
        "threshold": 0.8,
        "observed": None,
        "support": 0,
        "passed": False,
    } in result["checks"]
    assert "required_label_missing_from_evaluation" in result["reasons"]


def test_precision_floor_is_independent_of_recall_floor():
    result = evaluate_task_requirements(
        np.asarray(["pay", "pay", "bug", "bug"]),
        np.asarray(["pay", "bug", "bug", "bug"]),
        min_precision_by_label={"bug": 0.9},
    )
    assert result["passed"] is False
    assert result["metrics_by_label"]["bug"]["precision"] == pytest.approx(2 / 3)
    assert "minimum_precision_not_met" in result["reasons"]


def test_precision_is_unknown_when_required_label_is_never_predicted():
    result = evaluate_task_requirements(
        np.asarray(["pay", "bug", "pay", "bug"]),
        np.asarray(["pay", "bug", "pay", "bug"]),
        min_precision_by_label={"out_of_scope": 0.8},
    )
    check = next(
        item for item in result["checks"]
        if item["metric"] == "precision" and item["label"] == "out_of_scope"
    )
    assert check["observed"] is None
    assert check["predicted_support"] == 0
    assert check["passed"] is False
    assert result["metrics_by_label"]["out_of_scope"]["precision"] is None
    assert "required_label_never_predicted" in result["reasons"]
    assert "minimum_precision_not_met" not in result["reasons"]


def test_task_requirement_thresholds_must_be_finite_and_in_unit_interval():
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        CostConfig(min_recall_by_label={"out_of_scope": 1.1}).validate()
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        CostConfig(min_precision_by_label={"refund": float("nan")}).validate()


def test_cli_fails_closed_when_mandatory_label_is_absent(tmp_path, capsys):
    csv_path = create_demo_csv(tmp_path / "demo.csv", examples_per_class=50)
    run_dir = tmp_path / "runs"
    code = cli_main([
        "train", str(csv_path),
        "--strategies", "tfidf",
        "--out", str(run_dir),
        "--max-runtime-seconds", "300",
        "--max-candidates", "1",
        "--bootstrap-samples", "100",
        "--min-gate-examples-per-class", "5",
        "--min-recall-by-label", "label_that_does_not_exist=0.80",
    ])
    captured = capsys.readouterr()
    assert captured.out, captured.err
    payload = json.loads(captured.out)
    assert code == 5
    assert payload["business_requirements_passed"] is False
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    assert report["status"] == "task_requirements_not_met"
    assert report["business_requirements_passed"] is False
    assert report["task_requirements"]["final_model_validation"]["passed"] is False
    assert report["task_requirements"]["final_model_acceptance_gate"]["passed"] is False


def test_underpowered_gate_is_unknown_not_business_rejection(tmp_path, capsys):
    csv_path = create_demo_csv(tmp_path / "small-underpowered.csv", examples_per_class=12)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = sorted({row["label"] for row in __import__("csv").DictReader(handle)})
    output_dir = tmp_path / "underpowered-runs"
    code = cli_main([
        "train", str(csv_path),
        "--strategies", "tfidf",
        "--out", str(output_dir),
        "--max-runtime-seconds", "300",
        "--max-candidates", "1",
        "--bootstrap-samples", "100",
        "--min-gate-examples-per-class", "20",
        "--min-recall-by-label", f"{labels[0]}=0.0",
    ])
    captured = capsys.readouterr()
    assert captured.out, captured.err
    payload = json.loads(captured.out)
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    assert code == 4
    assert report["status"] == "acceptance_gate_underpowered"
    assert report["business_requirements_passed"] is None
    assert report["gate_passed_for_configured_metric"] is False
    assert report["overall_acceptance_passed"] is False


def test_task_ineligible_high_macro_f1_does_not_raise_eligibility_threshold():
    val_metrics = {
        "aggregate_winner": {"macro_f1": 0.95},
        "task_compliant": {"macro_f1": 0.90},
    }
    task_requirements = {
        "aggregate_winner": {"passed": False},
        "task_compliant": {"passed": True},
    }
    task_eligible, eligible, threshold, diagnostic = _validation_candidate_eligibility(
        val_metrics,
        task_requirements,
        min_macro_f1=None,
        noninferiority_margin=0.01,
    )
    assert task_eligible == ["task_compliant"]
    assert eligible == ["task_compliant"]
    assert threshold == pytest.approx(0.89)
    assert diagnostic == "task_compliant"


def test_no_task_compliant_candidate_stays_ineligible_but_keeps_diagnostic():
    val_metrics = {
        "higher_macro": {"macro_f1": 0.95},
        "lower_macro": {"macro_f1": 0.90},
    }
    task_requirements = {
        "higher_macro": {"passed": False},
        "lower_macro": {"passed": False},
    }
    task_eligible, eligible, threshold, diagnostic = _validation_candidate_eligibility(
        val_metrics,
        task_requirements,
        min_macro_f1=None,
        noninferiority_margin=0.01,
    )
    assert task_eligible == []
    assert eligible == []
    assert threshold == pytest.approx(0.94)
    assert diagnostic == "higher_macro"

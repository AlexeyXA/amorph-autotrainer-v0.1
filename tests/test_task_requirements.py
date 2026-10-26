from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from amorph_autotrainer.cli import main as cli_main
from amorph_autotrainer.cost import CostConfig
from amorph_autotrainer.demo import create_demo_csv
from amorph_autotrainer.metrics import evaluate_task_requirements


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
        "--bootstrap-samples", "50",
        "--min-gate-examples-per-class", "5",
        "--min-recall-by-label", "label_that_does_not_exist=0.80",
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 5
    assert payload["business_requirements_passed"] is False
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    assert report["status"] == "task_requirements_not_met"
    assert report["business_requirements_passed"] is False
    assert report["task_requirements"]["final_model_validation"]["passed"] is False
    assert report["task_requirements"]["final_model_acceptance_gate"]["passed"] is False

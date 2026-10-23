import json
import subprocess
import sys

import numpy as np
import pytest

from amorph_autotrainer.proofgate import Verdict, evaluate, write_report


def _balanced_labels(n_per_class=80):
    return np.asarray(["a"] * n_per_class + ["b"] * n_per_class)


def test_golden_accept_when_powered_and_candidate_is_noninferior():
    y = _balanced_labels()
    baseline = np.asarray(["a"] * 80 + ["a"] * 40 + ["b"] * 40)
    candidate = y.copy()

    report = evaluate(
        y,
        baseline,
        candidate,
        margin=0.02,
        n_bootstrap=200,
        seed=17,
        min_examples_per_class=5,
    )

    assert report.verdict == Verdict.ACCEPT
    assert report.subtype is None
    assert report.reason_codes == ()
    assert report.powered is True
    assert report.comparison["ci_lower"] >= -0.02
    assert report.to_legacy_gate_dict()["passed"] is True


def test_golden_reject_when_powered_candidate_is_worse_than_baseline():
    y = _balanced_labels()
    baseline = y.copy()
    candidate = np.asarray(["b"] * 80 + ["a"] * 80)

    report = evaluate(
        y,
        baseline,
        candidate,
        margin=0.02,
        n_bootstrap=200,
        seed=17,
        min_examples_per_class=5,
    )

    assert report.verdict == Verdict.REJECT
    assert "noninferiority_not_established" in report.reason_codes
    assert report.comparison["ci_lower"] < -0.02
    assert report.to_legacy_gate_dict()["passed"] is False


def test_golden_underpowered_gate_is_inconclusive_not_accept():
    y = np.asarray(["a"] * 4 + ["b"] * 4)
    baseline = y.copy()
    candidate = np.asarray(["a", "a", "a", "b", "b", "b", "b", "b"])

    report = evaluate(
        y,
        baseline,
        candidate,
        n_bootstrap=100,
        min_examples_per_class=5,
    )

    assert report.verdict == Verdict.INCONCLUSIVE
    assert report.subtype == "UNDERPOWERED"
    assert "min_examples_per_class_not_met" in report.reason_codes
    assert report.comparison is None
    assert report.to_legacy_gate_dict()["passed"] is False


def test_golden_self_comparison_is_inconclusive_without_zero_width_ci():
    y = _balanced_labels()
    predictions = y.copy()

    report = evaluate(y, predictions, predictions.copy(), n_bootstrap=100)

    assert report.verdict == Verdict.INCONCLUSIVE
    assert report.subtype == "SELF_COMPARISON"
    assert report.reason_codes == ("baseline_and_candidate_predictions_identical",)
    assert report.comparison is None
    assert report.to_legacy_gate_dict()["passed"] is False


def test_golden_length_mismatch_is_inconclusive_conformance_failure():
    y = _balanced_labels()
    baseline = y.copy()
    candidate = y[:-1]

    report = evaluate(y, baseline, candidate, n_bootstrap=100)

    assert report.verdict == Verdict.INCONCLUSIVE
    assert report.subtype == "CONFORMANCE_FAILED"
    assert "candidate_length_mismatch" in report.reason_codes
    assert report.conformance["passed"] is False
    assert report.comparison is None
    assert report.to_legacy_gate_dict()["passed"] is False


def test_prediction_label_absent_from_gate_truth_is_inconclusive():
    y = _balanced_labels()
    baseline = y.copy()
    candidate = y.copy()
    candidate[0] = "class-not-observed-in-gate"

    report = evaluate(y, baseline, candidate, n_bootstrap=100)

    assert report.verdict == Verdict.INCONCLUSIVE
    assert report.subtype == "CONFORMANCE_FAILED"
    assert "pred_candidate_contains_label_absent_from_y_gate" in report.reason_codes
    assert report.comparison is None
    assert report.to_legacy_gate_dict()["passed"] is False


def test_no_candidate_is_baseline_only_inconclusive():
    y = _balanced_labels()
    baseline = y.copy()

    report = evaluate(y, baseline, None, n_bootstrap=100)

    assert report.verdict == Verdict.INCONCLUSIVE
    assert report.subtype == "BASELINE_ONLY"
    assert "no_candidate_supplied" in report.reason_codes
    assert report.comparison is None
    assert report.powered is True
    assert report.to_legacy_gate_dict()["passed"] is False


def test_report_sha256_is_deterministic_and_tracks_input_changes(tmp_path):
    y = _balanced_labels()
    baseline = y.copy()
    candidate = np.asarray(["a"] * 80 + ["a"] * 5 + ["b"] * 75)
    report_a = evaluate(y, baseline, candidate, n_bootstrap=100, seed=11).to_dict()
    report_b = evaluate(y, baseline, candidate, n_bootstrap=100, seed=11).to_dict()
    report_c = evaluate(y, baseline, candidate, n_bootstrap=100, seed=12).to_dict()

    assert report_a["report_sha256"] == report_b["report_sha256"]
    assert report_a["input_fingerprints"]["payload_sha256"] == report_b["input_fingerprints"]["payload_sha256"]
    assert report_a["report_sha256"] != report_c["report_sha256"]

    output = tmp_path / "nested" / "report.json"
    written = write_report(evaluate(y, baseline, candidate, n_bootstrap=100, seed=11), output)
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed == written
    assert parsed["report_sha256"] == report_a["report_sha256"]


def test_json_cli_writes_report_for_inconclusive_self_comparison(tmp_path):
    y = _balanced_labels()
    source = tmp_path / "input.json"
    output = tmp_path / "proofgate-report.json"
    source.write_text(
        json.dumps({
            "y_gate": y.tolist(),
            "pred_baseline": y.tolist(),
            "pred_candidate": y.tolist(),
            "criteria": {"n_bootstrap": 100, "seed": 9},
            "metadata": {"run": "golden-cli"},
        }),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "amorph_autotrainer.proofgate", "--input", str(source), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout["verdict"] == Verdict.INCONCLUSIVE
    assert stdout["subtype"] == "SELF_COMPARISON"
    assert payload["metadata"] == {"run": "golden-cli"}
    assert len(payload["report_sha256"]) == 64



@pytest.mark.parametrize(
    ("criteria", "message"),
    [
        ({"margin": "0.02"}, "margin must be a finite real number"),
        ({"confidence": "0.95"}, "confidence must be a finite real number"),
        ({"min_macro_f1": "0.5"}, "min_macro_f1 must be a finite real number"),
        ({"margin": True}, "margin must be a finite real number"),
    ],
)
def test_numeric_criteria_reject_strings_and_booleans(criteria, message):
    y = _balanced_labels()
    with pytest.raises(ValueError, match=message):
        evaluate(y, y.copy(), np.asarray(["a"] * 79 + ["b"] + ["b"] * 80), **criteria)


def test_ragged_array_input_returns_inconclusive_conformance_report():
    y = [["a"], ["b", "c"]]
    baseline = [["a"], ["b", "c"]]
    candidate = [["a"], ["b", "c"]]

    report = evaluate(y, baseline, candidate, n_bootstrap=100)

    assert report.verdict == Verdict.INCONCLUSIVE
    assert report.subtype == "CONFORMANCE_FAILED"
    assert any(code.endswith("_array_conversion_failed") for code in report.reason_codes)
    assert report.comparison is None
    assert report.to_dict()["report_sha256"]

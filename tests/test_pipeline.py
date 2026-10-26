import csv
import json
from pathlib import Path

from amorph_autotrainer.cost import CostConfig
from amorph_autotrainer.cli import main as cli_main
from amorph_autotrainer.demo import create_demo_csv
from amorph_autotrainer.pipeline import (
    run_training,
    _proofgate_may_accept_candidate,
    _choose_candidate_by_objective,
)
from amorph_autotrainer.strategies import load_strategy_model, predict_loaded


def test_proofgate_cutover_accepts_only_verified_real_challengers():
    cases = [
        ("e5", "ACCEPT", "match", True),
        ("e5", "ACCEPT", "mismatch", False),
        ("e5", "ACCEPT", "not_comparable_underpowered", False),
        ("e5", "REJECT", "match", False),
        ("e5", "INCONCLUSIVE", "match", False),
        ("tfidf", "ACCEPT", "match", False),
    ]
    for strategy, verdict, parity, expected in cases:
        assert _proofgate_may_accept_candidate(strategy, verdict, parity) is expected


def test_end_to_end_tfidf_only(tmp_path):
    data_path = create_demo_csv(tmp_path / "demo.csv", examples_per_class=12)
    out = tmp_path / "runs"
    result = run_training(
        csv_path=data_path,
        output_dir=out,
        strategies=["tfidf"],
        export_features=True,
        config=CostConfig(
            max_runtime_seconds=300,
            max_candidates=1,
            expected_queries=50,
            bootstrap_samples=100,
            noninferiority_margin=0.02,
        ),
    )
    assert result["selected_strategy"] == "tfidf"
    assert result["report_path"]
    assert Path(result["manifest_path"]).is_file()
    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert report["dataset_sha256"]
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert "python_version" in manifest
    assert "numpy" in manifest["library_versions"]
    assert "scikit-learn" in manifest["library_versions"]
    assert "commit_sha" in manifest
    assert len(manifest["source_tree_sha256"]) == 64
    assert "near_duplicate_threshold" in manifest["config"]
    assert "max_near_duplicate_audit_rows" in manifest["config"]
    assert manifest["split_config"]["random_seed"] == 42
    assert "e5_model_revision" in manifest["config"]
    assert "strategy_fingerprints" in manifest
    assert "strategy_status" in manifest
    assert manifest["protocol_status"]["conformance_gate"] == "implemented"
    assert "tfidf" in manifest["conformance_gate"]
    assert Path(result["run_dir"], "replay.json").is_file()
    replay = json.loads(Path(result["run_dir"], "replay.json").read_text(encoding="utf-8"))
    assert replay["format"] == "amorph_replay_descriptor_v1"
    assert replay["descriptor_sha256"] == manifest["replay_descriptor"]["descriptor_sha256"]
    assert report["status"] in {
        "accepted", "target_not_met", "baseline_kept_candidate_rejected",
        "baseline_only", "baseline_self_comparison", "accepted_no_absolute_target",
        "acceptance_gate_underpowered", "acceptance_gate_inconclusive",
        "acceptance_gate_conformance_failed", "proofgate_legacy_parity_mismatch",
    }
    assert report["acceptance_gate_comparison"]["selected_candidate_scope"] == "baseline_self_comparison"
    if report["proofgate"]["selected_candidate_subtype"] == "UNDERPOWERED":
        assert any("Proof Gate returned INCONCLUSIVE / UNDERPOWERED" in warning for warning in report["warnings"])
    else:
        assert any("Proof Gate marked the TF-IDF self-comparison INCONCLUSIVE" in warning for warning in report["warnings"])
    assert report["proofgate"]["version"]
    assert report["proofgate"]["mode"] == "decision"
    assert report["proofgate"]["decision_source"] == "proofgate.evaluate"
    assert report["proofgate"]["cutover_complete"] is True
    proofgate_report = json.loads(
        (Path(result["run_dir"]) / report["proofgate"]["selected_candidate_report"])
        .read_text(encoding="utf-8")
    )
    assert proofgate_report["verdict"] == "INCONCLUSIVE"
    assert proofgate_report["subtype"] in {"SELF_COMPARISON", "UNDERPOWERED"}
    assert proofgate_report["comparison"] is None
    assert proofgate_report["report_sha256"] == report["proofgate"]["selected_candidate_report_sha256"]
    # After cut-over, self-comparison is INCONCLUSIVE and cannot be exposed as passed.
    assert report["selected_candidate_gate"]["passed"] is False
    assert report["proofgate"]["selected_candidate_verdict"] == "INCONCLUSIVE"
    assert report["proofgate"]["decision_passed"] is False
    if report["acceptance_gate_sufficient"]:
        assert proofgate_report["subtype"] == "SELF_COMPARISON"
        assert report["proofgate"]["legacy_parity_status"] == "intentional_inconclusive_self_comparison"
    else:
        assert proofgate_report["subtype"] == "UNDERPOWERED"
        assert report["proofgate"]["legacy_parity_status"] == "not_comparable_underpowered"
    assert report["proofgate"]["legacy_selected_gate_passed"] is None
    assert report["tco_assumption_status"] == "assumed_free_inputs"
    assert "acceptance_gate_sufficient" in report
    data_report = json.loads((Path(result["run_dir"]) / "data_report.json").read_text(encoding="utf-8"))
    assert data_report["split_method"] in {
        "stratified_exact_dedup_fallback",
        "milp_group_assignment_with_near_duplicate_groups",
        "milp_group_assignment_with_group_column_and_near_duplicates",
        "milp_group_assignment_with_group_column_near_duplicate_audit_skipped",
        "stratified_group_kfold",
    }
    assert (Path(result["run_dir"]) / "selected_features" / "train.npz").is_file()

    loaded = load_strategy_model(result["model_path"])
    pred = predict_loaded(loaded, ["Команда выиграла матч и вышла в финал турнира"])
    assert len(pred) == 1
    assert Path(result["database_path"]).is_file()


def test_baseline_only_status_is_explicit_when_gate_is_sufficient(tmp_path):
    data_path = create_demo_csv(tmp_path / "baseline-only.csv", examples_per_class=30)
    result = run_training(
        csv_path=data_path,
        output_dir=tmp_path / "runs",
        strategies=["tfidf"],
        config=CostConfig(
            max_runtime_seconds=300, max_candidates=1, expected_queries=100,
            bootstrap_samples=100, min_gate_examples_per_class=5,
        ),
    )
    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert report["acceptance_gate_sufficient"] is True
    assert report["status"] == "baseline_only"
    assert report["acceptance_gate_comparison"]["is_self_comparison"] is True
    assert any("No challenger passed the validation threshold" in warning for warning in report["warnings"])
    proofgate_info = report["proofgate"]
    proofgate_report = json.loads(
        (Path(result["run_dir"]) / proofgate_info["selected_candidate_report"])
        .read_text(encoding="utf-8")
    )
    baseline_proofgate_report = json.loads(
        (Path(result["run_dir"]) / proofgate_info["baseline_reference_report"])
        .read_text(encoding="utf-8")
    )
    assert proofgate_report["verdict"] == "INCONCLUSIVE"
    assert proofgate_report["subtype"] == "SELF_COMPARISON"
    assert proofgate_report["comparison"] is None
    assert baseline_proofgate_report["verdict"] == "INCONCLUSIVE"
    assert baseline_proofgate_report["subtype"] == "SELF_COMPARISON"
    assert baseline_proofgate_report["comparison"] is None
    # C2 cut-over uses Proof Gate as decision source: no synthetic self-comparison pass.
    assert report["final_acceptance_gate"]["passed"] is False
    assert report["proofgate"]["selected_candidate_verdict"] == "INCONCLUSIVE"
    assert report["proofgate"]["legacy_parity_status"] == "intentional_inconclusive_self_comparison"
    assert report["proofgate"]["legacy_selected_gate_passed"] is None
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["proofgate"]["selected_candidate_report_sha256"] == proofgate_report["report_sha256"]
    # A downloaded selected-model directory must carry every provenance file its
    # manifest references, not rely on sibling files in the run root.
    selected_dir = Path(result["manifest_path"]).parent
    assert (selected_dir / "replay.json").is_file()
    assert (selected_dir / "data_report.json").is_file()
    selected_proofgate_copy = json.loads(
        (selected_dir / proofgate_info["selected_candidate_report"]).read_text(encoding="utf-8")
    )
    assert selected_proofgate_copy["report_sha256"] == proofgate_report["report_sha256"]
    data_report = json.loads((Path(result["run_dir"]) / "data_report.json").read_text(encoding="utf-8"))
    assert proofgate_report["metadata"]["gate_split_sha256"] == data_report["splits"]["acceptance_gate"]["sha256"]
    replay = json.loads((Path(result["run_dir"]) / "replay.json").read_text(encoding="utf-8"))
    assert replay["extra"]["proofgate"]["selected_candidate_report_sha256"] == proofgate_report["report_sha256"]


def test_cli_exit_code_4_when_acceptance_gate_is_underpowered(tmp_path, capsys):
    data_path = create_demo_csv(tmp_path / "small.csv", examples_per_class=12)
    output_dir = tmp_path / "runs"
    code = cli_main([
        "train", str(data_path), "--out", str(output_dir),
        "--strategies", "tfidf", "--max-candidates", "1",
        "--max-runtime-seconds", "60", "--bootstrap-samples", "100",
        "--min-gate-examples-per-class", "20",
        "--max-near-duplicate-audit-rows", "1",
    ])
    captured = capsys.readouterr()
    assert code == 4
    payload = json.loads(captured.out)
    assert payload["status"] == "acceptance_gate_underpowered"
    assert payload["acceptance_gate_sufficient"] is False
    report_path = output_dir / payload["run_id"] / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proofgate_report = json.loads(
        (report_path.parent / report["proofgate"]["selected_candidate_report"])
        .read_text(encoding="utf-8")
    )
    assert proofgate_report["verdict"] == "INCONCLUSIVE"
    assert proofgate_report["subtype"] == "UNDERPOWERED"
    assert proofgate_report["comparison"] is None
    assert report["selected_candidate_gate"]["passed"] is False


def test_invalid_zero_runtime_is_rejected_before_any_candidate_or_model_loading(tmp_path, monkeypatch, capsys):
    data_path = create_demo_csv(tmp_path / "data.csv", examples_per_class=12)

    def forbidden_loader(_model_id, _revision=None):
        raise AssertionError("E5 model loader must not be reached for invalid runtime config")

    from amorph_autotrainer.strategies import E5Candidate
    monkeypatch.setattr(E5Candidate, "_load_encoder", staticmethod(forbidden_loader))
    output_dir = tmp_path / "runs-zero-budget"
    code = cli_main([
        "train", str(data_path), "--out", str(output_dir),
        "--strategies", "tfidf,e5", "--max-candidates", "2",
        "--max-runtime-seconds", "0",
    ])
    captured = capsys.readouterr()
    assert code == 1
    assert "max_runtime_seconds must be > 0" in captured.err
    assert not output_dir.exists()


def test_early_runtime_budget_exhaustion_returns_exit_code_3(tmp_path, capsys):
    data_path = create_demo_csv(tmp_path / "budget.csv", examples_per_class=12)
    output_dir = tmp_path / "runs"
    code = cli_main([
        "train", str(data_path), "--out", str(output_dir),
        "--strategies", "tfidf", "--max-candidates", "1",
        "--max-runtime-seconds", "0.0001", "--bootstrap-samples", "100",
    ])
    captured = capsys.readouterr()
    assert code == 3
    assert "Runtime budget exhausted" in captured.err



def test_cli_fails_closed_when_mandatory_label_recall_requirement_is_not_demonstrated(tmp_path, capsys):
    data_path = create_demo_csv(tmp_path / "task-requirement.csv", examples_per_class=30)
    output_dir = tmp_path / "runs-task-requirement"
    code = cli_main([
        "train", str(data_path), "--out", str(output_dir),
        "--strategies", "tfidf", "--max-candidates", "1",
        "--max-runtime-seconds", "300", "--bootstrap-samples", "100",
        "--min-recall-by-label", "oos=0.80",
    ])

    captured = capsys.readouterr()
    assert code == 5
    payload = json.loads(captured.out)
    assert payload["business_requirements_passed"] is False
    assert payload["task_requirements"]["configured"] is True
    assert payload["task_requirements"]["selected_candidate_validation_passed"] is False
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    assert report["status"] == "task_requirements_not_met"
    gate_result = report["task_requirements"]["baseline_acceptance_gate"]
    assert gate_result["passed"] is False
    assert "required_label_missing_from_evaluation" in gate_result["reasons"]
    assert any("no candidate is demonstrated" in item for item in report["warnings"])
    assert report["selector"]["diagnostic_only"] is True
    assert report["selector"]["selection_status"] == "diagnostic_only_no_validation_eligible_candidates"
    assert report["selector"]["eligible_candidates"] == []
    assert report["selector"]["candidates_considered"] == ["tfidf"]



def test_tco_mode_requires_explicit_strategy_cost_inputs_when_compute_rate_is_zero(tmp_path, capsys):
    data_path = create_demo_csv(tmp_path / "tco-unpriced.csv", examples_per_class=12)
    output_dir = tmp_path / "runs-unpriced"
    code = cli_main([
        "train", str(data_path), "--out", str(output_dir),
        "--strategies", "tfidf", "--max-candidates", "1",
        "--max-runtime-seconds", "300", "--selection-objective", "tco",
    ])

    captured = capsys.readouterr()
    assert code == 1
    assert "requires explicit" in captured.err
    assert "tfidf" in captured.err
    assert not output_dir.exists()


def test_tco_mode_accepts_explicit_zero_rates_but_reports_assumed_free_costs(tmp_path, capsys):
    data_path = create_demo_csv(tmp_path / "tco-priced.csv", examples_per_class=30)
    output_dir = tmp_path / "runs-priced"
    code = cli_main([
        "train", str(data_path), "--out", str(output_dir),
        "--strategies", "tfidf", "--max-candidates", "1",
        "--max-runtime-seconds", "300", "--selection-objective", "tco",
        "--strategy-usage-cost-eur-per-query", "tfidf=0",
        "--strategy-fixed-cost-eur", "tfidf=0",
        "--bootstrap-samples", "100",
    ])

    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["selection_objective"] == "tco"
    assert payload["selector"]["selection_objective"] == "tco"
    assert payload["selector"]["selected_strategy"] == "tfidf"
    assert payload["selector"]["candidate_costs"]["tfidf"]["estimated_tco_eur"] == 0
    assert payload["selector"]["candidate_tco_values_differ"] is False
    report = json.loads(Path(payload["report_path"]).read_text(encoding="utf-8"))
    assert report["selection_objective"] == "tco"
    assert report["estimated_tco"]["cost_interpretation"]
    assert any("assumed-free" in warning for warning in report["warnings"])


def test_tco_mode_requires_explicit_strategy_cost_inputs_even_with_compute_rate(tmp_path, capsys):
    data_path = create_demo_csv(tmp_path / "tco-partially-priced.csv", examples_per_class=12)
    output_dir = tmp_path / "runs-partially-priced"
    code = cli_main([
        "train", str(data_path), "--out", str(output_dir),
        "--strategies", "tfidf", "--max-candidates", "1",
        "--max-runtime-seconds", "300", "--selection-objective", "tco",
        "--compute-eur-per-hour", "25",
    ])

    captured = capsys.readouterr()
    assert code == 1
    assert "requires explicit" in captured.err
    assert "missing usage rates=['tfidf']" in captured.err
    assert "missing fixed costs=['tfidf']" in captured.err
    assert not output_dir.exists()


def test_tco_objective_can_override_runtime_proxy_when_candidate_costs_differ():
    shared = {
        "eligible_names": ["tfidf", "e5"],
        "fit_seconds_by_strategy": {"tfidf": 1.0, "e5": 10.0},
        "inference_seconds_per_query_by_strategy": {"tfidf": 0.01, "e5": 0.02},
    }
    costs = {
        "strategy_usage_cost_eur_per_query": {"tfidf": 1.0, "e5": 0.0},
        "strategy_fixed_cost_eur": {"tfidf": 0.0, "e5": 0.0},
    }
    runtime_selected, runtime_report = _choose_candidate_by_objective(
        **shared,
        config=CostConfig(selection_objective="runtime_proxy", expected_queries=100),
    )
    tco_selected, tco_report = _choose_candidate_by_objective(
        **shared,
        config=CostConfig(
            selection_objective="tco", expected_queries=100,
            compute_eur_per_hour=10.0,
            **costs,
        ),
    )

    assert runtime_selected == "tfidf"
    assert tco_selected == "e5"
    assert tco_report["selection_objective"] == "tco"
    assert tco_report["candidate_tco_values_differ"] is True
    assert tco_report["candidate_costs"]["e5"]["estimated_tco_eur"] < tco_report["candidate_costs"]["tfidf"]["estimated_tco_eur"]
    assert runtime_report["selection_objective"] == "runtime_proxy"


def test_tco_candidate_selector_fails_closed_for_unknown_strategy_costs():
    config = CostConfig(
        selection_objective="tco",
        expected_queries=100,
        compute_eur_per_hour=10.0,
        strategy_usage_cost_eur_per_query={"tfidf": 0.0},
        strategy_fixed_cost_eur={"tfidf": 0.0},
    )
    try:
        _choose_candidate_by_objective(
            ["tfidf", "e5"],
            fit_seconds_by_strategy={"tfidf": 1.0, "e5": 2.0},
            inference_seconds_per_query_by_strategy={"tfidf": 0.01, "e5": 0.02},
            config=config,
        )
    except ValueError as exc:
        assert "missing usage rates=['e5']" in str(exc)
        assert "missing fixed costs=['e5']" in str(exc)
    else:
        raise AssertionError("TCO selector must fail closed for unknown candidate cost components")


import csv
import json
from pathlib import Path

from amorph_autotrainer.cost import CostConfig
from amorph_autotrainer.cli import main as cli_main
from amorph_autotrainer.demo import create_demo_csv
from amorph_autotrainer.pipeline import run_training, _proofgate_may_accept_candidate
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
    assert report["proofgate"]["legacy_parity_status"] == "intentional_inconclusive_self_comparison"
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

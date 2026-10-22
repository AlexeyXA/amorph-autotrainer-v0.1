import csv
import json
from pathlib import Path

from amorph_autotrainer.cost import CostConfig
from amorph_autotrainer.cli import main as cli_main
from amorph_autotrainer.demo import create_demo_csv
from amorph_autotrainer.pipeline import run_training
from amorph_autotrainer.strategies import load_strategy_model, predict_loaded


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
    assert report["status"] in {"accepted", "target_not_met", "baseline_kept_candidate_rejected", "selected_no_absolute_target", "acceptance_gate_underpowered"}
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

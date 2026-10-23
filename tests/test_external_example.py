import csv
import json
import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sklearn_holdout_gate.py"


def _write_dataset(path: Path, *, groups_per_class: int = 0) -> None:
    labels = ("billing", "weather", "music")
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["text", "label"] + (["worker_id"] if groups_per_class else [])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if groups_per_class:
            for label in labels:
                for group_index in range(groups_per_class):
                    group_id = f"{label}-group-{group_index:03d}"
                    # Repeated rows in one source group must stay on the same side of the holdout.
                    for replicate in range(2):
                        writer.writerow({
                            "text": f"{label} support request category {label} group {group_index} variant {replicate}",
                            "label": label,
                            "worker_id": group_id,
                        })
        else:
            for label in labels:
                for sample_index in range(30):
                    writer.writerow({
                        "text": f"{label} support request topic {label} unique sample {sample_index}",
                        "label": label,
                    })


def _run(csv_path: Path, output_path: Path, *, group_column: str | None = None):
    command = [
        sys.executable, str(EXAMPLE), str(csv_path),
        "--n-bootstrap", "100",
        "--min-examples-per-class", "5",
        "--seed", "42",
        "--output", str(output_path),
    ]
    if group_column:
        command.extend(["--group-column", group_column])
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_standalone_sklearn_example_writes_hashed_proofgate_report(tmp_path):
    source = tmp_path / "messages.csv"
    output = tmp_path / "proofgate-report.json"
    _write_dataset(source)

    result = _run(source, output)

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert summary["verdict"] in {"ACCEPT", "REJECT", "INCONCLUSIVE"}
    assert report["verdict"] == summary["verdict"]
    assert report["metric"] == "macro_f1"
    assert report["metadata"]["holdout_role"] == "acceptance_gate_only_not_final_test"
    assert report["metadata"]["split_method"] == "stratified_row_holdout"
    assert report["metadata"]["dataset_sha256"]
    assert len(report["metadata"]["gate_row_indices_sha256"]) == 64
    assert len(report["metadata"]["train_row_indices_sha256"]) == 64
    assert len(report["input_fingerprints"]["payload_sha256"]) == 64
    assert len(report["report_sha256"]) == 64
    assert report["sample"]["n_examples"] > 0


def test_standalone_sklearn_example_accepts_explicit_group_safe_split(tmp_path):
    source = tmp_path / "grouped-messages.csv"
    output = tmp_path / "grouped-report.json"
    _write_dataset(source, groups_per_class=30)

    result = _run(source, output, group_column="worker_id")

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["metadata"]["split_method"] == "group_shuffle_holdout"
    assert report["metadata"]["group_column"] == "worker_id"
    assert report["metadata"]["train_row_count"] + report["metadata"]["gate_row_count"] == report["metadata"]["dataset_row_count"]
    assert len(report["metadata"]["train_group_ids_sha256"]) == 64
    assert len(report["metadata"]["gate_group_ids_sha256"]) == 64
    assert report["verdict"] in {"ACCEPT", "REJECT", "INCONCLUSIVE"}


def test_standalone_sklearn_example_does_not_silently_drop_invalid_rows(tmp_path):
    source = tmp_path / "invalid.csv"
    output = tmp_path / "must-not-exist.json"
    _write_dataset(source)
    with source.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text missing label", ""])

    result = _run(source, output)

    assert result.returncode == 2
    assert "empty label" in result.stderr
    assert not output.exists()

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .cost import CostConfig, estimate_tco_eur
from .data import load_records, split_records
from .experiment_db import ExperimentDB
from .metrics import classification_metrics, gate_candidate, paired_stratified_bootstrap_macro_f1
from .strategies import E5Candidate, TfidfCandidate, FittedCandidate
from .strategy_abi import build_replay_descriptor, run_conformance_checks


class RuntimeBudgetExceededError(RuntimeError):
    """Raised when the configured end-to-end runtime budget expires between stages."""


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_provenance() -> tuple[str | None, bool | None]:
    """Return the source commit and dirty-tree state when launched from a Git checkout."""
    module_path = Path(__file__).resolve()
    for directory in (module_path.parent, *module_path.parents):
        if not (directory / ".git").exists():
            continue
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"], cwd=directory,
                check=True, capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=directory,
                check=True, capture_output=True, text=True, timeout=2,
            ).stdout
            return commit or None, bool(status.strip())
        except (OSError, subprocess.SubprocessError):
            return None, None
    return None, None


def _source_tree_sha256() -> str:
    """Hash the installed Python package sources even when `.git` is absent."""
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    files = sorted(package_root.rglob("*.py"))
    for path in files:
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "amorph-autotrainer", "numpy", "scikit-learn", "scipy", "joblib",
        "sentence-transformers", "transformers", "torch",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _validate_strategy_names(names: list[str]) -> list[str]:
    supported = {"tfidf", "e5"}
    cleaned: list[str] = []
    for name in names:
        name = name.strip().lower()
        if not name:
            continue
        if name not in supported:
            raise ValueError(f"Unsupported strategy '{name}'. Supported: {sorted(supported)}")
        if name not in cleaned:
            cleaned.append(name)
    if "tfidf" not in cleaned:
        cleaned.insert(0, "tfidf")  # always retain a cheap, interpretable baseline
    return cleaned


def run_training(
    *,
    csv_path: str | Path,
    output_dir: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    group_column: str | None = None,
    strategies: list[str] | None = None,
    config: CostConfig | None = None,
    export_features: bool = False,
    e5_model_id: str | None = None,
    e5_model_revision: str | None = E5Candidate.default_model_revision,
    near_duplicate_threshold: float = 0.985,
    max_near_duplicate_audit_rows: int = 5000,
) -> dict[str, Any]:
    started_all = time.perf_counter()
    config = config or CostConfig()
    config.validate()
    strategy_names = _validate_strategy_names(strategies or ["tfidf"])
    strategy_names = strategy_names[:config.max_candidates]
    if "tfidf" not in strategy_names:
        strategy_names = ["tfidf"] + strategy_names[:-1]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    root = Path(output_dir) / run_id
    root.mkdir(parents=True, exist_ok=False)
    database = ExperimentDB(Path(output_dir) / "experiments.sqlite3")
    config_json = json.dumps({
        "text_column": text_column,
        "label_column": label_column,
        "group_column": group_column,
        "strategies": strategy_names,
        "cost_config": config.to_dict(),
        "export_features": export_features,
        "e5_model_id": e5_model_id or E5Candidate.model_id,
        "e5_model_revision": e5_model_revision,
        "near_duplicate_threshold": near_duplicate_threshold,
        "max_near_duplicate_audit_rows": max_near_duplicate_audit_rows,
    }, ensure_ascii=False, sort_keys=True)

    records, data_report, dataset_hash = load_records(
        csv_path, text_column, label_column, group_column=group_column
    )
    if time.perf_counter() - started_all >= config.max_runtime_seconds:
        raise RuntimeBudgetExceededError("Runtime budget exhausted during data loading/validation.")
    bundle = split_records(
        records, dataset_hash, data_report, seed=config.random_seed,
        near_duplicate_threshold=near_duplicate_threshold,
        max_near_duplicate_audit_rows=max_near_duplicate_audit_rows,
        min_examples_per_label_in_gate=config.min_gate_examples_per_class,
        min_examples_per_label_in_other_splits=1,
    )
    _write_json(root / "data_report.json", bundle.report)
    if time.perf_counter() - started_all >= config.max_runtime_seconds:
        raise RuntimeBudgetExceededError("Runtime budget exhausted during near-duplicate audit/splitting.")
    preprocessing_seconds = time.perf_counter() - started_all

    candidates: dict[str, FittedCandidate] = {}
    val_metrics: dict[str, dict[str, Any]] = {}
    candidate_errors: dict[str, str] = {}

    for name in strategy_names:
        elapsed = time.perf_counter() - started_all
        if elapsed >= config.max_runtime_seconds:
            candidate_errors[name] = "skipped: overall runtime budget exhausted before this candidate"
            continue
        candidate_dir = root / "candidates" / name
        created = _timestamp()
        try:
            if name == "tfidf":
                candidate = TfidfCandidate.fit(
                    bundle.train_texts, bundle.train_labels, bundle.val_texts
                )
                strategy_metadata = TfidfCandidate.save(candidate, candidate_dir)
            elif name == "e5":
                candidate = E5Candidate.fit(
                    bundle.train_texts, bundle.train_labels, bundle.val_texts,
                    model_id=e5_model_id,
                    model_revision=e5_model_revision,
                )
                strategy_metadata = E5Candidate.save(candidate, candidate_dir)
            else:
                raise ValueError(f"Unsupported strategy: {name}")

            metrics = classification_metrics(bundle.val_labels, candidate.val_pred)
            candidates[name] = candidate
            val_metrics[name] = metrics
            _write_json(candidate_dir / "validation_metrics.json", metrics)
            _write_json(candidate_dir / "strategy.json", strategy_metadata)

            cost = estimate_tco_eur(
                fit_seconds=preprocessing_seconds + candidate.fit_seconds,
                inference_seconds_per_query=candidate.latency_seconds_per_example,
                config=config,
            )
            database.record({
                "run_id": run_id,
                "strategy": name,
                "created_at_utc": created,
                "dataset_sha256": dataset_hash,
                "config_json": config_json,
                "status": "trained_validation_passed_to_selector",
                "fit_seconds": candidate.fit_seconds,
                "predict_latency_ms": candidate.latency_seconds_per_example * 1000,
                "validation_metrics_json": json.dumps(metrics),
                "test_metrics_json": None,
                "statistical_gate_json": None,
                "cost_json": json.dumps(cost),
                "artifact_path": str(candidate_dir),
                "error": None,
            })
        except Exception as exc:
            candidate_errors[name] = f"{type(exc).__name__}: {exc}"
            database.record({
                "run_id": run_id,
                "strategy": name,
                "created_at_utc": created,
                "dataset_sha256": dataset_hash,
                "config_json": config_json,
                "status": "failed",
                "error": candidate_errors[name],
            })
            # The baseline is required to return a meaningful result.
            if name == "tfidf":
                raise RuntimeError(f"Required TF-IDF baseline failed: {exc}") from exc

    # Strategy conformance is checked before selector/gate bootstrap use.
    strategy_conformance: dict[str, dict[str, Any]] = {}

    def predict_for_conformance(candidate: FittedCandidate, name: str, texts: list[str]) -> np.ndarray:
        if name == "e5":
            features = E5Candidate._encode(candidate.model["encoder"], texts)
            return candidate.model["classifier"].predict(features)
        return candidate.model.predict(texts)

    for name in list(candidates):
        candidate = candidates[name]
        probe_texts = bundle.val_texts[:min(3, len(bundle.val_texts))]
        try:
            check = run_conformance_checks(
                lambda texts, candidate=candidate, strategy_name=name:
                    predict_for_conformance(candidate, strategy_name, texts),
                strategy_name=name,
                sample_texts=probe_texts,
            )
            strategy_conformance[name] = check
        except Exception as exc:
            candidate_errors[name] = f"CONFORMANCE_FAILED: {type(exc).__name__}: {exc}"
            database.record({
                "run_id": run_id,
                "strategy": name,
                "created_at_utc": _timestamp(),
                "dataset_sha256": dataset_hash,
                "config_json": config_json,
                "status": "conformance_failed",
                "error": candidate_errors[name],
            })
            candidates.pop(name, None)
            val_metrics.pop(name, None)
            if name == "tfidf":
                raise RuntimeError(f"Required TF-IDF baseline failed strategy conformance: {exc}") from exc

    if not candidates:
        raise RuntimeError(f"No conforming candidate completed successfully. Errors: {candidate_errors}")
    if "tfidf" not in candidates:
        raise RuntimeError("Required TF-IDF baseline is missing after conformance checks")

    best_val_f1 = max(m["macro_f1"] for m in val_metrics.values())
    if config.min_macro_f1 is None:
        threshold = best_val_f1 - config.noninferiority_margin
    else:
        threshold = config.min_macro_f1

    eligible = [
        name for name, met in val_metrics.items()
        if met["macro_f1"] >= threshold
    ]
    selection_rule = "cheapest runtime proxy among validation candidates meeting threshold"
    if not eligible:
        # No method meets target: do not hide the failure; keep the best validation candidate
        # for final diagnosis, but the global quality gate will fail.
        eligible = [max(val_metrics, key=lambda name: val_metrics[name]["macro_f1"])]
        selection_rule = "no candidate met threshold; best validation candidate selected for diagnosis"

    # Runtime proxy is used even when free compute has a 0 EUR price.
    def runtime_proxy(name: str) -> float:
        c = candidates[name]
        return c.fit_seconds + config.expected_queries * c.latency_seconds_per_example

    selected_name = min(eligible, key=runtime_proxy)
    selected = candidates[selected_name]
    baseline = candidates["tfidf"]

    def predict_candidate(candidate: FittedCandidate, name: str, texts: list[str]) -> np.ndarray:
        if name == "e5":
            features = E5Candidate._encode(candidate.model["encoder"], texts)
            return candidate.model["classifier"].predict(features)
        return candidate.model.predict(texts)

    # Independent acceptance-gate split: never used for strategy selection or final reporting.
    gate_baseline_pred = predict_candidate(baseline, "tfidf", bundle.gate_texts)
    gate_selected_pred = predict_candidate(selected, selected_name, bundle.gate_texts)
    baseline_gate_metrics = classification_metrics(bundle.gate_labels, gate_baseline_pred)
    selected_gate_metrics = classification_metrics(bundle.gate_labels, gate_selected_pred)
    comparison = paired_stratified_bootstrap_macro_f1(
        bundle.gate_labels, gate_baseline_pred, gate_selected_pred,
        n_bootstrap=config.bootstrap_samples, seed=config.random_seed,
    )
    selected_gate = gate_candidate(
        selected_gate_metrics,
        baseline_gate_metrics,
        comparison,
        min_macro_f1=config.min_macro_f1,
        noninferiority_margin=config.noninferiority_margin,
    )

    baseline_gate = gate_candidate(
        baseline_gate_metrics,
        baseline_gate_metrics,
        paired_stratified_bootstrap_macro_f1(
            bundle.gate_labels, gate_baseline_pred, gate_baseline_pred,
            n_bootstrap=config.bootstrap_samples, seed=config.random_seed,
        ),
        min_macro_f1=config.min_macro_f1,
        noninferiority_margin=config.noninferiority_margin,
    )

    gate_class_counts = {
        str(label): int(np.sum(bundle.gate_labels == label))
        for label in sorted(set(bundle.gate_labels.tolist()))
    }
    gate_sufficient = (
        len(gate_class_counts) >= 2
        and min(gate_class_counts.values()) >= config.min_gate_examples_per_class
    )
    for gate_result in (selected_gate, baseline_gate):
        gate_result["observed_examples_per_class"] = gate_class_counts
        gate_result["minimum_examples_per_class"] = config.min_gate_examples_per_class
        gate_result["statistical_sample_status"] = "sufficient_for_configured_floor" if gate_sufficient else "underpowered"
        if not gate_sufficient:
            gate_result["passed"] = False
            gate_result.setdefault("reasons", []).append("acceptance_gate_underpowered")

    # Selection/rollback decisions use only validation and acceptance-gate data.
    # The untouched test split is evaluated only after final_name has been fixed.
    candidate_accepted = selected_gate["passed"]
    if selected_name == "tfidf":
        candidate_accepted = baseline_gate["passed"]
    if candidate_accepted:
        final_name = selected_name
        final_gate = selected_gate if selected_name != "tfidf" else baseline_gate
    else:
        final_name = "tfidf"
        final_gate = baseline_gate
    final_candidate = candidates[final_name]

    absolute_target_passed = (
        None if config.min_macro_f1 is None or not gate_sufficient
        else final_gate["candidate_macro_f1"] >= config.min_macro_f1
    )
    if not gate_sufficient:
        final_status = "acceptance_gate_underpowered"
    elif absolute_target_passed is False:
        final_status = "target_not_met"
    elif selected_name != final_name:
        final_status = "baseline_kept_candidate_rejected"
    elif absolute_target_passed is None:
        final_status = "selected_no_absolute_target"
    else:
        final_status = "accepted"

    # Final test is untouched by the selection and gate decisions. Results below are
    # report-only; they do not change which model is selected.
    baseline_test_pred = predict_candidate(baseline, "tfidf", bundle.test_texts)
    selected_test_pred = predict_candidate(selected, selected_name, bundle.test_texts)
    final_test_pred = baseline_test_pred if final_name == "tfidf" else selected_test_pred
    baseline_test_metrics = classification_metrics(bundle.test_labels, baseline_test_pred)
    selected_test_metrics = classification_metrics(bundle.test_labels, selected_test_pred)
    final_metrics = baseline_test_metrics if final_name == "tfidf" else selected_test_metrics
    final_artifact_dir = root / "selected"
    if final_artifact_dir.exists():
        import shutil
        shutil.rmtree(final_artifact_dir)
    final_artifact_dir.mkdir(parents=True)

    if final_name == "tfidf":
        TfidfCandidate.save(final_candidate, final_artifact_dir)
        strategy_meta = {"strategy": "tfidf", "model_file": "model.joblib", **final_candidate.metadata}
    else:
        strategy_meta = E5Candidate.save(final_candidate, final_artifact_dir)

    commit_sha, git_dirty = _git_provenance()
    source_tree_sha256 = _source_tree_sha256()
    strategy_classes = {"tfidf": TfidfCandidate, "e5": E5Candidate}
    strategy_fingerprints: dict[str, str] = {}
    strategy_status: dict[str, str] = {}
    for name, candidate in candidates.items():
        strategy_class = strategy_classes[name]
        strategy_status[name] = strategy_class.status().value
        if name == "e5":
            strategy_fingerprints[name] = strategy_class.fingerprint(
                code_revision=source_tree_sha256,
                model_id=candidate.model.get("model_id"),
                model_revision=candidate.model.get("model_revision"),
            )
        else:
            strategy_fingerprints[name] = strategy_class.fingerprint(code_revision=source_tree_sha256)

    protocol_status = {
        "strategy_abi": "implemented",
        "conformance_gate": "implemented",
        "gate_row_count_constraints": (
            "enforced_by_milp"
            if bundle.report.get("group_split_optimizer") is not None
            else "not_enforced_by_group_milp_fallback"
        ),
    }
    replay_descriptor = build_replay_descriptor(
        commit_sha=commit_sha,
        source_tree_sha256=source_tree_sha256,
        dataset_sha256=dataset_hash,
        dataset_revision=bundle.report.get("dataset_revision"),
        split_hashes=bundle.report.get("splits", {}),
        seed=config.random_seed,
        library_versions=_library_versions(),
        strategy_fingerprints=strategy_fingerprints,
        selected_strategy=final_name,
        acceptance_gate_status=final_status,
        acceptance_gate_sufficient=gate_sufficient,
        protocol_status=protocol_status,
        extra={
            "config": json.loads(config_json),
            "strategy_conformance": strategy_conformance,
            "split_gate_design": bundle.report.get("gate_design", {}),
        },
    )
    _write_json(root / "replay.json", replay_descriptor)
    manifest = {
        "format_version": 1,
        "project": "AMORPH AutoTrainer",
        "project_version": "0.1.0",
        "commit_sha": commit_sha,
        "git_working_tree_dirty": git_dirty,
        "source_tree_sha256": source_tree_sha256,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "library_versions": _library_versions(),
        "run_id": run_id,
        "strategy": final_name,
        "artifact_subdirectory": "selected",
        "created_at_utc": _timestamp(),
        "dataset_sha256": dataset_hash,
        "data_report": "data_report.json",
        "config": json.loads(config_json),
        "split_config": {
            "group_column": group_column,
            "near_duplicate_threshold": near_duplicate_threshold,
            "max_near_duplicate_audit_rows": max_near_duplicate_audit_rows,
            "random_seed": config.random_seed,
        },
        "strategy_metadata": strategy_meta,
        "strategy_fingerprints": strategy_fingerprints,
        "strategy_status": strategy_status,
        "protocol_status": protocol_status,
        "conformance_gate": strategy_conformance,
        "replay_descriptor_path": "replay.json",
        "replay_descriptor": replay_descriptor,
        "validation_metrics_all_candidates": val_metrics,
        "candidate_errors": candidate_errors,
        "selected_before_acceptance_gate": selected_name,
        "selection_rule": selection_rule,
        "acceptance_gate_metrics": {
            "baseline": baseline_gate_metrics,
            "candidate": selected_gate_metrics,
        },
        "selected_candidate_gate": selected_gate,
        "baseline_gate": baseline_gate,
        "final_acceptance_gate": final_gate,
        "final_test_metrics": final_metrics,
        "test_metrics_baseline_reference": baseline_test_metrics,
        "test_metrics_pre_gate_candidate_report_only": selected_test_metrics,
        "absolute_target_passed_on_gate": absolute_target_passed,
        "acceptance_gate_sufficient": gate_sufficient,
        "acceptance_gate_examples_per_class": gate_class_counts,
        "test_split_used_for_selection": False,
        "status": final_status,
        "estimated_tco": estimate_tco_eur(
            fit_seconds=final_candidate.fit_seconds,
            inference_seconds_per_query=final_candidate.latency_seconds_per_example,
            config=config,
        ),
        "limitations": [
            "The paired bootstrap interval estimates uncertainty on the independent acceptance-gate sample conditional on fitted models; it does not capture retraining variance.",
            "The final test split is report-only and is not used to select the model or trigger a fallback.",
            "Near-duplicate detection is a character n-gram heuristic; group/time/source leakage needs domain-specific policy.",
            "The runtime budget is best-effort between pipeline stages; an in-progress library fit is not forcibly interrupted.",
            "Estimated TCO depends on user-supplied rates and expected query volume.",
        ],
    }
    # The selected bundle is self-contained; root report also points to it.
    selected_manifest = dict(manifest)
    selected_manifest["artifact_subdirectory"] = "."
    _write_json(final_artifact_dir / "manifest.json", selected_manifest)
    root_manifest = dict(manifest)
    root_manifest["artifact_subdirectory"] = "selected"
    _write_json(root / "manifest.json", root_manifest)
    _write_json(root / "report.json", manifest)

    if export_features:
        try:
            from scipy import sparse
            features_dir = root / "selected_features"
            features_dir.mkdir(exist_ok=True)
            final_loaded_model = final_candidate.model
            if final_name == "tfidf":
                for split_name, texts in [
                    ("train", bundle.train_texts), ("validation", bundle.val_texts),
                    ("acceptance_gate", bundle.gate_texts), ("test", bundle.test_texts),
                ]:
                    matrix = final_loaded_model.named_steps["features"].transform(texts)
                    sparse.save_npz(features_dir / f"{split_name}.npz", matrix)
                manifest["features"] = {
                    "format": "scipy sparse NPZ",
                    "path": "selected_features/",
                    "warning": "Feature matrices can encode sensitive information; protect artifacts accordingly.",
                }
            else:
                for split_name, texts in [
                    ("train", bundle.train_texts), ("validation", bundle.val_texts),
                    ("acceptance_gate", bundle.gate_texts), ("test", bundle.test_texts),
                ]:
                    matrix = E5Candidate._encode(final_loaded_model["encoder"], texts)
                    np.savez_compressed(features_dir / f"{split_name}.npz", features=matrix)
                manifest["features"] = {
                    "format": "dense float embeddings in NPZ",
                    "path": "selected_features/",
                    "dimension": int(final_loaded_model["encoder"].get_sentence_embedding_dimension()),
                    "warning": "Embeddings can encode sensitive information; protect artifacts accordingly.",
                }
            _write_json(root / "report.json", manifest)
            selected_manifest = dict(manifest)
            selected_manifest["artifact_subdirectory"] = "."
            _write_json(final_artifact_dir / "manifest.json", selected_manifest)
            root_manifest = dict(manifest)
            root_manifest["artifact_subdirectory"] = "selected"
            _write_json(root / "manifest.json", root_manifest)
        except Exception as exc:
            manifest["feature_export_error"] = f"{type(exc).__name__}: {exc}"
            _write_json(root / "report.json", manifest)
            selected_manifest = dict(manifest)
            selected_manifest["artifact_subdirectory"] = "."
            _write_json(final_artifact_dir / "manifest.json", selected_manifest)
            root_manifest = dict(manifest)
            root_manifest["artifact_subdirectory"] = "selected"
            _write_json(root / "manifest.json", root_manifest)

    elapsed_total_seconds = time.perf_counter() - started_all
    runtime_budget = {
        "max_runtime_seconds": config.max_runtime_seconds,
        "actual_runtime_seconds": elapsed_total_seconds,
        "exceeded": elapsed_total_seconds > config.max_runtime_seconds,
        "enforcement": "best_effort_between_stages",
        "note": "A currently running sklearn/encoder operation is not forcibly interrupted; overrun is reported, and no next candidate is started after budget exhaustion.",
    }
    manifest["runtime_budget"] = runtime_budget
    manifest["selected_model_tco"] = estimate_tco_eur(
        fit_seconds=preprocessing_seconds + final_candidate.fit_seconds,
        inference_seconds_per_query=final_candidate.latency_seconds_per_example,
        config=config,
    )
    manifest["full_selection_run_tco"] = estimate_tco_eur(
        fit_seconds=elapsed_total_seconds,
        inference_seconds_per_query=final_candidate.latency_seconds_per_example,
        config=config,
    )
    manifest["estimated_tco"] = manifest["full_selection_run_tco"]
    _write_json(root / "report.json", manifest)
    selected_manifest = dict(manifest)
    selected_manifest["artifact_subdirectory"] = "."
    _write_json(final_artifact_dir / "manifest.json", selected_manifest)
    root_manifest = dict(manifest)
    root_manifest["artifact_subdirectory"] = "selected"
    _write_json(root / "manifest.json", root_manifest)

    # Update database with final gate information for selected and baseline candidates.
    for name in set([selected_name, "tfidf"]):
        cand = candidates[name]
        test_met = selected_test_metrics if name == selected_name else baseline_test_metrics
        gate_json = json.dumps(selected_gate if name == selected_name else baseline_gate, ensure_ascii=False) if name in (selected_name, "tfidf") else None
        database.record({
            "run_id": run_id,
            "strategy": name,
            "created_at_utc": _timestamp(),
            "dataset_sha256": dataset_hash,
            "config_json": config_json,
            "status": (
                final_status if name == final_name
                else "candidate_rejected" if name == selected_name and selected_name != final_name
                else "baseline_reference"
            ),
            "fit_seconds": cand.fit_seconds,
            "predict_latency_ms": cand.latency_seconds_per_example * 1000,
            "validation_metrics_json": json.dumps(val_metrics[name]),
            "test_metrics_json": json.dumps(test_met),
            "statistical_gate_json": gate_json,
            "cost_json": json.dumps(estimate_tco_eur(
                fit_seconds=preprocessing_seconds + cand.fit_seconds,
                inference_seconds_per_query=cand.latency_seconds_per_example,
                config=config,
            )),
            "artifact_path": str(root / "candidates" / name),
            "error": None,
        })

    return {
        "run_id": run_id,
        "run_dir": str(root),
        "database_path": str(Path(output_dir) / "experiments.sqlite3"),
        "status": final_status,
        "selected_strategy": final_name,
        "selected_before_acceptance_gate": selected_name,
        "validation_metrics": val_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "selected_candidate_test_metrics": selected_test_metrics,
        "acceptance_gate": selected_gate,
        "final_acceptance_gate": final_gate,
        "candidate_errors": candidate_errors,
        "report_path": str(root / "report.json"),
        "manifest_path": str(final_artifact_dir / "manifest.json"),
        "model_path": str(final_artifact_dir),
        "absolute_target_passed": absolute_target_passed,
        "acceptance_gate_sufficient": gate_sufficient,
        "runtime_budget": runtime_budget,
        "runtime_budget_exceeded": runtime_budget["exceeded"],
    }

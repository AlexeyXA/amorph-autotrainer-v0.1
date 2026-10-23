"""Proof Gate v0.1: evaluate precomputed predictions without training or model selection.

Only imports the existing metric helpers. This module deliberately does not import
AMORPH strategies, dataset loaders, split code, or model-training code.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..metrics import (
    classification_metrics,
    gate_candidate,
    paired_stratified_bootstrap_macro_f1,
)

PROOFGATE_VERSION = "0.1.0"
REPORT_SCHEMA_VERSION = "proofgate.report.v1"


def _finite_real(value: Any, name: str) -> float:
    """Accept real numeric values only; reject bools and numeric-looking strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


class Verdict:
    """String constants for the only three top-level Proof Gate decisions."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class GateReport:
    """Serializable, replay-oriented result for one paired prediction comparison."""

    verdict: str
    subtype: str | None
    reason_codes: tuple[str, ...]
    metric: str
    criteria: dict[str, Any]
    sample_size: int
    class_counts: dict[str, int]
    baseline_metrics: dict[str, Any] | None
    candidate_metrics: dict[str, Any] | None
    comparison: dict[str, Any] | None
    conformance: dict[str, Any]
    powered: bool
    input_sha256: str
    input_hashes: dict[str, str]
    metadata: dict[str, Any]
    legacy_gate_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible report data with a content digest."""
        payload: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "proofgate_version": PROOFGATE_VERSION,
            "verdict": self.verdict,
            "subtype": self.subtype,
            "reason_codes": list(self.reason_codes),
            "metric": self.metric,
            "criteria": self.criteria,
            "sample": {
                "n_examples": self.sample_size,
                "class_counts": self.class_counts,
                "powered": self.powered,
            },
            "baseline_metrics": self.baseline_metrics,
            "candidate_metrics": self.candidate_metrics,
            "comparison": self.comparison,
            "conformance": self.conformance,
            "input_fingerprints": {
                "payload_sha256": self.input_sha256,
                "array_sha256": self.input_hashes,
            },
            "metadata": self.metadata,
            "legacy_gate_result": self.legacy_gate_result,
            "decision_rule": {
                "noninferiority_boundary": "inclusive: ci_lower >= -margin",
                "absolute_target": "candidate Macro-F1 must be >= min_macro_f1 when configured",
                "inconclusive_never_accepts": True,
            },
        }
        payload["report_sha256"] = _sha256_json(payload)
        return payload

    def to_legacy_gate_dict(self) -> dict[str, Any]:
        """Adapter for AMORPH's existing gate_candidate-shaped manifest fields.

        For INCONCLUSIVE cases, passed is always false. A self-comparison does not
        acquire a synthetic zero-width confidence interval and is never represented
        as acceptance.
        """
        if self.legacy_gate_result is not None:
            legacy = dict(self.legacy_gate_result)
        else:
            baseline_f1 = (
                self.baseline_metrics.get("macro_f1")
                if self.baseline_metrics is not None
                else None
            )
            candidate_f1 = (
                self.candidate_metrics.get("macro_f1")
                if self.candidate_metrics is not None
                else None
            )
            legacy = {
                "passed": False,
                "reasons": list(self.reason_codes),
                "noninferiority_margin": self.criteria["margin"],
                "candidate_macro_f1": candidate_f1,
                "baseline_macro_f1": baseline_f1,
                "comparison": self.comparison,
                "interpretation": (
                    "Proof Gate returned INCONCLUSIVE; this is not an acceptance."
                ),
            }
        legacy["proofgate_verdict"] = self.verdict
        legacy["proofgate_subtype"] = self.subtype
        legacy["proofgate_reason_codes"] = list(self.reason_codes)
        if self.verdict == Verdict.INCONCLUSIVE:
            legacy["passed"] = False
        return legacy


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _to_builtin(value.item())
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return {"__nonfinite_float__": "nan"}
        if math.isinf(value):
            return {"__nonfinite_float__": "positive_infinity" if value > 0 else "negative_infinity"}
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        _to_builtin(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint_inputs(
    y_gate: Any,
    pred_baseline: Any,
    pred_candidate: Any,
    criteria: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
) -> tuple[str, dict[str, str]]:
    hashes = {
        "y_gate": _sha256_json(y_gate),
        "pred_baseline": _sha256_json(pred_baseline),
        "pred_candidate": _sha256_json(pred_candidate),
    }
    full = {
        "y_gate": _to_builtin(y_gate),
        "pred_baseline": _to_builtin(pred_baseline),
        "pred_candidate": _to_builtin(pred_candidate),
        "criteria": dict(criteria),
        "metadata": dict(metadata or {}),
    }
    return _sha256_json(full), hashes


def _has_missing_or_nonfinite_labels(values: np.ndarray) -> bool:
    for value in values.tolist():
        if value is None:
            return True
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return True
        if isinstance(value, str) and value.casefold() == "nan":
            return True
    return False


def _class_counts(y_gate: np.ndarray) -> dict[str, int]:
    counts = Counter(str(value) for value in y_gate.tolist())
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _safe_metrics(y_gate: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return classification_metrics(y_gate, predictions)


def evaluate(
    y_gate: Any,
    pred_baseline: Any,
    pred_candidate: Any | None,
    *,
    metric: str = "macro_f1",
    margin: float = 0.02,
    n_bootstrap: int = 500,
    seed: int = 42,
    min_examples_per_class: int = 5,
    min_macro_f1: float | None = None,
    confidence: float = 0.95,
    metadata: Mapping[str, Any] | None = None,
) -> GateReport:
    """Evaluate precomputed baseline/candidate predictions on a gate sample.

    ACCEPT requires sufficient class support, valid 1-D equal-length arrays, a real
    challenger, inclusive non-inferiority (CI lower bound >= -margin), and an
    optional absolute Macro-F1 target. REJECT means the powered gate failed criteria.
    Shape/length defects, underpowered samples, or no real challenger are INCONCLUSIVE.

    The bootstrap interval is conditional on the supplied predictions; it does not
    quantify model-retraining variability.
    """
    if metric != "macro_f1":
        raise ValueError("Proof Gate v0.1 supports metric='macro_f1' only")
    margin_value = _finite_real(margin, "margin")
    if margin_value < 0:
        raise ValueError("margin must be >= 0")
    confidence_value = _finite_real(confidence, "confidence")
    if not 0 < confidence_value < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    target_value = None if min_macro_f1 is None else _finite_real(min_macro_f1, "min_macro_f1")
    if target_value is not None and not 0 <= target_value <= 1:
        raise ValueError("min_macro_f1 must be None or a value in [0, 1]")
    if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap < 100:
        raise ValueError("n_bootstrap must be an integer >= 100")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(min_examples_per_class, int) or isinstance(min_examples_per_class, bool) or min_examples_per_class < 1:
        raise ValueError("min_examples_per_class must be an integer >= 1")

    criteria = {
        "metric": metric,
        "margin": margin_value,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "min_examples_per_class": min_examples_per_class,
        "min_macro_f1": target_value,
        "confidence": confidence_value,
    }
    metadata_dict = dict(metadata or {})
    input_sha256, input_hashes = _fingerprint_inputs(
        y_gate, pred_baseline, pred_candidate, criteria, metadata_dict
    )

    array_conversion_errors: list[str] = []
    try:
        raw_y = np.asarray(y_gate)
    except (TypeError, ValueError):
        raw_y = np.asarray([], dtype=object)
        array_conversion_errors.append("y_gate_array_conversion_failed")
    try:
        raw_baseline = np.asarray(pred_baseline)
    except (TypeError, ValueError):
        raw_baseline = np.asarray([], dtype=object)
        array_conversion_errors.append("pred_baseline_array_conversion_failed")
    if pred_candidate is None:
        raw_candidate = None
    else:
        try:
            raw_candidate = np.asarray(pred_candidate)
        except (TypeError, ValueError):
            raw_candidate = np.asarray([], dtype=object)
            array_conversion_errors.append("pred_candidate_array_conversion_failed")
    n = int(raw_y.shape[0]) if raw_y.ndim == 1 else 0
    counts: dict[str, int] = _class_counts(raw_y) if raw_y.ndim == 1 else {}

    def report(
        verdict: str,
        subtype: str | None,
        reasons: list[str],
        *,
        baseline_metrics: dict[str, Any] | None = None,
        candidate_metrics: dict[str, Any] | None = None,
        comparison: dict[str, Any] | None = None,
        conformance: dict[str, Any] | None = None,
        powered: bool = False,
        legacy: dict[str, Any] | None = None,
    ) -> GateReport:
        return GateReport(
            verdict=verdict,
            subtype=subtype,
            reason_codes=tuple(reasons),
            metric=metric,
            criteria=criteria,
            sample_size=n,
            class_counts=counts,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            comparison=comparison,
            conformance=conformance or {"passed": False, "errors": list(reasons)},
            powered=powered,
            input_sha256=input_sha256,
            input_hashes=input_hashes,
            metadata=metadata_dict,
            legacy_gate_result=legacy,
        )

    # Array shape and conformance checks are fail-closed and return a report rather
    # than raising, so callers can archive the failure evidence.
    shape_errors: list[str] = list(array_conversion_errors)
    if raw_y.ndim != 1:
        shape_errors.append("y_gate_must_be_one_dimensional")
    if raw_baseline.ndim != 1:
        shape_errors.append("pred_baseline_must_be_one_dimensional")
    if raw_y.ndim == 1 and raw_baseline.ndim == 1 and len(raw_y) != len(raw_baseline):
        shape_errors.append("baseline_length_mismatch")
    if raw_candidate is not None:
        if raw_candidate.ndim != 1:
            shape_errors.append("pred_candidate_must_be_one_dimensional")
        elif raw_y.ndim == 1 and len(raw_y) != len(raw_candidate):
            shape_errors.append("candidate_length_mismatch")
    if n == 0:
        shape_errors.append("empty_gate_sample")
    if raw_y.ndim == 1 and _has_missing_or_nonfinite_labels(raw_y):
        shape_errors.append("y_gate_contains_missing_or_nonfinite_label")
    if raw_baseline.ndim == 1 and _has_missing_or_nonfinite_labels(raw_baseline):
        shape_errors.append("pred_baseline_contains_missing_or_nonfinite_label")
    if raw_candidate is not None and raw_candidate.ndim == 1 and _has_missing_or_nonfinite_labels(raw_candidate):
        shape_errors.append("pred_candidate_contains_missing_or_nonfinite_label")

    # The gate labels define the observed class taxonomy for this v0.1 API.
    # If a prediction contains an unseen label, we cannot distinguish a genuinely
    # unknown output from a class omitted from the supplied gate. Do not let the
    # point metrics (which include predicted-only labels) and paired bootstrap
    # (which uses y_gate labels) silently evaluate different label universes.
    if not shape_errors and raw_y.ndim == 1 and raw_baseline.ndim == 1 and (
        raw_candidate is None or raw_candidate.ndim == 1
    ):
        try:
            observed_labels = set(raw_y.tolist())
            prediction_sets = [("pred_baseline", set(raw_baseline.tolist()))]
            if raw_candidate is not None:
                prediction_sets.append(("pred_candidate", set(raw_candidate.tolist())))
        except TypeError:
            shape_errors.append("labels_must_be_hashable_scalars")
        else:
            for prediction_name, prediction_labels in prediction_sets:
                if prediction_labels - observed_labels:
                    shape_errors.append(
                        f"{prediction_name}_contains_label_absent_from_y_gate"
                    )

    if shape_errors:
        return report(
            Verdict.INCONCLUSIVE,
            "CONFORMANCE_FAILED",
            shape_errors,
            conformance={"passed": False, "errors": shape_errors},
            powered=False,
        )

    y = raw_y
    baseline = raw_baseline
    candidate = raw_candidate
    class_counts = Counter(y.tolist())
    support_powered = len(class_counts) >= 2 and all(
        count >= min_examples_per_class for count in class_counts.values()
    )

    try:
        baseline_metrics = _safe_metrics(y, baseline)
    except (TypeError, ValueError) as exc:
        reason = f"baseline_metric_computation_failed:{type(exc).__name__}"
        return report(
            Verdict.INCONCLUSIVE, "CONFORMANCE_FAILED", [reason],
            conformance={"passed": False, "errors": [reason]},
        )

    if candidate is None:
        reasons = ["no_candidate_supplied"]
        if not support_powered:
            reasons.append("acceptance_gate_underpowered")
        return report(
            Verdict.INCONCLUSIVE,
            "BASELINE_ONLY",
            reasons,
            baseline_metrics=baseline_metrics,
            candidate_metrics=None,
            comparison=None,
            conformance={"passed": True, "errors": []},
            powered=support_powered,
        )

    try:
        candidate_metrics = _safe_metrics(y, candidate)
    except (TypeError, ValueError) as exc:
        reason = f"candidate_metric_computation_failed:{type(exc).__name__}"
        return report(
            Verdict.INCONCLUSIVE, "CONFORMANCE_FAILED", [reason],
            baseline_metrics=baseline_metrics,
            candidate_metrics=None,
            conformance={"passed": False, "errors": [reason]},
            powered=support_powered,
        )

    too_few_classes = len(class_counts) < 2
    low_support_labels = [
        str(label) for label, count in class_counts.items()
        if count < min_examples_per_class
    ]
    if too_few_classes or low_support_labels:
        reasons = []
        if too_few_classes:
            reasons.append("fewer_than_two_gate_classes")
        if low_support_labels:
            reasons.append("min_examples_per_class_not_met")
        return report(
            Verdict.INCONCLUSIVE,
            "UNDERPOWERED",
            reasons,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            comparison=None,
            conformance={"passed": True, "errors": []},
            powered=False,
        )

    if np.array_equal(baseline, candidate):
        return report(
            Verdict.INCONCLUSIVE,
            "SELF_COMPARISON",
            ["baseline_and_candidate_predictions_identical"],
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            comparison=None,
            conformance={"passed": True, "errors": []},
            powered=True,
        )

    try:
        comparison = paired_stratified_bootstrap_macro_f1(
            y,
            baseline,
            candidate,
            n_bootstrap=n_bootstrap,
            seed=seed,
            confidence=confidence_value,
        )
        legacy = gate_candidate(
            candidate_metrics,
            baseline_metrics,
            comparison,
            min_macro_f1=target_value,
            noninferiority_margin=margin_value,
        )
    except (TypeError, ValueError) as exc:
        reason = f"comparison_failed:{type(exc).__name__}"
        return report(
            Verdict.INCONCLUSIVE, "CONFORMANCE_FAILED", [reason],
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            conformance={"passed": False, "errors": [reason]},
            powered=True,
        )

    if legacy["passed"]:
        return report(
            Verdict.ACCEPT,
            None,
            [],
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            comparison=comparison,
            conformance={"passed": True, "errors": []},
            powered=True,
            legacy=legacy,
        )
    return report(
        Verdict.REJECT,
        None,
        list(legacy["reasons"]),
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        comparison=comparison,
        conformance={"passed": True, "errors": []},
        powered=True,
        legacy=legacy,
    )


def write_report(report: GateReport, path: str | Path) -> dict[str, Any]:
    """Write a deterministic, hashed report.json and return its payload."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload

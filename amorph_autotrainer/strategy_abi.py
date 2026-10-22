"""Formal strategy contract, conformance checks, fingerprints, and replay descriptor.

These helpers keep multi-strategy selection auditable without requiring a heavy
plugin framework. New strategies should implement the StrategyABI surface
(identity / version / status / capabilities / fingerprint / fit / predict path).
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np


class ComponentStatus(str, Enum):
    """Lifecycle status for strategies and protocol pieces (SSP-inspired, minimal)."""

    CANONICAL = "canonical"  # default baseline; treated as production-capable when gate passes
    IMPLEMENTED = "implemented"  # complete path, not necessarily the preferred default
    EXPERIMENTAL = "experimental"  # works in code; evidence may be incomplete
    LEGACY = "legacy"  # kept for replay only
    HYPOTHESIS = "hypothesis"  # design-only / out of scope for current acceptance


@runtime_checkable
class StrategyABI(Protocol):
    """Minimal contract expected of a trainable strategy class."""

    name: str

    @classmethod
    def identity(cls) -> str:
        ...

    @classmethod
    def version(cls) -> str:
        ...

    @classmethod
    def status(cls) -> ComponentStatus:
        ...

    @classmethod
    def capabilities(cls) -> dict[str, Any]:
        ...


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strategy_fingerprint(
    *,
    identity: str,
    version: str,
    hyperparameters: Mapping[str, Any],
    code_revision: str | None = None,
    source_commit_sha: str | None = None,
    model_revision: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Stable SHA-256 of strategy configuration used for replay comparisons.

    Includes optional code_revision (e.g. package source_tree_sha256 or git commit)
    so silent preprocessing changes invalidate prior TF-IDF vs TF-IDF comparisons.
    """
    payload: dict[str, Any] = {
        "identity": identity,
        "version": version,
        "hyperparameters": dict(hyperparameters),
        "code_revision": code_revision,
        "source_commit_sha": source_commit_sha,
        "model_revision": model_revision,
    }
    if extra:
        payload["extra"] = dict(extra)
    return sha256_hex(canonical_json(payload))


def run_conformance_checks(
    predict_fn: Callable[[list[str]], np.ndarray],
    *,
    strategy_name: str,
    sample_texts: list[str],
    expected_label_dtype: type = np.str_,
) -> dict[str, Any]:
    """Cheap pre-acceptance checks: types, lengths, empty input, non-crash.

    Fail-fast before bootstrap / large official-test evaluation.
    """
    errors: list[str] = []
    checks: dict[str, Any] = {"strategy": strategy_name}

    # Empty batch — preferred contract is ndarray length 0; sklearn TF-IDF may
    # reject zero-row inputs. Record soft status instead of hard-failing baseline.
    try:
        empty_pred = predict_fn([])
        if not isinstance(empty_pred, np.ndarray):
            errors.append("predict([]) must return numpy.ndarray")
        elif len(empty_pred) != 0:
            errors.append(f"predict([]) length must be 0, got {len(empty_pred)}")
        else:
            checks["empty_input"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["empty_input"] = f"unsupported: {type(exc).__name__}: {exc}"

    # Non-empty batch
    if not sample_texts:
        sample_texts = ["conformance probe"]
    try:
        pred = predict_fn(list(sample_texts))
        if not isinstance(pred, np.ndarray):
            errors.append("predict(sample) must return numpy.ndarray")
        elif len(pred) != len(sample_texts):
            errors.append(
                f"predict length mismatch: len(texts)={len(sample_texts)} len(pred)={len(pred)}"
            )
        else:
            checks["length_match"] = "ok"
            checks["prediction_dtype"] = str(pred.dtype)
            if pred.size and not (
                np.issubdtype(pred.dtype, np.str_)
                or np.issubdtype(pred.dtype, np.object_)
                or np.issubdtype(pred.dtype, np.integer)
            ):
                errors.append(f"unexpected prediction dtype: {pred.dtype}")
            if pred.size and np.any(pred.astype(str) == "nan"):
                errors.append("predictions contain string 'nan'")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"predict(sample) raised {type(exc).__name__}: {exc}")

    checks["passed"] = not errors
    checks["errors"] = errors
    if errors:
        raise ValueError(
            f"Strategy conformance failed for '{strategy_name}': " + "; ".join(errors)
        )
    return checks


def build_replay_descriptor(
    *,
    commit_sha: str | None,
    source_tree_sha256: str | None,
    dataset_sha256: str,
    dataset_revision: str | None,
    split_hashes: Mapping[str, Any],
    seed: int,
    library_versions: Mapping[str, Any],
    strategy_fingerprints: Mapping[str, str],
    selected_strategy: str,
    acceptance_gate_status: str,
    acceptance_gate_sufficient: bool,
    protocol_status: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Single canonical JSON answering: is this run replayable under the same policy?"""
    descriptor: dict[str, Any] = {
        "format": "amorph_replay_descriptor_v1",
        "commit_sha": commit_sha,
        "source_tree_sha256": source_tree_sha256,
        "dataset_sha256": dataset_sha256,
        "dataset_revision": dataset_revision,
        "split_hashes": dict(split_hashes),
        "seed": seed,
        "library_versions": dict(library_versions),
        "strategy_fingerprints": dict(strategy_fingerprints),
        "selected_strategy": selected_strategy,
        "acceptance_gate_status": acceptance_gate_status,
        "acceptance_gate_sufficient": acceptance_gate_sufficient,
        "protocol_status": dict(protocol_status),
    }
    if extra:
        descriptor["extra"] = dict(extra)
    descriptor["descriptor_sha256"] = sha256_hex(canonical_json(descriptor))
    return descriptor

import numpy as np
import pytest

from amorph_autotrainer.strategy_abi import (
    ComponentStatus,
    build_replay_descriptor,
    run_conformance_checks,
    strategy_fingerprint,
)
from amorph_autotrainer.strategies import E5Candidate, TfidfCandidate


def test_tfidf_exposes_strategy_abi_surface():
    assert TfidfCandidate.identity() == "tfidf"
    assert TfidfCandidate.version()
    assert TfidfCandidate.status() is ComponentStatus.CANONICAL
    assert "requires_gpu" in TfidfCandidate.capabilities()
    fp = TfidfCandidate.fingerprint(code_revision="abc")
    assert len(fp) == 64
    assert fp == TfidfCandidate.fingerprint(code_revision="abc")
    assert fp != TfidfCandidate.fingerprint(code_revision="def")


def test_e5_status_is_experimental():
    assert E5Candidate.status() is ComponentStatus.EXPERIMENTAL
    assert E5Candidate.capabilities()["optional_extra"] == "embeddings"


def test_conformance_passes_for_well_behaved_predict():
    def predict(texts):
        return np.asarray(["ok"] * len(texts), dtype=object)

    result = run_conformance_checks(predict, strategy_name="dummy", sample_texts=["a", "b"])
    assert result["passed"] is True


def test_conformance_fails_on_length_mismatch():
    def predict(texts):
        return np.asarray(["only-one"], dtype=object)

    with pytest.raises(ValueError, match="conformance failed"):
        run_conformance_checks(predict, strategy_name="bad", sample_texts=["a", "b"])


def test_replay_descriptor_has_stable_hash_field():
    d = build_replay_descriptor(
        commit_sha="deadbeef",
        source_tree_sha256="0" * 64,
        dataset_sha256="1" * 64,
        dataset_revision=None,
        split_hashes={"train": {"rows": 1, "sha256": "a"}},
        seed=42,
        library_versions={"numpy": "1.0"},
        strategy_fingerprints={"tfidf": "b" * 64},
        selected_strategy="tfidf",
        acceptance_gate_status="acceptance_gate_underpowered",
        acceptance_gate_sufficient=False,
        protocol_status={"strategy_abi": "implemented"},
    )
    assert d["format"] == "amorph_replay_descriptor_v1"
    assert len(d["descriptor_sha256"]) == 64


def test_strategy_fingerprint_changes_with_hyperparameters():
    a = strategy_fingerprint(
        identity="tfidf", version="1.0.0", hyperparameters={"C": 1}, code_revision=None
    )
    b = strategy_fingerprint(
        identity="tfidf", version="1.0.0", hyperparameters={"C": 10}, code_revision=None
    )
    assert a != b

import pytest

from experiments.llm_sft_oasst1.gate import paired_group_bootstrap


def _pairs(deltas):
    base, candidate = [], []
    for i, delta in enumerate(deltas):
        base.append({"tree_id": f"tree-{i}", "nll": 2.0})
        candidate.append({"tree_id": f"tree-{i}", "nll": 2.0 + delta})
    return base, candidate


def test_gate_accepts_clearly_noninferior_candidate():
    base, cand = _pairs([-0.10] * 24)
    result = paired_group_bootstrap(base, cand, seed=1, samples=200, margin=0.02, minimum_groups=20)
    assert result["verdict"] == "ACCEPT"
    assert result["ci_upper"] <= 0.02


def test_gate_rejects_candidate_with_clear_nll_regression():
    base, cand = _pairs([0.10] * 24)
    result = paired_group_bootstrap(base, cand, seed=1, samples=200, margin=0.02, minimum_groups=20)
    assert result["verdict"] == "REJECT"
    assert result["ci_lower"] > 0.02


def test_gate_returns_inconclusive_if_interval_overlaps_margin():
    base, cand = _pairs([-0.20] * 12 + [0.20] * 12)
    result = paired_group_bootstrap(base, cand, seed=1, samples=500, margin=0.02, minimum_groups=20)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["ci_lower"] <= 0.02 < result["ci_upper"]


def test_gate_fails_closed_when_underpowered():
    base, cand = _pairs([-0.2] * 8)
    result = paired_group_bootstrap(base, cand, seed=1, samples=200, margin=0.02, minimum_groups=20)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["ci_lower"] is None
    assert result["minimum_required_groups"] == 20


def test_gate_rejects_mismatched_tree_ids():
    base, cand = _pairs([-0.1] * 20)
    cand[-1]["tree_id"] = "different"
    with pytest.raises(ValueError, match="identical tree IDs"):
        paired_group_bootstrap(base, cand, seed=1, samples=200, margin=0.02, minimum_groups=20)


def test_gate_rejects_duplicate_tree_ids():
    base, cand = _pairs([-0.1] * 20)
    base.append(dict(base[0]))
    cand.append(dict(cand[0]))
    with pytest.raises(ValueError, match="Duplicate conversation tree"):
        paired_group_bootstrap(base, cand, seed=1, samples=200, margin=0.02, minimum_groups=20)


def test_gate_rejects_non_finite_nll():
    base, cand = _pairs([-0.1] * 20)
    cand[0]["nll"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        paired_group_bootstrap(base, cand, seed=1, samples=200, margin=0.02, minimum_groups=20)

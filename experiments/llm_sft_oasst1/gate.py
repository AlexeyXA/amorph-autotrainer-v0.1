"""Pure-Python paired bootstrap gate for the OASST1 SFT proof-of-concept."""
from __future__ import annotations

import math
import random
from statistics import mean
from typing import Any


def paired_group_bootstrap(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], *,
    seed: int, samples: int, margin: float, minimum_groups: int,
) -> dict[str, Any]:
    """Compare per-tree response NLL with a paired bootstrap; lower is better."""
    if samples < 100:
        raise ValueError("samples must be >= 100")
    if margin < 0 or minimum_groups < 1:
        raise ValueError("margin must be >= 0 and minimum_groups must be >= 1")
    base_by_tree = {row["tree_id"]: float(row["nll"]) for row in baseline}
    cand_by_tree = {row["tree_id"]: float(row["nll"]) for row in candidate}
    common = sorted(set(base_by_tree) & set(cand_by_tree))
    if len(common) != len(base_by_tree) or len(common) != len(cand_by_tree):
        raise ValueError("Baseline and candidate examples must have identical tree IDs")
    if any(not math.isfinite(base_by_tree[k]) or not math.isfinite(cand_by_tree[k]) for k in common):
        raise ValueError("NLL values must be finite")
    deltas = [cand_by_tree[key] - base_by_tree[key] for key in common]
    result: dict[str, Any] = {
        "metric": "unweighted mean across trees of per-example answer-token mean NLL; lower is better",
        "delta_definition": "candidate minus base; negative favours the adapter",
        "n_independent_conversation_trees": len(common),
        "noninferiority_margin_nats_per_token": margin,
        "bootstrap_samples": samples,
        "confidence": 0.95,
        "delta_mean": mean(deltas) if deltas else None,
        "ci_lower": None,
        "ci_upper": None,
        "verdict": "INCONCLUSIVE",
        "reason": "underpowered_gate",
    }
    if len(common) < minimum_groups:
        result["minimum_required_groups"] = minimum_groups
        return result
    rng = random.Random(seed)
    n = len(deltas)
    boot = [mean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    boot.sort()
    lower = boot[max(0, math.floor(0.025 * (samples - 1)))]
    upper = boot[min(samples - 1, math.ceil(0.975 * (samples - 1)))]
    result.update({"ci_lower": lower, "ci_upper": upper, "reason": "confidence_interval_overlaps_margin"})
    if upper <= margin:
        result.update({"verdict": "ACCEPT", "reason": "noninferiority_established"})
    elif lower > margin:
        result.update({"verdict": "REJECT", "reason": "inferiority_established"})
    return result

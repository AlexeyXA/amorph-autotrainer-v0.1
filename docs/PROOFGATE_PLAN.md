# Variant 3 plan: Proof Gate extraction (AMORPH → thin library)

**Owner decision context:** Prefer evidence/acceptance primitives over “another AutoML”.
**Status:** Plan only (2026-10-23). Not implemented as a separate package yet.

## One-liner

Accept baseline preds + candidate preds + gate labels (+ criteria) →
`ACCEPT` | `REJECT` | `INCONCLUSIVE` (+ honest subtypes) + replayable `report.json`.

**Does not train models. Does not select strategies. Does not replace TRL/PEFT.**

## Already in AMORPH (inventory to extract)

| Piece | Location (approx.) | Extract v0.1? |
|-------|--------------------|---------------|
| Paired bootstrap / non-inferiority | `metrics.py` (`gate_candidate`, bootstrap) | **yes** |
| Underpowered floor | pipeline + config `min_gate_examples_per_class` | **yes** |
| Honest statuses (`baseline_self_comparison`, …) | pipeline | **yes** |
| Conformance (type/length/ndim) | `strategy_abi.py` | **yes** |
| Replay / hashes | `strategy_abi.build_replay_descriptor`, manifest | **yes** |
| MILP group split | `data.py` | **no** (v0.2 optional) |
| TF-IDF / E5 / hybrid train | `strategies.py` | **no** |
| TCO selector | `pipeline.py` / `cost.py` | **no** |

## v0.1 API (freeze before code)

```text
proofgate.evaluate(
  y_gate,
  pred_baseline,
  pred_candidate | None,
  *,
  metric="macro_f1",
  margin=0.02,
  n_bootstrap=...,
  seed=...,
  min_examples_per_class=5,
) -> GateReport
```

### Verdicts

| Verdict | Meaning |
|---------|---------|
| `ACCEPT` | Gate powered; candidate non-inferior (and absolute target if set) |
| `REJECT` | Gate powered; candidate fails criteria |
| `INCONCLUSIVE` | Underpowered **or** conformance fail **or** no real challenger |
| `BASELINE_ONLY` / self-comparison | Optional subtype of INCONCLUSIVE — **never** market as ACCEPT |

Non-inferiority boundary: document **exact** code rule (AMORPH: inclusive `ci_lower >= -margin` if that is still true).

## Phases

### A — Spec (2 days)
- [ ] This RFC stable
- [ ] 5 golden synthetic cases: ACCEPT / REJECT / underpowered / self-comparison / length mismatch
- [ ] Field map: AMORPH manifest ↔ `GateReport`

### B — Module inside repo first (1 week)
- [ ] `amorph_autotrainer/proofgate/` or top-level `src/proofgate/` with **zero** imports from strategies/data train path
- [ ] Move/wrap bootstrap + gate + conformance + report writer
- [ ] `pytest` golden cases
- [ ] CLI: preds + labels in → `report.json`

### C — Single source of truth (3–5 days)
- [ ] AMORPH pipeline calls the same `evaluate()`
- [ ] Full AMORPH pytest green; no dual gate logic

### D — External example (3–5 days)
- [ ] `examples/sklearn_holdout_gate.py` (train outside AMORPH)
- [ ] `docs/PROOFGATE_LIMITATIONS.md`

### E — Later (not now)
- Slice/group degradation, ECE, leakage helpers, optional split extra
- Separate PyPI package name only after C+D stable

## Explicit non-goals (agents must not do these in the same PR)

- New embedders (BGE/GTE) or more hybrids
- Changing E5 production defaults
- Merging draft experiment PR #4
- LLM-as-judge / alignment forgetting harness (Variant 1)
- NCA benchmark suite (Variant 2)

## Success criteria

| Horizon | Criterion |
|---------|-----------|
| 2 weeks | Shared gate module; AMORPH tests green; golden verdicts pass |
| 1 month | External sklearn example + public report artifact |
| Adoption | Optional; honesty > hype |

## Agent assignment (GPT)

**Single next PR only:** Phase A+B start — `docs` already has this plan; implement `proofgate` module + golden tests; wire or dual-call from pipeline without behavior change on synthetic suite.

Report in issue #1 with `[GPT]`, commit SHA, CI URL. If blocked, say **Blocked:** with reason — do not go silent.

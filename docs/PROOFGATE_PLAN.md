# Variant 3 plan: Proof Gate extraction (AMORPH → thin library)

**Owner decision context:** Prefer evidence/acceptance primitives over “another AutoML”.
**Status (2026-10-23):** Phases **A–D implemented on `main`**. Not a separate PyPI package yet; lives as `amorph_autotrainer.proofgate`.

See also: `docs/READINESS.md`, `docs/PROOFGATE_LIMITATIONS.md`, `docs/PROOFGATE_API_V0_1.md`.

## One-liner

Accept baseline preds + candidate preds + gate labels (+ criteria) →
`ACCEPT` | `REJECT` | `INCONCLUSIVE` (+ honest subtypes) + replayable `report.json`.

**Does not train models. Does not select strategies. Does not replace TRL/PEFT.**

## Already in AMORPH (inventory)

| Piece | Location (approx.) | Extract v0.1? |
|-------|--------------------|---------------|
| Paired bootstrap / non-inferiority | `metrics.py` + `proofgate` | **yes — done** |
| Underpowered floor | config + proofgate | **yes — done** |
| Honest statuses | pipeline + proofgate | **yes — done** |
| Conformance (type/length/ndim) | strategy_abi + proofgate | **yes — done** |
| Replay / hashes | replay + proofgate reports | **yes — done** |
| MILP group split | `data.py` | **no** (stays AutoTrainer) |
| TF-IDF / E5 / hybrid train | `strategies.py` | **no** |
| TCO selector | `pipeline.py` / `cost.py` | **no** |

## v0.1 API

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

Non-inferiority boundary: inclusive `ci_lower >= -margin`.

## Phases

### A — Spec — **done**
- [x] RFC / plan
- [x] Golden synthetic cases
- [x] Field map AMORPH ↔ GateReport (`docs/PROOFGATE_API_V0_1.md`)

### B — Module — **done** (PR #10)
- [x] `amorph_autotrainer/proofgate/`
- [x] bootstrap + evaluate + report writer
- [x] pytest golden cases
- [x] CLI: `python -m amorph_autotrainer.proofgate`

### C — Single source of truth — **done** (PR #13 C1, PR #14 C2)
- [x] Dual-write then cut-over
- [x] Pipeline decisions from `proofgate.evaluate`
- [x] Legacy parity audit (shared helpers — wiring check, not independent stats)
- [x] Self-comparison never ACCEPT

### D — External example — **done** (PR #15)
- [x] `examples/sklearn_holdout_gate.py`
- [x] `docs/PROOFGATE_LIMITATIONS.md`

### E — Later (optional, not v0.1 freeze)
- [ ] Slice/group degradation, ECE, leakage helpers
- [ ] Separate PyPI package name
- [ ] Multi-seed promotion protocol for experimental strategies

## Explicit non-goals (still)

- New embedders (BGE/GTE) without owner ask
- Changing E5 production defaults
- Merging draft experiment PR #4 into defaults
- LLM-as-judge / alignment harness

## Remaining for “owner-ready”

1. Owner domain CSV run (`amorph-autotrainer train` or sklearn example) with artifact retention
2. Appropriate `group_id` when rows correlate
3. Read real Proof Gate verdict; do not treat `INCONCLUSIVE` as pass

# Gate design protocol 1+3 (Grok)

## Changes
1. **Eligibility (benchmark):** exclude train classes with &lt; `4 × min_gate` unique texts (default ≥20) or &lt; 4 worker groups.
3. **Gate design (core):** default split fractions train/val/gate/test = **0.45 / 0.15 / 0.25 / 0.15**; MILP requires ≥`min_gate_examples_per_class` rows of each label in the acceptance-gate split.

## Apply on main (if full files not merged yet)
```bash
patch -p1 < patches/0001-data-gate-design.patch
patch -p1 < patches/0002-pipeline-gate-design.patch
patch -p1 < patches/0003-benchmark-gate-design.patch
pytest -q
```

Or copy `amorph_autotrainer/data.py`, `pipeline.py`, `scripts/benchmark_massive_ru.py` from the Grok worktree / request GPT to merge.

## Tests
Local `pytest -q` passed after these changes (17 tests).

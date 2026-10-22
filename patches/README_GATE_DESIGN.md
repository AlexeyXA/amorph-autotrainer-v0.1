# Gate design protocol 1+3 (P0)

## What this does
1. **Eligibility (benchmark):** exclude train classes with < `4 × min_gate` unique texts (default ≥20) or < 4 worker groups.
3. **Gate design (core):** default split fractions train/val/gate/test = **0.45 / 0.15 / 0.25 / 0.15**; MILP requires ≥`min_gate_examples_per_class` **rows** of each label in the acceptance-gate split (not only presence).

Do **not** lower `min_gate_examples_per_class` below 5 to force a green gate.

## Apply on a clean tree from main
```bash
git checkout main && git pull
git checkout -b p0-gate-design-1-3   # or use existing PR branch
patch -p1 < patches/0001-data-gate-design.patch
patch -p1 < patches/0002-pipeline-gate-design.patch
patch -p1 < patches/0003-benchmark-gate-design.patch
pip install -e '.[test]'
pytest -q
```

## Local verification (Grok, 2026-10-22)
- Worktree with full sources applied: 20 pytest tests green.
- Branch `p0-gate-design-1-3` carries updated tests + complete patches.
- Prefer merging **full** `data.py` / `pipeline.py` / `benchmark_massive_ru.py` as source of truth; patches are the auditable delta vs main.

## After merge
Re-run MASSIVE workflow; require `acceptance_gate_sufficient == true` before treating the run as gate-passed.

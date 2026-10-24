# Readiness checklist (honest)

**Not a claim of universal production quality.** For v0.1, readiness means the implemented protocol is testable, documented, and has reproducible public-benchmark evidence. Production suitability still depends on the owner data run.

| # | Criterion | Status (2026-10-23) |
|---|-----------|---------------------|
| 1 | Green CI on `main` | **done** — run [#39923106458](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/39923106458) on `446745e57c05fa3f76c3c513ee10f970cd6d8275` succeeded (pytest, compileall, CLI smoke) |
| 2 | Powered acceptance gate (1+3) in executable split | **done** — hard row floors and fail-closed underpowered status |
| 3 | Strategy ABI + conformance + fingerprints | **done** |
| 4 | Proof Gate standalone API (Phase A+B) | **done** — `amorph_autotrainer.proofgate` |
| 5 | Single gate decision path (Phase C) | **done** — Proof Gate decides; legacy parity reuses shared metric/bootstrap helpers to check wiring, not as an independent statistical implementation |
| 6 | External scikit-learn example (Phase D) | **done** — `examples/sklearn_holdout_gate.py` |
| 7 | Proof Gate limitations and safe-use documentation | **done** — `docs/PROOFGATE_LIMITATIONS.md` |
| 8 | Owner domain CSV run with artifact hashes | **owner action required** — not yet demonstrated on the owner domain |
| 9 | Multi-seed protocol for promoting experimental E5/hybrid | **open** — not required for the v0.1 protocol freeze; required before stronger generalization claims |
| 10 | Task-specific critical-class acceptance constraints | **implemented in draft PR #27** — repeatable per-label recall/precision floors; checked on validation and acceptance gate independently of aggregate Proof Gate; test suite green on draft SHA `58dfee9d3642ae7a0144d238e5be56dc24f94efc` |
| 11 | Opt-in monetary TCO selector | **implemented in draft PR #27; CI recheck in progress** — `selection_objective=tco`; candidate-specific usage/setup costs; zero-compute TCO requires explicit rates; default remains `runtime_proxy`; see [TCO selector contract](TCO_SELECTOR.md) |

## Draft follow-up — public task verification and per-label constraints (2026-10-24)

These are **not merged into `main`**. See Draft PR [#27](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/pull/27).

- Reproducible public tasks: BANKING77, CLINC150 + OOS, and MASSIVE ru-RU. Latest successful public workflow: [#40080734847](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/40080734847).
- Key finding: CLINC150 E5 passed aggregate Macro-F1 Proof Gate but reached only 3.9% OOS recall on the untouched official test. This demonstrates that task-critical constraints must be configured separately; see [Issue #28](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/issues/28).
- Draft implementation adds `--min-recall-by-label LABEL=THRESHOLD` and `--min-precision-by-label LABEL=THRESHOLD`. Candidate eligibility checks them on validation; acceptance checks them separately on the gate. Final `business_requirements_passed` is false unless both validation and gate meet every configured requirement. CLI returns exit code 5 when the final model fails those requirements.
- Documentation: [task-specific acceptance contract](TASK_SPECIFIC_ACCEPTANCE.md). CI succeeded on source SHA `58dfee9d3642ae7a0144d238e5be56dc24f94efc`: [run #40084663031](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/40084663031).
- These controls only evaluate the supplied samples. They do not guarantee future OOS performance or handle arbitrary distribution shift; CLINC150 official test must not be used to tune a threshold or claim confirmatory success.

## Merged milestones

| Stage | Content | Owner | Status / evidence |
|-------|---------|-------|-------------------|
| A+B | Standalone Proof Gate API, three verdicts, deterministic report hashes and golden tests | GPT | **merged** — PR #10, included by subsequent commits |
| C1 | Shadow/dual-write and provenance without changing legacy decisions | GPT | **merged** — [PR #13](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/pull/13), merge SHA `9e5552a5e599b63e893dbd4120cceb963665ca45` |
| C2 | Proof Gate is the single decision source; inconclusive cases cannot be promoted to accepted | GPT | **merged** — [PR #14](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/pull/14), merge SHA `b80d6e22575c3939caaa23dd63b690fc6787edad` |
| D | Standalone scikit-learn example and limitations doc | GPT | **merged** — [PR #15](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/pull/15), merge SHA `446745e57c05fa3f76c3c513ee10f970cd6d8275` |
| Real-data integration check | Predeclared MASSIVE experiment on the C2 commit exercises ACCEPT plus legacy parity | GPT | **done as integration evidence**, details below; not an independent scientific replication |
| Owner domain | Run one real CSV with an appropriate group ID and retain run/report/split hashes | Owner | **required for readiness on the intended use case** |

## C2 real-challenger regression evidence

- Default-policy MASSIVE run [#39922255007](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/39922255007) succeeded on the C2 merge SHA. The default candidate was not validation-eligible; Proof Gate recorded `INCONCLUSIVE / SELF_COMPARISON`, `decision_passed=false`, and no synthetic zero-width confidence interval. Artifact ID `12487328876`; ZIP SHA-256 `ddfd02ecd6c35c904271bc20d157f13dd1b851bca0c375d581915a01f2a2874a`; all 23/23 inventory entries verified.
- Predeclared experiment [#39922255003](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/39922255003) on the same C2 commit selected the E5 C=0.1 / balanced candidate. Gate: 2,731 rows; TF-IDF Macro-F1 `0.435771`; E5 Macro-F1 `0.634771`; paired delta `+0.200598`; 95% CI `[+0.177388, +0.221808]`; 500 bootstrap replicates; Proof Gate `ACCEPT`; compatibility calculation `passed=true`; parity `match` using the same shared metric/bootstrap helpers (wiring evidence, not an independent statistical check). Artifact ID `12488475401`; ZIP SHA-256 `f7134298299b3adcddf192affd3d81d948a9968a45cf6803ca488fa3eb4932ee`; all 28/28 inventory entries verified.
- Dataset revision: `c4aa61e89895160619e18a94de72f19b99e72ca2`. This is an integration/parity regression run on a dataset/split/gate already inspected during previous E5 work; **do not claim a new independent statistical confirmation**. The public official test remains report-only and had been inspected earlier.
- Known benchmark warnings remain: the near-duplicate audit is skipped above 5,000 rows (worker-group split does not rule out near-duplicates across different groups), and zero-valued cost rates mean TCO is based on assumed-free inputs, not measured zero economic cost.

## Requirements before calling the owner's use case ready

1. Run either `amorph-autotrainer train` on the owner CSV with the correct text, label and group-ID columns, or use `examples/sklearn_holdout_gate.py` for the standalone example.
2. Keep a stable and appropriate unit of independence in `group_id` when rows from the same user/source may correlate. A group split does not replace a chronological split when the task is forecasting future data.
3. Inspect the actual Proof Gate verdict. `INCONCLUSIVE` is not an acceptance; `REJECT` means the predeclared criterion failed; `ACCEPT` only passes the specified gate conditions for that fitted model and sample.
4. Preserve the report, manifest, replay descriptor, dataset hash, split hashes and run SHA outside Git if the CSV contains sensitive or confidential data. Do not commit raw private data or credentials.

## Known non-goals for v0.1 freeze

- BGE/GTE productization or widening the strategy zoo.
- Merging experiment-only draft PR #4 into defaults.
- Claiming monetary TCO-optimal selection; current candidate selection uses a runtime proxy.
- Treating one holdout/bootstrap result as proof of generalization or future-distribution robustness.
- Claiming universal production readiness before the owner domain CSV run.

## Phase C rules (post cut-over)

1. Only a real challenger with Proof Gate `ACCEPT` and legacy parity `match` can pass. A parity mismatch fails closed.
2. A baseline self-comparison is `INCONCLUSIVE`; it does not pass the gate and has no artificial zero-width CI.
3. Underpowered and conformance-failed cases never become `ACCEPT` through fallback.
4. The final test remains report-only and never changes selection or rollback.

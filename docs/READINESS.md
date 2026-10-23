# Readiness checklist (honest)

**Not** a claim of universal production quality. “Ready” means the *protocol* is usable and documented.

| # | Criterion | Status (2026-10-23) |
|---|-----------|---------------------|
| 1 | Green CI on `main` | expected after each merge |
| 2 | Powered acceptance gate (1+3) in executable split | **done** |
| 3 | Strategy ABI + conformance + fingerprints | **done** |
| 4 | Proof Gate standalone API (Phase A+B) | **done** (`amorph_autotrainer.proofgate`) |
| 5 | Single gate decision path (Phase C) | **dual-write in this PR**; cut-over still open |
| 6 | External sklearn example (Phase D) | **open** |
| 7 | LIMITATIONS doc for Proof Gate | **open** |
| 8 | Owner domain CSV run with artifact hashes | **owner** |
| 9 | Multi-seed protocol for promoting experimental E5/hybrid | **open** (not required for v0.1 freeze) |

## Known non-goals for v0.1 freeze

- BGE/GTE productization
- Merging experiment-only PR #4 into defaults
- Monetary TCO-optimal selection
- LLM prompt classifiers

## Unified working plan (Grok + GPT)

| Stage | Content | Owner | Status |
|-------|---------|-------|--------|
| C1 shadow/dual-write | `proofgate.evaluate` in pipeline, legacy decides, match/mismatch audit | GPT | **this PR** |
| Roles + readiness docs | `AGENTS.md`, this file | Grok | **this PR** |
| C2 cut-over | decisions from Proof Gate where parity holds; self-comparison stays non-ACCEPT | GPT | next PR |
| D | sklearn example + LIMITATIONS | GPT | after C2 |
| Owner domain | one real CSV + group_id | Owner | parallel when ready |
| Later | TCO mode, near-dup policy, multi-seed | shared backlog | after D |

**Do not** in this PR: E5 default change, PR #4 merge, BGE/GTE, decision cut-over.

## Phase C rules (must hold)

1. Dual-write does **not** change selection/rollback.
2. Self-comparison: Proof Gate `INCONCLUSIVE`; legacy may still “pass” — parity status `intentional_inconclusive_self_comparison`, not a bug.
3. Cut-over only after documented ACCEPT/REJECT parity on real challenger cases.
4. Official test remains report-only.

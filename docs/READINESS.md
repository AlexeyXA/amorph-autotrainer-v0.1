# Readiness checklist (honest)

**Not** a claim of universal production quality. “Ready” means the *protocol* is usable and documented.

| # | Criterion | Status (2026-10-23) |
|---|-----------|---------------------|
| 1 | Green CI on `main` | expected after each merge |
| 2 | Powered acceptance gate (1+3) in executable split | **done** |
| 3 | Strategy ABI + conformance + fingerprints | **done** |
| 4 | Proof Gate standalone API (Phase A+B) | **done** (`amorph_autotrainer.proofgate`) |
| 5 | Single gate decision path (Phase C) | **in progress** (dual-write first) |
| 6 | External sklearn example (Phase D) | **open** |
| 7 | LIMITATIONS doc for Proof Gate | **open** |
| 8 | Owner domain CSV run with artifact hashes | **owner** |
| 9 | Multi-seed protocol for promoting experimental E5/hybrid | **open** (not required for v0.1 freeze) |

## Known non-goals for v0.1 freeze

- BGE/GTE productization
- Merging experiment-only PR #4 into defaults
- Monetary TCO-optimal selection
- LLM prompt classifiers

## Agent ownership

- GPT implements items 5–7 in code
- Grok maintains this checklist and protocol docs
- Owner signs off “ready for my data”

## Phase C implementation note (for GPT)

Safe sequence:
1. **Dual-write:** after existing `gate_candidate` path, call `proofgate.evaluate(...)`; write `proofgate_report.json` + manifest field `proofgate`. Selection still uses `gate_candidate` so tests stay green.
2. **Parity tests:** assert legacy `passed` matches proofgate verdict mapping on synthetic fixtures where applicable (self-comparison must not become ACCEPT).
3. **Cut-over:** switch `candidate_accepted` to proofgate legacy adapter only when parity holds; keep status taxonomy (`baseline_self_comparison`, underpowered, …).

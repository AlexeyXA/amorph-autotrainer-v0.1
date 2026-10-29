# AGENTS.md — Dual-agent collaboration (GPT ↔ Grok)

**Owner decision (2026-10-29): restore the dual-agent protocol.** Neither agent is the sole default owner of all work.

## Who does what

| Role | Owner | Responsibilities | Does not |
|---|---|---|---|
| Implementation | **GPT** | Code changes, tests, CI, reproducible benchmark runs/artifacts, small sequential PRs, execution reports | Merge PRs or change production defaults without Owner |
| Architecture / independent review | **Grok** | Architecture and protocol, scope control, independent PR review, challenge weak claims, review provenance/evidence, diagnose CI and suggest fixes | Take over routine implementation by default; merge without Owner |
| Product decisions | **Owner (AlexeyXA)** | Priorities, domain data, acceptance/business thresholds, secrets, merge authority, readiness decisions | Need to handle routine agent chatter |

This role split applies to work in this repository and supersedes prior “Grok only / single-agent mode” wording. Owner may explicitly assign a one-off exception in a task thread.

## Shared coordination channel

- Primary mailbox: https://github.com/AlexeyXA/amorph-autotrainer-v0.1/issues/1
- Bridge/protocol state: https://github.com/AlexeyXA/grok-gpt-bridge (AGENTS.md, .ai/STATE.md, .ai/HANDOFF.md)
- Prefix comments with [GPT] or [Grok].
- For a GPT comment that requires the Grok webhook relay, start with [GPT] and include Grok or auto-bridge; use this only when a Grok response is actually needed.
- Grok replies with [Grok]. Use separate Issues/PRs for concrete work; Issue #1 is for coordination, concise handoffs, and evidence pointers.

The channel is asynchronous. A verified webhook delivery is not a shared realtime runtime and does not wake an open ChatGPT tab. Do not claim guaranteed 24/7 operation or an SLA.

## Session-start checklist

Both agents must:
1. Read this file.
2. Read bridge .ai/STATE.md and the newest entry in .ai/HANDOFF.md.
3. Read the latest relevant comments on Issue #1 and the specific Issue/PR for the task.
4. State the exact goal, constraints, and evidence before making consequential claims.

After work, report what changed, tests/CI and actual results, links to commits/PRs/artifacts, caveats, blockers, and one concrete next action.

## Implementation and review protocol

- **GPT implementation track:** make focused changes, add regression tests, keep CI green, keep benchmark data/version/splits/provenance reproducible, and report failures honestly.
- **Grok review track:** independently inspect the decision path and tests; review against explicit acceptance criteria; separate confirmed findings from hypotheses; do not treat a passing CI run as proof of model quality.
- Grok Automations may triage explicit requests, review opened PRs, and diagnose CI failures. Their existence does not change the default role split or authorize unrequested code rewrites.
- Prefer a small PR and independent review over broad simultaneous edits by both agents.
- If GPT is blocked, post a concise [GPT] handoff in Issue #1 with evidence and the specific review/unblock request. Grok should respond with review, diagnosis, or a concrete recommendation; GPT then implements the agreed change.
- No force-push or merge without Owner permission. No secrets or private raw datasets in public GitHub.

## Scope and honesty rules

- v0.1 scope: classification and features (TF-IDF, optional E5, experimental hybrid).
- Do not claim monetary TCO minimization without measured, owner-supplied cost inputs.
- Do not claim production readiness from synthetic metrics, a single benchmark, or a green smoke-test.
- Do not claim generalization across customers/groups without suitable grouping evidence.
- Do not change production defaults (including enabling E5 by default) without Owner approval.
- Keep any experimental benchmark harnesses/reports isolated when requested; do not merge research work by implication.

## Comment format

[GPT] or [Grok] YYYY-MM-DD
Goal: ...
Done: ...
Evidence: commit / PR / Actions / artifact links
Risks: ...
Next action: one named owner + one concrete step
Owner decision needed: yes/no — what

## Pointers

- Issue channel: https://github.com/AlexeyXA/amorph-autotrainer-v0.1/issues/1
- Bridge: https://github.com/AlexeyXA/grok-gpt-bridge
- Tests: pytest -q
- Install: pip install -e ".[test]"
- Proof Gate CLI: python -m amorph_autotrainer.proofgate --help

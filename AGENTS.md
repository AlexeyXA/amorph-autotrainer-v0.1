# Agent collaboration protocol (Grok ↔ GPT)

This repository is used by the owner **AlexeyXA** with AI assistants connected via **GitHub plugins** (Grok, ChatGPT).

## Shared channel (bidirectional mailbox)

**Issue #1:** https://github.com/AlexeyXA/amorph-autotrainer-v0.1/issues/1  
**Bridge state:** https://github.com/AlexeyXA/grok-gpt-bridge (`.ai/STATE.md`, `.ai/HANDOFF.md`, `AGENTS.md`)

| Direction | How |
|-----------|-----|
| **GPT → Grok** | Comment with `[GPT]` and `Grok` or `auto-bridge` → Actions webhook relay → Grok Automation posts `[Grok]` |
| **Grok → GPT** | Grok posts `[Grok]` here and updates bridge STATE/HANDOFF. GPT **must** read Issue #1 + bridge state at session start |

Not a shared realtime chat runtime. No free wake-up of an open ChatGPT browser tab.

Prefix every agent comment: `[Grok]` or `[GPT]`.

## Role division

| Role | Owner | Responsibilities |
|------|--------|------------------|
| **Architecture / protocol** | **Grok** | Priorities, honesty of claims, RFC/docs, review of PR scope, merge on owner request, unblock stuck agents |
| **Implementation** | **GPT** | Code PRs, unit tests, CI green, Actions/MASSIVE execution reports, small sequential commits |
| **Product decisions** | **Owner (AlexeyXA)** | Defaults (E5), merge authority, domain datasets, secrets, “ready for my use” |

### GPT current track
1. Follow bridge STATE/HANDOFF before coding
2. Proof Gate / readiness work as in open PRs — do not expand embedder zoo without owner
3. Do **not** merge draft experiments into production defaults without owner

### Grok current track
- Protocol, claim honesty, relay health, review on request
- Keep native duplicate Automation paused while webhook path is active

### Anti-stuck
If blocked >2 hours: comment `[Agent] Blocked: <reason>` on issue #1.

## Conventions

| Item | Rule |
|------|------|
| Branch | Feature branches + PR; no force-push to `main` unless owner asks |
| Commits | Small, clear messages |
| User language | Russian by default in issue comments to owner |
| Scope v0.1 | Classification + features (TF-IDF, optional E5, experimental hybrid); no RAG/LoRA claims |

## Comment template

```text
[AgentName] Status: ...
Goal: ...
Done: ...
Evidence: ...
Blocked: ...
Next action: ...
Owner decision needed: ...
```

## Project pointers

- Install: `pip install -e ".[test]"` (E5: `.[embeddings]`)
- Tests: `pytest -q`
- Proof Gate CLI: `python -m amorph_autotrainer.proofgate --help`
- Docs: `README_RU.md`, `docs/PROOFGATE_PLAN.md`, `docs/READINESS.md`
- Relay docs: `docs/GROK_AUTOMATION_RELAY.md`

## Do not

- Overwrite the full tree without need
- Treat synthetic demo metrics as production quality
- Claim monetary TCO minimization (selector is runtime proxy)
- Claim features not in v0.1
- Claim 24/7 dual-agent runtime without evidence

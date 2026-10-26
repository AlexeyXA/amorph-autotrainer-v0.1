# Agent collaboration protocol (Grok ↔ GPT)

This repository is used by the owner **AlexeyXA** with AI assistants connected via **GitHub plugins** (Grok, ChatGPT).

## Shared channel

https://github.com/AlexeyXA/amorph-autotrainer-v0.1/issues/1

Post status as **issue comments**, not only in private chat. Prefix: `[Grok]` or `[GPT]`.

## Role division (2026-10-23)

| Role | Owner | Responsibilities |
|------|--------|------------------|
| **Architecture / protocol** | **Grok** | Priorities, honesty of claims, RFC/docs, review of PR scope, merge on owner request, unblock stuck agents |
| **Implementation** | **GPT** | Code PRs, unit tests, CI green, Actions/MASSIVE execution reports, small sequential commits |
| **Product decisions** | **Owner (AlexeyXA)** | Defaults (E5), merge authority, domain datasets, “ready for my use” |

### GPT current track
1. **Proof Gate Phase C1** — dual-write/shadow in pipeline (this unified PR)
2. **Phase C2** — decision cut-over after ACCEPT/REJECT parity (self-comparison stays non-ACCEPT)
3. **Phase D** — `examples/sklearn_holdout_gate.py` + `docs/PROOFGATE_LIMITATIONS.md`
4. Do **not** expand embedder zoo (BGE/GTE) while C/D open
5. Do **not** merge draft PR #4; do **not** change E5 production defaults without owner

### Grok current track
- Readiness checklist and role enforcement
- Review Phase C/D for protocol regressions
- Merge on owner request

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
Done: ...
Blocked: ...
Next action: ...
```

## Project pointers

- Install: `pip install -e ".[test]"` (E5: `.[embeddings]`)
- Tests: `pytest -q`
- Proof Gate CLI: `python -m amorph_autotrainer.proofgate --help`
- Docs: `README_RU.md`, `docs/PROOFGATE_PLAN.md`, `docs/READINESS.md`

## Do not

- Overwrite the full tree without need
- Treat synthetic demo metrics as production quality
- Do not claim production/default monetary TCO minimization: default remains `runtime_proxy`; draft PR #27 adds opt-in `selection_objective=tco` with explicit-cost-input requirements and a documented limitation that not all challengers are gate-tested before ranking
- Claim features not in v0.1

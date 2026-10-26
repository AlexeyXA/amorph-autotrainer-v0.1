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
1. **Phase C1, C2 and Phase D are complete and merged** — see `docs/READINESS.md` and PRs #13, #14, #15. Do not repeat these phases unless a concrete regression is found.
2. **P0: owner-domain readiness.** Keep the Russian owner runbook in PR #18 accurate; prepare one reproducible run on the owner's real labelled CSV only when it is available. Never upload private data to GitHub or substitute a public benchmark and call it owner-domain validation.
3. Before the owner-data run, do not claim domain readiness. Record only a safe summary and hashes; preserve raw data and complete artifacts locally when sensitive.
4. **Known product gap:** the current Selector uses a runtime proxy, not monetary TCO minimization. Do not claim cost optimality or implement/change defaults as though it were already delivered. TCO selection remains a distinct follow-up with explicit tests.
5. Do not expand the embedder zoo while the domain-readiness gap is open; do not merge draft PR #4; do not change E5 production defaults without owner approval.

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
- Claim monetary TCO minimization (selector is runtime proxy)
- Claim features not in v0.1

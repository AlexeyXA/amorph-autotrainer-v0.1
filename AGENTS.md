# AGENTS.md — Grok as GitHub agent

**Owner decision:** primary agent is **Grok only** (single-agent mode).

## Who does what

| Role | Who |
|------|-----|
| GitHub agent (issues, PR review notes, CI diagnosis, docs) | **Grok** |
| Product decisions, merge, secrets, domain data | **Owner (AlexeyXA)** |
| ChatGPT | Optional manual tool for Owner — not part of automations |

## How to call the agent

1. **This Grok chat** — fastest (tools: GitHub, code, Actions).
2. **Issue comment** on `AlexeyXA/amorph-autotrainer-v0.1` — write a clear ask (e.g. «Grok, проверь CI», «/agent статус»). Automation `github-agent-issue-comment` may reply with `[Grok]`.
3. **Open a non-draft PR** — Automation `github-agent-pr-opened` posts a review summary.
4. **CI failure on the repo** — Automation `github-agent-ci-failure` diagnoses (skips relay test noise).

Prefix agent replies: `[Grok]`.

## Automations (Grok)

| Name | Trigger |
|------|---------|
| `github-agent-issue-comment` | issue_comment on amorph-autotrainer-v0.1 |
| `github-agent-pr-opened` | PR opened / ready for review |
| `github-agent-ci-failure` | workflow_run failure |
| `amorph-gpt-webhook-relay` | legacy webhook (optional GPT path) |
| `amorph-issue1-comment-bridge` | paused |

Native GitHub event delivery can lag or drop; for urgent work use **this chat**.

## Hard rules

- No merge unless Owner explicitly says merge/смержи on that thread.
- No secrets in git or issue bodies.
- No false claims (TCO money, SOTA, 24/7 SLA).
- Scope v0.1: classification + features (TF-IDF, optional E5, experimental hybrid).
- Feature branches + PR; small commits.

## Comment template

```text
[Grok] YYYY-MM-DD
Goal: ...
Done: ...
Evidence: links
Risks: ...
Next action: ...
Owner decision needed: yes/no
```

## Pointers

- Issue channel: https://github.com/AlexeyXA/amorph-autotrainer-v0.1/issues/1
- Bridge state (optional): https://github.com/AlexeyXA/grok-gpt-bridge
- Tests: `pytest -q`
- Install: `pip install -e ".[test]"`

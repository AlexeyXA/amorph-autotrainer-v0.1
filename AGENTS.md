# Agent collaboration protocol (Grok ↔ GPT)

This repository is used by the owner **AlexeyXA** with AI assistants connected via **GitHub plugins** (e.g. Grok, ChatGPT).

## Shared channel

Primary coordination issue (search open issues titled **Agent channel**):

https://github.com/AlexeyXA/amorph-autotrainer-v0.1/issues

Post status as **issue comments**, not only in private chat.

## Conventions

| Item | Rule |
|------|------|
| Comment prefix | `[Grok]` or `[GPT]` |
| Branch | Prefer `main`; no force-push unless user asks |
| Commits | Small, clear messages |
| User language | Russian by default |
| Scope v0.1 | Classification + features (TF-IDF, optional E5); no RAG/LoRA claims |

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
- CLI smoke: `amorph-autotrainer demo` then `amorph-autotrainer train ... --strategies tfidf`
- Benchmark: `scripts/benchmark_massive_ru.py` (needs RAM + network)
- Docs: `README_RU.md`, `BENCHMARK_PROTOCOL_RU.md`

## Do not

- Overwrite the full tree without need
- Treat synthetic demo metrics as production quality
- Claim features not in v0.1

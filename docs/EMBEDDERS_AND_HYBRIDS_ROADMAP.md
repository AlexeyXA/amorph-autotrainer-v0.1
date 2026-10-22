# Embedders and hybrids roadmap (AMORPH AutoTrainer v0.1+)

Status vocabulary (Strategy ABI): `canonical` | `implemented` | `experimental` | `hypothesis`.

This document is a **design menu**, not a claim that every row is implemented.
Only strategies with ABI + fingerprint + gate evidence may enter production defaults.

## 1. Single strategies

| ID | Family | Representation | Head | RU/multi | Fit | Infer | Status | Notes |
|----|--------|----------------|------|----------|-----|-------|--------|-------|
| `tfidf` | sparse | word + char TF–IDF (FeatureUnion) | LogReg | any | low | very low | **canonical** | Current baseline |
| `e5` / `e5_small` | E5 | `intfloat/multilingual-e5-small` + `query:` prefix | LogReg | multi+RU | medium | medium | **experimental** | Implemented route; default `C=1.0`, `class_weight=None` |
| `e5_base` | E5 | multilingual-e5-base | LogReg | multi+RU | higher | higher | hypothesis | |
| `bge_m3` | BGE | `BAAI/bge-m3` | LogReg | multi+RU | med–high | med–high | **hypothesis** | Primary BGE candidate for RU/multi |
| `bge_small_en` | BGE | `BAAI/bge-small-en-v1.5` | LogReg | EN | lower | lower | hypothesis | EN-only domains |
| `bge_base_en` | BGE | `BAAI/bge-base-en-v1.5` | LogReg | EN | medium | medium | hypothesis | |
| `bge_large_en` | BGE | `BAAI/bge-large-en-v1.5` | LogReg | EN | high | high | hypothesis | Rarely worth default slot |
| `gte_small` | GTE | gte-small family | LogReg | mostly EN | lower | lower | hypothesis | |
| `gte_base` | GTE | gte-base | LogReg | EN | medium | medium | hypothesis | |
| `gte_large` | GTE | gte-large | LogReg | EN | high | high | hypothesis | |
| `gte_multilingual_base` | GTE | GTE-multilingual base (pin exact HF id + revision) | LogReg | multi+RU | med–high | medium | **hypothesis** | Primary GTE rival to E5/BGE-M3 on RU |
| `minilm` | other | multilingual MiniLM | LogReg | multi | low | low | hypothesis | Budget dense |
| `labse` | other | LaBSE | LogReg | multi | medium | medium | hypothesis | Legacy |
| `openai_emb` | API | vendor embeddings | LogReg | multi | API | API $ | hypothesis | Weak offline replay |
| `llm_pool` | LLM | hidden-state pooling | LogReg | depends | high | high | hypothesis | LLM-as-embedder |
| `llm_prompt` | generative | prompt → label | — | depends | — | very high | hypothesis | Different ABI; out of v0.1 gate shape |

### Dense slot comparison (E5 vs BGE vs GTE)

| Criterion | E5-small (now) | BGE-M3 | GTE-multilingual / GTE-base |
|-----------|----------------|--------|-----------------------------|
| AMORPH role | experimental dense | hypothesis dense alt | hypothesis dense alt |
| RU/multi | strong multi | strong multi | multi (GTE-multi) / EN (gte-base) |
| Relative cost | lower than M3 | higher than e5-small | small/base cheaper than large |
| Hybrid partner | `hybrid_tfidf_e5` | `hybrid_tfidf_bge_m3` | `hybrid_tfidf_gte_*` |
| Rule | Prefer **one** dense slot per run + tfidf (+ optional one hybrid) | avoid E5+BGE+GTE all at once |

## 2. Hybrids

| ID | Composition | Fusion | When useful | Infer cost | Priority | Status |
|----|-------------|--------|-------------|------------|----------|--------|
| (internal) | word ⊕ char TF–IDF | FeatureUnion | always | very low | — | **implemented** inside `tfidf` |
| `hybrid_tfidf_e5` | TF–IDF ⊕ E5-small | concat → LogReg | default hybrid | ~sum | **P1** | experimental (not in main yet) |
| `hybrid_tfidf_bge_m3` | TF–IDF ⊕ BGE-M3 | concat → LogReg | if E5 weak on domain | ~sum | **P1** | hypothesis |
| `hybrid_tfidf_gte_multi` | TF–IDF ⊕ GTE-multilingual | concat → LogReg | alt to BGE/E5 | ~sum | **P1** | hypothesis |
| `hybrid_tfidf_gte_base` | TF–IDF ⊕ GTE-base | concat | EN-only | ~sum | P2 | hypothesis |
| `hybrid_e5_bge_m3` | E5 ⊕ BGE-M3 | concat/blend | rarely worth 2× encoder | high | P3 | hypothesis |
| `hybrid_bge_gte` | BGE ⊕ GTE | concat/blend | rarely | high | P3 | hypothesis |
| `stack_tfidf_bge` | TF–IDF + BGE | stacking meta-clf | soft ensemble | 2 models + meta | P2 | hypothesis |
| `stack_tfidf_gte` | TF–IDF + GTE | stacking | same | 2 models + meta | P2 | hypothesis |
| `cascade_tfidf_e5` | TF–IDF then E5 if low conf | cascade | cost-aware | often ~tfidf | **P1** | hypothesis |
| `cascade_tfidf_bge_m3` | TF–IDF then BGE-M3 | cascade | cost-aware | often ~tfidf | **P1** | hypothesis |
| `cascade_tfidf_gte` | TF–IDF then GTE | cascade | cost-aware EN/multi | often ~tfidf | P2 | hypothesis |
| `blend_e5_bge` | E5 + BGE probs | weighted blend | A/B only | 2× dense | P3 | hypothesis |

## 3. Recommended strategy set S

```text
canonical:      tfidf
dense slot:     e5_small                 # current
alt dense:      bge_m3 | gte_multilingual_base   # pick ONE for A/B
hybrid P1:      hybrid_tfidf_e5
hybrid P1 alt:  hybrid_tfidf_bge_m3 | hybrid_tfidf_gte_multi
cost-aware:     cascade_tfidf_e5 | cascade_tfidf_bge_m3
out of v0.1:    llm_prompt, full cartesian product, silent default changes
```

**Do not** enable E5 + BGE-M3 + GTE in one run without a written protocol.

## 4. Fingerprint requirements (any new dense/hybrid)

Each candidate must record:

- `identity` (stable strategy id)
- `model_id` + **pinned `model_revision`**
- pooling / normalize / prefixes
- head hyperparameters (`C`, `class_weight`, …)
- fusion: `concat` | `stack` | `cascade` (+ cascade threshold)
- `commit_sha` / `source_tree_sha256`

## 5. Acceptance rules (unchanged)

1. Conformance gate (ABI) before statistical gate.
2. Validation eligibility; official test remains **report-only**.
3. Selector minimizes **runtime proxy** today, not monetary TCO — document honestly.
4. Do not promote experimental profiles (e.g. MASSIVE E5 C=0.1 balanced) to defaults without multi-seed protocol.
5. PR #4-style experiment branches stay unmerged unless intentionally productized.

## 6. Suggested next engineering (for GPT)

Ordered, small PRs:

1. **Optional:** `hybrid_tfidf_e5` (concat) as third strategy — ABI + fingerprint + tests; no default change.
2. **Optional A/B:** one of `bge_m3` or `gte_multilingual_base` with pinned revision, same linear-probe pattern as E5.
3. **Optional:** `cascade_tfidf_*` only after hybrid evidence; measure fraction routed to dense.
4. **Do not:** implement full embedder zoo in one PR.

## 5. Related evidence on main

- MASSIVE E5 experimental candidate (not default): `docs/MASSIVE_RU_E5_CANDIDATE_2026-10-22.md`
- Config profile: `configs/experimental/massive_ru_e5_c01_balanced.json`
- Agent channel: issue #1

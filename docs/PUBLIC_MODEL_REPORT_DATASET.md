# Public model-report data for AMORPH: GPT, Claude Opus, Gemini and Grok

**Prepared:** 2026-10-24  
**Source dataset revision:** `promptfoo/political-questions@7a4f845d61262c4727911d1ec3abd9a974a64847`  
**License stated by publisher:** CC BY 4.0  
**Dataset card:** https://huggingface.co/datasets/promptfoo/political-questions  
**Evaluation report/methodology:** https://www.promptfoo.dev/blog/grok-4-political-bias/

## What was found

This public Promptfoo evaluation contains 2,500 questions and responses from four model families:
- OpenAI GPT-4.1
- Anthropic Claude Opus 4
- Google Gemini 2.5 Pro
- xAI Grok 4

The source table includes a stable `question_id`, question text, topic axis, model-specific response columns and cross-model bias ratings. Promptfoo reports 10,000 model responses (four models × 2,500 questions) and says its report data is published under CC BY 4.0. This is a third-party evaluation of those model versions—not four separate datasets from four vendors’ own system cards.

Official model/system cards from OpenAI, Anthropic, xAI and Google normally report aggregate metrics/methodology, not a universal row-level CSV with `text,label,group_id`. The Promptfoo dataset is a more practical published alternative because it exposes the actual question-response examples for the four named model versions.

## Normalized files created by the preparation workflow

The GitHub Actions workflow downloads the pinned source, checks the schema/count, records SHA-256 provenance, and publishes the derived CSVs without uploading the original wide source CSV.

### 1. `model_response_source_attribution.csv` — raw derived data

Exactly three columns: `text,label,group_id`.

- `text`: the complete response from one model.
- `label`: one of `gpt-4.1`, `claude-opus-4`, `gemini-2.5-pro`, `grok-4`.
- `group_id`: the corresponding `question_id`; all four responses to the same question share this ID.

Expected size: 10,000 rows over 2,500 groups. This file is preserved as a faithful derived view of the source. It contains a few identical normalized responses with conflicting model labels, so the current AMORPH trainer correctly refuses to train on it without resolving those conflicts.

### 2. `model_response_source_attribution_conflict_free.csv` — recommended AMORPH input

Also exactly three columns: `text,label,group_id`. The preparation script normalizes each response using Unicode NFKC, trim, casefold and whitespace collapse. If the same normalized response occurs with multiple model labels, it removes **the entire question group** rather than deleting an arbitrary answer. This keeps all four model answers together and preserves equal model counts for each retained question.

For the pinned source revision on 2026-10-24, the audit found **3 conflicting normalized response texts affecting 10 response rows across 5 question groups**. Removing those groups retains 2,495 groups and 9,980 response rows (2,495 per model). The provenance JSON records counts and SHA-256 of the conflict-key set, but never publishes the raw conflicting text. The script recalculates these values every run and fails closed if the schema/source count changes.

Train using the conflict-free file and group by question ID:

```bash
amorph-autotrainer train model_response_source_attribution_conflict_free.csv \
  --text-column text --label-column label --group-column group_id \
  --strategies tfidf,e5 --max-candidates 2 \
  --out runs/model-source-attribution
```

### 3. `political_question_topic.csv` — topic classification

Exactly three columns: `text,label,group_id`.

- `text`: the source question.
- `label`: the source-provided topic `axis` (the actual pinned file contains `economic` and `social`).
- `group_id`: the source `question_id`.

Expected size: 2,500 rows. It is useful for a separate grouped topic-classification experiment.

### 4. `dataset_provenance.json`

Contains the pinned source revision and URL, source file SHA-256, original and filtered row/group counts, labels, conflict audit, hashes for every output CSV, license and limitations. The full raw source CSV stays in the runner cache and is not uploaded as a GitHub Actions artifact.

**What these datasets test:** source attribution and topic classification, plus AMORPH's CSV ingestion and group-aware splitting.

**What they do not test:** whether an answer is correct, useful, safe, unbiased or the cheapest acceptable output. Model identity is a source label, not an answer-quality label. The source-attribution task remains a public pipeline benchmark, not an economic evaluation of model performance.

## Additional token-cost profile and scenarios (not proof of savings)

The same workflow also produces three non-sensitive cost-planning tables:

- `llm_tco_workload_profile_promptfoo.csv`: 10,000 joined prompt/response rows with estimated input/output token lengths. Input token estimate is `ceil(question_characters/4) + 100 assumed instruction tokens`; output token estimate is `ceil(response_characters/4)`. This is a rough planning proxy, not the actual tokenization or billed usage from any API.
- `llm_tco_rate_card_2026-10-24.csv`: official tariff snapshot checked 2026-10-24, including GPT-6 Luna/Sol/Astra, Claude Opus 5.5, Grok 4.5 and Gemini 3.8 Flash introductory and post-2026 rates.
- `llm_tco_scenarios_promptfoo.csv`: price scenarios that apply each current price to the historical output-length profile for that provider's 2025-era model. Useful as a rough workload-profile scenario, but not an apples-to-apples candidate comparison because historical model responses have different lengths and current releases may generate different amounts of output/reasoning tokens.
- `llm_tco_common_profile_scenarios_promptfoo.csv`: apples-to-apples token-price scenario using the same estimated average input length (**118.6 tokens**) for every provider and three common output assumptions: pooled historical average (~41.28 tokens), fixed 50 output tokens, and fixed 100 output tokens. Each is calculated at 2,500 / 10,000 / 100,000 / 1,000,000 requests.

For a simple illustration, using the common 118.6 input-token estimate and fixed 50 output tokens per request at 1 million requests, the token-only estimates are about:

| Model / rate window | Estimated API token cost |
|---|---:|
| GPT-6 Luna (standard) | $36.86 |
| Gemini 3.8 Flash (intro price through 2026-12-31) | $276.45 |
| Gemini 3.8 Flash (price from 2027-01-01) | $552.90 |
| Grok 4.5 | $537.20 |
| GPT-6.1 Sol | $737.20 |
| Claude Opus 5.5 | $1,474.40 |
| GPT-6 Astra | $3,686.00 |

Official rate sources: [OpenAI pricing](https://developers.openai.com/api/docs/pricing), [Anthropic Claude Opus 5.5 pricing](https://platform.claude.com/docs/en/models/opus-5-5/overview), [xAI Grok pricing](https://docs.x.ai/developers/pricing), [Google Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing).

**These amounts are not a prediction of owner cost or a model-quality ranking.** They assume the same input/output token counts, use standard token list prices, and exclude tool/search fees, retries, caching effects and storage/write charges, model-specific reasoning-token differences, human review, infrastructure, tax, and the cost of incorrect answers. The historical-profile scenarios and common-profile scenarios must remain separate. Never call one “cheaper and just as good” without a quality gate on the intended task.

The prices in the rate card were verified on 2026-10-24. Gemini 3.8 Flash has an introductory API price through 2026-12-31; the published standard rate doubles for input/output from 2027-01-01. Refresh the rate card before any future business decision.

## How to use with AMORPH

After downloading the normalized CSV artifact from the workflow run:

```bash
amorph-autotrainer train model_response_source_attribution.csv \
  --text-column text \
  --label-column label \
  --group-column group_id \
  --strategies tfidf,e5 \
  --max-candidates 2 \
  --out runs/model-source-attribution
```

Do not split randomly by row without `--group-column group_id`: otherwise different models' answers to the same prompt can appear on both sides of the evaluation and inflate performance. Grouping by question ID reduces that leakage risk but does not detect every semantic duplicate across different prompts.

For the question topic task, replace the CSV with `political_question_topic.csv`.

## Important limitations

1. This evaluation dataset was generated in 2025 and covers GPT-4.1, Claude Opus 4, Gemini 2.5 Pro and Grok 4—not the current 2026 flagship versions. It is historical evidence, not a current model leaderboard.
2. The source-attribution task is not the same as choosing the cheapest model that solves a business problem. It must not be used to claim economic savings or to infer which model is best for a workload.
3. The accompanying political bias scores are cross-model judgements, not independent human gold labels. The two normalized CSVs do not treat those scores as ground truth.
4. The question-topic labels are source-provided axes; they may be coarse and need their own assessment for any production use.
5. Do not use the public dataset instead of a real business dataset. It is a publicly licensed benchmark for testing pipeline mechanics and one specific classification problem. The intended production domain still needs a separate evaluation.

## Citation

Promptfoo Team (2025), *Political Questions Dataset for LLM Bias Evaluation*, Hugging Face, CC BY 4.0. https://huggingface.co/datasets/promptfoo/political-questions

D'Angelo, M. (2025-07-24), *Evaluating political bias in LLMs*, Promptfoo. https://www.promptfoo.dev/blog/grok-4-political-bias/

# Public four-model Promptfoo dataset and TCO workload profile

**Snapshot:** 24 October 2026. **Source:** Promptfoo Political Questions / Political Bias Answers, pinned revision `7a4f845d61262c4727911d1ec3abd9a974a64847`, CC BY 4.0. [Dataset card](https://huggingface.co/datasets/promptfoo/political-questions) · [Published evaluation report](https://www.promptfoo.dev/blog/grok-4-political-bias/).

## Available data

- `model_response_source_attribution.csv`: 10,000 archived answers, 2,500 each from GPT-4.1, Claude Opus 4, Gemini 2.5 Pro and Grok 4. Columns `text,label,group_id`; label is the historical response model and group_id is the source question ID.
- `political_question_topic.csv`: 2,500 questions with the provided `economic/social` axis label. Same three columns.
- `llm_tco_workload_profile_promptfoo.csv`: 10,000 joined query/response rows; includes estimated input/output token counts for cost planning.
- `llm_tco_rate_card_2026-10-24.csv`: official published rates for current model versions from OpenAI, Anthropic, xAI and Google checked on the snapshot date.
- `llm_tco_scenarios_promptfoo.csv`: 28 standard-rate scenarios at Nq=2,500 / 10,000 / 100,000 / 1,000,000.

The preparation workflow is [`.github/workflows/prepare-promptfoo-model-report-dataset.yml`](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/blob/research/real-task-verification-20261024/.github/workflows/prepare-promptfoo-model-report-dataset.yml). Latest successful prepare run: [#40085733823](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/40085733823).

## Grouping rule

In the response-attribution dataset, every prompt produced four archived answers; all four share one `group_id`. **Keep all answers for one question in the same split.** Otherwise, the classifier can see other answers to the same prompt in training and produce an optimistic test result. Here group_id is a question identifier, not a customer/session ID.

## TCO calculations: useful, but not billed usage

The public benchmark has Nq=2,500 unique questions. The 10,000/100,000/1,000,000 query counts in the CSV are volume scenarios, not observed owner traffic.

Token profile uses an explicit approximation: `ceil(character_count / 4)` plus 100 assumed instruction tokens for each input. Exact current-provider token counts are not available from the report data. The output profile is based on the 2025 archived responses from GPT-4.1, Claude Opus 4, Gemini 2.5 Pro and Grok 4; the cost scenarios apply current list prices to that older response-length profile. Current models may generate more/less text and may bill additional reasoning tokens. Therefore, this is a **planning scenario for token API charges**, not a bill, not a production savings claim, and not the full TCO.

Excluded from the figures: cache hits/writes/storage; tools/search/grounding; retries and failed calls; current-model hidden reasoning tokens or output-length changes; batch/priority tiers; hosting, integration, evaluation, human review, data curation, taxes, and the cost of wrong answers. These components must be added once measured/known. The report's labels measure model source or question topic, not factual correctness, usefulness, safety or business value.

## Current official pricing sources

- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing): current short-context Standard rates used for GPT-6 Luna/Sol/Astra. Cached input prices are stored but not applied in this no-cache scenario.
- [Claude Opus 5.5 pricing](https://platform.claude.com/docs/en/models/opus-5-5/overview): $4 input / $20 output per million tokens; cache read $0.20; Batch API advertises 50% off input and output. Batch/cache are not applied in this standard-rate scenario.
- [xAI pricing](https://docs.x.ai/developers/pricing): Grok 4.5 short-context $2 input / $0.30 cached / $6 output per million. Its long-context rates apply once the prompt reaches 200k tokens.
- [Google Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing?hl=en): Gemini 3.8 Flash introductory standard rates $0.75 input / $3.75 output through 2026-12-31, then $1.50 / $7.50 from 2027-01-01.

## What this proves for AMORPH

This is actual labeled public benchmark material with meaningful prompt groups, plus a transparent public price snapshot. It can test group-aware splitting and provide a baseline workload-cost scenario. It does **not** prove the four current models' present-day comparative quality because the archived answers are from earlier versions; it does not measure the owner's Nq; and no current GPT/Grok/Claude/Gemini API calls were made. To calculate invoice-grade TCO, run the same queries against the current model IDs, record billed input/output/reasoning/cached tokens and latency from each provider response, and include evaluation/human/operational costs.

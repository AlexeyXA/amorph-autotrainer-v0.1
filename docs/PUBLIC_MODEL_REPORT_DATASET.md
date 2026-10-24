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

The GitHub Actions workflow `Prepare public four-model report dataset` downloads the pinned source, checks the schema/count, records SHA-256 provenance, and publishes these artifacts:

### 1. `model_response_source_attribution.csv`

Exactly three columns: `text,label,group_id`.

- `text`: the complete response from one model.
- `label`: one of `gpt-4.1`, `claude-opus-4`, `gemini-2.5-pro`, `grok-4`.
- `group_id`: the corresponding `question_id`; all four answers to the same question share this ID.

Expected size: 10,000 rows if all four model responses are present for all 2,500 questions. The script fails closed rather than silently dropping a missing response.

**What this task tests:** a classifier’s ability to identify the originating model family from its response, with all responses to the same prompt kept in one split. It is useful for testing AMORPH's CSV ingestion, group-aware splitting and repeatability across real model outputs.

**What it does not test:** whether an answer is correct, useful, safe, unbiased or the cheapest acceptable output. Model identity is the label, not answer quality.

### 2. `political_question_topic.csv`

Exactly three columns: `text,label,group_id`.

- `text`: the source question.
- `label`: the source-provided topic `axis`.
- `group_id`: the source `question_id`.

Expected size: 2,500 rows. This is a simpler topic-classification dataset derived from the same evaluation source.

### 3. `dataset_provenance.json`

Contains the pinned source revision and URL, the source file SHA-256, row counts, group counts, per-class counts, output file hashes, licence and limitations. The original wide source CSV is used only in the runner’s cache and is not uploaded as an artifact.

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

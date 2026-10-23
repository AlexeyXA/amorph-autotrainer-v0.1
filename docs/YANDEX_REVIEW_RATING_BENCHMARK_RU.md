# Yandex Maps review-rating benchmark (experimental)

## Purpose

This experiment checks AMORPH on Russian-language user-written reviews with user star ratings. It is **not** an intent-classification benchmark: labels are rating buckets (1–2 stars = negative, 3 = neutral, 4–5 = positive; rating 0 is excluded). A rating is a noisy proxy for sentiment, not a human annotation of sentiment.

## Source and license

- Original Yandex repository: https://github.com/yandex/geo-reviews-dataset-2023
- Hugging Face data mirror: https://huggingface.co/datasets/d0rj/geo-reviews-dataset-2023
- The source describes 500,000 Russian reviews about organizations in Russia, published January–July 2023. The repository states MIT license.
- The run resolves and records an immutable Hugging Face dataset revision and SHA-256 of the downloaded Parquet file.

## Leakage policy

- Model input contains only review text; organization name, address, and rubrics are excluded from features.
- An opaque SHA-256 group ID is derived from normalized organization address + name and used only for group-disjoint splitting.
- Normalized exact-text duplicates are removed globally before splitting. If identical normalized text has conflicting rating buckets, all rows for that text are removed.
- The deterministic sample is capped at 4,900 rows, with at most eight reviews per organization.
- The workflow uploads provenance and metric JSON only, not raw reviews, prepared CSV, or model weights.

## Interpretation limits

- This tests rating-bucket prediction on public reviews, not support-intent classification and not the owner's metal-products distribution.
- There is no official independent test split. Internal final-test metrics become observed when viewed; further tuning requires a new untouched holdout or dataset revision.
- The rating-derived target can disagree with the sentiment expressed in the text.
- Monetary TCO is not measured because runtime and labor rates are zero-valued assumptions in this benchmark configuration.
- No production-quality or commercial-readiness claim follows from this single run.

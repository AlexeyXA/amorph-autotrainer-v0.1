# Task-specific acceptance requirements (A1)

## Purpose

Aggregate Macro-F1 and the statistical Proof Gate do not guarantee that a critical class meets its own operating requirement. This opt-in contract adds per-label recall and precision floors. It does not change defaults when no floors are supplied.

## CLI

```bash
amorph-autotrainer train labelled.csv \
  --text-column text --label-column label \
  --strategies tfidf,e5 --max-candidates 2 \
  --min-macro-f1 0.80 \
  --min-recall-by-label oos=0.80 \
  --min-precision-by-label refund=0.95 \
  --out runs
```

Repeat either per-label option for each mandatory label. Values must be finite numbers in [0, 1]. Labels must match the CSV labels after the reader's normal conversion. These example values are illustrative, not global defaults.

## Decision contract

1. The selector requires candidates to meet the aggregate validation threshold, if configured, and all declared per-label floors on the validation split.
2. If no candidate is eligible, AMORPH may retain the best aggregate candidate for diagnosis only; it is not considered eligible for acceptance.
3. The selected candidate and TF-IDF baseline are checked separately against the same criteria on the independent acceptance-gate sample.
4. A candidate cannot be accepted merely because the aggregate Proof Gate returns ACCEPT: it must also pass all configured per-label requirements and have passed validation eligibility.
5. `business_requirements_passed` is `true` only when the final model passes every configured requirement on validation and acceptance-gate samples; `false` means the CLI exits with code 5. When no per-label criteria are configured, it is `null` (unknown/not applicable).
6. The final test split remains report-only and is not used to choose a model, tune thresholds, or trigger rollback.

## Limitations

- These are point estimates on supplied samples, not confidence intervals for individual labels.
- If a required label is absent from an evaluation sample, its recall check fails closed because recall cannot be demonstrated for that sample.
- Passing these checks does not guarantee performance under future distribution shift or compensate for an invalid label taxonomy, incorrect group IDs, or non-representative data.
- Thresholds should be chosen and recorded before examining acceptance-gate/test outcomes. Never tune a threshold against an already inspected official test set.
- The checks complement, rather than replace, statistical uncertainty analysis and the Proof Gate.

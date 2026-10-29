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
5. `gate_passed_for_configured_metric` reports only the raw aggregate Proof Gate verdict for its configured metric. It is not an overall acceptance decision. The nested `proofgate.decision_passed` is a legacy gate projection after sample-power and parity fail-closed overrides; it also is not overall task acceptance. Read `status`, `overall_acceptance_passed`, and `business_requirements_passed` for the full decision.
6. `business_requirements_passed` is `true` only when the final strategy passes every configured requirement on validation and a sufficiently powered acceptance-gate sample. It is `false` when validation already demonstrates a mandatory failure, or when a sufficiently powered gate fails a mandatory requirement. It is `null` when a passing validation result is followed by an underpowered gate, since the overall business contract cannot then be established.
7. CLI outcome matches the report: a demonstrated task-requirement failure exits with code 5; an underpowered gate with no validation-side failure exits with code 4. A baseline kept after rejecting a challenger is not the same as accepting the challenger.
8. If a required label is never predicted, its precision is undefined: the report records `observed: null` and reason `required_label_never_predicted`, rather than presenting `zero_division=0` as a measured zero.
9. The final test split remains report-only and is not used to choose a model, tune thresholds, or trigger rollback.

## Limitations

- These are point estimates on supplied samples, not confidence intervals for individual labels.
- If a required label is absent from an evaluation sample, its recall check fails closed because recall cannot be demonstrated for that sample.
- Precision is undefined when the model predicts no examples of a class; the report uses null and a distinct reason rather than treating it as a measured zero.
- The aggregate Proof Gate can report ACCEPT while the final strategy is rejected by mandatory task-specific criteria. Read aggregate-gate fields and overall status separately.
- Passing these checks does not guarantee performance under future distribution shift or compensate for an invalid label taxonomy, incorrect group IDs, or non-representative data.
- Thresholds should be chosen and recorded before examining acceptance-gate/test outcomes. Never tune a threshold against an already inspected official test set.
- The checks complement, rather than replace, statistical uncertainty analysis and the Proof Gate.

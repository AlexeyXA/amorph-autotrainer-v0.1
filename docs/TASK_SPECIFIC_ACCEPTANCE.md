# Task-specific acceptance requirements (v0.1)

## Why this exists

The statistical Proof Gate compares a challenger with TF-IDF using the configured aggregate Macro-F1 rule. Passing that rule does **not** imply that every critical class, operational segment or business risk meets its own requirement.

In the public CLINC150 run on 2026-10-24, E5 received aggregate Proof Gate `ACCEPT`, while its recall for the official OOS class on the untouched official test was only 3.9%. That official test is already observed and must not be used to tune a threshold or confirm a fix. See [the benchmark report](PUBLIC_TASK_BENCHMARK_REPORT_2026-10-24.md) and [Issue #28](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/issues/28).

## CLI

You can declare per-label floors explicitly:

```bash
amorph-autotrainer train labelled.csv \
  --text-column text --label-column label \
  --strategies tfidf,e5 --max-candidates 2 \
  --min-macro-f1 0.80 \
  --min-recall-by-label oos=0.80 \
  --min-precision-by-label refund=0.95 \
  --out runs
```

Each `--min-recall-by-label` and `--min-precision-by-label` option may be repeated for multiple labels. Values must be finite numbers in `[0, 1]`; label names must match the CSV label values exactly after the CSV reader's normal conversion. The example thresholds are illustrative only, not AMORPH defaults or a recommendation for every task.

## Decision contract

1. The selector first checks aggregate validation eligibility and every declared per-label requirement on the validation split. Candidates that fail a requirement are not eligible.
2. If no candidate meets all declared criteria, the best Macro-F1 candidate is retained for diagnosis only; it is not marked validation-eligible and cannot be accepted as a challenger.
3. The pre-gate selected candidate is evaluated against the same declared per-label requirements on the independent acceptance-gate split. An aggregate Proof Gate `ACCEPT` cannot override a task-specific failure.
4. The final model's `business_requirements_passed` is true only when every configured requirement passes on both validation and acceptance-gate samples. False means the CLI exits with code 5. When no per-label requirements are configured, the value is null/unknown rather than true.
5. The official test remains report-only and does not affect selection, fallback or gate decisions.

The report records `task_requirements.criteria`, `validation_by_candidate`, `selected_candidate_acceptance_gate`, `baseline_acceptance_gate`, `final_model_validation`, `final_model_acceptance_gate` and `business_requirements_passed` separately from the statistical `proofgate.decision_passed`.

## What this still does not prove

- Thresholds are point-estimate constraints on the supplied samples; this version does not calculate a separate confidence interval for each per-label precision/recall.
- A class absent from a validation or gate sample fails closed for a configured recall requirement. That is not the same as estimating its future recall.
- The same thresholds cannot protect against unrepresented distribution shift, a bad label taxonomy, missing source groups, or semantic near-duplicates that the audit does not find.
- Passing validation and gate constraints does not establish business suitability unless thresholds and data are representative and were declared before evaluation.
- Task-specific constraints are quality gates, not a cost model. This PR branch has an opt-in `--selection-objective tco`, which estimates monetary cost from measured fit/latency and declared rates; it is not merged into `main` and does not prove savings. TCO mode must fail closed when candidate-specific usage/fixed costs are unknown; use explicit `0` only when a component is known to be free. Shared costs are reported but do not affect the ranking.

For OOS, fraud, safety, rare incidents or other critical labels, define the operating requirement and collect representative validation/gate examples before making an acceptance claim. If the distribution is expected to shift, a separate independent fresh holdout is needed for confirmatory evidence.

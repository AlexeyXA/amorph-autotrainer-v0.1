# Proof Gate v0.1 API and AMORPH manifest mapping

Status: in-repository Phase A+B implementation. This is a gate over already-computed labels and predictions, not a model trainer or strategy selector.

## API

~~~python
from amorph_autotrainer.proofgate import evaluate, write_report

report = evaluate(
    y_gate,
    pred_baseline,
    pred_candidate,  # pass None when no challenger exists
    metric="macro_f1",
    margin=0.02,
    n_bootstrap=500,
    seed=42,
    min_examples_per_class=5,
    min_macro_f1=None,
)
write_report(report, "report.json")
~~~

The first release supports Macro-F1 only because the paired confidence interval currently implemented in AMORPH is for candidate-minus-baseline Macro-F1. A different metric cannot silently reuse that interval.

## Verdict semantics

- ACCEPT: powered sample, valid paired inputs, a real challenger, the configured absolute target (if any) is met, and the paired-bootstrap lower confidence bound satisfies the inclusive rule ci_lower >= -margin.
- REJECT: powered sample and real challenger, but a configured criterion fails.
- INCONCLUSIVE: underpowered support, invalid shapes/lengths/labels, no challenger, or identical baseline/candidate prediction arrays. It is never mapped to a successful acceptance.
- Subtypes include UNDERPOWERED, CONFORMANCE_FAILED, BASELINE_ONLY, and SELF_COMPARISON.

Underpowered gates and self-comparisons do not get synthetic zero-width confidence intervals. For conformance defects, the evaluator returns a report with errors instead of raising, preserving evidence that the gate did not run.

The bootstrap interval is conditional on the fitted models' supplied predictions; it does not estimate variation caused by retraining.

## Replay and report format

The report has schema version proofgate.report.v1, criterion values, support counts, baseline/candidate aggregate metrics, comparison interval when calculated, conformance result, metadata, SHA-256 hashes for each input array and the full input payload, plus a SHA-256 digest of the report body. The report intentionally stores input hashes rather than raw label/prediction values; callers retain the input JSON if a future replay is required.

CLI:

~~~bash
proofgate --input gate-input.json --output report.json
# alternatively
python -m amorph_autotrainer.proofgate --input gate-input.json --output report.json
~~~

Input JSON shape:

~~~json
{
  "y_gate": ["a", "a", "b", "b"],
  "pred_baseline": ["a", "b", "b", "b"],
  "pred_candidate": ["a", "a", "b", "b"],
  "criteria": {
    "metric": "macro_f1",
    "margin": 0.02,
    "n_bootstrap": 500,
    "seed": 42,
    "min_examples_per_class": 5,
    "min_macro_f1": null
  },
  "metadata": {
    "dataset_revision": "pinned-version",
    "gate_split_sha256": "..."
  }
}
~~~

Do not use that four-row illustrative sample with the default five-examples-per-class floor as evidence of a powered result; the example only shows the JSON schema.

## AMORPH manifest field map (Phase C integration target)

| AMORPH field | Proof Gate v0.1 field |
|---|---|
| acceptance_gate_metrics.baseline | baseline_metrics |
| acceptance_gate_metrics.candidate | candidate_metrics |
| selected_candidate_gate.comparison / final_acceptance_gate.comparison | comparison |
| selected_candidate_gate.passed / final_acceptance_gate.passed | verdict == ACCEPT only |
| acceptance_gate_sufficient and observed_examples_per_class | sample.powered, sample.class_counts |
| warnings / candidate_errors for conformance | conformance, subtype, reason_codes |
| replay.json input identity | input_fingerprints.payload_sha256 |
| report.json and manifests | report_sha256 |

In particular, a legacy AMORPH self-comparison can currently have a zero-delta interval; Proof Gate intentionally treats identical prediction arrays or a missing candidate as INCONCLUSIVE instead. Full pipeline migration is a separate Phase C task so this PR does not create parallel decision logic inside the training pipeline.

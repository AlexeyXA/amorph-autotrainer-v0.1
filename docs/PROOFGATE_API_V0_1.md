# Proof Gate v0.1 API and AMORPH manifest mapping

Status: Phase A+B standalone API and Phase C decision cut-over are implemented. The live pipeline uses Proof Gate verdicts for acceptance; the legacy calculation is retained only as an independent parity auditor and can only veto an ACCEPT on mismatch. Proof Gate evaluates precomputed labels and predictions; it is not a model trainer or strategy selector.

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

The observed labels in `y_gate` define the class vocabulary for v0.1. If baseline or candidate predictions contain a label not present in `y_gate`, the verdict is `INCONCLUSIVE` (`CONFORMANCE_FAILED`). The API cannot infer from predictions alone whether that label is out-of-taxonomy or a class missing from the supplied gate. This fail-closed rule also keeps point metrics and the paired bootstrap from silently using different class sets.

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

## AMORPH manifest and replay mapping (Phase C cut-over)

| AMORPH field / artifact | Meaning after cut-over |
|---|---|
| proofgate/selected_candidate.json | Decision-source GateReport for selected candidate versus TF-IDF |
| proofgate/baseline_reference.json | Explicit baseline self-comparison; always INCONCLUSIVE, with no synthetic CI |
| proofgate.selected_candidate_report_sha256 | Digest of the selected-candidate report; linked from report and replay.json |
| proofgate.legacy_parity_status | `match`, `mismatch`, or explicit non-comparable / intentional-inconclusive reason |
| selected_candidate_gate / final_acceptance_gate | Compatibility dictionaries adapted from Proof Gate reports; INCONCLUSIVE has `passed=false` |
| proofgate.decision_source / cutover_complete | `proofgate.evaluate` / `true` |
| selected artifact directory | Includes `replay.json`, `data_report.json` and referenced `proofgate/` reports |

The pipeline accepts a non-baseline candidate only when Proof Gate returns ACCEPT and the independent legacy parity audit agrees. A parity mismatch suppresses acceptance and is recorded as `proofgate_legacy_parity_mismatch`; the legacy calculation cannot promote REJECT or INCONCLUSIVE. A TF-IDF self-comparison is INCONCLUSIVE, has no synthetic confidence interval, and does not claim gate acceptance. Final-test data remains outside selection and gate decisions.

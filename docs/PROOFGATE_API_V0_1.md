# Proof Gate v0.1 API and AMORPH manifest mapping

Status: Phase A+B standalone API is implemented. Phase C is in dual-write/shadow stage: the pipeline emits Proof Gate reports alongside the legacy gate, but the legacy gate remains the active decision source until a separate parity-backed cut-over PR. Proof Gate itself only evaluates precomputed labels and predictions; it is not a model trainer or strategy selector.

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

## AMORPH manifest and replay mapping (Phase C dual-write)

| AMORPH field / artifact | Meaning during dual-write |
|---|---|
| proofgate/selected_candidate.json | Shadow GateReport for the selected candidate versus TF-IDF |
| proofgate/baseline_reference.json | Shadow report that deliberately classifies the baseline self-comparison as INCONCLUSIVE |
| proofgate.selected_candidate_report_sha256 | Digest of the selected-candidate report; also linked from replay.json |
| proofgate.shadow_parity_status | `match`, `mismatch`, or an explicit non-comparable reason |
| selected_candidate_gate / final_acceptance_gate | Legacy `gate_candidate` results; still drive the decision in this stage |
| proofgate.decision_source / cutover_complete | Explicitly records `legacy_gate_candidate` / `false` until a separate cut-over PR |
| selected artifact directory | Includes `replay.json`, `data_report.json` and referenced `proofgate/` reports |

During dual-write, a Proof Gate self-comparison or missing challenger is INCONCLUSIVE and has no synthetic confidence interval. The legacy path may still emit a tautological zero-width interval and remains decision-active temporarily; the report marks this transitional state explicitly. The separate cut-over PR must switch decisions to Proof Gate, retain the status taxonomy, and must not merge until applicable parity tests and CI pass.

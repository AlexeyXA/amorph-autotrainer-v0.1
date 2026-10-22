# MASSIVE ru-RU: experimental E5 candidate record

**Status:** strong dataset-specific candidate; not a universal default.  
**Production default:** unchanged on main — E5 Logistic Regression C=1.0, class_weight=None.

## Profile

- Dataset: AmazonScience/massive, locale ru-RU
- Pinned dataset revision: c4aa61e89895160619e18a94de72f19b99e72ca2
- Strategy: intfloat/multilingual-e5-small + LogisticRegression
- Pinned model revision: 614241f622f53c4eeff9890bdc4f31cfecc418b3
- Variant: C=0.1, class_weight=balanced, max_iter=1000
- Reproduction command:

~~~bash
python scripts/benchmark_massive_ru.py \
  --with-e5 \
  --output-dir benchmark_runs/massive_ru_e5_c01_balanced \
  --max-runtime-seconds 7200 \
  --e5-classifier-c 0.1 \
  --e5-class-weight balanced
~~~

The JSON profile at configs/experimental/massive_ru_e5_c01_balanced.json is a provenance record/reference profile; it is not auto-loaded by the CLI.

## Evidence

- Control run: [#11](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/39743323483), E5 C=1.0, class_weight=None.
- Variant run: [#12](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/39743872366), success.
- Variant branch head: bd1cb64595e8bf90f666316809adac00908bc914.
- Workflow execution manifest source commit: a8b0e8a21f11a7998585355483d7260dbd90a0c9 (GitHub pull-request merge ref used by this run).
- Production base commit at experiment time: 4bbd3d3899a4dacf7a78eb47b9a940b89519af55.
- Package source-tree SHA-256: 924aa8e25847eaacabb6674ae96d41b81ae4825f7e60d6b23dce2ee367154f96.
- Dataset revision and prepared CSV hashes are the same in control and variant. Prepared train CSV: 11,387 rows, SHA-256 1ead2c4667897fe1afda03b241814e8765eb256524664f3384eb3fc3b15323f0; official-test CSV: 2,974 rows, SHA-256 aaa90bd680ce470167e75b58520c1def0d8f8353deb86ae368a871653e27bdd4.
- Internal split row counts: train 4,917; validation 1,639; acceptance gate 2,731; internal test 1,639.
- Variant artifact: amorph-massive-ru-39743872366-1, ID 12410736231; SHA-256 99f5ce649a52c64ba1586325dea229dab04f595777e00b3a5102d665a3633eac; all 18 inventory hashes matched.

## Results

### Validation (selection-only)

| Metric | TF-IDF | E5 C=1.0 / None | E5 C=0.1 / balanced |
|---|---:|---:|---:|
| Macro-F1 | 0.450398 | 0.274119 | **0.590038** |
| Accuracy | 0.545455 | 0.406955 | 0.640635 |
| Balanced accuracy | 0.468798 | 0.327378 | 0.730751 |

The tuned E5 variant passed the validation eligibility threshold; the control E5 configuration did not.

### Separate acceptance gate

- Tuned E5 Macro-F1: 0.634771
- TF-IDF baseline Macro-F1: 0.433699
- Paired Macro-F1 delta: +0.202630
- 95% bootstrap interval: [+0.179311, +0.223791], 500 resamples
- Status: gate passed; final pipeline status accepted_no_absolute_target because no absolute minimum Macro-F1 target was configured.

### Official test — report-only, not used to select or accept the model

| Metric | TF-IDF | E5 C=1.0 / None | E5 C=0.1 / balanced |
|---|---:|---:|---:|
| Accuracy | 0.621722 | 0.531944 | **0.660054** |
| Macro-F1 (test-label universe) | 0.447810 | 0.285151 | **0.592888** |
| Balanced accuracy | 0.432933 | 0.301088 | **0.685975** |
| Macro-F1 (fixed 56 training-label universe) | 0.471800 | 0.300427 | **0.624650** |
| Balanced accuracy (fixed 56-label universe) | 0.456126 | 0.317218 | **0.722723** |

Three official-test labels (29, 41, 7) have no training support after the declared benchmark filters; there are 56 training-supported labels among 59 observed official-test labels (94.92%). The new per-class report will keep these unsupported classes visible with training support 0 instead of silently omitting them.

## Interpretation and limitations

1. This is strong evidence for this pinned MASSIVE ru-RU task and split, not proof that the variant is universally best for AMORPH.
2. The experiment changes both C and class_weight simultaneously. It does not isolate their individual causal effects.
3. The comparison is one predeclared split/seed and bootstrap resamples fitted-model predictions; it does not estimate retraining variance.
4. Near-duplicate audit is skipped above 5,000 rows. Worker-ID grouping does not prove there are no near-duplicates across different workers.
5. The official test is report-only and has already been inspected for this profile; do not use it to tune more variants and then claim the same test is an untouched final evaluation.
6. The selector still minimizes its runtime proxy, not monetary TCO.
7. No E5 default or generic task policy is changed by this evidence record.

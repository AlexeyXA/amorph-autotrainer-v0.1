# AMORPH AutoTrainer — public real-task verification

**Date:** 2026-10-24  
**Successful workflow:** [GitHub Actions #40080257083](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/40080257083)  
**Code revision used for the completed benchmark run:** `0672f785fc7c9eccf76a1b953510ea73518fd737`  
**Dataset/model outputs:** artifacts linked from the workflow run above; no raw CSVs were uploaded.

## Scope and result in one paragraph

Three public labelled classification tasks were run through AMORPH's current TF-IDF / E5 pipeline on CPU using fixed seeds and pinned sources. BANKING77 and MASSIVE ru-RU selected TF-IDF; their Proof Gate is honestly `INCONCLUSIVE / SELF_COMPARISON`, because there was no validation-eligible challenger. CLINC150 selected E5 and passed the configured Proof Gate with `ACCEPT` and legacy parity `match`. These are public benchmark results, **not owner-domain proof or proof of economic savings**. Cost rates in this run were zero/assumed-free and the selector still uses runtime proxy, not monetary TCO.

## Results

Validation numbers are the AMORPH internal validation splits created only from each official training partition. Official test numbers are report-only and are computed after selection and Proof Gate.

| Dataset | Raw official train / test | Internal validation Macro-F1 (TF-IDF / E5) | Selected | Proof Gate | Official test (TF-IDF) | Official test (E5) |
|---|---:|---:|---|---|---|---|
| BANKING77 | 10,003 / 3,080; 77 classes | 0.8630 / 0.7809 | TF-IDF | INCONCLUSIVE / SELF_COMPARISON | Acc 0.8620; Macro-F1 0.8607 | Acc 0.7903; Macro-F1 0.7660 |
| CLINC150 + OOS class | 15,100 / 5,500; 151 classes | 0.9072 / 0.9307 | E5 | **ACCEPT**, parity match | Acc 0.7456; Macro-F1 0.8142 | Acc 0.7542; Macro-F1 0.8244 |
| MASSIVE ru-RU | 11,387 prepared train / 2,974 test; 56 train labels, 59 test labels | 0.4517 / 0.2739 | TF-IDF | INCONCLUSIVE / SELF_COMPARISON | Acc 0.6214; fixed-train-label Macro-F1 0.4716 | Acc 0.5313; fixed-train-label Macro-F1 0.3000 |

### CLINC150 gate details

The gate compared E5 with TF-IDF on 3,775 internal gate rows (25 examples per each of 151 classes). Paired Macro-F1 delta was **+0.01097**, 95% bootstrap CI **[+0.00194, +0.02066]**, 500 replicates; the configured non-inferiority margin was 0.01. The rule passed, and legacy parity matched.

Interpretation is intentionally narrow: there was no absolute `min_macro_f1` target in this run; acceptance means the predeclared relative gate conditions passed on that gate sample, not that the model is fit for any business use. The official test contains 1,000 `oos` examples, while official training contains 100. `oos` is treated as a known 151st class here, so this benchmark does not prove open-world OOD detection.

### BANKING77 interpretation

The source files were downloaded from the pinned PolyAI-LDN repository revision and verified by SHA-256. E5 was worse than TF-IDF on validation and on official test. Since the chosen candidate is the TF-IDF baseline, Gate correctly returns `INCONCLUSIVE / SELF_COMPARISON`; no challenger acceptance is claimed.

### MASSIVE ru-RU interpretation

Pinned dataset revision: `c4aa61e89895160619e18a94de72f19b99e72ca2`, locale `ru-RU`, CC BY 4.0. Source train and test parquet hashes and prepared CSV hashes are recorded below.

The prepared training set has 11,387 rows after required-field filtering, removal of 72 rows belonging to conflicting normalized-text groups, and removal of 55 rows from low-support labels; 56 labels remain in training. The official test has 2,974 rows across 59 labels, including three labels without training support. This limits interpretation of fixed-training-universe scores. Near-duplicate audit was skipped above AMORPH's configured cap; worker grouping for internal splits does not prove there are no cross-worker near-duplicates. Gate is inconclusive, not accepted.

## Data sources, licence and hashes

### BANKING77
- Source revision: `PolyAI-LDN/task-specific-datasets@9d081458ff52e53cf7e848f414e6e9344e4e6696`
- Licence: CC BY 4.0. Dataset card / source: https://huggingface.co/datasets/PolyAI/banking77
- Original train SHA-256: `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`
- Original test SHA-256: `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`
- Prepared train SHA-256: `70b6a942fc4bbb0d33b95505103f07d37930fc15bc31720f1cf2e141bb3fae4a`
- Prepared official test SHA-256: `537842ab25f621ca3e1dc6c065b1c59769acdc1b937ddae7b5fff19e7677c51f`

### CLINC150
- UCI dataset card: https://archive.ics.uci.edu/dataset/570/clinc150
- Pinned source mirror revision: `clinc/oos-eval@828f8093932c8fe6ca7936c3d2e52903b1c523de`
- Licence stated on the UCI card: CC BY 4.0
- Source `data_full.json` SHA-256: `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`
- Prepared train SHA-256: `e6882599e1f258267bf2059686c28af253800ddf6d1a19c2ae9930947a3a9d6d`
- Prepared official test SHA-256: `9596c6a420a4cc887dbb55c45483bc4528d301e41836cb51c98e1771b82b3095`
- UCI CDN returned an expired TLS certificate during initial download attempts. The final successful run fetches a pinned upstream raw file and rejects any SHA-256 mismatch; TLS verification is not disabled.

### MASSIVE ru-RU
- Dataset: https://huggingface.co/datasets/AmazonScience/massive
- Revision: `c4aa61e89895160619e18a94de72f19b99e72ca2`
- Licence: CC BY 4.0
- Source train parquet SHA-256: `89ecdc8ddfd00b38d05624c8061dbd10ea7b63f4587ee82750e0e48ebdfb91eb`
- Source test parquet SHA-256: `d53f1c58d753e64b7cf7f10face6c4a57ac93e7f9db82826b43058eacaf2e46a`
- Prepared train SHA-256: `1ead2c4667897fe1afda03b241814e8765eb256524664f3384eb3fc3b15323f0`
- Prepared official test SHA-256: `aaa90bd680ce470167e75b58520c1def0d8f8353deb86ae368a871653e27bdd4`

## Limitations that matter for the product

1. No owner/private production CSV was available; these results demonstrate operation on public benchmarks only.
2. BANKING77 and CLINC150 contain no reliable user/customer/session group ID; internal splits are row-wise and cannot claim group-disjoint generalization.
3. The completed workflow run above predates the newest supplemental exact train/test overlap audit. The updated runner adds Unicode-normalized overlap counts/hashes and a diagnostic for rows that do not overlap; that newer full matrix run is tracked separately in Actions.
4. The current product tests classification strategies (TF-IDF and E5), **not GPT/Grok/Gemini APIs**. No hosted LLM API was called and no API key or paid inference was used.
5. The Gate currently evaluates one configured classification metric and does not by itself measure business-weighted error costs, safety, hallucinations, regression on old generative capabilities, or future distribution shift.
6. The Selector still chooses by runtime proxy. With zero monetary rates, any TCO output is an assumed-free scenario, not a claim of zero cost or minimum money spend.
7. No production defaults were changed. PR #27 remains Draft and must be reviewed before merge.

## Next steps

1. Finish the supplemental per-class and train/test overlap run; use the complete official-test metrics and the non-overlap diagnostic together.
2. Preserve this benchmark evidence as reproducibility/regression coverage, not as a substitute for a business-specific task.
3. Run one authorised owner-domain CSV before claiming production suitability.
4. Only then implement/validate `selection_objective=tco`, with explicit rates/horizon/query volume and tests showing the lowest-cost candidate passes all quality gates.


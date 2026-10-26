# AMORPH AutoTrainer — public real-task verification

**Baseline audit date:** 2026-10-24  
**Latest successful public benchmark + provenance regression run (2026-10-26):** [GitHub Actions #40279920378](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/40279920378)  
**Latest code revision:** `cb8c8eb7e626771d2c32f1ff09572e51f8d30a12`  
**Historical exact-text-overlap baseline:** workflow [#40080734847](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/40080734847), source SHA `8206499af4b4a76a28145559bd6aff2b71f72778`.  
**Latest artifacts (report/provenance/manifests + candidate models; no raw CSVs uploaded):** BANKING77 [artifact 12631747967](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/40279920378/artifacts/12631747967), digest `sha256:48c41dd52139e602511515c2acfa57d9caed5edea2f866add9b52f9ed3246f51`; CLINC150 [artifact 12631977679](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/40279920378/artifacts/12631977679), digest `sha256:d89b4872f8241efd791578676d4c97e15f5702a6dfe34248fcb508865abf060b`; MASSIVE ru-RU [artifact 12631832811](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/40279920378/artifacts/12631832811), digest `sha256:2b4bacc019e82e6c639f84c0d66b772bca745cd22a8f08624279eec1995962d0`.  
**Latest compact result table:** [`public_task_benchmark_results_2026-10-26.csv`](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/blob/research/real-task-verification-20261024/data/public_task_benchmark_results_2026-10-26.csv). No raw CSVs were uploaded.

## Scope and result in one paragraph

The latest full public run is workflow #40279920378 on source SHA `cb8c8eb7e626771d2c32f1ff09572e51f8d30a12`; all three matrix jobs completed successfully and each produced an artifact. BANKING77 and MASSIVE ru-RU selected the TF-IDF baseline, with Proof Gate `INCONCLUSIVE / SELF_COMPARISON`. In CLINC150, E5's aggregate Proof Gate still returns `ACCEPT`, but the explicit diagnostic constraint `min_recall_by_label[oos]=0.80` fails on validation (recall 0.20, support 15) and the gate (recall 0.20, support 25). Therefore the final model is TF-IDF as a diagnostic fallback, the selector has no validation-eligible candidate, and the internal run status is `task_requirements_not_met` with `business_requirements_passed=false`. This is the intended distinction between aggregate statistical acceptance and a mandatory task requirement. The CLINC official-test OOS recall remains 10.2% for TF-IDF and 3.9% for E5; that test was already inspected and is report-only, so this is regression/wiring evidence, **not fresh confirmatory evidence**. These public runs do not prove owner-domain suitability or money savings; the selector remains `runtime_proxy`, not monetary TCO.

## Latest provenance regression run — 2026-10-26

The workflow above ran exactly on the current Draft PR #27 head and completed successfully. The `execution_manifest.json` files were checked inside all three downloaded artifact ZIPs, and their GitHub-reported archive digests were independently matched against local SHA-256 hashes.

| Dataset | Artifact | Internal status | Proof Gate / task contract | Business requirement | Runtime |
|---|---|---|---|---|---:|
| BANKING77 | [12631747967](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/40279920378/artifacts/12631747967) | `baseline_self_comparison` | `INCONCLUSIVE / SELF_COMPARISON`; only TF-IDF validation-eligible | Not configured (`null`) | 95 s |
| CLINC150 | [12631977679](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/40279920378/artifacts/12631977679) | `task_requirements_not_met` | Pre-gate E5 `ACCEPT`; OOS recall floor fails on validation and gate; final TF-IDF diagnostic fallback | `false` | 168 s |
| MASSIVE ru-RU | [12631832811](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/40279920378/artifacts/12631832811) | `baseline_self_comparison` | `INCONCLUSIVE / SELF_COMPARISON`; only TF-IDF validation-eligible | Not configured (`null`) | 89 s |

For MASSIVE ru-RU, this run specifically verifies the workflow manifest fix: the result JSON stores the pipeline object under `internal_amorph_run`, whereas the generic BANKING77/CLINC runner uses `internal_run`. The previous workflow extracted status only from `internal_run`, silently losing the MASSIVE Proof Gate and internal status. The current manifest correctly records `proofgate_verdict=INCONCLUSIVE`, `internal_run_status=baseline_self_comparison` and `business_requirements_passed=null`. Unknown/not-configured criteria remain null rather than becoming success.

Archive SHA-256 values: BANKING77 `48c41dd52139e602511515c2acfa57d9caed5edea2f866add9b52f9ed3246f51`; CLINC150 `d89b4872f8241efd791578676d4c97e15f5702a6dfe34248fcb508865abf060b`; MASSIVE ru-RU `2b4bacc019e82e6c639f84c0d66b772bca745cd22a8f08624279eec1995962d0`. The compact CSV is pinned to run #40279920378 and source SHA `cb8c8eb7e626771d2c32f1ff09572e51f8d30a12`.

This rerun uses the same previously inspected public dataset revisions and official tests. It checks reproducibility, guard behavior and provenance fields; it must not be described as an independent confirmation of generalization or business performance.

## Results

Validation numbers are the AMORPH internal validation splits created only from each official training partition. Official test numbers are report-only and are computed after selection and Proof Gate.

| Dataset | Raw official train / test | Internal validation Macro-F1 (TF-IDF / E5) | Selected | Proof Gate | Official test (TF-IDF) | Official test (E5) |
|---|---:|---:|---|---|---|---|
| BANKING77 | 10,003 / 3,080; 77 classes | 0.8630 / 0.7809 | TF-IDF | INCONCLUSIVE / SELF_COMPARISON | Acc 0.8620; Macro-F1 0.8607 | Acc 0.7903; Macro-F1 0.7660 |
| CLINC150 + OOS class | 15,100 / 5,500; 151 classes | 0.9072 / 0.9307 | TF-IDF (diagnostic fallback; no eligible candidate) | E5 aggregate Proof Gate **ACCEPT**, but mandatory OOS guard **FAIL**; final status `task_requirements_not_met` | Acc 0.7456; Macro-F1 0.8142 | Acc 0.7542; Macro-F1 0.8244 |
| MASSIVE ru-RU | 11,387 prepared train / 2,974 test; 56 train labels, 59 test labels | 0.4517 / 0.2739 | TF-IDF | INCONCLUSIVE / SELF_COMPARISON | Acc 0.6214; fixed-train-label Macro-F1 0.4716 | Acc 0.5319; fixed-train-label Macro-F1 0.3004 |

### CLINC150 gate details

The gate compared E5 with TF-IDF on 3,775 internal gate rows (25 examples per each of 151 classes). Paired Macro-F1 delta was **+0.01097**, 95% bootstrap CI **[+0.00194, +0.02066]**, 500 replicates; the configured non-inferiority margin was 0.01. The rule passed, and legacy parity matched.

Interpretation is intentionally narrow: there was no absolute `min_macro_f1` target in this run; acceptance means the predeclared relative gate conditions passed on that gate sample, not that the model is fit for any business use. The official test contains 1,000 `oos` examples, while official training contains 100. `oos` is treated as a known 151st class here, so this benchmark does not prove open-world OOD detection.

**Critical OOS result on the official test:** training has 100 OOS rows, but official test has 1,000. TF-IDF predicted 112 examples as OOS (precision 0.9107, recall **0.1020**, F1 0.1835); E5 predicted only 39 (precision 1.0000, recall **0.0390**, F1 0.0751). Thus E5's aggregate Gate ACCEPT does **not** show acceptable OOS detection. The internal gate sampled 25 rows per class and did not represent the official test's 1,000-OOS prevalence. A task that needs open-world detection must have a separate predeclared OOS recall/precision/coverage gate and a suitable evaluation protocol; the current one-score Macro-F1 Gate is insufficient.

**Exact train/test text overlap audit:** after Unicode NFKC + trim + casefold + whitespace collapse, 2 of 5,500 official-test rows overlap an input in official train (0.0364%; 2 unique normalized strings). Both test rows have a label not observed for that same normalized input in train. The test set remains unchanged. On the 5,498 rows without normalized exact overlap, E5 accuracy/Macro-F1 are 0.7543/0.8246 and TF-IDF 0.7457/0.8142. This is a small count, but the label conflict merits attention and is explicitly recorded via the SHA-256 of overlap keys rather than publishing text.

### BANKING77 interpretation

The source files were downloaded from the pinned PolyAI-LDN repository revision and verified by SHA-256. E5 was worse than TF-IDF on validation and on official test. Since the chosen candidate is the TF-IDF baseline, Gate correctly returns `INCONCLUSIVE / SELF_COMPARISON`; no challenger acceptance is claimed.

**Exact train/test text overlap audit:** 7/3,080 official-test rows (0.2273%) have a normalized exact input match in official train; all 7 share the same label as their train match. The test remains unchanged. On 3,073 test rows without these matches, TF-IDF accuracy/Macro-F1 are 0.8617/0.8603 and E5 are 0.7908/0.7662, so the overall conclusion does not change. SHA-256 of the set of normalized overlap keys: `e911a21c1879ca61390b055fa6d6850901bf3f9aac9ac2144a31fe7edf8722b2`.

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
3. The successful workflow #40080734847 includes Unicode-normalized exact train/test input overlap counts and hashed overlap-key sets for BANKING77 and CLINC150, and reports diagnostics on test rows without those exact overlaps. These are exact-string checks, not semantic near-duplicate detection. The MASSIVE runner has a different data-preparation path and does not yet apply this same official train-vs-test text overlap check.
4. The current product tests classification strategies (TF-IDF and E5), **not GPT/Grok/Gemini APIs**. No hosted LLM API was called and no API key or paid inference was used.
5. The Gate currently evaluates one configured classification metric and does not by itself measure business-weighted error costs, safety, hallucinations, regression on old generative capabilities, or future distribution shift.
6. The Selector still chooses by runtime proxy. With zero monetary rates, any TCO output is an assumed-free scenario, not a claim of zero cost or minimum money spend.
7. No production defaults were changed. PR #27 remains Draft and must be reviewed before merge.

## Next steps

1. Add a dedicated OOS acceptance criterion; CLINC150 demonstrates that aggregate Macro-F1 acceptance can coexist with very low OOS recall.
2. Preserve this benchmark evidence as reproducibility/regression coverage, not as a substitute for a business-specific task.
3. Run one authorised owner-domain CSV before claiming production suitability.
4. Only then implement/validate `selection_objective=tco`, with explicit rates/horizon/query volume and tests showing the lowest-cost candidate passes all quality gates.


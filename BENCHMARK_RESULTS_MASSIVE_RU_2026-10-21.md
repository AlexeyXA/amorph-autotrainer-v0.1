# MASSIVE ru-RU benchmark — 2026-10-21

## Status

- GitHub Actions run: [#9 — completed successfully](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/39572201595)
- Artifact: [download run evidence (ZIP)](https://github.com/AlexeyXA/amorph-autotrainer-v0.1/actions/runs/39572201595/artifacts/12332543164)
- Benchmark source commit: `91cd7f16f89209160238c2b5847a3a62f70b6fa5`
- Dataset: `AmazonScience/massive`, locale `ru-RU`, pinned dataset revision `c4aa61e89895160619e18a94de72f19b99e72ca2`
- End-to-end workflow elapsed time: 126 seconds (2026-10-21 08:50:44–08:52:50 UTC); pipeline-reported runtime: 101.94 seconds.
- **Important:** the workflow/script completed, but AMORPH's internal acceptance status is `acceptance_gate_underpowered`. The model did **not** pass the predeclared acceptance gate. Treat the reported external metrics as a preliminary, task-specific comparison, not production acceptance.

## Official MASSIVE test metrics

The official test partition was left unchanged and evaluated only after the internal candidate selection/gate. Both candidates were scored against the same 2,974 rows.

| Strategy | Accuracy | Macro-F1 | Balanced accuracy | Macro-F1 on 58-label training taxonomy | Balanced accuracy on 58-label training taxonomy |
|---|---:|---:|---:|---:|---:|
| TF-IDF + classifier | 0.6564 | 0.5274 | 0.5086 | 0.5365 | 0.5173 |
| multilingual-E5 + classifier | 0.5666 | 0.3666 | 0.3705 | 0.3729 | 0.3769 |

Within this run, TF-IDF exceeded E5 by 8.98 percentage points in accuracy and 16.09 percentage points in conventional macro-F1. This is a single benchmark run, not a claim of statistical significance or universal superiority.

Internal validation metrics (used during selection):
- TF-IDF: accuracy 0.6077, macro-F1 0.4889, balanced accuracy 0.4901.
- E5: accuracy 0.5049, macro-F1 0.3440, balanced accuracy 0.3734.
- Internal selector's provisional choice: TF-IDF.
- Final gate: `acceptance_gate_underpowered`; its configured minimum is 5 gate examples per class, and several retained classes fell below that floor. Do not interpret selector choice as a successful AMORPH quality-gate approval.

## Data handling and caveats

The official MASSIVE training file contains conflicting labels for some identical normalized utterances. To preserve AMORPH's fail-closed duplicate validation, the benchmark removed **all training rows** belonging to each normalized-text group with conflicting labels, rather than choosing an arbitrary label. It recorded 23 ambiguous text groups / 72 rows and hashes for the excluded keys and records.

After that, labels without enough independent training support for AMORPH's group-safe four-way split were excluded wholesale: labels `37` (4 unique texts, 2 worker groups) and `41` (18 unique texts, 3 worker groups). This removed 23 additional rows. The effective model-training taxonomy therefore has **58 labels**, not all original 60. The official test was not filtered: it has 2,974 rows and includes label `41`, which has no corresponding training examples; this limits interpretation of conventional macro-F1. Label `37` is absent from the official test.

After exact deduplication, 10,957 training records entered the internal pipeline. A joint SciPy/HiGHS MILP assigned 56 worker groups into leakage-safe internal splits:
- Train: 5,478 rows (49.995%)
- Validation: 1,644 rows (15.004%)
- Acceptance gate: 1,644 rows (15.004%)
- Internal test: 2,191 rows (19.996%)

The solver returned a valid feasible assignment at its 10-second time limit; it did not prove optimality. Every retained label appeared in all four internal splits, and each worker group was assigned to exactly one split. The character n-gram near-duplicate audit was skipped because the dataset exceeded its configured 5,000-row audit cap; therefore worker-group separation is verified, but near-duplicate separation is not fully established. The official test may also contain annotation-worker identities present in the official train partition.

## Reproduction configuration

Command:
```bash
python scripts/benchmark_massive_ru.py \
  --with-e5 \
  --output-dir benchmark_runs/massive_ru_e5 \
  --max-runtime-seconds 7200
```

- Python 3.12.15; PyTorch CPU build 2.14.1+cpu; scikit-learn 1.9.1; SciPy 1.18.1; sentence-transformers 6.1.0.
- Random seed: 42; paired bootstrap samples: 500; configured minimum acceptance-gate support: 5 examples/class; non-inferiority margin: 0.01.
- E5 model: `intfloat/multilingual-e5-small`, revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- The 58-label training taxonomy and filtering decisions are explicitly reported in `dataset_provenance.json`.

## SHA-256 evidence

Pinned source Parquet files:
- MASSIVE ru-RU train Parquet: `89ecdc8ddfd00b38d05624c8061dbd10ea7b63f4587ee82750e0e48ebdfb91eb`
- MASSIVE ru-RU official test Parquet: `d53f1c58d753e64b7cf7f10face6c4a57ac93e7f9db82826b43058eacaf2e46a`

Prepared benchmark data and result files:
- Prepared train CSV (11,419 rows before AMORPH exact dedup): `1f4f9b7b7e36c37d3133fe1fea285ad2923b0979fc2c442495f55f4496993082`
- Unmodified official test CSV (2,974 rows): `aaa90bd680ce470167e75b58520c1def0d8f8353deb86ae368a871653e27bdd4`
- `benchmark_result.json`: `1bd8de413f3c7fc3fd2286c167ff4ec6f2a0565089c95cf3c12ebad273589996`
- `dataset_provenance.json`: `6a854238f0d94167b4a35fd7551b9a782fa2313dcdabb74900550b0c7dae6416`
- Locally downloaded artifact ZIP: `d41e9b755ccd0f485224dc5c635edbe3edb62ca757ad1da91977e725b253d648`

The artifact also contains `execution_manifest.json`, internal split hashes/report, serialized TF-IDF/E5 candidate artifacts and the experiment SQLite database. The execution manifest includes SHA-256, size and path for every exported file.

## Interpretation

The benchmark script and artifact pipeline completed. The current evidence suggests TF-IDF is the stronger baseline than this E5 configuration for this filtered MASSIVE ru-RU intent-classification setup. However, the training taxonomy was reduced from 60 to 58 labels, the internal acceptance gate was underpowered, and the near-duplicate audit was skipped at the size cap. A stronger next iteration should resolve the gate-support design and near-duplicate audit policy, then rerun before making acceptance or production-quality claims. MASSIVE evaluates intent classification, not RAG or LoRA.

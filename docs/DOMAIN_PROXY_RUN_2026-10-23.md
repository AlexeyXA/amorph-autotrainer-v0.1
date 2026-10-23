# Domain-proxy run (2026-10-23)

**Not owner private data.** Synthetic support-intent CSV used only to exercise the protocol end-to-end and show what READINESS item 8 looks like in the manifest.

## Dataset (hashes only)

| Field | Value |
|-------|--------|
| rows | 1050 |
| labels | 5 (billing, shipping, account, product, cancel) |
| groups | 70 |
| `source_sha256` | `dce29311f848ae4137c117cdcef2fe524bc43d463bde07233ff9a43adab85eec` |

## Run A — AutoTrainer (`tfidf` only)

| Field | Value |
|-------|--------|
| run_id | `20261023T084157Z-e949fb4d` |
| status | `baseline_only` |
| `acceptance_gate_sufficient` | `true` |
| `decision_source` | `proofgate.evaluate` |
| `cutover_complete` | `true` |
| Proof Gate verdict | **`INCONCLUSIVE` / `SELF_COMPARISON`** |
| `decision_passed` | `false` |
| `legacy_parity_status` | `intentional_inconclusive_self_comparison` |
| selected report_sha256 | `786d232206cefa21f1b512b01a44094c75ff026db3fb800476f3835171a7e01d` |

Expected: single strategy → no challenger → not ACCEPT.

## Run B — Standalone Proof Gate (weak vs strong sklearn)

Group-safe holdout (~25% gate). Baseline: word TF–IDF `max_features=80`, `C=0.1`. Challenger: word+char TF–IDF, `C=2`, `class_weight=balanced`.

| Field | Value |
|-------|--------|
| n_train / n_gate | 780 / 270 |
| groups train / gate | 52 / 18 (disjoint) |
| baseline macro-F1 (gate) | ~0.940 |
| candidate macro-F1 (gate) | 1.0 |
| Δ mean / 95% CI | ~+0.060 / [+0.037, +0.088] |
| Proof Gate verdict | **`ACCEPT`** |
| powered | `true` |
| report_sha256 | `813899fde856f8b6c8c3c80e74f20b7aae52a6f9e0b034423a8ccffaf8101af7` |

## What this does **not** claim

- Not the owner’s production domain
- Not multi-seed generalization
- Not a reason to change E5 defaults or merge PR #4
- Synthetic separability is easy; real domains will often see REJECT or INCONCLUSIVE

## Owner next step for READINESS #8

Repeat Run A (and optional multi-strategy) on **private** CSV; paste only status / verdict / subtype / hashes — never raw text.

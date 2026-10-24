# TCO selector — draft v0.1

## Status

Implemented in Draft PR #27 on branch `research/real-task-verification-20261024`. **Not merged into `main`; default selection remains `runtime_proxy`.** This is an estimation/selection mechanism for the current local classification candidates (TF-IDF, E5 and experimental hybrid). It does not connect GPT, Grok or Gemini APIs and does not turn public API pricing tables into model adapters.

## Objectives

- `selection_objective=runtime_proxy` (default): preserve the current rule, minimizing candidate fit seconds + expected queries × measured latency.
- `selection_objective=tco`: among candidates that pass aggregate Macro-F1 validation eligibility and configured task-specific per-label floors on validation, minimize estimated euro TCO. Exact monetary ties are broken by runtime proxy and explicitly reported.

The selected challenger still has to pass Proof Gate and per-label checks on the separate acceptance-gate split. If it fails, the current pipeline falls back to TF-IDF. **The current implementation does not yet evaluate every validation-eligible challenger through the acceptance gate before ranking all gate-passing candidates; do not claim a globally minimum-cost accepted challenger when another challenger was not gate-tested.** That is a separate follow-up required for a full cost-optimal search policy.

## Cost function

For strategy (j) over a user-specified query horizon (Q):

`C_j = C_compute,j + C_shared + C_usage,j(Q) + C_fixed,j`

where:
- `C_compute,j` = (preprocessing + candidate-fit seconds + expected query inference seconds) × compute EUR/hour ÷ 3600;
- `C_usage,j(Q)` = strategy-specific EUR/query × `expected_queries`;
- `C_fixed,j` = candidate-specific setup or horizon-allocated cost;
- `C_shared` = fixed data + maintenance + human costs supplied in common for every candidate.

The report records fit/latency, each cost component, expected query count, objective, all eligible candidate estimates, exact-cost ties, cost inputs and warnings. Shared costs appear in TCO totals but cancel out of relative candidate ranking if identical for each candidate. Candidate-specific recurring/setup fees are needed when methods have different costs not captured by compute time.

## CLI

Example for a local classifier measured on a server billed at €2/hour:

```bash
amorph-autotrainer train data.csv \
  --text-column text --label-column label \
  --strategies tfidf,e5 --max-candidates 2 \
  --selection-objective tco \
  --compute-eur-per-hour 2.00 \
  --expected-queries 100000 \
  --fixed-data-cost-eur 25 \
  --maintenance-cost-eur 10 \
  --human-hours 1 --human-eur-per-hour 30 \
  --min-macro-f1 0.80 \
  --out runs
```

Example with candidate-specific usage costs, e.g. externally billed per-query charges or cost differences not represented by local compute time:

```bash
amorph-autotrainer train data.csv \
  --strategies tfidf,e5 --selection-objective tco \
  --expected-queries 100000 \
  --compute-eur-per-hour 0 \
  --strategy-usage-cost-eur-per-query tfidf=0.0001 \
  --strategy-usage-cost-eur-per-query e5=0.00002 \
  --strategy-fixed-cost-eur tfidf=5 \
  --strategy-fixed-cost-eur e5=25 \
  --out runs
```

When `compute_eur_per_hour=0`, TCO mode requires explicit `--strategy-usage-cost-eur-per-query` and `--strategy-fixed-cost-eur` entries for every requested strategy. An explicit zero is allowed, but the report must disclose assumed-free inputs; zero is not evidence that the resource actually costs nothing. If a positive uniform compute rate is supplied but strategy-specific usage/setup entries are missing, missing components are assumed zero with a warning.

## Output contract

- `selection_objective`: requested objective.
- `selector.candidate_costs`: detailed estimates for the eligible candidates.
- `selector.minimum_estimated_tco_eur`: minimum estimated monetary TCO among those candidates.
- `selector.tco_tied_candidates` / `selector.tie_breaker`: show ties; runtime only breaks a tie.
- `selector.candidate_tco_values_differ`: whether estimated TCO actually differs across the eligible set.
- `estimated_tco`, `selected_model_tco`, `full_selection_run_tco`: operational estimate fields at different scopes.
- `tco_assumption_status` and warnings: disclose zero or missing cost assumptions.

A real TCO claim still requires measured/justified compute rates, human/data/maintenance costs, realistic query count and a comparable horizon. Current API-market snapshot is not automatically wired to a hosted-LLM candidate; TF-IDF/E5 are the only candidates in this implementation.

## Known limitations / next steps

1. Candidate eligibility before gate uses validation and task-specific point estimates. Per-label confidence intervals and multi-seed variability are not yet implemented.
2. The pipeline gates the preselected candidate, then falls back to TF-IDF if it fails. Full minimum-TCO selection among all gate-passing challengers requires running Proof Gate per challenger on the same predeclared gate set without using official test data, then selecting among accepted candidates.
3. Cost coverage is only as accurate as supplied assumptions. The API usage rate should include input/output tokens, cache discounts, tools, retries and realistic query profiles when an actual API strategy adapter is added.
4. Current selector is not a hosted LLM router. No GPT/Grok/Gemini calls are made by this branch.
5. Public datasets do not establish the owner's business utility or financial savings; a domain CSV and baseline cost measurement are still required.

## Regression tests

The draft tests:
- TCO can choose a slower candidate if its specified total euro cost is lower.
- `runtime_proxy` remains the default and ignores monetary overrides for ranking.
- equal-cost candidates are reported as tied and runtime proxy is only the tie-breaker.
- TCO mode with zero compute rate fails closed if per-strategy cost inputs are missing.
- explicit zero rates are accepted but surfaced as assumed-free in the report.

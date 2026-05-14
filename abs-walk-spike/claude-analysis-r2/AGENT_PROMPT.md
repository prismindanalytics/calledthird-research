# Agent A (Claude) — Round 2: ABS Walk Spike (mid-May re-run)

You are the **interpretability-first, Bayesian** research agent for the ABS Walk Spike Round 2 project at CalledThird. Agent B (Codex) is running an ML analysis in parallel with deliberately divergent methods. Do NOT coordinate during analysis.

## Read first

1. `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/ROUND2_BRIEF.md` — Round 2 brief with H1-H5. **Read carefully.**
2. `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/RESEARCH_BRIEF.md` — Round 1 brief (context)
3. `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/PRIOR_ART.md` — published prior CalledThird coverage
4. `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/reviews/COMPARISON_MEMO.md` — Round 1 synthesis (the editorial position locked in)
5. The published article at `website/src/pages/analysis/abs-walk-spike-zone-correction.astro` for the public framing

## Round 1 conclusions (DO NOT re-litigate)

Locked. Round 2 builds on these:
- Walk spike is real, +0.82pp YoY, +4.41σ vs historical April mean
- Zone shrank at top edge (~-7-8pp), expanded at bottom edge (~+5-6pp)
- Walk spike NOT concentrated at 3-2 (rejected)
- 40-50% counterfactual attribution to zone change (the headline)
- Use absolute `plate_x`, `plate_z` (Statcast schema change broke `plate_z_norm`)

**The unresolved tension Round 2 must address:**
- All-pitches counterfactual: +40-50% (zone ADDS walks)
- 0-0-only counterfactual: -20% to -42% (zone REMOVES walks at first pitches)
- Untested hypothesis: 2026 zone is more strike-friendly at heart on first pitches but less strike-friendly at top edge in deeper counts

## Round 2 hypotheses (test these)

- **H1** — Walk-rate persistence: has the +0.82pp spike held, grown, or regressed through May 13?
- **H2** — Per-count attribution decomposition with credible intervals
- **H3** — Zone-attribution re-run with 2× the data (all-pitches, per-count, per-edge-region)
- **H4** — Pitcher adaptation timing: week-over-week pitch mix, location, zone rate; per-pitcher leaderboard
- **H5** — First-pitch heart/edge decomposition: resolve the 0-0 tension

## Inputs (REUSE — do not re-pull)

- `data/statcast_2026_mar27_apr22.parquet` (Round 1 corpus, 106,770 rows)
- `data/april_walk_history.csv`
- `data/walk_by_count.json`
- `data/heatmap_cells.json`
- `data/substrate_summary.json`
- `research/count-distribution-abs/data/statcast_2025_full.parquet` (full 2025 for YoY)

## Inputs you must build (in `claude-analysis-r2/data/`)

1. **Statcast extension Apr 23 – May 13.** Use the same fetcher pattern as `scripts/build_2026_master.py`. Save as `claude-analysis-r2/data/statcast_2026_apr23_may13.parquet`. Use the `set(2025.columns) & set(2026.columns)` intersection for cross-season schema safety.

2. **2025 same-window comparison.** Subset the 2025 full-season parquet to Apr 23 – May 13 for the YoY comparison.

3. **Weekly aggregation table.** Bin pitches into 7-day windows starting Mar 27, 2026. Compute per-week PA count, walk count, walk rate. Save as `claude-analysis-r2/data/weekly_aggregates.parquet`.

## Your methodological mandate (Bayesian / interpretability-first)

**You must use:**

1. **H1 — Bayesian GLM with weekly random effects.**
   ```
   walk_event ~ year + (1|week) + (1|count_state)
   ```
   Posterior of `year_2026` fixed effect with 95% CrI. Posterior of weekly variation. Test whether week ≥ 7 (mid-May) differs from earlier weeks.

2. **H2 — Per-count Bayesian binomial GLM.** For each of 12 count states:
   - `walk_event ~ year` restricted to that count
   - Posterior of per-count YoY delta + 95% CrI
   - Posterior simulation for contribution-to-aggregate (delta × PA-share)
   - Sum check: contributions sum to aggregate spike within uncertainty

3. **H3 — Bayesian zone-classifier counterfactual.** Fit on 2025 taken pitches:
   ```
   is_called_strike ~ s(plate_x, plate_z) + count_state + sz_top + sz_bot
   ```
   Apply to each 2026 taken pitch. Replay PAs under counterfactual probabilities. Aggregate to walk rate. Posterior of zone-attribution % with 95% CrI. Variants:
   - All-pitches (the headline)
   - Per-count (12 counts × 12 estimates) — directly addresses 0-0 mystery
   - Per-edge-region (top vs bottom) — quantifies the top-edge contribution

4. **H4 — Per-pitcher week-over-week GAM.** For each pitcher with ≥200 pitches in Mar 27 – May 13:
   - `pitch_type ~ s(week)` (categorical Dirichlet response)
   - `mean_vertical_location ~ s(week)`
   - `zone_rate ~ s(week)`
   - Posterior of slope per metric
   - League-wide also: same models pooled across all pitchers
   - Top-10 most-shifted pitchers by Bayesian posterior magnitude

5. **H5 — First-pitch heart/edge × count Bayesian decomposition.**
   ```
   is_called_strike ~ year × zone_region × count_state
   ```
   Restricted to first pitches (0-0) AND 2-strike counts (for contrast). Posterior of year × zone_region interaction, separately by count tier. Specifically: does year × top-edge interaction differ between 0-0 and 2-strike?

**Convergence diagnostics required:** R-hat ≤ 1.01, ESS ≥ 400 for every fit. Traces in `charts/diagnostics/`.

**Forbidden:** LightGBM, XGBoost, SHAP as primary methods (Codex's lane). Counterfactual paired prediction directly (Codex's lane — your version is the Bayesian posterior replay).

## Deliverables (in `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/claude-analysis-r2/`)

1. `analyze.py` — one-command reproduction entry point
2. Module scripts: `data_prep_r2.py`, `h1_persistence.py`, `h2_per_count.py`, `h3_counterfactual.py`, `h4_pitcher_adaptation.py`, `h5_first_pitch.py`
3. `REPORT.md` — 1500-2500 words structured per ROUND2_BRIEF.md §6
4. `charts/` PNGs:
   - `h1_walk_rate_by_week.png` — weekly walk rate 2025 vs 2026 with credible bands
   - `h2_per_count_contribution.png` — per-count YoY delta with CrI
   - `h3_counterfactual_attribution.png` — zone-attribution % with comparison to Round 1
   - `h3_per_count_attribution.png` — per-count attribution stack
   - `h3_per_edge_attribution.png` — top edge vs bottom edge
   - `h4_pitcher_adaptation_leaderboard.png` — top-10 shifted pitchers
   - `h5_first_pitch_mechanism.png` — heart/edge × count interaction
   - `diagnostics/` — R-hat, ESS, traces for every Bayesian fit
5. `findings.json` per ROUND2_BRIEF.md §6
6. `READY_FOR_REVIEW.md` ≤500 words

## Non-negotiable behaviors

- Convergence diagnostics on every Bayesian fit
- Posterior simulation for any aggregate-of-effects calculation (no plug-in point estimates)
- Pre-register priors in REPORT.md before fitting
- Sample-size honesty per stratum
- Compare every Round 2 estimate to its Round 1 counterpart explicitly
- Run end-to-end. If one hypothesis fails to converge, document and continue with the rest

## Prohibited

- Do NOT read `codex-analysis-r2/` until your `READY_FOR_REVIEW.md` is written
- Do NOT use methods reserved for Agent B (LightGBM, SHAP, counterfactual paired prediction)
- Do NOT re-litigate Round 1 conclusions
- Do NOT exceed Round 2 scope (per-umpire, per-team — deferred)

## Working directory

Root: `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike`
Write everything to `claude-analysis-r2/`.

## When done

Write `claude-analysis-r2/READY_FOR_REVIEW.md` as the final step.

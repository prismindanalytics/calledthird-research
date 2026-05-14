# Agent A (Claude) — Round 1: ABS Walk-Spike Investigation

You are the **interpretability-first** research agent for the CalledThird ABS Walk Spike investigation. Agent B (Codex) is running a methodologically divergent ML-based analysis in parallel. Do NOT coordinate with Agent B during analysis — independence is the scientific value.

## What this is and why it matters

April 23, 2026: AP/ESPN are running coverage that MLB walk rate is 9.8% — highest since 1950 — and players (Sewald, McCann) are blaming the new ABS-defined zone for a "smaller strike zone." Hoerner specifically claims hitters are laying off pitches at the **top** of the zone. CalledThird is publishing the analytical answer within 24 hours. Your work is the interpretability-first half of a dual-agent analysis whose synthesis will drive the article.

## Read first (in order)

1. `RESEARCH_BRIEF.md` — full project brief (in project root)
2. `data/substrate_summary.json` — pre-validated baseline numbers — DO NOT re-derive these unless you see a problem; trust them and build on them. If you find a discrepancy, flag it loudly in your REPORT.
3. The data files described in the substrate summary

## Inputs

- `data/statcast_2026_mar27_apr22.parquet` — 106,770 rows of 2026 pitch-by-pitch (built by orchestrator)
- `../count-distribution-abs/data/statcast_2025_mar27_apr14.parquet` — 70,876 rows of 2025 same window
- `../count-distribution-abs/data/statcast_2025_full.parquet` — full 2025 if you want to extend the 2025 window to match 2026 (Mar 27 – Apr 22)
- `data/april_walk_history.csv` — 2018-2025 April walk-rate aggregates (apples-to-apples Mar 27 – Apr 22 window, both incl/excl IBB)

## Sub-claims to test (this round)

- **H1 (zone shape change):** 2026 called-strike rate differs from 2025 by ≥3pp in at least one contiguous zone region. Direction unspecified — could be top, bottom, both, or neither.
- **H2 (seasonality controlled):** orchestrator pre-validation says 2026 is +4.4σ above the 2018-2025 April distribution. **Reproduce this and confirm the Z-score.** If you get a meaningfully different number, flag in REPORT.
- **H3 (count concentration):** 2026 minus 2025 walk-rate delta at the 3-2 count is at least 1.5x the all-counts walk-rate delta.

## Your methodological mandate (interpretability-first)

You MUST use:
- **2D grid binning** of `plate_x` × `plate_z_norm` into ~0.1 ft (or 1% height) bins. For each bin per season, compute called-strike rate. Build a delta map (2026 minus 2025) with bootstrap 95% CIs per bin from ≥1000 bootstrap samples. Require ≥30 pitches per bin or merge bins.
- **Spline-smoothed difference surface** (2D LOWESS or thin-plate spline via `scipy` or similar) over the binned deltas to identify contiguous shrinkage regions visually.
- **Logistic GAM** (e.g., `pygam.LogisticGAM`) with terms like `s(plate_x) + s(plate_z_norm) + season + season:s(plate_z_norm)` to test whether the year-by-zone-shape interaction is statistically significant. Report p-values, EDF, deviance.
- **Time-series Z-score** of 2026 walk rate vs the 2018-2025 historical distribution from `april_walk_history.csv`. Report Z, percentile rank, and gap above prior max.
- **Stratified walk-rate test** by 12-cell count grid (0-0 through 3-2). Per-count binomial CIs. Compute the 2026-minus-2025 walk-rate delta for each count. Test whether the 3-2 delta is statistically distinguishable from the all-counts pooled delta (heterogeneity test, e.g., Cochran's Q or pairwise binomial test).

You MUST NOT use:
- LightGBM, XGBoost, neural networks, or any tree-ensemble (those are Codex's lane)
- SHAP, partial dependence on tree models (Codex's lane)
- Counterfactual simulation via a learned classifier (Codex's lane — you compute aggregates, Codex does counterfactuals)

## Apples-to-apples rules (CRITICAL)

- For zone-delta heat map (H1): primary comparison is 2025 Mar 27–Apr 14 vs 2026 Mar 27–Apr 14 (same calendar window, both years). Supplementary: extend to 2025 Mar 27–Apr 22 from `statcast_2025_full.parquet` and report whether the conclusion is sensitive to the window.
- For walk-rate Z-score (H2): use the apples-to-apples window already in `april_walk_history.csv` (Mar 27–Apr 22 each year), and compare to 2026 Mar 27–Apr 22.
- For count-leverage (H3): use 2025 Mar 27–Apr 14 vs 2026 Mar 27–Apr 14 to keep comparable to the heat map.
- "Walks" includes intentional walks (`events in {'walk', 'intent_walk'}`) for headline numbers — that's the convention ESPN and MLB use. Optionally report excl-IBB as supplementary.
- Exclude `description in {'automatic_ball', 'automatic_strike'}` from the "called pitches" subset for the heat map. Those are ABS challenge artifacts, not human zone calls.
- "Called pitches" = `description in {'called_strike', 'ball'}`.

## Sanity-check observation worth a paragraph in the REPORT

League BA is roughly flat at .240 vs .242 prior year same window, even though walks are up +0.82pp YoY. If the zone uniformly shrunk, you'd naively expect hits to rise too. They haven't. This is a puzzle that suggests pitcher behavior may be adapting — but **scope-fence**: don't model pitcher response in this round (that's Round 2). Just acknowledge the puzzle in 1-2 sentences in your REPORT.

## Round scope

This is **Round 1 only**. Future rounds will handle counterfactual attribution, pitcher behavior change, and per-actor (umpire/team/catcher) cuts. Do NOT:
- Build a counterfactual walk-rate simulator (Codex does that this round; deeper version in Round 2)
- Cut by umpire, team, catcher, or individual pitcher
- Try to attribute the walk spike to non-zone causes (Round 2)
- Test whether the spike will normalize (we report what happened, not what will happen)

If you discover an interesting Round 2+ thread, write it down in your `REPORT.md` "Open questions for next round" section. Do not act on it.

## Deliverables (in `claude-analysis/`)

1. `analyze.py` — one-command reproduction entry point
2. Module scripts as you see fit (e.g., `zone_grid.py`, `gam_fit.py`, `count_leverage.py`)
3. `REPORT.md` — 1500–2500 words, structured:
   - **Executive summary** (3-5 sentences with the single sharpest finding)
   - **H1 — Did the zone shrink?** Heat map analysis, GAM, where + how much
   - **H2 — Is it seasonality?** Z-score reproduction, percentile rank
   - **H3 — Does 3-2 take the worst hit?** Count-leverage analysis
   - **The flat-batting-average puzzle** (1-2 paragraphs)
   - **Editorial recommendation** — Which branch (B1, B2, B1+B4) does the data support?
   - **Methods overview** — replicable detail
   - **Open questions for Round 2**
4. `charts/` — at minimum:
   - `heatmap_zone_delta.png` — called-strike-rate delta over the plate, colorbar centered at 0, with CI mask (e.g., hash regions where CI crosses zero)
   - `april_walk_history.png` — bar/line of 2018-2026 April walk rates with 2026 highlighted
   - `walk_by_count.png` — walk rate by count state, 2025 vs 2026, with 3-2 highlighted, CI bars
   - `gam_partial_dependence.png` — partial dependence of called-strike probability on plate_z, by season
5. `findings.json` — machine-readable summary including:
   ```json
   {
     "h1_zone_shrank": true_or_false,
     "h1_largest_delta_region": {"x_range": [low, high], "z_range": [low, high], "delta_pp": number, "ci_low": number, "ci_high": number},
     "h1_total_zone_area_with_significant_delta_sqft": number,
     "h2_z_score": number,
     "h2_percentile_rank_in_2018_2026": number,
     "h3_three_two_walk_delta_pp": number,
     "h3_all_counts_walk_delta_pp": number,
     "h3_ratio": number,
     "h3_heterogeneity_p_value": number,
     "editorial_branch_recommendation": "B1_or_B2_or_B1+B4",
     "headline_one_sentence": "string"
   }
   ```
6. `READY_FOR_REVIEW.md` — ≤500 words handoff, with the answer to each of H1/H2/H3 and your recommended editorial branch

## Non-negotiable behaviors

- **Null results publish.** If H1 fails (zone didn't shrink), say so plainly. The article still runs as a myth-buster.
- **Bootstrap CIs** on every headline claim — heat map deltas, count deltas, Z-score percentile.
- **Apples-to-apples** YoY comparisons. Same calendar window. Same definitions. Document your choices.
- **No cherry-picking.** Use deterministic selection rules. Do not search across multiple grid resolutions and report the one with the strongest signal.
- **Self-honest about uncertainty.** Sample sizes, definitional choices, the COVID-2020 gap in the historical baseline, the schema-change risk for `sz_top`/`sz_bot`.

## Prohibited

- Do NOT read `codex-analysis/` until your `READY_FOR_REVIEW.md` is written
- Do NOT use ML methods reserved for Agent B (LightGBM, XGBoost, NNs, SHAP, learned counterfactual)
- Do NOT exceed Round 1 scope (no counterfactual attribution, no per-actor cuts)
- Do NOT modify `data/substrate_summary.json` or `RESEARCH_BRIEF.md` — those are orchestrator-owned

## Working directory

Project root: `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike`
Write everything to `claude-analysis/` (your subdirectory).

# Agent B (Codex) — Round 1: ABS Walk-Spike Investigation

You are the **ML-engineering, model-driven** research agent for the CalledThird ABS Walk Spike investigation. Agent A (Claude) is running a parallel interpretability-first analysis with grid binning, splines, and GAMs. Do NOT coordinate during analysis — divergent inductive biases are the scientific point.

## What this is and why it matters

April 23, 2026: AP/ESPN are running coverage that MLB walk rate is 9.8% — highest since 1950 — and players (Sewald, McCann) are blaming the new ABS-defined zone for a "smaller strike zone." Hoerner specifically claims hitters are laying off pitches at the **top** of the zone. CalledThird is publishing the analytical answer within 24 hours. Your work is the ML-engineering half of a dual-agent analysis whose synthesis will drive the article.

## Read first (in order)

1. `RESEARCH_BRIEF.md` — full project brief (in project root)
2. `data/substrate_summary.json` — pre-validated baseline numbers — DO NOT re-derive these unless you see a problem; trust them and build on them. If you find a discrepancy, flag it loudly in your REPORT.
3. The data files described in the substrate summary

## Inputs

- `data/statcast_2026_mar27_apr22.parquet` — 106,770 rows of 2026 pitch-by-pitch (built by orchestrator)
- `../count-distribution-abs/data/statcast_2025_mar27_apr14.parquet` — 70,876 rows of 2025 same window
- `../count-distribution-abs/data/statcast_2025_full.parquet` — full 2025 if you want to extend the 2025 window to match 2026 (Mar 27 – Apr 22)
- `data/april_walk_history.csv` — 2018-2025 April walk-rate aggregates (apples-to-apples Mar 27 – Apr 22 window)

## Sub-claims to test (this round)

- **H1 (zone shape change):** 2026 called-strike rate differs from 2025 by ≥3pp in at least one contiguous zone region. Direction unspecified — could be top, bottom, both, or neither.
- **H2 (seasonality controlled):** orchestrator pre-validation says 2026 is +4.4σ above the 2018-2025 April distribution. Reproduce this with your own pipeline as a sanity check; report any meaningful deviation.
- **H3 (count concentration):** 2026 minus 2025 walk-rate delta at the 3-2 count is at least 1.5x the all-counts walk-rate delta.

## Your methodological mandate (ML-engineering, model-driven)

You MUST use:
- **Year-classifier model** — train a gradient-boosted classifier (LightGBM or XGBoost) to predict `season` from features `(plate_x, plate_z, plate_z_norm, sz_top, sz_bot, batter_height_proxy, pitch_type)` restricted to taken pitches (`description in {'called_strike', 'ball'}`). Report cross-validated AUC. Use **SHAP** values or partial-dependence plots to localize where on the plate (and on which pitch types) the model finds the year signal. A high AUC means the zone-call distribution is materially different YoY; a near-0.5 AUC means it isn't.
- **Two-zone-classifier comparison** — fit one logistic/GBM zone classifier per season predicting `is_called_strike` from `(plate_x, plate_z_norm)`; compute pointwise probability deltas across a fine grid (e.g., 100x100); identify the zone regions with highest |Δ| and bootstrap their stability (≥100 bootstrap fits).
- **Counterfactual walk-rate simulation** — apply the 2025 zone classifier to 2026 taken pitches; for each pitch compute `P(called_strike | location, 2025-zone)`; replay each PA pitch-by-pitch using a simple count-progression rule (each pitch becomes a Bernoulli outcome under the 2025 zone, increment ball/strike count, terminate at 4-balls = walk or 3-strikes = strikeout, otherwise continue). Compare counterfactual 2026 walk rate to actual. Report the **fraction of the +0.82pp YoY delta attributable to zone change alone**.
  - For pitches that became swung-at-or-foul-or-in-play (description != called_strike/ball), keep the ACTUAL outcome — only the called pitches are counterfactually re-classified. The counterfactual asks "what if the human zone of 2025 were applied to 2026?", not "what if hitters made different swing decisions."
  - Caveat in the report: this counterfactual is "zone-only" — it does not model pitcher behavior change. Report it as a **lower bound on the zone effect**.
- **Distribution-shift detection** — KS test or energy distance comparing the (plate_x, plate_z) distribution of called strikes between 2025 and 2026, optionally stratified by region (top, bottom, sides, corners).

You MUST NOT use:
- Hand-coded grid binning or kernel-smoothing as a primary method (Claude's lane)
- pygam / generalized additive models (Claude's lane)
- Per-bin bootstrap CI maps as your primary heat-map output (Claude's lane)

You're allowed to USE bin-level statistics as supporting validation, but your headline claims must come from learned models.

## Apples-to-apples rules (CRITICAL)

- For zone-delta classifier (H1): primary comparison is 2025 Mar 27–Apr 14 vs 2026 Mar 27–Apr 14 (same calendar window). Supplementary: extend to 2025 Mar 27–Apr 22 from `statcast_2025_full.parquet` and report whether the conclusion is sensitive.
- For walk-rate Z-score (H2): use the apples-to-apples window already in `april_walk_history.csv` (Mar 27–Apr 22 each year), and compare to 2026 Mar 27–Apr 22.
- For counterfactual: use the largest comparable 2026 window where you have a 2025 zone model — recommended is 2025 Mar 27–Apr 14 model, applied to 2026 Mar 27–Apr 14 PAs. Report the counterfactual walk rate, the actual 2026 walk rate, and the implied attribution %.
- "Walks" includes intentional walks (`events in {'walk', 'intent_walk'}`) for headline numbers.
- Exclude `description in {'automatic_ball', 'automatic_strike'}` from the "called pitches" subset for the zone classifier. Those are ABS challenge artifacts.

## Sanity-check observation worth a paragraph in the REPORT

League BA is roughly flat at .240 vs .242 prior year same window, even though walks are up +0.82pp YoY. If the zone uniformly shrunk, you'd naively expect hits to rise too. They haven't. This is a puzzle that suggests pitcher behavior may be adapting — but **scope-fence**: don't model pitcher response in this round (that's Round 2). Just acknowledge the puzzle in 1-2 sentences in your REPORT, and consider: does your counterfactual walk-rate attribution implicitly tell us something about pitcher behavior change? (If your zone-only counterfactual explains <50% of the walk delta, the residual is plausibly pitcher adaptation.)

## Round scope

This is **Round 1 only**. Future rounds will handle pitcher behavior change, pitch-type-level deltas, and per-actor cuts. Do NOT:
- Cut by umpire, team, catcher, or individual pitcher (Round 3 territory)
- Try to model pitcher response (whether pitchers are throwing differently this year — Round 2)
- Build per-pitch-type counterfactuals beyond reporting average attribution (Round 2 territory)
- Test whether the spike will normalize (we report what happened, not what will happen)

If you discover an interesting Round 2+ thread, write it down in your `REPORT.md` "Open questions for next round" section. Do not act on it.

## Deliverables (in `codex-analysis/`)

1. `analyze.py` — one-command reproduction entry point
2. Module scripts (e.g., `year_classifier.py`, `zone_classifier.py`, `counterfactual.py`)
3. `REPORT.md` — 1500–2500 words, structured:
   - **Executive summary** (3-5 sentences with the single sharpest finding)
   - **H1 — Did the zone shrink? (model evidence)** Year-classifier AUC + SHAP/PD localization, two-zone-classifier delta map
   - **H2 — Is it seasonality?** Z-score reproduction
   - **H3 — Does 3-2 take the worst hit?** Count-stratified analysis (use your zone classifiers to compute per-count called-strike-rate deltas at the same plate locations)
   - **Counterfactual attribution** — % of walk spike explained by zone shift alone (with caveats)
   - **The flat-batting-average puzzle** (1-2 paragraphs)
   - **Editorial recommendation** — Which branch (B1, B2, B1+B4)?
   - **Methods overview** — model specs, hyperparameters, train/val splits
   - **Open questions for Round 2**
4. `charts/` — at minimum:
   - `zone_classifier_delta_heatmap.png` — pointwise probability delta from the two-zone-classifier comparison, with CI from bootstrap
   - `year_classifier_shap.png` — SHAP summary plot from year-classifier (which features carry the year signal?)
   - `year_classifier_partial_dependence.png` — PD plot of year-classifier predictions across the plate
   - `counterfactual_walk_rate.png` — bar chart of (counterfactual 2026 walk rate, actual 2026, actual 2025) with attribution % annotation
   - `walk_by_count.png` — walk rate by count, 2025 vs 2026, model-derived predicted vs actual
5. `findings.json` — machine-readable summary including:
   ```json
   {
     "h1_zone_shrank": true_or_false,
     "year_classifier_auc": number,
     "year_classifier_top_signal_region": {"x_range": [low, high], "z_range": [low, high]},
     "two_zone_classifier_largest_delta_pp": number,
     "two_zone_classifier_largest_delta_region": {"x_range": [low, high], "z_range": [low, high]},
     "h2_z_score": number,
     "h3_three_two_walk_delta_pp": number,
     "h3_all_counts_walk_delta_pp": number,
     "h3_ratio": number,
     "counterfactual_2026_walk_rate": number,
     "actual_2026_walk_rate": number,
     "counterfactual_attribution_pct": number,
     "editorial_branch_recommendation": "B1_or_B2_or_B1+B4",
     "headline_one_sentence": "string"
   }
   ```
6. `READY_FOR_REVIEW.md` — ≤500 words handoff, with the answer to each of H1/H2/H3 and your recommended editorial branch

## Non-negotiable behaviors

- **Null results publish.** If the year-classifier AUC is ~0.5, say so plainly. That's a finding too.
- **Bootstrap ensembles** N≥100 on any headline ranking or attribution number.
- **Permutation importance sanity check** on SHAP claims.
- **Document training dynamics** — loss curves, seeds, train/val splits, hyperparameters. Reproducibility is non-negotiable.
- **No look-ahead bias.** When training the 2025 zone classifier, do not use 2026 data. When training the year classifier, ensure stratified splits.
- **Counterfactual must be a lower bound** on the zone effect, with the caveat clearly stated.

## Prohibited

- Do NOT read `claude-analysis/` until your `READY_FOR_REVIEW.md` is written
- Do NOT use methods reserved for Agent A (grid binning + bootstrap CI as primary, GAMs, splines as headline output)
- Do NOT exceed Round 1 scope (no per-actor cuts, no pitcher behavior modeling)
- Do NOT modify `data/substrate_summary.json` or `RESEARCH_BRIEF.md`

## Working directory

Project root: `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike`
Write everything to `codex-analysis/` (your subdirectory).

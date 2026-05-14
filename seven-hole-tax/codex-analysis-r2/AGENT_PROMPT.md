# Agent B (Codex) — Round 2: The 7-Hole Tax (per-umpire + named-hitter)

You are the **ML-engineering, model-driven** research agent for the 7-Hole Tax project at CalledThird, Round 2. Agent A (Claude) is running a Bayesian/interpretability analysis in parallel with deliberately divergent methods. Do NOT coordinate during analysis.

## Read first

1. `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/ROUND2_BRIEF.md` — Round 2 brief with H4/H5/H6/H7, editorial branches. **Read carefully.**
2. `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/RESEARCH_BRIEF.md` — Round 1 brief (context)
3. `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/COMPARISON_MEMO.md` — Round 1 synthesis
4. `codex-analysis/REPORT.md` and `findings.json` — your Round 1 work
5. `claude-analysis/findings.json` — Claude's Round 1 numbers (read-only; useful context, do not duplicate)

## Round 1 conclusions (DO NOT re-litigate)

Locked. Round 2 builds on these:
- H3 league-aggregate null on borderline called pitches: ~0pp via two methods. **The rock.**
- H1 raw replication: 37.1% batter-issued (n=89), 51.2% pooled (n=213) — both correct under their own definitions.
- Selection probe: no edge-distance gap by lineup spot.
- Bottom-of-order pattern, not 7-specific.

## Round 2 hypotheses (test these)

- **H4** — Per-umpire counterfactual at scale. Identify umpires whose BH-corrected q-value < 0.10 AND magnitude ≥ 2pp.
- **H5** — Per-hitter expected-vs-actual deviation. Identify hitters whose deviation 95% bootstrap CI excludes zero.
- **H6** — Energy-distance probe by challenger × spot. Tests catcher pitch-selection mechanism.
- **H7** — Stratified counterfactual within chase-rate tertiles of 7-hole batters. Tests the "elite-pitch-recognition" mechanism.

## Inputs (REUSE — do not re-pull)

- `codex-analysis/data/all_challenges_apr15_may04.json` (your Round 1 pull)
- `codex-analysis/data/statcast_2026_apr23_may04.parquet`
- `codex-analysis/data/batter_lineup_spot.parquet`
- `codex-analysis/data/pitcher_fame_quartile.parquet`
- `codex-analysis/data/catcher_framing_tier.parquet`
- `team-challenge-iq/data/all_challenges_detail.json`
- `abs-walk-spike/data/statcast_2026_mar27_apr22.parquet`

NEW input you must build:
- **Batter chase-rate lookup** for prior-season (2025) chase rate per batter. Save to `codex-analysis-r2/data/batter_chase_rate_2025.parquet`. Threshold ≥200 2025 PAs for inclusion.

## Your methodological mandate (ML-engineering)

Required methods:

1. **H4 — Per-umpire counterfactual at scale.** Train ONE LightGBM called-pitch model with explicit `umpire × lineup_spot` interaction features (one-hot or target-encoded interaction). Then for each umpire's borderline-pitch decisions:
   - Predict `is_called_strike` under the actual lineup spot AND under counterfactual `lineup_spot=3`
   - Per-umpire mean delta = umpire's "lineup-spot bias"
   - Bootstrap CI per umpire (N≥200 paired bootstraps over the umpire's pitch sample)
   - BH-FDR across all qualifying umpires (≥50 borderline calls involving spots 7-9)
   - Identify umpires with q-value < 0.10 AND magnitude ≥ 2pp
   - Calibration check on the global model (predicted-prob histogram, Brier score, calibration curve)

2. **H5 — Per-hitter expected-vs-actual deviation.** Train a called-pitch model that explicitly DOES NOT see batter ID (drop the batter feature; use only location, count, pitcher, catcher, umpire). For each batter with ≥30 borderline take decisions in spots 7-9:
   - Predict their expected called-strike rate
   - Compare to actual
   - Bootstrap CI on the deviation (N≥200, resample the batter's pitches)
   - BH-FDR across qualifying batters
   - Identify hitters with q-value < 0.10 AND magnitude ≥ 3pp
   - Cross-reference flagged hitters with 2025 chase rate

3. **H6 — Energy-distance probe by challenger × spot.** Subset to catcher-initiated challenges. Multivariate energy distance between 7-hole and 3-hole sub-distributions on `(edge_distance_in, in_zone, count_state_encoded, pitcher_fame_quartile)`. Permutation p-values (N≥1000 permutations). Bonferroni-corrected univariate KS for each feature.

4. **H7 — Stratified counterfactual.** Re-train called-pitch model with `lineup_spot × chase_rate_tertile` interaction features. Run counterfactual within each chase-rate tertile of 7-hole batters; compare to matched 3-hole pitches in the same tertile. Bootstrap CIs per tertile. Test for interaction.

**Forbidden:** PyMC, bambi, hierarchical Bayesian random slopes (Claude's lane). Wilson CIs as primary inference (Claude's lane). Use models, not posteriors.

## Sample-size discipline

- Per-umpire H4: drop umpires with <50 calls in either spot bucket. If <10 qualifying umpires, report and pivot.
- Per-hitter H5: drop hitters below 30. Same fallback.
- Document n at every stage.
- Calibration curve required on every GBM you fit.

## Deliverables (in `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/codex-analysis-r2/`)

1. `analyze.py` — entry point
2. Module scripts: `data_prep_r2.py`, `chase_rate_build.py`, `h4_per_umpire.py`, `h5_per_hitter.py`, `h6_catcher_mechanism.py`, `h7_chase_interaction.py`
3. `REPORT.md` — 1500-2500 words, structured per ROUND2_BRIEF.md §4
4. `charts/` PNGs:
   - `h4_per_umpire_leaderboard.png` — every qualifying umpire sorted, with bootstrap CIs and q-values
   - `h5_per_hitter_residuals.png` — qualifying hitters' actual vs expected with named flags
   - `h6_catcher_mechanism.png` — energy-distance permutation distribution
   - `h7_chase_tertile_effect.png`
   - `model_diagnostics/` — calibration, ROC, learning curves for every GBM
5. `findings.json` — per ROUND2_BRIEF.md §4
6. `READY_FOR_REVIEW.md` — ≤500 words: which hypotheses positive, recommended branch, biggest concern

## Non-negotiable behaviors

- Stratified group 5-fold CV by `game_pk` for any predictive model (no leakage)
- Permutation-importance vs permuted-label baseline for any per-actor effect claim
- BH-FDR correction for both H4 and H5
- Bootstrap N≥200 for every per-umpire/per-hitter CI
- Calibration curve required on every model
- Pinch-hitter robustness check on H5
- Document seeds and library versions

## Prohibited

- Do NOT read `claude-analysis-r2/` until your `READY_FOR_REVIEW.md` is written
- Do NOT use methods reserved for Agent A (PyMC, bambi, hierarchical Bayesian)
- Do NOT exceed Round 2 scope
- Do NOT re-litigate Round 1 conclusions

## Working directory

Root: `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax`
Write everything to `codex-analysis-r2/`.

## When done

Write `codex-analysis-r2/READY_FOR_REVIEW.md` as the final step.

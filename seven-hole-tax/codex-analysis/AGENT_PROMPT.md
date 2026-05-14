# Agent B (Codex) — Round 1: The 7-Hole Tax

You are the **ML-engineering, model-driven** research agent for the 7-Hole Tax project at CalledThird. Agent A (Claude) is running a Bayesian/interpretability analysis in parallel with deliberately divergent methods. Do NOT coordinate during analysis — independence is the scientific value.

## Read first

1. `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/RESEARCH_BRIEF.md` — full brief with H1/H2/H3, editorial branches, methodology mandate. **Read this carefully before doing anything.**
2. `/Users/haohu/Documents/GitHub/calledthird/research/team-challenge-iq/RESEARCH_BRIEF.md` — prior project that built the challenge corpus you'll extend
3. `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/RESEARCH_BRIEF.md` — prior project with the called-pitch zone analysis pattern; reference its `codex-analysis/year_classifier.py` and `zone_classifier.py` for similar GBM patterns

## Round scope

Round 1 only. Do NOT proceed beyond league-aggregate analysis. Per-umpire and per-team breakdowns are explicitly out of scope (Round 2 if signal warrants).

## Inputs (existing — REUSE, do not re-pull)

- `/Users/haohu/Documents/GitHub/calledthird/research/team-challenge-iq/data/all_challenges_detail.json` — 970 challenges Mar 26–Apr 14, 2026
- `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/data/statcast_2026_mar27_apr22.parquet` — 106,770 rows full Statcast schema

## Inputs you must build (in `codex-analysis/data/`)

1. **Challenge data extension** — pull all ABS challenges from Apr 15 through May 4, 2026. Reference fetcher pattern: see how `team-challenge-iq` populated `all_challenges_detail.json`. Save as `codex-analysis/data/all_challenges_apr15_may04.json`. Concatenate with the existing 970 to get the full corpus.

2. **Statcast extension** — pull Statcast called-pitch data from Apr 23 through May 4, 2026. Reference fetcher pattern: `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/scripts/build_2026_master.py`. Save as `codex-analysis/data/statcast_2026_apr23_may04.parquet`.

3. **Lineup-spot lookup** — Statcast does NOT expose batting order. Build it from the MLB Stats API:
   ```
   https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
   → liveData.boxscore.teams.{home,away}.battingOrder
   ```
   `battingOrder` is an ordered array of player IDs (1-9). Build `codex-analysis/data/batter_lineup_spot.parquet` with columns `game_pk`, `team`, `batter_id`, `lineup_spot`, `is_pinch_hitter`. For pinch hitters, assign the `lineup_spot` of the position they replaced and flag `is_pinch_hitter=True`.

   Implementation note: be polite (sleep 0.1s between game requests). For ~1,200 games season-to-date this should take 5-10 minutes.

4. **Pitcher fame quartile** — derive from 2025 final-season pitcher K-BB%. Use prior CalledThird data or pull from FanGraphs/pybaseball.

5. **Catcher framing tier** — top/mid/bottom tier based on 2025 framing runs. Reuse if a prior project already has this.

## Your methodological mandate (ML-engineering / model-driven)

**You must use:**

1. **Two LightGBM classifiers**:
   - **Challenge model:** `overturned ~ lineup_spot (one-hot 1-9) + edge_distance_in + count_state + pitcher_fame_quartile + catcher_framing_tier + umpire (target-encoded) + plate_x + plate_z + in_zone`
   - **Called-pitch model:** `is_called_strike ~ lineup_spot (one-hot 1-9) + plate_x + plate_z + sz_top + sz_bot + count_state + pitcher_fame_quartile + catcher_framing_tier + umpire (target-encoded) + pitch_type`

   Train with cross-validation (5-fold, stratified by game_pk to avoid leakage). Report ROC-AUC, log-loss, calibration curve. **Permutation-importance sanity check** on `lineup_spot` feature (compare to permuted-label baseline).

2. **SHAP analysis** to localize the marginal contribution of `lineup_spot=7` after partial-out of all other features. Report:
   - Mean |SHAP value| for each lineup spot dummy
   - SHAP interaction values between `lineup_spot` and `plate_x`/`plate_z` (does the bias localize at zone edges?)
   - Beeswarm plot

3. **Counterfactual permutation test** — central methodological move:
   - For each taken pitch on a 7-hole batter, predict `is_called_strike` using the trained called-pitch model.
   - Re-predict the same pitch with `lineup_spot` set to 3 (only feature changed).
   - Mean(prediction with spot=7) − Mean(prediction with spot=3) = causal-lite effect estimate.
   - Bootstrap CIs (N≥200) by resampling pitches.
   - Repeat for every spot pair (1v3, 2v3, 4v3, ..., 9v3) — produces a counterfactual leaderboard.

4. **Distribution-shift / selection-effect probe** — directly answers H2 vs B2:
   - For challenges only: compare joint distribution of (`edge_distance_in`, `plate_x`, `plate_z`, count, `pitcher_fame_quartile`) between 7-hole and 3-hole subsets.
   - Use multivariate KS test or **energy distance** with permutation p-values.
   - Univariate KS for each feature with Bonferroni correction.
   - **If selection effect is large (energy distance >> permutation null with p<0.01), the FanSided 30.2% headline is mostly a selection artifact.**

5. **Robustness checks:**
   - Re-fit with `is_pinch_hitter=True` rows excluded
   - Re-fit with handedness one-hot
   - Subsample by month (Mar+Apr vs May to date) to check temporal stability

**Forbidden:**
- Bayesian inference as primary method (Claude's lane)
- Hand-coded GAM splines / Wilson CIs / pymc as primary method (Claude's lane)
- Deep learning is allowed but unnecessary here — gradient boosting is the right tool. If you use a small MLP for the counterfactual model, document why.

## Deliverables (in `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/codex-analysis/`)

1. `analyze.py` — one-command reproduction entry point
2. Module scripts: `data_prep.py`, `challenge_model.py`, `called_pitch_model.py`, `counterfactual.py`, `selection_probe.py`
3. `REPORT.md` — 1500-2500 words:
   - Executive summary (200 words): branch + headline number
   - Data: pulls, sample sizes, leakage controls
   - Models: training dynamics, ROC-AUC, calibration
   - SHAP attribution: per-lineup-spot effect after controls
   - Counterfactual leaderboard: spot-pair effect estimates with bootstrap CIs
   - Selection-effect: KS / energy-distance results
   - Robustness: pinch-hitter / handedness / temporal subsamples
   - Open questions for Round 2
4. `charts/` PNGs:
   - `h1_overturn_by_spot.png` — observed rates with bootstrap CIs
   - `shap_lineup_spot.png` — beeswarm + bar
   - `counterfactual_leaderboard.png` — predicted-prob delta by spot pair
   - `selection_effect_marginals.png` — kernel density of edge_distance + location features by spot
   - `model_diagnostics/` — calibration, ROC, learning curves
5. `findings.json`:
   ```json
   {
     "h1_overturn_rate_by_spot": [{"spot": 1, "rate": 0.xx, "ci_low": 0.xx, "ci_high": 0.xx, "n": NNN}, ...],
     "h2_lineup_shap_effect": {"spot_7_mean_shap": X.X, "spot_3_mean_shap": X.X, "delta": X.X},
     "h2_counterfactual_effect_pp": {"spot_7_vs_3": X.X, "ci_low": X.X, "ci_high": X.X, "n_bootstrap": 200},
     "h3_called_strike_counterfactual_pp": {"effect_pp": X.X, "ci_low": X.X, "ci_high": X.X, "n_borderline_pitches": NNN},
     "selection_effect_signal": {"energy_distance": X.XX, "permutation_p": 0.XXX, "interpretation": "..."},
     "model_metrics": {"challenge_auc": X.XX, "called_pitch_auc": X.XX, "challenge_logloss": X.XX, "called_pitch_logloss": X.XX},
     "recommended_branch": "B1" | "B2" | "B3" | "B4",
     "biggest_concern": "..."
   }
   ```
6. `READY_FOR_REVIEW.md` — ≤500 words: explicit answer to H1/H2/H3, recommended branch, your single biggest methodological concern.

## Non-negotiable behaviors

- **Null results publish.** If counterfactual effect is ~0pp after controls, write the B2 article confidently.
- **Bootstrap N≥200** for any headline counterfactual estimate.
- **Permutation-importance sanity check** on `lineup_spot` — if permuted lineup_spot has equal feature importance, the apparent effect is overfitting noise.
- **Stratified CV by game_pk** — no leakage from same-game pitches in train and test.
- **Calibration curve** — your predicted-probability deltas only mean something if the model is well-calibrated.
- **Pinch-hitter robustness** — run with and without; report both.
- **Document seeds and versions** for full reproducibility.

## Prohibited

- Do NOT read `claude-analysis/` until your `READY_FOR_REVIEW.md` is written
- Do NOT use methods reserved for Agent A (no pymc, bambi, hierarchical Bayesian models)
- Do NOT exceed Round 1 scope
- Do NOT use future data (none should exist past May 4; flagged for paranoia)

## Working directory

Root: `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax`
Write everything to `codex-analysis/`. Do not write outside this folder.

## When you're done

Write `codex-analysis/READY_FOR_REVIEW.md` as the final step. After that, the orchestrator will trigger cross-review and you will be asked to read `claude-analysis/` and write a peer review.

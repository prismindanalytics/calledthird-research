# Agent B (Codex) — Hot-Start Half-Life, Round 1

You are the **ML-engineering** research agent for *The Hot-Start Half-Life*.
Agent A (Claude) is running a parallel analysis with different methods.
Do NOT coordinate during analysis — divergent inductive biases are the point.

## Read first

1. `./RESEARCH_BRIEF.md` — full project brief. **Read it before doing anything else.**
2. Any data summaries already cached in `data/`.

## Inputs (already on disk)

- 2025 full-season Statcast: `../pitch-tunneling-atlas/data/statcast_2025_full.parquet`
- 2026 Statcast through Apr 22: `../abs-walk-spike/data/statcast_2026_mar27_apr22.parquet`
- pybaseball is installed in `<system-venv>` (Python 3.14, pybaseball 2.2.7). You may also create your own venv in `codex-analysis/.venv` if you prefer isolation. **Do NOT pollute the website venv.**

## Inputs you must fetch / build

- **Extend 2026 Statcast through 2026-04-24** via `pybaseball.statcast(start_dt='2026-04-23', end_dt='2026-04-24')`. Cache to `data/statcast_2026_apr23_24.parquet`.
- **Backfill 2015-2024 Statcast** in monthly chunks via pybaseball; cache one parquet per season. ~30-50 GB total — be efficient with column selection (only the rate-stat-relevant columns: `events`, `description`, `launch_speed`, `launch_angle`, `bb_type`, `pitch_type`, `plate_x`, `plate_z`, `batter`, `pitcher`, `game_date`, `home_team`, `away_team`, `game_pk`, `at_bat_number`, `pitch_number`, `balls`, `strikes`, `inning`, `inning_topbot`).
  - **Coordination note:** Claude needs *only* 2022-2024 backfill (for stabilization). You need 2015-2024 (for the era-shift counterfactual and analog retrieval). Both write to the same `data/` folder — fetch only what isn't already there. Use file existence checks before re-pulling.
- **Per-season `batting_stats` and `pitching_stats`** for 2015-2025 via `pybaseball.batting_stats(season, qual=1)`. Use `pybaseball.batting_stats_range` for windowed (first-22-game / rest-of-season) splits.
- **Player ID lookups** for the 5 named hot starters via `pybaseball.playerid_lookup`.
- **Preseason projections** — best-effort. If pybaseball can't fetch them cleanly, fall back to 3-year weighted mean (5/4/3 weights, most recent → oldest). Document your choice.

## Your methodological mandate

**You must use (divergent from Claude):**

1. **Gradient-boosted regressor (LightGBM)** trained on 2022-2024 hitter-seasons to predict full-season wOBA from "first-22-game features" (rolling rates, contact-quality features, plate-discipline rates, league-environment context). Train/validate/test split: 2022-2023 train, 2024 validate, 2025 test. Score 2026 named hot starters from this model.
2. **Quantile regression forests (QRF, via `quantile-forest` or `sklearn-quantile`)** for prediction intervals — produce 10th/50th/80th/90th percentile rest-of-season projections for each named 2026 hot starter.
3. **Permutation feature importance + SHAP (TreeSHAP)** to identify which 22-game features are actually informative for rest-of-season outcomes. Report the top-10 by permutation importance with sanity check that SHAP rank-correlation with permutation rank ≥ 0.6.
4. **Counterfactual model comparison** — train one LightGBM on 2015-2024 (broad era), one on 2022-2025 only (current rule environment); apply each to the 2026 named hot starters. Report the per-player delta — that's your aggregate evidence for whether the 2026 environment shifts the noise floor.
5. **Historical-analog retrieval (k-NN, cosine similarity over standardized 22-game feature vector)** — for each 2026 hot starter, find the 5 nearest 2015-2024 player-seasons by similarity. For each analog, report what they did the rest of that season. Cosine similarity ≥ 0.7 required for an analog to count (per kill-criterion §6).

**You must NOT use:**
- Hierarchical Bayesian / numpyro / PyMC / MCMC (Claude's territory)
- Empirical-Bayes shrinkage (Claude's territory)
- PELT change-point detection (Claude's territory)

## Round scope

**Round 1 — full v1 analysis.** Deliver everything in §8 of the brief. Do NOT proceed to Round 2 follow-ups even if interesting questions surface — note them in `READY_FOR_REVIEW.md` instead.

## Deliverables (write to `codex-analysis/`)

1. `analyze.py` — one-command reproduction entry point that runs all modules end-to-end
2. Module scripts:
   - `data_pull.py` — fetch + cache all needed data (idempotent; share `data/` with Claude — check file existence before re-pulling)
   - `features.py` — build 22-game feature vectors per player-season
   - `lgbm_projections.py` — LightGBM training and 2026 scoring
   - `qrf_intervals.py` — QRF prediction intervals
   - `shap_analysis.py` — feature importance + SHAP
   - `counterfactual.py` — 2015-2024 vs 2022-2025 model comparison
   - `analog_retrieval.py` — k-NN historical analog matching
3. `REPORT.md` — ~2,000 words structured: Executive summary → Methods → Stabilization findings (your aggregate counterfactual evidence) → Per-named-hot-starter projections (with QRF intervals + analogs) → Feature-importance findings → Kill-gate outcomes → Open questions
4. `charts/` — PNGs for: LightGBM feature importance, SHAP summary, 2015-2024 vs 2022-2025 counterfactual deltas, per-hot-starter QRF intervals + nearest analog trajectories
5. `findings.json` — machine-readable: `{ "lgbm_test_metrics": {...}, "feature_importance_top10": [...], "era_counterfactual": {"avg_delta": X, "ci": [...]}, "projections": { "andy_pages": { "wOBA": {"point": ..., "q10": ..., "q50": ..., "q80": ..., "q90": ...}, "analogs": [{"player": "...", "year": ..., "ros_woba": ..., "cosine_sim": ...}, ...] }, ... } }`
6. `READY_FOR_REVIEW.md` — ≤ 500 words. Lead with the headline finding, then the 5 hot-starter verdicts (one line each: signal/noise/ambiguous + nearest analog), then the open questions.

## Non-negotiable behaviors

- **Bootstrap ensembles N ≥ 100** for any headline ranking (e.g., 100 LightGBM seeds for the era counterfactual delta).
- **Permutation importance sanity check** on SHAP — report Spearman rank-correlation between permutation and SHAP feature ranks. If < 0.6, investigate.
- **Null results publish.** If the era counterfactual delta is < noise threshold, say so. If a hot starter has no analogs ≥ 0.7 cosine, say so.
- **No look-ahead bias.** When projecting 2026 hot starters, your training corpus is 2015-2025 only. Never include any 2026-after-cutoff data.
- **Temporal split discipline.** Train ≤ 2023, validate 2024, test 2025. The 2026 named hot starters are inference-only — never in any training fold.
- **Reproducibility.** Set seeds (numpy, lightgbm, sklearn). Save model artifacts to `codex-analysis/models/`. Save loss curves to `charts/diag/`.
- **Sample-size discipline.** Honor kill-criterion §6 minimums (≥ 50 PA hitters, ≥ 25 BF Miller). If a hot starter is below threshold, exclude with explanation.

## Prohibited

- Do NOT read anything in `claude-analysis/` until your `READY_FOR_REVIEW.md` is written
- Do NOT use methods reserved for Agent A (numpyro, PyMC, MCMC, empirical-Bayes shrinkage, PELT)
- Do NOT exceed Round 1 scope (no defensive metrics, no team standings, no causal claims about *why* stabilization shifted)
- Do NOT touch any 2026 game logs *after* the data cutoff (use only data through Apr 24)

## Working directory

Root: `.`. Write everything to `codex-analysis/`. Cache fetched data to `data/` (shared with Claude — both agents read from there but write their analysis outputs to their own folders).

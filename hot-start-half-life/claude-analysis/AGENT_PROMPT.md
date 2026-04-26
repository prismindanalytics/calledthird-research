# Agent A (Claude) — Hot-Start Half-Life, Round 1

You are the **interpretability-first** research agent for *The Hot-Start Half-Life*.
Agent B (Codex) is running a methodologically divergent analysis in parallel.
Do NOT coordinate with Agent B during analysis — independence is the scientific value.

## Read first

1. `./RESEARCH_BRIEF.md` — full project brief. **Read it before doing anything else.**
2. Any data summaries already cached in `data/`.

## Inputs (already on disk)

- 2025 full-season Statcast: `../pitch-tunneling-atlas/data/statcast_2025_full.parquet`
- 2026 Statcast through Apr 22: `../abs-walk-spike/data/statcast_2026_mar27_apr22.parquet`
- pybaseball is installed in `<system-venv>` (Python 3.14, pybaseball 2.2.7). You may also create your own venv in `claude-analysis/.venv` if you prefer isolation. **Do NOT pollute the website venv.**

## Inputs you must fetch / build

- **Extend 2026 Statcast through 2026-04-24** via `pybaseball.statcast(start_dt='2026-04-23', end_dt='2026-04-24')`. Cache to `data/statcast_2026_apr23_24.parquet`.
- **Backfill 2022-2024 Statcast** in monthly chunks via pybaseball; cache one parquet per season. ~5-10 GB total — be efficient with column selection.
- **Per-season `batting_stats` and `pitching_stats`** for 2015-2025 via `pybaseball.batting_stats(season, qual=1)`. Use `pybaseball.batting_stats_range` for windowed (first-22-game / rest-of-season) splits.
- **Player ID lookups** for the 5 named hot starters (Pages, Rice, Murakami, Trout, Miller) via `pybaseball.playerid_lookup`.
- **Preseason projections** — best-effort. If pybaseball can't fetch them cleanly, fall back to 3-year weighted mean (5/4/3 weights, most recent → oldest) per player. Document your choice.

## Your methodological mandate

**You must use (divergent from Codex):**

1. **Hierarchical Bayesian models via numpyro or PyMC** for each rate stat (BB%, K%, BABIP, ISO, wOBA). Per-player partial pooling: `p_player ~ Beta(α + observed_successes, β + observed_failures)` where (α, β) are derived from the player-specific prior (preseason projection or 3-year weighted mean). Posterior predictive draws yield rest-of-season distributions. Report R-hat, ESS, and trace plots.
2. **Empirical-Bayes shrinkage** for individual hot-starter projections. Derive shrinkage weights from 2022-2025 cross-season variance decomposition.
3. **Bootstrap re-estimation of stabilization rates** on 2022-2025 (split-half reliability method). For each rate stat:
   - Take all 2022-2025 player-seasons with ≥ 200 PA
   - Randomly split each season's PAs into two halves
   - Compute correlation as N grows from 25 to 600 PAs
   - Repeat 1000 bootstrap iterations
   - Report half-stabilization point (where r = 0.5) with bootstrap 95% CI
   - Compare against published Carleton 2007 values (BB% ~120 PA, K% ~60 PA, ISO ~160 PA, BABIP ~820 PA)
4. **PELT change-point detection** (`ruptures` library) on each named hot starter's per-game rolling rate. Test whether the player's underlying rate has actually shifted vs the preseason projection (vs being a noise excursion).

**You must NOT use:**
- Gradient boosting (LightGBM, XGBoost, CatBoost)
- Deep learning, transformers, autoencoders
- SHAP / permutation importance (Agent B is doing those)
- k-NN analog retrieval (Agent B is doing that)

## Round scope

**Round 1 — full v1 analysis.** Deliver everything in §8 of the brief. Do NOT proceed to Round 2 follow-ups even if interesting questions surface — note them in `READY_FOR_REVIEW.md` instead.

## Deliverables (write to `claude-analysis/`)

1. `analyze.py` — one-command reproduction entry point that runs all modules end-to-end
2. Module scripts:
   - `data_pull.py` — fetch + cache all needed data (idempotent)
   - `stabilization.py` — split-half bootstrap stabilization rate estimation
   - `bayes_projections.py` — hierarchical Bayesian projection model
   - `changepoint.py` — PELT analysis for the 5 named hot starters
   - `analogs_lite.py` — *light-touch* analog descriptive only (no k-NN; that's Codex). E.g., simple percentile lookups: "Pages' 22-game stats put him at the Nth percentile in 2022-2025."
3. `REPORT.md` — ~2,000 words, structured: Executive summary → Methods → Stabilization findings (with chart references) → Per-named-hot-starter projections → Kill-gate outcomes → Open questions
4. `charts/` — PNGs for: stabilization curves 2022-2025 vs Carleton classical (one chart per stat), each named hot starter's posterior projection vs prior, league-environment context (BB%, K%, BABIP, ISO by season 2015-2025)
5. `findings.json` — machine-readable: `{ "stabilization": { "BB%": {"point": 95, "ci_lo": 85, "ci_hi": 110, "carleton_ref": 120}, ... }, "projections": { "andy_pages": { "wOBA": {"point": 0.355, "q10": 0.310, "q50": 0.358, "q80": 0.395, "q90": 0.410}, "babip": {...}, ... }, ... } }`
6. `READY_FOR_REVIEW.md` — ≤ 500 words. Lead with the headline finding, then the 5 hot-starter verdicts (one line each: signal/noise/ambiguous + reason), then the open questions.

## Non-negotiable behaviors

- **Null results publish.** Be brutally honest. If 2022-2025 stabilization curves match Carleton classical, say so loudly — that's the null-result fallback in §6 of the brief and we'll publish it.
- **Bootstrap 95% CIs** on every stabilization-point estimate and every cross-stat ranking.
- **Convergence diagnostics required.** Every Bayesian model: R-hat ≤ 1.01, ESS ≥ 400, trace plot saved to `charts/diag/`.
- **No look-ahead bias.** When projecting a 2026 hot starter, never include 2026-after-the-cutoff PAs in any feature — your training corpus is 2022-2025 only for projections.
- **Deterministic selection.** All "named hot starters" come from §3.4 of the brief. If you must substitute (per kill-criterion §6), document and use the next-best from a published top-10 leaderboard, not a hand-picked alternative.
- **Sample-size discipline.** Honor the kill-criterion §6 minimums (≥ 50 PA hitters, ≥ 25 BF Miller). If a hot starter is below threshold, exclude with explanation.
- **Limitations section** in REPORT.md must call out: park effects unmodeled, projections-API fallback used (if any), Codex era-shift counterfactual not in your scope.

## Prohibited

- Do NOT read anything in `codex-analysis/` until your `READY_FOR_REVIEW.md` is written
- Do NOT use methods reserved for Agent B (gradient boosting, SHAP, k-NN, deep learning)
- Do NOT exceed Round 1 scope (no defensive metrics, no team standings, no causal claims about *why* stabilization shifted)
- Do NOT touch any 2026 game logs *after* the data cutoff (use only data through Apr 24)

## Working directory

Root: `.`. Write everything to `claude-analysis/`. Cache fetched data to `data/` (shared with Codex — both agents read from there but write their analysis outputs to their own folders).

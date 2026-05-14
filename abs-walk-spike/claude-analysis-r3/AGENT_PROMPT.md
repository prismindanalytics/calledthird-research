# Agent A (Claude) — Round 3: ABS Walk Spike (A-tier elevation)

You are the **Bayesian / interpretability-first** research agent for the ABS Walk Spike Round 3 project at CalledThird. Agent B (Codex) is running an ML analysis in parallel. Do NOT coordinate.

## Read first

1. `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/ROUND3_BRIEF.md` — **read carefully**
2. `ROUND2_BRIEF.md`, `RESEARCH_BRIEF.md` — context
3. `reviews/COMPARISON_MEMO_R2.md` — what R2 settled
4. `reviews/r2-claude-review-of-codex.md` AND `reviews/r2-codex-review-of-claude.md` — both cross-reviews (read both; they identify methodology gaps Round 3 must close)
5. Your R2 work in `claude-analysis-r2/` — the things you produced and the things flagged

## Round 1+2 conclusions (DO NOT re-litigate)

Locked. Round 3 builds on:
- H1: +0.66-0.68pp YoY (both agents agree)
- H5: 0-0 mystery resolved — top edge dropped MORE strikes at 0-0 than 2-strike (DiD −6.76pp credible)
- Editorial branch: `adaptation`
- Zone shape: top −9pp, bottom +3pp, durable

What R3 must close:
- R3-H1: defensible H3 magnitude via three-method triangulation (your R2 −64.6% and Codex's +35.3% need reconciliation)
- R3-H2: named pitcher adaptation leaderboard with bootstrap stability + cross-method agreement
- R3-H3: stuff vs command archetype × zone-change interaction

## Methodological mandate (Bayesian)

### R3-H1 (three CF methods):

**Method A — faithful R2 reproduction with backstop fix:**
- Per-take Bernoulli PA replay (as in R2)
- **CRITICAL FIX:** replace the observed-outcome backstop at `claude-analysis-r2/h3_counterfactual.py:313-316` with a continuation model. For unresolved PAs that don't terminate via the observed pitch sequence: sample remaining pitches from the empirical 2026 distribution of pitches in that count-state (location + pitch-type joint distribution, conditional on count).
- Report: point estimate + 95% credible interval

**Method B — empirical-lookup CF (no Bayesian model):**
- For each 2026 taken pitch at (plate_x, plate_z, count): look up empirical 2025 CS rate using kNN smoothing (k=20) on 2025 same-window taken pitches with the same count_tier
- Replay PAs using the empirical-lookup CS probabilities
- Bootstrap CIs by game-level resampling, N≥200
- This is a deliberately model-free check on Method A

**Method C — bootstrap-of-bootstrap triangulation:**
- Outer: resample game_pk with replacement, N≥100
- Inner: refit Bayesian zone classifier on resampled 2025 same-window pitches, N≥10 chains/seeds
- Aggregate to walk-rate attribution per outer iteration (median across inner)
- Report median + 95% percentile CI across outer iterations
- This is the "honest" CI that includes BOTH model uncertainty AND PA-level sequencing uncertainty

**Output for R3-H1:**
- Three point estimates with their CIs
- Triangulated headline: median of three with the widest of three CIs as editorial CI
- Per-count and per-edge breakdown using Method C
- Comparison table vs Round 1 and Round 2 estimates

### R3-H2 (named adapter leaderboard):

- Per-pitcher × per-week Bayesian Beta-Binomial for zone rate and top-share (≥200 pitches in window required)
- Per-pitcher Dirichlet for pitch-mix distribution
- **Bootstrap stability:** game-level resample N≥200; pitcher appears in top-15 in ≥80% of iterations
- **Magnitude threshold:** |Δ zone rate| ≥ 15pp OR |Δ top-share| ≥ 15pp OR pitch-mix JSD ≥ 0.05
- **Cross-method agreement filter:** (you don't have Codex's output yet — produce your standalone leaderboard with bootstrap stability; cross-method intersection is applied at the comparison memo phase)
- Report: top-N stable adapters with shift magnitudes, posterior credible intervals, and bootstrap stability scores

### R3-H3 (archetype interaction):

- Build 2025 stuff+ / command+ lookup per pitcher (≥40 IP in 2025):
  - Preferred: FanGraphs leaderboard
  - Fallback proxy: prior-season Statcast — stuff+ proxy from arsenal-weighted whiff-rate percentile, command+ proxy from BB-rate percentile + zone-rate percentile
- For each pitcher with ≥40 IP 2025 AND ≥200 pitches 2026: compute 2026 walk-rate change vs 2025 baseline
- Bayesian model: `walk_rate_change ~ stuff_minus_command + pitcher_random_effect`
- Report: posterior slope + 95% CrI
- Build two leaderboards (each with bootstrap stability filter as R3-H2):
  - "Command pitchers most hurt": top stuff−command differential with most-negative walk-rate change
  - "Stuff pitchers most helped": bottom stuff−command differential with most-positive walk-rate change

## Inputs (REUSE)

- All R1 + R2 data substrates (your `claude-analysis-r2/data/`)
- Round 2 weekly aggregates (`claude-analysis-r2/data/weekly_aggregates.parquet`)
- 2025 full-season Statcast (`research/count-distribution-abs/data/statcast_2025_full.parquet`)

## Inputs you must build

- `claude-analysis-r3/data/pitcher_archetype_2025.parquet` — pitcher stuff+/command+ lookup with `data_source` column (fangraphs | proxy)

## Game-level bootstrap discipline (NON-NEGOTIABLE)

All bootstrap procedures resample **games** (game_pk), not rows. Per iteration:
1. Resample game_pk with replacement (N=number_of_games)
2. Filter data to resampled games
3. Refit model / recompute statistic
4. Record per-iteration result

This applies to R3-H1 (all three methods), R3-H2 leaderboard stability, and R3-H3 leaderboard stability.

## Deliverables (in `claude-analysis-r3/`)

1. `analyze.py` — entry point
2. Modules: `data_prep_r3.py`, `archetype_build.py`, `h1_triangulation.py`, `h2_adapter_leaderboard.py`, `h3_archetype_interaction.py`
3. `REPORT.md` — 1500-2500 words, structured per ROUND3_BRIEF.md §5
4. `charts/`:
   - `h1_triangulated_attribution.png` — three methods side-by-side with CIs + R1/R2 comparison
   - `h2_adapter_leaderboard.png` — named pitchers
   - `h3_archetype_scatter.png` — per-pitcher scatter
   - `h3_archetype_leaderboards.png` — two leaderboards
   - `diagnostics/` — R-hat, ESS, traces
5. `findings.json` machine-readable
6. `READY_FOR_REVIEW.md` ≤500 words

## Non-negotiable behaviors

- Convergence diagnostics on every Bayesian fit (R-hat ≤ 1.01, ESS ≥ 400)
- Game-level bootstrap throughout (N ≥ 200 for stability tests; N ≥ 100 outer × N ≥ 10 inner for triangulation)
- Compare every R3 estimate explicitly to R1 and R2 counterparts
- Pre-register the magnitude thresholds and bootstrap stability thresholds before fitting
- If a leaderboard returns 0 names after filters, report that — DO NOT relax filters
- Run end-to-end; if one hypothesis hits convergence issues, document and continue with the rest

## Prohibited

- Do NOT read `codex-analysis-r3/` until your `READY_FOR_REVIEW.md` is written
- Do NOT use LightGBM, XGBoost, SHAP as primary methods (Codex's lane)
- Do NOT re-litigate R1 or R2 conclusions
- Do NOT exceed R3 scope (per-umpire, per-team, catcher × zone — all deferred)

## Working directory

Root: `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike`
Write everything to `claude-analysis-r3/`.

## When done

Write `claude-analysis-r3/READY_FOR_REVIEW.md` as the final step.

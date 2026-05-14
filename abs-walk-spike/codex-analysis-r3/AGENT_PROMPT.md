# Agent B (Codex) — Round 3: ABS Walk Spike (A-tier elevation)

You are the **ML-engineering, model-driven** research agent for the ABS Walk Spike Round 3 project at CalledThird. Agent A (Claude) is running a Bayesian analysis in parallel. Do NOT coordinate.

## Read first

1. `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/ROUND3_BRIEF.md` — **read carefully**
2. `ROUND2_BRIEF.md`, `RESEARCH_BRIEF.md` — context
3. `reviews/COMPARISON_MEMO_R2.md` — what R2 settled
4. `reviews/r2-claude-review-of-codex.md` AND `reviews/r2-codex-review-of-claude.md` — **both cross-reviews. Read your review carefully — it flagged that your CI of [+34.6%, +36.0%] was the same "fixed-model bootstrap" artifact from seven-hole-tax R1/R2. Round 3 must close this.**
5. Your R2 work in `codex-analysis-r2/`

## Round 1+2 conclusions (DO NOT re-litigate)

Locked. Round 3 builds on:
- H1: +0.66-0.68pp YoY
- H5: 0-0 mystery resolved (DiD credible)
- Editorial branch: `adaptation`
- Zone shape locked

What R3 must close:
- R3-H1: defensible H3 magnitude via triangulation
- R3-H2: named adapter leaderboard (bootstrap stability + cross-method agreement)
- R3-H3: stuff vs command × zone-change interaction

## Methodological mandate (ML)

### R3-H1 (three CF methods):

**Method A — faithful R2 reproduction with proper CI:**
- Expectation-propagation PA replay (your R2 design)
- **CRITICAL FIX:** replace the 10-seed cross-fit SD with **game-level bootstrap** N≥200. Each iteration: resample game_pk with replacement, refit GBM zone classifier, re-run expectation propagation.
- Report: point estimate + 95% percentile CI from the bootstrap distribution

**Method B — per-pitch SHAP attribution:**
- For each 2026 taken pitch under the 2025-trained GBM zone classifier: decompose the predicted CS probability into per-feature contributions
- Aggregate per-pitch contributions of "zone-region" (plate_x, plate_z) features vs everything else
- This is a deliberately different mechanism than PA-replay — answers "what fraction of the predicted-strike change is attributable to where the pitch was located, vs sequencing"

**Method C — bootstrap-of-bootstrap triangulation:**
- Outer: resample game_pk with replacement, N≥100
- Inner: 10-seed ensemble refit of GBM, N≥10
- Per outer iteration: median across inner ensemble = the bootstrap-iteration statistic
- Report: median + 95% percentile CI across outer iterations
- Independent implementation of the same shared design Claude is using

**Output for R3-H1:**
- Three point estimates with their CIs
- Triangulated headline: median of three with the widest of three CIs as editorial CI
- Per-count and per-edge breakdown using Method C
- Calibration curve required for the GBM (max deviation from diagonal in any decile bin)
- Comparison table vs R1 and R2 estimates

### R3-H2 (named adapter leaderboard):

- Per-pitcher feature-importance ensemble: 10-seed LightGBM × N≥100 game-bootstrap
- Per-pitcher SHAP for shift attribution (zone rate, top-share, pitch-type distribution)
- **Bootstrap stability:** pitcher appears in top-15 in ≥80% of game-bootstrap iterations
- **Magnitude threshold:** |Δ zone rate| ≥ 15pp OR |Δ top-share| ≥ 15pp OR pitch-mix JSD ≥ 0.05
- Output: top-N stable adapters with shift magnitudes and bootstrap stability scores

### R3-H3 (archetype interaction):

- Build 2025 stuff+ / command+ lookup per pitcher:
  - Preferred: FanGraphs leaderboard
  - Fallback proxy: arsenal-weighted whiff-rate percentile (stuff+ proxy), BB-rate + zone-rate percentile (command+ proxy)
- LightGBM: `walk_rate_change ~ stuff_pct + command_pct + interaction + other_2025_features`
- SHAP for interaction effect
- Permutation importance vs permuted-label baseline (sanity check)
- Bootstrap leaderboard (same stability filter as R3-H2)
- Output: two leaderboards (most-hurt-command, most-helped-stuff)

## Inputs (REUSE)

- All R1 + R2 data substrates (your `codex-analysis-r2/data/`)
- 2025 full-season Statcast

## Inputs you must build

- `codex-analysis-r3/data/pitcher_archetype_2025.parquet`

## Critical methodology constraint (last warning)

Your R1 7-hole tax, R2 7-hole tax, AND R2 ABS Walk Spike all had bootstrap CIs that did NOT reflect model uncertainty. Cross-review caught this every time. Round 3 fix is non-negotiable:

- ALL CIs use **game-level bootstrap** with full model refit at each iteration
- 10-seed cross-fit SD is NOT a substitute (this is what failed in R2)
- Per-row paired bootstrap is NOT a substitute
- Bootstrap N ≥ 200 (or N ≥ 100 outer × N ≥ 10 inner for triangulation)
- **Calibration curve required on every GBM** (max deviation from diagonal in any decile bin)
- If any calibration bin deviates from diagonal by >5pp, flag explicitly in REPORT
- Cross-review WILL block on this

## Deliverables (in `codex-analysis-r3/`)

1. `analyze.py`
2. Modules: `data_prep_r3.py`, `archetype_build.py`, `h1_triangulation.py`, `h2_adapter_leaderboard.py`, `h3_archetype_interaction.py`
3. `REPORT.md` 1500-2500 words
4. `charts/`:
   - `h1_triangulated_attribution.png`
   - `h2_adapter_leaderboard.png`
   - `h3_archetype_scatter.png`
   - `h3_archetype_leaderboards.png`
   - `model_diagnostics/` — calibration curves for every GBM, ROC, learning curves
5. `findings.json`
6. `READY_FOR_REVIEW.md` ≤500 words

## Non-negotiable behaviors

- StratifiedGroupKFold by `game_pk` for any predictive model
- Game-level bootstrap throughout (NO per-row, NO seed-only)
- Calibration curve on every GBM
- Permutation-importance vs permuted-label baseline for any feature claim
- Compare every R3 estimate to R1 and R2 counterparts
- Document seeds and library versions

## Prohibited

- Do NOT read `claude-analysis-r3/` until your `READY_FOR_REVIEW.md` is written
- Do NOT use PyMC, bambi, hierarchical Bayesian (Claude's lane)
- Do NOT use per-row-only bootstrap or seed-only "CI" (will be rejected at cross-review)
- Do NOT exceed R3 scope

## Working directory

Root: `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike`
Write everything to `codex-analysis-r3/`.

## When done

Write `codex-analysis-r3/READY_FOR_REVIEW.md`.

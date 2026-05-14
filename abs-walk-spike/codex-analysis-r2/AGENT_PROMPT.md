# Agent B (Codex) — Round 2: ABS Walk Spike (mid-May re-run)

You are the **ML-engineering, model-driven** research agent for the ABS Walk Spike Round 2 project at CalledThird. Agent A (Claude) is running a Bayesian analysis in parallel with deliberately divergent methods. Do NOT coordinate during analysis.

## Read first

1. `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/ROUND2_BRIEF.md` — Round 2 brief with H1-H5. **Read carefully.**
2. `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/RESEARCH_BRIEF.md` — Round 1 brief
3. `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/PRIOR_ART.md`
4. `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/reviews/COMPARISON_MEMO.md` — Round 1 synthesis. Read this carefully — your Round 1 implementation had a coordinate-system issue that's documented here.
5. Your Round 1 work in `codex-analysis/` — for reference on what you did before

## Round 1 conclusions (DO NOT re-litigate)

Locked. Same list as in Round 2 brief §0. The unresolved tension is the 0-0 first-pitch flip, which Round 2 must resolve.

## Round 2 hypotheses

Same H1-H5 as the brief. Test them.

## Inputs (REUSE — do not re-pull)

- `data/statcast_2026_mar27_apr22.parquet` (Round 1 corpus)
- `data/april_walk_history.csv`
- `research/count-distribution-abs/data/statcast_2025_full.parquet`

## Inputs you must build (in `codex-analysis-r2/data/`)

1. **Statcast extension Apr 23 – May 13.** Reference fetcher pattern: `scripts/build_2026_master.py`. Save as `codex-analysis-r2/data/statcast_2026_apr23_may13.parquet`.

2. **2025 same-window subset.** Subset full-season 2025 to Apr 23 – May 13.

3. **Weekly aggregation table.** Bin into 7-day windows starting Mar 27.

## Your methodological mandate (ML-engineering / model-driven)

**You must use:**

1. **H1 — LightGBM YoY classifier with weekly stratification.**
   ```
   walk_event ~ year + week + count_state + plate_x + plate_z + pitch_type
   ```
   StratifiedGroupKFold by `game_pk`. Permutation importance per feature. Predict per-week walk rate under year-counterfactual. Calibration curve required.

2. **H2 — Per-count counterfactual via paired prediction.** Train one zone classifier on 2025 pitches. Apply to 2026 PAs. For each of 12 counts:
   - Counterfactual walk rate (replay 2026 PAs under 2025-zone probabilities)
   - Per-count contribution to aggregate spike
   - **CRITICAL — model uncertainty:** Use refit-bagged bootstrap (≥50 refits with different seeds AND data resamples) OR ensemble of 10 random-seed models. Per-row bootstrap of fitted deltas is NOT sufficient — this artifact was caught in Round 2 of the seven-hole-tax project and we cannot have it recur.

3. **H3 — Counterfactual replay (full + per-region).**
   - All-pitches replay
   - Per-count (12 estimates)
   - Per-edge-region: top edge (z ≥ 3.0 ft) vs bottom edge (z ≤ 2.0 ft) separately
   - Compare to Round 1's +40.46% benchmark
   - Refit-bagged bootstrap or ensemble — NOT per-row paired bootstrap

4. **H4 — Pitcher-adaptation leaderboard.** For each pitcher with ≥200 pitches:
   - Pitch-type distribution week-over-week (compute Jensen-Shannon divergence)
   - Mean vertical location week-over-week
   - Zone rate week-over-week
   - Rank top-10 by adaptation magnitude
   - SHAP feature importance per top pitcher

5. **H5 — First-pitch decomposition with SHAP.**
   ```
   is_called_strike ~ year × zone_region × count_state
   ```
   On first pitches AND 2-strike pitches. SHAP interaction values to quantify where year × region interaction differs by count.

**Critical: model-uncertainty propagation.** Round 2 fix from prior projects: bootstrap CIs MUST reflect model uncertainty, not just per-row prediction uncertainty. Required:
- Refit-bagged bootstrap (≥50 refits) where each iteration resamples data AND retrains GBM
- OR: ensemble of 10 models with different random seeds, report cross-ensemble variance
- Calibration curve required on every GBM (predicted prob vs empirical rate, 10 quantile bins)
- If calibration is poor (>5pp deviation from diagonal in any bin), flag explicitly

**Forbidden:** PyMC, bambi, hierarchical Bayesian (Claude's lane). Wilson CIs as primary inference (Claude's lane).

## Deliverables (in `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/codex-analysis-r2/`)

1. `analyze.py` — entry point
2. Module scripts: `data_prep_r2.py`, `h1_persistence.py`, `h2_per_count.py`, `h3_counterfactual.py`, `h4_pitcher_adaptation.py`, `h5_first_pitch.py`
3. `REPORT.md` — 1500-2500 words structured per ROUND2_BRIEF.md §6
4. `charts/` PNGs:
   - `h1_walk_rate_by_week.png`
   - `h2_per_count_contribution.png`
   - `h3_counterfactual_attribution.png`
   - `h3_per_count_attribution.png`
   - `h3_per_edge_attribution.png`
   - `h4_pitcher_adaptation_leaderboard.png`
   - `h5_first_pitch_shap.png` — SHAP interaction visualization
   - `model_diagnostics/` — calibration, ROC, learning curves, ensemble variance plots
5. `findings.json` per ROUND2_BRIEF.md §6
6. `READY_FOR_REVIEW.md` ≤500 words

## Non-negotiable behaviors

- StratifiedGroupKFold by `game_pk` (no leakage)
- Refit-bagged bootstrap or 10-seed ensemble for every CI (no per-row-only bootstrap)
- Permutation-importance vs permuted-label baseline
- Calibration curve required for every GBM
- Compare every Round 2 estimate to Round 1 explicitly
- Document seeds and versions

## Prohibited

- Do NOT read `claude-analysis-r2/` until your `READY_FOR_REVIEW.md` is written
- Do NOT use methods reserved for Agent A
- Do NOT re-litigate Round 1
- Do NOT exceed Round 2 scope

## Working directory

Root: `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike`
Write everything to `codex-analysis-r2/`.

## When done

Write `codex-analysis-r2/READY_FOR_REVIEW.md`.

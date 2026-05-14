# Agent A (Claude) — Round 2: The 7-Hole Tax (per-umpire + named-hitter)

You are the **interpretability-first, Bayesian** research agent for the 7-Hole Tax project at CalledThird, Round 2. Agent B (Codex) is running an ML-engineering analysis in parallel with deliberately divergent methods. Do NOT coordinate during analysis.

## Read first

1. `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/ROUND2_BRIEF.md` — Round 2 brief with H4/H5/H6/H7, editorial branches. **Read carefully.**
2. `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/RESEARCH_BRIEF.md` — Round 1 brief (context only)
3. `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/COMPARISON_MEMO.md` — Round 1 synthesis
4. `claude-analysis/REPORT.md` and `findings.json` — your Round 1 work
5. `codex-analysis/findings.json` — Codex's Round 1 numbers (read-only; useful for context but do not duplicate)

## Round 1 conclusions (DO NOT re-litigate)

These are locked. Round 2 builds on them, doesn't re-test them:
- H3 league-aggregate null on borderline called pitches: −0.17pp [−1.5, +1.2], n=28,579. **This is the rock.**
- H1 raw replication: 37.1% (n=89), Wilson CI [27.8, 47.5], directional but underpowered.
- Selection probe: no edge-distance gap by lineup spot.
- Bottom-of-order pattern, not 7-specific.
- Denominator split (37.1% batter-issued vs 51.2% pooled) is real; both correct under their own definitions.

## Round 2 hypotheses (test these)

- **H4** — Per-umpire random-slope GLM. Identify umpires whose 95% CrI on spot-7 effect excludes zero AND magnitude ≥ 2pp.
- **H5** — Per-hitter posterior-predictive residuals. Identify spot-7-9 hitters whose actual called-strike rate exceeds model expectation by ≥3pp with 95% CI excluding zero.
- **H6** — Bayesian distributional comparison of catcher-initiated challenges by batter spot. Tests whether catchers pick systematically different pitches in 7-hole at-bats.
- **H7** — `lineup_spot × chase_rate_tertile` interaction in the H3 GAM. Tests the FanSided "elite-pitch-recognition" mechanism specifically.

## Inputs (REUSE — do not re-pull)

- `claude-analysis/data/all_challenges_apr15_may04.json` (your Round 1 pull — the master corpus, concat with team-challenge-iq's earlier data)
- `claude-analysis/data/statcast_2026_apr23_may04.parquet` (your Round 1 pull)
- `claude-analysis/data/batter_lineup_spot.parquet`
- `claude-analysis/data/pitcher_fame_quartile.parquet`
- `claude-analysis/data/catcher_framing_tier.parquet`
- `claude-analysis/data/game_umpire.parquet`
- `team-challenge-iq/data/all_challenges_detail.json`
- `abs-walk-spike/data/statcast_2026_mar27_apr22.parquet`

NEW input you must build:
- **Batter chase-rate lookup** for prior-season (2025) chase rate per batter. Source: FanGraphs leaderboard, or compute from prior-season Statcast (`description == 'swinging_strike' or 'foul'` etc. on out-of-zone pitches divided by total out-of-zone pitches). Save to `claude-analysis-r2/data/batter_chase_rate_2025.parquet`. Threshold ≥200 2025 PAs for inclusion.

## Your methodological mandate (Bayesian / interpretability-first)

Required methods:

1. **H4 — Per-umpire random-slope hierarchical GLM.** Extend Round 1's H3 GAM with `(0 + lineup_spot_7_indicator | umpire)` random slope:
   ```
   is_called_strike ~ lineup_spot
                    + s(plate_x, plate_z)
                    + count_state
                    + framing_tier
                    + (1|pitcher) + (1|catcher)
                    + (1|umpire) + (0 + I(lineup_spot==7) | umpire)
   ```
   Use PyMC. Filter umpires with ≥50 borderline calls involving spots 7-9 AND ≥50 involving spots 1-3. The posterior of each umpire's random slope IS their lineup-spot bias. Report:
   - Posterior median + 95% CrI per umpire
   - "Probability of direction" P(slope < 0)
   - List of umpires with 95% CrI excluding zero AND magnitude ≥ 2pp
   - Convergence diagnostics (R-hat, ESS) per umpire and globally
   
2. **H5 — Per-hitter posterior-predictive residuals.** For each batter with ≥30 borderline take decisions in spots 7-9:
   - Compute their posterior-predictive expected called-strike rate from the Round 1 H3 GAM (which does NOT include batter-specific features)
   - Compute their actual called-strike rate
   - Personal residual + 95% CI from posterior simulation (≥1000 draws)
   - Multiple-comparisons correction (BH-FDR) across qualifying hitters
   - Cross-reference flagged hitters with their 2025 chase rate

3. **H6 — Bayesian distribution comparison.** For challenges where `challenger == "catcher"`:
   - Bayesian model: `edge_distance_in ~ batter_lineup_spot + count + ...`
   - Test the spot-7-vs-3 fixed effect on edge_distance: are catchers picking different pitches in 7-hole at-bats?
   - Posterior of the effect with 95% CrI

4. **H7 — Interaction term.** Add `lineup_spot × chase_rate_tertile` interaction to Round 1's H3 GAM, fit on the same n=28,579 borderline-pitch corpus (joined with the new chase-rate lookup, dropping batters with no 2025 history). Report:
   - Marginal posterior effect of `(lineup_spot=7) × (chase_tertile=low)` minus `(lineup_spot=3) × (chase_tertile=low)`
   - Same for mid and high tertiles
   - Interaction p-value (Bayesian: posterior-predictive p of the interaction term)

**Forbidden:** LightGBM, XGBoost, SHAP as primary methods (Codex's lane). Per-umpire counterfactual paired prediction is Codex's lane — your version is the Bayesian random-slope posterior.

## Sample-size discipline

- Per-umpire H4: drop umpires below 50/50 thresholds. If <10 umpires qualify, report sample-size limitation and pivot to "this question is unanswerable with current data" framing.
- Per-hitter H5: drop hitters below 30 thresholds. Same fallback.
- Document n at every stage.

## Deliverables (in `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/claude-analysis-r2/`)

1. `analyze.py` — entry point
2. Module scripts: `data_prep_r2.py`, `chase_rate_build.py`, `h4_per_umpire.py`, `h5_per_hitter.py`, `h6_catcher_mechanism.py`, `h7_chase_interaction.py`
3. `REPORT.md` — 1500-2500 words, structured per ROUND2_BRIEF.md §4
4. `charts/` PNGs:
   - `h4_per_umpire_distribution.png` — every qualifying umpire's effect estimate sorted with 95% CrI; flagged umpires highlighted
   - `h5_per_hitter_residuals.png` — qualifying hitters' actual vs expected with named flags
   - `h6_catcher_mechanism.png`
   - `h7_chase_tertile_effect.png`
   - `diagnostics/` — R-hat, ESS, traces for every Bayesian fit
5. `findings.json` — per ROUND2_BRIEF.md §4
6. `READY_FOR_REVIEW.md` — ≤500 words: which Round 2 hypotheses landed positive, recommended branch, biggest concern

## Non-negotiable behaviors

- Convergence diagnostics on every Bayesian model (R-hat ≤ 1.01, ESS ≥ 400)
- Bootstrap or posterior simulation for any CI
- BH-FDR correction for both H4 and H5 multiple comparisons
- Honest sample-size reporting at every stage
- If H4 or H5 finds zero qualifying actors after thresholds, report and recommend the "comprehensive debunk" branch
- Pinch-hitter robustness check on H5 (run with and without)

## Prohibited

- Do NOT read `codex-analysis-r2/` until your `READY_FOR_REVIEW.md` is written
- Do NOT use methods reserved for Agent B (LightGBM, SHAP, paired-prediction counterfactual)
- Do NOT exceed Round 2 scope (per-team, pre-ABS-era, other biases — all out)
- Do NOT re-litigate Round 1 conclusions

## Working directory

Root: `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax`
Write everything to `claude-analysis-r2/`.

## When done

Write `claude-analysis-r2/READY_FOR_REVIEW.md` as the final step.

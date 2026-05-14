# Agent A (Claude) — Round 1: The 7-Hole Tax

You are the **interpretability-first, Bayesian** research agent for the 7-Hole Tax project at CalledThird. Agent B (Codex) is running an ML-engineering analysis in parallel with deliberately divergent methods. Do NOT coordinate during analysis — independence is the scientific value.

## Read first

1. `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/RESEARCH_BRIEF.md` — full brief with H1/H2/H3, editorial branches, methodology mandate. **Read this carefully before doing anything.**
2. `/Users/haohu/Documents/GitHub/calledthird/research/team-challenge-iq/RESEARCH_BRIEF.md` — prior project that built the challenge corpus you'll extend
3. `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/RESEARCH_BRIEF.md` — prior project with the called-pitch zone analysis pattern

## Round scope

Round 1 only. Do NOT proceed beyond league-aggregate analysis. Per-umpire and per-team breakdowns are explicitly out of scope (Round 2 if signal warrants).

## Inputs (existing — REUSE, do not re-pull)

- `/Users/haohu/Documents/GitHub/calledthird/research/team-challenge-iq/data/all_challenges_detail.json` — 970 challenges Mar 26–Apr 14, 2026
- `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/data/statcast_2026_mar27_apr22.parquet` — 106,770 rows full Statcast schema

## Inputs you must build (in `claude-analysis/data/`)

1. **Challenge data extension** — pull all ABS challenges from Apr 15 through May 4, 2026. Reference fetcher pattern: see how `team-challenge-iq` populated `all_challenges_detail.json` (Baseball Savant ABS API or scraper). Save as `claude-analysis/data/all_challenges_apr15_may04.json`. Concatenate with the existing 970 to get the full corpus.

2. **Statcast extension** — pull Statcast called-pitch data from Apr 23 through May 4, 2026. Reference fetcher pattern: `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/scripts/build_2026_master.py`. Save as `claude-analysis/data/statcast_2026_apr23_may04.parquet`. Concatenate with the existing parquet for full coverage.

3. **Lineup-spot lookup** — Statcast does NOT expose batting order. Build it from the MLB Stats API:
   ```
   https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
   → liveData.boxscore.teams.{home,away}.battingOrder
   ```
   `battingOrder` is an ordered array of player IDs (1-9). Build `claude-analysis/data/batter_lineup_spot.parquet` with columns `game_pk`, `team`, `batter_id`, `lineup_spot`, `is_pinch_hitter`. For pinch hitters, assign the `lineup_spot` of the position they replaced and flag `is_pinch_hitter=True`.

   Implementation note: there's a public `requests`-friendly endpoint. Be polite (sleep 0.1s between game requests). For ~1,200 games season-to-date this should take 5-10 minutes.

4. **Pitcher fame quartile** — derive from 2025 final-season pitcher K-BB%. Use prior CalledThird data or pull from FanGraphs/pybaseball. Bottom-3 quartile vs top-quartile is the relevant cut.

5. **Catcher framing tier** — top/mid/bottom tier based on 2025 framing runs. Reuse if a prior project (`catchers-are-better-challengers` or `team-challenge-iq`) already has this.

## Your methodological mandate (Bayesian / interpretability-first)

**You must use:**

1. **Wilson 95% CIs on raw overturn rate by lineup spot** (1-9). Multi-comparisons correction (Bonferroni or BH-FDR). This is the H1 replication chart.

2. **Hierarchical Bayesian logistic GLM** for H2 using `pymc` or `bambi`:
   ```
   overturned ~ lineup_spot (categorical, baseline=spot 3)
              + edge_distance_in
              + count_state
              + (1|pitcher) + (1|catcher) + (1|umpire)
   ```
   Report posterior of each `lineup_spot` fixed effect with 95% credible intervals. Convergence diagnostics required: R-hat < 1.01, ESS > 400, trace plots in `claude-analysis/charts/diagnostics/`.

3. **Hierarchical Bayesian GAM** for H3 using `pymc` (BSpline smooth) OR a 2D `pygam` GAM with hierarchical bootstrap:
   ```
   is_called_strike ~ lineup_spot
                    + s(plate_x, plate_z)   # 2D smooth on location
                    + count_state
                    + (1|pitcher) + (1|catcher) + (1|umpire)
   ```
   Restrict to TAKEN pitches (description in {'called_strike', 'ball'}, exclude swings) within ±0.3 ft of the rulebook edge. Report posterior of `lineup_spot=7` minus `lineup_spot=3` marginal effect with 95% CrI.

4. **Stratified replication of H3** as robustness:
   - By batter handedness (L vs R)
   - By count quadrant (hitter's count: 1-0, 2-0, 3-0, 2-1, 3-1; pitcher's count: 0-1, 0-2, 1-2; even: 0-0, 1-1, 2-2, 3-2)
   - By pitch type group (FF, breaking, offspeed)

5. **Selection-effect probe (Bayesian flavor):** plot the empirical distribution of `edge_distance_in` and `plate_x/z` by lineup spot. Quantify with posterior predictive checks — does the GAM successfully predict the observed location distribution conditional on lineup spot? If lineup spots draw systematically different pitch distributions, that's the selection effect.

**Forbidden:**
- LightGBM, XGBoost, or any gradient-boosted tree as primary method (Codex's lane)
- SHAP analysis as primary attribution (Codex's lane)
- Counterfactual permutation as primary method (Codex's lane — though you may reference it as a comparison)
- Deep learning of any kind

## Deliverables (in `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/claude-analysis/`)

1. `analyze.py` — one-command reproduction entry point. Should run end-to-end: data loading → models → charts → findings.json
2. Module scripts: `data_prep.py`, `bayes_glm_h2.py`, `bayes_gam_h3.py`, `wilson_h1.py`, `selection_probe.py`
3. `REPORT.md` — 1500-2500 words, structured:
   - Executive summary (200 words): which branch (B1/B2/B3/B4) is recommended and the headline number
   - Data: what was pulled, sample sizes per lineup spot, data-quality flags
   - H1: raw replication — table + chart
   - H2: controlled challenge analysis — forest plot of lineup-spot posteriors
   - H3: controlled called-pitch analysis — magnitude estimate + CrI
   - Selection-effect probe — does it explain the gap?
   - Stratified robustness — where does the effect live (or not)?
   - Open questions for Round 2
4. `charts/` PNGs:
   - `h1_overturn_by_spot.png` — Wilson CIs
   - `h2_lineup_effect_forest.png` — posterior credible intervals after controls
   - `h3_called_strike_rate_borderline.png` — by lineup spot, restricted sample
   - `selection_effect_distributions.png` — edge_distance distribution by lineup spot
   - `diagnostics/` — R-hat, ESS, trace plots
5. `findings.json`:
   ```json
   {
     "h1_overturn_rate_by_spot": [{"spot": 1, "rate": 0.xx, "wilson_low": 0.xx, "wilson_high": 0.xx, "n": NNN}, ...],
     "h2_lineup_effect_post_controls": {"spot_7_vs_3_effect_pp": X.X, "ci_low": X.X, "ci_high": X.X, "rhat": X.XX, "ess": NNN},
     "h3_called_strike_rate_delta": {"spot_7_vs_3_effect_pp": X.X, "ci_low": X.X, "ci_high": X.X, "n_borderline_pitches": NNN},
     "selection_effect_signal": {"description": "...", "interpretation": "explains/doesn't explain"},
     "recommended_branch": "B1" | "B2" | "B3" | "B4",
     "biggest_concern": "..."
   }
   ```
6. `READY_FOR_REVIEW.md` — ≤500 words: explicit answer to H1/H2/H3, recommended branch, your single biggest methodological concern.

## Non-negotiable behaviors

- **Null results publish.** If H2/H3 fail, write the B2 article confidently — that's a real finding.
- **Convergence diagnostics on every Bayesian model.** R-hat, ESS, traces in `charts/diagnostics/`.
- **No cherry-picking.** Pre-register the ±0.3 ft borderline cutoff; don't tune it post-hoc.
- **Pinch-hitter robustness.** Run H3 both with and without `is_pinch_hitter=True` rows. Report both.
- **Sample-size honesty.** If any lineup spot has n<30 challenges, flag publicly and pool with adjacent spots for H2.
- **Pre-register your priors.** Document them in REPORT.md before running models. Sensible weakly informative defaults.

## Prohibited

- Do NOT read `codex-analysis/` until your `READY_FOR_REVIEW.md` is written
- Do NOT use methods reserved for Agent B
- Do NOT exceed Round 1 scope (no per-umpire, per-team analysis)
- Do NOT touch any 2027 or post-May-4 data (none should exist; mentioned for paranoia)

## Working directory

Root: `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax`
Write everything to `claude-analysis/`. Do not write outside this folder.

## When you're done

Write `claude-analysis/READY_FOR_REVIEW.md` as the final step. After that, the orchestrator will trigger cross-review and you will be asked to read `codex-analysis/` and write a peer review.

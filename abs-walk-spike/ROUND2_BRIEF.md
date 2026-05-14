# Round 2 Brief — ABS Walk Spike (mid-May re-run)

**Status:** Round 2 — agents launching
**Target deliverable:** Flagship CalledThird article, 24-48h Round 2 ship
**News hook expiration:** ~5-7 days (FanGraphs has published TWO competing pieces on the same beat: "Where Are 2026's Extra Walks Coming From?" and "The Strike Zone Is Shrinking. Here's How." — we must move fast)
**Principal risk:** With three weeks of fresh data, the headline 40-50% zone attribution number could shift meaningfully. The narrative depends on where it lands.

---

## 0. What Round 1 settled (do NOT re-litigate)

These conclusions are locked from Round 1; do not re-test:

- **Walk spike is real and not seasonality.** 9.77% over Mar 27-Apr 22 vs historical mean 9.02% (2018-2025), Z = +4.41σ.
- **Zone changed shape in absolute coordinates.** Top edge (z ≈ 3.2-3.9 ft): ~-7-8pp called-strike rate drop. Bottom edge (z ≈ 1.0-2.0 ft): ~+5-6pp expansion.
- **Walk spike is NOT concentrated at 3-2.** 3-2 walk-rate delta = -0.11pp vs +0.82pp all-counts (Cochran's Q p = 0.67).
- **40-50% counterfactual attribution to zone change.** Two independent implementations: +40.46% and +49.40%.
- **Statcast `sz_top`/`sz_bot` schema change** between 2025 and 2026 broke `plate_z_norm`. Use absolute plate coordinates.
- **The 0-0 first-pitch tension is unresolved.** All-pitches counterfactual: +40-50% (zone adds walks). 0-0-only counterfactual: -20% to -42% (zone REMOVES walks at first pitches). First-called-pitch counterfactual: +12%. Hypothesis (untested): 2026 zone is more strike-friendly on first pitches but less strike-friendly in late counts at the top edge.

What's open and Round 2 must answer:
- Whether the +0.82pp YoY spike has held, grown, or regressed with ~3 more weeks of data (FanGraphs reports walk rate now ~9.5% through May 8 — directionally similar)
- Whether the 40-50% zone-attribution number holds with 2× the data
- Per-count attribution decomposition: which counts contribute most to the spike?
- The 0-0 first-pitch flip mystery: what's actually happening on first pitches that pushes more PAs into hitter's counts?
- Pitcher adaptation timing: have pitchers shifted pitch mix, location, zone rate week-over-week?
- FanGraphs reports 3-4 contributing factors (zone shrinkage, zone-rate drop 50.7%→47.2%, pitch-mix shifts, swing-rate drop). Quantify each with CIs.

---

## 1. The Round-2 sub-hypotheses

### H1 — Walk-rate persistence
**Hypothesis:** The YoY walk-rate spike has not regressed. Through May 13, the YoY delta is ≥0.8pp.

**Operational definition:** Pull 2026 Mar 27 – May 13 vs same window 2025. Compute walk rate, intentional-walks-included. Report YoY delta with bootstrap CI.

**Possible outcomes:**
- Spike held or grown → confirms durability; story is "the new equilibrium"
- Spike regressed substantially → bigger story: "pitchers found the new floor"
- Mixed (held in some counts, regressed in others) → leads into H2

### H2 — Per-count attribution decomposition
**Hypothesis:** The +0.82pp YoY walk-rate spike decomposes into per-count contributions. Specifically: which counts contribute most to the aggregate spike?

**Operational definition:** For each of 12 count states, compute the YoY walk-rate delta and the contribution to the all-counts spike (delta × PA-share). Bootstrap CIs per count. The contributions sum to the total spike.

**The 0-0 mystery (specifically):** Round 1 reported 0-0 delta = +0.82pp and 3-2 delta = -0.11pp — i.e., the per-count walk-rate change is broadly distributed, NOT concentrated at any specific count. But PA flow analysis showed 3-0 traffic increased more than the total walk-rate jump. Round 2 must resolve: is the spike from per-count walk-rate increases, or from more PAs flowing into walk-prone counts (3-0, 3-1), or both?

### H3 — Zone-attribution re-run
**Hypothesis:** With 2× the data, the 40-50% zone-attribution holds. The new estimate falls in [30%, 60%].

**Operational definition:** Re-run the counterfactual replay: classify each 2026 taken pitch with the 2025 zone classifier; recompute would-be walk rate; report fraction of YoY spike attributable to zone change. Two independent implementations (Bayesian + ML) must converge to within 15pp of each other.

**Counterfactual variants required:**
- All-pitches replay (the headline number)
- Per-count replay (resolve the 0-0 tension)
- Per-edge-region replay (top edge vs bottom edge separately — quantify the top-edge contribution specifically)

### H4 — Pitcher adaptation timing
**Hypothesis:** Pitchers have begun adapting to the new zone. Either pitch mix has shifted (more sinkers / fewer four-seamers at the top), location has shifted (lower release / lower target), or zone rate has shifted (more deliberate misses for whiffs).

**Operational definition:** Compute week-over-week:
- Pitch-type share by week (FF, SI, SL, CH, etc.)
- Average vertical location of called/swung-at pitches by week
- Zone rate (% of pitches in rulebook zone) by week, with CIs

Test whether week-over-week trends are statistically significant. Build per-pitcher leaderboard of who shifted most (top 10 pitchers by adaptation magnitude, with ≥200 pitches in window).

**Cross-reference:** FanGraphs reports league-wide zone rate down 50.7% → 47.2% in 2026. Verify and decompose.

### H5 — The 0-0 first-pitch flip mechanism
**Hypothesis:** The 0-0 first-pitch flip (zone REMOVES walks at 0-0 but ADDS them overall) reflects a real causal mechanism: 2026 zone is more strike-friendly at the heart on first pitches (reducing 1-0 traffic) but less strike-friendly at the top edge on later counts (allowing more terminal balls in deep counts).

**Operational definition:** Decompose first-pitch (0-0) called pitches into heart-zone vs edge-zone. Test:
- Has the heart-zone CS rate at 0-0 changed YoY?
- Has the top-edge CS rate at 0-0 changed differently than in deeper counts?
- Do pitches at the top of the zone become balls at different rates in 0-0 vs 2-strike counts?

**Three possible answers:**
- Mechanism confirmed → publish as the resolution of Round 1's loose thread
- Mechanism rejected (no heart/edge asymmetry by count) → publish as "the 0-0 mystery remains; here's what it isn't"
- Mechanism partially confirmed → publish with caveats

---

## 2. Editorial branches

Three publishable framings depending on what comes back:

| Round 2 finding | Article frames as | Working title |
|---|---|---|
| **H1 + H3 + H4 + H5 all positive** (spike held, attribution stable, pitcher adaptation visible, mechanism resolved) | Comprehensive update | *"Three weeks later: the walk spike held. Pitchers haven't adapted. Here's the count-by-count decomposition — and the 0-0 mystery is solved."* |
| **H1 confirms, H3 stable, H4 partial, H5 resolved** | Mechanism piece | *"We Know How the ABS Walk Spike Works Now. Every Count. Every Edge. The First-Pitch Mystery, Resolved."* |
| **H1 shows regression** | Pitcher-adaptation piece | *"Three Weeks Later, Pitchers Started Adapting. Here's Where the Walk Spike Is Settling."* |
| **All-mixed** | Honest update | *"We Said 40-50% of the Walk Spike Was the Zone. After Three Weeks More Data, Here's What Actually Changed."* |

Any of the four framings is A-tier IF execution is clean. The "comprehensive update" is the highest ceiling.

---

## 3. Round structure

**Round 2 only** for this project. Single-round, news-anchored — FanGraphs is publishing.

---

## 4. Data

### 4.1 Primary corpus

**Existing (REUSE, do not re-pull):**
- `data/statcast_2026_mar27_apr22.parquet` — 106,770 rows (the Round 1 corpus)
- `data/april_walk_history.csv` — historical April walk rates 2018-2025
- `data/heatmap_cells.json` — Round 1 grid-binned heat-map cells
- `data/walk_by_count.json` — Round 1 walk-by-count breakdown
- `research/count-distribution-abs/data/statcast_2025_full.parquet` — full 2025 season for YoY comparison

**Fresh pull required:**
- `data/statcast_2026_apr23_may13.parquet` — Statcast for the new 21-day window. Use same fetcher pattern as `scripts/build_2026_master.py`. Expected ~80K rows.
- Use the same `set(2025.columns) & set(2026.columns)` intersection rule as Round 1 to avoid schema drift.

### 4.2 Critical comparability rules (re-stated from Round 1)
- Walk events: include both `walk` and `intent_walk` for headline numbers (ESPN/MLB convention).
- Exclude `automatic_ball` and `automatic_strike` from called-pitch subsets (these are ABS challenge artifacts).
- Use ABSOLUTE `plate_x`, `plate_z` for zone analysis. Round 1 confirmed `plate_z_norm` is broken due to Statcast `sz_top`/`sz_bot` schema change.
- For week-over-week (H4), bin into 7-day windows starting Mar 27. Final week may be partial.

### 4.3 Known data-quality issues
- Statcast publishing lag for very recent dates (May 13-14). If PA counts look low for the most recent day vs nearby days, exclude that day.
- 2026 sz_top/sz_bot is deterministic per-batter (schema change documented Round 1). For any height-normalized analysis, use Lahman roster heights or omit normalization.

---

## 5. Methodology

### 5.1 Agent A (Claude) — interpretability-first, Bayesian

**Mandate:** Hierarchical Bayesian models with count-state random effects and weekly time-varying components. Every claim ladders down to a posterior + chart a reader can replicate.

Required methods:

1. **H1 — YoY persistence GLM.** Bayesian logistic GLM on walk-event ~ year + week + (1|count_state) over Mar 27 – May 13 window. Report posterior of `year_2026` fixed effect with 95% CrI. Stratify by week to test for regression toward 2025.

2. **H2 — Per-count posterior attribution.** Per-count Bayesian binomial model: walk-rate ~ year + (1|count_state). Posterior of per-count YoY delta with credible intervals. Contribution-to-aggregate via posterior simulation: each count's contribution = (delta × PA-share) with full propagation of uncertainty.

3. **H3 — Counterfactual replay with Bayesian zone classifier.** Train a Bayesian logistic GAM zone classifier on 2025 taken pitches: `is_called_strike ~ s(plate_x, plate_z) + count_state`. Apply to every 2026 taken pitch to predict its 2025-era called-strike probability. Replay each PA under counterfactual classifier; aggregate to walk rate. Report posterior of zone-attribution % with 95% CrI. Include per-count and per-edge-region variants.

4. **H4 — Week-over-week trend GAM.** For each pitcher with ≥200 pitches in window, fit `pitch-type-share ~ s(week)`, `mean-vertical-location ~ s(week)`, `zone-rate ~ s(week)`. Posterior of slope coefficients. League-wide also as descriptive.

5. **H5 — First-pitch heart/edge decomposition.** Bayesian model: `is_called_strike ~ year × zone-region × count-state` restricted to first pitches and 2-strike counts. Test whether the year × zone-region interaction differs by count.

Convergence diagnostics required: R-hat ≤ 1.01, ESS ≥ 400. Traces and summary diagnostics in `claude-analysis-r2/charts/diagnostics/`.

**Forbidden:** LightGBM, XGBoost, SHAP as primary methods (Codex's lane). Counterfactual paired prediction (Codex's lane — Claude's counterfactual is the Bayesian posterior replay).

### 5.2 Agent B (Codex) — ML-engineering, model-driven

**Mandate:** Gradient-boosted classifiers with cross-validated counterfactual paired prediction; SHAP attribution; explicit feature-importance baselines.

Required methods:

1. **H1 — YoY persistence with weekly stratification.** LightGBM binary classifier: `walk_event ~ year + week + count_state + plate_x + plate_z + pitch_type`. Permutation importance per feature. Predict per-week walk rate under year-counterfactual.

2. **H2 — Per-count counterfactual.** Train one LightGBM zone classifier on 2025 pitches. Apply to 2026 PAs. For each of 12 count states, compute counterfactual walk rate via paired prediction. Bootstrap CIs (N≥200). Sum to aggregate contributions.

3. **H3 — Counterfactual replay (full + per-region).** Same as Round 1 but with 2× data. All-pitches, per-count, per-edge-region (top vs bottom). Bootstrap CIs. Compare to Round 1's +40.46% benchmark.

4. **H4 — Pitcher-adaptation leaderboard.** For each pitcher (≥200 pitches), compute week-over-week shift in pitch-type distribution (Jensen-Shannon divergence), mean vertical location, zone rate. Rank top-10 most-shifted pitchers. SHAP per-pitcher.

5. **H5 — First-pitch decomposition with SHAP.** Train classifier: `is_called_strike ~ year × zone-region × count-state` on first pitches. SHAP interaction values to localize where year × region differs by count.

**Critical methodology constraint (from R1 + R2 7-hole tax learnings):** Codex's bootstrap CIs in past rounds have suffered from "fixed-model bootstrap" artifacts where CIs are narrower than parameter uncertainty warrants. Round 2 fix:
- Refit-bagged bootstrap: resample data AND refit GBM at each iteration (N≥50 refits minimum)
- OR: explicit model-uncertainty quantification via ensembling (10 random-seed models)
- Calibration curve required on every GBM

**Forbidden:** PyMC, bambi, hierarchical Bayesian (Claude's lane). Posterior credible intervals as primary inference (Claude's lane).

**Why the divergence matters:** Claude produces Bayesian posteriors with proper uncertainty propagation. Codex produces ML attribution with explicit feature interpretability (SHAP). Convergence on zone-attribution % across both methods locks the headline. Divergence becomes the comparison memo's centerpiece.

---

## 6. Round 2 Deliverables per agent

Each agent must produce in `claude-analysis-r2/` or `codex-analysis-r2/`:

1. `analyze.py` — one-command reproduction entry point
2. Module scripts: `data_prep_r2.py`, `h1_persistence.py`, `h2_per_count.py`, `h3_counterfactual.py`, `h4_pitcher_adaptation.py`, `h5_first_pitch.py`
3. `REPORT.md` — 1500-2500 words covering H1-H5 + recommended editorial branch
4. `charts/` PNGs at minimum:
   - `h1_walk_rate_by_week.png` — weekly walk rate 2025 vs 2026 with CIs
   - `h2_per_count_contribution.png` — bar chart of per-count YoY delta with CIs
   - `h3_counterfactual_attribution.png` — zone-attribution % with comparison to Round 1
   - `h4_pitcher_adaptation_leaderboard.png` — top-10 most-shifted pitchers
   - `h5_first_pitch_mechanism.png` — heart/edge × count interaction
   - `diagnostics/` — convergence diagnostics (Claude) / model-uncertainty calibration (Codex)
5. `findings.json` — machine-readable summary:
   ```json
   {
     "h1_walk_rate": {"rate_2025": X.X, "rate_2026": X.X, "yoy_delta_pp": X.X, "ci_low": X.X, "ci_high": X.X, "n_2026": NNN},
     "h2_per_count": [{"count": "0-0", "delta_pp": X.X, "ci_low": X.X, "ci_high": X.X, "contribution_to_aggregate_pp": X.X, ...}, ...],
     "h3_zone_attribution_pct": {"all_pitches": X.X, "ci_low": X.X, "ci_high": X.X, "per_count": {...}, "top_edge": X.X, "bottom_edge": X.X},
     "h4_adaptation": {"league_zone_rate_2025": X.X, "league_zone_rate_2026": X.X, "league_weekly_trend": "...", "top_adapters": [{"pitcher": "...", "shift_magnitude": X.X, ...}]},
     "h5_first_pitch_mechanism": {"heart_zone_0_0_yoy_delta_pp": X.X, "top_edge_2_strike_yoy_delta_pp": X.X, "interaction_credible": true/false},
     "recommended_branch": "comprehensive-update | mechanism | adaptation | honest-update",
     "biggest_concern": "..."
   }
   ```
6. `READY_FOR_REVIEW.md` — ≤500 words

---

## 7. Sample-size discipline

- H4 per-pitcher: ≥200 pitches in window required. If <30 pitchers qualify, league-level only.
- Bootstrap N ≥ 200 for all CIs.
- Refit-bagged bootstrap (Codex) ≥ 50 refits minimum.
- Convergence (Claude): R-hat ≤ 1.01, ESS ≥ 400.

---

## 8. Scope fence

**IN scope:**
- H1-H5 as defined
- League-aggregate and per-pitcher (with sample-size thresholds)
- Comparison to Round 1 numbers explicitly

**OUT of scope (deferred to Round 3 if signal warrants):**
- Per-umpire breakdown (research-queue item: "ABS Walk Spike — per-umpire / team / catcher cuts")
- Per-team breakdown
- Catcher framing × zone-change interaction
- Pre-ABS-era comparison

---

## 9. Timeline

| Hour | Focus | Gate |
|------|-------|------|
| 0 | Brief approved, agents launched | Round 1 data reused, May data pull scripted |
| 0-6 | Both agents run H1-H5 + data pull | Both `READY_FOR_REVIEW.md` exist |
| 6-10 | Cross-review | Both reviews in `reviews/` (r2- prefix) |
| 10-14 | Comparison memo + A-tier assessment | `reviews/COMPARISON_MEMO_R2.md` |
| 14-36 | Article draft | Draft via `calledthird-editorial` |
| 36-48 | Astro + charts + OG + ship | Live |

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Walk spike has regressed; story flips | This becomes Branch 3 (pitcher-adaptation piece) — still A-tier |
| Zone-attribution number drifts substantially from 40-50% | Honest update piece (Branch 4). We own the original take; updating it is a brand feature, not a bug. |
| Codex bootstrap CIs are again too narrow | R2 mandate: refit-bagged bootstrap (≥50 refits) OR model ensembling. Cross-review will catch any remaining artifact. |
| FanGraphs publishes a deeper piece in the meantime | We have count-decomposition + pitcher leaderboard + 0-0 mystery resolution they don't. Move fast. |
| 0-0 mystery resolves as "no mechanism" | Still publishable: "the 0-0 mystery remains; here's what it isn't" — and we own the question |
| Two agents diverge on H3 magnitude | Comparison memo resolves; report range (Round 1 reported 40-49% range, accepted) |

---

## 11. How to Run

```bash
cd /Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike

# Agents read this brief + Round 1 artifacts. Do NOT re-pull Mar 27 – Apr 22 data.
# Each agent extends to May 13 in their own data folder.

# Launch:
# - Claude via Agent tool (background)
# - Codex via direct CLI to codex exec with this prompt
```

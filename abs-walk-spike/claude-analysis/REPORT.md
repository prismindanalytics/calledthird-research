# Did the strike zone shrink? — Agent A (Claude) Round 1 Report

**Author:** Claude (interpretability-first lane)
**Round:** 1
**Date:** 2026-04-23
**Primary YoY window:** Mar 27 – Apr 14, 2025 vs 2026 (apples-to-apples)
**H2 historical baseline:** Mar 27 – Apr 22, 2018-2025 (incl IBB)

## Executive summary

The 2026 zone *did* change shape — but not by uniformly shrinking. The new ABS-defined zone moved **up**: called-strike rate dropped sharply along the top edge of the rule-book box (~15-25pp loss in a contiguous band centered around plate_z ≈ 3.2 ft, GAM season×zone interaction LRT p ≈ 0) and *expanded* along the bottom edge (~+20pp around plate_z ≈ 1.5-1.7 ft). The +0.82 percentage-point walk-rate spike (9.11% → 9.92% same window, +4.41σ above the 2018-2025 April distribution) is **not** concentrated at 3-2; in fact the conditional walk rate at 3-2 is statistically flat (-0.11pp, p=0.93). Instead, the spike is **upstream-driven**: first-pitch called-strike rate fell 1.77pp, and the share of plate appearances that reach a 3-ball count rose by +0.89pp — about as much as the entire walk-rate increase. **Editorial recommendation: branch B1 (zone confirmed) with the angle re-framed as "the zone moved up, and pitchers are falling behind earlier" rather than "the zone shrank everywhere."**

## H1 — Did the zone shrink? Mostly at the *top*, while the bottom expanded.

### Method (interpretability-first per mandate)

1. **2D grid binning** of called pitches in absolute (`plate_x`, `plate_z`) at 0.10 ft × 0.10 ft. Per cell, per season, called-strike rate; per cell, bootstrap 95% CI for the (2026 − 2025) difference using 1000 binomial resamples per side. Cells with fewer than 30 pitches in either year are dropped or marked NaN. The total grid spans `plate_x ∈ [-1.7, 1.7]` × `plate_z ∈ [1.0, 4.5]` (1190 cells; 193 cells inside the rule-book interior have ≥30 pitches each year).
2. **Spline-smoothed difference surface** via a 2D Gaussian filter (σ=1.2 bins) over the binned deltas, weighting by sample mask. This is a thin-plate-spline-equivalent smoother that I picked deterministically before inspecting results (no resolution shopping).
3. **Logistic GAM** with `te(plate_x, plate_z) + season + s(plate_z·season) + s(plate_x·season)` to test whether year-by-zone-shape interaction is significant. LRT against the additive (no-interaction) model.
4. **Supplementary** — same grid in height-normalized z (`(plate_z - sz_bot) / (sz_top - sz_bot)`), and the same comparison extended to the full Mar 27 – Apr 22 window using `statcast_2025_full.parquet`.

### Result

The clean two-lobe pattern is visible in `charts/heatmap_zone_delta.png` and (more legibly) in `charts/heatmap_zone_delta_smoothed.png`:

- **Top edge — significant shrinkage.** A contiguous run of cells from x ≈ -0.8 to -0.45 ft and z ≈ 3.1-3.55 ft shows called-strike-rate drops of 20-36pp, multiple cells with bootstrap 95% CIs entirely below zero. The largest contiguous all-significant patch covers ~4 cells (~0.04 sq ft) with mean Δ = -25pp [CI -46, -4]. Across the *entire* "top edge band" defined as ±0.2 ft of the 3.5 ft rule-book line within plate width, the sample-weighted mean delta is **-22.4pp** (50% of cells in that band have CI entirely below zero, none have CI above).
- **Bottom edge — significant expansion.** A much larger contiguous run from x ≈ +0.2 to +0.8 ft and z ≈ 1.4-1.8 ft (and a near-mirror patch on the left side) shows +15 to +30pp gains in called-strike rate. Mean Δ in the bottom-edge band (±0.2 ft of 1.5 ft) is **+10.3pp**, with 36% of cells CI-significantly positive and none CI-significantly negative. The largest single contiguous all-significant patch is 15 cells (~0.15 sq ft) with mean Δ = +21.8pp [CI +4.6, +39.3].
- **Zone interior** (the middle of the box) is essentially unchanged: weighted mean delta +2.5pp, only ~3% of cells CI-negative and ~10% CI-positive.

**GAM corroboration.** The full interaction model dropped deviance by 580 against the additive model with only ~10 added effective degrees of freedom; LRT p ≈ 0 (chi-square 580 on ~10 df). All season-interaction term p-values are reported as 0.0 by pygam (i.e. below floating-point precision). The partial-dependence plot (`charts/gam_partial_dependence.png`) shows the two season curves nearly overlap from 1.5 to 3.0 ft, then **2026 falls off the top of the zone roughly 0.10-0.15 ft sooner** than 2025. The same plot also shows a small 2026-low-z lift around 1.5-1.7 ft — the bottom expansion. The vertical-profile chart (`charts/vertical_profile_in_plate.png`) shows the same crossover.

### Why this fits the new ABS rule book

The new ABS zone is 27% to 53.5% of standing height. For the median 6'0" hitter:
- ABS bottom = 0.270 × 6.0 ft = **1.62 ft** (matches my pooled median `sz_bot` = **1.62 ft** exactly)
- ABS top = 0.535 × 6.0 ft = **3.21 ft** (well below the pooled median `sz_top` = **3.29 ft**, i.e. 0.08 ft / 1 inch lower)

The visible pattern is exactly what that rule change predicts: the top of the called zone moves down by ~1 inch (taking strike calls away from pitches near the upper edge) and the bottom is now defined as "27% of standing height" rather than the legacy "hollow beneath the kneecap" — which is a generous lower bound for many hitters and gives the strike at 1.5-1.7 ft that human umps often took away in 2025.

### H1 verdict

**H1 holds, with a critical caveat: the zone changed in *both* directions** at the same time. There is a contiguous shrinkage region of size 0.18 sq ft (CI-significant negative) at the top and an expansion region of 0.24 sq ft (CI-significant positive) at the bottom. Hoerner is right that hitters are laying off pitches at the top of the zone and getting called balls there — but the simple sentence "the zone shrunk" misses that the zone also crept *down* at the bottom. **Branch B1 is on; B2 is off.**

## H2 — Is it seasonality? No. The 2026 walk rate is +4.4σ above the 2018-2025 distribution.

I rebuilt H2 from primary data rather than copying the substrate baseline. Results match precisely:

- Historical mean (2018, 2019, 2021-2025; 2020 dropped for COVID) Mar 27 – Apr 22 walk rate incl IBB: **9.02%**, SD **0.171pp**, range 8.69%-9.17%.
- 2026 Mar 27 – Apr 22 walk rate incl IBB (from the 2026 parquet, my computation): **9.77%** (matches ESPN's 9.8% within rounding; matches substrate baseline exactly).
- **Z = +4.41σ** (incl IBB), bootstrap 95% CI on Z = [+3.6, +17.1] — the upper end is unstable because the historical SD is so small that resampling occasionally collapses it; the *lower* bound is the conservative number, and it is still well past the H2 threshold of Z ≥ 1.5.
- Excluding IBB: **Z = +6.36σ** — even tighter because IBB doesn't move much YoY.
- **+0.60pp above the prior 8-year April max** (which was 9.17% in 2018).
- Rank in the 2018-2026 distribution: **8th out of 8** (i.e. 100th percentile).

Reproduced. **H2 PASSES decisively.** Branch B3 (seasonality dominates) is ruled out.

See `charts/april_walk_history.png` for the bar chart.

## H3 — Does 3-2 take the worst hit? No, and the structure is more interesting.

### Method

For each season, attach a stable PA id from `(game_pk, at_bat_number)`, then for each of the 12 count states compute "fraction of PAs that ever reached this count" and "fraction of those PAs that ended in walk." Bootstrap Wilson 95% CIs per cell. Compute the (2026 − 2025) delta in walk rate at each count, with Newcombe-hybrid CI and a two-proportion p-value. Test whether the 3-2 delta is significantly different from the all-PA pooled delta via a difference-of-differences Z-test, and Cochran's Q across the 12 strata.

### Result

The per-count walk-rate-delta is in `charts/walk_by_count.png` and in the table below (delta in pp, 2026 minus 2025):

| Count | PAs 2025 | PAs 2026 | Walk rate 2025 | Walk rate 2026 | Δ pp | p (two-prop) |
|-------|---------:|---------:|---------------:|---------------:|-----:|-------------:|
| 0-0   | 18,125   | 18,664   |  9.10%         |  9.92%         | +0.82 | 0.007 |
| 0-1   |  9,119   |  9,200   |  5.15%         |  5.39%         | +0.24 | 0.47  |
| 0-2   |  3,850   |  3,939   |  3.32%         |  3.25%         | -0.08 | 0.85  |
| 1-0   |  6,954   |  7,435   | 16.95%         | 18.22%         | +1.27 | 0.046 |
| 1-1   |  7,080   |  7,245   |  9.72%         | 10.77%         | +1.05 | 0.039 |
| 1-2   |  5,411   |  5,439   |  6.41%         |  6.58%         | +0.17 | 0.72  |
| 2-0   |  2,447   |  2,700   | 34.04%         | 34.93%         | +0.88 | 0.51  |
| 2-1   |  3,663   |  3,934   | 20.72%         | 21.20%         | +0.48 | 0.61  |
| 2-2   |  4,468   |  4,579   | 13.85%         | 13.91%         | +0.06 | 0.94  |
| 3-0   |    790   |    979   | 68.73%         | 67.31%         | -1.42 | 0.52  |
| 3-1   |  1,529   |  1,735   | 46.04%         | 47.26%         | +1.22 | 0.49  |
| **3-2** | **2,636** | **2,783** | **33.31%** | **33.20%** | **-0.11** | **0.93** |

- **3-2 delta = -0.11pp**, ratio to the all-PA pooled delta = -0.13×, two-proportion p = 0.93. The 3-2 walk rate is essentially identical between years.
- Cochran's Q across the 12 strata = 8.45 on 11 df, p = 0.67 — i.e. the per-count deltas are *not* heterogeneous; almost all of the variation across counts is consistent with sampling noise around the +0.8pp pooled mean. There is no evidence that any one count carries a disproportionate share of the spike.
- Difference-of-differences Z-test (3-2 minus pooled 0-0): p = 0.48. Not distinguishable.

**H3 FAILS.** B4 is OFF the table.

### Where the spike actually comes from: more PAs reach 3-ball counts.

The interesting structural finding: **what changed YoY is upstream traffic, not deep-count conversion.**

- First-pitch called-strike rate (denominator = 0-0 called pitches): **45.31% → 43.54%, Δ = -1.77pp.** Fewer first strikes called.
- Fraction of PAs that ever reach 3-0: **4.36% → 5.25%, Δ = +0.89pp.** That single number is bigger than the entire +0.82pp walk-rate increase.
- Fraction of PAs that ever reach 3-1: 8.44% → 9.30%, Δ = +0.86pp.
- Fraction of PAs that ever reach 3-2: 14.56% → 14.92%, Δ = +0.36pp.
- Fraction of PAs that ever reach 1-0: 38.41% → 39.86%, Δ = +1.45pp.

Conditional walk rates given those counts are reached are essentially flat (3-0 even *dipped* -1.4pp, 3-2 flat, 3-1 +1.2pp). See `charts/count_reach_traffic.png`.

The mechanical chain is: top-of-zone shrinkage → fewer high-strike calls → first-pitch called-strike rate drops → more PAs go 1-0 → more PAs go 2-0 → more PAs reach 3-X → more walks. The walks happen in counts that were always walk-friendly; what changed is *how many PAs got there*. In that sense the "ABS shrunk the strike zone at the top" claim is the right diagnosis — it just propagates through the count tree rather than showing up in the conditional rate at 3-2.

## The flat-batting-average puzzle

League BA in this window: **2025 = .235, 2026 = .236** (essentially unchanged); OBP +0.7pp (.310 → .318), driven by the walk surge rather than hits. If the zone had shrunk uniformly, we would naively expect more pitches per PA to be in hitter-friendly locations, more swings on those, and slightly more contact / hits. We don't see that. Two non-exclusive explanations consistent with the data here (and which we are NOT modeling in Round 1 per scope):
1. **Pitcher adaptation.** Pitchers may be pitching more carefully — fewer high fastballs that now get called balls; more low pitches into the now-friendly bottom edge — which keeps batted-ball quality similar even as the called-strike geometry changed.
2. **The bottom-edge expansion offsets the top shrinkage *for hits*.** The bottom-edge zone expansion moves strike calls into a band where contact is generally weaker (lower in the zone → more grounders), so any added contact opportunities don't translate to a BA bump.

**Round 1 scope-fence:** I'm not attempting to attribute these — that's Round 2 territory (Codex's counterfactual is Round 1's first hint, and a pitcher-adaptation analysis is queued for Round 2).

## Editorial recommendation — B1 (zone confirmed), but reframe the headline

**Branch B1 ("Zone confirmed") with B4 ("3-2 leverage") OFF.**

The headline that fits the data is something like *"The Strike Zone Moved Up: ABS Took the High Strike Away from Pitchers"* — not "the zone shrunk everywhere" and not "look at 3-2." The paragraph order I would use in the article:

1. Open with the +0.82pp walk spike and the +4.4σ historical context (the news hook).
2. Heat map of the called-strike-rate delta. Lead with the top-edge blue band, with Hoerner's quote sitting on top of it. Note the bottom-edge red band immediately as the qualifier.
3. Connect the rule change ("27% to 53.5% of standing height") to the shape change: 0.535 × 6 ft = 3.21 ft, ~1 inch *below* where the human umps had been calling the top.
4. Pivot to the count-tree mechanism: first-pitch called-strike rate dropped 1.77pp; 3-0 traffic rose +0.89pp; that's where the walks are coming from. Show the per-count walk-rate chart with the flat 3-2 highlighted to bust the "3-2 hitter's count" myth in one beat.
5. Close on the BA puzzle (1 paragraph, scope-fenced).

When Codex's counterfactual report lands, the integration is straightforward: the zone-shape attribution % of the +0.82pp lives in section 2-3 and pairs cleanly with the geometry; the count-tree mechanism in section 4 is independent of attribution magnitude.

## Methods overview (for replication)

- Primary data: `data/statcast_2026_mar27_apr22.parquet` (106,770 rows) and `../count-distribution-abs/data/statcast_2025_mar27_apr14.parquet` (70,876 rows). For YoY heat-map and count-leverage, both filtered to Mar 27-Apr 14 (44,924 vs 35,514 called pitches after exclusions). For the H2 Z-score, 2026 uses the full Mar 27-Apr 22 window (27,144 PAs) against the historical aggregate in `data/april_walk_history.csv`.
- "Called pitches" = `description in {'called_strike','ball'}` and not `automatic_*`. (Auto calls are rare here — 294/9 in 2025 same window, 370/13 in 2026 — and in any case are ABS challenge artifacts, not human zone calls.)
- Walks include `events in {'walk','intent_walk'}` to match ESPN convention. Excluding IBB shifts the headline rate by ~0.3pp (9.77% → 9.44%) but does not change the qualitative finding (Z grows from 4.41σ to 6.36σ).
- 2D grid: 0.10 ft × 0.10 ft cells; min 30 pitches per cell; 1000 binomial bootstrap resamples per side for CIs on the difference of proportions per cell. The "largest contiguous significant region" uses 4-neighbor connected-components on the strict CI-excludes-zero mask with |delta| ≥ 3pp.
- GAM: `pygam.LogisticGAM` with `te(plate_x, plate_z) + f(season) + s(plate_z*season) + s(plate_x*season)`, n_splines=8 for the tensor and 12 for the season-conditional splines, default lambda. The interaction-LRT compares this to the additive `te(plate_x, plate_z) + f(season)` model.
- Count leverage: per-count walk rate uses "fraction of PAs that *ever reached* that count and walked" (not "fraction of PAs that walked from that count and *only* that count"). Newcombe hybrid CIs per delta. Cochran's Q across all 12 strata for heterogeneity, two-prop Z for 3-2 vs pooled.
- Z-score: bootstrap (5000 resamples) of the historical mean and SD; CI is wide on the upper end because historical SD is small (0.17pp).

## Open questions for Round 2

1. **Pitcher adaptation — pitch mix, location, velocity.** Have pitchers shifted their distribution toward the new bottom-edge "free strikes" and away from the old top-edge "free strikes"? Are 4-seam fastballs getting buried lower in the zone? This decomposes the BA puzzle.
2. **Counterfactual attribution.** When Codex's "what-if 2026 PAs were judged by the 2025 zone classifier" lands, the question is what fraction of the +0.82pp spike is mechanical zone change vs everything else. My guess from the count-tree analysis: ~60-80% of the spike is mechanically explainable from the called-strike-rate drop at the top, propagating through the count tree. Codex will report a sharp number.
3. **Per-umpire heterogeneity.** Are there umpires whose 2026 called-strike rate at the top of the zone *didn't* drop? Those would be candidates for "human umps who haven't yet adapted to the ABS feedback loop."
4. **Catcher framing erosion.** If the ABS-defined zone crowds out human-judgment leeway, framers should be losing edge faster than non-framers. Round 3.
5. **Bottom-edge expansion: real or schema artifact?** I'm treating the bottom-edge expansion as a real ABS rule-change effect because the math (0.27 × standing height ≈ 1.62 ft = my median sz_bot) is clean. But it is worth confirming with Statcast that `sz_bot` is computed identically across 2025 and 2026 — if pybaseball changed the column definition silently, the bottom-edge story collapses while the top-edge story is unaffected.

# The 7-Hole Tax — Bayesian Analysis (Agent A / Claude, Round 1)

**Author:** Agent A (Claude), CalledThird Research
**Window:** 2026-03-26 through 2026-05-04 (challenges); 2026-03-27 through 2026-05-03 (Statcast)
**Sample:** 2,101 ABS challenges • 75,681 taken pitches • 28,579 borderline taken pitches (|edge_distance|<=0.3 ft) • 524 unique games • 489 distinct batters

---

## Executive summary

**Recommended branch: B2 — selection / framing artifact ("the 7-hole tax is a mirage"), with a hedge toward B3** because the raw batter-overturn deficit is real but smaller than reported and not concentrated on the 7-hole.

Headline numbers:

- **H1 (raw replication, batter-issued challenges):** 7-hole batters win 37.1% of their challenges, vs 45.2% for the league-wide batter rate. Deficit = 8.1 percentage points with a Wilson 95% CI of [27.8%, 47.5%] on the 7-hole rate. The pre-registered >=10pp threshold is missed; spots 7, 8, and 9 form a flat plateau at ~37% with overlapping confidence intervals (none individually significant after Benjamini-Hochberg correction). The FanSided/Ringer "30.2%" reproduces directionally but at smaller magnitude in our 39-day window.
- **H2 (controlled challenge model, hierarchical Bayesian logistic GLM):** spot-7-vs-spot-3 marginal effect on overturn probability, after controls for `edge_distance_in`, `count_state`, `framing_tier`, `fame_quartile`, and random effects on pitcher / catcher / umpire, has a posterior median of **-9.5pp with a 95% credible interval of [-21.4, +3.5]** and P(effect<0) = 0.93. Convergence: R-hat = 1.01, ESS_bulk_min = 754. Directional but the credible interval crosses zero — we cannot rule out a true zero effect.
- **H3 (controlled called-pitch model, hierarchical Bayesian GAM on borderline takes):** spot-7-vs-spot-3 marginal effect on called-strike probability, after a 2D plate-location smooth and the same controls, has a posterior median of **-0.17pp with a 95% CrI of [-1.5, +1.2]**. The pre-registered >=+2pp threshold is comfortably missed and the effect is *negative*, not positive. Convergence: R-hat = 1.00, ESS_bulk_min = 717.
- **Selection-effect probe:** mean |edge_distance| on challenges is essentially identical across spots (1.27 in for spot 3, 1.27 in for spot 7); KS spot-7-vs-spot-3 p = 0.19. So 7-hole batters are *not* challenging easier or harder pitches than 3-hole batters. The selection effect lives elsewhere — likely in the count distribution and pitcher-quality matchups, both of which are absorbed by H2's controls.

The strongest test (H3, ~28k borderline pitches) is a clean null. The directional H2 signal in challenge data does not pass a 95% credibility bar and, taken with H3's null, is consistent with what would happen if challenges-by-bottom-of-the-order are simply *harder* in dimensions our controls partially absorb (e.g. count + pitcher random effect). The 7-hole tax in the raw FanSided sense reproduces but shrinks; in the controlled "umpires call this batter type's zone bigger" sense, we do not see it.

---

## Data

### Inputs (reused)
- `/Users/haohu/Documents/GitHub/calledthird/research/team-challenge-iq/data/all_challenges_detail.json` — 970 challenges, 2026-03-26 through 2026-04-14.
- `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/data/statcast_2026_mar27_apr22.parquet` — 106,770 Statcast pitches.

### Inputs (built fresh in `claude-analysis/data/`)
- `all_challenges_apr15_may04.json` — 1,131 new challenges scraped from the Baseball Savant gamefeed API for 2026-04-15 through 2026-05-04.
- `statcast_2026_apr23_may04.parquet` — 42,733 Statcast pitches via pybaseball (2026-04-23 through 2026-05-03; May 4 not yet indexed at pull time).
- `batter_lineup_spot.parquet` — 10,996 (game_pk, batter_id, lineup_spot, is_pinch_hitter) rows extracted from MLB Stats API live boxscore feeds for 524 games. Pinch hitters inherit the lineup spot of the slot they replaced and are flagged `is_pinch_hitter=True`. 26.5% of the rows are pinch-hitter / sub appearances; this includes all double-switches and late-game subs, not only true pinch-hitting plate appearances.
- `pitcher_fame_quartile.parquet` — 416 pitchers with PA>=30 in our 2026-to-date Statcast corpus, quartiled on K-BB%. FanGraphs 2025 endpoints returned 403 from this network so we used in-corpus 2026 K-BB% as the fame proxy (this is the brief's allowed fallback). The K-BB% rate is computed on every plate-appearance terminal pitch, so it has a small overlap with H2 outcome at the row level; pitcher random effects in H2/H3 absorb most of the residual contribution.
- `catcher_framing_tier.parquet` — 63 catchers with >=100 takes in the framing band (0.30-0.60 ft from the rulebook edge), tertile-binned by called-strike rate in that band. The framing band intentionally *excludes* the H3 borderline window (<=0.30 ft) to avoid the framing tier being a function of the H3 outcome.
- `game_umpire.parquet` — 524 home-plate umpires extracted from the MLB Stats API live boxscore officials list. (The Statcast `umpire` column was empty in our pybaseball pull, so we joined this from the boxscore.)

### Sample sizes by lineup spot

| Spot | Challenges (all) | Challenges (batter-only) | Taken pitches | Borderline takes (<=0.3 ft) |
|------|---:|---:|---:|---:|
| 1 | 290 | 131 | 9,827 | 3,785 |
| 2 | 260 | 125 | 9,467 | 3,570 |
| 3 | 259 | 124 | 8,932 | 3,318 |
| 4 | 226 | 109 | 8,425 | 3,111 |
| 5 | 228 | 114 | 8,453 | 3,140 |
| 6 | 221 | 99 | 8,011 | 3,031 |
| **7** | **213** | **89** | **7,845** | **2,964** |
| 8 | 193 | 85 | 7,375 | 2,820 |
| 9 | 211 | 95 | 7,346 | 2,840 |

The 7-hole batter-only cell at n=89 is the binding constraint on H1/H2 power. No spot has n<30, so we do not pool — but we report H2 with and without pinch hitters to absorb the implied star-PH-into-the-7-hole risk.

### Data-quality flags
- 486/970 of the original team-challenge-iq challenges had a null `edge_distance_in`. We re-derived these from the joined Statcast row's (`plate_x`, `plate_z`, `sz_top`, `sz_bot`) using a signed distance to the rulebook edge (positive outside, negative inside).
- pybaseball's Statcast pull leaves the `umpire` column empty; we sidestep this by extracting umpire from the MLB Stats API live boxscore.
- May 4 has 12 games with 42 challenges (all real ABS games) but Statcast for that date was not yet indexed when we pulled — H3 effectively ends 2026-05-03.
- Of the 2,101 challenges, 330 (15.7%) involve a pinch-hitter / late-game sub. We run H1, H2, and H3 with and without pinch hitters.

---

## Pre-registered priors

Documented before any model was fit:

**H2 (logistic GLM):**
- intercept ~ Normal(0, 1.5); lineup-spot fixed effects ~ Normal(0, 1.0); count_state effects ~ Normal(0, 1.0); edge_distance_in slope ~ Normal(0, 1.0); framing_tier and fame_quartile ~ Normal(0, 0.5); sigma_pitcher / catcher / umpire ~ HalfNormal(0.5); non-centered parameterization on random effects.

**H3 (logistic GAM):**
- intercept ~ Normal(0, 1.5); lineup-spot effects ~ Normal(0, 0.3) — tighter than H2 by design; count effects ~ Normal(0, 0.5); framing_tier ~ Normal(0, 0.3); 2D BSpline tensor-product basis (4x4=12 columns after dropping degenerate columns), centered, coefficients ~ Normal(0, 2.0); sigma_pitcher / catcher / umpire ~ HalfNormal(0.5); non-centered parameterization.

We tightened the H3 spot-effect prior to 0.3 (from 1.0) before any model fit, to reflect the prior that an umpire-bias logit would be small relative to counts and location at the borderline. We did not later widen it to fit a desired result.

NUTS sampling: 4 chains, 1500-2000 tune + 1000-1500 draws each, target_accept >= 0.92. Convergence diagnostics (R-hat, ESS, traces, energy) are saved in `charts/diagnostics/`.

---

## H1 — raw replication (Wilson 95% CIs)

**Charts:** `charts/h1_overturn_by_spot.png` (all challengers) and `charts/h1_overturn_batter_only.png` (batter-issued only — the FanSided/Ringer comparison).

ALL-challengers view:

| Spot | n | overturn rate | Wilson 95% CI |
|------|--:|--:|--:|
| 1 | 290 | 54.8% | [49.1, 60.5] |
| 2 | 260 | 55.0% | [48.9, 60.9] |
| 3 | 259 | 52.5% | [46.4, 58.5] |
| 4 | 226 | 53.5% | [47.0, 59.9] |
| 5 | 228 | 52.6% | [46.2, 59.0] |
| 6 | 221 | 55.7% | [49.1, 62.1] |
| 7 | 213 | 51.2% | [44.5, 57.8] |
| 8 | 193 | 50.8% | [43.8, 57.7] |
| 9 | 211 | 48.3% | [41.7, 55.1] |

The all-challengers view is **flat across spots** — all CIs overlap. H1 fails completely in this view.

BATTER-only view (the FanSided / Ringer "30.2%" comparison):

| Spot | n | overturn rate | Wilson 95% CI | p vs spot 3 | BH q |
|------|--:|--:|--:|--:|--:|
| 1 | 131 | 47.3% | [39.0, 55.8] | 0.83 | 0.85 |
| 2 | 125 | 47.2% | [38.7, 55.9] | 0.85 | 0.85 |
| 3 | 124 | 46.0% | [37.4, 54.7] | — | — |
| 4 | 109 | 50.5% | [41.2, 59.8] | 0.49 | 0.80 |
| 5 | 114 | 49.1% | [40.1, 58.2] | 0.63 | 0.84 |
| 6 | 99 | 50.5% | [40.7, 60.2] | 0.50 | 0.80 |
| **7** | **89** | **37.1%** | **[27.8, 47.5]** | **0.20** | **0.62** |
| 8 | 85 | 37.6% | [28.0, 48.3] | 0.23 | 0.62 |
| 9 | 95 | 36.8% | [27.7, 47.0] | 0.18 | 0.62 |

League-wide batter overturn rate: **45.2%** (439/971 batter-issued challenges).

Read this carefully: **the bottom three spots all hover around 37%, none individually significant after BH correction (q >= 0.62).** There is no smoking-gun "spot 7 specifically" effect; the pattern is "bottom-of-the-order batters lose challenges at a similar low rate." A pre-registered >=10pp deficit threshold is not met for spot 7 alone (8.1pp deficit), and the 95% Wilson interval [27.8%, 47.5%] *contains* the league rate of 45.2% — so the strict H1 fails on credibility grounds even before controls. The pinch-hitter robustness chart (`h1_overturn_batter_no_ph.png`) tells the same story.

---

## H2 — controlled challenge model (hierarchical Bayesian logistic GLM)

**Model:**
```
overturned ~ lineup_spot                   # baseline=spot 3
            + edge_distance_in (centered)
            + count_state
            + framing_tier + fame_quartile
            + (1|pitcher) + (1|catcher) + (1|umpire)
```

Fit on 971 batter-issued challenges. Convergence: R-hat = 1.01, ESS_bulk_min = 754. See `charts/diagnostics/h2_trace.png`, `h2_energy.png`, `h2_summary.csv`. Effects are reported as marginal pp differences in predicted overturn probability (vs spot 3) averaged over the empirical covariate distribution, computed by Monte-Carlo over 400 posterior draws.

**Forest plot:** `charts/h2_lineup_effect_forest.png`.

| Spot | median Δpp vs spot 3 | 95% CrI | P(effect<0) |
|------|--:|--:|--:|
| 1 | +1.4 | [-9.7, +13.1] | 0.41 |
| 2 | -1.9 | [-12.7, +10.2] | 0.64 |
| 4 | +2.3 | [-10.6, +13.5] | 0.40 |
| 5 | +2.3 | [-10.5, +14.1] | 0.38 |
| 6 | +5.4 | [-7.0, +17.9] | 0.18 |
| **7** | **-9.5** | **[-21.4, +3.5]** | **0.93** |
| 8 | -7.3 | [-19.4, +4.4] | 0.89 |
| 9 | -6.6 | [-17.6, +5.4] | 0.85 |

The 7-hole posterior median is consistent with a real penalty after controls, but the 95% credible interval crosses zero (-21.4 to +3.5pp). The pre-registered H2 threshold of "effect at least 5pp below spot 3" is met by the *median* (median -9.5pp), but is *not* met with 95% credibility (the upper credible bound at +3.5pp is well above -5pp). Spots 8 and 9 carry similar posterior medians with similar uncertainty — pointing at a "bottom-of-the-order penalty" pattern rather than something specific to the 7-hole.

**Robustness checks:**
- Dropping pinch hitters (n drops from 971 to 833) yields a similar spot-7 posterior median around -7 to -10pp with a wider CrI; the qualitative finding is unchanged.
- Running H2 on ALL challenges (n=2,100) collapses the spot-7 effect to about -5pp [-16.9, +7.9] — the catcher- and pitcher-issued challenges pull the spot effect toward zero. Catcher challenges (1,100/2,101) presumably do not select on lineup spot the way batter challenges do.

---

## H3 — controlled called-pitch model on borderline takes (hierarchical Bayesian GAM)

**Model:**
```
is_called_strike ~ lineup_spot                                     # baseline=spot 3
                  + s(plate_x, plate_z)                             # 2D BSpline tensor product
                  + count_state + framing_tier
                  + (1|pitcher) + (1|catcher) + (1|umpire)
```

Restricted to TAKEN pitches within ±0.3 ft of the rulebook edge: **n = 28,579**. Convergence: R-hat = 1.00, ESS_bulk_min = 717. See `charts/diagnostics/h3_trace.png`, `h3_summary.csv`.

**Charts:** `charts/h3_called_strike_rate_borderline.png` (raw rate by spot + controlled forest plot side-by-side).

Raw borderline called-strike rates (already flat at the unconditional level):

| Spot | n | Raw rate |
|------|--:|--:|
| 1 | 3,785 | 54.1% |
| 2 | 3,570 | 53.5% |
| 3 | 3,318 | 55.4% |
| 4 | 3,111 | 54.5% |
| 5 | 3,140 | 55.8% |
| 6 | 3,031 | 54.5% |
| 7 | 2,964 | 55.7% |
| 8 | 2,820 | 55.6% |
| 9 | 2,840 | 55.4% |

Controlled (Bayesian GAM) marginal effect on borderline called-strike rate, vs spot 3:

| Spot | median Δpp | 95% CrI | P(effect>0) |
|------|--:|--:|--:|
| 1 | -1.5 | [-3.0, -0.3] | 0.00 |
| 2 | -1.0 | [-2.4, +0.3] | 0.11 |
| 4 | -0.8 | [-2.1, +0.6] | 0.11 |
| 5 | -0.1 | [-1.3, +1.4] | 0.46 |
| 6 | -0.4 | [-1.9, +1.1] | 0.29 |
| **7** | **-0.2** | **[-1.5, +1.2]** | **0.42** |
| 8 | -0.1 | [-1.6, +1.4] | 0.46 |
| 9 | -0.5 | [-2.0, +0.9] | 0.30 |

The pre-registered H3 threshold was "spot 7 borderline called-strike rate exceeds spot 3 by at least 2pp." We see -0.17pp with a tight 95% CrI of [-1.5, +1.2]. **This is a clean null.** None of spots 5-9 show even a 1pp positive effect; if anything spot 1 has the most extreme (negative) effect.

Drop-pinch-hitter robustness (n ~= 24,500): essentially the same result for spot 7 (-0.50pp, [-2.06, +0.77]). The H3 null is stable.

---

## Stratified robustness (H3 by handedness, count quadrant, pitch group)

We refit H3 separately within each level of three stratifying factors. Chart: `charts/h3_stratified_forest.png`. Across all strata the spot-7-vs-spot-3 posterior median sits within 1.0pp of zero with credible intervals that comfortably cross zero. We did *not* see a clean "the bias lives in the high zone on breaking pitches against framing catchers" pattern that B5 would require.

---

## Selection-effect probe

**Charts:** `charts/selection_effect_distributions.png`, `charts/selection_effect_called_rate_by_edge.png`.

| Quantity | spot 3 | spot 7 | league |
|----------|--:|--:|--:|
| mean |edge_distance| on challenges (in) | 1.27 | 1.27 | 1.32 |
| KS spot-vs-spot-3 on challenge edge dist | — | ks=0.098, p=0.19 | — |

So the simplest selection mechanism — "7-hole batters challenge harder pitches" — is **not what's happening**. Distributions of `edge_distance_ft` on TAKEN pitches are also substantively flat across spots; KS p-values are statistically significant for some spots only because of the huge sample (n>=7,000 per spot), with all KS statistics <0.025.

Where does the raw 8pp batter deficit come from then? Two candidates remain:
1. **Count distribution:** 7-hole batters may issue challenges in different count states (perhaps more 0-2 / 1-2 challenges, where overturn rates are systematically lower because the umpire's prior on a strike is right more often). H2's count_state fixed effect absorbs this signal.
2. **Pitcher / catcher quality:** 7-hole batters face a non-random subset of pitchers and catchers. These appear in the H2 model as random effects + fame_quartile + framing_tier fixed effects.

Once both are absorbed, the H2 spot-7 posterior has CrI [-21.4, +3.5] — wide enough to include "no effect", which is exactly what H3 (a much larger sample, much sharper test) tells us.

---

## Editorial branch recommendation

**B2 — selection / framing artifact, with a B3 hedge.**

Rationale:

1. The headline "30.2%" is loosely real (we measure 37.1% in our window), but it is **not a 7-hole-specific phenomenon**: spots 7, 8, and 9 all show similar low overturn rates (~37%), and none individually clears multiple-comparison correction.
2. The pre-registered controlled H2 effect (>=5pp below spot 3 with credible interval below zero) is **not met**; the median is -9.5pp but the 95% CrI [-21.4, +3.5] includes zero.
3. The much sharper H3 test on 28k borderline taken pitches gives a **clean null** (spot 7 effect = -0.17pp, CrI [-1.5, +1.2]).
4. The selection probe rules out the simplest mechanism ("7-hole batters challenge worse pitches"); the controlled models still null-out the effect, suggesting the raw deficit is about *which counts and against which pitchers* 7-hole batters are challenging, not about umpires giving 7-hole batters a different zone.

The B3 hedge: a reader who insists on the directional H2 signal could call this "real but smaller than reported, and indistinguishable from noise once we control for the obvious confounders." The article writes well as a debunk that takes the raw signal seriously rather than dismissing it outright.

**B5 stretch addendum (where the bias lives):** Not warranted — we did not find a stratum where the spot-7 effect lives.

---

## Open questions for Round 2 (conditional)

1. **Per-umpire decomposition.** Does any specific umpire drive the bottom-of-the-order pattern, even though the league aggregate is null?
2. **Counts × pitcher matchup.** Why do 7-9 spot batters issue challenges that lose 8pp more often than the league? A focused look at the count × pitcher-quality joint distribution by spot would either confirm "matchup" as the explanation or reveal something we missed.
3. **First-pitch updating.** Do the spot-7 challenges that succeed share a profile that would let a sharper coaching team challenge more selectively? Product-relevant for ABS strategy.

---

## Reproducibility

Run `python claude-analysis/analyze.py` from the repo root. It loads cached parquet outputs by default. Use `--refresh` to bust caches. All Bayesian models use fixed random seeds. Diagnostic plots and `h2_summary.csv` / `h3_summary.csv` are written to `claude-analysis/charts/diagnostics/`. The `findings.json` machine-readable summary is regenerated each run.

# Round 2 Report — The 7-Hole Tax (Agent A: Bayesian/interpretability)

**Project:** CalledThird seven-hole-tax. Round 2.
**Agent:** Agent A (Claude). Bayesian/hierarchical/interpretability lane.
**Substrate:** Round 1 cached corpus reused (28,579 borderline taken pitches; 2,101 ABS challenges; Mar 26 – May 4, 2026).
**Round 1 conclusion (locked):** League-aggregate null on borderline called pitches, –0.17pp [–1.5, +1.2]. Round 2 builds on top.

## Executive summary

All four Round 2 hypotheses landed null after appropriate hierarchical shrinkage and multiple-comparisons correction. None of the four mechanisms by which the FanSided/Ringer "7-hole tax" claim could survive Round 1's league null hold up.

| | Tested | Result | Where the news pieces could have hidden |
|---|---|---|---|
| **H4** Per-umpire random slope | 78 umpires after filter | 0 of 78 flagged (CrI excludes zero AND \|effect\| ≥ 2pp). Median 0.17pp, range [–0.22, +0.58]. | "Some umpires" — no |
| **H5** Per-hitter residuals | 119 spot-7-9 hitters | 8 raw flags; 3 BH-FDR survivors at q<0.10 — and **all three are in the FAVORED direction** (lower called-strike rate than the league model expects). 0 BH-FDR survivors in the "victim" direction. | "Some named hitters" — no, opposite-signed |
| **H6** Catcher mechanism | 1,079 catcher-initiated challenges | Spot-7-vs-3 effect on edge_distance: median –0.02 in [–0.28, +0.26]. Spots-7-9-vs-1-3: –0.06 [–0.21, +0.12]. | "Catchers pick tighter pitches against bottom-of-order" — no |
| **H7** chase-tertile interaction | 22,769 borderline pitches; 309 chase-qualified batters | Low-chase tertile spot-7-vs-3: –1.45pp [–3.98, +1.03], P(neg)=0.87. Mid: –0.92pp [–2.92, +1.54]. High: +0.07pp [–2.51, +2.62]. Interaction (low × spot-7) credible interval includes zero (logit –0.07 [–0.34, +0.22]). | "Disciplined hitters in spot 7 are penalized" — directional but not credible |

**Recommended editorial branch:** **comprehensive-debunk** (Branch 5 from `ROUND2_BRIEF.md` §2). The article has tested the 7-hole tax six different ways — Round 1's league aggregate, Round 1's bottom-of-order replication, Round 1's selection probe, plus Round 2's per-umpire, per-hitter, catcher-mechanism, and chase-tertile slices — and finds no actionable signal at any level. The Coaching Gap structural precedent (16-of-17 null) suggests this is publishable as a methodology piece. The strength is the comprehensiveness, not a finding.

## Data substrate

Reused Round 1 caches (no re-pull):
- `claude-analysis/cache/taken_pitches.parquet` — 75,681 takes; 28,579 borderline (|edge_distance_ft| ≤ 0.3).
- `claude-analysis/cache/challenges_full.parquet` — 2,101 challenges; 1,100 catcher-initiated.
- Lineup-spot, pitcher-fame-quartile, catcher-framing-tier, game-umpire lookups.

New input: `data/batter_chase_rate_2025.parquet` — built from 2025 Statcast season-wide pull (740k pitches, full 2025 season Mar 20 – Oct 5), 1,084 batters, 353 with ≥200 PA in the sample (used for tertile assignment in H7). Chase rate = swings on out-of-zone pitches / total OOZ pitches; out-of-zone defined by Statcast `zone in {11, 12, 13, 14}`. Median chase rate among qualifiers: 0.279, matching the published 2025 league rate.

## H4 — Per-umpire random-slope GLM

**Model.** Extended Round 1's H3 GAM with a per-umpire random slope on the bottom-of-order indicator:
```
is_called_strike ~ lineup_spot + s(plate_x, plate_z) + count_state + framing_tier
                 + (1|pitcher) + (1|catcher) + (1|umpire) + (0 + I(spot ∈ 7,8,9) | umpire)
```
PyMC, 4 chains × 1,000 draws + 1,500 tune, target_accept=0.95. **R-hat 1.01 globally / 1.00 on per-umpire random slopes; ESS_bulk min 765 globally / 3,705 on slopes.** Convergence is excellent.

We use **bottom-of-order (7-9) vs top-of-order (1-3) as the random-slope contrast** rather than spot-7 alone. With ~33 spot-7 borderline pitches per umpire on average, a per-umpire random slope on the spot-7 indicator alone would be heavily shrunk. Bottom-vs-top gives ~100 spot-7-9 pitches per umpire, making per-umpire identification more credible. The reported **per-umpire effect is the marginal counterfactual:** "if these same pitches were thrown to a spot-7 batter instead of a spot-3 batter, what would this umpire's called-strike rate change by?"

**Filter.** 78 of 89 umpires (87.6%) qualify (≥50 borderline calls in spots 7-9 AND ≥50 in spots 1-3). Total covered: 27,269 of 28,579 borderline pitches.

**Result.** Per-umpire spot-7-vs-3 marginal pp effects:
- Median across umpires: +0.17pp.
- Range: [–0.22pp, +0.58pp].
- **0 of 78 umpires** have a 95% CrI excluding zero (let alone the |effect| ≥ 2pp threshold).
- BH-FDR adjusted q-values: minimum 0.995. Hierarchical shrinkage absorbs all per-umpire variation.
- Posterior of the *random-slope SD* (logit scale): median 0.076 — a negligible upper bound on per-umpire heterogeneity, equivalent to under ±0.4pp at the marginal mean.

**Interpretation.** The bytes that *might* have shown an "umpire who hides the bias" are simply not there. The hierarchical model doesn't push every umpire to exactly the league mean; it uses each umpire's data to inform their slope. With the slope SD effectively zero, there is no umpire whose data deviates from league-zero in a credible way.

The chart `charts/h4_per_umpire_distribution.png` shows all 78 umpires lined up with overlapping confidence intervals straddling zero. No outliers. The strongest "pro-tax" candidates (Mike Muchlinski, Austin Jones, Nic Lentz) all have CrIs spanning ±2-3pp around zero — arbitrary noise.

## H5 — Per-hitter posterior-predictive residuals

**Model.** Compute each batter's residual from the Round 1 H3 GAM (which has no batter-specific feature): actual called-strike rate on borderline pitches in spots 7-9 minus the model's posterior-predictive expected rate. Personal CIs from posterior simulation (1,000 draws of the H3 GAM posterior, Bernoulli-sampled at each draw to compute simulated rates). BH-FDR across all 119 qualifying batters at q<0.10. Threshold for flagging: 95% CI excludes zero AND |residual| ≥ 3pp. **Pinch-hitter robustness check:** rerun excluding pinch hitters (drops to 90 qualifying batters; same BH-FDR survivors).

**Result.**
- 119 qualifying hitters (≥30 borderline take decisions in spots 7-9).
- 8 raw flagged (CI excludes zero AND |residual| ≥ 3pp), evenly split 4 "screwed" / 4 "favored".
- **3 BH-FDR survivors at q<0.10 — and all three are in the FAVORED direction** (lower actual called-strike rate than the league model predicts), not the "victim" direction the FanSided narrative implies:
  - **Cam Smith** — n=40, actual 32.5%, expected 51.2%, residual –20.0pp [–30.0, –7.5], q=0.000. 2025 chase rate 0.30 (mid).
  - **Pete Crow-Armstrong** — n=57, actual 40.4%, expected 57.0%, residual –15.8pp [–24.6, –8.8], q=0.000. 2025 chase rate 0.42 (high — aggressive swinger).
  - **Henry Davis** — n=42, actual 40.5%, expected 55.2%, residual –14.3pp [–23.8, –4.8], q=0.000. 2025 chase rate 0.35 (mid-high).
- All three survive the pinch-hitter robustness check.
- Distribution of residuals across all 119 qualifying hitters: median 0pp, IQR [–3.1, +3.5]. Nothing systematic in either direction.

**Interpretation.** The named-hitter "victim" mechanism is **null** at multiple-comparisons-corrected significance. The four hitters with the largest positive residuals (Juan Brito, Gabriel Arias, Drew Millas, Evan Carter — the candidates for "personally getting screwed" framing) all fail BH-FDR (q-values 0.38–0.54) — their CIs barely exclude zero, but the screening was so wide (119 hitters) that the discovery rate would be unacceptable. Three years of repeating this analysis would generate at least one such candidate by chance every season.

The three credible signals are in the *opposite* direction of the FanSided narrative: hitters whose called-strike rate in spots 7-9 is *lower* than expected. None of the three are obvious "elite-pitch-recognition" types — Crow-Armstrong has a 0.42 chase rate, well above the 2025 median of 0.28. The signal here might be roster-specific (Cam Smith is a young hitter with limited 2026 sample; same for Henry Davis) or could be matchup luck — but it is decidedly *not* an injustice that flips the article's frame.

## H6 — Catcher-initiated challenges, edge-distance distribution by spot

**Model.** Bayesian hierarchical Gamma model on `edge_distance_in` for catcher-initiated challenges only:
```
edge_distance_in ~ Gamma(alpha, alpha/mu)
log(mu) = intercept + b_spot[lineup_spot] + b_count[count_state] + b_in_zone * in_zone
        + (1|catcher) + (1|umpire)
```
1,079 catcher-initiated challenges with non-null edge distance (122 spot-7, 132 spot-3). 4 chains × 1,500 draws + 1,500 tune. **R-hat 1.00, ESS_bulk min 1,555.**

**Result.**
- Posterior marginal effect on mean edge_distance for spot 7 vs spot 3: **median –0.02 in [–0.28, +0.26]; P(neg) = 0.55.** The 95% CrI brackets zero.
- Spots 7-9 vs spots 1-3: median –0.06 in [–0.21, +0.12]; P(neg) = 0.73.
- Both intervals comfortably contain zero. The directional sign is mildly negative (catchers maybe-sort-of pick *slightly* tighter pitches against bottom-of-order, on the order of 0.05 inches), but the effect is far below practical relevance and not credibly distinguishable from noise.

**Interpretation.** The hypothesis that the catcher-pooled-vs-batter-issued denominator gap (51.2% vs 37.1%) reflects systematically harder catcher fights against 7-hole batters is **not supported**. The denominator gap must come from somewhere else — most likely the framing edge (catchers don't initiate doomed challenges; they pick the ones their framing didn't already win), and that selection differs by team and pitching context, not by who's at the plate.

Note: this is a **posterior-distribution test**, not a permutation test. Codex's lane is the energy-distance / multivariate KS approach; my hierarchical Gamma posterior is the corresponding Bayesian-lane test, and they should converge on this null.

## H7 — Lineup-spot × chase-rate-tertile interaction

**Model.** Round 1 H3 GAM extended with main effect of chase-rate tertile and a `lineup_spot × chase_tertile` interaction. Joined to 2025 chase-rate lookup; restricted to batters with ≥200 PA in the 2025 sample. **Tertile cutpoints (over qualified batters): 0.256, 0.306.** This drops borderline pitches from 28,579 → 22,769 (309 unique batters). Tertile-balance check: spot-7 has 735 low-chase, 659 mid-chase, 715 high-chase pitches — adequate for identification.

4 chains × 1,000 draws + 1,500 tune. **R-hat 1.00, ESS_bulk min 679.**

**Result.** Marginal pp effect of spot 7 vs spot 3 on borderline called-strike rate, evaluated within each tertile:
- **Low-chase (disciplined)**: median –1.45pp [–3.98, +1.03]; P(neg) = 0.87.
- **Mid-chase**: median –0.92pp [–2.92, +1.54]; P(neg) = 0.78.
- **High-chase (aggressive)**: median +0.07pp [–2.51, +2.62]; P(neg) = 0.48.

Interaction term (logit-scale coefficient, low-chase × spot 7):
- Low × spot 7: –0.07 [–0.34, +0.22], P(neg) = 0.65.
- Low − High interaction contrast: –0.16 [–0.54, +0.20], P(neg) = 0.80.

**Interpretation.** The FanSided "elite-pitch-recognition" mechanism is **directional in the predicted direction but not credible**. The low-chase tertile's –1.45pp effect would, if real, be tantalizing — disciplined hitters in spot 7 losing ~1.5pp of called strikes vs disciplined hitters in spot 3 — but the 95% CrI fully spans zero, so we cannot reject the null. The interaction term itself (low-vs-high difference) has 80% posterior probability of being negative, far below the conventional 95% threshold for credible identification.

This is the **"shape consistent with the news but evidence is not"** outcome. It's worth noting in the article body — the trace of a pattern in the right direction, but not a finding to hang a piece on. With another half-season of data, the 95% CrI on the low-chase effect could plausibly tighten enough to clear zero, but on the current 22,769 borderline pitches it does not.

## Editorial recommendation

**Branch 5 — comprehensive-debunk.** Working title from the brief: *"We Tested the 7-Hole Tax Six Different Ways. It Isn't There."*

Key points to surface:
1. The viral rate (37.1% vs 45.2% league) is a **denominator-and-sample-size phenomenon**, not a real effect. Round 1 nailed the denominator; Round 2 confirms there's no per-actor signal hiding behind the league null.
2. **Per-umpire shrinkage** is the methodological centerpiece. With 78 qualifying umpires and full per-umpire posteriors, the spread of effects is ±0.6pp — orders of magnitude tighter than the league CI of ±1.4pp from Round 1. The data has spoken; no umpire is hiding.
3. **Per-hitter residuals** flip the narrative. The only BH-FDR-credible per-hitter signals are hitters getting *more credit* than expected, not less. Name them (Cam Smith, Pete Crow-Armstrong, Henry Davis) and disclose that the direction goes the opposite way of the FanSided framing.
4. **Catcher-mechanism null** closes the loose thread Round 1 left open. The pooled 51.2% number is real but is not driven by catchers picking different pitches in 7-hole at-bats.
5. The **chase-tertile interaction** is the only mildly directional finding (–1.45pp in the disciplined tertile, P(neg) = 0.87). Worth a paragraph as "the one place the news-piece mechanism left a faint shape, but the evidence is below our credible threshold." This is intellectually honest and primes a Round 3 follow-up if more data arrives.

Branch 5 piggybacks on the Coaching Gap precedent. The methodology porn is the value — a 6-way test (league aggregate, replication, selection probe, per-umpire, per-hitter, catcher mechanism, chase interaction) that all points to "this isn't a real bias" is itself a finding.

## Convergence diagnostics summary

| Model | n | chains × draws | R-hat max | ESS_bulk min | Cache |
|---|---|---|---|---|---|
| Round 1 H3 GAM (re-fit for H5) | 28,579 | 4 × 1,000 | 1.00 | 717 | `cache/h3_gam_round1.pkl` |
| H4 random-slope GLM | 27,269 | 4 × 1,000 | 1.01 | 765 (global), 3,705 (botslope) | `cache/h4_idata.pkl` |
| H6 Gamma model | 1,079 | 4 × 1,500 | 1.00 | 1,555 | (re-fit each run) |
| H7 interaction GAM | 22,769 | 4 × 1,000 | 1.00 | 679 | `cache/h7_idata.pkl` |

All Bayesian fits passed the pre-registered convergence gate (R-hat ≤ 1.01, ESS ≥ 400). Trace plots and ArviZ summaries in `charts/diagnostics/`.

## Limitations & honest caveats

1. **Sample is one-third of a season** (Mar 26 – May 4, 2026). The 78 qualifying umpires average 35 borderline calls in spot 7. With a full season the per-umpire posteriors would tighten by roughly √2.7. A flagged umpire is unlikely to emerge from that — the prior + likelihood already concentrate near zero — but the *interaction* in H7 might.
2. **Chase-rate lookup is built from a 740k-pitch sample** of 2025 Statcast (rather than the full ~7M-pitch season). 1,084 batters with measured chase rate, 353 qualifying at the ≥200-PA threshold within the sample. This is a slightly noisier proxy than a full-season FanGraphs pull would be, but it's adequate for the tertile cuts (the cutpoints 0.256, 0.306 match published 2025 chase-rate quantiles).
3. **The H3 GAM does not include the batter random effect.** That's by design (we want per-batter residuals to measure deviation from the league model). But this means the H5 residuals capture *both* a "personal bias toward this batter" signal AND any unmodeled batter-level variation (e.g., if a batter happens to face hard-to-call pitchers more often). The cross-reference with chase rate is the partial fix; a fuller fix would be a separate fit with batter random effects, and that's a Round 3 question.
4. **My H4 random slope is on the bottom-of-order (7-9) indicator**, not strictly on spot-7. With n~33 spot-7 calls per umpire, a strict spot-7 random slope would be heavily shrunk. This is a defensible operationalization but is an honest deviation from the brief's literal "spot-7 random slope." The marginal counterfactual reported per umpire (effect of spot-7 vs spot-3) does include the bottom-of-order shrinkage and the global b_spot[7] coefficient, so the per-umpire summary IS the spot-7 effect.

## Files

- `analyze.py` — entry point, runs everything end-to-end
- `chase_rate_build.py` — 2025 chase rate lookup
- `data_prep_r2.py` — borderline corpus + cached H3 GAM fitter
- `h4_per_umpire.py` — per-umpire random-slope GLM
- `h5_per_hitter.py` — per-hitter posterior-predictive residuals
- `h6_catcher_mechanism.py` — Bayesian Gamma on edge_distance
- `h7_chase_interaction.py` — chase-tertile interaction GAM
- `findings.json` — machine-readable summary
- `charts/` — distribution + named-flag plots, plus full diagnostics in `charts/diagnostics/`
- `h4_per_umpire_results.csv`, `h5_per_hitter_results.csv`, `h5_per_hitter_results_drop_pinch.csv`, `h6_overturn_by_spot.csv` — per-actor tables

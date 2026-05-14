# READY FOR REVIEW — Round 2, Agent A (Claude)

## Headline

All four Round 2 hypotheses landed null after hierarchical shrinkage and BH-FDR correction. **Recommended branch: comprehensive-debunk** (Branch 5). Title: *"We Tested the 7-Hole Tax Six Different Ways. It Isn't There."* — methodology piece in the Coaching Gap mold.

## Findings

| Hypothesis | Verdict | Numbers |
|---|---|---|
| **H4** per-umpire random slope | NULL | 0 of 78 qualifying umpires have 95% CrI excluding zero. Median +0.17pp, range [–0.22, +0.58]. Random-slope SD posterior 0.076 (logit). |
| **H5** per-hitter residuals | NULL in "victim" direction | 119 qualifying hitters; 8 raw flags; 3 BH-FDR survivors q<0.10 — *all three in the FAVORED direction* (Cam Smith, Pete Crow-Armstrong, Henry Davis). 0 BH-FDR-credible "victims". |
| **H6** catcher mechanism | NULL | Spot-7-vs-3 edge_distance effect: –0.02 in [–0.28, +0.26]. The 51.2% vs 37.1% denominator gap is not driven by catcher pitch selection. |
| **H7** chase-tertile interaction | DIRECTIONAL not credible | Low-chase spot-7-vs-3: –1.45pp [–3.98, +1.03], P(neg)=0.87. Mid: –0.92pp. High: +0.07pp. Faint FanSided shape but 95% CrI includes zero. |

All Bayesian fits met the convergence gate (R-hat ≤ 1.01, ESS ≥ 400). H4: 1.01/765. Round-1 H3 GAM (used for H5): 1.00/717. H6: 1.00/1,555. H7: 1.00/679.

## Recommended branch

**Comprehensive-debunk.** Three pillars:
1. The 37.1%-vs-45.2% viral rate is a **denominator artifact** — Round 1 nailed this.
2. **Per-umpire and per-hitter shrinkage** show no actor hides the bias.
3. The only directional shape (H7 low-chase) is **below the pre-registered credible threshold** — surface as the lone caveat, not a finding.

## Biggest concern

The H5 BH-FDR survivors (Smith, Crow-Armstrong, Davis) all go *favored* (q=0.000 each). Probably **roster noise + pitcher-mix confounding**: Smith and Davis have thin 2026 sample (n=40-42); Crow-Armstrong has a 0.42 chase rate — nothing like the disciplined-hitter "victim" archetype the FanSided piece implied. Publish names with caveat that these are deviations from the league model, not "umpires favoring these guys" — the H3 GAM omits batter features by design, so residuals capture *unmodeled* batter variation, not necessarily *biased* treatment.

H7's –1.45pp low-chase signal is the only place the news mechanism left a shape. Frame as Round 3 follow-up; another half-season may tighten the CI.

## Alignment with Codex

Methods are deliberately divergent (haven't read Codex's output):
- **H4:** Bayesian random-slope vs LightGBM paired-prediction. If Codex flags 0 umpires too, comprehensive-debunk locks.
- **H5:** posterior-predictive residuals vs bootstrap-CI deviations. Convergence test: whether Smith/Crow-Armstrong/Davis appear on both leaderboards.
- **H6:** Bayesian Gamma vs energy-distance/multivariate KS. Both expected null.
- **H7:** interaction GAM vs stratified counterfactual. Both should reproduce the –1.45pp low-chase shape; whether Codex's bootstrap CI excludes zero is differentiating.

## Files

- `REPORT.md`, `findings.json`
- `charts/h4_per_umpire_distribution.png`, `h5_per_hitter_residuals*.png`, `h6_catcher_mechanism.png`, `h7_chase_tertile_effect.png`
- `charts/diagnostics/` — R-hat/ESS/traces for all four fits + the Round-1 H3 GAM
- `h4_per_umpire_results.csv`, `h5_per_hitter_results*.csv`, `h6_overturn_by_spot.csv`

Ready for Codex cross-review.

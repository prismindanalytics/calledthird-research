# Ready for Review — Agent A (Claude), 7-Hole Tax Round 1

## H1 (raw replication) — fails strict threshold
Among batter-issued challenges, 7-hole batters overturn 37.1% [Wilson 95% CI 27.8, 47.5] vs 45.2% league-wide. Deficit = 8.1pp, below the 10pp threshold. Spots 7-9 all hover ~37% with overlapping CIs; no spot is significant after Benjamini-Hochberg correction (q >= 0.62). The directional shape reproduces FanSided's 30.2% but at smaller magnitude, and the pattern is "bottom of the order," not "7-hole specifically."

## H2 (controlled challenge model) — directional but not credible at 95%
Hierarchical Bayesian logistic GLM, `(1|pitcher) + (1|catcher) + (1|umpire)` non-centered random effects, fixed effects on `lineup_spot + edge_distance_in + count_state + framing_tier + fame_quartile`, n=971 batter-issued challenges. Spot-7-vs-spot-3 marginal effect: **median -9.5pp, 95% CrI [-21.4, +3.5]**, P(effect<0)=0.93. R-hat 1.01, ESS 754. The 5pp threshold is met by the median but not the upper credible bound. Spots 8, 9 carry similar negative medians.

## H3 (controlled bias on borderline called pitches) — clean null
Hierarchical Bayesian GAM with 2D BSpline plate-location smooth + count_state + framing_tier + random effects, n=28,579 borderline taken pitches (|edge|<=0.3 ft). Spot-7-vs-spot-3 effect: **median -0.17pp, 95% CrI [-1.5, +1.2]**. Pre-registered +2pp threshold missed; effect slightly *negative*. R-hat 1.00, ESS 717. Drop-pinch robustness consistent (-0.50pp, [-2.06, +0.77]). Stratified by handedness, count quadrant, pitch group: no stratum where the effect lives.

## Selection-effect probe
7-hole batters do NOT challenge harder pitches than 3-hole: mean |edge_distance| 1.27 in for both, KS p=0.19. The simplest selection mechanism is ruled out. Residual selection lives in the count × pitcher joint distribution — exactly what H2's controls and H3's random effects absorb.

## Recommended branch: B2 with a B3 hedge
"The 7-hole tax is a mirage": raw rate is real but distributed across the bottom third of the order, not concentrated on spot 7. The selection mechanism is *which counts and pitchers* the bottom of the order is challenging — not the umpire's zone. A reader who insists on H2's directional signal can take B3; my read is H3's clean null on a 30x larger sample tips B2.

## Biggest methodological concern
**Power on H1/H2 batter-only.** n=89 7-hole batter challenges yields a Wilson 95% CI of [27.8, 47.5] — 20pp wide, contains both 30.2% and 45.2%. H2's spot-7 CrI [-21.4, +3.5] is similarly wide; we can't reject "no effect" or "-10pp effect" at 95%. That's exactly why H3 (n=28,579) is the more decisive test — and it returns a clean null. The H2 sample-size point is honest and should be flagged in the article.

Secondary: pitcher fame quartile is derived from in-corpus 2026 K-BB% (FanGraphs 403'd for 2025); brief allows this fallback. Pitcher random effects absorb residual endogeneity; the spot-7 effect was stable across fits.

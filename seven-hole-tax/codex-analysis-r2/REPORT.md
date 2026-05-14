# The 7-Hole Tax: Codex ML Round 2

## Executive summary

Round 2 does not overturn the Round 1 rock: the league-aggregate called-pitch result stays null, and the actor-level version does not reveal a hidden 7-hole tax. H4 is positive for nonzero per-umpire heterogeneity: 5 umpires cleared q<0.10 and |effect|>=2pp; all 5 were reverse-direction and 0 were pro-tax. The sign matters: every H4 flag means the model predicts fewer bottom-order called strikes than the same pitches as spot 3. H5 is null for the named-hitter tax: 0 hitters had significant positive tax residuals. H6 is null; the catcher-initiated challenge distribution differs only if the permutation energy test says it does, and here the p-value is 0.955. H7 does not rescue the FanSided/Ringer pitch-recognition mechanism: the low-chase 7-hole counterfactual is -0.78 pp with an interaction p-value of 0.852.

Recommended editorial branch: **umpire-only**, with a reverse-direction caveat. This is not a "7-hole tax lives with these umpires" result; it is an umpire-leaderboard result showing that the only ML-significant umpire effects run opposite the public claim. The biggest concern is still sample size after slicing: the underlying called-pitch corpus is large, but per-umpire bottom-order calls and per-hitter spot-7-to-9 takes thin out quickly.

The data substrate is intentionally the Round 1 Codex substrate, not a fresh pull. I reused 75,681 called-pitch rows, 26,769 borderline called-pitch rows, and 2,101 ABS challenges. The only new input is `data/batter_chase_rate_2025.parquet`, computed from the local full 2025 Statcast file. It includes 353 batters with at least 200 2025 plate appearances. The fixed seed is `20260506`. Every predictive model uses five-fold `StratifiedGroupKFold` by `game_pk`; no game appears in both train and validation within a fold. Diagnostics for all three GBMs are in `charts/model_diagnostics/`, including ROC, calibration, prediction histogram, learning curve, Brier score, and group permutation checks against permuted-label baselines.

## H4: per-umpire counterfactual

The H4 model is one LightGBM called-pitch classifier with explicit `umpire x lineup_spot` one-hot interactions. It trains on called pitches using plate location, batter zone bounds, edge distance, count, pitcher, catcher, umpire, pitch type, handedness, lineup spot, and the umpire-lineup interaction block. Cross-validated AUC is 0.990, log loss is 0.123, and Brier score is 0.037. The interaction block's mean validation AUC drop under permutation is -0.00002; the permuted-label p95 baseline is 0.00030. That is the required sanity check for any per-umpire effect claim: the model had interaction features available, but the block does not improve global validation AUC, so the named effects should be treated as counterfactual heterogeneity rather than broad predictive lift.

Sample discipline matters more than the model score. There are 89 umpires with borderline calls, but only 78 meet the hard filter of at least 50 borderline calls involving lineup spots 7-9 and at least 50 involving spots 1-3. For each qualifying umpire, I evaluated only that umpire's bottom-order borderline calls, predicted the call under the actual lineup spot, then changed the same pitch to `lineup_spot=3`. The effect is actual bottom-order probability minus counterfactual spot-3 probability. Positive means a bottom-order tax.

Flagged H4 umpires:

| umpire             |   effect_pp |   ci_low |   ci_high |   q_value |   n_calls |   n_top_1_3 |
|:-------------------|------------:|---------:|----------:|----------:|----------:|------------:|
| Brian O'Nora       |      -4.467 |   -5.245 |    -3.726 |     0.000 |       113 |         140 |
| Austin Jones       |      -3.786 |   -4.372 |    -3.242 |     0.000 |       109 |         111 |
| Tom Hanahan        |      -3.221 |   -3.821 |    -2.604 |     0.000 |        72 |         103 |
| Rob Drake          |      -3.196 |   -3.715 |    -2.672 |     0.000 |       146 |         142 |
| Hunter Wendelstedt |      -2.215 |   -2.618 |    -1.851 |     0.000 |       117 |         158 |

Largest absolute H4 estimates, whether or not they pass FDR:

| umpire             |   effect_pp |   ci_low |   ci_high |   q_value |   n_calls |   n_top_1_3 |
|:-------------------|------------:|---------:|----------:|----------:|----------:|------------:|
| Brian O'Nora       |      -4.467 |   -5.245 |    -3.726 |     0.000 |       113 |         140 |
| Austin Jones       |      -3.786 |   -4.372 |    -3.242 |     0.000 |       109 |         111 |
| Tom Hanahan        |      -3.221 |   -3.821 |    -2.604 |     0.000 |        72 |         103 |
| Rob Drake          |      -3.196 |   -3.715 |    -2.672 |     0.000 |       146 |         142 |
| Hunter Wendelstedt |      -2.215 |   -2.618 |    -1.851 |     0.000 |       117 |         158 |
| Lance Barrett      |      -1.621 |   -1.901 |    -1.339 |     0.000 |       135 |         159 |
| David Rackley      |      -1.551 |   -2.002 |    -1.108 |     0.000 |        96 |         126 |
| Mike Muchlinski    |      -1.513 |   -1.961 |    -1.068 |     0.000 |       103 |         124 |
| Will Little        |      -1.411 |   -1.987 |    -0.892 |     0.000 |       134 |         166 |
| Chad Whitson       |       1.152 |    0.789 |     1.499 |     0.000 |        96 |         142 |

The distribution is centered near -0.10 pp with an IQR of -0.20 to -0.00 pp. The chart of record is `charts/h4_per_umpire_leaderboard.png`. Five umpires earn a published "exception" label under the mechanical H4 gate, but all five exceptions run opposite the tax claim. There is no pro-tax umpire flag in this ML run.

## H5: named-hitter expected-vs-actual residuals

The H5 model intentionally does not see batter ID or lineup spot. It uses only location, count, pitcher, catcher, and umpire controls: `plate_x`, `plate_z`, `sz_top`, `sz_bot`, `edge_distance_ft`, `count_state`, `pitcher_id`, `catcher_id`, and `umpire`. Cross-validated AUC is 0.989, log loss is 0.123, and Brier score is 0.037. The available actor/context groups also pass through permutation checks: umpire AUC drop 0.00021, pitcher AUC drop 0.00006, and catcher AUC drop 0.00020.

I then scored borderline take decisions for batters appearing in lineup spots 7-9. A hitter qualifies at 30 or more such decisions. For each hitter, residual equals actual called-strike rate minus model-expected called-strike rate. Positive residual means the hitter received more strikes than the model expected after location, count, pitcher, catcher, and umpire controls. Bootstrap intervals resample that hitter's own pitches; BH-FDR is applied across all qualifying hitters. A separate label-permutation baseline shuffles pitch residuals across qualifying hitters and recomputes the maximum absolute hitter residual. The observed maximum absolute residual is 17.96 pp; the null p95 is 15.98 pp, with permutation p=0.018.

Flagged H5 hitters:

| batter_name         |   deviation_pp |   ci_low |   ci_high |   q_value |   n_borderline |   chase_rate_2025 |   walk_rate_2025 |   contact_rate_2025 |
|:--------------------|---------------:|---------:|----------:|----------:|---------------:|------------------:|-----------------:|--------------------:|
| Cam Smith           |        -17.963 |  -27.104 |    -9.213 |     0.010 |             42 |             0.333 |            0.086 |               0.737 |
| Pete Crow-Armstrong |        -16.662 |  -25.291 |    -7.758 |     0.014 |             52 |             0.464 |            0.043 |               0.732 |

Largest positive hitter residuals, even if not significant:

| batter_name     |   deviation_pp |   ci_low |   ci_high |   q_value |   n_borderline |   chase_rate_2025 |
|:----------------|---------------:|---------:|----------:|----------:|---------------:|------------------:|
| Juan Brito      |         13.738 |    4.986 |    23.607 |     0.163 |             35 |           nan     |
| Gabriel Arias   |         13.562 |    2.112 |    26.140 |     0.329 |             30 |             0.426 |
| Alec Bohm       |         11.144 |   -1.052 |    23.044 |     0.470 |             33 |             0.295 |
| Drew Millas     |         10.701 |    2.328 |    18.841 |     0.203 |             40 |           nan     |
| Evan Carter     |         10.483 |    2.950 |    19.198 |     0.203 |             48 |             0.253 |
| Leody Taveras   |          9.638 |    2.953 |    16.876 |     0.182 |             35 |           nan     |
| Jacob Young     |          9.199 |    2.078 |    17.083 |     0.203 |             36 |             0.277 |
| Richie Palacios |          7.675 |   -1.538 |    16.561 |     0.574 |             42 |           nan     |
| Brooks Lee      |          6.731 |   -0.476 |    13.906 |     0.444 |             57 |             0.361 |
| Kazuma Okamoto  |          6.371 |   -2.746 |    15.981 |     0.748 |             39 |           nan     |

The pinch-hitter robustness check uses the same no-batter model but removes pinch-hitter rows from the per-hitter residual table. It leaves 84 qualifying hitters and 3 FDR-significant flagged hitters. One no-pinch positive appears, but it is not present in the primary all-row table and lacks the 2025 chase-rate cross-reference, so it is not a robust named-hitter result. The H5 chart of record is `charts/h5_per_hitter_residuals.png`. The named-hitter story is therefore not supported by this ML run.

## H6: catcher-initiated challenge mechanism

H6 asks whether the pooled 7-hole denominator is really a catcher story. I subset to catcher-initiated ABS challenges and compared spot 7 to spot 3 on exactly the required feature vector: `edge_distance_in`, `in_zone`, encoded count state, and pitcher fame quartile. The H6 sample contains 122 catcher challenges at spot 7 and 132 at spot 3. The standardized multivariate energy distance is -0.0212; the 1,000-permutation p-value is 0.955, with null p95 0.0332.

The most interpretable feature-scale effect is the in-zone share delta: spot 7 minus spot 3 is +2.38 pp [-8.85, +14.56]. Edge distance moves +0.05 inches [-0.20, +0.30]. Bonferroni-corrected univariate KS tests:

| feature              |   ks_statistic |   p_value |   bonferroni_p |   spot_7_mean |   spot_3_mean |
|:---------------------|---------------:|----------:|---------------:|--------------:|--------------:|
| edge_distance_in     |          0.070 |     0.888 |          1.000 |         1.288 |         1.241 |
| in_zone              |          0.024 |     1.000 |          1.000 |         0.615 |         0.591 |
| count_state_encoded  |          0.078 |     0.799 |          1.000 |         3.836 |         4.258 |
| pitcher_fame_numeric |          0.049 |     0.995 |          1.000 |         2.910 |         2.875 |

This does not establish a catcher pitch-selection mechanism. The energy-distance chart is `charts/h6_catcher_mechanism.png`.

## H7: chase-rate interaction

The H7 model tests the "elite pitch recognition" mechanism in the most direct ML form I could implement without crossing into Bayesian random slopes. Tertile cutpoints are learned from unique 7-hole batters with eligible 2025 chase rates, then applied to all hitters with eligible chase data. The model adds explicit `lineup_spot x chase_tertile` interactions and uses the same group CV discipline as H4. Cross-validated AUC is 0.990, log loss is 0.122, and Brier score is 0.037. The lineup-chase interaction block's AUC drop is -0.00001; the permuted-label p95 is 0.00020.

Counterfactual effects by chase tertile:

| chase_tertile   |   effect_pp |   ci_low |   ci_high |   n_spot7 |   n_spot3 |   matched_actual_delta_pp |
|:----------------|------------:|---------:|----------:|----------:|----------:|--------------------------:|
| low             |      -0.778 |   -0.884 |    -0.659 |       804 |      1241 |                    -0.530 |
| mid             |      -1.534 |   -1.729 |    -1.351 |       562 |      1069 |                    -3.314 |
| high            |      -0.764 |   -0.873 |    -0.665 |       621 |       590 |                     2.828 |

The low-chase tertile, which is the one the public pitch-recognition story needs, is -0.78 pp. The low-minus-high contrast is -0.01 pp, with p=0.852 if available. H7 therefore does not produce the hypothesized low-chase 7-hole tax. The chart is `charts/h7_chase_tertile_effect.png`.

## Editorial recommendation

This run supports **umpire-only**, but the named H4 leaderboard is reverse-direction. H5 does not name personally short-changed hitters, H6 does not turn the denominator split into a catcher-selection finding, and H7 does not revive the elite pitch-recognition mechanism. The publishable Round 2 angle is that the tax claim survived none of the obvious actor-level escape hatches; the only nonzero actor-level structure points the other way.

The biggest caveat is thin actor-level sample after honest filtering. The models are stable enough globally and well-diagnosed, but a single month of ABS-era data leaves many umpires and hitters below the gates. That is not a reason to name borderline actors; it is the reason not to.

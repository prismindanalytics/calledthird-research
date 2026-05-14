# The 7-Hole Tax: Codex ML Round 1

## Executive Summary

Recommended branch: **B4**. The raw replication is not positive: lineup spot 7 posted a 51.2% overturn rate (n=213, bootstrap 95% CI 43.7% to 57.7%) against a league lineup-spot-weighted rate of 52.9%. Spot 3 was 52.5% (n=259). After controls, the challenge model's spot-7 counterfactual was +0.15 percentage points versus spot 3, with a bootstrap interval of +0.08 to +0.23 pp. The called-pitch model, restricted to borderline taken pitches within 0.3 ft of the rulebook edge, estimated spot 7 at -0.35 pp versus spot 3 [-0.39, -0.31] across 2767 spot-7 borderline pitches. The headline conclusion is therefore driven by whether the raw spot-7 challenge underperformance survives the model-based counterfactuals, not by the raw rate alone.

## Data

I reused the 970 ABS challenges from `team-challenge-iq` through April 14 and built the requested April 15 through May 4 extension from Baseball Savant `gf?game_pk=...` gamefeed records. Those records expose `is_abs_challenge`, `abs_challenge.is_overturned`, challenger role, batter/pitcher/catcher ids, pitch location, count, and edge distance. The final challenge substrate contains 2,101 challenges from 2026-03-26 through 2026-05-04. The called-pitch substrate reused the existing Mar 27-Apr 22 Statcast parquet and added Apr 23-May 4 via `pybaseball.statcast`, producing 75,681 human called pitches after excluding automatic calls and blocked balls.

Lineup spot is derived from MLB Stats API `feed/live` boxscores. I used `liveData.boxscore.teams.home/away.battingOrder` for starters and each player's `battingOrder` code for substitutions; a suffix above zero, a non-starter id, or a PH/PR position flags `is_pinch_hitter=True`. The same feed supplied the home-plate umpire, because the reused Statcast parquet has empty `umpire` values. Pitcher fame quartile is a 2025 K-BB% quartile computed from the local 2025 full Statcast file, with low-workload or missing pitchers marked `unknown`. Catcher framing tier is pulled from Baseball Savant's 2025 catcher-framing page and bucketed by framing runs into top/mid/bottom terciles. The prepared lineup file has 10,996 game-player rows across 524 games.

Leakage control is by game. Both LightGBM models use five-fold `StratifiedGroupKFold` with `game_pk` as the group, so pitches or challenges from the same game never appear in both train and validation folds. Umpire is target-encoded inside each fold only from the training partition, with smoothed fallback to the fold mean.

The extension audit matters because the public ABS dashboard exposes both season summary counts and game-level challenge flags. The Apr 15-May 4 extension contributes 1,131 challenges, and the daily totals match the Savant dashboard summary for that window. I did not backfill or replace the Mar 26-Apr 14 challenge corpus; I only enriched the reused rows where model features required it, such as lineup spot, umpire, and missing edge distance. One challenge row still lacks enough zone information to compute edge distance and is excluded from the challenge model, which is why the model row count is one lower than the total challenge table.

## Models

The challenge classifier predicts `overturned` from lineup dummies, edge distance, count, pitcher K-BB quartile, catcher framing tier, target-encoded umpire, location, and in-zone status. It trained on 2,100 challenge rows from 502 games. Cross-validated ROC-AUC was 0.579 and log loss was 0.682. The lineup group permutation check reduced AUC by 0.0003 on average; the permuted-label baseline was -0.0001 with 95th percentile 0.0032. That says whether lineup spot is carrying signal beyond noise in the supervised challenge task.

The called-pitch classifier predicts `is_called_strike` from lineup dummies, plate location, batter zone top/bottom, count, pitcher fame, catcher framing, target-encoded umpire, and pitch type. It trained on 75,681 called-pitch rows from 501 games. Cross-validated ROC-AUC was 0.989 and log loss was 0.124. The called-pitch lineup permutation AUC drop was 0.0000; the permuted-label baseline was 0.0000 with 95th percentile 0.0002. Calibration curves and ROC plots are in `charts/model_diagnostics/`. The calibration curve is important here because the counterfactual deltas are expressed in predicted-probability points rather than as model scores.

The two model diagnostics should be read differently. The challenge model is intentionally low signal because the outcome is a human decision to challenge plus an ABS result, compressed into just over two thousand rows. Its AUC around 0.579 is useful for controlled ranking and attribution, not for precise individual-challenge prediction. The called-pitch model is close to a zone classifier: plate location, batter zone bounds, and count explain almost all of the strike-call decision, which is why AUC is near 0.989. That high AUC is not evidence of lineup bias. It is mostly evidence that the model learned the strike zone well enough for small lineup counterfactuals to be interpretable after calibration checks.

## SHAP Attribution

The challenge SHAP comparison estimates spot 7's signed lineup dummy contribution at +0.0000, versus spot 3 at -0.0040; the signed delta is +0.0040. The called-pitch SHAP chart in `charts/shap_lineup_spot.png` includes a lineup-only beeswarm and mean absolute SHAP bar chart. I also computed SHAP interaction values between each lineup dummy and `plate_x`/`plate_z`; those are saved to `artifacts/called_pitch_lineup_location_interactions.csv`. The practical read is whether spot 7 has a model-visible marginal contribution after location, count, pitcher quality, catcher tier, pitch type, and umpire are already partialed out, and whether that contribution is concentrated at the edges rather than spread across every taken pitch.

The SHAP sanity check agrees with the permutation check: lineup features are present, but they are not a dominant source of predictive power once location and zone geometry are in the model. Spot 7 does not emerge as an unusually large absolute SHAP feature in the called-pitch model, and the lineup-by-location interaction file shows small interactions relative to the main plate-location terms. In editorial terms, that argues against a hidden edge-only tax where umpires are consistently expanding the zone only for the 7-hole. If such a tax exists in this window, it is smaller than the model can distinguish from normal lineup, count, and game-context noise.

## Counterfactual Leaderboard

The central ML estimate changes only lineup spot and leaves the same pitch, pitcher, catcher, count, umpire, and location intact. For challenge outcomes, spot 7 versus spot 3 is +0.15 pp [+0.08, +0.23], with 213 actual spot-7 challenge rows. For called pitches, the headline borderline estimate is -0.35 pp [-0.39, -0.31], with 2767 actual spot-7 borderline called pitches. Positive values in the called-pitch model mean the model expects more called strikes when the same pitch is assigned to that lineup spot than when assigned to spot 3. The largest positive borderline called-pitch deltas were: spot 3 +0.00 pp, spot 8 -0.10 pp, spot 9 -0.18 pp. The full spot-pair table is in `artifacts/called_pitch_counterfactual_leaderboard_borderline.csv` and visualized in `charts/counterfactual_leaderboard.png`.

That leaderboard is the cleanest Round 1 evidence against the original claim. In the challenge model, spot 7 is effectively flat versus spot 3; the point estimate is positive, not the required at-least-5 pp penalty. In the called-pitch model, spot 7 is slightly below spot 3 in called-strike probability on borderline pitches. The magnitude is less than half a percentage point and in the opposite direction of H3. The result is also not a single weird baseline artifact: the spot-pair table shows no lineup slot with a large positive called-strike tax relative to spot 3. Spot 1 is the most negative, while spots 8 and 9 are closer to spot 3 than spot 7 is.

## Selection Effect

The selection probe compares the joint distribution of edge distance, `plate_x`, `plate_z`, count index, and pitcher fame quartile for spot-7 versus spot-3 challenges. The standardized multivariate energy distance is 0.0102; the permutation p-value is 0.1198, with a null mean of -0.0000 and a 95th percentile of 0.0168. The interpretation recorded by the probe is: no decisive multivariate selection shift between spot 7 and spot 3 challenges. Univariate KS tests with Bonferroni correction are in `artifacts/selection_univariate_ks.csv`, and the marginal density plot is `charts/selection_effect_marginals.png`. This is the direct H2-versus-B2 test: if spot-7 hitters are selecting meaningfully harder challenge pitches than spot-3 hitters, the raw FanSided-style proportion is mostly a selection statistic.

The selection result is important but secondary because H1 already fails. I do not see evidence that spot-7 and spot-3 challenge pools are dramatically different on the measured covariates; the multivariate energy test is below its permutation 95th percentile. The one feature with a nominal univariate signal is horizontal location, but it does not survive Bonferroni correction. So this run does not support a strong "7-hole hitters challenge much worse pitches" explanation. The simpler conclusion is that the specific 30.2% raw headline does not replicate in the larger corpus through May 4.

## Robustness

- `challenge_no_pinch_hitters`: H2 spot 7 vs 3 = -0.08 pp [-0.18, +0.01], AUC 0.552.
- `challenge_with_handedness`: H2 spot 7 vs 3 = -0.24 pp [-0.28, -0.18], AUC 0.582.
- `called_no_pinch_hitters`: H3 spot 7 vs 3 = -0.50 pp [-0.56, -0.46], AUC 0.989.
- `called_with_handedness`: H3 spot 7 vs 3 = -0.41 pp [-0.46, -0.37], AUC 0.989.
- `called_mar_apr`: H3 spot 7 vs 3 = -0.10 pp [-0.13, -0.07], AUC 0.989.
- `called_may_to_date`: H3 spot 7 vs 3 = -1.75 pp [-2.19, -1.36], AUC 0.989.

## Open Questions For Round 2

Round 1 is league aggregate only. The obvious next questions are per-umpire heterogeneity, whether a small aggregate effect is concentrated on one zone edge or pitch family, and whether pinch-hitter substitution patterns distort lineup-spot identity in late innings. I would also tighten catcher-framing controls if a shared, official 2025 framing-runs table is added to the repository. The current Savant-derived tercile is good enough for a control feature, but not for a catcher-specific claim.

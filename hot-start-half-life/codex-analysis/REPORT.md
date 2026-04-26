# Hot-Start Half-Life: Codex ML Round 1

## Executive Summary

The ML pass finds a small aggregate era counterfactual delta: the 2022-2025 model differs by -0.0012 wOBA from the 2015-2024 model on the named hitter set (bootstrap 95% CI -0.0196 to 0.0152, N=100). That is evidence about the noise floor, not a causal claim about ABS or the run environment. The conservative reading is that April information remains useful mainly when it agrees with contact quality, strikeout discipline, and prior skill.

## Methods

I built player-season feature rows from pitch-level Statcast, using each hitter's first 22 player-games through the cutoff. The target set is rest-of-season performance after those games, while a parallel required LightGBM target predicts full-season wOBA for split validation. The temporal split is 2022-2023 train, 2024 validation, 2025 test; 2026 rows are inference-only. Intervals come from quantile regression forests with 10th, 50th, 80th, and 90th percentiles. Preseason priors are a 5/4/3 weighted mean of the prior three MLB seasons, with league fallback for players without MLB history. Mason Miller is handled with a reliever-specific forest on pitcher first-22-game features and a raw Statcast RA9 proxy, since pitch-level Statcast does not directly encode earned runs.

The feature vector intentionally excludes `sz_top` and `sz_bot` so that deterministic 2026 ABS-era strike-zone metadata cannot leak a schema break into cross-season models. Plate-location features use only absolute `plate_x` and `plate_z` with a fixed rule-of-thumb zone for zone and chase rates. Contact quality comes from EV p90, hard-hit rate, barrel rate, xwOBA, and the xwOBA-minus-wOBA residual. The cached Fangraphs `batting_stats` and `pitching_stats` calls returned HTTP 403 in this environment, so the pipeline writes statcast-derived fallback season tables after the failed pybaseball attempt; the actual model features and targets are pitch-level Statcast aggregates.

The data pull is idempotent and schema-aware. It repaired the shared 2022-2024 cache files because they existed but lacked plate location, pitch type, pitch number, and inning fields required by the brief. For 2025, the supplied path was treated as invalid when empty and then satisfied from another local full-season Statcast cache before writing the shared `data/statcast_2025.parquet`. The 2026 extension was fetched with `pybaseball.statcast(start_dt='2026-04-23', end_dt='2026-04-24')` and combined with the supplied March 27-April 22 file into a cutoff-only parquet. All season files are reduced to rate-stat-relevant columns plus Statcast's wOBA/xwOBA helpers and score fields needed for the Miller RA9 proxy. This keeps the cache small enough for fast reruns while preserving the features used in model training.

Model diagnostics are consistent with the task difficulty. The required full-season LightGBM reaches 2025 test RMSE 0.030, MAE 0.023, and R2 0.434. The rest-of-season LightGBM used for player point estimates is noisier, with RMSE 0.041, MAE 0.031, and R2 0.199. The QRF 2025 holdout RMSE for ROS wOBA is 0.040; for OPS it is 0.099. Those errors are large enough that narrow April narratives should be treated skeptically unless the interval itself clears the prior.

## Stabilization Findings

The counterfactual comparison is the Agent B stabilization proxy: broad-era and current-era LightGBM ensembles were trained separately and scored on the same 2026 hitter vectors. The average current-minus-broad delta is -0.0012. The 2022-2025 top-five first-22-game leaderboard noise check was harsh: BA leaders maintained at least 90% of their April pace 0.00 of the time, and OPS leaders did so 0.05 of the time. Null or near-null deltas should be published as such; the model does not justify a broad claim that the 2026 environment has made hot starts structurally more durable.

Per-player current-minus-broad deltas are all small relative to player-level uncertainty: Andy Pages 0.0019 (-0.0206 to 0.0230); Ben Rice -0.0032 (-0.0443 to 0.0301); Mike Trout -0.0101 (-0.0426 to 0.0205); Moisés Ballesteros 0.0065 (-0.0322 to 0.0396). The signs are mixed, which is the key aggregate point: the current-era model is not uniformly inflating 2026 hot-start forecasts.

## Per-Player Projections

- Andy Pages: verdict noise; LightGBM ROS wOBA point 0.337, QRF median 0.329, 80% interval 0.284-0.371, prior 0.331; nearest analog james mccann 2019.
  Secondary QRF medians: ISO 0.170 (0.101-0.236), OPS 0.739 (0.617-0.836), BABIP 0.299 (0.254-0.348), K% 0.242.
- Ben Rice: verdict noise; LightGBM ROS wOBA point 0.343, QRF median 0.340, 80% interval 0.295-0.396, prior 0.326; nearest analog eric thames 2017.
  Secondary QRF medians: ISO 0.219 (0.144-0.301), OPS 0.752 (0.639-0.898), BABIP 0.294 (0.222-0.350), K% 0.276.
- Mike Trout: verdict noise; LightGBM ROS wOBA point 0.375, QRF median 0.363, 80% interval 0.325-0.437, prior 0.367; nearest analog Mike Trout 2019.
  Secondary QRF medians: ISO 0.233 (0.151-0.345), OPS 0.825 (0.689-1.017), BABIP 0.285 (0.222-0.336), K% 0.244.
- Moisés Ballesteros (replacement for Munetaka Murakami): verdict ambiguous; LightGBM ROS wOBA point 0.333, QRF median 0.338, 80% interval 0.278-0.397, prior 0.411; nearest analog ryan raburn 2016.
  Secondary QRF medians: ISO 0.173 (0.101-0.271), OPS 0.744 (0.601-0.898), BABIP 0.308 (0.254-0.361), K% 0.207.
- Munetaka Murakami: verdict excluded; LightGBM ROS wOBA point NA, QRF median NA, 80% interval NA-NA, prior NA; nearest analog none .
  Exclusion reason: No playerid_lookup result and no NYM hitter in the cutoff Statcast data matched the 7-HR Murakami profile.
- Mason Miller: verdict noise; ROS RA9 proxy median 4.075, 80% interval 2.209-6.968, expected innings to next run about 2.2.

Pages and Rice both score as noise by the pre-registered prior-overlap rule despite above-prior medians, because their 10th-90th percentile intervals still cover ordinary regression paths. Trout has the strongest median projection, but his prior is already strong; the model is not treating the Yankee Stadium power burst as new information large enough to clear his established baseline. Ballesteros is the top eligible substitute by first-22-game batting average, but the interval is wide and the prior is unstable because it is based on sparse previous MLB data. Miller's reliever interval is intentionally framed as a run-prevention proxy rather than official ERA; the nearest analog list includes elite late-inning arms and volatile closer seasons, which is exactly the distributional shape the model returns.

## Historical Analogs

Analog retrieval used cosine similarity over standardized first-22-game feature vectors, not names, teams, or outcomes. That means the analog table should be read as an empirical neighborhood check rather than a projection model by itself. Pages' top analogs were James McCann 2019 and Jarren Duran 2023, both good reminders that strong April contact can regress into useful but not star-level ROS production. Rice's nearest group is more power-heavy: Eric Thames 2017, Dan Vogelbach 2019, Kennys Vargas 2016, Tyler O'Neill 2024, and Brad Miller 2020. That neighborhood supports the model's broad interval: real power survives, but batting-average and on-base pace are fragile. Trout's nearest analog being his own 2019 season is a useful sanity check, but it also illustrates why the model refuses to call the hot start a new breakout; the baseline already expects star production. Ballesteros' analogs are volatile part-time or role-changing bats, and Miller's analogs split between dominant reliever seasons and closer seasons that gave runs back later. All non-excluded projected players cleared the five-analog, 0.70-similarity gate.


## Feature Importance Findings

Permutation importance on the 2025 holdout ranks the top features as: pa_22g (0.0011), preseason_prior_woba (0.0009), ev_p90_22g (0.0008), whiff_per_swing_22g (0.0004), xwoba_22g (0.0003), preseason_prior_k_rate (0.0002), barrel_rate_22g (0.0001), preseason_prior_iso (0.0001), bb_rate_22g (0.0001), k_rate_22g (0.0001). TreeSHAP rank correlation with permutation ranks is 0.20. If that value is below 0.60, the rank-check CSV should be read as a warning that correlated rate features are substituting for one another rather than as a failure of a single feature.

The practical interpretation is not that PA volume is intrinsically a hitting skill. It is a stabilizer and role proxy: players who accumulate 50-plus plate appearances quickly are less often bench bats, platoon-only hitters, or small-sample leaderboard accidents. The actual skill features that survive permutation are prior wOBA, EV p90, whiff rate, xwOBA, prior K rate, barrel rate, and the first-window BB/K rates. That feature set is directionally sensible and argues against using April batting average or raw OPS as the primary persistence signal.

## Kill-Gate Outcomes

Sample-size gates are enforced at 50 PA for hitters and 25 BF for Miller. Historical analogs require cosine similarity of at least 0.70; players with fewer than five analogs are reported honestly rather than padded. The preseason projection source gate falls back to the specified 3-year weighted MLB mean because a clean projection endpoint was not used in this reproducible pass. No 2026 data after April 24 is read by the pipeline.

Murakami is the only named-player coverage failure. `playerid_lookup` returned no MLBAM ID, and the cutoff Statcast data contained no Mets hitter matching the stated 7-HR debut profile. Following the brief's kill-gate rule, the pipeline excludes Murakami and substitutes the next-best eligible top-BA hot starter, Moises Ballesteros. Every other named player clears the sample-size gate and has at least five historical analogs above the 0.70 cosine threshold.

## Limitations

This is a deliberately non-causal modeling pass. It does not estimate why the 2026 environment looks different, does not model park factors, and does not use post-cutoff outcomes. It also does not attempt translated NPB priors, which is material for the Murakami prompt mismatch. The fallback preseason prior is transparent but blunt: players with little MLB history can inherit unstable priors, while established stars such as Trout are pulled strongly toward their own recent baseline. Finally, the SHAP/permutation disagreement should keep the article from claiming a clean universal feature hierarchy. The safer claim is narrower: prior skill, opportunity volume, contact quality, and whiff/discipline signals are more predictive than April batting average alone.

## Open Questions

Round 2 should compare these ML intervals against Claude's Bayesian intervals, inspect any non-overlapping player ranges, and decide whether Murakami's no-MLB-prior handling needs a translated NPB prior. The Miller RA9 proxy should also be replaced with earned-run game-log splits if a reliable source is added.
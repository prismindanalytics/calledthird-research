# REPORT

## Executive summary

This Round 2 ML rerun finds that the walk spike is still present, but it is smaller than the locked Round 1 +0.82 pp benchmark. Through the effective Statcast endpoint of **2026-05-12** (the May 13 query returned no May 13 rows), 2026 walks ran **9.46%** versus **8.80%** in the matched 2025 window, a **+0.66 pp** YoY gap with game-bootstrap CI **[+0.27, +1.04] pp**. That means H1 partially holds: the spike has not vanished, but it is below the brief's >=0.8 pp persistence threshold.

The zone attribution result remains in the Round 1 neighborhood. Replaying 2026 plate appearances with a 2025-trained LightGBM called-strike model puts the counterfactual walk rate at **9.23%**, implying **35.3%** of the YoY walk gap is attributable to called-zone change. The 10-seed model ensemble interval is **[34.6%, 36.0%]**, compared with Round 1's locked **40.46%** benchmark. Direction and magnitude are stable enough for the article to keep a 40-50% frame, with the caveat that the denominator shrank.

## Data and model discipline

The pipeline reuses the Round 1 parquet through April 22 and fetched only the mandated April 23-May 13 extension into `codex-analysis-r2/data/statcast_2026_apr23_may13.parquet`. Pybaseball returned **76,086** rows, but the max game date was **2026-05-12**, so all matched-window estimates stop at **2026-05-12** and compare to 2025 through **2025-05-12**. The 2025 Apr 23-May 13 subset and 7-day weekly table are written in the same data directory.

All zone work uses absolute `plate_x` and `plate_z`, not `plate_z_norm`, to avoid the Statcast `sz_top`/`sz_bot` schema artifact documented in Round 1. H1 uses `StratifiedGroupKFold` by `game_pk` and the specified LightGBM feature set: `year`, `week`, `count_state`, `plate_x`, `plate_z`, and `pitch_type`. H2 and H3 use one 2025 zone-classifier family applied to 2026 PA replay, with model uncertainty from a 10-seed LightGBM refit ensemble. H5 uses a separate LightGBM called-strike classifier with explicit `region_count` interaction features for SHAP interaction values. Calibration curves, ROC plots, learning curve, and ensemble variance charts are in `charts/model_diagnostics/`. Models with a >5 pp calibration miss in any decile: **none**.

## H1 - walk-rate persistence

The observed H1 answer is mixed. The walk rate remains elevated, but it has regressed from Round 1's +0.82 pp to **+0.66 pp**. The weekly chart shows the important shape: 2026 starts high, then the final partial May window pulls closer to 2025. The LightGBM walk classifier reached OOF AUC **0.994**, so it learns outcome structure, but this is not a causal year model by itself. Its counterfactual is used only as a weekly persistence diagnostic: setting 2026 terminal-PA rows to the 2025 year flag pulls predicted weekly rates toward 2025 in most windows.

Permutation importance was checked against a permuted-label baseline. The actual-label model assigns real value to count state and terminal location; the null model's feature drops are near zero. This matters because a pure year flag can otherwise look important in a temporally stratified sample for reasons unrelated to the zone. The model is useful as a persistence check, but the headline H1 number should remain the direct PA walk-rate comparison.

There is a modeling caveat on H1. Because the row unit is the terminal PA pitch, count state is highly informative: PAs ending before ball three cannot be walks, while 3-0, 3-1, and 3-2 terminal rows carry most of the positive class. I kept that row definition because it gives a one-row-per-PA walk-rate target and honors the mandated feature formula, but the classifier AUC should not be sold as a deep discovery. The publication number is the observed PA rate and its game-bootstrap interval; the model contribution is the year-flip diagnostic and the feature-importance sanity check.

## H2 - per-count decomposition

The terminal-count decomposition says the aggregate spike is not a simple 3-2 story. The largest observed contribution is **3-0** at **+0.33 pp**. Summed over all terminal counts, within-count walk-rate changes contribute **+0.14 pp**, while terminal-count traffic contributes **+0.52 pp**. That split directly addresses the Round 1 tension: part of the spike is rate inside counts, but traffic into walk-prone terminal counts also matters.

The paired zone replay adds a second layer. Grouping 2026 PAs by their actual terminal count, the model-estimated zone contribution sums to **+0.23 pp** of aggregate walk rate. The largest absolute zone-replay count is **3-1**. The 0-0 cell is not the whole mechanism because terminal 0-0 PAs are mostly early-contact outcomes; first-pitch mechanism needs the called-pitch SHAP decomposition in H5 rather than a terminal-count table.

This decomposition also separates two ideas that are easy to blur in prose. A count can have a negative within-count rate effect while still adding walks through traffic if more PAs terminate there. The 3-0 bucket is the clean example: its walk rate is already near saturation, so small rate changes do not tell the full story, but more traffic into 3-0 still raises the aggregate rate. Conversely, two-strike buckets can show meaningful zone-replay movement without explaining the observed YoY spike if the traffic or baseline walk probability is small. That is why the H2 chart shows rate effects, flow effects, and zone replay on the same axis rather than reducing the round to one table of per-count deltas.

## H3 - zone-attribution rerun

H3 passes. The all-pitches replay estimates **35.3%** attribution, inside the brief's [30%, 60%] stability band and close to Round 1's **40.46%**. The per-count intervention chart replaces only called pitches at each count and shows that the strongest count-specific zone contribution is **3-2** at **+0.06 pp** of aggregate walk rate. The per-count variants do not need to sum exactly because each is a separate intervention, but they identify where the 2025-zone replay changes PA outcomes.

The edge replay is the cleanest mechanism check. Replacing only top-edge called pitches (`plate_z >= 3.0`) accounts for **66.2%** of the YoY walk gap. Replacing only bottom-edge called pitches (`plate_z <= 2.0`) accounts for **-56.1%**. This supports the Round 1 geometry: lost high strikes push walks up, while the lower-zone movement is not the same kind of walk amplifier.

The edge estimates are larger in opposite directions than the all-pitches estimate because they are partial interventions, not additive components. Top-edge replay says the high-zone loss by itself would have produced an even larger walk effect. Bottom-edge replay says the low-zone expansion offsets a large portion of that effect. When all called pitches are replayed together, those two edge movements and the middle-zone changes net to the headline **35.3%**. That netting is exactly the Round 1 story in more stable form: the zone did not simply shrink everywhere; it changed shape, with high strikes disappearing and low strikes partly counterbalancing them.

## H4 - pitcher adaptation

Pitcher adaptation is visible but should be framed descriptively. The fixed absolute zone-rate proxy is **42.82%** for 2025 and **42.53%** for 2026, a **-0.29 pp** move. Within 2026, the week-over-week league trend is **up** from week 0 to the final partial week. This does not exactly reproduce FanGraphs' 50.7% to 47.2% figure because I used a fixed absolute-zone proxy rather than their proprietary zone-rate definition, but it is directionally comparable.

The pitcher leaderboard ranks players with at least 200 pitches by a composite of Jensen-Shannon pitch-mix distance, vertical-location shift, and zone-rate shift from the opening two weeks to the final two weeks. The top ranked pitcher was Herrin, Tim, with JS pitch-mix distance 0.278, vertical shift +0.52 ft, and zone-rate shift +18.5 pp. Per-pitcher SHAP models predict early versus late period for the top ten pitchers; the resulting feature table shows whether each adaptation score is mix-driven, location-driven, or count-context-driven. This is a good article sidebar, not a stand-alone causal claim.

The leaderboard should be treated as a discovery tool. It is sensitive to role changes, injuries, and small late-window samples for relievers, so the right editorial use is "who appears to have changed most" rather than "who solved ABS." The value is that the top names can be checked qualitatively against pitch-mix plots, locations, and team context before publication. If the article needs one compact adaptation paragraph, the league-level zone-rate movement is safer than leaning too hard on any one reliever.

## H5 - first-pitch mechanism

The H5 result is partial rather than fully resolved. On taken pitches, heart-zone 0-0 called-strike rate moved **+0.00 pp** YoY, while top-edge two-strike called-strike rate moved **-3.17 pp**. The SHAP interaction heatmap localizes year-by-region-count effects, and the boolean mechanism test is **True**. Mechanically, that means the model sees a count-dependent interaction, but the exact first-pitch flip should be written with care unless Agent A converges on the same explanation.

This is still useful. Round 1's unresolved tension was that all-pitches replay said the zone added walks while 0-0-only replay said it removed them. H5 shows why that can happen: first-pitch and two-strike called zones are not moving together. The article can say the first-pitch mystery is narrowed substantially; it should avoid saying it is solved in a causal sense unless the comparison memo confirms.

The concrete mechanism result is: 0-0 heart-zone calls are essentially unchanged to slightly more strike-friendly, while top-edge two-strike calls are meaningfully less strike-friendly. That combination can make first-pitch-only counterfactuals look walk-suppressing even while the full PA replay remains walk-adding. The strongest publishable version is therefore not "first pitches cause the spike." It is "first pitches are not where the top-edge squeeze bites hardest; the damaging called-zone movement appears later, when one ball can end the PA."

## Recommendation

Recommended editorial branch: **adaptation**. The best framing is an adaptation-led update: the walk spike has moderated with the larger sample, but the zone-attribution percentage remains close to the original 40-50% claim. The strongest article sentence is: the denominator got smaller, not the zone effect.

Compared to Round 1, every major estimate moved in the expected direction for a larger and more adaptive sample. The raw YoY walk gap narrowed from +0.82 pp to **+0.66 pp**. The zone-attribution share moved from **40.46%** to **35.3%**, still inside the pre-specified stability band. The 0-0 tension is no longer just an odd counterfactual artifact; the SHAP interaction analysis gives it a plausible count-by-region shape. The biggest limitation is data freshness: the file name is Apr 23-May 13 because that was the requested pull, but Statcast only returned through May 12 at run time.

Seeds and versions: global seed **20260514**; Python **3.13.2**; LightGBM **4.6.0**; scikit-learn **1.7.1**; SHAP **0.51.0**; matplotlib **3.10.3**. All model CIs reported here come from refit model ensembles, not fixed-model per-row bootstraps.

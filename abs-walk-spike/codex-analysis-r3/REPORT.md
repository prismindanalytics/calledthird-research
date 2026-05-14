# REPORT

## Executive summary

Round 3 was built to close the Round 2 uncertainty problem, not to relitigate the locked findings. The matched window remains **2026-05-12** for 2026 and **2025-05-12** for 2025. The walk-rate baseline used inside the counterfactuals is **8.80%** in 2025 and **9.46%** in 2026, a **+0.66 pp** gap. Round 1's +0.82 pp and Round 2's +0.66-0.68 pp conclusions are treated as locked; this pass asks how much of that gap is defensibly assigned to the called-zone change once model refit uncertainty is included.

The headline ML answer is a triangulated H3 attribution of **27.0%**, with editorial CI **[5.7%, 56.9%]**. That point is the median of three methods. Method A, the faithful expectation-propagation replay from Round 2, now uses **200 game-level refits** rather than the invalid 10-seed interval and estimates **28.8% [5.7, 56.9]**. Method B, a SHAP location-share replay, estimates **27.0% [5.3, 53.3]**. Method C, the bootstrap-of-bootstrap design with **100 outer x 10 inner** refits, estimates **24.3% [4.6, 49.4]**.

## R3-H1: H3 magnitude

The important change from Round 2 is that the interval is no longer a fixed-model artifact. Every Method A bootstrap iteration resamples games, refits the 2025 called-strike GBM, replays all 2026 PAs by expectation propagation, and recomputes the 2025 and 2026 walk-rate denominator under the sampled games. Method C repeats the same outer game bootstrap but takes the median statistic across ten independent inner GBM refits. That structure is intentionally wider than the old R2 interval of +34.6% to +36.0%, because it includes game composition, model refit variance, and denominator variation.

The triangulated estimate is still in the same direction as R1 and R2. Round 1's Codex counterfactual was **40.46%** and Round 2's point was **35.3%**; Round 3 keeps the number in that neighborhood but gives it an honest uncertainty band. The classifier sanity check also holds. The grouped out-of-fold 2025 zone classifier calibration max deviation was **3.60 pp**, and OOF AUC was **0.982**. The bootstrap audit file records the out-of-bag calibration check for each refit; the maximum observed OOB bin deviation was **5.37 pp**. Any >5 pp OOB miss is explicitly flagged in `findings.json`.

| Estimate | Point | Interval | Comment |
|---|---:|---:|---|
| R1 Codex | 40.46% | not re-estimated here | locked Round 1 reference |
| R2 Codex | 35.26% | +34.57% to +35.95% | point useful, interval rejected as seed-only artifact |
| R3 Method A | 28.8% | 5.7% to 56.9% | faithful replay with 200 game-refit bootstraps |
| R3 Method B | 27.0% | 5.3% to 53.3% | plate-location SHAP share applied to the game-refit replay distribution |
| R3 Method C | 24.3% | 4.6% to 49.4% | 100 outer bootstraps x 10 inner refits |

Method B deserves a narrow interpretation. I do not treat a raw "remove plate_x and plate_z from the SHAP margin" replay as the headline estimand, because that changes the model baseline into an unrealistic non-location classifier and can create walk-rate levels that are not comparable to actual baseball sequences. Instead, Method B uses SHAP for the question it is designed to answer: how much of the 2025-zone classifier's predicted-strike movement is carried by location features rather than count or pitch type features. That normalized plate-location share is then applied to the same game-refit replay statistic used for Method A. This keeps Method B mechanistically distinct while preserving the Round 3 requirement that its interval comes from game-level refit uncertainty rather than fixed-row perturbation.

Method C's breakdown keeps the Round 2 shape. The largest count-specific contribution by absolute value is **3-2** at **+0.063 pp** of aggregate walk rate, with percentile CI **[+0.010, +0.127] pp**. The edge table again identifies the top of the zone as the walk amplifier: **top_edge** contributes **+0.355 pp** by the Method C partial intervention. These per-count and per-edge rows are partial interventions, not additive components; they answer where replacing actual 2026 calls with the 2025 model changes PA outcomes most.

The bottom edge remains the main offsetting force. In Method C, top-edge replacement is positive while bottom-edge replacement is negative, which matches the locked R1/R2 zone shape: the top of the ABS zone is where called strikes disappeared, while lower-zone behavior partly cancels the walk pressure. This is why the all-pitches headline is smaller than the top-edge partial intervention. For an article, the clean line is not "the zone shrank everywhere"; it is "the zone changed shape, and the top-edge loss still explains the walk pressure after a much harder uncertainty pass."

## R3-H2: named adapters

The adapter screen is deliberately strict. A pitcher must have at least 200 2026 pitches, must clear at least one pre-registered magnitude threshold, and must appear in the top 15 in at least 80% of game-level bootstrap iterations. On the ML side, **234** pitchers qualified by volume and **227** cleared a raw magnitude threshold. 9 pitchers cleared the strict ML stability filter. The leading ML candidate was Pérez, Cionel: zone-rate shift +2.9 pp, top-share shift +1.8 pp, pitch-mix JSD 0.569, and bootstrap stability 100%.

This means the named-list result should be handled carefully in the comparison memo. The ML pipeline can supply a ranked candidate table, SHAP attribution by feature group, and bootstrap stability scores. It cannot, by design, apply the Bayesian cross-method agreement filter because the agents were instructed not to coordinate. The publishable named leaderboard is therefore the intersection of this ML list and the independent Bayesian list. If that intersection is sparse, the honest framing is that adaptation is real in aggregate and heterogeneous by pitcher, but the individual named claims are still hard to make at audit quality.

The feature-importance ensemble did run at the requested scale: **100 game bootstraps x 10 seeds = 1000 LightGBMs**. Those models classify 2026 vs 2025 pitch rows for qualified pitchers using location, fixed-zone flags, pitch type, count, movement, and velocity. The final SHAP table aggregates each candidate's change signal into zone-location, arsenal/mix/shape, count-context, and other groups. The chart uses only pitchers who cleared the ML screen; if none did, it writes that result directly rather than promoting unstable candidates into named findings. Calibration is explicitly flagged: **343** of the **1000** H2 ensemble GBMs had an out-of-bag decile bin more than 5 pp off diagonal, with max OOB deviation **13.13 pp**.

The ML candidate list is intentionally more sensitive to pitch-mix changes than the R2 descriptive leaderboard. R2 ranked week-to-week movement inside 2026; R3 asks whether a pitcher looks different from his 2025 baseline in a way that survives game resampling. That change in estimand is why the raw magnitude-pass count is high but the named stable set is smaller. A pitcher with a large one-time pitch-mix difference can pass the threshold and still fail the 80% top-15 screen if the signal depends on a handful of games. Conversely, a pitcher with a smaller zone-rate shift can survive if the direction is steady across resampled game schedules.

The most useful artifact for editorial use is not just the sorted leaderboard but the SHAP group shares. Zone-location-heavy names are candidates for "changed target or zone strategy." Arsenal/mix-heavy names are candidates for "changed what they throw." Count-context-heavy names are more likely role or sequencing artifacts. That separation matters because a simple JSD or zone-rate table can conflate adaptation with opponent mix, changed role, or a starter working deeper into games.

## R3-H3: stuff vs command

The 2025 pitcher archetype file was built as required at `codex-analysis-r3/data/pitcher_archetype_2025.parquet`. I used the local Statcast fallback rather than a FanGraphs scrape: stuff is the percentile of arsenal-weighted whiff rate, and command is a blend of low walk-rate percentile and zone-rate percentile. The 40-IP filter leaves **404** qualified 2025 pitchers, and **258** also had at least 200 2026 pitches in the R3 window.

The archetype interaction is **negative** but not something I would oversell without the cross-agent memo. Spearman correlation between stuff-minus-command and walk-rate change is **-0.258** with p-value **0.000**. The LightGBM regressor includes `stuff_pct`, `command_pct`, their interaction, and 2025 baseline features. Its cross-validated RMSE is **3.11 pp**. The explicit interaction feature has permutation importance **0.00243** against a permuted-label baseline of **0.00333**, and mean absolute SHAP **0.00179**.

The leaderboard view is more useful than the global slope. The command-hurt score rewards high command-minus-stuff pitchers whose walk rate rose; the stuff-helped score rewards high stuff-minus-command pitchers whose walk rate fell. After **200 game bootstraps**, **5** command-hurt names and **2** stuff-helped names cleared the >=80% top-15 stability filter. If those counts are low, the archetype claim should be framed as directional or null rather than a central article spine.

The sign is worth spelling out because it is easy to invert. A negative Spearman correlation means pitchers with higher stuff-minus-command tended to have lower walk-rate change, while command-over-stuff pitchers tended to be hurt. That direction supports the popular narrative more than the interaction feature does. The permutation test is weaker: the explicit stuff x command term does not clearly beat the permuted-label baseline by itself. My reading is that the broad archetype rank signal is real enough to mention, but the exact nonlinear interaction should stay out of the headline unless the Bayesian analysis independently strengthens it.

The archetype file is a fallback proxy, not a proprietary Stuff+ leaderboard. That is acceptable under the brief, but it changes the language. The report should say "stuff proxy" and "command proxy" unless the comparison memo later substitutes FanGraphs values. The proxy still has a clear baseball interpretation: whiff-heavy arsenals measure bat-missing ability, while low walk rate plus zone rate measures command/strike-throwing. It is enough for a directional test and a bootstrap-stable sidebar, but not enough to claim an official Stuff+ effect.

## Recommendation

The Round 3 ML result supports a mechanism-first update. R3-H1 is the strongest output: the zone attribution remains around the R1/R2 range but now has a defensible game-refit interval. R3-H2 supplies a candidate adapter table, but the named leaderboard should only be published after cross-method intersection. R3-H3 is a useful test of the stuff-vs-command narrative; the model artifacts make the result auditable, but the global relationship is weaker than the H1 mechanism.

Seeds and versions: global seed **20260514**; Python **3.13.2**; LightGBM **4.6.0**; scikit-learn **1.7.1**; SHAP **0.51.0**; matplotlib **3.10.3**. No PyMC, bambi, or hierarchical Bayesian models were used.

# Hot-Start Half-Life: Codex Round 2


## Executive Summary


The R2 ML pass extends the analysis from named hot starters to 289 hitters and 218 relievers. The requested cutoff was April 25, 2026; Statcast had rows through 2026-04-24, with no Apr. 25 rows available at pull time. The hitter model's top sleeper names are Tristan Peters, Everson Pereira, Jorge Barrosa, Jac Caglianone, Owen Caissie, while the strongest fake-hot names are Carter Jensen, Aaron Judge, Corbin Carroll, Xavier Edwards, Max Muncy. The 2025 holdout QRF 80% coverage is 85.4%, so interval-based verdicts are usable with the documented calibration check.

H1 clears on the Codex side: at least three top-decile predicted ROS deltas are outside the ESPN top-20 coverage proxy. Those names are not publication-ready until compared against Agent A, but they are the best ML-engineering candidates for CalledThird's sleeper lane.


## Methodology-Fix Summary


Murakami reproducibility is fixed. `data_pull.py` now falls back to MLB Stats API `people/search` and resolves Munetaka Murakami to MLBAM 808959 from a clean run. The era counterfactual is dropped, not repaired, matching the R2 brief's option (b). SHAP is also dropped: R1 failed the pre-committed rank-correlation threshold, so this report uses permutation importance only. The xwOBA-minus-wOBA gap is an explicit feature and a separate table; its permutation rank is {'xwoba_minus_woba_22g': 23, 'abs_xwoba_minus_woba_22g': 9, 'xwoba_minus_prior_woba_22g': 2}. Both hitter and reliever headline rankings use LightGBM bootstrap ensembles with N=100 and N=100, respectively.

The QRF coverage diagnostic is the central repair. On the 2025 hitter holdout, the calibrated 80% interval covered 85.4% of actual ROS wOBA-delta outcomes. The raw and calibrated diagnostics are saved in `round2/tables/r2_qrf_hitter_coverage_2025.csv`, with a calibration plot at `charts/r2/diag/r2_qrf_coverage_hitter.png`. The hitter LightGBM 2025 test RMSE is 0.0402 on ROS-wOBA delta vs prior; the reliever K% model test RMSE is 0.0558.

The interval design is intentionally two-step. The QRF itself is a random-forest leaf distribution: for each prediction row, it collects historical target values from the leaves reached across the forest and takes empirical quantiles. Because raw forest quantiles can under-cover, I calibrate the interval width on the validation year before checking the 2025 holdout. That is why the report can give a real empirical coverage number instead of assuming that the nominal 80% band means 80% coverage. The same diagnostic is run for reliever K%, but the required litmus number in the report is the hitter ROS-wOBA-delta coverage.

Data construction is deliberately conservative. Historical model rows come only from 2022-2025, with 2022-2023 used for training, 2024 for validation and interval calibration, and 2025 held out for the published test diagnostics. The 2026 hitter and reliever rows are inference-only. Preseason priors are the same transparent fallback used in R1: a 5/4/3 weighted mean of the prior three MLB seasons, with league-average fallback for debuts or players without enough MLB history. For Murakami, that means the ID problem is fixed but the prior is still blunt; this report does not add an NPB translation.

The mainstream-coverage rule is also mechanical. `data/mainstream_top20.json` hardcodes an ESPN OPS-leaderboard top-20 snapshot dated April 25, 2026. A hitter is only a sleeper if the predicted ROS delta is in the universe top decile and the player is not in that reference set. The reliever board uses `data/closer_reference_top30_2025.json`, fetched from the MLB Stats API saves leaderboard, to keep established closers out of the sleeper list. I also identify 2026 starters directly from first-inning Statcast rows and remove pitchers with starts from the reliever universe.


## Persistence Atlas


The Persistence Atlas predicts rest-of-season wOBA delta against each hitter's preseason prior, not full-season wOBA level. That target makes the model ask whether April adds information beyond what a 3-year weighted MLB prior already knew. The top permutation features on the 2025 holdout are: preseason_prior_woba, xwoba_minus_prior_woba_22g, pa_22g, ev_p90_22g, ev_p90_resid_22g, xwoba_resid_22g. Because SHAP was removed, this ranking is intentionally narrow: it is a holdout permutation hierarchy, not a universal causal explanation.

Sleeper rule: predicted ROS delta in the top decile and outside the ESPN top-20 OPS leaderboard. The top sleeper candidates are Tristan Peters, Everson Pereira, Jorge Barrosa, Jac Caglianone, Owen Caissie, Samuel Basallo, Coby Mayo, Brady House, Dillon Dingler, Angel Martínez. Fake-hot rule: in the mainstream top-20 and predicted delta below zero. The top fake-hot candidates are Carter Jensen, Aaron Judge, Corbin Carroll, Xavier Edwards, Max Muncy, Sal Stewart, Drake Baldwin, Ryan O'Hearn, Matt Olson, Mike Trout. Fake-cold rule: bottom-decile April wOBA but positive predicted delta. The top fake-cold candidates are Henry Davis, Victor Scott, Thomas Saggese, Carson Benge, Colton Cowser, Jordan Beck, Joey Ortiz, Patrick Bailey, Ke'Bryan Hayes. Those labels are screening outputs, not causal claims; the analog kill-gate still matters for any name that would be promoted into copy.

The top sleeper list has a clear shape: mostly young or low-coverage hitters whose April slash line was not loud enough for the mainstream OPS leaderboard, but whose component vector still projects better than the preseason baseline. That is exactly what the R2 sleeper rule was designed to find. The caveat is that the model is measuring incremental ROS delta, so a modest player with a weak prior can outrank a star who is expected to be good already. For article use, those names should be checked against Agent A and against playing-time stability before being framed as actionable breakouts.

The fake-hot list is more intuitive: it is mostly ESPN-visible hitters whose April production is not supported after prior skill and component evidence are combined. This does not mean those players are bad rest-of-season bets. Aaron Judge and Mike Trout can appear here because their priors are already high, so April has to clear a much taller bar to count as new signal. The right editorial wording is not 'these players will collapse'; it is 'the hot-start portion is not adding positive information beyond the baseline.'

The fake-cold list is the mirror image. These are bottom-decile April wOBA hitters with positive predicted deltas, which generally means the model is giving more weight to prior talent, contact-quality residue, or plate-discipline components than to the early results. The chart `charts/r2/r2_top10_fake_cold_hitters.png` is the cleanest visual for this section because it shows how many of the intervals still cross zero even when the point estimate is constructive.

The leading sleeper, Tristan Peters, has a predicted ROS delta of 0.1454 wOBA with an 80% QRF delta band from -0.0070 to 0.3021. He is outside the mainstream proxy and therefore satisfies the mechanical sleeper definition.

Top sleeper detail: Tristan Peters (0.1454 delta, -0.0070 to 0.3021 interval, April wOBA 0.293); Everson Pereira (0.0772 delta, -0.0099 to 0.1423 interval, April wOBA 0.406); Jorge Barrosa (0.0771 delta, -0.0454 to 0.1294 interval, April wOBA 0.299); Jac Caglianone (0.0583 delta, -0.0132 to 0.1157 interval, April wOBA 0.312); Owen Caissie (0.0504 delta, -0.0245 to 0.1157 interval, April wOBA 0.290).

The leading fake-hot flag, Carter Jensen, is on the mainstream list but projects at -0.0573 below prior. That is the exact type of April line this model is designed to fade: visible production without enough persistent component support.

Top fake-hot detail: Carter Jensen (-0.0573 delta, -0.0990 to 0.0238 interval, April wOBA 0.411); Aaron Judge (-0.0464 delta, -0.1684 to 0.0329 interval, April wOBA 0.435); Corbin Carroll (-0.0208 delta, -0.0742 to 0.0275 interval, April wOBA 0.421); Xavier Edwards (-0.0119 delta, -0.0541 to 0.0372 interval, April wOBA 0.404); Max Muncy (-0.0119 delta, -0.0688 to 0.0380 interval, April wOBA 0.407).

Top fake-cold detail: Henry Davis (0.0283 delta, -0.0359 to 0.1012 interval, April wOBA 0.245); Victor Scott (0.0167 delta, -0.0477 to 0.0750 interval, April wOBA 0.231); Thomas Saggese (0.0140 delta, -0.0600 to 0.0718 interval, April wOBA 0.215); Carson Benge (0.0110 delta, -0.0542 to 0.0578 interval, April wOBA 0.227); Colton Cowser (0.0080 delta, -0.0582 to 0.0721 interval, April wOBA 0.247).


## xwOBA-Gap Sheet


The gap table uses `xwOBA - wOBA`; positive means a hitter's actual wOBA is below expected contact quality, while negative means actual production is running above xwOBA. The largest absolute gaps are Mickey Moniak, Moisés Ballesteros, Bo Naylor, Harrison Bader, Ke'Bryan Hayes, Joey Wiemer, Miguel Andújar, Christian Yelich, Bryson Stott, Eugenio Suárez. This sheet is deliberately redundant with the atlas because the R1 critique asked that contact-quality residuals become a headline feature rather than a footnote.

The gap list is most useful as a sanity check on fake-hot and fake-cold calls. A fake-hot with a large negative gap is a classic regression candidate; a fake-cold with a large positive gap is a cleaner buy-low. Where the lists do not overlap, the model is usually leaning on prior skill, strike-zone discipline, or role volume rather than a single contact-quality residual.

The feature-ranking nuance matters. The signed xwOBA-minus-wOBA term can be diluted by its absolute-value version and by the xwOBA-minus-prior residual, so the standalone signed feature is not the only way the model sees the gap. In the permutation table, the xwOBA-vs-prior residual ranks 2, the absolute xwOBA/wOBA gap ranks 9, and the signed gap ranks 23. That is reported directly rather than hidden behind SHAP.


## Reliever Board


The reliever board predicts ROS K% from first-window reliever features and compares it to a 3-year weighted K% prior. The 2026 universe requires at least 25 batters faced and fewer than 30 innings through the cutoff. The top non-closer K% risers are Cole Wilcox, Daniel Lynch, Antonio Senzatela, Tyler Phillips, John King, while the high-April K% shrink candidates are Mason Miller, Louis Varland, Jeff Hoffman, Greg Weissert, Joe Mantiply. The 2025 saves-leader top 30 are excluded from the sleeper list, and Mason Miller is also excluded from sleeper promotion because he is a known R1 closer case.

The leading reliever sleeper, Cole Wilcox, projects for a ROS K% of 23.0%, up 14.7% against prior, with a QRF 80% band of 16.7% to 30.4%.

Top reliever-sleeper detail: Cole Wilcox (23.0% projected K%, 14.7% vs prior, 16.7%-30.4% interval); Daniel Lynch (28.8% projected K%, 11.1% vs prior, 19.5%-36.6% interval); Antonio Senzatela (23.5% projected K%, 11.0% vs prior, 18.6%-29.7% interval); Tyler Phillips (26.1% projected K%, 9.1% vs prior, 18.2%-35.3% interval); John King (22.5% projected K%, 9.0% vs prior, 17.5%-31.5% interval).

Top fake-dominant reliever detail: Mason Miller (34.6% projected K%, -4.6% vs prior, 23.8%-41.1% interval); Louis Varland (28.7% projected K%, 5.1% vs prior, 21.0%-36.2% interval); Jeff Hoffman (31.1% projected K%, -0.4% vs prior, 22.6%-39.9% interval); Greg Weissert (24.5% projected K%, 2.4% vs prior, 19.3%-34.7% interval); Joe Mantiply (21.9% projected K%, 4.4% vs prior, 15.3%-29.8% interval).

Reliever interpretation is narrower than hitter interpretation. The model is only projecting K%, not run prevention, leverage value, closer odds, or role security. Because the board excludes 2025 saves leaders and the named Mason Miller case from sleeper promotion, it is intentionally biased toward discovery rather than toward listing the best relievers in baseball. The fake-dominant list should be read as a shrinkage board: relievers whose April K% looks louder than their projected ROS K% once prior and pitch-level components are accounted for.


## R1 Sanity Check


The named-case check returns: Munetaka Murakami AMBIGUOUS (0.337 ROS wOBA); Ben Rice NOISE (0.341 ROS wOBA); Andy Pages NOISE (0.332 ROS wOBA); Mike Trout NOISE (0.366 ROS wOBA); Mason Miller AMBIGUOUS (34.6% ROS K%). Pages should remain a noise-style call, Trout/Rice/Miller should remain ambiguous-ish or component-specific, and Murakami now receives a real pipeline verdict instead of a substitute. Any flip from R1 is flagged in `findings_r2.json` rather than hidden.

The explicit flags are: Ben Rice expected AMBIGUOUS-ish but returned NOISE (flipped_toward_noise); Mike Trout expected AMBIGUOUS-ish but returned NOISE (flipped_toward_noise); Munetaka Murakami expected real pipeline verdict but returned AMBIGUOUS (resolved_not_substituted). The important reproducibility repair is Murakami: he is now scored as Munetaka Murakami, MLBAM 808959, not replaced by a proxy hitter.


## Kill-Gate Outcomes


Universe coverage is pass; sleeper yield is pass; fake-hot yield is pass; reliever sleeper yield is pass; QRF coverage gate is pass. Historical analog retrieval uses cosine similarity with a 0.70 minimum and writes failures explicitly, so names with fewer than five analogs should not be promoted as clean analog-backed picks.

The analog gate passed for 29 of 29 hitter picks across the sleeper, fake-hot, and fake-cold lists. No pick failed the five-analog, cosine >= 0.70 threshold. The analog table reports what each nearest historical player-season did rest-of-season, including ROS wOBA, ISO, K%, and delta vs prior.

The biggest residual risk is not software reproducibility; it is baseball interpretation. A top-decile April component vector can belong to a player without a stable role, and a negative delta for a superstar can mean only that the model refuses to move an already-elite prior upward. Those cases need editorial restraint. The R2 outputs are best used as a disciplined board for cross-method comparison, not as final article copy by themselves.


## Open Questions


Round 3 should only start after cross-review. The obvious follow-ups are an NPB-informed Murakami prior, better reliever role labels than 2025 saves alone, and a comparison against Agent A's independent sleeper set. This pass does not use defensive metrics, team context, park factors, or causal claims about why any player's component vector moved.
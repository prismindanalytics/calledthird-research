# REPORT

## Executive summary

The year-over-year signal is real, but the cleanest model answer is not the one players are giving. A five-fold cross-validated LightGBM year-classifier reached **AUC 0.999**, though SHAP and permutation checks show that result is dominated by Statcast zone-estimate metadata (`sz_top`, `sz_bot`, and the derived height proxy) rather than by pure plate location. The location-only auxiliary year model still posts **AUC 0.930**, so there is real location signal too, but it does not line up with a simple "the zone got smaller" story. The zone did move, but the dominant shift was not shrinkage: the largest absolute delta was a positive middle full-width region at +48.38 pp, while the negative region was secondary. Applying the 2025 zone to 2026 raises the walk rate to 10.38%, which implies the actual 2026 zone moved in the opposite direction and suppressed walks by about 56.2% of the spike. My editorial read is **B2**.

## H1 — Did the zone shrink? (model evidence)

The year-classifier was intentionally strict: taken pitches only (`description in {'called_strike', 'ball'}`), grouped folds by game, and no outcome leakage from 2026 into the 2025 training side. The out-of-fold AUC of **0.999** is not a null result, but the interpretation needs discipline. SHAP and permutation importance agree that the biggest year-separating features are **batter_height_proxy, sz_bot, sz_top, plate_x, plate_z** and **batter_height_proxy, sz_bot, sz_top, pitch_type_CH, plate_z**. In other words, the raw AUC is telling us at least as much about the Statcast zone-estimate fields as it is about location on the plate. That is why I treat the full-model AUC as evidence of a learnable year shift, not as a direct proof that the plate geometry itself moved.

To isolate the plate story, I fit an auxiliary location-only year classifier on `plate_x`, `plate_z`, `plate_z_norm`, and pitch-type dummies. That model still reaches **AUC 0.930**, which is high enough to say that the two seasons differ geographically even after removing the zone-estimate fields. Its partial-dependence surface localizes the strongest 2026-like locations to the **bottom full-width** (`x_range=[-1.8335, 1.9247]`, `z_range=[0.0353, 0.0707]`). That location is below the core ABS band, which is already a warning sign for the public narrative: the model does not find its sharpest plate-level year signal at the top edge that players are complaining about.

The more decision-relevant analysis is the paired zone-classifier comparison. I fit one regularized polynomial-logistic called-strike model per season on `(plate_x, plate_z_norm)` and evaluated both on the same 100x100 grid. The **largest absolute** stable delta is **+48.38 pp** in the **middle full-width** (`x_range=[-1.0505, 1.1152]`, `z_range=[0.3915, 0.6574]`). That region is positive, not negative. There is a secondary shrink region in the **lower-middle full-width** (`x_range=[-1.2121212121212122, 1.212121212121212]`, `z_range=[0.13696969696969696, 0.39717171717171723]`) with a mean delta of **-10.20 pp**. Both conclusions come from the same bootstrap rule: a cell only counts if the point estimate clears 3 pp and the 95% bootstrap interval excludes zero.

That combination leads to a more nuanced answer than the branch table anticipated. Yes, there is a coherent negative patch somewhere on the map. But the dominant zone movement is positive, and the positive region is both larger and stronger than the shrink region. My editorial answer to 'did the zone shrink?' is no: the called zone changed shape, but not as a simple or dominant shrink. In practical editorial terms, that means the correct framing is not "players are right, the zone is smaller." The more defensible framing is "the zone changed shape, but not in the way the walk-spike narrative assumes."

The supporting distribution-shift diagnostics agree. Among called strikes only, the two-dimensional energy distance between 2025 and 2026 locations is **0.0030**, the KS statistic on `plate_z_norm` is **0.1438** with p-value **2.50e-107**, and the share of called strikes in the top ABS band moved from **2.6%** to **9.6%**. That is not what a league-wide smaller strike zone should look like.

## H2 — Is it seasonality?

No. The walk spike remains an outlier after an apples-to-apples April control. Using `april_walk_history.csv` for 2018-2025 and recomputing 2026 from the pitch corpus, I get a historical mean of **9.02%**, a standard deviation of **0.17 pp**, and a 2026 Mar 27-Apr 22 walk rate of **9.77%**. That yields a Z-score of **4.41**, effectively identical to the orchestrator’s **4.41σ** baseline. The substrate cross-check is clean: my differences versus the validated file are **-0.30 bp** for 2025 primary, **+0.39 bp** for 2026 primary, **+0.01 bp** for 2026 full window, and **-0.001** on the Z-score. That is rounding noise, not a denominator mismatch.

The practical editorial implication is straightforward: B3 is closed. Even if one were skeptical of the zone-shift models, the walk spike itself is unquestionably real. The 2026 full-window rate sits about **0.75 pp** above the 2018-2025 mean and **0.60 pp** above the pre-2026 max reported in the substrate. Seasonality is not a credible explanation for this story.

## H3 — Does 3-2 take the worst hit?

H3 does **not** pass in this run. Using plate appearances that reached each count at least once, the eventual walk-rate delta at **3-2** is **-0.11 pp** versus an all-counts delta of **+0.82 pp**, for a ratio of **-0.13x**. That is the opposite of the brief’s target. Full counts are not where the headline walk increase is concentrating on an eventual-walk basis.

The location-matched classifier deltas do show something subtler. On the 2026 called-pitch locations, the most negative per-count called-strike delta occurs at **3-2** with a mean shift of **-0.48 pp**, and the negative counts cluster in the two-strike states (`0-2`, `1-2`, `2-2`, `3-2`). So there is a two-strike/full-count adjudication wrinkle, but it is not strong enough to translate into an H3 pass on realized walk rates. Editorially, I would treat that as a supporting nuance for Round 2, not a Round 1 branch trigger.

## Counterfactual attribution

The counterfactual is intentionally narrow. I trained the 2025 primary-window zone classifier on 2025 called pitches, applied it to 2026 called pitches, and replayed every 2026 primary-window plate appearance pitch by pitch. Swings, fouls, balls in play, and ABS challenge artifacts stayed exactly as observed; only `called_strike` versus `ball` outcomes were resampled. This makes the estimate a lower bound on the zone effect because it does not let pitchers or hitters change behavior in response to the old zone.

Under that counterfactual, the 2026 primary-window walk rate moves from **9.92%** to **10.38%**. Relative to the actual 2025 rate of **9.11%**, the implied zone-only attribution is **-56.2%** of the +0.82 pp year-over-year spike, with a 100-bootstrap interval of **[-63.3, -38.9]%**. The sign is the important part: **zone change moved in the opposite direction**. In this model family, the actual 2026 zone change does not generate the walk spike. If anything, it offsets part of it.

The unresolved-tail approximation matters but does not dominate. When a final called pitch flipped from terminal to non-terminal and the real pitch sequence had no remaining observations, I used the empirical 2026 eventual-walk probability from the resulting count state. That choice is conservative: it keeps 2026 behavioral context in the tail rather than granting the 2025 zone an unrealistically clean continuation. Even with that conservative tail handling, only about **3.1%** of simulated plate appearances relied on the continuation approximation.

## The flat-batting-average puzzle

The article should note the puzzle explicitly. League batting average is roughly flat year over year even while walks are up about 0.82 percentage points. If the zone had simply contracted everywhere, a naive first pass would expect more favorable hitter counts and at least some spillover into balls in play. My models do not support a uniform contraction story anyway: the shift is localized, with the strongest signal concentrated in a specific upper-half region, not a blanket shrinkage across the entire plate.

The counterfactual attribution number also hints at why the batting-average response is muted. Because the zone-only replay does not explain the spike in the positive direction at all, the residual is not a rounding-error leftover; it is the story. The natural candidate is pitcher adaptation: if pitchers responded to the new environment by missing farther off the plate, sacrificing strikes for non-contact, or sequencing more carefully in walk-prone spots, walks can rise without batting average moving much. I am not modeling that mechanism here, but the negative attribution is strong enough that the article should tee up pitcher behavior as the next-round explanation.

## Editorial recommendation

My recommendation is **B2**. The cleanest one-sentence framing is: **The early ABS zone does not look smaller overall: the dominant called-zone change is a middle full-width expansion, and the zone-only counterfactual moves walks the wrong way.**. If the newsroom wants the sharpest branch logic:

- H1: FAIL as an editorial shrink test
- H2: PASS
- H3: FAIL

That combination supports a **B2** frame. The most accurate Round 1 story is not that ABS made the zone smaller. It is that the called zone changed, but the model evidence says the dominant change either moved the other way or, at minimum, does not explain the walk spike readers are seeing.

## Methods overview

The year-classifier used LightGBM with `learning_rate=0.035`, `num_leaves=31`, `min_child_samples=80`, `subsample=0.9`, `colsample_bytree=0.9`, and early stopping over grouped five-fold cross-validation. The main feature set was `plate_x`, `plate_z`, `plate_z_norm`, `sz_top`, `sz_bot`, `batter_height_proxy=(sz_top-sz_bot)/0.265`, and one-hot pitch-type indicators after collapsing rare pitch types into `OTHER`. Because the SHAP audit showed the zone-estimate fields dominating the full-model AUC, I also fit a location-only auxiliary year classifier for the plate-localization chart and reported its cross-validated AUC separately.

The zone classifiers used separate regularized polynomial-logistic models per season on `is_called_strike ~ plate_x + plate_z_norm`, with `PolynomialFeatures(degree=2)` and `LogisticRegression(C=0.2)`. The headline map uses a 100x100 grid over the observed support, and a region only counts if the point estimate clears 3 pp and the pointwise 95% bootstrap interval excludes zero. I ran **100 bootstrap model pairs** for the delta surface and **100 bootstrap 2025-zone refits** for the counterfactual attribution. Random seeds were fixed from `GLOBAL_SEED=20260423` and written into the artifacts JSON so the run is reproducible.

## Open questions for Round 2

- How much of the residual walk spike is explained by pitch-location drift, pitch-type mix changes, or changes in zone-avoidance strategy by pitchers?
- Do certain pitch classes drive the upper-zone shift disproportionately once we explicitly decompose by fastballs, breaking balls, and offspeed?
- Does the unexplained residual cluster in hitter-friendly counts other than 3-2, suggesting adaptation in sequencing rather than just terminal-pitch adjudication?
- How sensitive is the attribution estimate to replacing the conservative continuation approximation with a more explicit continuation model for extended plate appearances?

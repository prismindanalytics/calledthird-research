# Hot-Start Half-Life - Codex Round 3 Report

## Executive Summary

Round 3 implements the Codex-side blocking fixes without expanding the universe or adding new external joins. I chose QRF calibration Path B: R3 no longer calls the interval layer calibrated. The interval outputs are raw leaf-quantile random-forest bands, and the report states their empirical behavior directly. On the requested train <= 2023, validate 2024, test 2025 diagnostic split, raw hitter coverage is 85.0% on 2024 validation and 85.0% on 2025 holdout; raw reliever coverage is 81.0% and 81.8%. The nonnegative conformal margin remains 0.0000 for hitters and 0.0000 for relievers, so no R3 claim relies on a fake calibration step.

The hitter sleeper board now filters `preseason_prior_woba > 0`, which removes Tristan Peters's zero-prior arithmetic accident. I kept the ranking target as predicted delta because the editorial object is "under-discussed improvement versus prior"; the zero-prior filter is the specific fix that prevents debuts from winning mechanically. The top-10 hitter board is now Everson Pereira, Jorge Barrosa, Samuel Basallo, Jac Caglianone, Dillon Dingler, Coby Mayo, Kyle Karros, Leody Taveras, Brady House, Ildemaro Vargas. The reliever board applies the same low-prior discipline with a 10% prior K-rate floor; the top five are Antonio Senzatela, Blade Tidwell, Daniel Lynch, John King, Caleb Kilian. The R2 cross-method floor survives on the Codex side: Caglianone, Pereira, Barrosa, Basallo, Dingler, Lynch, Senzatela, and King all remain in the output substrate.

The fake-hot rule is stricter: a mainstream hitter must project below `prior - 1 prior SD`. The prior SD is anchored to the 2024 validation SD of ROS wOBA around the deterministic preseason prior, with player multi-year volatility allowed to widen it. One hitter cleared the strict fake-hot rule: Carter Jensen.

Named-starter verdicts are unchanged from Codex R2: Andy Pages NOISE, Ben Rice NOISE, Mike Trout NOISE, Munetaka Murakami AMBIGUOUS, Mason Miller AMBIGUOUS. These labels use raw QRF intervals, not calibrated intervals, so confidence is intentionally modest where the band crosses zero.

## Fix-by-Fix Status

1. QRF calibration framing: done via Path B. R3 reports raw QRF coverage and explicitly drops calibrated-language claims. The 2024 diagnostic margin is reported, but it is not sold as a meaningful calibration.
2. Tristan Peters zero-prior accident: done. `preseason_prior_woba > 0` is required for hitter sleeper eligibility, and Peters is listed as a killed R2 pick.
3. Sleeper ranking rule: done by the permitted filter path. I did not switch to `pred_ros_woba`, because that would turn the board into an absolute talent list and would drop several low-prior cross-method prospect signals. The R3 rule is top-decile predicted delta, non-mainstream, positive preseason prior, ranked by predicted delta.
4. Fake-hot rule: done. The R2 `pred_delta_mean < 0` rule is gone. The new screen requires projected ROS wOBA to fall at least one prior-SD below the preseason prior.
5. xwOBA-gap hedge: done. The reported importance table has one xwOBA-gap feature, `xwoba_minus_prior_woba_22g`. The other two variants are not part of the headline ranking; they remain only as model controls, which the R3 brief allowed.

## Validation and Reproducibility Notes

All R3 model diagnostics use the same temporal discipline: train seasons are 2022 and 2023, validation is 2024, and test is 2025. The 2026 files are inference-only. The production 2026 rankings use bootstrap ensembles and QRFs fit on the same 2022-2023 training split, while the reported error, coverage, feature importance, and prior-SD floor come from validation or holdout data rather than the 2026 outcomes. Seeds are fixed through `SEED = 20260425`, and the new model artifacts are saved under `models/r3/`.

The bootstrap count is 100 for both headline ranking models. That matters because the sleeper boards now expose `pred_delta_bootstrap_sd` or `pred_k_rate_bootstrap_sd` in the saved tables; the point estimate decides rank, but the report keeps uncertainty visible. QRF bands are used as raw interval diagnostics only. They over-cover nominal 80% for hitters by roughly five percentage points on both validation and test, and the reliever bands are closer to nominal but still not calibrated by an active conformal adjustment.

## Corrected Sleeper Rankings

### Hitters

| Rank | Player | Prior | Pred ROS | Delta | Raw 80% Delta |
|---:|---|---:|---:|---:|---|
| 1 | Everson Pereira | 0.220 | 0.324 | +0.104 | [-0.008, +0.166] |
| 2 | Jorge Barrosa | 0.190 | 0.273 | +0.083 | [-0.045, +0.145] |
| 3 | Samuel Basallo | 0.246 | 0.312 | +0.066 | [-0.034, +0.106] |
| 4 | Jac Caglianone | 0.240 | 0.295 | +0.055 | [-0.026, +0.115] |
| 5 | Dillon Dingler | 0.288 | 0.343 | +0.055 | [-0.030, +0.105] |
| 6 | Coby Mayo | 0.238 | 0.283 | +0.045 | [-0.027, +0.115] |
| 7 | Kyle Karros | 0.279 | 0.323 | +0.044 | [-0.034, +0.106] |
| 8 | Leody Taveras | 0.280 | 0.319 | +0.039 | [-0.034, +0.105] |
| 9 | Brady House | 0.255 | 0.294 | +0.039 | [-0.027, +0.106] |
| 10 | Ildemaro Vargas | 0.301 | 0.337 | +0.036 | [-0.035, +0.105] |

The main rank change is mechanical and intended: Peters drops out. Pereira and Barrosa move to the top, Caglianone remains top three, and Basallo plus Dingler remain in the top ten. The top-10 list is still a delta board, so it should be read as "players the model upgrades most versus their prior," not "best projected rest-of-season hitters."

### Relievers

| Rank | Player | Prior K% | Pred ROS K% | Delta | Raw 80% K% |
|---:|---|---:|---:|---:|---|
| 1 | Antonio Senzatela | 12.5% | 24.5% | 12.0% | [17.9%, 31.8%] |
| 2 | Blade Tidwell | 12.8% | 22.5% | 9.7% | [16.0%, 30.4%] |
| 3 | Daniel Lynch | 17.7% | 27.4% | 9.7% | [18.8%, 39.3%] |
| 4 | John King | 13.5% | 22.4% | 8.8% | [17.4%, 31.8%] |
| 5 | Caleb Kilian | 14.8% | 22.6% | 7.8% | [16.3%, 32.5%] |

Cole Wilcox drops because his R2 rank was driven by an 8.3% prior K-rate. The R3 reliever floor is intentionally light, but it prevents the same low-prior arithmetic shape that broke Peters. Varland is not in the R3 top-five sleeper list.

## Strict Fake-Hot Result

| Player | Prior | Prior SD | Threshold | Pred ROS | Shortfall SD |
|---|---:|---:|---:|---:|---:|
| Carter Jensen | 0.404 | 0.051 | 0.353 | 0.345 | 1.16 |

The tightened fake-hot screen is no longer a list of stars who merely project below their own elite priors. Judge and Trout do not clear the rule. Carter Jensen clears because the model projects him more than one prior-SD below a short, very hot preseason prior, and his April xwOBA-minus-prior gap is negative. This is the only R3 Codex fake-hot name I would let into article copy; the old R2 list should not ship.

## Named Starter Verdicts

| Player | R3 verdict | Confidence | Evidence |
|---|---|---|---|
| Andy Pages | NOISE | high | April wOBA 0.395, prior 0.331, projected ROS 0.328 with raw 80% delta band [-0.053, +0.039]. |
| Ben Rice | NOISE | medium | April wOBA 0.521, prior 0.326, projected ROS 0.345 with raw 80% delta band [-0.046, +0.065]. |
| Mike Trout | NOISE | medium | April wOBA 0.423, prior 0.367, projected ROS 0.375 with raw 80% delta band [-0.054, +0.057]. |
| Munetaka Murakami | AMBIGUOUS | low | April wOBA 0.418, prior 0.291, projected ROS 0.350 with raw 80% delta band [-0.028, +0.106]. |
| Mason Miller | AMBIGUOUS | medium | April K% 65.9%, prior 39.1%, projected ROS 33.3% with raw 80% K% band [23.1%, 43.5%]. |

No named-starter verdict changed from R2; the R3 changes affect framing and rankings rather than the named verdict labels. The most important interpretive change is not a label flip; it is interval honesty. Rice and Trout still fail the Codex signal rule because their raw delta bands cross zero and their point estimates are not large enough to override the prior. Murakami remains the closest hitter to a signal, but the raw lower bound remains below zero and the prior is still a league-style fallback rather than a true NPB translation. Mason Miller remains an elite strikeout arm, but Codex does not certify the April strikeout rate as a sustainable jump over his already-high prior.

## Feature Importance Framing

R3 reports one xwOBA-gap feature: `xwoba_minus_prior_woba_22g`. It ranked 2 in the reported 2025 holdout permutation table and 2 before removing the two unreported gap variants from the displayed ranking. The other gap variants are not used as separate evidence in the report. This keeps the useful contact-quality signal while removing the R2 hedge where three correlated versions of the same idea could be quoted opportunistically.

## What Changed From R2

The killed R2 picks are: Tristan Peters (zero-prior hitter sleeper removed by preseason_prior_woba > 0 filter), Cole Wilcox (low-prior reliever dropped by R3 prior K% floor), Louis Varland (excluded from R3 sleeper-reliever list; remains fake-dominant). The QRF section changed from a calibration claim to a raw-coverage diagnostic. The fake-hot board changed from a loose negative-delta screen to a stricter prior-SD shortfall screen. The feature-importance section changed from three xwOBA-gap variants to one reported gap feature.

The corrected outputs are deliberately less theatrical. R3 preserves the cross-method sleeper core while removing the specific artifacts the cross-review found. It also leaves room for honest disagreement with Agent A on Rice, Trout, Murakami, or Miller if their corrected Bayesian pipeline lands elsewhere.

## Open Questions

Future work should build a real NPB-to-MLB prior for Murakami, validate whether reliever priors should use role-specific recent windows, and decide whether a truly calibrated interval should be obtained by tuning QRF quantile levels rather than applying a nonnegative conformal expansion to an already over-covering interval. Those are Round 4 questions and were not used here.

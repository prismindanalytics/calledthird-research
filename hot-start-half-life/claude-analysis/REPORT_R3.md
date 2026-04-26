# Hot-Start Half-Life — Claude (Agent A) Round 3 Report

**Date:** 2026-04-25
**Methodology lane:** Bayesian / interpretability-first
**Status:** All four R2 cross-review blockers closed; convergence-check substrate generated.

---

## 1. Executive summary

R3 closes the four methodology defects Codex identified in cross-review of my R2 work. The two most consequential changes are:

1. **The contact-quality blend was validated on a 2025 holdout — and the learned coefficients reverse my R2 Rice/Trout SIGNAL calls.** A linear blend trained on 2022-2024 player-seasons (`first-22-games` window mapping to ROS wOBA, with features wOBA + xwOBA + EV p90 + HardHit% + Barrel% + 3-year prior wOBA + intercept) beats a wOBA-only baseline by 1.2 wOBA-points RMSE on 2025 (0.0359 vs 0.0371; n_train=692, n_holdout=340). The learned coefficient on raw window wOBA is essentially zero (-0.023); the dominant features are xwOBA (0.165) and the 3-year prior (0.279). When this learned blend is applied to Rice and Trout, both ROS-vs-prior delta posteriors collapse onto zero (Rice q10/q50/q90 = -.002/+.002/+.006; Trout = -.003/+.001/+.005). **Both downgrade from R2 SIGNAL to R3 NOISE** with high confidence — this is the convergence-with-Codex outcome the brief calls "the good outcome."

2. **The hierarchical Bayesian model is now the production wOBA estimator, not a side-output.** R2 Codex correctly noted the partial-pooling shared-kappa NUTS fit existed but rankings used per-player conjugate updates — pure window dressing. In R3 I fit shared-kappa wOBA AND xwOBA models (kappa R-hat = 1.004/1.004, ESS = 611/1506; both inside the brief's R-hat <= 1.01 / ESS >= 400 bar) and feed those per-player rho_p posterior samples directly into the learned blend. The kappa q50 is ~211 for wOBA (universe pools toward each player's 3-year prior with ~211 effective PA of weight) and ~602 for xwOBA (heavier pooling — xwOBA is more stable across players).

3. **The stabilization bootstrap, corrected to direct player-season resampling, leaves the wOBA-shifted finding INTACT** — and actually tightens the CI. R2 sampled players then chose one season per player per draw (changed estimand). R3 samples N player-seasons directly with replacement from the 1,433 qualifying. The corrected wOBA half-stab estimate is 451 PA [95% CI 335-638] vs Carleton 280 — non-overlapping. ISO also flags shifted under the corrected method (point 199, CI 162-228, vs Carleton 160 at the boundary). BB% / K% / BABIP remain consistent with Carleton.

4. **Varland is now FAKE-DOMINANT only.** The two reliever-board rules are made mutually exclusive: any reliever flagged fake-dominant cannot also be on the sleeper list. Varland (obs - post = +.113) clears the fake-dominant gate and is excluded from sleepers. The 3-year weighted prior is the methodologically defensible choice over a late-2025 closer-window prior.

**Headline R3 verdicts on the 5 named hot starters:**

| Player | R2 verdict | **R3 verdict** | Confidence | One-line evidence |
|---|---|---|---|---|
| Andy Pages | NOISE | **NOISE** | low | Delta q50 = -0.007; learned blend confirms BABIP excursion. |
| Ben Rice | SIGNAL | **NOISE** | high | Delta q50 = +0.002 once learned coefficients replace 50/50. |
| Mike Trout | SIGNAL | **NOISE** | high | Delta q50 = +0.001; learned coef puts only +0.165 on xwOBA. |
| Munetaka Murakami | SIGNAL | **SIGNAL** | medium | Delta q50 = +0.021 [+.018 / +.025]; cleanest survivor. |
| Mason Miller (K%) | SIGNAL | **SIGNAL** | medium | K% +0.089 above prior, survives reliever-K% shrinkage. |

**Top-10 R3 sleeper hitters:** Caglianone, Barrosa, Pereira, Basallo, Peraza, House, Davis, Lockridge, Mayo, Vivas.

**Top-5 R3 sleeper relievers:** Senzatela, Kilian, King, Weissert, Phillips.

**Killed from R2:** Dingler, Vargas, Lopez (hitter sleepers — drop out of top-15 once learned blend is applied); Varland and Lynch (reliever sleepers — both fail the new mutual-exclusion coherence rule).

---

## 2. Fix-by-fix status (with kill-gate verdicts)

### Fix 1: Contact-quality blend mislabeling — **CLOSED (Path A: validated learned blend)**

**Claim now supported:** The R3 ROS estimator uses a learned linear blend of raw observed wOBA, observed xwOBA, observed EV p90, observed HardHit%, observed Barrel%, and the 3-year-weighted prior wOBA, with coefficients fit on 2022-2024 player-seasons and validated on a held-out 2025 panel of 340 player-seasons. The learned blend beats a wOBA + prior baseline by 1.2 wOBA-points of RMSE on the holdout (0.0359 vs 0.0371). Each of EV p90, HardHit%, and Barrel% receives a non-zero learned coefficient and contributes to ROS_wOBA at the per-player level.

| Feature | Coefficient | Notes |
|---|---|---|
| wOBA_window | -0.024 | Essentially zero — raw window wOBA is dominated by BABIP noise; xwOBA carries the contact-quality signal. |
| xwOBA_window | +0.165 | Primary contact-quality signal. |
| EV_p90_window | +0.003 | Small but positive (per-mph) contribution. |
| HardHit_window | -0.025 | Negative — likely correlates with launch-angle profile that hurts wOBA ROS. Multicollinearity with EV. |
| Barrel_window | +0.039 | Positive — barrels have predictive power beyond xwOBA. |
| prior_wOBA | +0.279 | Strongest single predictor; the prior is the dominant anchor. |
| intercept | -0.103 | League-average shift. |

**Kill-gate verdict:** PASS. Either path acceptable; learned blend RMSE < wOBA-only baseline RMSE on 2025 by 1.2 wOBA-points (threshold was 0.2 wOBA-points). ADOPT. The R2 framing of "Rice/Trout SIGNAL via contact-quality features" is **retracted** — under the learned coefficients both downgrade to NOISE with high confidence.

### Fix 2: Hierarchical labeling honesty — **CLOSED (Path A: integrated)**

**Claim now supported:** wOBA and xwOBA priors in the production R3 ranking come from shared-kappa NUTS partial-pooling models (one fit per metric across all 279 universe hitters). Per-player rho_p posterior samples become the wOBA_window and xwOBA_window inputs to the learned blend; the per-player conjugate Beta-Binomial path used in R2 is dropped from the wOBA pipeline.

Convergence diagnostics:
- wOBA fit: kappa q50 = 210.7, R-hat = 1.0044, ESS = 611
- xwOBA fit: kappa q50 = 602.3, R-hat = 1.0040, ESS = 1506

Both inside R-hat <= 1.01 / ESS >= 400 bar.

The conjugate Beta-Binomial path is retained ONLY for side-channel rates (BB%, K%, BABIP, ISO, HardHit%, Barrel%) that the production blend does not consume from posterior space. Those side channels are still useful for the per-player diagnostic display but no longer drive any verdict.

**Kill-gate verdict:** PASS. Codex's "hierarchical is window dressing" critique closed.

### Fix 3: Stabilization bootstrap estimand — **CLOSED**

**Estimand corrected:** R3 samples player-seasons directly with replacement from the 1,433 qualifying-player-seasons (>= 200 PA in 2022-2025). A single batter's multiple seasons can co-appear in one draw; some player-seasons can appear multiple times in one draw. The R2 method (sample players then choose one season per player) is dropped.

| Stat | R3 half-stab PA | 95% CI | Carleton | Verdict |
|---|---|---|---|---|
| BB% | 118 | [96, 147] | 120 | consistent_with_carleton |
| K% | 54 | [50, 65] | 60 | consistent_with_carleton |
| ISO | 199 | [162, 228] | 160 | **shifted** (Carleton at boundary) |
| BABIP | 629 | [580, 6400*] | 820 | consistent_with_carleton (right-censored upper) |
| **wOBA** | **451** | **[335, 638]** | **280** | **shifted** |

*BABIP upper CI hits the 1600-PA grid maximum x 4 censoring placeholder for 12 of 150 draws.*

**Kill-gate verdict:** wOBA-shifted finding **survives and tightens** under the corrected estimand (R2's CI was 298-3600 with right-censoring at the upper bound; R3's is 335-638 — non-overlapping with Carleton 280). ISO now also crosses the threshold (the lower bound 162 is above Carleton 160). BB%, K%, BABIP remain consistent. The R3 finding is "wOBA half-stab is meaningfully higher than Carleton in the 2022-2025 era; ISO is at the boundary."

### Fix 4: Varland coherence — **CLOSED**

**Rule change:** Sleeper and fake-dominant lists are now **mutually exclusive**. Any reliever satisfying the fake-dominant gate (`obs_K% >= 0.30 AND obs_K% - post_K%_q50 >= 0.08`) is removed from sleeper consideration, regardless of post-prior delta.

**Varland disposition:** FAKE-DOMINANT only.
- April K% = .415 on 41 BF; post K% q50 = .301; obs - post = +.113 (clears the 0.08 fake-dominant gate).
- Disclosed prior choice: 3-year weighted (BF~700+ across 2023-2025), not a late-2025 closer-window prior. Methodologically defensible — the late-2025 narrow window would be a post-trade role-change cherry-pick.
- The R3 sleeper list (Senzatela, Kilian, King, Weissert, Phillips, Martin) excludes anyone on the fake-dominant board.

**Kill-gate verdict:** Varland appears on at most one list (FAKE-DOMINANT). Lynch is also moved (was R2 sleeper #3, R2 fake-dominant #2, now R3 fake-dominant only). King, who was NOT fake-dominant in R2, stays on the sleeper list.

---

## 3. Re-ranked sleeper lists

### Top-10 sleeper hitters (R3, post-fix)

| Rank | Player | PA | Prior wOBA | April wOBA | R3 ROS q50 | ROS-vs-prior q10 / q50 / q90 |
|---|---|---|---|---|---|---|
| 1 | Caglianone, Jac | 79 | .240 | .328 | .321 | +.077 / +.082 / +.087 |
| 2 | Barrosa, Jorge | 55 | .178 | .344 | .257 | +.075 / +.079 / +.083 |
| 3 | Pereira, Everson | 50 | .220 | .416 | .290 | +.066 / +.071 / +.075 |
| 4 | Basallo, Samuel | 73 | .246 | .339 | .312 | +.061 / +.066 / +.071 |
| 5 | Peraza, Oswald | 70 | .240 | .406 | .298 | +.054 / +.058 / +.063 |
| 6 | House, Brady | 90 | .255 | .301 | .306 | +.046 / +.051 / +.055 |
| 7 | Davis, Henry | 54 | .259 | .222 | .306 | +.043 / +.047 / +.051 |
| 8 | Lockridge, Brandon | 70 | .266 | .284 | .305 | +.034 / +.038 / +.042 |
| 9 | Mayo, Coby | 69 | .288 | .286 | .325 | +.033 / +.037 / +.041 |
| 10 | Vivas, Jorbit | 63 | .270 | .389 | .302 | +.027 / +.032 / +.036 |

The R2 convergent-with-Codex five (Caglianone, Pereira, Barrosa, Basallo, Dingler) — four survive R3 in the top 4. **Dingler dropped** out of the top 15 because his learned-blend ROS q50 (+.021) sits below Mayo (+.037), Vivas (+.032), and several others. Dingler still has positive delta but the top-decile cutoff under the learned coefficients is +0.018, and he sits inside that band — a borderline downgrade rather than a kill.

### Top-5 sleeper relievers (R3, post-fix)

| Rank | Player | BF | Prior K% | April K% | R3 post K% q50 | Delta | Shrinkage |
|---|---|---|---|---|---|---|---|
| 1 | Senzatela, Antonio | 58 | .119 | .259 | .181 | +.062 | 0.55 |
| 2 | Kilian, Caleb | 41 | .148 | .293 | .206 | +.058 | 0.58 |
| 3 | King, John | 40 | .137 | .275 | .185 | +.049 | 0.64 |
| 4 | Weissert, Greg | 42 | .216 | .333 | .259 | +.043 | 0.63 |
| 5 | Phillips, Tyler | 60 | .169 | .267 | .211 | +.043 | 0.54 |

R2's convergent three (Lynch, Senzatela, King): Senzatela and King survive; Lynch drops because he is on the fake-dominant board (obs - post = +.132, clears the 0.08 gap).

---

## 4. Re-verdicted named hot starters

The R2-vs-R3 verdict diff is dominated by the learned-blend reversal of Rice and Trout.

### Andy Pages — **NOISE (low confidence)** — unchanged
- April wOBA .403, xwOBA .360, EV p90 106.2 — the wOBA / xwOBA gap of -.043 is the BABIP-luck story.
- R3 hier-pooled wOBA q50 = .352; learned-blend ROS q50 = .327; delta vs prior = -0.007.
- 4-method consensus from R2 holds; no evidence to revise.

### Ben Rice — **NOISE (high confidence)** — **changed from R2 SIGNAL**
- April wOBA .549, xwOBA .481, EV p90 106.4 — the contact quality is genuinely strong.
- The R2 SIGNAL verdict came from a hand-tuned 50/50 wOBA + xwOBA blend that effectively averaged the .549 and .481 to give a .47-ish ROS prior, well above his .345 3-year prior.
- R3's learned coefficients put -0.024 on raw window wOBA, +0.165 on xwOBA, and +0.279 on the prior. Rice's hier-pooled xwOBA q50 = .396 (heavily pulled toward his .345 prior). The blend evaluation gives ROS_wOBA q50 = .347 — virtually identical to his prior. Delta-vs-prior 80% interval is [-0.002, +0.006].
- Editorial read: April was a real performance; the learned model says it doesn't rewrite the prior. The downgrade from SIGNAL to NOISE is the convergence-with-Codex outcome.

### Mike Trout — **NOISE (high confidence)** — **changed from R2 SIGNAL**
- April wOBA .420, xwOBA .476, EV p90 108.7 — elite contact quality.
- Same mechanism as Rice: R2's hand-tuned blend over-weighted xwOBA. R3's learned coefficients give Trout's hier-pooled xwOBA q50 = .394 (his prior was .362, so partial pooling holds it close), and ROS q50 = .363, virtually equal to his prior. Delta = +0.001 [-0.003, +0.005].
- The R2 SIGNAL framing was an artifact of the unvalidated 50/50.

### Munetaka Murakami — **SIGNAL (medium confidence)** — unchanged
- April wOBA .427, xwOBA .435, EV p90 110.8 (HIGHEST in the universe) on 103 PA.
- Prior is league-average (.326) because he has no MLB history; the R3 partial-pooling model can't dramatically push his prior up without evidence beyond the window. But the blend's coefficient on Barrel% (+0.039) and EV p90 (+0.003) credits the underlying contact for a real ROS lift.
- R3 ROS q50 = .347; delta = +.021 [+.018, +.025]. Tighter and lower-magnitude than R2's +0.064 (R2 over-credited xwOBA), but the SIGNAL verdict survives at medium confidence.
- Caveat: prior is league-average by necessity. An NPB-translation prior would be more informative.

### Mason Miller — **SIGNAL (medium confidence)** — unchanged
- K%-only verdict; streak survival probabilities remain killed.
- April K% .650 on 40 BF; 3-year prior K% .407; R3 posterior K% q50 = .495; rise of +0.089 above prior.
- The shrinkage_to_prior is 0.636 (heavy — only 36% of the data overrode the prior), but the 9-point K% rise above an already-elite prior is durable signal. He IS on the fake-dominant board (obs - post = +.155) but as a closer he's filtered out of the sleeper list anyway; the editorial framing is "K% rise is real, streak length is not credibly forecastable."

---

## 5. What changed from R2 and why

| Item | R2 | R3 | Reason |
|---|---|---|---|
| Rice verdict | SIGNAL | NOISE | Learned blend coefficients on 2025 holdout reverse the hand-tuned 50/50 |
| Trout verdict | SIGNAL | NOISE | Same mechanism — xwOBA gets +0.165 not +0.500 |
| Murakami delta_q50 | +0.064 | +0.021 | Hierarchical pooling shrinks his "elite April" closer to league-average prior |
| Sleeper hitter top-10 includes | Dingler, Vargas, Lopez | NOT Dingler/Vargas/Lopez (replaced by Peraza, House, Davis, Lockridge, Mayo) | Learned blend down-weights raw window wOBA; their "real" delta vs prior is smaller than R2's hand-tuned blend implied |
| Sleeper reliever #1 | Varland | Senzatela | Varland is FAKE-DOMINANT under R3 mutual-exclusion rule |
| Sleeper relievers top-5 includes | Varland, Senzatela, Lynch, Kilian, King | Senzatela, Kilian, King, Weissert, Phillips | Varland and Lynch fail mutual-exclusion |
| wOBA stab finding | "shifted" (CI 298-3600, right-censored) | "shifted" (CI 335-638, well-defined) | Direct player-season bootstrap tightens CI; finding strengthens |
| ISO stab finding | consistent_with_carleton | shifted (boundary) | New finding; CI 162-228 sits above Carleton 160 |
| Hierarchical fit role | side-output sanity check | production estimator | Codex critique closed |

---

## 6. Open questions (out of R3 scope)

1. **NPB-translation prior for Murakami.** His SIGNAL verdict relies on a league-average prior because no MLB history exists. With an NPB-translation prior in the .335-.350 range, his R3 ROS delta would tighten further but probably stay positive.
2. **Component-stat hierarchical fits.** I integrated the hierarchical model for wOBA and xwOBA only. Doing the same for HardHit%, Barrel%, and EV p90 would give properly-shrunk inputs to the learned blend (currently those go in raw). Likely small effect — those stats stabilize quickly — but worth a Round 4 tighten.
3. **Component-stat changepoints for Mason Miller.** The killed streak model could be replaced with a `delta_run_exp` accumulation per BF, but that requires re-pulling 3 years of pitch-level data with that column.
4. **Real fake-hot list under tighter rule.** R3 produces only 1 fake-hot name (under the universe-SD threshold of -0.034). The "no fake-hot list ships" outcome is publishable but worth understanding — is the universe SD too forgiving, or is there genuinely no big April-overperformer in the mainstream-top-20 set this year?

---

## 7. Convergence substrate

`r3_convergence_check.json` is the file the comparison memo will read. Schema as specified in §4.3 of the R3 brief, with all 5 named-starter verdicts (NOISE/NOISE/NOISE/SIGNAL/SIGNAL), top-10 sleeper hitters, top-5 sleeper relievers, killed picks, and verdicts changed from R2.

If Codex's R3 also reverses Rice/Trout to NOISE — that's the cleanest possible outcome for the article: both methods now agree the published-leaderboard names are mostly noise, and the convergent sleeper set (with Caglianone, Pereira, Basallo at minimum) is the differentiated content.

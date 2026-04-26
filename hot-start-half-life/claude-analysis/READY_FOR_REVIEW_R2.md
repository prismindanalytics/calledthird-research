# Claude (Agent A) — READY FOR REVIEW R2

## Top-5 sleeper hitters
(Top decile of predicted ROS-vs-prior wOBA delta AND not in ESPN top-20 by OPS.)

1. **Jac Caglianone (KC)** — 79 PA, obs .328 / xwOBA .350, prior .240, ROS-vs-prior +0.051 (q10 +0.032). EV90 111 mph, HardHit 55%.
2. **Everson Pereira (NYY)** — 50 PA, obs .416 / xwOBA .403, prior .220, +0.038 (+0.020). HardHit 49%, Barrel 14%.
3. **Jorbit Vivas (NYM)** — 63 PA, obs .389 / xwOBA .316, +0.036. wOBA ahead of xwOBA — partial BABIP-luck.
4. **Samuel Basallo (BAL)** — 73 PA, obs .339 / xwOBA .374, +0.035. xwOBA ahead of wOBA — under-luck.
5. **Jorge Barrosa (ARI)** — 55 PA, obs .344 / xwOBA .247, +0.032. Smaller signal; weak prior amplifies.

Bonus: **Dillon Dingler (DET)** xwOBA .438 vs wOBA .345 (+0.093 gap).

## Top-5 sleeper relievers
(K%-rise >= +0.04 vs 3yr prior; not in 2025 top-30 saves.)

1. **Louis Varland (BAL)** — prior K .239 -> post .301 (+0.062). 817 BF prior anchor.
2. **Antonio Senzatela (COL)** — prior K .119 -> post .181 (+0.062). Converted starter.
3. **Daniel Lynch (KC)** — prior K .172 -> post .231 (+0.060).
4. **Caleb Kilian (CHC)** — prior K .148 -> post .206 (+0.058). Thin prior.
5. **John King (TEX)** — prior K .137 -> post .185 (+0.049).

## Fake-hot / fake-cold (kill-gates failed)

H2 (fake hot >= 3): **FAIL** — only 2 mainstream-listed hitters with negative ROS-vs-prior delta (Aaron Judge, Xavier Edwards). Drop section per brief.

H3 (fake cold >= 3): **FAIL** — only 1 (Henry Davis, +0.009). Section dropped.

The xwOBA-gap analysis is the article-worthy alternative — top under-performers (xwOBA >> wOBA): Bo Naylor (+.123), Harrison Bader (+.119), Ke'Bryan Hayes (+.111), Patrick Bailey (+.104), Austin Wells (+.098). Credible buy-low names; Bayesian projection doesn't push their priors positive.

## Methodology fix status

| Fix | Status |
|---|---|
| Murakami reproducibility (MLB Stats API) | **done** |
| Stabilization player-season cluster bootstrap | **done** |
| Projection prior includes contact quality | **done** |
| Mason Miller streak model | **killed** (per brief) |
| Hierarchical labeling honesty | **done** (kappa ~ HalfNormal partial pooling; R-hat 1.009, ESS 1,247) |

## Findings to flag

1. **R1 stabilization claim collapses.** With proper cluster bootstrap, only wOBA still shifted vs Carleton (upper CI wide due to right-censoring). BB%, K%, ISO, BABIP all CIs contain Carleton. R1's "non-overlapping CIs" was a within-player partition artifact.
2. **Two named-starter flips: Rice and Trout.** Both R1-AMBIGUOUS -> R2-SIGNAL once contact-quality enters the prior. Rice ROS-vs-prior CI [+0.026, +0.066]; Trout [+0.005, +0.045]. Pages NOISE, Murakami SIGNAL, Miller K% SIGNAL all hold.
3. **Mason Miller K% finding convergent** (R1 .500, R2 .495 posterior median). Streak probabilities (65/45/33%) are killed.

## Round 3 deferrals (not pursued)

NPB-translation prior for Murakami; delta_run_exp reliever streak model; era counterfactual with matched samples; park-factor-adjusted xwOBA gap; out-of-sample coverage of per-component shrinkage priors; stabilization grid extension to tighten BABIP/wOBA upper CIs.

Universe: 279 hitters, 233 relievers (both above brief thresholds).

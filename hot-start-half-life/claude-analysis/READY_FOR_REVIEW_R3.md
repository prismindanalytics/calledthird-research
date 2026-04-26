# Claude (Agent A) — READY FOR REVIEW R3

**Date:** 2026-04-25
**Methodology lane:** Bayesian / interpretability-first

## R3 fix-status checklist (all 4 closed)

| # | Fix | Status | Module |
|---|---|---|---|
| 1 | Contact-quality blend mislabeling | **done** (Path A: validated learned blend; ADOPT) | `r3_blend_validation.py` |
| 2 | Hierarchical labeling honesty | **done** (Path A: integrated as production wOBA + xwOBA estimator) | `r3_hierarchical_production.py` |
| 3 | Stabilization bootstrap estimand | **done** (direct player-season resampling) | `r3_stabilization.py` |
| 4 | Varland coherence | **done** (mutual-exclusion rule) | `r3_reliever_board.py` |

## Named-starter R3 verdicts

| Player | R2 | **R3** | Conf | Delta-vs-prior posterior q10/q50/q90 |
|---|---|---|---|---|
| Andy Pages | NOISE | **NOISE** | low | -0.010 / -0.007 / -0.003 |
| Ben Rice | SIGNAL | **NOISE** | high | -0.002 / +0.002 / +0.006 |
| Mike Trout | SIGNAL | **NOISE** | high | -0.003 / +0.001 / +0.005 |
| Munetaka Murakami | SIGNAL | **SIGNAL** | medium | +0.018 / +0.021 / +0.025 |
| Mason Miller (K%) | SIGNAL | **SIGNAL** | medium | K% +.089 above prior |

Rice and Trout downgraded because the learned blend (validated on 2025 holdout, RMSE 0.0359 vs wOBA-only baseline 0.0371; gain +1.2 wOBA-pts) puts only +0.165 on xwOBA and +0.279 on the prior — R2's 50/50 was over-crediting xwOBA. **This is the convergence-with-Codex outcome the brief flagged as the GOOD outcome.**

## Top-10 sleeper hitters (R3)

1. Caglianone, Jac (+.082) [convergent w/ Codex R2]
2. Barrosa, Jorge (+.079) [convergent]
3. Pereira, Everson (+.071) [convergent]
4. Basallo, Samuel (+.066) [convergent]
5. Peraza, Oswald (+.058)
6. House, Brady (+.051)
7. Davis, Henry (+.047)
8. Lockridge, Brandon (+.038)
9. Mayo, Coby (+.037)
10. Vivas, Jorbit (+.032)

## Top-5 sleeper relievers (R3)

1. Senzatela, Antonio (+.062 K%, shrink 0.55) [convergent]
2. Kilian, Caleb (+.058, 0.58)
3. King, John (+.049, 0.64) [convergent]
4. Weissert, Greg (+.043, 0.63)
5. Phillips, Tyler (+.043, 0.54)

## Killed from R2

- **Hitter sleepers dropped:** Dingler, Vargas, Lopez (still positive delta but below R3 top-decile cutoff once learned blend down-weights raw wOBA).
- **Reliever sleepers dropped:** Varland (FAKE-DOMINANT — coherence), Lynch (FAKE-DOMINANT — coherence).

## What changed from R2

1. **Rice / Trout NOISE downgrades** — learned-blend reverses both.
2. **wOBA stabilization "shifted" strengthens** — direct player-season bootstrap CI is 335-638 PA (R2 had 298-3600 right-censored). ISO also flags (CI 162-228 vs Carleton 160).
3. **Hierarchical fit is now production** — shared-kappa NUTS (R-hat 1.004, ESS 611/1506).
4. **Reliever board coherence** — mutual-exclusion rule. Varland/Lynch leave sleepers; King/Senzatela survive.

## Substrate written

- `claude-analysis/r3_convergence_check.json`
- `claude-analysis/findings_r3.json`
- `claude-analysis/REPORT_R3.md`
- `claude-analysis/charts/r3/*.png`

## Round-4 candidates (NOT in R3 scope)

1. NPB-translation prior for Murakami.
2. Hierarchical fits for HardHit/Barrel/EV p90.
3. `delta_run_exp` per-BF reliever streak survival.
4. Whether "no fake-hot under stricter rule" (1 name) is null.

# READY_FOR_REVIEW — Claude (Agent A), Round 1

## Headline

**Stabilization rates have shifted in the 2022-2025 era for three of five rate stats.** wOBA half-stabilization needs **~75% more PA** today (489 PA vs Carleton's 280; 95% bootstrap CI 396-569). ISO needs **~24% more PA** (198 vs 160; CI 176-238). BABIP stabilizes **~24% faster** (627 PA vs 820; CI 583-747). BB% (122 vs 120) and K% (54 vs 60) match Carleton 2007/2013. **H1 supported.**

The structural read: wOBA is slower despite BABIP being faster — the slowdown is in the *power* half (HRs, XBH), not the singles half. Shift restrictions tightened BABIP signal-to-noise; launch-angle / post-deadened-ball variance loosened ISO's.

## Per-starter verdicts

- **Andy Pages — NOISE.** Every stat's prior sits inside the posterior 80% interval. PELT detects no break in 2026. ROS wOBA q50 = .344, q90 = .370. The .409 BA is a 94-PA BABIP excursion the model liquidates immediately.

- **Ben Rice — AMBIGUOUS.** BB%, K%, ISO are signals (.200 BB% real, .429 ISO real); BABIP and wOBA are noise. wOBA shrinks from .500 to posterior q50 = .368, q90 = .395. Real All-Star, not the MVP-trajectory pace the headline implies.

- **Munetaka Murakami — SIGNAL.** 4 of 5 stats register signal vs the deliberately weak league prior (60-PA effective). Posterior wOBA q50 = .378, q90 = .427. Caveat: NPB→MLB translation prior would tighten this.

- **Mike Trout — AMBIGUOUS.** Only K% (.200 vs .298 prior) is a clean signal. BABIP suppression (.228 vs .301) is the swing factor; wOBA q50 = .373.

- **Mason Miller — AMBIGUOUS, STREAK PROBABILISTICALLY DURABLE.** K% rises to posterior q50 = .500 (signal). Streak simulation: **65% probability ≥ 5 more scoreless IP, 45% ≥ 10 IP, 33% ≥ 15 IP**.

## Noise floor

Of 20 player-seasons that led 22-game wOBA in 2022-2025, **only 10% sustained ≥ 85% of that wOBA the rest of season**. Median wOBA decline = -0.135 points. April leaders generally don't persist.

## Convergence

All 27 Bayesian fits: R-hat ≤ 1.003, ESS ≥ 2,800. Stabilization bootstrap: 200 iterations; point estimates stable to ±2 PA.

## Kill-gates

Stabilization shift ≥ 10%: **PASS** (3 of 5). Named-starter coverage: **PASS** (Miller 41 BF). Cross-agent agreement: **pending Codex**. Historical-analog quality: N/A (Codex's lane).

## Open questions for Round 2

1. ISO/wOBA shift mechanism — HR vs XBH variance decomposition.
2. BABIP CI tightening via cross-season player pooling.
3. NPB→MLB translation prior for Murakami.
4. Trout durability-adjusted projection.
5. Pitch-level `delta_run_exp` survival model for Miller's streak.
6. Causal mechanism behind wOBA stabilization shift (out of brief scope).

## Caveats

FanGraphs `leaders-legacy.aspx` was 403'd today; all aggregates computed from Statcast (upstream source). Murakami required manual MLBAM lookup via MLB stats API. 2026 `sz_*` columns excluded everywhere.

## Files

`analyze.py`, `data_pull.py`, `stabilization.py`, `bayes_projections.py`, `changepoint.py`, `analogs_lite.py`, `findings.json`, `charts/` (stabilization, league env, per-player projection + changepoint, MCMC trace plots).

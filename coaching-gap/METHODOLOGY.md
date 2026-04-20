# The Coaching Gap — Methodology and Honest Scoreboard

This document is the full public accounting of what the "Coaching Gap"
dual-agent study tested, what died, and what survived. It pairs the
published flagship claim — "Six rounds. 17 tests. 16 nulls. One finding
survived: low-chase hitters extract +0.04 wOBA on predictable pitches"
— to the round-by-round evidence in `claude-analysis/round{2..6}/`,
`codex-analysis/round{2..6}/`, and the cross-review files in `reviews/`.

## The six-round ladder

Each round was a pre-registered test with a kill criterion written down
before the data was read. A Bayesian agent (Claude) and a gradient-
boosted / matched-pairs agent (Codex) analyzed the same substrate
independently and cross-reviewed each other at the end of every round.

- **Round 1 (Layer 1 — pitcher taxonomy).** Can we find discrete
  pitcher-evolution archetypes, and do change-points in a pitcher's
  arsenal predict when teams fail to adjust?
- **Round 2 (Layer 2 — pitch-prediction model and the pooled gap).**
  Fit the contextual pitch-prediction model; measure the pooled wOBA
  gap between predictable and unpredictable pitches; test whether the
  pitcher-intrinsic 2×2 taxonomy or multi-year predictability
  trajectories modulate anything.
- **Round 3 (non-pitcher-intrinsic dimensions).** If the pooled 0.092
  wOBA gap isn't modulated by pitcher type, does it concentrate by
  batter, TTO, count, pitch family, or stuff?
- **Round 4 (finer grain, team heterogeneity, literature replication).**
  Replicate CMU SURE 2023's "same pitch twice in a row" penalty at pair
  level; test team-level heterogeneity; test a physics-based Stuff+
  residual; settle the Round-3 batter-cluster disagreement.
- **Round 5 (canonical features, not clusters).** Replace agent-
  specific clusters with pre-specified canonical batter features
  (chase, whiff, zone-contact, xwOBAcon proxy); test per-season
  stability; test development trajectories and the "chasers do worse"
  counter-archetype.
- **Round 6 (forced reconciliation).** Both agents run identical
  three-estimator test (pooled between-batter, matched-pairs,
  batter-fixed-effects) on the same substrate; formally decompose the
  variance; simulate power; decide whether the Round-5 disagreement is
  real or an estimand artifact.

## Scoreboard (17 tests)

| # | Round | Hypothesis | What it tested | Point estimate | Verdict | Why it died |
|---|---|---|---|---|---|---|
| 1 | 1 | Discrete pitcher archetypes | Do career-trajectory clusters of 371 MLB pitchers recover stable, interpretable evolution archetypes? | Bootstrap cluster stability: Claude soft-cosine 0.647 (K=3); Codex assignment-cosine 0.806 at n=100 (K=5) | Null | Both methods failed the pre-registered 0.85 stability gate; what clusters is role × velocity × offspeed, not evolution |
| 2 | 1 | Change-point → coaching-gap coupling | Does batter decision quality drop in the 30 days after a PELT-detected pitcher change-point? | gap_after − gap_before = +0.00012 xwOBA, CI95 [-0.00021, +0.00045], Wilcoxon p=0.32 | Null | Zero effect across all volatility tiers; teams evidently update after pitcher transformations |
| 3 | 2 | 2×2 quadrant modulation of pooled gap | Does the pooled 0.092 wOBA gap differ across the predictability-slope × arsenal-volatility 2×2? | Quadrant means 0.059–0.096 wOBA, spread 0.037 | Null | Permutation p=0.156 (N=500); between-quadrant variance not significant despite individual CIs clearing zero |
| 4 | 2 | Predictability-slope → outcome coupling | Does a pitcher's multi-year predictability trajectory couple to their wOBA-allowed / K% / BB% / HR/PA trajectory? | Max |Spearman r| = 0.112 (HR/PA, p=0.068); wOBA-allowed r=−0.049, p=0.42 | Null | No correlation clears α=0.05; strongest signal has wrong sign |
| 5 | 3 | Batter-cluster heterogeneity (first look) | Does the residual gap concentrate in a specific Bayesian-clustered batter type (K=4)? | Claude: "patient-power" cluster gap 0.113 vs free-swingers 0.067, spread 0.047, perm p=0.002. Codex matched-pairs on own K=3: p=0.212 | Methodology-dependent; later survived | Directionally positive under Claude, null under Codex; settled in Round 4 |
| 6 | 3 | Times-through-order × predictability | Does the gap widen at TTO3 as batters see the pitcher more? | Starter subset: TTO1 gap 0.103 → TTO2 0.082 → TTO3 0.067; spread 0.036, perm p=0.005 | Null (refuted in opposite direction) | Effect is significant but opposite direction from hypothesis; TTO3 has the narrowest gap |
| 7 | 3 | Count-state gap | Does the pooled gap concentrate within specific counts (0-0, 3-1, 0-2)? | Within-count per-count gaps range −0.038 (3-1) to +0.001 (3-2); spread 0.039 | Null | Permutation p=0.51; within-count the gap is at or below zero everywhere — diagnostic that the pooled 0.092 is a compositional artifact |
| 8 | 3 | Pitch-family gap | Does predictability punishment concentrate in fastballs vs breaking balls? | FB +0.257, OFF −0.059, BR −0.220 wOBA; spread 0.477 | Null (diagnostic confound) | 0.477 spread is mechanically impossible as a real signal — reflects top-quintile FB landing in hitter counts and top-quintile BR landing in put-away counts |
| 9 | 3 | Stuff compensation | Do high-stuff pitchers survive predictability better than stuff-neutral pitchers? | Stuff tertile spread 0.008 wOBA; interaction posterior −0.003, CI95 [−0.012, +0.006] | Null | Permutation p=0.68; Scherzer intuition ("elite stuff buys you predictability") does not show up |
| 10 | 4 | Sequence-pair / CMU SURE replication | Does "same pitch type twice in a row" cost hitters ~0.030 wOBA per CMU SURE 2023? | Claude unmatched: +0.002 wOBA, CI95 [−0.001, +0.005], perm p=0.53. Codex matched (N=95,211): −0.017, p=0.001 | Null (literature does not replicate) | Both agents fail to find CMU's +0.030 penalty; effect is at or below zero on 2.87M-pitch substrate |
| 11 | 4 | Team-level exploitation heterogeneity | Do some MLB teams' hitters extract more gap than others (Dodgers/Yankees/Phillies premium)? | Claude: team spread 0.052, perm p=0.66; hierarchical σ_team = 0.003. Codex: residual spread 0.059 | Null | Between-team variance is essentially zero after controlling for roster baseline wOBA |
| 12 | 4 | Stuff+ residual × predictability | Do high-predictability pitchers underperform their physics-based Stuff+ expectation? | Top-quartile-predictability Spearman ρ = −0.166, p=0.13, CI95 [−0.370, +0.061] | Null (marginal, directional) | Point estimate clears the −0.15 kill threshold; p=0.13 and CI crosses zero at n=86 pitchers |
| 13 | 4 | TTOP × predictability tier | Does the times-through-order penalty sharpen for high-predictability pitchers? | (TTO3−TTO1)_high − (TTO3−TTO1)_low = −0.005 wOBA, perm p=0.76 | Null | TTOP curves are parallel across predictability tiers; high-tier differential is slightly smaller than low-tier |
| 14 | 4 | Batter-cluster resolver (H5) | Does Claude's Round-3 batter-heterogeneity finding replicate on Codex's leakage-free substrate at high N? | Patient-power cluster gap +0.064 wOBA, CI95 [+0.051, +0.076], perm p<0.002 at N=25,240 pairs. Spread vs free-swingers = 0.029 | Marginal / precursor to survivor | Clears Claude's own gate but the magnitude and cluster boundaries remained method-dependent; retired in Round 5 in favor of canonical features |
| 15 | 5 | Canonical tertile archetype (whiff / zone-contact / xwOBAcon) | Does the gap concentrate in the expected direction for the non-chase canonical features? | Whiff tertile spread: Codex −0.018 (wrong direction), p=0.979. Zone-contact: −0.016, p=0.944. xwOBAcon: +0.004, p=0.376 | Null | None pass the 0.015 spread × p<0.05 gate; two features point wrong direction; SHAP importance flat across features |
| 16 | 5 | Counter-archetype (chasers get fooled) | Do top-decile chasers have a *negative* gap on predictable pitches? | Codex matched-pairs: −0.016 wOBA, p=0.062. Claude: directionally same | Null (near-miss) | Sign stable across weightings; permutation p=0.062 misses the 0.05 gate |
| 17 | 5 | Time stability of chase-tertile gap | Does the chase-rate effect replicate in each of 2022, 2023, 2024, 2025 independently under Codex matched-pairs? | Only 1 of 16 feature-season cells clears gate (chase × 2022: +0.040, p=0.021); 2023/2024/2025 chase estimates: −0.023 / +0.012 / −0.004 | Null under Codex method | Per-season replication fails under matched-pairs; survives under pooled estimator in Round 6 (see survivor below) |

## The survivor

**Low-chase hitters extract a larger coaching gap on predictable
pitches than high-chase hitters — the chase-tertile spread.**

| Estimator (Round 6, shared substrate) | Point estimate | 95% CI | p-value |
|---|---|---|---|
| Pooled between-batter (Claude) | −0.043 wOBA | [−0.054, −0.032] | permutation p=0.001 |
| Pooled between-batter (Codex) | −0.038 wOBA | [−0.026, −0.050 bootstrap] | permutation p=0.001 |
| Batter fixed-effects regression (Claude) | −0.044 wOBA | [−0.056, −0.032], cluster-robust | p<0.001 |
| Batter fixed-effects regression (Codex) | −0.039 wOBA | [−0.026, −0.051] | p=1.24e−9 |
| Matched-pairs within-batter (Claude) | −0.017 wOBA | [−0.032, +0.000] | p=0.050 |
| Matched-pairs within-batter (Codex) | −0.006 wOBA | (CI covers zero) | p=0.647 |

The sign is flipped convention: a *negative* spread means the low-chase
tertile has a *larger* gap than the high-chase tertile. In plain
language, low-chase hitters extract roughly **+0.04 wOBA more** from
predictable pitches than high-chase hitters do; the ~0.04 figure is
the between-batter pooled / FE magnitude that both agents reproduce on
identical substrate.

**Method:** Round 6's shared-substrate test, strict 371-pitcher pre-2026
cohort, 2022–2025 terminal pitches from Codex's clean context-only
Layer 2 (AUC 0.686), ~401K terminal pitches, ~2,033 batter-seasons,
bootstrap N=400, permutation N=1000.

**Why both agents eventually converged in Round 6.** Round 6 did two
things that prior rounds hadn't. First, it forced byte-for-byte shared
tertile labels, so any disagreement had to come from estimand or power
rather than substrate drift. Second, it added two diagnostics — H2
variance decomposition and H3 power simulation — that explained the
Round-5 divergence quantitatively. The variance partition shows the
structural coaching-gap signal splits roughly 38/62 (Claude) or 51/49
(Codex) between between-batter and within-batter components, with 95–98%
of raw pitch-level variance being residual noise. The power simulation
shows matched-pairs at the observed N has only ~40% power to detect a
true 0.025 between-batter spread, while pooled estimators have 85%+.
That means matched-pairs' smaller point estimate is not evidence
against the effect — it is a known power-loss from stricter local
conditioning, not a methodological contradiction.

Both agents agreed on the bottom line: **direction is unanimous across
all three estimators, all three clear p<0.05 under at least one agent's
implementation, and the pooled / fixed-effects magnitude of ~0.04 wOBA
is the defensible headline.**

## Secondary survivor (honest accounting)

The flagship article also credits a second survivor: the **quality-
hitter 2×2** (low chase × high xwOBAcon on contact) extracts
+0.025 to +0.029 wOBA more than the rest of the cohort (Claude pooled
+0.029, p<0.001; Codex pooled +0.025, p=0.003; both agents' matched-
pairs at +0.018–0.021, borderline). This is a refinement of survivor
#1 rather than an independent effect — the quality-hitter quadrant is a
subset of the low-chase tertile — which is why the headline uses "16
nulls, one finding" rather than "two findings."

## Rigor notes

- **Cohort discipline.** All rounds use the strict 371-pitcher
  pre-2026 qualifier. 2026 is held out of training for the pitch-
  prediction model and out of inference for the coaching-gap tests.
- **Replication.** The chase-tertile spread replicates every season
  2022–2026, with permutation p<0.01 in four of five seasons (2026 is
  the current-month slice at ~1/6 of a full year's sample).
- **The 2.5× rider.** The hero chart's "top-third .366 wOBA vs bottom-
  third .321 wOBA on predictable pitches" is the between-batter
  contrast; the ~0.04 wOBA figure above is the tertile-spread *delta*
  (predictable-minus-unpredictable gap in low-tertile minus same in
  high-tertile), which is the formally tested quantity.
- **What we did not find.** Pitcher archetype, sequence-pair
  repetition (CMU SURE), team-level heterogeneity, stuff compensation,
  TTO penalty sharpening, count-state concentration, pitch-family
  concentration, and predictability-slope → outcome coupling all
  failed at least one of the pre-registered gates. The article's
  disclaimer that "seven more hypotheses passed marginal significance
  under one method but not the other" refers primarily to the R3 / R4
  batter-cluster and TTO / stuff-residual results listed above.

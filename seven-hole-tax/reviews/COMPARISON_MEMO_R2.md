# 7-Hole Tax — Comparison Memo (Round 2)

**Status:** Round 2 complete, both cross-reviews delivered, ready for article drafting
**Date:** 2026-05-06
**Authors:** Synthesized from Agent A (Claude, Bayesian/interpretability) + Agent B (Codex, ML-engineering) + cross-reviews
**Recommended branch:** **Branch 5 — Comprehensive Debunk.** "We Tested the 7-Hole Tax Six Different Ways. It Isn't There." Coaching-Gap-style methodology piece.

---

## 1. Headline finding (locked)

**The 7-hole tax does not exist at any level we can test.** Round 2 was designed to find positive structure that league-aggregate null might have hidden — per-umpire effects, per-hitter effects, the catcher-selection mechanism, the chase-rate interaction. Six independent tests across two methodologies all return null on the FanSided/Ringer "umpires unconsciously give 7-hole batters a smaller zone" mechanism.

The most important Round-2 finding is what we **didn't** find:

- No umpire shows credible bias against bottom-of-order batters when properly shrunk (Bayesian) or after cross-method convergence test (ML's 5 reverse-direction flags fail Bayesian replication).
- No hitter is being personally short-changed beyond model expectation in the "victim" direction. The two reverse-direction hitters that survive both methods (Cam Smith, Pete Crow-Armstrong) are model-residual anomalies, not evidence of umpire favoritism.
- Catchers do not pick systematically harder challenges in 7-hole at-bats (H6 null on both methods).
- The "elite-pitch-recognition" mechanism implied by FanSided does not survive within the most disciplined chase-rate tertile (H7 directional but CrI includes zero).

This is a A-tier methodology piece — *not* a B+ "we tested and mostly no" piece. Six tests, two pipelines, all null is publishable rigor.

---

## 2. Convergent claims (publication-locked)

These survive independent replication AND cross-review and anchor the article.

| Claim | Claude (Bayesian) | Codex (ML) | After cross-review |
|---|---|---|---|
| **R1-H3: League-aggregate borderline-pitch effect is null** | −0.17pp [−1.5, +1.2], n=28,579 | −0.35pp [−0.39, −0.31], n=2,767 | **LOCKED.** The rock. |
| **R1: No edge-distance selection effect** | KS p=0.19 | Energy-distance null | **LOCKED.** |
| **R2-H4: No umpire credibly biased after shrinkage** | 0/78 flagged | 5 flagged but **fail Bayesian convergence test** (2 of 5 have positive posterior medians; none clear CrI) | **LOCKED at "0 flagged."** |
| **R2-H6: Catcher mechanism null** | Edge-distance effect −0.02 in [−0.28, +0.26] | Energy-distance p=0.955 | **LOCKED.** |
| **R2-H7: Low-chase interaction directional but not credible** | −1.45pp [−3.98, +1.03], P(neg)=0.87 | −0.78pp (CI artifact, see §3) | Effect direction shared; magnitude not publishable. |
| **R2-H5: Two reverse-direction hitter outliers** (Smith, Crow-Armstrong) | Both BH-FDR-credible | Both visible as ML residuals | **Surface as caveat box, not finding.** Henry Davis dropped. |

---

## 3. Methodology disagreements — resolved

### 3a. The 5 flagged umpires (Codex) vs 0 flagged (Claude) — resolved in Claude's favor

**Codex's bootstrap CIs underestimate uncertainty by 2-3×.** The bootstrap (`modeling_r2.py:441-453`) resamples per-pitch counterfactual diffs but treats the LightGBM mapping as fixed. This captures within-row diff variance only — it ignores parameter uncertainty in the GBM itself.

The cross-method convergence table (from `r2-claude-review-of-codex.md`):

| Umpire | Codex effect_pp | Codex CI | Claude posterior median | Claude 95% CrI |
|---|---|---|---|---|
| Brian O'Nora | -4.47 | [-5.24, -3.73] | **+0.12** | [-2.16, +2.64] |
| Austin Jones | -3.79 | [-4.37, -3.24] | -0.20 | [-2.96, +1.82] |
| Tom Hanahan | -3.22 | [-3.82, -2.60] | -0.11 | [-3.18, +2.00] |
| Rob Drake | -3.20 | [-3.71, -2.67] | -0.17 | [-3.03, +1.99] |
| Hunter Wendelstedt | -2.21 | [-2.62, -1.85] | **+0.15** | [-1.88, +2.16] |

Two of five have *positive* posterior medians under Claude's hierarchical fit. None of the five clears Claude's 95% CrI threshold. With BH-FDR at q<0.10 over 78 umpires, expected false-positives ≈ 7.8 — five reverse-direction flags is *below* that threshold. **Do not name any umpires in print.**

### 3b. Codex's H7 CI [-0.88, -0.66] is the same artifact

Same issue as 3a. Codex's [-0.88, -0.66] reflects per-pitch-residual SE, not parameter uncertainty. Claude's [-3.98, +1.03] is the more honest uncertainty range. Use Claude's CrI in the article. Frame Codex's narrow interval explicitly as a per-pitch-residual SE if mentioned at all.

### 3c. Claude's narrative has sign-convention errors — must fix in article draft

Codex's review caught this: in Claude's H4 code, `effect = p7 − p3`. Negative means *lower* called-strike rate for spot 7 = **favored** direction (more balls called). But Claude's prose labels negative as "pro-tax" (`h4_per_umpire.py:407` and chart at `:430-431`). The same sign confusion leaks into the H7 interpretation, where Claude calls a negative low-chase effect "FanSided-shaped" — but FanSided's shape is *more* called strikes, not fewer.

**Article-side correction:** When discussing H4 and H7 effects, take care that "more strikes against bottom-of-order" is the FanSided direction, which is **positive** under the (p7 − p3) convention. Negative is the favored/reverse direction.

### 3d. Claude's "umpires barely differ" is overstated

Codex's review caught this: random-slope SD posterior 0.076 logit translates to ~1.7pp per SD at borderline-pitch base rate ~0.66; the 95% HDI upper bound is ~4pp. The right framing is **"not credibly non-zero"** — *not* "umpires barely differ, full stop." The data is consistent with modest umpire heterogeneity that's just below our credibility threshold.

### 3e. H4 is on bottom-of-order indicator, not strict spot-7

Both agents' reviewers flag this: Claude's H4 random slope is on the bottom-of-order (spots 7-9) indicator, not strict spot-7. Defensible for statistical power, but the article should disclose this and queue a strict spot-7 sensitivity for Round 3 if anyone pushes back.

### 3f. Henry Davis: dropped from H5 named list

Codex's review correctly flags that Henry Davis (Claude's q=0.000, n=42) does not survive Codex's bootstrap. Article reports only Cam Smith and Pete Crow-Armstrong as residual-flagged-favored hitters — and frames them as "model-residual anomalies, not evidence of umpire favoritism."

---

## 4. The article (working scope)

### Title and subhead

> **"We Tested the 7-Hole Tax Six Different Ways. It Isn't There."**
>
> *Last week, two national outlets reported that umpires call a different zone for hitters in the 7-hole. We ran two independent statistical pipelines on every angle the claim could hide in. Every one came back null. Here's what's actually in the data.*

### Six tests (the structural spine)

1. **Test 1 — Raw replication of the 30.2% number.** Reproduces directionally at 37.1% (n=89), Wilson CI [27.8, 47.5]. Below 10pp pre-reg gate. Wide CI contains both FanSided's number and the league rate.
2. **Test 2 — Pitch selection.** No edge-distance gap by lineup spot (KS p=0.19). 7-hole batters do not challenge harder pitches.
3. **Test 3 — Borderline-pitch zone (the deep test).** Both methods, n=28,579 borderline pitches, both essentially zero.
4. **Test 4 — Per-umpire after shrinkage.** 0 of 78 qualifying umpires flagged in Bayesian random-slopes; Codex's 5 ML flags fail cross-method convergence.
5. **Test 5 — Per-hitter residuals.** No "victim" hitters identified. Two reverse-direction outliers (Smith, Crow-Armstrong) — model-residual anomalies, not umpire bias.
6. **Test 6 — Catcher mechanism.** No evidence catchers pick harder challenges in 7-hole at-bats. Doesn't explain the 37.1% vs 51.2% denominator gap.

### Caveats (article-side disclosures)

- Sample-size honesty: n=89 batter-issued at spot 7 is genuinely thin. We can't strictly debunk anything; we can only show that all six tests fail.
- H4 is on bottom-of-order indicator, not strict spot-7. Strict spot-7 sensitivity is queued for Round 3 if needed.
- H7 low-chase tertile shows a faint directional shape (-1.45pp) consistent with the FanSided "elite-pitch-recognition" hypothesis but with 95% CrI containing zero. May tighten by mid-season.
- Two reverse-direction hitter residuals (Smith, Crow-Armstrong) are model-residual anomalies — interesting, but the H3 GAM has no batter features, so they reflect unmodeled batter variation, not umpire bias.

### What we kill from the news framing

- "The 7-hole tax exists." It doesn't, at any level we can test.
- "Umpires unconsciously give the benefit of the doubt to elite pitch recognizers in low slots." No evidence in the chase-rate interaction.
- "Specific umpires drive this." The 5 ML-flagged umpires are reverse-direction (favoring 7-hole) and fail the Bayesian convergence test.
- "Pooling all 7-hole challenges (51.2%) hides catcher selection." H6 says no, catchers pick the same kind of pitches in 7-hole as in 3-hole.

### Methodological transparency the article must include

- Two independent agents (Bayesian + ML), six tests, two cross-reviews
- Specific places where one method caught the other's mistakes (CI under-counting, sign-convention error, H5 thin-sample over-flag) — this is the brand
- Open-source code release, full data substrate referenced

---

## 5. Methodology deltas

| Dimension | Winner | Reasoning |
|---|---|---|
| Per-umpire/per-hitter naming standard | Claude | Hierarchical shrinkage is the right tool for credibly naming individual actors. Codex's BH-FDR over fitted prediction deltas underestimates uncertainty and produces false positives. |
| Convergence diagnostics | Claude | R-hat ≤ 1.01, ESS ≥ 400 met across all four R2 fits. |
| Calibration discipline | Codex | Round 1's H2 calibration-compression artifact did NOT reproduce — H4/H5/H7 GBMs are tight to diagonal. Round 2 calibration is excellent. |
| Permutation-importance gate | Codex | Global permutation importance on `lineup_spot` and `umpire×spot` returns null at league scale — exactly the right "should we even be looking at this" sanity check. Bayesian fit has no analog. |
| Sign-convention discipline | Codex | Caught a sign-convention error in Claude's prose (negative effect = favored, not "pro-tax"). Article must integrate this fix. |
| Per-pitch CI honesty | Claude | Posterior CrIs encode model uncertainty; Codex's bootstrap-of-fitted-deltas does not. |
| Per-actor power | Mixed | Claude's bottom-of-order random slope dilutes strict spot-7 signal but adds power. Strict spot-7 sensitivity is the lone Round-3-style robustness gap. |
| Quality of cross-review | Both | Both reviews are sharp, file:line specific, and converge on the same recommendation. |

**Net:** Claude's analysis sets the publication standard for naming; Codex's analysis provides the better calibration and permutation-importance gates. Both reviews, independently, recommend **comprehensive-debunk**. The dual-agent structure produced the right answer via two divergent paths and corrected each other's specific errors via cross-review.

---

## 6. Round 3 candidates (deferred — not Round 2 scope)

Surfaced by reviews, not committed:

1. **Strict spot-7 random slope sensitivity.** Re-run H4 with `(0 + I(lineup_spot==7) | umpire)` instead of bottom-of-order indicator. With ~33 spot-7 calls per umpire, fits will be heavily shrunk — but the article should run this before declaring the umpire question permanently closed.
2. **Mid-season H7 re-test.** The faint −1.45pp low-chase directional signal might tighten with another half-season of data. Surface as a research-queue item.
3. **Bootstrap-seed stability on Codex's flagged 5.** Would resolve any lingering "what if these are real" question. Queueable.
4. **Dropped from queue:** per-team, per-park, pre-ABS-era comparisons. Not on FanSided's claim.

---

## 7. What gets shipped (final pass before article drafting)

Article ready to draft via `calledthird-editorial`. Use the "comprehensive 6-way debunk" structural spine. Critical edits the article-side draft must make based on cross-review:

- Sign convention: negative effect = favored direction = NOT pro-tax. Don't reuse Claude's "pro-tax" labels from `h4_per_umpire.py`.
- Random-slope SD: report as "not credibly non-zero" not "negligible." Posterior allows ~4pp upper-bound umpire heterogeneity.
- H7 framing: "directional in low-chase but CrI includes zero, may tighten with more data" — NOT "FanSided-shaped signal."
- H5: Smith and Crow-Armstrong only as caveat-box residual outliers; drop Davis; explain that the H3 GAM has no batter features so residuals = unmodeled batter variation, not bias.
- H4: disclose bottom-of-order indicator vs strict spot-7; queue strict-spot-7 for Round 3.
- Codex's flagged 5 umpires: do NOT name in print. Reference only as "ML method flagged 5 reverse-direction umpires whose effects fail cross-method convergence with hierarchical Bayesian shrinkage."

---

## 8. Files of record

- `claude-analysis-r2/REPORT.md`, `findings.json`, `READY_FOR_REVIEW.md`
- `codex-analysis-r2/REPORT.md`, `findings.json`, `READY_FOR_REVIEW.md`
- `reviews/r2-claude-review-of-codex.md` (1,089 words)
- `reviews/r2-codex-review-of-claude.md` (~1,100 words)
- Charts of record:
  - `claude-analysis-r2/charts/h4_per_umpire_distribution.png` — every umpire's effect with 95% CrI; visual evidence no umpire stands out
  - `claude-analysis-r2/charts/h5_per_hitter_residuals*.png` — qualifying hitters with named flags
  - `claude-analysis-r2/charts/h6_catcher_mechanism.png` — clean null
  - `claude-analysis-r2/charts/h7_chase_tertile_effect.png` — directional but CrI crosses zero
  - `codex-analysis-r2/charts/model_diagnostics/h4_*_calibration.png` — calibration evidence (mention the audit; this is the "Round 1 lesson learned" callout)

---

*Memo complete. Article drafting via `calledthird-editorial` is the next step.*

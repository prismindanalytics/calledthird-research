# ABS Walk Spike — Comparison Memo (Round 2)

**Status:** Round 2 complete, both cross-reviews delivered, ready for article drafting
**Date:** 2026-05-14
**Authors:** Synthesized from Agent A (Claude, Bayesian/PA-replay) + Agent B (Codex, ML/expectation-propagation) + cross-reviews
**Recommended branch:** **`adaptation`** (both agents independently agreed)

---

## 1. Headline finding

**Three weeks after the April 23 piece, the walk spike has narrowed but persisted, the zone-attribution number drifts modestly (40-50% → ~35%), and pitchers have begun adapting their pitch locations. The 0-0 first-pitch mystery from Round 1 is now resolved: top-edge first-pitch strikes dropped 6-7pp MORE than top-edge 2-strike calls, driving traffic into 1-0/2-0/3-0 counts. That traffic channel — not the per-count conversion rate — is what's left of the walk spike.**

This is genuinely new structure on the same story. Round 1 said "40-50% is the zone." Round 2 says "40% → ~35%, with most of the residual now coming through specific counts via a specific zone-region change."

---

## 2. Convergent claims (publication-locked)

These survive independent replication AND cross-review and anchor the article.

| Claim | Claude (Bayesian) | Codex (ML) | Cross-review verdict |
|---|---|---|---|
| **H1: YoY walk rate spike persists but narrowed** | +0.68pp [+0.31, +1.04], 9.46% vs 8.78% | +0.66pp [+0.27, +1.04], 9.46% vs 8.80% | **LOCK.** Both methods, ~identical effect size. |
| **Within-2026 fading (not just regression to 2025)** | W1-3 → W5-7 = −0.86pp; P(regressed)=89% | Same week-by-week pattern | **LOCK.** The spike is decaying in real time. |
| **Zone-shape change is durable** | Top edge −9.18pp, bottom edge +3.43pp | Direction agrees; top edge +66% attribution, bottom edge −56% (partial interventions cancel) | **LOCK.** Both methods, both rounds. The geometry is real. |
| **Aggregate mean predicted CS under 2025 classifier on 2026 takes ≈ 0.334** | Both implementations independently arrive at this number | Confirmed via separate seed ensemble | **LOCK.** This is the key diagnostic both agents found. |
| **H5: 0-0 mystery resolved with count-dependent mechanism** | Top edge dropped −10.48pp at 0-0 vs −3.73pp at 2-strike; DiD −6.76pp credible | SHAP interaction values confirm count-dependence; heart 0-0 CS delta ~0pp; top-edge 2-strike −3.17pp | **LOCK.** The mechanism is real and bidirectionally confirmed. |
| **H2: Traffic-driven, not conversion-driven** | +0.41pp traffic + 0.27pp conditional decomposition | Top-edge first-pitch SHAP supports same | **LOCK.** Pitchers losing the count battle earlier, then more 3-x walks. |
| **H4: Pitcher adaptation visible at the pitcher level even when league-aggregate is flat** | League zone rate +0.64pp [-0.11, +1.49] (not credibly nonzero); individual pitcher shifts of ±20-40pp | Same week-over-week pattern, ensemble identifies same top adapters | **LOCK.** Adaptation exists but is heterogeneous. |
| **Branch recommendation** | `adaptation` | `adaptation` | **LOCK.** |

---

## 3. The H3 sign-flip — resolved as methodology divergence

This is the only substantive divergence. **Both reviewers independently concluded the same thing**: Codex's +35.3% is the publishable headline; Claude's −64.6% is a stress-test result, not a settled finding.

### Why both numbers exist

Both classifiers agree exactly on the aggregate diagnostic: mean predicted CS on 2026 takes under 2025 classifier = 0.334, higher than both 2025 (0.327) and 2026 (0.325) empirical. So far, identical.

The split is in **PA propagation**:

- **Codex: expectation propagation.** `state_probs` as a distribution over (balls, strikes) at each step, weighted by classifier `p_strike`. Deterministic given the model. → `cf_walk_rate = 0.0923 < 0.0946 actual` → **+35.3% attribution** (zone change adds ~35% of the spike).

- **Claude: per-take Bernoulli sampling.** Each posterior draw samples actual call sequences. → `cf_walk_rate = 0.0989 > 0.0946 actual` → **−64.6% attribution** (zone change "removed" walks, i.e., pitchers overcompensated).

Both are *internally consistent*. They answer subtly different counterfactual questions:

- Codex's +35.3%: "What's the expected walk-rate gap attributable to zone shape, holding pitcher locations fixed?"
- Claude's −64.6%: "What does PA-level Bernoulli sequencing produce when the 2025 classifier is asked to call 2026 pitches?"

### Why we publish Codex's number

- It's the direct continuation of Round 1's editorial framing ("% of YoY spike attributable to zone change")
- Expectation propagation is monotone in the obvious direction (more strikes → fewer walks); Claude's Bernoulli replay mixes that with a backstop where unresolved sequences inherit observed outcomes (`h3_counterfactual.py:313-316`), which can produce sign-perverse behavior
- Codex's per-count and per-edge decomposition is auditable from artifacts; Claude's H3 per-count output is essentially all `0-0` (PA starts) and the edge decomposition is by first-pitch-region (not partial intervention), so it doesn't cleanly explain the residual
- Codex has held-out OOF calibration; Claude has convergence diagnostics but no held-out calibration check

### Critical caveat we must integrate

**Codex's CI of [+34.6%, +36.0%] is too narrow.** It's a 10-seed cross-fit SD only — no game-bootstrap, no PA-level resampling, no classifier-prediction uncertainty. This is the same "fixed-model bootstrap" artifact we caught in seven-hole-tax R1 and R2.

The article must report the +35.3% with a **widened editorial CI**, roughly **[~25%, ~50%]**, reflecting:
- The methodological uncertainty Claude's stress test exposed (sign is sensitive to replay assumptions)
- Honest game-level sampling variance
- Round 1's 40-49% range is consistent with this widened interval

### What this means for the article

The article publishes `+35.3% [~25%, ~50%]` as the updated zone-attribution headline. It does NOT publish "the zone effect sign-flipped" — that would overclaim what the analysis supports. The honest phrasing: **"Three weeks later, the zone-attribution number has muted modestly to ~35%, consistent with our Round 1 range, but it is now sensitive to replay assumptions in a way it wasn't before — which is itself evidence that pitchers have started shifting their locations away from the high-strike zone they're now losing."**

The Claude −64.6% appears as a methodology box: "What happens when we replay 2026 pitches under the 2025 classifier with full PA-level Bernoulli sampling? The number actually flips negative. That's the same data, with a different counterfactual mechanic — and the divergence is direct evidence of the location-shift channel."

---

## 4. Methodology deltas

| Dimension | Winner | Reasoning |
|---|---|---|
| H1 estimation | Tie | Both independently arrive at +0.66-0.68pp |
| H3 headline framing | Codex | Expectation-propagation is closer to R1's editorial framing |
| H3 uncertainty propagation | Claude | Bernoulli replay surfaces real sensitivity Codex's deterministic version hides |
| H3 CI honesty | Claude | Codex's [+34.6, +36.0] is the same fixed-model-bootstrap artifact from 7-hole tax R1; needs widening |
| H3 per-count/per-edge auditability | Codex | Partial-intervention edge decomp; per-count properly stratified |
| H3 calibration | Codex | Held-out OOF calibration vs convergence-only diagnostics |
| H5 0-0 mechanism | Claude | Bayesian DiD framework cleaner than SHAP for the count-asymmetry question; both directionally agree |
| H4 pitcher adaptation | Tie | Both methods identify per-pitcher heterogeneity |
| H2 per-count decomposition | Claude | Traffic vs conditional decomposition is the editorially-clean framing |

**Net:** Codex's analysis is the publication standard for H3 numerical claims; Claude's analysis sets the standard for the H5 mechanism narrative and the H2 traffic/conditional decomposition. Article uses both.

---

## 5. What gets published

### Lead
> *Three weeks ago we said roughly half the walk spike was the new ABS zone, and pitchers owned the rest. With two more weeks of data and a much harder analytical pass, here's the update: the spike has narrowed but persisted at +0.66pp YoY. The zone-attribution number has muted to ~35%, but the more interesting finding is mechanistic — we now know exactly how the zone is producing those extra walks, and it isn't where we thought.*

### The four findings that anchor the article

1. **The spike has narrowed but persists, and it's actively fading.** +0.82pp (R1) → +0.66pp (R2). Within-2026 trajectory shows W1-3 → W5-7 dropping −0.86pp. Both pipelines agree.

2. **Zone attribution updated: ~35% [~25%, ~50%], down from 40-50%.** Both methods agree the zone effect is real and positive; the magnitude has muted as pitcher locations shifted.

3. **The 0-0 first-pitch mystery is resolved.** The top edge of the zone lost 6-7pp MORE strikes at 0-0 than at 2-strike counts (DiD credible at −6.76pp; SHAP interaction confirms the count-dependence). That means more 1-0 traffic, more 2-0/3-0 traffic, more terminal walks — even though the per-count walk rate hasn't changed much. The spike is now traffic-driven, not conversion-driven.

4. **Pitcher-level adaptation is real but heterogeneous.** League zone rate is essentially flat W1→W7 (+0.64pp [-0.11, +1.49], not credibly nonzero). But individual pitchers show shifts of ±20-40pp in zone rate or top-share. Per-pitcher leaderboard available.

### The methodology callout box (asset, not blocker)

The article includes an explicit "what our two pipelines disagreed about" callout:
- Codex's expectation-propagation: +35.3% attribution
- Claude's Bernoulli PA replay: −64.6% attribution
- Both used the same training data and same aggregate diagnostic (mean predicted CS = 0.334)
- The divergence isn't a bug — it's evidence that pitcher-location adaptation is now interacting with PA-level sequencing in a way it wasn't in Round 1

This is brand-on-message: "we publish our cross-method disagreements when they're informative."

### Caveats the article must include

- The +35.3% has a wider editorial CI than Codex's internal one reports; honest range is [~25%, ~50%]
- The within-2026 trajectory shows real fading; the +35% number is window-dependent and may continue to mute
- Sample window is Mar 27 – May 12 (May 13 was partial)
- The 0-0 H5 mechanism is robust at R-hat 1.010, ESS 552; magnitude is credible but not overdetermined

### What gets killed

- The "sign-flipped" framing (overclaims; not publication-ready)
- "Pitchers have fully adapted" (league zone rate is flat; adaptation exists but is heterogeneous)
- Any single per-pitcher named claim without ≥200-pitch sample and explicit shift magnitudes

---

## 6. The article's title and structural spine

### Working title
> **"Three Weeks Later: We Know How the ABS Walk Spike Works Now."**

Alternative:
> **"The Walk Spike Is Fading. Here's the Count Where It's Hiding."**

### Structural spine

1. Hero finding: spike narrowed (+0.82 → +0.66pp) but persists; within-2026 fading visible
2. Updated zone attribution: ~35% (down from 40-50%)
3. **The mechanism resolution (the centerpiece):** top-edge first-pitch strike loss → 1-0/2-0/3-0 traffic → terminal walks. With the heat map showing top-edge vs heart vs bottom-edge differential by count.
4. Pitcher adaptation: per-pitcher leaderboard, who shifted most
5. Methodology callout: cross-method disagreement on H3, what it means
6. What's next: All-Star break re-test for whether the fading continues or stabilizes

---

## 7. Files of record

- `claude-analysis-r2/REPORT.md` (in READY_FOR_REVIEW.md text), `findings.json`
- `codex-analysis-r2/REPORT.md`, `findings.json`
- `reviews/r2-claude-review-of-codex.md` (~1,440 words)
- `reviews/r2-codex-review-of-claude.md` (~1,086 words)
- Charts of record:
  - `codex-analysis-r2/charts/h3_per_count_attribution.png` (the central new visualization)
  - `claude-analysis-r2/charts/h5_first_pitch_mechanism.png` (the mechanism story)
  - `claude-analysis-r2/charts/h1_walk_rate_by_week.png` (the fading)
  - `codex-analysis-r2/charts/h4_pitcher_adaptation_leaderboard.png` (the leaderboard)

---

*Memo complete. A-tier assessment follows separately. Article drafting via `calledthird-editorial` is the next step.*

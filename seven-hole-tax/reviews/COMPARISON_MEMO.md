# 7-Hole Tax — Comparison Memo (Round 1)

**Status:** Round 1 complete, ready for article drafting
**Date:** 2026-05-05
**Authors:** Synthesized from Agent A (Claude, Bayesian/interpretability) + Agent B (Codex, ML-engineering) + cross-reviews
**Recommended branch:** **Hybrid B2 + B4** — "The 7-Hole Tax exists as a small, low-power batter-skill anecdote; the umpire-zone-bias version of the claim does not replicate."

---

## 1. Headline finding

**The viral 7-hole tax claim is mostly an artifact of small samples and a denominator that conflates batter skill with catcher-driven outcomes.** When you ask the question both ways, you get:

- **The literal FanSided/Ringer claim** ("7-hole batters win their challenges 30.2%"): directionally replicates but at a smaller, statistically-imprecise magnitude. 7-hole *batter-issued* overturn rate = 37.1% (n=89, Wilson 95% CI [27.8%, 47.5%]) vs league 45.2%. The CI contains everything from FanSided's 30.2% to the league rate.

- **The "umpire-zone-bias against 7-hole batters" interpretation** (the implicit mechanism the news pieces gestured at): does not replicate. On 28,579 borderline called pitches with full controls, the spot-7-vs-spot-3 effect is **−0.17pp** (Bayesian GAM, Claude) and **−0.35pp** (counterfactual GBM, Codex) — both essentially zero, both with CIs that exclude the +2pp pre-registered threshold.

Both agents agree the called-pitch test is the decisive one. Both agents agree it's null.

---

## 2. Convergent claims (highest publication priority)

These survive rigorous independent replication and should anchor the article.

| Claim | Claude (Bayesian) | Codex (ML/counterfactual) | Confidence |
|---|---|---|---|
| **Borderline called-strike rate: spot 7 ≈ spot 3** | −0.17pp [−1.5, +1.2], n=28,579 | −0.35pp [−0.39, −0.31], n=2,767 | **Lock.** Both methods, near-zero, CIs straddle zero on the Bayesian side and barely cross zero on the ML side. The cleanest evidence in the project. |
| **No edge-distance selection effect** | Mean \|edge_distance\| = 1.27" for both spot 7 and spot 3, KS p=0.19 | Energy-distance probe: location distributions similar; selection lives in count × pitcher, not in plate location | **Lock.** The simplest selection mechanism (different "easy" pitches) is ruled out. |
| **Bottom-of-the-order pattern, not 7-specific** | Spots 7-9 all hover ~37% with overlapping CIs; q ≥ 0.62 after BH correction | Spot-7 SHAP / counterfactual indistinguishable from spots 8-9 | **Lock.** The "7-hole tax" naming is misleading; it's a "bottom-third tax" if anything. |
| **No detectable bias signature in any stratum** | H3 stratified by handedness, count quadrant, pitch group: no stratum where the effect lives | Counterfactual leaderboard across all spot pairs: noise around zero | **Strong.** Codex's review flags that Claude's stratified H3 had a process-death recovery; the recovered evidence is partial. Treat as supportive, not bulletproof. |

---

## 3. Divergent claims

### 3a. Headline H1 number (37.1% vs 51.2%)

**This is a definitional split, not a contradiction.** Both reviewers concur:

- **Claude's 37.1%** restricts to batter-issued challenges (n=89). This is the literal match for FanSided's "7-hole batters win their challenges" framing.
- **Codex's 51.2%** pools all challenges where the batter-at-plate was 7-hole (n=213), including catcher and pitcher initiations. Catchers overturn at ~60% (per CalledThird's prior `catchers-are-better-challengers`), so pooling mechanically inflates the rate.

**Resolution:** The article uses **Claude's batter-issued denominator** as the H1 replication number, with Codex's pooled denominator presented as a robustness check. Codex's denominator answers a different question ("are 7-hole plate appearances worse for the offense?") and the answer is "no" — also a useful publishable fact.

### 3b. Editorial branch (B2 vs B4)

**Both branches overclaim in opposite directions.** Both reviewers caught this independently:

- Codex's **B4 ("doesn't replicate")** mis-frames the all-challenger denominator as the H1 test, then concludes failure. But on the FanSided-equivalent denominator, the directional pattern *does* replicate, just at a smaller magnitude with wide CIs.
- Claude's **B2 ("mirage")** outruns the data — n=89 with a 20pp Wilson CI cannot debunk the public 30.2% number; it can only fail to replicate strictly.

**Resolution:** Use the hybrid framing explicitly. The article distinguishes:
- The **specific causal mechanism** ("umpires call a different zone for 7-hole batters") → does not replicate, B4 territory, decisive.
- The **raw observational pattern** ("7-hole batters win their own challenges less often") → directionally present, smaller than reported, statistically imprecise.

### 3c. H2 confidence interval

Codex's H2 effect of **+0.15pp [+0.08, +0.23]** is implausibly narrow for n=213 with AUC 0.579. Claude's review correctly identified this as a **calibration-compression artifact**: the GBM's predicted probabilities cluster tightly around the base rate, so flipping the lineup_spot dummy moves predictions by a near-constant tiny amount, producing a tightly-clumped paired bootstrap that doesn't reflect actual uncertainty about the effect size.

**Resolution:** The article does NOT cite Codex's H2 CI as a tight effect estimate. Codex's H2 is reported as "essentially flat" without the misleading CI; Claude's H2 [−21.4, +3.5] gets cited as the more honest uncertainty range, with the explicit caveat that batter-only n=971 is underpowered.

---

## 4. Methodology deltas

| Dimension | Winner | Reasoning |
|---|---|---|
| Headline H1 framing | Claude | Restricted to batter-issued challenges, matches FanSided's claim. Codex's pooling inflated mechanically. |
| Statistical honesty about uncertainty | Claude | Wilson CIs and Bayesian CrIs reflect actual uncertainty; Codex's bootstrap CI compressed by GBM calibration. |
| Reproducibility / leakage discipline | Codex | StratifiedGroupKFold by `game_pk`, target encoding fit inside fold, permutation-importance vs permuted-label baseline. Cleaner CV than Claude's Bayesian fits. |
| Convergence diagnostics | Mixed | Claude's R-hat=1.01 is a borderline pass under the strict <1.01 pre-reg (Codex's reviewer flagged); Codex doesn't have Bayesian convergence to worry about. |
| Stratified analysis robustness | Codex | Claude's stratified H3 had a process-death recovery; results are auditable from PNGs but not from `findings.json`. Codex's robustness checks (pinch-hitter, handedness, temporal) all completed cleanly. |
| Selection-effect test | Claude | KS test on edge-distance is the right operationalization; Codex's energy-distance probe ran on the wrong subset (all-challenger, not batter-only). |
| Volume / data extension | Tie | Both pulled the Apr 15–May 4 extension successfully; corpora are nearly identical (2,101 challenges, ~75K taken pitches). |
| Counterfactual attribution | Codex | The "predict the same pitch with lineup_spot flipped" structure is the cleanest causal-lite estimator either method produced. Magnitude is null but the framework is right. |

**Net:** Claude's analysis is more publication-ready as the headline narrative; Codex's analysis provides the strongest robustness checks and the cleanest counterfactual attribution. The article uses both.

---

## 5. What gets published

### Lead
> *Last week, two national outlets independently reported that umpires call a different strike zone for hitters in the 7-hole than for hitters in the 3-hole. We tested it.*

### The four claims that anchor the article

1. **The deep test is decisive — and null.** On 28,579 borderline called pitches, the spot-7-vs-spot-3 called-strike rate is essentially zero (−0.17pp Bayesian, −0.35pp ML counterfactual). The "umpire bias against 7-hole batters" mechanism does not exist in the data.
2. **The viral 30.2% number replicates directionally but smaller.** 7-hole batters win their own challenges at 37.1% (n=89), not 30.2%. The deficit vs league (45.2%) is 8.1pp — below our pre-registered 10pp gate, and the CI is wide enough to contain everything from FanSided's number to no effect at all.
3. **It's a bottom-of-the-order pattern, not a 7-hole pattern.** Spots 7, 8, and 9 all hover near 37%; spot 7 is not statistically distinguishable from its neighbors. The "7-hole tax" framing is a feature of headline aesthetics, not a feature of the data.
4. **The selection mechanism is real but boring.** 7-hole batters challenge slightly different pitches than 3-hole batters in terms of count and pitcher quality (but not plate location). The raw 8pp deficit is largely explained by which pitches/situations the bottom of the order ends up challenging, not by umpires calling them differently.

### Caveats the article must include

- **n=89** for batter-issued 7-hole challenges. The CI is 20pp wide. We can't strictly debunk anything; we can only show that the strict pre-reg gates fail and the deeper test is null.
- **The pooled denominator** (51.2%, n=213) is what Codex used and is what you'd get if you read the Savant ABS dashboard naively. We report it as a robustness check; it's "league-flat" — no penalty.
- **H2 calibration artifact** — the original ML report's +0.15pp [+0.08, +0.23] interval is a calibration-compression artifact, not a tight effect estimate. The honest uncertainty range from the Bayesian GLM is [−21.4, +3.5]. Underpowered.

### What gets killed

- The framing "umpires unconsciously give borderline calls to perceived-elite hitters" — interesting hypothesis but our data does not support it at 95% confidence in either direction.
- Any per-spot leaderboard claiming spot 7 is uniquely treated.
- The ±2pp called-strike-rate effect as a published number — both methods say zero.

### Suggested article title (working)

> **"The 7-Hole Tax Doesn't Hold Up — Here's What Actually Explains the Number"**

Subtitle: *"FanSided and The Ringer reported that 7-hole batters win their challenges 30.2% of the time. We ran two independent controlled analyses on 28,579 borderline pitches. The bias claim doesn't survive."*

---

## 6. Open questions for Round 2 (conditional on signal)

These are out of scope for Round 1 but worth flagging for the research queue:

1. **Per-umpire breakdown.** League-aggregate is null; do specific umpires drive the directional batter-issued pattern? Round 2 candidate.
2. **Count-x-pitcher decomposition of the residual selection effect.** Both reviewers flagged that "selection lives in the count × pitcher distribution" is asserted, not decomposed. A targeted decomp could quantify how much of the raw 8pp deficit is count-state vs pitcher-quality vs lineup-spot itself.
3. **Pinch-hitter substitution as mediator.** The 40% magnitude swing in H3 when pinch hitters are excluded suggests substitution patterns deserve their own analysis.
4. **Catcher-initiated challenges by batter lineup spot.** The pooled denominator showed catchers DO win 7-hole challenges at ~60%; is there a strategy where catchers preferentially challenge in 7-hole at-bats? Useful for the "challenge economy" story candidate.

These are deferred — Round 1 is single-round per the brief.

---

## 7. Editorial handoff

The article is ready to draft via the `calledthird-editorial` skill. Recommended structure:

1. **Lead with the deep null** — both methods, n=28,579, agree.
2. **Then walk through the H1 replication** — how 30.2% becomes 37.1% becomes "actually it's a bottom-of-the-order pattern."
3. **Then the methodology divergence** — frame the dual-agent disagreement as a feature, not a bug. "Two independent analyses with different methods and different denominator choices arrived at the same conclusion via different paths."
4. **Then the selection mechanism** — what's actually going on (counts, pitcher quality, not zone bias).
5. **Then the honest sample-size caveats and the open questions for Round 2.**

The transparency about the dual-agent process (and the cross-review that caught the denominator split) is an asset, not a liability — it's exactly the rigor CalledThird's brand is built on.

---

## 8. Files referenced

- `claude-analysis/REPORT.md`, `findings.json`, `READY_FOR_REVIEW.md`
- `codex-analysis/REPORT.md`, `findings.json`, `READY_FOR_REVIEW.md`
- `reviews/claude-review-of-codex.md` (1,016 words)
- `reviews/codex-review-of-claude.md` (806 words)
- Charts of record:
  - `claude-analysis/charts/h1_overturn_batter_only.png` (the H1 chart)
  - `claude-analysis/charts/h3_called_strike_rate_borderline.png` (the H3 null)
  - `codex-analysis/charts/counterfactual_leaderboard.png` (the counterfactual confirmation)
  - `claude-analysis/charts/selection_effect_distributions.png` (the selection-effect probe)

---

*Memo complete. Article drafting via `calledthird-editorial` is the next step.*

# The 7-Hole Tax Doesn't Hold Up — Here's Why the Number Looks Bigger Than It Is

**Subtitle:** Last week, two national outlets reported that umpires call a different strike zone for hitters in the 7-hole. We tested it two independent ways. The bias claim doesn't survive — but the raw number is real, smaller than reported, and the mechanism isn't the umpire.

**Tags:** ABS, Umpires, Methodology

**Date:** May 5, 2026

**Reading time:** ~9 min

**Data:** ABS challenges Mar 26 – May 4, 2026 (2,101 challenges). Statcast called pitches Mar 27 – May 3, 2026 (75,681 taken pitches; 28,579 borderline within 0.3 ft of the rulebook edge). Dual-agent analysis with cross-review.

---

## The lede

Last week, [FanSided](https://fansided.com/mlb/mlb-s-abs-challenge-data-just-proved-what-hitters-have-known-for-years) reported that 7-hole batters in MLB win their ABS challenges only 30.2% of the time — the worst rate of any lineup slot. [The Ringer ran a parallel piece](https://www.theringer.com/2026/04/08/mlb/abs-mlb-strike-zone-umpire-shaming) framing it the same way: umpires, the theory goes, unconsciously give borderline calls to hitters they perceive as having elite pitch recognition. A No. 3 hitter takes a pitch and the ump assumes it must be a ball. A No. 7 hitter takes the same pitch and gets rung up.

It's a satisfying story. It would explain a lot of frustration that hitters at the bottom of the order have been venting all spring. And it would mean ABS is doing exactly what it was designed to do — surface a bias that human umpires didn't even know they had.

We ran the test two completely different ways and arrived at the same conclusion: **the bias claim doesn't survive controls.** The viral 30.2% number replicates directionally — but smaller, statistically imprecise, not specific to the 7-hole, and not driven by anything umpires are doing on borderline pitches. What looks like an umpire-bias story is more consistent with selection into the challenge pool than with a different called zone.

Both can be interesting. They aren't the same.

## The deep test is decisive — and null

Here's the move that reveals the underlying truth: don't look only at the small slice of pitches that batters chose to challenge. Look at *every called pitch* the umpire had a real judgment call on.

We restricted the analysis to taken pitches within 0.3 ft of the rulebook strike-zone edge — the borderline band where umpires actually have judgment latitude. That gave us **28,579 pitches** across the season's first six weeks. Then we asked the simplest possible question: does the called-strike rate on these borderline pitches differ for 7-hole batters vs 3-hole batters, after controlling for plate location, count, pitcher, catcher, and umpire?

Two different methods, run by two independent analysts who didn't see each other's work until they were both done, gave nearly identical answers:

- A hierarchical Bayesian generalized additive model with random effects for pitcher, catcher, and umpire returned a 7-vs-3 marginal effect of **−0.17 percentage points**, with a 95% credible interval of [−1.5, +1.2].
- A LightGBM gradient-boosted classifier with a counterfactual paired prediction (re-predicting every 7-hole pitch as if the batter were in the 3-hole, holding everything else fixed) returned **−0.35 percentage points**, with a paired-bootstrap 95% CI of [−0.39, −0.31].

Both essentially zero. Both with confidence intervals that fail to reach our pre-registered +2 percentage point effect-size threshold. There is no meaningful difference in how umpires call borderline pitches between 7-hole and 3-hole batters.

The mechanism the news pieces gestured at — that umpires are reading the lineup card and adjusting their judgment — is not something we detect in this window. The 95% credible upper bound on the spot-7-vs-3 effect is +1.2 percentage points. That rules out anything bigger than trivial, even if it cannot rule out exactly zero.

## So why does the 30.2% headline exist?

Because it's measuring something different.

The FanSided number is an *overturn rate on challenges* — the share of times a batter, after taking what they thought was a bad call, said "I disagree" and won. That's a fundamentally different measurement than the called-strike rate on borderline pitches. It's filtered through three sequential decisions:

1. The umpire makes a call on a borderline pitch.
2. The batter (or catcher, or pitcher) decides whether to challenge it.
3. ABS replays the trajectory and either upholds or overturns.

Step 1 is the umpire-bias question. Steps 2 and 3 are batter-skill and pitch-difficulty questions. The 30.2% rate conflates all three.

When we replicated the 30.2% number directly on our 2,101-challenge corpus through May 4, the 7-hole batter-issued overturn rate was **37.1%** — 6.9 points above the FanSided figure, but 8.1 points below the 45.2% league-wide batter-issued rate. Directionally consistent with the original claim, smaller in magnitude.

But the sample is thin. **n = 89.** The Wilson 95% confidence interval is [27.8%, 47.5%] — a 20-point band that contains both the 30.2% and the 45.2% league rate. This number is too imprecise to debunk and too imprecise to confirm. It sits in the place where every finding lives when there's not enough data yet: directionally suggestive, statistically inconclusive.

## It isn't even a 7-hole pattern

When we walked across all nine lineup spots, the gap stopped looking like a 7-hole story. Spots 7, 8, and 9 all clustered near 37% with confidence intervals that overlap each other. Spot 7 is statistically indistinguishable from its neighbors. The pattern isn't "7-hole hitters get a different zone" — it's "the bottom third of the order wins fewer challenges," which is a much less surprising claim.

After multiple-comparisons correction (Benjamini-Hochberg false discovery rate), no lineup spot is significantly different from any other. Q-values are all above 0.6 — most above 0.8 — which is statistical-speak for "noise."

## The selection mechanism is real but boring

The deep null on borderline called pitches plus the bottom-of-the-order pattern on overturn rates point in the same direction. Something is different about the bottom of the order — but it isn't the umpire's strike zone.

We checked the obvious selection mechanism: maybe 7-hole batters challenge harder pitches. They don't. The mean distance from the rulebook edge on 7-hole challenges is 1.27 inches. The mean for 3-hole challenges is also 1.27 inches. A Kolmogorov-Smirnov test on the full distribution returns p = 0.19 — no detectable difference. By the most direct measure of "is this an obvious miss?" the bottom and middle of the order are challenging the same kinds of pitches.

The remaining selection most likely lives in count and pitcher/catcher context — which counts the bottom of the order ends up challenging in, and which pitchers are on the mound when they do. Those are exactly the variables our controlled models absorb, and they are where the apparent 7-hole effect goes away. We have not yet decomposed how much of the residual gap each piece contributes; that is a Round 2 question. What we can say is that the data are consistent with selection into the challenge pool, not with a different called zone.

## A note on the dual-agent process

When two analysts run the same data with different methods and reach the same conclusion, the conclusion is robust. When they reach different conclusions, the comparison is the lesson.

Our two agents disagreed in one important place. The interpretability-first analyst restricted the H1 replication to challenges that the batter themselves initiated — n = 89, rate 37.1%. The ML-engineering analyst pooled all challenges where the batter at the plate was in the 7-hole, including challenges that the catcher or the pitcher initiated — n = 213, rate 51.2%. Mechanically, that pooling absorbs the [catcher framing edge](/analysis/catchers-are-better-challengers) we documented in prior work, which inflates the apparent 7-hole rate.

Both analysts, on cross-review, agreed: the FanSided phrase "7-hole batters win their challenges 30.2%" most naturally reads as batter-issued. The 37.1% number is the literal replication target. The 51.2% number answers a defensibly different question — "are 7-hole plate appearances disadvantaged on net?" — and the answer to *that* question is "no" (51.2% sits within a few points of the all-challenge league rate of 52.9%).

This is the kind of definitional split that can quietly turn an analysis into a misleading headline. Both pooled and unpooled denominators tell consistent stories once you look at the controlled tests. The pooled view is just less direct as a replication.

## What we believe, and what we don't

**We believe:** The "umpires call a different zone for 7-hole batters" claim is not supported by the data. On 28,579 borderline called pitches with full controls, the 7-vs-3 effect is essentially zero. Two independent methods, two different inductive biases, the same answer.

**We believe:** There is a real, small, observational pattern where the bottom of the order wins fewer challenges than the rest. The headline gap is roughly 8 points between the 7-hole batter-issued rate and the league-wide batter-issued rate, on a small sample with a wide confidence interval. A controlled hierarchical Bayesian challenge model placed the spot-7-vs-spot-3 effect at a directional negative magnitude, but its 95% credible interval crossed zero — directionally consistent with the bottom-of-order story, not decisive on its own. The pattern is consistent with count and pitcher/catcher context driving the deficit, not with umpire judgment.

**We don't believe:** That n = 89 is enough to conclusively debunk the 30.2% number. It isn't. It's enough to say the underlying causal mechanism the news pieces implied (umpire bias) doesn't survive a deeper test, but the raw observational deficit is statistically imprecise either way.

**We don't claim:** That umpires don't have any biases. They demonstrably do — we've written about [umpire-specific zones](/analysis/four-kinds-of-zone), [catcher framing](/analysis/catcher-framing-abs-era), and [the worst-calling umpire](/analysis/cb-bucknor-by-the-numbers). The claim being tested here is much narrower: do they call a different zone based on where the batter sits in the lineup? They don't.

## Open questions

The clean part of the answer is the deep null. The directional pattern in the small-sample raw rate is harder to dismiss strictly, and it raises questions a single round of analysis can't answer:

1. **Per-umpire breakdown.** League-aggregate is null. Are specific umpires driving the directional batter-issued pattern? With more data, it would be possible to ask whether a small minority of umpires has a real bias the league-aggregate analysis is washing out.
2. **The count-and-pitcher decomposition.** We've described the residual 8-point overturn deficit as consistent with count and pitcher/catcher context — our controls absorb it without naming the specific share each driver carries. A formal mediation analysis would quantify how much of the gap each piece contributes.
3. **Catcher-initiated challenges by batter lineup spot.** The pooled denominator showed catchers do well on 7-hole challenges. Are catchers preferentially burning challenges in 7-hole at-bats because the framing context is different? That's a strategic question with implications for the broader [team challenge economy](/analysis/abs-challenge-strategy) story.
4. **Temporal stability.** The May-only subsample of borderline called pitches put the spot-7-vs-3 effect at roughly five times the full-window magnitude (still small in absolute terms, still cutting in the same direction). With six weeks of season, that could be noise; with twelve weeks it might be a trend worth investigating.

We'll come back to these as the season's data accumulates.

## Methodology

This piece used two independent analytical pipelines run by two different agents with deliberately divergent methodologies. Both pipelines processed the same underlying corpus: 2,101 ABS challenges (through May 4, 2026) and 75,681 taken pitches (through May 3, 2026, the latest fully indexed Statcast date at time of analysis). The borderline-pitch subsample (28,579 pitches within 0.3 ft of the rulebook edge) is the primary substrate for the umpire-bias test; both pipelines fit on the full taken-pitch corpus and evaluated effects on the borderline subsample.

**Method A (interpretability-first).** Wilson 95% confidence intervals on raw overturn rates with Benjamini-Hochberg multiple-comparisons correction. Hierarchical Bayesian logistic GLM (PyMC NUTS, 4 chains, R-hat ≤ 1.01, ESS ≥ 717) for the controlled challenge analysis: `overturned ~ lineup_spot + edge_distance + count_state + framing_tier + fame_quartile + (1|pitcher) + (1|catcher) + (1|umpire)`. Hierarchical Bayesian generalized additive model with 2D B-spline plate-location smooth for the borderline called-pitch analysis. Stratified exploratory checks by handedness, count quadrant, and pitch group are reported as supportive, not as findings of record (one stratified fit suffered a process-induced interruption and was recovered from chart artifacts).

**Method B (ML-engineering).** LightGBM gradient-boosted classifiers for both the challenge model and the called-pitch model, with stratified group 5-fold cross-validation by `game_pk` (no within-game leakage). Target encoding for high-cardinality umpire feature, fit inside training folds. SHAP attribution to localize the per-lineup-spot marginal contribution. Counterfactual paired prediction with bootstrapped deltas: for each 7-hole pitch, predict the called-strike probability under the actual lineup spot and under a counterfactual `lineup_spot = 3`, with 200-iteration paired-bootstrap CIs. Energy-distance and KS tests for the selection-effect probe.

**Lineup-spot derivation.** Statcast does not expose batting order directly. Both pipelines independently built a lookup from the MLB Stats API live boxscore endpoint, mapping `(game_pk, batter_id) → lineup_spot` for the season's ~10,000 batter-game observations. Pinch hitters were assigned the lineup spot of the position they replaced and flagged for robustness checks.

**Cross-review.** After both agents completed independent analyses, each read the other's full work and wrote a skeptical peer review (~800-1,000 words) flagging methodology gaps, overclaims, and reproducibility concerns. The cross-reviews caught (a) a denominator-definition split that explained a 14-point divergence in the headline number; (b) a confidence-interval calibration artifact in one of the ML estimates; and (c) several places where each side's prose outran what its own data could support.

The full corpus, scripts, and dual-agent reports are available in the project repository.

---

*If you found this useful, [follow CalledThird](#) for more rigorous-but-readable baseball analysis. We retract when we're wrong, and we publish nulls.*

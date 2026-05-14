# We Tested the 7-Hole Tax Six Different Ways. It Isn't There.

**Subtitle:** Last week, two national outlets reported that umpires call a different strike zone for hitters in the 7-hole. We ran two independent statistical pipelines on every angle the claim could hide in. Six tests, two methods, all null.

**Tags:** ABS, Umpires, Methodology
**Target read time:** 11 min
**Target length:** ~2,200 words

---

## The claim

Last week, two analytical baseball outlets reported the same striking number. **7-hole batters win their ABS challenges 30.2% of the time** — the worst rate of any lineup spot. *FanSided* and *The Ringer* both ran with it. The implied mechanism: umpires unconsciously give borderline-call benefit-of-the-doubt to hitters they perceive as having elite pitch recognition, and they don't extend the same generosity to the bottom of the order.

It's a great hypothesis. The unconscious-bias framing fits a familiar pattern. The number is screenshotable. The implication — that the strike zone is a shifting social construct rather than a fixed geometric region — is exactly the kind of thing baseball Twitter loves.

But neither piece ran a controlled analysis. They reported proportions across batting order positions and let the reader fill in the causal story. We wanted to know whether the story survives once you control for the things you'd want to control for: pitch location, count, pitcher quality, catcher framing, the umpire making the call. So we ran the test.

We ran it six different ways, with two independent statistical pipelines, across two rounds of analysis. **Every one of the six tests came back null.**

This piece walks through every test we ran, what it would have shown if the 7-hole tax were real, and what it actually shows. The headline finding is that the bias claim does not exist at any level we can credibly test. The more interesting finding is what's *actually* in the data — which isn't nothing, but isn't the FanSided story either.

---

## The setup

Our corpus through May 4, 2026: **2,101 ABS challenges** across the regular season to date. **75,681 taken pitches** with full Statcast tracking. **28,579 of those pitches** sit within ±0.3 ft of the rulebook strike-zone edge — the borderline region where umpires actually have judgment latitude.

We pulled batting order from the MLB Stats API boxscore feed for every game (Statcast doesn't expose lineup spot directly). We built pitcher-quality tiers from prior-season K-BB%. We built catcher framing tiers from prior-season framing runs. We linked every challenge to its umpire.

Then we asked the question two ways. **Agent A** (a hierarchical Bayesian model with random effects on umpire, pitcher, and catcher) and **Agent B** (a gradient-boosted classifier with counterfactual paired prediction) ran independent analyses with deliberately divergent methods. Each agent produced a full report. Each agent reviewed the other's work. We followed the dual-agent protocol that powered our [Pitch Tunneling Atlas](/analysis/pitch-tunneling-atlas) and [Coaching Gap](/analysis/coaching-gap-patience) pieces.

The point of running two methods is robustness — when both agents converge on a finding, the finding is locked in. When they diverge, the disagreement is itself the story. Both happened in this project. The convergence is the headline. The divergences became the cross-review section.

---

## Test 1: Does the viral 30.2% number replicate?

The most direct test: does the 30.2% rate hold up in our larger, more recent corpus?

The answer is "directionally, yes — but not at that magnitude, and not statistically."

Restricting to **batter-issued challenges** (the natural reading of "7-hole batters win their challenges"), we get **37.1% on n=89** challenges, with a Wilson 95% CI of **[27.8%, 47.5%]**. The league-wide batter-issued overturn rate is 45.2%. The 7-hole deficit is 8.1pp.

That deficit is below our pre-registered 10pp gate for "the headline number replicates." The CI is 20pp wide and contains everything from FanSided's 30.2% to roughly the league rate. With n=89, the Wilson CI honestly cannot rule out either the original number or the null hypothesis of "no effect." The original report wasn't *wrong* — it was a small-sample observation that didn't survive the size of the window we're now able to measure.

There's also a different way to compute the rate that produces a different answer. If you pool *all* challenges where a 7-hole batter is at the plate — including catcher- and pitcher-initiated challenges — the rate jumps to **51.2% on n=213**, essentially league-flat. That's because catchers in our prior research [overturn at 60.6% across all positions](/analysis/catchers-are-better-challengers), so pooling them in with batter-issued challenges mechanically inflates the 7-hole at-bat rate.

Both numbers are correct under their own definitions. Neither one supports a clean read of "umpires uniquely punish 7-hole batters."

---

## Test 2: Are 7-hole batters challenging worse pitches?

If 7-hole batters are choosing harder fights — picking calls that are systematically further from the rulebook edge, or further from the umpire's typical zone — the lower overturn rate would be a selection effect, not a bias effect.

We tested this directly. The mean absolute distance from the rulebook edge is **1.27 inches** for both 7-hole and 3-hole challenges (all challenger types pooled, n=213 / n=259). A Kolmogorov-Smirnov test on the full edge-distance distribution returns **p = 0.19**. The same direction holds when restricted to batter-issued challenges (1.32 in vs 1.29 in). There is no signal that 7-hole batters — or 7-hole at-bats more broadly — face different challenged pitches than 3-hole.

So selection on pitch location can't explain the gap. Whatever's happening lives elsewhere — in the count distribution, in pitcher matchups, or, theoretically, in umpire bias.

---

## Test 3: The deep test — borderline-pitch zone analysis

This is the test that matters. The challenge-overturn-rate analyses above are limited to the small subset of pitches that get challenged. The actual question — *does the umpire's strike zone differ for 7-hole batters?* — is best answered by looking at *every* called pitch in the borderline region, not just the ones a batter or catcher pushed back on.

We took every taken pitch within ±0.3 ft of the rulebook edge — **28,579 pitches** — and modeled `is_called_strike` as a function of lineup spot, plate location (2D smooth), count state, pitcher fame quartile, catcher framing tier, and umpire random effects.

Both methods agree on the answer.

| Method | Spot 7 vs Spot 3 effect | 95% interval |
|---|---|---|
| Hierarchical Bayesian GAM | **−0.17pp** | [−1.5, +1.2] |
| LightGBM counterfactual paired prediction | **−0.35pp** | [−0.39, −0.31] |

Both essentially zero. (The narrower ML interval is a paired bootstrap over fitted prediction deltas with the LightGBM mapping held fixed; it is not a model-uncertainty interval. Treat it as a precision-of-fitted-deltas diagnostic, and use the wider Bayesian interval as the publishable uncertainty statement. We discuss this in the methodology callout below.)

We pre-registered a +2pp threshold for "the bias is real" before fitting either model. Both methods miss it by an order of magnitude. The deep test — the one that asks the actual question instead of a chargeback proxy — finds nothing.

We stratified by handedness (L vs R), by count quadrant (hitter's vs pitcher's count), and by pitch type (fastball vs breaking vs offspeed). No stratum where the effect lives.

---

## Test 4: Could it be specific umpires?

A league-wide null might mask a real effect that lives in a few specific umpires. Maybe the median umpire is unbiased, but a handful of umpires really do call a different zone for 7-hole batters, and the average washes out their effect.

We tested this with proper hierarchical shrinkage. **78 umpires qualified** (≥50 borderline calls each in lineup spots 7-9 *and* in spots 1-3). Each umpire's random slope was estimated on a bottom-of-order indicator (spots 7-9 vs 1-3), not strict spot-7, to preserve statistical power; the league-mean spot-7 effect from the main GAM (the −0.17pp number above) is what we contrast against. A strict-spot-7 random-slopes sensitivity is queued for our All-Star-break re-test. Each umpire's posterior was estimated jointly with all other umpires, with partial pooling toward the league mean.

**Zero of 78 umpires had a 95% credible interval excluding zero AND a magnitude ≥ 2pp.** The full per-umpire effect distribution sits in a narrow band around zero, with the random-slope SD posterior at 0.076 logit (roughly 1.7pp per SD). The 95% upper bound on umpire heterogeneity is about 4pp — meaning the data is *consistent* with modest umpire-by-umpire variation, but no individual umpire stands out as a credibly biased actor.

The ML method initially flagged five umpires, all in the *reverse* direction — fewer called strikes for 7-hole batters, i.e. a smaller called-strike zone, not the larger zone the tax story requires. Cross-review found that those flags don't survive Bayesian replication: two of the five have positive posterior medians under hierarchical shrinkage, and none clears the Bayesian credible-interval threshold, despite ML effects as large as −4.47pp.

The cross-method convergence test is what publication standards in this kind of analysis should look like. The five ML flags are the kind of finding that survives a single methodology and dissolves under the second one. We don't name any umpires in this article, in either direction.

---

## Test 5: Could it be specific hitters?

The same logic at the hitter level. A league-wide null could mask a small set of bottom-of-order hitters who are personally getting short-changed — the "specific victims" version of the FanSided claim.

We computed posterior-predictive residuals for every batter with ≥30 borderline take decisions in spots 7-9. **119 hitters qualified.** For each, we computed their actual called-strike rate on borderline pitches versus what the controlled model expected (given their pitches' locations, counts, pitcher and catcher quality, and umpires).

After BH-FDR correction, **zero hitters were credibly worse off** than the model expected.

Two hitters surfaced in both methods as credibly *better off* than expected — Cam Smith and Pete Crow-Armstrong. Claude had one additional Bayesian-only hitter flag, but it did not survive the Codex bootstrap robustness check, so we do not treat it as publication-level. Neither cross-method hitter is a disciplined-hitter archetype that fits the FanSided "elite-pitch-recognition" narrative — Crow-Armstrong's 2025 chase rate was 0.42, well above league average.

The right read on those names is that they're **model-residual anomalies, not evidence of umpire favoritism.** Our controlled model intentionally has no batter-specific features (we didn't want to absorb the very effect we were looking for); the residuals capture *unmodeled* batter variation, not biased treatment. They're interesting-but-not-actionable, and we'd want a larger sample before reading anything causal into them.

---

## Test 6: The catcher mechanism

The denominator split between Test 1's 37.1% (batter-issued) and 51.2% (pooled) hinted at one possible mechanism: maybe catchers, who initiate more challenges than batters, are systematically picking *different* pitches in 7-hole at-bats than they do in 3-hole at-bats. If catchers selectively challenge harder calls when the bottom of the order is up, that would inflate the pooled rate without saying anything about umpire bias.

We tested this directly. Restricting to catcher-initiated challenges, we modeled edge-distance as a function of batter lineup spot, with all the same controls. The spot-7-vs-3 effect on catchers' challenged-pitch edge-distance is **−0.02 inches with a 95% CrI of [−0.28, +0.26]**. The Codex pipeline's energy-distance probe across the joint feature distribution returns **p = 0.955.**

Both methods agree: **catchers do not pick systematically different challenges in 7-hole at-bats.** The denominator split between 37.1% and 51.2% is a real numerical fact, but it isn't a catcher-strategy story. It's an arithmetic byproduct of catcher-initiated challenges having a higher overturn rate league-wide than batter-initiated challenges.

---

## What's actually in the data

The bias mechanism doesn't exist at any level we can test. But there is a real pattern in the descriptive numbers.

**Spots 7, 8, and 9 all hover around 37% on batter-issued overturn rate**, with overlapping CIs. After Benjamini-Hochberg correction, no individual spot is statistically distinguishable from any other. So whatever's going on, it's a "bottom third of the order" pattern, not a "7-hole specifically" pattern. The "7-hole tax" naming is headline aesthetics, not a feature of the data.

The most likely explanation is **count and pitcher selection** — bottom-of-order hitters end up in different count states and face different pitchers than the top of the order. Those interactions absorb most of the raw 8pp deficit once you condition on them in a controlled model. We didn't fully decompose count × pitcher in this round, but both methods' selection-effect probes converge on this being where the residual signal lives.

We also tested whether the FanSided "elite-pitch-recognition" mechanism survives within the most disciplined chase-rate tertile of bottom-of-order batters. The Bayesian estimate is **−1.45pp [−3.98, +1.03]** on low-chase 7-hole batters. Under the p7 − p3 convention, that negative sign is the reverse/favored direction — fewer called strikes for 7-hole batters, not the larger zone the tax story requires — and the 95% CrI contains zero. Probability of the effect being negative is 0.87, so the low-chase slice is worth retesting but is not evidence for the FanSided mechanism.

That's the lone low-chase shape worth retesting, but it is not a tax signal as signed. We're flagging it as a research-queue item for an All-Star-break re-test, not a finding.

---

## What the dual-agent process caught

Two methodological details worth flagging, because they're the kind of thing that gets baked into a published article and then becomes The Number People Cite.

**The narrow CI artifact.** Codex's pipeline initially reported the H7 low-chase effect at −0.78pp with a 95% interval of [−0.88, −0.66]. That's tighter than warranted for an 804-row spot-7 low-chase stratum and a 1,241-row spot-3 comparison. Cross-review found that the bootstrap was resampling per-pitch fitted prediction deltas while treating the underlying gradient-boosted model as fixed — capturing within-row variance but not parameter uncertainty. The more honest publication interval is the Bayesian estimate's [−3.98, +1.03], which we use throughout this piece. We caught the same kind of artifact in [our April 23 ABS Walk Spike analysis](/analysis/abs-walk-spike-zone-correction); it's a recurring failure mode of bootstrap-of-fitted-deltas methodology.

**The sign convention error.** One of our drafts referred to negative effects (lower called-strike rate for spot 7) as "pro-tax" — but a *lower* called-strike rate means *more balls* are being called, which is the *favored* direction, not the taxed direction. The cross-review caught the error before it shipped. We mention this not to perform humility but because errors of this kind get into published baseball analytics constantly, and they meaningfully change how a reader interprets a chart. We'd rather catch them than not.

---

## What we're not claiming

A null result is not a debunk. We're not saying umpires have zero bias against any subset of hitters under any conditions. We're saying:

- The specific FanSided/Ringer mechanism — "umpires give 7-hole batters a larger called-strike zone than 3-hole batters" — does not exist at any of six levels we tested
- The 30.2% headline number is real but is the floor of a wide confidence interval that contains the league rate
- The "7-hole" framing is a feature of where the headline number landed, not a feature of where the underlying pattern lives
- A faint directional shape exists in the low-chase tertile that may tighten with more data; we're flagging it for an All-Star re-test

Our sample on the most direct test — batter-issued challenges from 7-hole hitters — is genuinely thin. n=89. Our wider tests on borderline called pitches use far more data (28,579) but answer a slightly different question. The article is written within those sample-size limits. A reader who is unconvinced by the null because they want a stronger test is, mostly, asking for more data, which we'll have by July.

---

## Methodology

Two independent agents — one hierarchical Bayesian (`PyMC` random-slopes GLM + GAM), one ML (`LightGBM` with counterfactual paired prediction + SHAP + energy-distance selection probe) — analyzed the same 28,579-borderline-pitch corpus with deliberately divergent methods across two rounds. Each agent then read the other's full work and wrote a skeptical peer review (800-1,100 words each). Cross-review caught the CI artifact, the sign-convention error, and refined which named hitters survived to publication.

All code, data substrates, and per-actor leaderboards are open source on GitHub: [seven-hole-tax repository](https://github.com/prismindanalytics/calledthird). Convergence diagnostics for every Bayesian fit (R-hat ≤ 1.01, ESS ≥ 400) are in the diagnostics folder.

The faint H7 low-chase signal and a strict spot-7 random-slopes sensitivity are queued for a Round 3 re-test at the All-Star break, when we'll have roughly twice the sample.

If anything in this analysis is wrong, we'd like to know. The point of publishing the methodology is so that someone else can find a flaw we missed.

---

## Summary

| Test | What it asks | Result |
|---|---|---|
| 1. Raw replication | Does the 30.2% number hold up? | Directional but underpowered (37.1%, n=89, CI [27.8, 47.5]) |
| 2. Pitch selection | Are 7-hole batters challenging worse pitches? | No (KS p=0.19) |
| 3. Borderline-pitch zone | Do umpires call a different zone? | No (~0pp on n=28,579, both methods) |
| 4. Per-umpire | Is there a credibly-biased umpire? | No (0/78 flagged after shrinkage) |
| 5. Per-hitter | Is there a credibly-victim hitter? | No (2 cross-method reverse-direction residual outliers, model-residual not bias) |
| 6. Catcher mechanism | Do catchers pick harder fights in 7-hole? | No (effect 0pp, p=0.955) |

Six tests. Two methods. Two cross-reviews. All null on the bias claim.

The 7-hole tax is not in the data we have, in any form we can credibly measure. We tested.

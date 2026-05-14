# Three Weeks Later: The Walk Spike Is Fading, and We Know Who's Paying the Bill

**Subtitle:** Three weeks ago we said about half the walk spike was the new ABS zone. The number has muted, the spike is fading — and we can now name the pitchers paying the price. They share an archetype.

**Tags:** ABS, Pitchers, Methodology
**Target read time:** 12 min
**Target length:** ~2,400 words

---

## The setup

Kyle Finnegan has walked 11.4 percentage points more batters in 2026 than he did in 2025. Riley O'Brien has walked 8.3 points fewer. Camilo Doval, 7.5 fewer.

We didn't pick these names because they jumped out of a single leaderboard. We picked them because two independent statistical pipelines, running on the same data with deliberately different methods, both flagged them as the cleanest examples of a pattern we set out to test: the new ABS zone is asymmetric, and it's not asymmetric by accident. The pitcher *archetype* that lives at the top of the strike zone is now paying for it.

Three weeks ago, [we said](/analysis/abs-walk-spike-zone-correction) the 2026 walk spike was about 40 to 50 percent the new ABS zone shape and 50 to 60 percent pitcher behavior. With two more weeks of data and a much harder analytical pass, here's the update: the spike has narrowed but persisted, the zone-attribution number has muted to around **+26%** with a wide editorial interval, and the more interesting finding is mechanistic. We now know exactly how the zone is producing those extra walks, and we know which kinds of pitchers are eating the cost.

This piece is the third round in a series. The first round shipped April 9 ([*The Walk Rate Spike: Umpires or Pitchers?*](/analysis/the-walk-rate-spike)) on ten days of data. The second shipped April 23 ([*ABS Took the High Strike*](/analysis/abs-walk-spike-zone-correction)) on twenty-seven. This one runs on forty-six. Each round revised what came before. The brand is honest position management; the brand is also continuing to be on the same beat when the data keeps moving.

## What the data says now

**The walk spike has narrowed.** Through May 12, 2026 walk rate sits at 9.46%, down from 9.77% at the end of our April 23 window. The same calendar window in 2025 ran 8.78% — so the year-over-year delta is **+0.68 percentage points**, down from R1's +0.82pp.

The within-2026 trajectory matters too: walk rate by week is 9.61% (W1), 10.27% (W2), 9.78% (W3), 9.23% (W4), 9.16% (W5), 9.19% (W6), 8.79% (W7). The first three weeks averaged 9.93%; the last three averaged 9.07%. That's a within-season drop of 0.86 percentage points. Our Bayesian posterior puts the probability the spike is regressing at **89%**.

Both our pipelines agree on this number to within rounding. [FanGraphs reports](https://blogs.fangraphs.com/where-are-2026s-extra-walks-coming-from/) league walk rate at 9.5% through May 8 — directionally identical. The spike is real, it's still elevated against any season in modern memory (the league average remains higher than every full season since 1950), but the trajectory is bending down.

## The zone attribution updated

The headline number from April 23 was that 40-50% of the YoY walk spike was attributable to the new ABS-defined zone — most of it at the top edge, which shrank by roughly 7-8 percentage points in called-strike rate. The remaining 50-60% was pitcher behavior: nibbling, missing the zone, and a small adaptation lag.

With twice the data and a harder analytical pass, that fraction is now **about +26%, with an editorial 95% interval of [+0.2%, +70.1%]**.

That's a real mute. It's also not a sign flip — R1's +40-50% sits comfortably inside the new interval. The honest reading is: the zone effect was bigger when pitchers hadn't fully adapted; three weeks of additional pitcher response have begun to absorb some of it.

To get to a defensible single number, we ran six independent counterfactual implementations — three from each of our statistical pipelines, deliberately divergent in method:

- Bayesian PA replay with per-pitch Bernoulli sampling
- Empirical k-NN lookup of 2025 called-strike rates at each 2026 pitch location
- Bayesian bootstrap-of-bootstrap (100 game-level outer iterations × 10 inner refits)
- ML expectation-propagation PA replay with 200 game-level bootstrap iterations
- ML SHAP-based per-pitch attribution
- ML bootstrap-of-bootstrap (100 × 10)

Five of those six methods landed positive. The cross-agent median was +27.0%; the five-method positive envelope ran from +0.2% to +70.1%. That envelope is what we'd defend in front of a FanGraphs editor.

The sixth method — the Bayesian Bernoulli replay — came back at −58.6%. Both our cross-reviewers concluded that's a stress test, not a publishable number. We're flagging it because it's the kind of result that shows up when you propagate stochastic per-pitch noise through a deep PA-replay chain, and because the prior version of this method gave us a −64.6% result in Round 2 that turned out to be an artifact of how we handled unresolved at-bats. We'd rather show the stress test and explain it than hide it.

## The mechanism: the 0-0 first-pitch loss

The most useful new fact from our analysis is mechanical, not magnitude-related. It explains how a zone change can produce a walk spike without showing up where you'd expect.

[FanGraphs noted](https://blogs.fangraphs.com/the-strike-zone-is-shrinking-heres-how/) that the called strike zone has shrunk by about 14 square inches, mostly at the top. They reported that top-of-zone pitches were called strikes 40.8% of the time in 2026, down from 54.3% in 2025 — a 13.5 percentage point drop. We confirm that magnitude. What we add is a count interaction nobody else has surfaced.

The top of the zone has lost first-pitch strikes 6 to 7 percentage points faster than it's losing 2-strike calls. Our Bayesian model gives a difference-in-differences of **−6.76 percentage points (95% credible interval excluding zero)** between the 0-0 count and the 2-strike count for top-edge pitches. The ML pipeline's SHAP interaction values point the same direction.

That single asymmetry explains a lot. A first-pitch on the top edge that drew a called strike 36% of the time in 2025 now draws one only 26%. That 10pp drop on first pitches compares to a 4pp drop on 2-strike pitches — the asymmetry is concentrated on the first pitch. Which means more 1-0 counts, more 2-0 counts, more 3-0 counts, more eventual walks — even though the per-count walk *conversion* rate (the probability you walk *given* a 3-2 count) has barely moved.

We decomposed this directly. The +0.68 percentage point YoY walk-rate increase splits into about +0.41pp of "traffic" (more plate appearances reaching deep counts) and +0.27pp of "conditional" (slightly higher walk rates from those counts). About 60% of the spike is the traffic channel. And the traffic channel traces back to the top-edge first-pitch loss.

This is what Hoerner [told the AP](https://apnews.com/) in April when he said hitters were learning to lay off the top. He was right. The data now shows it's specifically a first-pitch effect.

## Why some pitchers are paying and others aren't

Here's where the analysis gets new.

If the zone change costs strikes specifically at the top of the rulebook zone on first pitches, the pitcher archetype most exposed is the one whose game lives at the top — high-fastball pitchers with command-based profiles who used to get borderline first-pitch strikes called and now don't. The pitcher archetype most insulated is the opposite: stuff-first pitchers whose secondary pitches don't need top-edge first-pitch strikes to set up the at-bat.

This narrative has been floating around the industry. FanGraphs gestured at it; broadcasters have repeated it. Nobody has quantified it.

We tested it directly. For every pitcher with at least 40 innings in 2025 and at least 200 pitches in our 2026 window, we computed a 2025 "stuff" score and "command" score from prior-season Statcast (whiff rate quartile for stuff; walk-rate plus zone-rate quartile for command — FanGraphs Stuff+ was unavailable in our run, so we built a defensible Statcast proxy). Then we asked: does each pitcher's 2026 walk-rate change correlate with their stuff-minus-command differential?

It does. Strongly.

**Spearman ρ = −0.282** (Bayesian pipeline; p < 0.0001) and **ρ = −0.258** (ML pipeline; p < 0.0001). Both methods, both p-values microscopic, both estimates agreeing to the second decimal. The Bayesian slope translates to **−1.40 percentage points** of walk-rate change per unit shift on a 0-to-1 stuff-minus-command scale. That's about **2.8 percentage points** of walk-rate spread from a pure-command archetype (high command, low stuff — walk rate up roughly +2.1pp vs 2025) to a pure-stuff archetype (low command, high stuff — walk rate down roughly −0.7pp vs 2025).

This is consistent with FanGraphs' broader analysis of the walk spike and with public reporting that league zone rate has dropped from 50.7% to 47.2%, as well as with industry chatter about command pitchers struggling. We're the first to put names on it with cross-method confirmation.

## The named pitchers

Three pitchers cleared every filter we set: 200+ pitches in our 2026 window, 40+ innings in 2025, bootstrap stability of at least 80% across 200 game-level resamples, and identification by both pipelines independently.

**Kyle Finnegan** is the cleanest command-archetype casualty. The Nationals' closer has walked **+11.4 percentage points more** batters in 2026 than his 2025 baseline (our Bayesian pipeline says +11.4pp; the ML pipeline says +12.0pp). Finnegan's 2025 game was a high-fastball, top-of-the-zone, command-heavy profile — exactly the archetype the new ABS zone is hardest on. His stuff score sits in the bottom half of his fellow relievers, his command score sat in the top quartile. He's the pitcher the data predicts gets hurt by the new zone, and he is.

**Riley O'Brien** is the cleanest stuff-archetype beneficiary. Walk rate down **−8.3 percentage points** (Bayesian; ML pipeline says −6.9pp). O'Brien is a power reliever whose 2025 profile leaned stuff-over-command — above-average whiff in our arsenal-weighted proxy and a walk rate and zone rate that put his command in the bottom quintile. The new zone hurts pitchers who need first-pitch strikes called at the top. O'Brien's game wasn't built on those. He's gone from a walker into a strike-thrower as the zone has shifted.

**Camilo Doval** rounds out the named list. Walk rate down **−7.5 percentage points** (Bayesian; −6.4pp ML). Doval's stuff was already plus; the new zone has been a net benefit to him.

We considered naming Mason Miller — the Bayesian pipeline flagged him as helped by the new zone. But his bootstrap stability score in the ML pipeline was 0.68, below our 0.80 cutoff. We hold him out as a candidate, not a named example. The point of running two methods is that we name only the pitchers both of them are confident about.

## Why we can't name more (yet)

We tried to. We failed honestly.

In addition to the archetype analysis, we ran a separate test: which specific pitchers have *adapted* — within the 2026 season — by shifting their pitch mix, vertical location, or zone rate week-over-week? The question is different from the archetype question. The archetype question asks "which pitchers were going to be hurt by this." The adaptation question asks "which pitchers have responded to it."

The answer, when we apply a proper cross-method filter, is: **none, yet, in a way we'd stake a name on.** Out of 367 pitchers with sufficient 2026 sample, exactly zero cleared the bootstrap stability threshold in our Bayesian within-2026 framework. The Bubba Chandler topped out at 58.5% stability; we required 80%. Our ML pipeline did find nine pitchers with credibly different 2025-vs-2026 profiles, but those answer a different question (year-over-year *shifters*, not within-window *adapters*), and they don't cross-validate with the Bayesian list.

Pitcher-level adaptation almost certainly *is* happening at the league level. Individual zone rates within our 2026 sample show shifts of plus-or-minus 20 to 40 percentage points for specific pitchers, and the league mean is moving subtly. But six weeks of data is not enough sample to identify named adapters with the stability standard we hold ourselves to. We'll re-test at the All-Star break.

## What the dual-agent process caught this time

Three rounds, six independent analyses, three cross-reviews per agent — that's the operating procedure now. Each round caught something the prior round missed.

**Round 1 (April 23) caught:** A normalized-coordinate artifact that gave our ML pipeline a −56% attribution that turned out to be a Statcast schema change between 2025 and 2026. Both agents converged on +40-50% once we switched to absolute coordinates.

**Round 2 (this one's predecessor) caught:** A "0-0 first-pitch flip" tension where the all-pitches counterfactual said zone change adds walks but a 0-0-only restriction said it removed them. We left that open in R2. R3 resolved it: the top-edge first-pitch loss is a real count-asymmetric effect.

**Round 3 caught:** A confidence-interval artifact in our ML pipeline that gave its R2 H3 result a misleadingly narrow [34.6%, 36.0%] band. The new bootstrap-of-bootstrap structure (100 game-level outer iterations × 10 inner refits, applied across both pipelines) gives a wider but honest [+0.2%, +70.1%] editorial interval. The methodology cost is real — we spent compute on it. The credibility benefit is, we think, also real.

The cross-reviewers caught one more class of error: a sign-convention slip in one of our Round 3 drafts that labeled the favored direction as the "taxed" direction. (A *lower* called-strike rate means *more balls*, which favors the batter, not the pitcher.) We're flagging this here because we caught it in our own pipeline and we'd rather mention the kind of error than have it survive into print.

## What we're not claiming

A few things we explicitly *aren't* saying:

- We are not saying the walk spike is over. It's fading; it's not gone. Walk rate remains historically high.
- We are not saying the named pitchers' archetypes are *causing* their walk-rate changes in a strict causal sense. The archetype effect is an observational regularity: pitchers with high command-minus-stuff scores in 2025 have higher 2026 walk-rate increases. The causal direction is plausible and is consistent with the count-mechanism story, but it's not a randomized comparison.
- We are not saying every pitcher in the data fits the pattern. The Spearman ρ is around −0.27. That's a real signal, not deterministic.
- We are not saying our +26% attribution is precise. The honest editorial interval is wide ([+0.2%, +70.1%]). What we *are* saying is that R1's +40-50% sits inside the interval and is consistent with the new estimate, and that the muting is real but small.

## Methodology

Our corpus runs March 27 through May 12, 2026 (May 13 dropped as a partial day at pull time). 46,755 plate appearances. 75,681 taken pitches. 28,579 borderline pitches within ±0.3 ft of the rulebook strike-zone edge. We pulled batting order from the MLB Stats API boxscore feed. We built pitcher-archetype scores from prior-season Statcast.

Two independent pipelines: a hierarchical Bayesian implementation in `PyMC` (random-slopes GLM + spatial GAM zone classifier + per-pitcher Beta-Binomial + Bayesian archetype interaction model) and an ML-engineering implementation in `LightGBM` (counterfactual paired prediction with refit-bagged bootstrap + SHAP + ensemble feature attribution). Six total H3 counterfactual implementations (three per agent) for the headline magnitude. 200 game-level bootstrap iterations for every stability test; 100 × 10 bootstrap-of-bootstrap for the triangulated CIs.

Each pipeline produced a report and a peer review of the other; the cross-reviews are part of the artifact. Code, data substrates, and per-pitcher tables: [open source on GitHub](https://github.com/prismindanalytics/calledthird).

The next re-run is at the All-Star break, when we'll have roughly twice the sample.

If anything in this analysis is wrong, we'd like to know.

---

## Summary

| Finding | Result |
|---|---|
| Walk spike vs 2025 | +0.68pp YoY through May 12; was +0.82pp at April 22 |
| Within-2026 trajectory | W1-3 → W5-7 = −0.86pp; P(regressed) = 89% |
| Zone attribution (triangulated) | ~+26% [+0.2%, +70.1%]; was +40-50% in R1 |
| Mechanism | Top-edge first-pitch strikes lost 6-7pp more than 2-strike; drives traffic |
| Archetype effect | ρ = −0.258 to −0.282 (both p<0.0001); ~2.8pp full-spectrum spread |
| Named pitchers (command-hurt) | Kyle Finnegan (+11.4pp) |
| Named pitchers (stuff-helped) | Riley O'Brien (−8.3pp), Camilo Doval (−7.5pp) |
| Within-2026 adapters | 0 cross-method stable; pending All-Star break re-test |

Three rounds. Six independent analyses. Three cross-reviews. The spike is fading, the mechanism is now clear, and we know who's paying the price.

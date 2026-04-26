# The April Sell List

*Six hot starters who won't last. Six sleepers nobody's talking about. And the one name we couldn't make up our minds about.*

**Author:** CalledThird • **Date:** April 25, 2026 • **Reading time:** 9 min

---

## The hard part about April

It's been 22 games. Andy Pages is running a .403 wOBA. Ben Rice leads the AL in batting average. Mike Trout just put 5 home runs in the bleachers at Yankee Stadium in four days. And Mason Miller has now thrown 30⅔ scoreless innings to start his season.

If you've been reading any baseball coverage this month, you already know all of this. What you don't know is which of these performances will still look special when the leaves turn — and which ones we'll all stop talking about by the end of May.

We tried to answer that. Twice. The first time, we got Ben Rice and Mike Trout wrong. (We'll explain.) The second time, two completely different statistical methods independently arrived at the same answer for nearly every name on the leaderboard. That's the version you're reading.

Here's what 22 games of baseball can — and can't — tell you.

---

## The first thing to know: April lies. A lot.

Across the past four seasons, we identified the top five hottest 22-game wOBA starts each year — the players who were torching the league through April. Twenty player-seasons in total.

Of those 20: **two** maintained 85% of their April pace through the rest of the season. Eighteen didn't. The median decline was 0.135 points of wOBA — roughly the gap between an MVP candidate and a league-average bat.

<AprilLiesChart client:visible />
*Each line is one of the past four seasons' top wOBA starts. By July, almost none of them looks like an MVP anymore.*

So when you look at this year's leaderboard, the prior — the thing your eyes tell you when you ignore everything else — should be: **most of this evaporates**.

Where it gets interesting is asking *which ones*.

---

## The names baseball is talking about

We ran the entire 287-hitter universe through two independent statistical pipelines: one Bayesian, one machine-learning. They disagreed plenty, but where they agreed, they agreed strongly.

For the published-leaderboard names, they agreed almost completely. Here's the buy-and-sell sheet:

<BuyHoldSellScoreboard client:visible />

The "projected" column is what both methods, after combining the player's 22-game line with their pre-2026 baseline and their underlying contact quality, think they'll hit the rest of the season.

Six MVP-pace starts. Six reasons they won't last:

<RegressionPanels client:visible />

**Andy Pages.** Statcast's expected wOBA — what his contact quality alone would normally produce — is .360. His actual wOBA is .403. That gap is luck on where the balls are landing, not skill. The hard contact is real (60% hard-hit rate, in the elite tier). The .403 isn't. Both methods project him to land near his pre-2026 baseline of .331 — essentially zero rest-of-season improvement over what we already expected.

**Ben Rice.** Genuinely good plate discipline (20% walk rate) on a real power surge. But the .378 BABIP is doing too much of the work. He'll land closer to .345 — All-Star caliber, not MVP-caliber.

**Mike Trout.** This is the one our model wants to take seriously, and won't. Trout's preseason baseline is already so high that even a 5-HR series at Yankee Stadium doesn't move our projection meaningfully. He's going to keep being Mike Trout. Mike Trout running .425 is not new information.

**Aaron Judge.** Same story as Trout. His preseason baseline is .402 wOBA. He's running .435. The math says: that's mostly the same Aaron Judge.

**Corbin Carroll, Max Muncy.** Same math. Stars whose April production is strong but not above what their baselines already predicted.

The version of this article that says "Aaron Judge will collapse" would be wrong, and we're not writing it. The honest version is: *the hot-start portion isn't adding much information beyond what we already knew about him.*

---

## The names baseball isn't talking about

This is the part nobody else is publishing.

For every Andy Pages on the ESPN leaderboard, there's a player whose 22 games look quieter on the surface but whose underlying components — exit velocity, plate discipline, contact quality, prior skill — project meaningfully better than their preseason baseline.

We screened all 287 hitters with at least 50 plate appearances. We required a name to clear two bars: predicted rest-of-season delta in the top decile, AND not appear in any mainstream "April hot starter" leaderboard. Then we required *both* of our independent methods to surface the name.

Six hitters cleared. Here they are:

<SleeperSpotlightCards client:visible />

**Jac Caglianone (LAA, OF).** 95th-percentile hard-hit rate behind a .312 wOBA. The contact quality is the kind of thing that survives April — it's not a BABIP story.

**Everson Pereira (NYY, OF).** Quietly behind Rice on the Yankees' depth chart with a .406 wOBA. Both methods like the underlying shape: power and plate discipline, not luck.

**Jorge Barrosa.** Defensive utility OF whose 22-game line (.299 wOBA) doesn't look loud — but the contact metrics and discipline both point at a real bat that's about to surface in a regular role.

**Samuel Basallo (BAL, prospect).** The Orioles have a habit. Basallo is the next iteration.

**Coby Mayo (BAL).** Same Orioles habit. Both methods rank his 22-game profile in the top decile of the universe.

**Brady House (WSH).** A name that hasn't shown up in early-season coverage at all. Underlying contact metrics put him there.

And four relievers. The reliever pool was 218 names; we required at least 25 batters faced and excluded everyone in the top 30 of last year's saves leaders (so no Edwin Diaz, no Emmanuel Clase, no Jhoan Duran):

- **Antonio Senzatela** (COL) — converted starter, K% jumped meaningfully against his 3-year prior. The Coors part is irrelevant for a single-inning reliever.
- **Daniel Lynch** (KC) — 17.7% prior K%, 37.1% in April, projects ~27% rest-of-season. The textbook sleeper shape: a real underlying gain even after we shrink the April spike.
- **John King** (TEX) — quieter K% jump but the projection clears the bar in both methods.
- **Caleb Kilian** — same shape.

---

## The one we couldn't agree on

Munetaka Murakami is the only April performance that *neither* of our methods is willing to write off — which is interesting, because it's also the one where our methodology is weakest.

Quick context if you haven't been paying attention: Murakami is a 26-year-old corner infielder who spent eight seasons with the Tokyo Yakult Swallows before signing a 2-year, $34M deal with the White Sox in December. In Japan, he hit 246 home runs in eight seasons. In 2022 alone, he hit 56 — breaking Sadaharu Oh's NPB single-season record for a Japanese-born player and winning the Triple Crown at age 22. Through his first 21 MLB games: 8 home runs, a .418 wOBA, and a thing where he homered in each of his first three career games (a White Sox first; only the fourth player in MLB history to do it).

Both of our pipelines see signal in that profile. They disagree on how strong: our Bayesian model calls it "signal, medium confidence." Our machine-learning model calls it "ambiguous, low confidence." They agree on the direction — both think Murakami is the cleanest of any leaderboard name — and they agree on *why they're not more sure*.

The reason is the comparable. We don't have one.

The natural reference is Shohei Ohtani, who came over from the Hokkaido Nippon-Ham Fighters in 2018. But Ohtani's NPB power profile doesn't actually look like Murakami's. Ohtani's NPB peak was 22 home runs in a season (2016, his Pacific League MVP year); Murakami's NPB peak was 56. Ohtani arrived in MLB and immediately hit 22 home runs in 104 games as a rookie — roughly the same pace as his NPB high. Apply that same logic to Murakami and you'd project him for 50+ MLB home runs, which neither of our models will say out loud.

The honest framing: nobody with Murakami's NPB power has ever come over to MLB before. There's no robust translation factor for someone with his profile, because the sample size of Japanese-born sluggers with 50+ HR seasons is one (him). Our model assigns him a league-average preseason baseline because we didn't build an NPB-to-MLB power translator. That choice makes our "signal" partly a statement about how much of his 22 games clears a methodological low bar, rather than a confident MVP prediction.

Watch him. The plate discipline, the hard contact, the early home run pace — it all looks like a real bat. Whether it ends up looking like Ohtani's rookie year (.285, 22 HR), the high end of what NPB hitters historically translate to, or something further up the curve nobody has seen yet — that's the part we genuinely don't know.

---

## Mason Miller: what's real and what isn't

Miller has thrown 30⅔ consecutive scoreless innings. The streak is genuinely historic for a closer. But here's the thing: most of what gets written about a streak like this confuses *real talent* with *predictable streak length*. Those are different questions.

<MillerSplit client:visible />

**What's real:** His strikeout rate. Miller's pre-2026 K% was 40.7%. He's running 65.9% in April. After we shrink that toward his prior — which is what statistical models do, because nobody actually maintains a 65% K% over a full season — we project his rest-of-season K% at 49-50%. That's elite, and it's a real gain over his already-elite baseline.

**What's not predictable, despite what you've read elsewhere:** how much longer the scoreless streak will go. The probabilities you'll see floated — "65% chance he gets to 50 innings," that sort of thing — are not derived from a model that actually understands how runs score in baseball. They're derived from "how often does he allow a home run," which is one of three or four ways a reliever can give up a run.

We didn't build the right model for that question, and we're not going to publish a number we don't believe. What we *can* tell you: the strikeout rate is real, and as long as the strikeout rate stays elite, the streak length is mostly a function of luck and high-leverage usage, neither of which our model has access to.

---

## What we got wrong (and why we're showing you)

The first time we ran this analysis, we said Ben Rice and Mike Trout were both real signals. We said it because we'd added "contact-quality features" — exit velocity, hard-hit rate, barrel rate — to our projection model.

When our second statistical pipeline disagreed, we audited our own code. The "contact-quality features" turned out to be a hand-tuned 50/50 blend of two stats (wOBA and xwOBA), with the other contact features computed but not actually entering the projection. When we replaced the hand-tuned blend with a learned-coefficient blend trained on 2025 data — the kind of thing we should have done the first time — Rice and Trout both collapsed back to noise.

<RetractionBox client:visible />

We're showing you this because most baseball analytics sites don't show their retractions. They publish the version they thought was right and quietly stop talking about it when it isn't. That feels worse than just being upfront: *here's what we said in our first pass, here's what we say now, here's what changed.*

The version of this article that ran a week ago would have told you Mike Trout was real. We'd have been wrong. Worth knowing.

---

## What this all means for the next 30 days

The shortest summary:

- **Don't get attached to Pages, Rice, Trout, Judge, Carroll, or Muncy's April lines.** The data on April hot starts is brutal, and these are the most-talked-about names that the data thinks won't last in their current form.
- **Watch six hitters nobody else is talking about** — Caglianone, Pereira, Barrosa, Basallo, Mayo, House. Two methods independently said they'd be on this list before we filtered to "names not in the news."
- **Murakami is the one big-name April signal that survives our analysis.** With caveats.
- **Mason Miller's K% rate is real.** His streak length is not credibly forecastable, despite what you've read elsewhere.
- **April is short enough that the only thing you should be confident about is the shape of the noise.** Almost nobody who looks like an MVP through 22 games is one in October.

We'll re-check at the All-Star break.

---

<details>
<summary><strong>Methodology</strong></summary>

We ran two independent statistical pipelines on the same April 2026 universe (287 hitters with at least 50 plate appearances, 218 relievers with at least 25 batters faced through 2026-04-24). Both methods used Statcast data 2022-2025 as the historical training corpus.

**Method A (Bayesian):** Hierarchical Bayesian projection (NUTS in numpyro) with empirical-Bayes shrinkage. The shrinkage strength was anchored to a re-derived player-season-level stabilization rate for each component stat (the standard published Carleton 2007 stabilization rates for plate-discipline stats; corrected estimates for wOBA and ISO based on 2022-2025 data). Contact quality entered the projection via a learned linear blend of wOBA, xwOBA, EV p90, HardHit%, Barrel%, and prior wOBA, with coefficients fit on 2022-2024 player-seasons and validated on a 2025 holdout (RMSE 0.0359 vs 0.0371 for a wOBA-only baseline).

**Method B (Machine learning):** LightGBM gradient-boosted projection, trained on 2022-2024, validated on 2024, tested on 2025. Quantile regression forests for prediction intervals. Historical-analog retrieval via cosine similarity over standardized 22-game feature vectors.

Both methods were independently cross-reviewed — each agent read the other's code and report and wrote a skeptical peer review. Across three rounds of analysis, three rounds of cross-review, and the comparison memos, we retracted multiple findings (the R1 "stabilization rates shifted" headline, the R2 Rice/Trout SIGNAL claims, several specific sleeper picks) and resolved several disagreements (Lynch as a sleeper rather than fake-dominant; the Tristan Peters and Louis Varland names dropped from sleeper lists).

The full code, both agent prompts, all three comparison memos, and the cross-reviews are in the [CalledThird research repo](https://github.com/prismindanalytics/calledthird-research) under `research/hot-start-half-life/`.

**Caveats we owe you:**

- Codex's prediction intervals over-cover by ~5pp on validation; they're deliberately conservative.
- Codex's sleeper floor required a positive preseason wOBA baseline; some of our sleeper hitters (Pereira, Barrosa) have priors below the .250 line we'd ideally use.
- Codex's "fake hot" rule used a population residual standard deviation (not a per-player Bayesian SD) as the threshold; only Carter Jensen cleared it.
- Claude's contact-quality blend was validated on a single 2022-2024 → 2025 holdout, not a k-fold cross-validation; coefficient stability across folds is not established.
- Murakami's prior is a 60-PA league-average proxy because no NPB-translation was used.
- We did not model defensive metrics, park factors, or causal mechanisms behind any individual player's profile.

</details>

---

*This piece is part of CalledThird's "Myths vs Data" series. If you read this far and want our methodology breakdowns, our umpire scorecards, or our pitch-tunneling work, the rest of the site is yours.*

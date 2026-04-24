# Prior CalledThird ABS Coverage

These published pieces are the editorial backdrop for the ABS Walk Spike investigation. The new article must reference them — either confirming, extending, or revising prior conclusions. Honest position management is core to CalledThird's voice.

## The most directly relevant prior piece

### "The Walk Rate Spike: Umpires or Pitchers?" — April 9, 2026
**Slug:** `the-walk-rate-spike`
**File:** `website/src/pages/analysis/the-walk-rate-spike.astro`
**Window:** 2026 Mar 27 – Apr 5 (10 days, 38K pitches) vs same window 2025
**Headline claim:** "The data says pitchers are nibbling more. The ABS zone squeeze hypothesis is rejected."

Key quantitative claims the new analysis must engage with:
- Walk rate up 0.96pp in the matched window: 8.71% (2025) → 9.67% (2026)
- **Shadow-zone called-strike rate ROSE +4pp** (65.7% → 69.7%) — umpires calling MORE borderline strikes, not fewer
- Waste-zone pitch share rose from 39.0% to 42.0% (+3.0pp)
- Heart-zone pitch share fell from 12.2% to 11.3% (-0.9pp)
- Conclusion: "Pitchers are nibbling. The behavioral change appears to be on the pitcher side, not the umpire side."

**The new article's editorial position relative to this:**
With 17 more days of data (Apr 6 – Apr 22) and a more sophisticated methodology (full 2D heat map, GAM, learned counterfactual), we will:
- **CONFIRM** if the shadow-zone CS rate is still up YoY and the zone overall didn't shrink
- **EXTEND** by adding the heat map (where exactly things changed) and the counterfactual (% attributable)
- **REVISE** if the new data shows the zone DID shrink in some region — explicit acknowledgement that our prior conclusion was based on early data and should be updated

The Apr 9 article's "pitchers nibble, not umpire squeeze" claim is testable directly against H1 of this brief. If H1 fails (no contiguous zone region shrank ≥3pp), Apr 9 is reaffirmed. If H1 passes (zone DID shrink), Apr 9 needs a meaningful update.

This is exactly the kind of position management that builds CalledThird's credibility. We were one of the first analytical voices on this; we can either own the original take or update it transparently.

## Other relevant prior pieces (citation candidates)

### "ABS Isn't Rewriting Every At-Bat. It's Repricing the Last Pitch." — April 16, 2026
**Slug:** `the-count-that-matters`
**Direct relevance to H3:** Already established that 3-2 generates 24% of ABS run value despite being only 2.8% of called pitches (8.3x leverage ratio). Late counts (2-2, 3-0, 3-1, 3-2) = 13% of pitches but 47% of value (3.6x).

**Why this matters for our article:** H3 (does 3-2 take the worst hit on walks?) is a natural extension. If 3-2 walks are up disproportionately, the run-value impact is even bigger than the raw walk-rate delta suggests because of this leverage concentration.

### "The Best ABS Challengers Are Catchers" — April 16, 2026
**Slug:** `catchers-are-better-challengers`
**Direct relevance to Reddit discourse:** Established catchers overturn 60.6% vs batters' 45.5% (OR=1.85, p<0.001). This is the same fact u/altfillischryan cited on r/baseball and used to argue "the challenge system is net adding strikes, not removing them" — a defense of the zone.

**Why this matters for our article:** Acknowledges that the challenge system itself isn't the cause of the walk spike — challenges net-add strikes. So if walks are up, the cause must be in the underlying zone calling (or pitcher behavior), not in the challenge system. Strengthens the H1 question.

### "The Anatomy of a Missed Call" — April 5, 2026
**Slug:** `anatomy-of-a-missed-call`
**Foundational piece:** 7.2% miss rate, half-inch cliff. Establishes the base rate of umpire error. Useful citation for any zone-accuracy claim.

### "Four Kinds of Zone" — April 6, 2026
**Slug:** `four-kinds-of-zone`
**Per-umpire variation:** 21pp gap in borderline strike rates. Useful context for the per-umpire round (Round 3) and as an aside in Round 1.

### "The Challenge System Is Quietly Favoring Defense" — April 6, 2026
**Slug:** `abs-challenge-strategy`
**Challenge economics:** 55% overturn, defense vs batter gap. Worth citing in passing.

### "Catcher Framing in the ABS Era" — April 5, 2026
**Slug:** `catcher-framing-abs-era`
**Counterintuitive:** Good framing may gain value in ABS era (influences batter's decision to challenge). Worth a callback if the article touches on framing.

### "After a Fight, the Zone Gets Cleaner" — April 10, 2026
**Slug:** `the-zone-after-a-fight`
**Methodology cousin:** Uses YoY heat-map deltas similar to what this round will produce. The +2.0pp accuracy improvement after fights came from the same kind of analysis. Relevant for how to present the chart.

## How to use this in the article

1. **Open with the Apr 9 piece position.** "Two weeks ago, we wrote that the walk spike was pitchers nibbling, not umpires squeezing. Now that the players have weighed in publicly, blaming the zone, here's what 17 more days of data say."

2. **Cite specifically.** When making a claim that builds on prior work, link to the prior piece by slug. Example: `[Two weeks ago we found](/analysis/the-walk-rate-spike) that shadow-zone called-strike rate had actually risen +4pp YoY...`

3. **Be explicit about updates.** If the new analysis revises the Apr 9 conclusion, say so plainly. "Our earlier conclusion was based on 10 days. With 27 days of data, the picture is more nuanced: ..."

4. **Use the count-leverage piece for the 3-2 sidebar.** If H3 holds, frame as "3-2 was already the most valuable count on the plate (we showed in [The Count That Matters](/analysis/the-count-that-matters) that it generates 8.3x its called-pitch share in run value). Now it's also where the walk spike concentrates most."

5. **Acknowledge the Reddit discourse.** "Players, fans, and analysts are all pointing in different directions. Catchers are still net-adding strikes through challenges ([as we showed](/analysis/catchers-are-better-challengers)), so the spike isn't from challenges themselves. It's from..."

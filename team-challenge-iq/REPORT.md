# Research Report: Twins vs Diamondbacks Challenge Strategy

Prepared on April 16, 2026 inside `research/team-challenge-iq/codex-analysis-2026-04-16/`.

## Executive Summary
- Normalized sample: **940 ABS challenges** from **March 27 to April 14, 2026**, with both the **Twins** and **Diamondbacks** playing **17 ABS-tracked games** in that window.
- Verdict: **publish as a contrast story, not a simple "who's best" ranking.**
- The cleanest answer is:
  - **Minnesota is the leverage hunter.**
  - **Arizona is the precision team.**
- The Twins challenged much more often:
  - **45 challenges** vs **30** for Arizona
  - **2.65 challenges per game** vs **1.76 per game**
- Arizona won more often on raw overturn rate:
  - **60.0%** vs **51.1%** for Minnesota
- But Minnesota still captured more total run value:
  - **2.162 runs** vs **1.255 runs**
- Their strategy split is stark:
  - **Minnesota late-count share**: **31.1%**
  - **Arizona late-count share**: **13.3%**
  - **Arizona catcher share**: **70.0%**
  - **Minnesota catcher share**: **53.3%**
- Arizona's profile is more conservative:
  - fewer challenges overall
  - heavier catcher ownership
  - more obvious pitches on average (**1.59"** from the edge vs **1.40"** for Minnesota)
- Minnesota's profile is more aggressive:
  - much heavier late-count exposure
  - bigger total opportunity
  - but worse conversion inside that richer mix
- The selection/execution split makes the difference explicit:
  - **Minnesota selection bonus**: **+0.267 runs**
  - **Arizona selection bonus**: **-0.283 runs**
  - **Minnesota execution bonus**: **-0.630 runs**
  - **Arizona execution bonus**: **-0.145 runs**

## Recommended Framing
- Best angle: **The Twins and Diamondbacks Are Solving ABS in Opposite Ways.**
- Alternate angle: **Minnesota Buys Leverage. Arizona Buys Certainty.**
- Avoid a simplistic "Arizona is better" or "Minnesota is better" headline. The more accurate answer is that the teams are optimizing different things.

## What Holds Up
1. **Minnesota is the volume-and-leverage side of the matchup.**
   The Twins challenge far more often and spend a much larger share of that volume in late counts, especially `2-2` and `3-2`.
2. **Arizona is the cleaner selection side.**
   The Diamondbacks use catchers more, challenge fewer pitches, and target more obvious misses on average.
3. **The teams are not separating on just one axis.**
   Minnesota wins on opportunity and total value. Arizona wins on overturn rate and cleaner role structure.
4. **Late counts are the real dividing line.**
   Minnesota ran **14 late-count challenges** and captured **1.074 runs** there.
   Arizona ran only **4 late-count challenges** and captured **0.000 runs** there.
5. **Both teams still leave value on the table.**
   Even Arizona, the cleaner precision team, came in slightly below what a league-average club would have been expected to capture from its chosen count mix. Minnesota left much more on the board because its mix was richer and harder to cash.

## What Does Not Hold Up
- A clean **"Arizona is just the smarter team"** claim. Arizona is more selective and more catcher-led, but the Twins still produced the larger value haul.
- A clean **"Minnesota cracked the code"** claim. The Twins are buying the right kinds of spots, but their execution inside that mix still lagged expectation.
- A strong claim that one strategy is already decisively superior. The early sample says the strategies are different; it does not yet prove one is the stable winner.

## Key Tables
### 1. Head-to-head summary
| Team | Games | Challenges | Challenges/game | Overturn | Total value | Value/challenge | Catcher share | Late-count share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MIN | 17 | 45 | 2.65 | 51.1% | 2.162 | 0.048 | 53.3% | 31.1% |
| AZ | 17 | 30 | 1.76 | 60.0% | 1.255 | 0.042 | 70.0% | 13.3% |

### 2. Role split
| Team | Challenger | Challenges | Overturn | Total value | Value/challenge | Mean edge |
| --- | --- | --- | --- | --- | --- | --- |
| MIN | Batter | 21 | 33.3% | 0.340 | 0.016 | 1.51" |
| MIN | Catcher | 24 | 66.7% | 1.822 | 0.076 | 1.32" |
| AZ | Batter | 9 | 44.4% | 0.261 | 0.029 | 2.53" |
| AZ | Catcher | 21 | 66.7% | 0.994 | 0.047 | 1.31" |

### 3. Count-bucket split
| Team | Bucket | Challenges | Overturn | Total value |
| --- | --- | --- | --- | --- |
| MIN | Early | 13 | 46.2% | 0.438 |
| MIN | Middle | 18 | 61.1% | 0.650 |
| MIN | Late | 14 | 42.9% | 1.074 |
| AZ | Early | 14 | 78.6% | 0.510 |
| AZ | Middle | 12 | 58.3% | 0.745 |
| AZ | Late | 4 | 0.0% | 0.000 |

## Charts
### 1. League volume-efficiency context
![Volume efficiency matrix](charts/volume_efficiency_matrix.png)

### 2. Twins vs Diamondbacks role split
![Twins vs Diamondbacks role split](charts/twins_dbacks_role_split.png)

### 3. Twins vs Diamondbacks count buckets
![Twins vs Diamondbacks count buckets](charts/twins_dbacks_count_buckets.png)

### 4. Selection vs execution matrix
![Selection execution matrix](charts/selection_execution_matrix.png)

### 5. Count-level challenge detail
![Twins vs Diamondbacks count detail](charts/twins_dbacks_count_detail.png)

## Strongest Editorial Takeaways
- **The Twins are buying more of the expensive counts.**
  Their late-count share is more than double Arizona's, and that alone changes the shape of the story.
- **The Diamondbacks are building a cleaner challenge process.**
  Their mix is more catcher-led and more conservative on edge distance, which helps explain the stronger overturn rate.
- **Minnesota's batter side is the real weakness.**
  Twins catchers went **16 for 24** on overturns and produced **1.822 runs**. Twins batters went just **7 for 21** and produced **0.340 runs**.
- **Arizona's caution has a real tradeoff.**
  The Diamondbacks' approach keeps their overturn rate high, but it also keeps them out of the highest-value late counts.
- **This is a strategy contrast, not a final leaderboard.**
  If you want a publishable story, the best promise is not "who is best" but "what kind of ABS team each club is becoming."

## Method
1. Loaded `all_challenges_detail.json` and filtered to **March 27 through April 14, 2026**.
2. Defined the **challenging team** as:
   - `team_batting` for batter challenges
   - `team_fielding` for catcher and pitcher challenges
3. Built count buckets as:
   - **early**: `0-0`, `0-1`, `1-0`
   - **middle**: `0-2`, `1-1`, `1-2`, `2-0`, `2-1`
   - **late**: `2-2`, `3-0`, `3-1`, `3-2`
4. Counted games from `all_games_with_abs.json` using team appearances as home or away club.
5. Built selection and execution bonuses using league-average **count-specific run value per challenge**:
   - **selection bonus** = value implied by the team's chosen count mix vs league-average mix
   - **execution bonus** = actual value minus the value expected from that chosen mix

## Caveats
- This is still a small early-season sample for team strategy work.
- Arizona's late-count sample is only **4 challenges**, so that zero-value result is directionally useful but not yet stable.
- The run-value baseline comes from the repo's existing challenge log rather than a new independent RE288 rebuild.
- This report is centered on **Minnesota vs Arizona**, not a complete league-rank adjudication.

## Bottom Line
- **Best publishable claim**: the Twins and Diamondbacks are pursuing distinct ABS strategies, with Minnesota prioritizing leverage and Arizona prioritizing cleaner, catcher-led precision.
- **Best one-sentence framing**: *Minnesota is challenging like a team that wants every high-stakes shot it can buy, while Arizona is challenging like a team that would rather take fewer swings and make them cleaner.*
- **What not to claim**: that either team has already solved ABS in a final way. The stronger claim is that the contrast in strategy is already visible and meaningful.

# Research Report: The Catcher Edge in ABS

Prepared on April 15, 2026 inside `research/abs-catcher-edge/codex-analysis-2026-04-15/`.

## Executive Summary
- Normalized sample: **940 ABS challenges** from **March 27 to April 14, 2026**, including **429 batter challenges**, **503 catcher challenges**, and **8 pitcher challenges**.
- Verdict: **publish**. The strongest signal in the `team-challenge-iq` data is not just team variance. It is that **catchers are materially better ABS challengers than batters**.
- The raw gap is large: catchers posted a **60.6%** overturn rate versus **45.5%** for batters. That yields a raw catcher-vs-batter odds ratio of **1.85** with a chi-square p-value of **4.96e-06**.
- Catchers are not just farming easier misses. Their challenged pitches were actually **closer to the edge on average**: **1.19 inches** from the border versus **1.44 inches** for batters.
- The role edge survives a simple control model. In a logistic regression with **count bucket** and **edge distance** included, the catcher indicator still carried an odds ratio of **1.66** with a **95% CI of 1.13 to 2.43** and **p = 0.009**.
- Catchers outperform in every count bucket:
  - **Early**: 67.1% overturn vs 48.6%
  - **Middle**: 58.1% vs 48.2%
  - **Late**: 52.5% vs 37.8%
- The value gap widens as leverage rises. In late counts, catchers captured **0.119 runs per challenge** versus **0.090** for batters.
- The biggest count-level catcher edges show up at **0-1 (+30.2 pp)**, **0-0 (+26.2 pp)**, **0-2 (+24.7 pp)**, **2-2 (+24.1 pp)**, and **3-2 (+14.1 pp)**.
- Important caveat: team role mix alone is not the full answer. Among teams with at least 20 challenges, **catcher share has essentially no correlation with total value captured (`r = 0.024`)**.

## Recommended Framing
- Best angle: **The Best ABS Challengers Aren't Batters. They're Catchers.**
- Alternate angle: **Catchers Are Winning MLB's ABS Learning Curve.**
- Narrower strategic angle: **Seeing the pitch is more valuable than simply owning the challenge.**

## What Holds Up
1. **Catchers beat batters on the headline metric.**
   The overturn gap is too large to dismiss as noise in the full March 27 to April 14 sample.
2. **Catchers are doing it on tighter decisions.**
   Their average challenged pitch was roughly a quarter-inch closer to the edge, which cuts against the easy explanation that catchers only wait for obvious misses.
3. **The catcher edge survives controls.**
   Once count bucket and edge distance are included, the catcher role still carries a statistically significant advantage.
4. **Late counts magnify the gap.**
   The role difference is not just an early-count housekeeping phenomenon. In late counts, catchers still win more often and extract more run value per decision.
5. **The team-policy takeaway is narrower than the role takeaway.**
   Some catcher-heavy teams are strong, but role share by itself does not explain the team leaderboard. The story is **catcher skill**, not a universal one-line policy rule.

## What Does Not Hold Up
- A simplistic **"just let catchers challenge everything"** claim. Team-level catcher share does not reliably map to total value or overturn rate.
- A pure **"catchers only win because they challenge obvious misses"** claim. The edge-distance data cut the other way.
- A count-by-count absolutist claim. Batters did run better in a few specific counts such as **1-0**, **2-0**, and **3-1**, though those pockets are smaller and some are sample-fragile.

## Key Tables
### 1. Overall role comparison
| Challenger | Challenges | Overturn | 95% CI | Runs / challenge | Avg edge distance |
| --- | --- | --- | --- | --- | --- |
| Batter | 429 | 45.5% | 40.8% to 50.2% | 0.050 | 1.44" |
| Catcher | 503 | 60.6% | 56.3% to 64.8% | 0.062 | 1.19" |

### 2. Count-bucket split
| Bucket | Batter overturn | Catcher overturn | Batter value | Catcher value |
| --- | --- | --- | --- | --- |
| Early | 48.6% | 67.1% | 0.026 | 0.034 |
| Middle | 48.2% | 58.1% | 0.044 | 0.056 |
| Late | 37.8% | 52.5% | 0.090 | 0.119 |

### 3. Biggest catcher count edges
| Count | Batter n | Catcher n | Catcher minus batter overturn | Catcher minus batter value |
| --- | --- | --- | --- | --- |
| 0-1 | 29 | 56 | +30.2 pp | +0.020 |
| 0-0 | 80 | 115 | +26.2 pp | +0.021 |
| 0-2 | 22 | 32 | +24.7 pp | +0.023 |
| 2-2 | 52 | 37 | +24.1 pp | +0.026 |
| 3-2 | 47 | 50 | +14.1 pp | +0.078 |

## Charts
### 1. Overall role summary
![Overall role summary](charts/overall_role_summary.png)

### 2. Count-bucket profiles
![Bucket role profiles](charts/bucket_role_profiles.png)

### 3. Edge-bucket profiles
![Edge role profiles](charts/edge_role_profiles.png)

### 4. Count-level catcher advantage
![Count role advantage](charts/count_role_advantage.png)

### 5. Team role-mix context
![Team role mix](charts/team_role_mix.png)

## Strongest Editorial Takeaways
- **Catchers look like the best natural ABS operators in the current system.**
  They win more often, create more value per challenge, and do it on slightly tougher pitches.
- **The role gap is not limited to safe counts.**
  The late-count numbers still favor catchers, which is the most important place for a strategic edge to survive.
- **3-2 is the best example of catcher value under pressure.**
  Catchers matched batter volume almost one-for-one at full counts (**50 vs 47 challenges**) but more than doubled the value haul (**8.280 runs vs 4.140**) while challenging much closer pitches on average (**0.94" vs 1.88"**).
- **Early counts may be where catcher expertise shows up most clearly.**
  At **0-0**, **0-1**, and **0-2**, catcher overturn rates were roughly **25 to 30 percentage points** better than batter rates.
- **Team implementation still matters.**
  A catcher-heavy mix is not a cheat code. For example:
  - **LAD** is heavily catcher-led (**73.3% catcher share**) and strong.
  - **MIA** is also catcher-led (**70.8%**) and productive.
  - But **CWS** is catcher-led too (**62.5%**) and still struggled.
  - Meanwhile **NYY** leaned batter-heavy (**35.9% catcher share**) and still performed well overall.

## Method
1. Loaded `all_challenges_detail.json` and filtered to **March 27 through April 14, 2026**.
2. Kept `challenge_value` as the repo baseline, filling nulls with zero where unsuccessful challenges were stored as missing rather than explicit zero.
3. Defined count buckets as:
   - **early**: `0-0`, `0-1`, `1-0`
   - **middle**: `0-2`, `1-1`, `1-2`, `2-0`, `2-1`
   - **late**: `2-2`, `3-0`, `3-1`, `3-2`
4. Compared batter and catcher challenges on:
   - overturn rate
   - mean run value per challenge
   - average `edge_distance_in`
5. Estimated a logistic model:
   `overturned ~ catcher_flag + edge_distance_in + C(count_bucket)`
6. Built team role-mix context using the actual **challenging team**:
   - `team_batting` when the challenger is the batter
   - `team_fielding` when the challenger is the catcher

## Caveats
- This is still an early-season sample. The role effect is strong already, but some individual count splits remain small.
- Pitchers were effectively irrelevant in this window, with just **8** challenges, so this report is really about **batters vs catchers**.
- The control model is intentionally simple. It adjusts for count bucket and edge distance, not every possible game-state variable.
- The team scatter should be read as context, not as proof that role share has no causal effect. It only shows that share alone does not explain outcomes cleanly in this sample.

## Bottom Line
- **Best publishable claim**: catchers are materially better ABS challengers than batters in the current 2026 sample, and that edge remains after controlling for where and when the pitch happened.
- **Best one-sentence framing**: *If ABS is creating a new baseball skill, the early evidence says the best practitioners are the players who receive the pitch, not the ones who take it.*
- **What not to claim**: that every team should blindly hand all challenges to catchers. The cleaner claim is that catchers appear to see challenge-worthy pitches better, but organizations still need a full decision framework around that edge.

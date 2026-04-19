# Research Report: The ABS Count Economy

Prepared on April 15, 2026 inside `research/abs-count-leverage/codex-analysis-2026-04-15/`.

## Executive Summary
- Normalized sample: **70,876 matched-window pitches in 2025**, **73,513 matched-window pitches in 2026**, **37,528 called pitches in the 2026 window**, and **940 ABS challenges from March 27 to April 14, 2026**.
- The raw challenge log contains **970** challenges through April 14. This report excludes the **30 March 26** challenges from pitch-normalized metrics because the paired local pitch-level parquet window starts on March 27.
- Verdict: **publish, but make leverage concentration the headline.** The count grid did move slightly toward hitter counts, but the editorially strong signal is that **ABS value is overwhelmingly concentrated in late counts, especially 3-2 and 2-2**.
- The structural shift is real but small: **1-0 (+0.30 pp), 2-0 (+0.22 pp), 3-0 (+0.22 pp), 3-1 (+0.20 pp)** all ticked up in 2026, while **0-1 (-0.35 pp)** and **1-2 (-0.21 pp)** ticked down. Mean pitches per plate appearance moved only **3.910 -> 3.939**, and the share of **6+ pitch** plate appearances only moved **20.9% -> 21.3%**.
- The full count is the real ABS center of gravity: **3-2 accounts for just 2.84% of called pitches, but 10.32% of challenges and 23.56% of captured run value.** It generates **90.99 challenges per 1,000 called pitches** and **11.65 runs per 1,000 called pitches**, both the clear highs.
- Late counts as a bucket (**2-2, 3-0, 3-1, 3-2**) represent only **13.0% of called pitches**, but they generate **25.7% of challenges** and **47.3% of all captured run value**.
- Catchers still drive the system, but the ownership changes by count. At **0-0**, catchers take **58.1%** of challenges and post a **68.7%** overturn rate. At **2-2**, batters take **57.1%** of the volume, but catchers still nearly match the run value because they convert more efficiently. At **3-2**, volume is basically even and catchers still generate the larger share of value (**8.28 runs vs 4.14**).
- Team strategy splits cleanly into two layers:
  - **Selection bonus**: teams like **ATH, BOS, MIN, SEA, ATL** are steering more of their challenge volume into richer late counts.
  - **Execution bonus**: teams like **BAL, NYY, DET, SF, MIA** are outperforming what their chosen count mix would normally return.

## Recommended Framing
- Best angle: **The ABS Count That Actually Matters Is 3-2.**
- Alternate angle: **ABS Isn't Rewriting Every At-Bat. It's Repricing the Last Pitch.**
- Secondary team angle: **Some teams buy better count mix. Others cash those counts better.**

## What Holds Up
1. **The count-distribution story is context, not the lede.**
   There is a slight 2026 tilt toward ball-heavy counts, and the chi-square test is significant (`p = 6.36e-05`), but the biggest single-state shift is only **+0.30 percentage points**. That is not a strong enough editorial payoff on its own.
2. **Run value is more concentrated than overturn rate.**
   The count with the best overturn rate is **3-0 (80.0%)**, but that count is rare and only returns **1.44 runs per 1,000 called pitches**. The count with the most run leverage is **3-2**, despite only a **39.2%** overturn rate.
3. **The near-edge full count is the premium ABS asset.**
   Just **21** full-count challenges within **1 inch** of the edge produced **8.97 runs**, or about **17.0% of all captured run value** in the March 27 to April 14 sample.
4. **Early-count challenge volume is easy to see but not very valuable.**
   The early-count bucket (**0-0, 0-1, 1-0**) absorbs **56.7% of called pitches** and **38.9% of challenges**, but only **21.2% of total run value**.
5. **Teams can create edge two different ways.**
   The strategy matrix shows that richer count mix helps, but actual leaderboard position is still driven more by how well teams convert inside those counts than by count selection alone.

## What Does Not Hold Up
- A sweeping **"ABS has structurally changed every plate appearance"** claim. The matched-window movement is measurable but too small to carry a headline by itself.
- A simplistic **"the smartest teams just challenge more late counts"** leaderboard. That explains part of the mix, but clubs like **BAL** and **NYY** are getting most of their advantage from execution, not just selection.
- A pitcher-driven challenge story. There are only **8 pitcher-initiated challenges** in the normalized sample.

## Key Tables
### 1. Count leverage leaderboard
| Count | Called-share | Challenge-share | Value-share | Challenges / 1k called | Runs / 1k called | Overturn |
| --- | --- | --- | --- | --- | --- | --- |
| 3-2 | 2.84% | 10.32% | 23.56% | 90.99 | 11.65 | 39.2% |
| 2-2 | 5.64% | 9.68% | 16.75% | 42.97 | 4.17 | 41.8% |
| 3-1 | 2.22% | 4.15% | 4.64% | 46.76 | 2.94 | 51.3% |
| 2-1 | 4.38% | 6.49% | 7.57% | 37.13 | 2.43 | 52.5% |
| 3-0 | 2.31% | 1.60% | 2.36% | 17.34 | 1.44 | 80.0% |
| 0-0 | 33.44% | 21.06% | 10.63% | 15.78 | 0.45 | 58.1% |

### 2. Team strategy examples
These are **challenging-team** metrics, not batting-team-only metrics.

| Team | Challenges | Selection bonus | Execution bonus | Late-count share | Actual value |
| --- | --- | --- | --- | --- | --- |
| ATH | 41 | +0.396 | +0.091 | 34.1% | 2.787 |
| BOS | 16 | +0.380 | -0.283 | 56.2% | 0.995 |
| MIN | 45 | +0.267 | -0.630 | 31.1% | 2.162 |
| BAL | 29 | +0.005 | +0.858 | 20.7% | 2.489 |
| NYY | 39 | -0.269 | +0.813 | 17.9% | 2.732 |
| DET | 33 | -0.063 | +0.785 | 30.3% | 2.573 |

How to read it:
- **Selection bonus** asks whether a team's count mix is richer than league average.
- **Execution bonus** asks whether the team outperformed the run value a league-average club would have captured from that same mix.

## Charts
### 1. Matched-window count shift
![Count distribution delta](charts/count_distribution_delta.png)

### 2. Called-pitch share vs challenge share vs value share
![Pressure and value by count](charts/pressure_value_by_count.png)

### 3. Value density and overturn context
![Value density by count](charts/value_density_by_count.png)

### 4. Role ownership by count
![Role mix by count](charts/role_mix_by_count.png)

### 5. Edge-distance heatmap
![Edge value heatmap](charts/edge_value_heatmap.png)

### 6. Team selection vs execution matrix
![Team selection vs execution](charts/team_selection_execution.png)

## Strongest Editorial Takeaways
- **3-2 is the premium challenge count.** It is challenged far more often than its called-pitch frequency would suggest, and it produces by far the most run value per opportunity.
- **2-2 is the supporting co-star.** It does not reach full-count density, but it still contributes nearly **one-sixth** of all captured value.
- **0-0 is not where the action is.** It dominates visual volume and public memory, but it is a relatively poor source of value.
- **Catchers still run the ecosystem**, especially early, but hitter ownership rises in the highest-friction two-strike counts.
- **Team-level headlines should be careful.** The most publishable version is not "Team X solved ABS." It is "some clubs are buying better counts, but the best totals still come from who cashes those counts."

## Method
1. Loaded the full matched local 2025 window from `statcast_2025_mar27_apr14.parquet`.
2. Built the 2026 matched local window by concatenating the daily March 27 to April 5 parquet files with `statcast_2026_apr06_14.parquet`.
3. Filtered `all_challenges_detail.json` to March 27 through April 14, 2026 so pitch-normalized metrics use the same local pitch window.
4. Filled missing `challenge_value` with zero. In the source JSON, many unsuccessful challenges store null rather than explicit zero.
5. Defined the **challenging team** as:
   - `team_batting` when the challenger is a batter
   - `team_fielding` when the challenger is a catcher or pitcher
6. Normalized count pressure by **called pitches** (`ball` + `called_strike`) rather than all pitches, because those are the direct challenge opportunities.
7. Computed team **selection bonus** as the gap between a team's chosen count mix and a league-average count mix.
8. Computed team **execution bonus** as the gap between actual captured value and the value a league-average club would have been expected to capture from that same count mix.
9. Used Wilson 95% intervals for count-level overturn-rate context in the value-density chart.

## Caveats
- The pitch-normalized view excludes **March 26, 2026** only because the local pitch-level parquet window in the paired research folder begins on March 27.
- `challenge_value` comes from the repo's existing challenge log and is treated as the analysis baseline here; this report does not independently rebuild the RE288 model from first principles.
- This is still an early-season sample. The leverage ranking by count is strong already, but individual team estimates can move meaningfully as volume grows.
- The called-pitch opportunity denominator is a pragmatic one. It captures observed opportunities, not every theoretical miss that might have been challengeable with different behavior.

## Bottom Line
- **Best publishable claim**: ABS is not creating equal pressure across the count grid. It is concentrating value in the last serious decision points of an at-bat, especially **3-2** and **2-2**.
- **Best one-sentence framing**: *The count that really matters in MLB's ABS era is not 0-0, where the noise is loudest, but 3-2, where a small slice of called pitches is generating nearly a quarter of the system's run value.*
- **What not to claim**: that teams have already solved ABS in a stable, repeatable way. The cleaner claim is that teams are separating into different strategies, but execution inside the same counts still matters more than pure mix.

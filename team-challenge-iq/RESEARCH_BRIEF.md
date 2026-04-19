# Research Brief: Team Challenge IQ — Who's Winning the ABS Meta-Game?

## Hypothesis
There is meaningful, statistically significant variance in how MLB teams use the ABS challenge system in 2026 — not just in volume, but in **quality per challenge**. A subset of teams (Twins, Diamondbacks, Reds) are extracting measurable run value from better decision-making, while others are wasting challenges on low-probability calls. This is a new strategic dimension that distinguishes organizations.

## The Hook
ESPN has been [actively tracking team-level ABS performance](https://www.espn.com/mlb/story/_/id/48305211/2026-mlb-abs-challenge-system-tracker-team-player-rankings). Sports Illustrated [just called the Twins "the early kings of MLB's new ABS Challenge System"](https://www.si.com/mlb/twins/onsi/minnesota-twins-news/twins-are-early-kings-of-mlbs-new-abs-challenge-system). The narrative is out there — but nobody has run our version: RE288-weighted run value per challenge, with proper sample-size caveats and a Wilson CI visualization.

This is the Moneyball angle: **who figured out the new meta first?**

## Pillar
ABS & Umpires (already overrepresented — but this is a genuinely new sub-angle that builds on, not duplicates, our April 6 challenge strategy article).

## Data Available

### In `data/` folder:
- **`all_challenges_detail.json`** — 970 individual challenges across 242 games (Mar 26 – Apr 14, 2026). Full detail: team_batting, team_fielding, challenger, count, inning, plate_x, plate_z, zone, edge_distance_in, in_zone, initial_call, final_call, overturned, challenge_value, catcher, batter, pitcher, umpire.
- **`all_games_with_abs.json`** — 242 game-level records with challenge counts and outcomes.
- **`abs_leaderboard_catchers.json`** — 54 catchers ranked by challenges/overturns/value.
- **`abs_leaderboard_counts.json`** — 12 count states with challenge breakdowns.
- **`abs_leaderboard_batters.json`** — 129 batters with challenge data.
- **`savant_abs_batting_team.csv`** — Baseball Savant's team-level leaderboard with their expected-vs-actual metrics (use as cross-check).

### Key fields per challenge:
- `team_batting`, `team_fielding` — which team was at bat / fielding
- `challenger` — who initiated ("batter" / "catcher" / "pitcher")
- `overturned` — boolean
- `challenge_value` — our RE288-weighted run value
- `edge_distance_in` — how far from the zone edge (proxy for "obvious" vs "close")
- `in_zone` — was the pitch actually in the rulebook zone
- `initial_call` / `final_call` — Strike/Ball before and after ABS

## Key Facts (from web research)
- 970 total challenges through Apr 14, 2026 (~54% overturn rate league-wide)
- Twins lead in volume (42 challenges, 60% overturn per SI)
- Diamondbacks lead in efficiency (74% win rate per ESPN)
- Chad Whitson led umpires in overturns against him early
- ABS is making games ~4 min longer on average

## Analysis Steps

### 1. Team Challenge Volume & Outcome
- For each of 30 teams: total challenges initiated (as batter or fielder), overturn rate, total run value gained/lost (sum of `challenge_value` for overturned challenges, signed by which side benefits).
- Rank by: volume, overturn %, run value captured, run value per game played.
- Wilson 95% CIs on overturn rates (small samples matter — some teams have ≤10 challenges).

### 2. The Volume-Efficiency Matrix
- Scatter plot: X = total challenges (volume), Y = overturn rate (quality). Size = run value captured. Highlight Twins (high volume), Diamondbacks (high efficiency), and any outliers.
- This is the signature visualization — the "Moneyball chart" of ABS strategy.

### 3. Who's Challenging? (Role Breakdown)
- Batter vs catcher vs pitcher: per team, what's the role split? Do teams with catcher-heavy challenges (framing-aware) do better than teams leaning on batters?
- Compare to the 54-catcher leaderboard — which teams have catchers in the top 20 overturn rate?

### 4. Edge-Distance Profile
- For each team's challenges, what's the average `edge_distance_in`? Teams that only challenge obvious misses (>1" off the edge) should have high overturn rates but limited volume. Teams that gamble on close calls (<0.5") have more volume but lower overturn rates.
- This is a proxy for "challenge discipline."

### 5. Count-State Selection
- Using `abs_leaderboard_counts.json` + per-team breakdown: which teams over-index on high-leverage counts (2-strike, 3-ball)? RE288 weights amplify the value of overturns in those counts.
- Expected run value of a team's challenge distribution vs actual.

### 6. Head-to-Head: Twins vs Diamondbacks
- Mini case study contrasting the two public narratives: Twins (high volume, solid overturn) vs Diamondbacks (high precision, lower volume). Which is producing more run value? What does their challenge distribution look like by count / role / edge?

## Kill Criteria
- If team volume is too small for most clubs to get Wilson CIs tighter than ±15pp, the team-by-team comparison is noise — we pivot to "early trends + warning" story.
- If run-value differences between top and bottom teams are <1 run per 10 games, it's real but too small for a headline.
- If the Twins/Diamondbacks narrative doesn't hold up in our data (e.g., Diamondbacks aren't actually leading), we pivot to "the real leaders nobody's talking about."

## Article Framing
- **If confirmed:** "Challenge IQ: Which Teams Are Winning the ABS Meta-Game"
- **If mixed:** "The Twins Are Leading on Volume — But [X team] Is Actually Winning on Run Value"
- **If rejected:** "No Team Has Cracked the ABS Code Yet — Here's What They're Missing"

Lead with the volume-efficiency matrix. Name names. The Diamondbacks/Twins narrative already has traction — we either confirm it rigorously or correct it.

## Methodological Notes
- **Sample size discipline**: All percentages need Wilson CIs. Highlight where n < 20 with a warning icon.
- **Multiple comparisons**: With 30 teams, ~1-2 will appear "significantly different" by chance. Use Bonferroni or similar — or just be honest about the false-positive risk.
- **Run value direction**: When a team CHALLENGING wins → positive value for them. When a team is CHALLENGED AGAINST and loses → negative value for them. Sum both sides.
- **Use rulebook zone expanded by one baseball radius** (1.45" / 12 = 0.121 ft) for `in_zone` comparisons — match the ABS definition.

## Output
- Charts: team volume-efficiency matrix, run-value leaderboard, role breakdown bars, edge-distance distribution by team, count-selection heatmap, Twins-vs-Diamondbacks comparison.
- Summary: top 5 / bottom 5 tables with CIs, key differentiators.
- Recommendation: publish / revise / wait.

# CB Bucknor: The Data Behind Baseball's Most Controversial Umpire — Research Proposal

## Context
CB Bucknor is currently trending nationally after the ABS challenge system exposed his calls in early 2026. In a March 28 BOS@CIN game, 8 of his calls were challenged and 6 overturned — including back-to-back called third strikes both reversed. Yahoo Sports, Fox News, and multiple outlets have covered the story. A 2010 ESPN survey of 100 active players named him the worst umpire in MLB. The ABS system is now quantifying what fans have believed for years.

CalledThird has unique data to tell this story rigorously:
- 2025 full-season zone classification (4,174 pitches, 28 games)
- 2026 game-by-game data with ABS challenge outcomes
- The BOS@CIN game as #1 on our Umpire Impact leaderboard (4.08 wOBA)
- Zone style quadrant: "Tight Struggler" (low accuracy + conservative borderline zone)

## Research Questions

### Q1: How bad is Bucknor compared to the league?
- **2025**: 91.02% accuracy — 3rd worst of 83 qualified umpires (only Bruce Dreckman and Laz Diaz worse)
- 13.4 wrong calls per game (league mean: 10.9)
- Average miss distance: 1.34" (highest in MLB — when he misses, he misses by more)
- Tasks for researchers: Verify these ranks. Compute Wilson CIs on his accuracy vs league mean. Is the gap statistically significant?

### Q2: What kind of umpire is he?
- Classified as "Tight Struggler" — low accuracy AND conservative zone
- BSR 41% (below median 42.7%) — he doesn't call borderline pitches as strikes
- FS% 43.5% — when he's wrong, it's slightly more missed strikes than false strikes
- 212 missed strikes vs 163 false strikes in 2025
- Tasks: Compare his zone shape to the 3 other quadrants. Is his profile consistent across seasons? Is there a specific part of the zone where he misses most (horizontal vs vertical edge)?

### Q3: What happened in the BOS@CIN game (March 28, 2026)?
- Our data: 90.6% accuracy, 21 missed calls, 4.08 total challenge value
- 8 ABS challenges, 6 overturned (75% overturn rate vs league 55%)
- Highest single-game umpire impact in our 2026 database
- Game decided 6-5 (CIN won) — umpire impact potentially exceeded margin of victory
- Tasks: Reconstruct the game pitch-by-pitch. Which calls were challenged? What were the count states? How much run value shifted? Did the errors systematically favor one team?

### Q4: Has the ABS system changed his behavior?
- 2025: 91.02% accuracy across 28 games
- 2026 early data: Need to compare (do we have multiple 2026 games?)
- National reporting says his calls were overturned 78% in one game (Rays-Brewers)
- Tasks: Compare his 2025 and 2026 accuracy, BSR, and false strike patterns. Is there evidence of adjustment or regression?

### Q5: How does he compare to the best umpire?
- Bucknor: 91.02%, 13.4 wrong/game, 1.34" miss distance
- Edwin Jimenez (best): 94.57%, 7.7 wrong/game, 1.01" miss distance
- The gap: 3.55pp accuracy, 5.7 more wrong calls per game, 0.33" farther off on misses
- Tasks: Compute the run-value impact of this gap. Over a season, how many more runs does Bucknor's error rate shift compared to the best umpire?

## Data Available
| Source | Description | Location |
|--------|-------------|----------|
| 2025 umpire personality | 83 umpires, full season | `archive/published-research/analysis-abs/data/umpire_personality_2025_classified.json` |
| 2025 umpire profiles | KV data with zone style | `website/pipeline/nightly/data/umpire_2025_profiles.json` |
| 2026 game reports | D1 `game_reports` table | Cloudflare D1 (query via API) |
| 2026 ABS challenges | D1 `game_abs_challenges` table | Cloudflare D1 (query via API) |
| 2026 called pitches | D1 `called_pitches` table | Cloudflare D1 (query via API) |
| 2026 all pitches | D1 `all_pitches` table | Cloudflare D1 (query via API) |
| BOS@CIN game report | calledthird.com | `/report?date=2026-03-28&game=bos-vs-cin` |

## External Sources for Validation
- [Yahoo Sports: Bucknor's worst overturned calls](https://sports.yahoo.com/articles/c-b-bucknor-worst-overturned-131750084.html)
- [Yahoo Sports: 6 calls overturned in Reds game](https://sports.yahoo.com/articles/c-b-bucknor-overturned-calls-180940465.html)
- [SF Today: ABS exposes worst umpire](https://nationaltoday.com/us/ca/san-francisco/news/2026/04/02/mlbs-new-abs-challenge-system-exposes-umpire-cb-bucknor-as-worst-in-baseball/)
- [Umpire Scorecards](https://umpscorecards.com) — independent umpire accuracy tracking
- [Baseball-Reference CB Bucknor](https://www.baseball-reference.com/bullpen/C._B._Bucknor) — career history
- Wikipedia: 2010 ESPN player survey ranking him worst

## Proposed Article Angle
**"CB Bucknor by the Numbers"** — not a hit piece, but a rigorous data profile. The story:

1. **The reputation** (brief): Fans have believed he's the worst for years. The 2010 survey. The viral clips. Now ABS is measuring it.
2. **The 2025 baseline**: 3rd worst accuracy in MLB. Tight Struggler zone. Highest average miss distance. These aren't anecdotes — they're 4,174 pitches.
3. **The BOS@CIN game**: Pitch-by-pitch reconstruction. 21 missed calls, 4.08 wOBA shifted, game decided by 1 run. The single highest-impact umpire game in our database.
4. **The zone shape**: Where exactly does he miss? Horizontal edges? Vertical? Is it systematic or random?
5. **The comparison**: Bucknor vs Jimenez (best). The gap in run value over a season.
6. **The honest question**: Is this a skill issue (fixable with training) or a limitation (some umpires are just less accurate)? The data alone can't answer this, but it can frame the question.

## Editorial Guidelines
- **Tone**: Respectful and data-driven. Not mocking. Bucknor is a professional umpire who has worked MLB for 28+ years. The data tells the story — we don't need to editorialize.
- **Fairness**: Show his BEST games alongside his worst. Note games where he was above league average. Include Wilson CIs showing the uncertainty in his stats.
- **Context**: He's 3rd worst, not the absolute worst. Dreckman and Diaz are below him. The gap between worst and best is 4.5pp — significant but not as extreme as viral clips suggest.
- **ABS framing**: The ABS system isn't punishing Bucknor specifically — it's measuring all umpires equally. He just happens to be at the tail of the distribution.

## Analysis Tasks for Independent Researchers

### Researcher 1: Statistical Profile
1. Verify Bucknor's 2025 ranks (accuracy, wrong calls/game, miss distance) against the full umpire dataset
2. Compute Wilson 95% CIs on his accuracy vs league mean — is the gap significant?
3. Analyze his zone shape: where on the plate does he miss most? Use the 4,174 called pitches to build a miss-rate heatmap by zone region
4. Compute per-season run-value impact: how many more runs does Bucknor shift per game vs the average umpire? vs the best umpire?
5. Check for count-state bias: is he worse in high-leverage counts (3-2, 2-2)?

### Researcher 2: Game-Level Analysis
1. Reconstruct the BOS@CIN March 28 game from our called_pitches and game_abs_challenges data
2. For each wrong call: what was the count, who was batting, what was the pitch type, how far from the zone edge?
3. Compute directional impact: did the errors systematically favor one team? (false strikes hurt batter = help pitcher's team; missed strikes = opposite)
4. Compare BOS@CIN to his other 2026 games — is it an outlier or typical?
5. If external ABS data is available, validate our 6/8 overturn count against what Yahoo/Savant reports

## Kill Criteria
- If Bucknor's accuracy is within the normal range (not statistically different from league mean at 95% CI), the "worst umpire" framing doesn't hold
- If the BOS@CIN game is an extreme outlier and his other games are average, the story becomes "one bad game" not "persistent pattern"
- If we can't reconstruct the game pitch-by-pitch with confidence, the game-level section should be cut

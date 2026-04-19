# Research Brief: Where ABS Challenges Actually Carry the Most Leverage

## Hypothesis
ABS challenge value is not spread evenly across the count grid. The late counts that already feel tense to players and fans, especially **3-2** and **2-2**, are where challenges create a disproportionate share of run value even when overturn rates are not the highest there. Teams can improve their ABS results either by steering challenge volume into those richer counts or by executing better once they get there.

## Why This Follows the Latest Two Folders
- `count-distribution-abs/` asks whether ABS changed the structure of plate appearances.
- `team-challenge-iq/` asks which teams are winning the ABS decision game.
- This brief bridges them: **if the count grid is only moving a little, the real question is where the challenge system matters most inside that grid.**

## Core Questions
1. Which count states generate the most challenge volume relative to how often those counts produce called pitches?
2. Which counts generate the most run value per called-pitch opportunity?
3. Are the best counts for overturn rate the same as the best counts for run value?
4. How does challenge ownership shift by count: batter, catcher, or pitcher?
5. Which teams are buying better count mix, and which teams are simply executing better than expected within their chosen counts?

## Data Available
- `research/count-distribution-abs/data/statcast_2025_mar27_apr14.parquet`
- `research/count-distribution-abs/data/2026-03-27.parquet` through `2026-04-05.parquet`
- `research/count-distribution-abs/data/statcast_2026_apr06_14.parquet`
- `research/team-challenge-iq/data/all_challenges_detail.json`

## Planned Output
- A reproducible analysis package with report, tables, and charts
- A count-level leverage table
- A role-by-count breakdown
- A team strategy matrix separating count selection from execution
- Publication guidance on whether the best editorial angle is structural change, leverage concentration, or team strategy

## Kill Criteria
- If 3-2 and 2-2 do not materially over-index on value once we normalize by called-pitch opportunity, the leverage thesis weakens.
- If team strategy differences collapse after adjusting for count mix, the team angle becomes sidebar material rather than the main piece.
- If the count-distribution shift itself is large enough to dominate the story, that becomes the headline and leverage becomes secondary context.

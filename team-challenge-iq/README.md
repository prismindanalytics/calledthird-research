# Team Challenge IQ — Twins vs Reds

**Published article:** [Minnesota Buys Leverage. Cincinnati Buys Certainty.](https://calledthird.com/analysis/twins-vs-reds-abs)

## The Question

Which teams are winning the ABS meta-game — and how?

## Key Findings

1. **Minnesota leads volume** — 52 challenges, 3.2/game, 30.8% in late counts (where value concentrates).
2. **Cincinnati leads efficiency** — 27 challenges, 74.1% overturn rate (highest in MLB), 91.7% overturn in early counts (11/12).
3. **These are opposite strategies** — volume × leverage vs selective precision. Both extract value; neither is clearly better.
4. **Cincinnati's batters are unusually good** — 71.4% overturn rate vs league batter average of 45.5%.
5. **Minnesota's batter side is the weak link** — 40% overturn rate drags down the team.
6. **Selection vs execution:** Minnesota wins on count selection (+0.267), loses on execution (-0.630). Cincinnati neutral on both but consistent.

## Files

| File | Description |
|------|-------------|
| `RESEARCH_BRIEF.md` | Hypothesis and team-level metrics spec |
| `REPORT.md` | Full head-to-head breakdown |
| `analyze.py` | Team-level aggregation script |

## Methodology

- **Data:** 940 ABS challenges through Apr 14, 2026 from CalledThird D1
- **Challenging team:** `team_batting` for batter challenges, `team_fielding` for catcher challenges
- **Selection bonus:** gap between team's count mix and league-average mix, valued at league-average per-count rates
- **Execution bonus:** actual value minus expected value from chosen mix

## Citation

```
CalledThird (2026). "Minnesota Buys Leverage. Cincinnati Buys Certainty."
https://calledthird.com/analysis/twins-vs-reds-abs
```

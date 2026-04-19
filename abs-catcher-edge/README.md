# The Best ABS Challengers Are Catchers

**Published article:** [The Best ABS Challengers Are Catchers](https://calledthird.com/analysis/catchers-are-better-challengers)

## The Question

Who is better at ABS challenges — batters or catchers?

## Key Findings

1. **Catchers overturn 60.6% of challenges vs batters' 45.5%** (chi-square p = 4.96×10⁻⁶, OR = 1.85).
2. **Catchers challenge closer pitches** — avg 1.19" from edge vs batters' 1.44" — and still win more often.
3. **Logistic regression controlling for count bucket and edge distance:** catcher OR = 1.66 [1.13, 2.43], p = 0.009. The edge survives controls.
4. **Gap holds in every count bucket:** Early +18.5pp, Middle +9.9pp, Late +14.7pp.
5. **At 3-2:** catchers challenge pitches half the distance from the edge as batters, and win 14.1pp more often.
6. **Team policy isn't the answer:** No correlation between catcher-share and team value captured (r = 0.024). Individual catcher skill matters more than a blanket policy.

## Files

| File | Description |
|------|-------------|
| `RESEARCH_BRIEF.md` | Hypothesis and data spec |
| `REPORT.md` | Full findings with count-bucket and handedness breakdowns |
| `analyze.py` | Analysis script |

## Methodology

- **Data:** 940 ABS challenges (Mar 27 – Apr 14, 2026); 429 batter, 503 catcher, 8 pitcher
- **Zone model:** Rulebook zone expanded by one baseball radius (1.45")
- **Statistical tests:** Chi-square, logistic regression, Wilson 95% CIs

## Citation

```
CalledThird (2026). "The Best ABS Challengers Are Catchers."
https://calledthird.com/analysis/catchers-are-better-challengers
```

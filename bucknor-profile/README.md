# CB Bucknor By The Numbers — Umpire Profile

**Published article:** [CB Bucknor By The Numbers](https://calledthird.com/analysis/cb-bucknor-by-the-numbers)

## The Question

How does CB Bucknor — long derided as an umpire — actually grade out in the Statcast/ABS era? We profile him against 83 qualified umpires from the 2025 season and follow his 2026 starts as they happen.

## Key Findings

1. **Third-worst accuracy in MLB:** 91.02% vs league 92.55%, 95% CI [90.11%, 91.85%]. Gap vs rest-of-league is statistically significant (p = 0.0002).
2. **Highest average miss distance:** 1.34" when wrong — #1 (worst) of 83 qualified umpires. Estabrook is #2 at 1.31". This is his most distinctive stat.
3. **Tight Struggler quadrant:** BSR 41.0% (doesn't call borderline pitches as strikes), FS% 43.5% (errors lean toward missed strikes). Tight zone + low accuracy.
4. **Miss magnitude is the outlier:** His BSR rank is 31st-of-83 — not extreme. But when Bucknor is wrong, he's wrong by more than anyone else.

## Files

| File | Description |
|------|-------------|
| `RESEARCH_PROPOSAL.md` | Hypothesis, methodology, and analysis plan |
| `analyze.py` | Full analysis: accuracy, miss distance, quadrant classification, significance tests |
| `findings.md` | Complete findings memo with verified rankings, CIs, and per-game context |

## Data

- **2025 season:** 83 qualified umpires from our umpire personality dataset
- **2026 tracking:** nightly cache files (112 games, 541 ABS challenges as of analysis)
- **Distinction throughout:** post-challenge final calls vs inferred pre-ABS human calls
- **Statistical tests:** Wilson confidence intervals, z-test for accuracy gap, independent verification of rank claims

## Methodology Notes

v2 of the analysis corrected four issues:
1. Distinguishes post-challenge final calls from pre-ABS human calls
2. Directional impact uses full initial-call picture (including overturned challenges)
3. Season impact uses cache-wide avg challenge-value per wrong call
4. Full-cache claims independently verified by loading report/ABS files

## Citation

```
CalledThird (2026). "CB Bucknor By The Numbers."
https://calledthird.com/analysis/cb-bucknor-by-the-numbers
```

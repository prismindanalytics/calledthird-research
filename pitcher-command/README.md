# Do Pitchers Lose Their Command? — Within-Outing Plate-Location Scatter

**Published article:** [Do Pitchers Lose Their Command?](https://calledthird.com/analysis/do-pitchers-lose-command)

## The Question

Within a single outing, do starting pitchers lose command as their pitch count climbs? We measure plate-location dispersion (the scatter of where pitches cross the zone) from the first third vs the last third of each start.

## Key Findings

1. **Population mean is flat:** Average plate-location scatter is ~15.2–15.3" at every pitch-count bucket from 1 through 105. Correlation between pitch number and scatter is r = 0.007.
2. **The distribution is asymmetric:** 14.0% of 4,892 true starts blow up (>20% scatter increase) vs 5.2% that tighten — a 2.7:1 ratio. This holds after adjusting for pitch-type mix.
3. **Short outings blow up most:** 23.4% blow-up rate in 30–59 pitch outings vs 12.2% in 90+ pitch outings. Survivor bias — pitchers who get pulled early are often pulled because they're wild.
4. **Metric caveat:** We measure plate-location dispersion, not command in the biomechanical sense. Without catcher target data, we can't separate miss-from-intent from intentional location variety.

## Files

| File | Description |
|------|-------------|
| `memo.md` | Final research memo with full findings, including starter-only analysis |
| `pull_2025_pitcher_variance.py` | 2025 season scatter computation from Statcast |

## Data

- **2025 full season:** 729,827 pitches with location data, 4,892 true starts (30+ pitches), 6,801 total qualifying appearances
- **Starter identification:** pitcher who threw the first pitch for each half-inning side
- **Metric:** plate-location scatter = sqrt(std(plate_x)² + std(plate_z)²) × 12, in inches
- **Within-outing split:** first third vs last third of pitches thrown

## Citation

```
CalledThird (2026). "Do Pitchers Lose Their Command?"
https://calledthird.com/analysis/do-pitchers-lose-command
```

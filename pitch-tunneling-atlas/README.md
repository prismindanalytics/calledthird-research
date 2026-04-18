# The Pitch Tunneling Atlas

League-wide pitch tunneling model measuring how deceptive each MLB pitcher's arsenal is based on trajectory physics.

**Published articles:**
- [The Pitch Tunneling Atlas](https://calledthird.com/analysis/pitch-tunneling-atlas) — Findings and leaderboard
- [The Physics Behind the Tunneling Atlas](https://calledthird.com/analysis/tunneling-atlas-physics) — Methodology, validation, limitations

**Live data:** [calledthird.com/explore#pitchers](https://calledthird.com/explore#pitchers) — 2025 and 2026 tunneling scores, updated nightly.

## The Question

Does pitch tunneling — two pitches looking identical at the batter's decision point, then diverging to the plate — actually predict outcomes, or is it just a narrative?

## The Approach

We computed where every pitch in the 2025 season (739,820 pitches) passes through the batter's decision point (~24 feet from plate) using Statcast's constant-acceleration trajectory equations. Then we decomposed deception into:

- **Plate separation** — how different pitches end up at the plate (pitch diversity)
- **Decision-point tightness** — how similar they look at the commit point (pure tunneling)

And tested whether either predicts whiff rate beyond raw stuff (velocity, spin, movement).

## Key Findings

1. **Plate separation adds +9.0% R² to whiff prediction** beyond velocity+spin+movement (p<0.001)
2. **Decision-point tightness adds +0.8% more** (p=0.016, coefficient negative as expected)
3. **Plate diversity dominates tunneling** by ~11x in explanatory power — but tunneling IS real
4. **FF-KC is the best tunnel pair** in MLB (9.8" average divergence)
5. **Craig Kimbrel leads** the league in divergence (13.3"); **Taylor Rogers** is the anti-tunnel (-1.5")
6. **Whiff rate only** — CSW% and xwOBA show no significant tunneling effect

## Files

| File | Description |
|------|-------------|
| `RESEARCH_BRIEF.md` | Hypothesis, methodology, physics equations, kill criteria |
| `analyze_claude.py` | Claude's analysis — divergence metric, R² decomposition |
| `analyze_codex.py` | Codex's analysis — stabilized ratio, sensitivity analysis, handedness splits |
| `COMPARISON_MEMO.md` | Cross-review synthesis: where they agree, disagree, and which methodology to use |

## Dependencies

```bash
pip install pybaseball pandas numpy scipy statsmodels matplotlib
```

## Reproducing

1. Pull 2025 Statcast data:
   ```python
   from pybaseball import statcast
   df = statcast(start_dt='2025-03-27', end_dt='2025-09-30')
   df.to_parquet('data/statcast_2025_full.parquet')
   ```
2. Run either analysis script:
   ```bash
   python analyze_claude.py  # or analyze_codex.py
   ```

Expected runtime: ~2 minutes on a 2024 MacBook Pro.

## Trajectory Physics

For any pitch, position at distance `y` from the plate:

```
x(t) = release_pos_x + vx0·t + 0.5·ax·t²
z(t) = release_pos_z + vz0·t + 0.5·az·t²
```

where `t` is solved from `y(t) = y_target` (decision point = 23.9 ft).

**Validation:** Model has 4.92" mean plate error due to constant-acceleration approximation (unmodeled drag changes). The bias is systematic and **cancels for relative comparisons** between pitch types — which is all we need for tunneling metrics. See [physics companion article](https://calledthird.com/analysis/tunneling-atlas-physics) for full details.

## Sensitivity

Decision-point distance tested at 20, 23, 23.9, 25, 28 feet. Rankings stable (Spearman ≥ 0.84 across all distances).

## Limitations

- **Centroid-based**, not sequential — doesn't capture pitch-ordering effects
- **Reliever bias** — fewer pitch pairs per pitcher means less dilution, so relievers dominate the leaderboard
- **Whiff-rate signal only** — CSW% and xwOBA show no significant tunneling effect
- **Physics model** valid for relative comparisons, not literal sub-inch trajectory reconstruction

## Citation

```
CalledThird (2026). "The Pitch Tunneling Atlas."
CalledThird.com. https://calledthird.com/analysis/pitch-tunneling-atlas
```

## License

CC BY 4.0 — reproduce, extend, or critique freely.

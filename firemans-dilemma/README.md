# The Fireman's Dilemma — Inherited Runners and the Outs Gradient

**Published article:** [The Fireman's Dilemma](https://calledthird.com/analysis/the-firemans-dilemma)

## The Question

How much of a reliever's inherited runner outcomes is driven by entry situation vs individual skill?

## Key Findings

1. **The outs gradient dominates:** 44% strand rate at 0 outs, 61% at 1 out, 82% at 2 outs.
2. **League IR strand rate:** 68.3% in 2025.
3. **Individual variance is wide:** 12.6pp standard deviation across 171 relievers with 15+ IR.
4. **Skill vs noise is unresolved:** Cross-season correlation between 2025 full-season and 2026 early strand rate is near zero (r = 0.098), but 2026 samples are too thin.
5. **Inning doesn't matter:** 65-70% strand rate is flat across innings 4-9.

## Files

| File | Description |
|------|-------------|
| `memo.md` | Final analytical memo |
| `compute_ir_from_playbyplay.py` | Per-entry IR attribution from MLB play-by-play API (responsiblePitcher field) |
| `pull_2025_inherited_runners.py` | 2025 season data pull |

## Methodology Evolution

The analysis went through 5 iterations (55% heuristic → 54% → 23% → 67.4% boxscore API → play-by-play API). The final approach uses MLB's official `responsiblePitcher` field for per-entry attribution rather than our own join heuristics.

- **Data source:** MLB Stats API boxscore + play-by-play endpoints
- **2025 sample:** 4,044 reliever entries, 6,516 inherited runners
- **Matching:** responsiblePitcher → pitcher who allowed the IR-advancing event

## Citation

```
CalledThird (2026). "The Fireman's Dilemma."
https://calledthird.com/analysis/the-firemans-dilemma
```

# ABS Count Leverage — Where Run Value Actually Concentrates

**Published article:** [ABS Isn't Rewriting Every At-Bat. It's Repricing the Last Pitch.](https://calledthird.com/analysis/the-count-that-matters)

## The Question

The count distribution under ABS has barely shifted. So where is the value being generated?

## Key Findings

1. **3-2 is 2.8% of called pitches but 24% of all ABS run value** — an 8.3x concentration ratio.
2. **Late counts (13% of called pitches) generate 47% of value** — a 3.6x leverage multiplier.
3. **The near-edge 3-2 sweet spot:** 21 challenges within 1 inch of the zone edge at 3-2 produced 8.97 runs — ~17% of all ABS value.
4. **Count distribution shift is statistically significant but small** — biggest move is +0.30pp at 1-0.
5. **Team selection vs execution:** The Athletics and Red Sox buy better count mix; Orioles and Yankees cash their counts better; the Twins have volume but lose on execution.

## Files

| File | Description |
|------|-------------|
| `RESEARCH_BRIEF.md` | Hypothesis, data, methodology |
| `REPORT.md` | Full findings with tables and statistics |
| `analyze.py` | Analysis script |

## Methodology

- **Data:** 940 ABS challenges (Mar 27 – Apr 14, 2026) + 73K pitches matched-window 2025 vs 2026
- **Value metric:** RE288 count-state linear weights
- **Count buckets:** Early (0-0, 0-1, 1-0), Middle (0-2, 1-1, 1-2, 2-0, 2-1), Late (2-2, 3-0, 3-1, 3-2)

## Citation

```
CalledThird (2026). "ABS Isn't Rewriting Every At-Bat. It's Repricing the Last Pitch."
https://calledthird.com/analysis/the-count-that-matters
```

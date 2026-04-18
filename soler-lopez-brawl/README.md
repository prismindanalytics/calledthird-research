# Do Umpires Call Differently After Bench-Clearing Incidents?

**Published article:** [After a Fight, the Zone Gets Cleaner](https://calledthird.com/analysis/the-zone-after-a-fight)

## The Question

After bench-clearing incidents (like the April 2026 Soler-Lopez brawl), do umpires call a wider or tighter zone in the next game to assert control? Or is that just a narrative?

## The Approach

7 Statcast-era bench-clearing incidents (2019-2026). For each, we identified the home plate umpire and tracked their next game behind the plate. Then we compared called-strike rate, accuracy, and out-of-zone strike rate to their season baseline.

## Key Findings

1. **No significant change in zone size** — called-strike rate shifts -1.3pp from baseline (p=0.302, CI includes zero)
2. **Accuracy improves significantly** — +2.0pp in the next game (p=0.001), all 6 umpires above baseline
3. **Out-of-zone strike rate drops** — -2.6pp (p=0.004, 6/6 below baseline)
4. **The story:** umpires don't change the zone after a fight. They get more precise.

## Files

| File | Description |
|------|-------------|
| `RESEARCH_BRIEF.md` | Hypothesis, methodology, kill criteria |
| `incidents.json` | The 7 incidents with game_pks and umpire assignments |
| `analyze_claude.py` | Claude's analysis |
| `analyze_codex.py` | Codex's analysis (stronger physics validation) |
| `COMPARISON_MEMO.md` | Cross-review synthesis |

## The Incidents

| Date | Teams | Key Players | Umpire |
|------|-------|-------------|--------|
| 2026-04-07 | LAA vs ATL | Soler/Lopez | Edwin Moscoso |
| 2024-05-01 | MIL vs TB | Peralta/Siri | Alex MacKay |
| 2023-08-05 | CWS vs CLE | Anderson/Ramirez | Mark Wegner |
| 2022-06-26 | SEA vs LAA | Winker/Rendon | John Bacon |
| 2022-04-27 | NYM vs STL | Arenado/Cabrera | Jeremie Rehak |
| 2020-08-09 | HOU vs OAK | Laureano | Nick Mahrley |
| 2019-07-30 | PIT vs CIN | Kela/Garrett/Puig | Larry Vanover |

Soler-Lopez (2026) lacks a completed next game in our sample and is excluded from paired statistical tests.

## Methodology Notes

- **Zone model:** Rulebook zone expanded by one baseball radius (1.45" / 12 = 0.121 ft) on all sides
- **Sample:** 6 complete incident/next-game pairs; 1 pending
- **Statistical tests:** One-sample t-tests on (next game − umpire baseline) deltas
- **Limitation:** n=6 is small. The OOZ finding is exploratory (multiple metrics tested)

## Citation

```
CalledThird (2026). "After a Fight, the Zone Gets Cleaner."
CalledThird.com. https://calledthird.com/analysis/the-zone-after-a-fight
```

## License

CC BY 4.0 — reproduce, extend, or critique freely.

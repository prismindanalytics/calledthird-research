# The Fireman's Dilemma — Research Memo (Final v2, Per-Entry Play-by-Play)

## Key Change from Previous
Replaced the flawed boxscore+Statcast join with **per-entry analysis from the MLB play-by-play API**. The `responsiblePitcher` field in each scoring event provides official scorer attribution at the event level, and entry context (outs, inning) comes from the first pitch event.

**Qualification**: IRS is derived from `responsiblePitcher` events (play-by-play), while IR totals come from boxscore. A small number of entries (~0.1%) show discrepancies between the two sources (e.g., Brenan Hanifee 1 IR / 1 IRS in play-by-play vs 1 / 0 in boxscore). The method is substantially more accurate than any previous version but is not a perfect per-entry decomposition.

## Data Sources
- **2025 per-entry** (play-by-play API): ~4,038 entries (after deduping 6 games appearing on two dates), ~6,506 IR, ~2,062 IRS. Strand rate: **68.3%**. Note: displayed counts use the raw output (4,044 / 6,516 / 2,064); deduped totals differ by <0.2% and do not change any qualitative findings.
- **2026 per-entry** (play-by-play API): 225 entries, 374 IR, 122 IRS. Strand rate: **67.4%**.
- **Cross-season**: 44 relievers with 10+ IR in 2025 and 3+ IR in 2026. r = 0.098.
- Entry context (inning, outs) from `playEvents[0].count.outs` in the first at-bat the reliever faces.
- Validation: 2025 per-entry aggregate (68.3%) matches boxscore aggregate (68.4%).

## Key Findings

### 1. The Outs Gradient Is Real, Steep, and Consistent (Per-Entry Play-by-Play-Backed)

| Outs at Entry | 2025 Entries | 2025 Strand | 2026 Entries | 2026 Strand |
|--------------|-------------|-------------|-------------|-------------|
| 0 | 611 | **43.8%** | 36 | **38.5%** |
| 1 | 1,449 | **61.4%** | 77 | **61.7%** |
| 2 | 1,928 | **82.0%** | 109 | **79.0%** |

This is the headline finding. Entering with 0 outs means the majority of inherited runners score (56% scoring rate). With 2 outs, only 18% score. The gradient is consistent across both seasons.

The 2025 data provides large per-entry sample sizes (611 / 1,449 / 1,928). Each entry has IRS derived from play-by-play `responsiblePitcher` events and IR from the boxscore pitcher-game total — substantially more accurate than previous game-level projections, though not a perfect per-entry decomposition (see qualification above).

### 2. By Inning (2025, 4,044 entries)
| Inning | Entries | IR | Strand |
|--------|---------|---:|--------|
| 3 | 85 | 155 | 60% |
| 4 | 210 | 347 | 70% |
| 5 | 603 | 993 | 67% |
| 6 | 1,028 | 1,595 | 68% |
| 7 | 982 | 1,518 | 70% |
| 8 | 731 | 1,205 | 69% |
| 9 | 272 | 466 | 65% |

Strand rate is remarkably flat across innings 4-9 (65-70%). The entry inning doesn't matter much — it's the outs that matter.

### 3. Extra Innings (2025)
56 entries, 98 IR, 76.5% strand. Higher strand rate than regular innings — possibly because extra-inning entries are often automatic-runner situations with 0 outs (runner placed on 2B), and the first goal is just to prevent the run.

### 4. Cross-Season Persistence: Inconclusive
- r = 0.098, n = 44, Fisher 95% CI ≈ [-0.21, 0.38]
- Consistent with modest positive, zero, or modest negative persistence
- 2026 sample too thin (3-5 IR per reliever)

### 5. 2025 Leaderboard (15+ IR, true sorted, 171 qualified)

Top 10:
| # | Reliever | IR | IRS | Strand % |
|---|----------|---:|----:|----------|
| 1 | Max Lazar | 15 | 0 | 100.0% |
| 2 | Ben Casparius | 29 | 1 | 96.6% |
| 3 | Bryan King | 23 | 1 | 95.7% |
| 4 | Erik Sabrowski | 16 | 1 | 93.8% |
| 5 | Michael Kelly | 15 | 1 | 93.3% |
| 6 | Fernando Cruz | 27 | 2 | 92.6% |
| 7 | Daniel Lynch IV | 21 | 2 | 90.5% |
| 8 | Tyler Ferguson | 29 | 3 | 89.7% |
| 9 | Yariel Rodriguez | 36 | 4 | 88.9% |
| 10 | Adrian Morejon | 45 | 5 | 88.9% |

Bottom 10:
| # | Reliever | IR | IRS | Strand % |
|---|----------|---:|----:|----------|
| 162 | Victor Vodnik | 21 | 11 | 47.6% |
| 163 | Alek Jacob | 17 | 9 | 47.1% |
| 164 | Aaron Bummer | 30 | 16 | 46.7% |
| 165 | Justin Topa | 24 | 13 | 45.8% |
| 166 | Jacob Latz | 18 | 10 | 44.4% |
| 167 | Jonathan Loaisiga | 18 | 10 | 44.4% |
| 168 | Andrew Saalfrank | 18 | 10 | 44.4% |
| 169 | Taylor Rogers | 32 | 18 | 43.8% |
| 170 | Colin Poche | 15 | 11 | 26.7% |
| 171 | Joey Wentz | 22 | 18 | 18.2% |

Wilson 95% CIs: Morejon 88.9% [76.5%, 95.2%], Wentz 18.2% [7.3%, 38.5%].

## Proposed Article: "The Fireman's Dilemma"

1. **The number**: 68% strand rate = 1 in 3 inherited runners scores.
2. **The outs cliff**: 44% → 61% → 82%. The steepest finding. Entering with 0 outs = majority score. 2 outs = most stranded. Play-by-play-backed with 4,044 entries across the full 2025 season.
3. **Inning doesn't matter**: 65-70% across innings 4-9. Flat.
4. **The leaderboard**: 2025 full season. Wide Wilson CIs. Descriptive, not skill rankings.
5. **The persistence question**: Inconclusive (r = 0.098). Open question.
6. **The takeaway**: The entry situation is strongly associated with strand outcomes — 0-out entries have dramatically worse strand rates than 2-out entries. This is descriptive, not causal: 0-out and 2-out entries differ in leverage, base state, and managerial context. But the gradient is steep enough to be directionally useful — a manager weighing whether to pull a starter mid-inning should know that the inherited runner situation gets materially harder with fewer outs.

## Data Validation
- 2025 per-entry aggregate (68.3%) closely matches boxscore aggregate (68.4%) — 0.1pp difference
- 2026 per-entry aggregate (67.4%) matches boxscore aggregate (67.4%)
- Outs field from `playEvents[0].count.outs` validated against known game situations
- Known discrepancies: ~0.1% of entries show IRS mismatches between play-by-play and boxscore (e.g., Hanifee, Detmers). IR counts come from boxscore, IRS from play-by-play `responsiblePitcher` — the two sources are not perfectly reconciled
- 6 duplicated pitcher-game rows from games appearing on two dates (rain delays/makeups) — deduping changes totals by <0.2%

## References
- [MLB IR-A%](https://www.mlb.com/glossary/advanced-stats/inherited-runs-allowed-percentage)
- [MLB Stats API play-by-play](https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay) — `responsiblePitcher` field
- [MLB Stats API boxscore](https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore)

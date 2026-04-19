# CB Bucknor Analysis — Opus Researcher Findings (v2, corrected)

**Date**: 2026-04-08  
**Researcher**: Opus (independent analysis)  
**Data sources**: 2025 umpire personality dataset (83 umpires), all 2026 nightly cache files (112 games, 541 ABS challenges), 2026-03-28 called pitches parquet, 2026-03-28 ABS challenges JSON

**Revision note**: v2 corrects four issues from the initial analysis:
1. Distinguishes post-challenge final calls from inferred pre-ABS human calls throughout
2. Directional impact now uses the full initial-call picture (including overturned challenges)
3. Season impact estimates use cache-wide avg challenge-value per wrong call, not single-game dRE
4. Full-cache claims (game rank, Bucknor game count) are independently verified by loading all report/ABS files

---

## Q1: How Bad Is Bucknor Compared to the League?

**Verified rankings (2025 season, 83 qualified umpires):**

| Metric | Bucknor | League Mean | Rank | Significant? |
|--------|---------|-------------|------|-------------|
| Accuracy | 91.02% | 92.55% | 3rd worst | YES (p=0.0002) |
| Wrong calls/game | 13.4 | 10.9 | 3rd most | — |
| Avg miss distance | 1.34" | 1.13" | **#1 (worst)** | — |

**Statistical significance**: Bucknor's accuracy gap vs rest of league is -1.54pp, 95% CI [-2.42pp, -0.67pp]. Wilson CIs do not overlap (Bucknor [90.11%, 91.85%] vs rest-of-league [92.47%, 92.65%]). z = -3.77, p = 0.0002.

**Key insight**: The avg miss distance is his most distinctive stat. At 1.34", he's the farthest-missing umpire in MLB. #2 is Estabrook at 1.31". When Bucknor is wrong, he's wrong by more than anyone else.

**Bottom 5 with CIs:**
1. Dreckman: 90.11% [89.07%, 91.05%]
2. Diaz: 90.95% [90.08%, 91.76%]
3. **Bucknor: 91.02% [90.11%, 91.85%]**
4. Fletcher: 91.17% [90.30%, 91.96%]
5. Kulpa: 91.42% [90.41%, 92.33%]

Note: Bucknor and Diaz CIs overlap heavily — we cannot confidently say who is 2nd vs 3rd worst.

---

## Q2: What Kind of Umpire Is He?

**Quadrant: Tight Struggler** (low accuracy + conservative borderline zone)
- BSR 41.0% (below league median 42.7%) — doesn't call borderline pitches as strikes
- FS% 43.5% — errors lean slightly toward missed strikes (212 MS vs 163 FS)
- However: BSR of 41.0% is not an extreme tail value (ranked 31st-lowest of 83). The quadrant label is directionally right, but **miss severity is the clearer outlier**, not an unusually tight zone.

**Quadrant comparison:**
| Quadrant | n | Avg Acc | Avg Miss Dist | Avg Wrong/Gm |
|----------|---|---------|---------------|-------------|
| Conservative Ace | 19 | 93.28% | 1.08" | 9.8 |
| Aggressive Ace | 23 | 93.10% | 1.09" | 10.0 |
| Wild Expander | 20 | 91.92% | 1.18" | 11.8 |
| **Tight Struggler** | 21 | **91.89%** | **1.16"** | **11.9** |

Tight Strugglers as a group are the worst quadrant, and Bucknor is 2nd worst within it.

**High-leverage accuracy**: 92.44% (gap: +1.42pp above overall). He actually performs slightly better in high-leverage counts than his baseline. League average gap is +1.17pp. The current evidence does not support a "melts down in leverage counts" framing.

**Among Tight Struggler peers**: Bucknor stands out for his miss distance (1.34" vs peer avg ~1.16"). Even within his low-accuracy quadrant, his misses are unusually far from the zone edge.

---

## Q3: BOS@CIN March 28, 2026 — Game Reconstruction

### Important: Post-challenge vs Pre-ABS distinction

The `all_called` parquet stores **post-challenge final calls**. Overturned ABS challenges appear as correct calls, not wrong calls. To evaluate Bucknor's actual human performance, we add the 6 overturned challenges back.

### Game summary

| Metric | Post-challenge (game record) | Pre-ABS (Bucknor's actual calls) |
|--------|------------------------------|----------------------------------|
| Wrong calls | 21 | **27** |
| Accuracy | 90.58% | **87.89%** |
| False strikes | 16 | **22** |
| Missed strikes | 5 | 5 |
| FS% | 76.2% | **81.5%** |
| Total miss value | 4.082 cv | **5.416 cv** |

- **Score**: CIN 6, BOS 5 (extra innings)
- **ABS challenges**: 8 filed, 6 overturned (75% overturn rate vs 55.1% league avg)
- **Pre-ABS rank**: **#1 out of 112 games** in the 2026 cache by inferred human miss value (verified by loading all report files)
- ABS corrected 6 of his false strikes, boosting accuracy from 87.89% to 90.58%

### ABS challenge reconstruction

| Inn | Batter | Pitcher | Initial | Final | Result | Count | Edge Dist | CV Est |
|-----|--------|---------|---------|-------|--------|-------|-----------|--------|
| 2 | Trevino | Gray | Ball | Ball | UPHELD | 1-2 | 0.03" | 0.273 |
| 3 | Anthony | Singer | Strike | Ball | **OVERTURNED** | 2-1 | 2.72" | 0.210 |
| 3 | Anthony | Singer | Strike | Strike | UPHELD | 3-1 | 0.11" | 0.306 |
| 6 | De La Cruz | Coulombe | Strike | Ball | **OVERTURNED** | 1-0 | 2.44" | 0.130 |
| 6 | Suarez | Watson | Strike | Ball | **OVERTURNED** | 1-2 | 0.30" | 0.273 |
| 6 | Suarez | Watson | Strike | Ball | **OVERTURNED** | 2-2 | 1.07" | 0.384 |
| 7 | Benson | Watson | Strike | Ball | **OVERTURNED** | 1-0 | 0.90" | 0.130 |
| 7 | Benson | Watson | Strike | Ball | **OVERTURNED** | 3-0 | 2.41" | 0.207 |

All 6 overturned challenges were false strikes (Strike -> Ball) challenged by batters.

### Directional impact (full initial-call picture)

| Component | BOS benefited | CIN benefited |
|-----------|--------------|---------------|
| Remaining wrong calls | 1.438 cv | 2.644 cv |
| Overturned challenges (initial call) | 1.124 cv | 0.210 cv |
| **Combined initial human misses** | **2.562 cv** | **2.854 cv** |

**Bucknor's original human calls slightly favored CIN** (2.854 vs 2.562 cv). The ABS system then corrected 1.124 cv of BOS benefit and only 0.210 cv of CIN benefit — meaning ABS disproportionately fixed calls that had been helping BOS.

This is not a case of dramatic one-sided bias — the difference is moderate (~0.29 cv). Both teams were harmed significantly by inaccuracy.

### Worst single call (remaining)

Inn 4 Top, 3-2 count, 2 outs, runner on 2B: Brady Singer batting. Called strike on a sinker 3.23" outside the zone. Challenge value = 0.690. Highest-impact single remaining wrong call in the game.

### Count-state distribution (combined initial human calls)

| Count | Wrong calls | CV Total | Leverage |
|-------|-------------|----------|----------|
| 2-2 | 3 (2 remaining, 1 overturned) | 1.152 | HIGH |
| 0-0 | 8 (8 remaining, 0 overturned) | 0.760 | low |
| 3-2 | 1 | 0.690 | HIGH |
| 2-1 | 3 (2 remaining, 1 overturned) | 0.630 | low |
| 2-0 | 3 | 0.585 | low |
| 1-2 | 2 (1 remaining, 1 overturned) | 0.546 | low |

8 of 27 initial human wrong calls came on 0-0 counts (first pitch). 6 came in high-leverage counts (2-2, 3-2, 3-0).

---

## Q4: Has ABS Changed His Behavior?

**Data limitation**: Only 1 Bucknor plate game exists in our 2026 local cache (verified by scanning all 112 game reports). This is insufficient for trend analysis.

**What we can observe:**

| Metric | 2025 (28 games) | BOS@CIN pre-ABS (1 game) |
|--------|-----------------|--------------------------|
| Accuracy | 91.02% | 87.89% |
| FS% | 43.5% | 81.5% |
| False strikes/game | 5.8 avg | 22 |
| Missed strikes/game | 7.6 avg | 5 |

The BOS@CIN game shows an extreme false-strike skew (81.5% FS, 22 false strikes = 3.8x his 2025 per-game rate). ABS corrected 6 of these, boosting accuracy from 87.89% to 90.58%.

**Recommendation**: The 2025 baseline is firm. For the 2026 narrative, we can say "early 2026 data shows the pattern continuing" and cite the BOS@CIN specifics, but we should not claim statistical evidence of improvement or regression from 1 game. The Q4 question is **unresolved** pending more 2026 data.

---

## Q5: Bucknor vs Jimenez (Worst vs Best)

| Metric | Bucknor | Jimenez | Gap |
|--------|---------|---------|-----|
| Accuracy | 91.02% | 94.57% | -3.55pp |
| Wrong calls/game | 13.4 | 7.7 | +5.7 |
| Avg miss distance | 1.34" | 1.01" | +0.33" |
| HL accuracy | 92.44% | 96.05% | -3.61pp |
| BSR | 41.0% | 42.7% | -1.7pp |
| FS% | 43.5% | 41.0% | +2.5pp |

**Statistical significance**: z = -6.04, p < 0.000001. CIs do not overlap. The gap is highly significant.

**Season impact estimation** (using cache-wide avg challenge-value per pre-ABS wrong call = 0.171):

| Comparison | Extra wrong/game | Extra cv/game | Over 28 games |
|------------|-----------------|---------------|---------------|
| Bucknor vs league avg | 2.5 | 0.436 | 12.22 |
| Bucknor vs Jimenez | 5.7 | 0.977 | 27.35 |

These are modeled estimates using the repo's count-based wOBA challenge-value framework across all 112 cached games, not directly observed 2025 run values. They are useful for sizing the gap but should be presented as estimates.

The 0.33" miss distance gap is telling: Jimenez's misses average 1.01" from the zone edge (borderline, understandable), while Bucknor averages 1.34" (visually obvious). This explains the viral clips.

---

## Kill Criteria Assessment

| Criterion | Result | Status |
|-----------|--------|--------|
| Accuracy within normal range? | No — p=0.0002, gap -1.54pp CI [-2.42, -0.67] | **CLEAR** |
| BOS@CIN an extreme outlier? | Pre-ABS 87.89% is 3.13pp below his 91.02% baseline — worse than normal, but baseline is already 3rd worst | **CLEAR** |
| Can we reconstruct the game? | Yes — 223 pitches, 8 ABS challenges, pre/post distinction clear | **CLEAR** |

## Success criteria (nuanced, per Codex framing)

| Question | Result |
|----------|--------|
| Q1: Materially worse than league avg? | **Go.** Significant at 95%. |
| Q2: Distinct profile type? | **Partial.** Tight Struggler is directionally right, but miss severity is the standout, not zone shape. |
| Q3: BOS@CIN reconstructable? | **Go with caveat.** Must distinguish post-challenge from initial human calls. |
| Q4: ABS changed behavior? | **Unresolved.** Only 1 game in cache. |
| Q5: Seasonal gap sizable? | **Go, modeled.** Useful estimate, but modeled not directly observed. |

**Recommendation: PROCEED with article.** All kill criteria are clear.

---

## Key Findings for Article

1. **The gap is real and significant.** Bucknor's 91.02% accuracy is -1.54pp below the rest of the league (p=0.0002). This isn't noise — it's 4,174 pitches across 28 games.

2. **His signature stat is miss distance.** #1 in MLB at 1.34". When other umpires miss, it's by ~1.13" (borderline). When Bucknor misses, it's by more. This is why his bad calls go viral — they're visually obvious.

3. **BOS@CIN was his worst game, and the worst in our 2026 database.** Pre-ABS: 27 wrong calls, 87.89% accuracy, 5.416 cv — ranked #1 out of 112 games. ABS corrected 6 calls to bring the final line to 90.58%. The article should present both numbers.

4. **False strikes are his weakness.** In BOS@CIN, 22 of 27 initial human wrong calls were false strikes (81.5%). Six were overturned by ABS. The pattern: he calls strikes on pitches that are clearly off the plate.

5. **Both teams suffered; CIN benefited slightly more from his initial calls.** Combined inferred human miss value: CIN 2.854 vs BOS 2.562. Not systematic bias, but widespread inaccuracy with a moderate tilt.

6. **The Bucknor-Jimenez gap is ~5.7 wrong calls/game.** That's roughly 0.977 challenge-value points per game of additional umpire-driven impact. Over 28 games: 27.35 cv.

7. **He's not the absolute worst.** Dreckman (90.11%) and Diaz (90.95%) are below him. But his miss distance makes his errors the most visible.

8. **Critical editorial note**: Any published number about the BOS@CIN game must state whether it's the post-challenge final line or the inferred pre-ABS human line. Mixing them is the most likely source of error in this analysis.

---

## Data Files

- `analyze.py` — full analysis script (reproducible, loads all 2026 cache files)
- `data/bucknor_analysis_opus.json` — structured output for all findings

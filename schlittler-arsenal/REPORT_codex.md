# Research Report: Cam Schlittler's Arsenal

Prepared on April 12, 2026 inside `research/schlittler-arsenal/codex-analysis-2026-04-12/`.

## Executive Summary
- Local sample: **231 classified pitches** plus **1 automatic-strike row** (**232 total Statcast rows**), **3 starts**, **22 strikeouts**, **0 walks**, and **8 hits allowed** through April 7, 2026.
- Verdict: **publish, but narrow the thesis**. The data strongly support a real arsenal redesign and a rare near-even three-fastball mix. They do **not** support a "largest movement spread in baseball" claim.
- Mix shift: Schlittler went from **55.3% 4-seam / 18.9% cutter / 7.1% sinker** in 2025 to **31.6% / 29.4% / 30.3%** in the first three starts of 2026.
- Balance is the standout trait: among **101** right-handed 2025 pitchers who threw at least 50 of all three fastballs, Schlittler's **2026 mix balance ranks at the 100th percentile**, while his movement-spread triangle is only **66th percentile**. The story is the **combination** of balance plus above-average separation, not separation alone.
- The cutter remake is the loudest pitch-level change: **+1.6 mph**, **-2.2 inches horizontal change**, and **+7.2 inches vertical change** versus 2025. Relative to 2025 right-handed cutters, the 2026 cutter already sits around the **91st percentile in velocity** and **85th percentile in vertical movement**.
- Early role separation is real: the **4-seam** owns the whiffiest swing profile (**48.8%** on swings), the **sinker** owns the called-strike profile (**56.7%**, vs **40.5%** league), and the **cutter** sits in the middle as the bridge pitch.

## Recommended Framing
- Best angle: **Cam Schlittler Didn't Build the Biggest Fastball Spread in Baseball. He Built One of the Most Balanced Ones.**
- Alternate angle: **The New Cutter Is the Key to Cam Schlittler's Three-Fastball Attack.**
- Editorial recommendation: **publish now**, but keep the promise limited to:
  - the mix change is real
  - the cutter reshape is real
  - the three-fastball usage balance is unusual
  - the role split is visible already

## What Holds Up
1. **The arsenal changed, not just the outcomes.**
   Schlittler was a 4-seam-first pitcher in 2025. In 2026 he is essentially splitting the workload across FF, FC, and SI.
2. **The cutter is the fulcrum pitch.**
   The 2026 cutter is harder and carries much more vertical life than his 2025 version. That is the cleanest evidence that something material changed.
3. **The fastballs are doing different jobs.**
   The 4-seam is being used higher and farther from the heart of the zone, which helps explain the low called-strike rate and high whiff rate.
   The sinker is landing much closer to the zone interior, which helps explain the strong called-strike returns.
   The cutter lives between those poles, giving him a glove-side fastball that still looks like part of the same tunnel family.
4. **Balance is the rare part.**
   The 2026 three-fastball mix is almost perfectly even. The nearest 2025 shape-and-usage matches were **Cole Winn, Luis Severino, Walker Buehler, Zack Kelly**, but even those were less balanced.

## What Does Not Hold Up
- The raw movement spread is **good, not unprecedented**. Schlittler's 2026 triangle area lands around the **66th percentile** of eligible 2025 right-handed three-fastball pitchers, not the extreme top of the population.
- The early whiff and called-strike rates are **directionally interesting, not stabilized**. We only have 231 classified pitches and just over 100 called pitches.
- The local ABS-style called-pitch parquet stops on **April 5, 2026**, so the April 7 start cannot be audited pitch-by-pitch for rulebook correctness from the precomputed `all_called` files.

## Pitch-Mix Change Table
| Pitch | 2025 usage | 2026 usage | Usage delta | Velo delta | Horiz move delta | Vert move delta |
| --- | --- | --- | --- | --- | --- | --- |
| 4-Seam | 55.3% | 31.6% | -23.7 pp | -0.7 mph | -0.2 in | +1.2 in |
| Cutter | 18.9% | 29.4% | +10.5 pp | +1.6 mph | -2.2 in | +7.2 in |
| Sinker | 7.1% | 30.3% | +23.2 pp | -1.0 mph | -2.8 in | -1.1 in |
| Curve | 14.7% | 7.8% | -6.9 pp | +1.3 mph | -4.0 in | +4.1 in |

## Outcome Snapshot: 2026 vs 2025 RHP League
| Pitch | 2026 zone | 2026 whiff | 2026 called strike | 2026 CSW | League zone | League whiff | League called strike | League CSW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4-Seam | 47.9% | 48.8% | 23.3% | 38.4% | 56.4% | 18.7% | 33.7% | 26.3% |
| Cutter | 54.4% | 25.0% | 34.4% | 29.4% | 54.0% | 20.3% | 32.6% | 26.5% |
| Sinker | 67.1% | 17.5% | 56.7% | 34.3% | 57.4% | 11.7% | 40.5% | 27.1% |
| Curve | 33.3% | 28.6% | 27.3% | 27.8% | 43.5% | 29.3% | 31.0% | 30.2% |

## Historical Three-Fastball Neighbors
These are **shape-and-usage comps**, not performance comps.

| Pitcher | FF share | FC share | SI share | Triangle area | Distance |
| --- | --- | --- | --- | --- | --- |
| Cole Winn | 40.8% | 28.2% | 31.0% | 75.7 | 2.78 |
| Luis Severino | 42.6% | 26.8% | 30.6% | 40.6 | 2.81 |
| Walker Buehler | 43.2% | 28.8% | 28.0% | 59.2 | 3.03 |
| Zack Kelly | 29.4% | 47.6% | 23.0% | 41.0 | 3.08 |
| Shawn Armstrong | 37.7% | 31.4% | 30.8% | 54.2 | 3.19 |
| Lou Trivino | 25.0% | 39.2% | 35.8% | 41.3 | 3.24 |

## Charts
### 1. Movement profile
![Movement profile](charts/movement_profile.png)

### 2. Pitch-mix shift
![Pitch mix shift](charts/pitch_mix_shift.png)

### 3. Velocity context
![Velocity distributions](charts/velocity_distributions.png)

### 4. Location roles
![Location roles](charts/location_roles.png)

### 5. Balance vs spread
![Balance vs spread](charts/balance_vs_spread.png)

## Bottom Line
- **Publishable claim**: Schlittler has turned a 4-seam-led arsenal into a highly balanced three-fastball attack, and the cutter reshape is the clearest mechanical/pitch-design lever behind it.
- **Do not claim**: that his movement spread is uniquely extreme league-wide.
- **Best single sentence**: *Cam Schlittler's first three starts look less like random heater luck and more like a deliberate three-fastball redesign, with a remade cutter turning an already-hard arsenal into a balanced, role-specific attack.*

## Method
1. Loaded `statcast_2025_full.parquet` as the right-handed 2025 league baseline and every local 2026 parquet through `statcast_2026_apr06_11.parquet` for Schlittler's three starts.
2. Built a rulebook strike zone from `plate_x`, `plate_z`, `sz_top`, and `sz_bot`, expanded by one baseball radius to account for ball-center tracking.
3. Calculated pitch-level rates from Statcast descriptions:
   whiff rate = whiffs / swings,
   chase rate = swings on out-of-zone pitches / out-of-zone pitches,
   called-strike rate = called strikes / called pitches,
   CSW = (called strikes + whiffs) / total pitches.
4. Compared Schlittler's three-fastball mix against right-handed 2025 pitchers with at least 50 FF, 50 FC, and 50 SI.
5. Defined `triangle_area` as the movement-space area formed by the FF, FC, and SI centroids in inches.
   Defined `entropy` as normalized pitch-share balance across those three pitches.

## Caveats
- This is still a **231-pitch classified sample** with one additional automatic-strike row, so outcome rates can move quickly.
- The three-fastball comp table compares 2026 Schlittler to **2025** pitchers because that is the full local league baseline provided in the repo.
- Statcast movement (`pfx_x`, `pfx_z`) is not the same thing as Hawkeye active-spin or induced vertical break modeling.
- The game log confirms **22 strikeouts and 0 walks** from the local Statcast event rows, but this report does not re-estimate ERA or opponent quality.

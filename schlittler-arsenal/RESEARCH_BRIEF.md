# Research Brief: Cam Schlittler's Arsenal — Building a Three-Fastball Monster

## Hypothesis
Cam Schlittler's early 2026 dominance (22 K, 0 BB, 1.62 ERA through 3 starts) is driven by an elite three-fastball approach — 4-seam, sinker, and cutter — with movement profiles that create an unusually wide tunneling spread for a single-pitcher arsenal. His Statcast profile maps to a specific archetype of sustained MLB success.

## The Hook
Schlittler is the biggest pitching story in baseball right now. First Yankee to open with consecutive starts of 5+ scoreless IP and 7+ K. 22:0 K:BB ratio. FanGraphs has him atop the Pitching+ leaderboard. Pitcher List and Pinstripe Alley both ran deep-dives on his fastballs.

CalledThird's angle: we have every pitch from all 3 starts in Statcast. We can show the movement profile, the tunneling geometry, and how his pitch mix compares to the league — not just the box score.

## Pillar
Pitcher Profiles (currently 3 articles, needs current-season content)

## Key Facts (from web research)
- **Age:** 25, RHP, 6'6" 215 lbs, 7th round 2022 (Northeastern)
- **2026 line (3 starts):** 2-0, 1.62 ERA, 16.2 IP, 22 K, 0 BB
- **Pitch mix shift:** 4-seam dropped to 35%, sinker usage jumped from 6.1% (2025) to 23.5%, cutter has a new grip with more movement
- **Pitcher ID:** 693645

## Data Available

### In `data/` folder:
- `statcast_2025_full.parquet` — Full 2025 season (league baseline, ~739K pitches)
- `2026-03-27.parquet` through `2026-04-05.parquet` — Daily Statcast files (includes Schlittler starts 1-2)
- `statcast_2026_apr06_11.parquet` — Apr 6-11 data (includes Schlittler start 3, Apr 7)

### Schlittler in the data:
- **232 total pitches across 3 starts** (148 from Mar 27 + Apr 1, 84 from Apr 7)
- **Pitch types:** FF (4-seam): ~73, FC (cutter): ~68, SI (sinker): ~70, CU (curve): ~18, SL (slider): ~2
- **Games:** 2026-03-27 (vs SF), 2026-04-01 (vs SEA?), 2026-04-07 (vs OAK?)

## Analysis Steps

### 1. Movement Profile (pfx_x, pfx_z)
- Plot Schlittler's full arsenal vs league RHP pitch-type clouds (same approach as the Imai article)
- Show the three-fastball cluster: FF, SI, FC spread across the movement space
- Compare to 2025 league averages for each pitch type

### 2. Velocity Profile
- Distribution of release_speed by pitch type
- Compare to league RHP distributions
- Note: his 4-seam and sinker are both mid-to-high 90s

### 3. Pitch Mix Change (2025 vs 2026)
- If 2025 Schlittler data exists in the full-season file, show the shift: sinker from 6% to 24%, 4-seam from dominant to 35%
- The cutter's new grip → movement comparison

### 4. Location Heatmap
- Where does each pitch type land? (plate_x, plate_z scatter by pitch type)
- Is he living on edges (nibbling) or attacking the zone?
- Called-strike rate by zone location

### 5. Outcome Analysis
- Whiff rate by pitch type (swinging_strike / total swings at each pitch)
- Called-strike rate by pitch type
- Compare to league averages
- The 22:0 K:BB ratio — is it the location (zone rate), the movement, or both?

### 6. Historical Comp
- Find pitchers with similar three-fastball profiles from 2025 data (cluster by FF+SI+FC usage %)
- Who else throws this mix, and how did they perform?

## Kill Criteria
- If Schlittler's Statcast profile is unremarkable (movement and velocity near league average), the story weakens
- If the sample is too thin (232 pitches may be enough for a profile but not for reliable rate stats)
- Story becomes "too early to tell" rather than "here's something special"

## Article Framing
- **If confirmed:** "Cam Schlittler's Three-Fastball Blueprint: The Statcast Profile Behind 22 K and 0 BB"
- **If mixed:** "What Cam Schlittler's Statcast Data Actually Shows — And What Needs More Time"
- Lead with movement profile chart (the visual hook)
- Include the pitch mix shift narrative (cutter grip change, sinker emergence)
- Honest about sample size throughout

## Output
- Charts: movement profile, velocity distributions, location heatmap, pitch mix comparison, outcome table
- Summary: key stats with league context
- Recommendation: publish / revise / wait for more data

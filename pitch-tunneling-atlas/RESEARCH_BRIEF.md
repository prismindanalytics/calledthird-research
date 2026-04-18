# Research Brief: The Pitch Tunneling Atlas

## Overview
Build a league-wide pitch tunneling model that quantifies how deceptive each pitcher's arsenal is — measuring how similar pitches look out of the hand and how much they diverge by the time the batter must decide. This is not a single article but an analytical framework: a leaderboard, per-pitcher profiles, and the methodology to identify elite deception vs mechanical redundancy.

Nobody publishes a systematic tunneling metric across the full league. Baseball Savant shows movement profiles. FanGraphs references tunneling anecdotally. CalledThird would own the definitive public quantitative framework.

## Core Physics

### The Decision Point
A batter facing a 95 mph fastball has roughly **400 milliseconds** from release to plate arrival. But the commit/no-commit decision happens earlier — research suggests batters must decide to swing by the time the ball is approximately **23-25 feet from the plate** (~170ms after release). This means batters base their swing decision on the pitch's position and trajectory at a point roughly **35-40% of the way to the plate**.

### What Tunneling Means
Two pitches "tunnel" well when:
1. They pass through **nearly the same point** at the decision distance (~23 feet from plate)
2. They **diverge significantly** by the time they reach the plate

A pitcher who throws a 4-seam and a changeup from the same release point, through the same tunnel window, but arriving at different plate locations with different velocities — that's elite tunneling. A pitcher who telegraphs pitch type from the release point has poor tunneling.

### The Math
For each pitch, we can compute its position at any distance from the plate using the Statcast trajectory data:

```
x(t) = x0 + vx0*t + 0.5*ax*t²
y(t) = y0 + vy0*t + 0.5*ay*t²
z(t) = z0 + vz0*t + 0.5*az*t²
```

Where (x0, y0, z0) = release point, (vx0, vy0, vz0) = initial velocity, (ax, ay, az) = acceleration.

The `y` coordinate runs from pitcher to catcher (~55 feet). We solve for `t` when `y = y_decision` (the decision point, ~40 feet from plate = ~15 feet from release) to get the position at the decision point.

**Tunnel distance** = distance between two pitch types at the decision point
**Plate distance** = distance between them at the plate
**Tunneling score** = plate_distance / tunnel_distance (higher = more deceptive)

## Research Questions

### Q1: Per-Pitcher Tunneling Scores
For every pitcher with 200+ pitches in 2025:
- Compute the **tunnel distance** and **plate distance** for every pair of pitch types they throw (FF-SL, FF-CH, SI-FC, etc.)
- Compute a **tunneling score** per pitch pair: plate_divergence / tunnel_convergence
- Aggregate into an **overall deception score** per pitcher (weighted by usage of each pair)
- Rank all 654 qualified pitchers

**Key output:** Who has the most deceptive arsenal in MLB? Who has the least? Is there a correlation between tunneling score and outcomes (whiff rate, xwOBA-against)?

### Q2: Release Point Consistency
Before computing tunnel distances, measure each pitcher's release point consistency:
- Standard deviation of release_pos_x, release_pos_y, release_pos_z **within** each pitch type
- Standard deviation of release point **across** pitch types (low = better tunneling precondition)
- Are there pitchers with tight release points per pitch type but different release points between types? (This would telegraph pitch type.)

**Key output:** Release point consistency leaderboard. Correlation with tunneling score.

### Q3: Which Pitch Pairs Tunnel Best League-Wide?
Across all pitchers:
- Which pitch-type combinations produce the highest average tunneling scores? (e.g., FF-CH? SI-SL? FC-CU?)
- Is there a "best tunnel pair" that elite pitchers tend to exploit?
- How does the Schlittler FF-SI-FC trio rank in tunnel distance vs. league?

**Key output:** Pitch-pair tunneling heatmap. "The 10 Best Tunnel Combinations in Baseball."

### Q4: Does Tunneling Predict Outcomes?
The critical validation question:
- Correlate per-pitcher tunneling score with: overall whiff rate, CSW%, xwOBA-against, K%
- Control for velocity and stuff quality (spin rate, movement magnitude)
- Does tunneling add predictive value BEYOND raw stuff? If a pitcher has average velocity but elite tunneling, do they outperform stuff-only models?

**Key output:** Regression analysis. "Tunneling explains X% of outcome variance beyond velocity and movement alone."

### Q5: The 2026 Tunneling Update
For pitchers with enough 2026 data:
- Has their tunneling score changed year-over-year?
- Case studies: Schlittler's cutter transformation (we already know the movement changed — did the tunneling improve?)
- Any pitchers whose tunneling degraded in 2026? (Could explain performance drops.)

**Key output:** "Pitchers Whose Deception Changed in 2026" — connects to the Schlittler article.

### Q6: Tunneling vs Batter Handedness
Does tunneling effectiveness differ against LHB vs RHB?
- Some pitch pairs might tunnel beautifully from one batter's perspective but poorly from the other
- Compute separate tunneling scores for vs-LHB and vs-RHB
- Which pitchers have the most asymmetric tunneling profiles?

**Key output:** "The Pitchers Who Are Only Deceptive From One Side."

## Data Available

### In `data/` folder:
- **`statcast_2025_full.parquet`** — Full 2025 season, 739,820 pitches, 654 pitchers with 200+
- **`statcast_2026_apr06_14.parquet`** — 2026 data through Apr 14
- Daily 2026 parquets through Apr 5

### Key Statcast Fields for Tunneling:
| Field | Description | Coverage |
|-------|-------------|----------|
| `release_pos_x` | Horizontal release position (feet) | 99% |
| `release_pos_y` | Distance from plate at release (feet) | 99% |
| `release_pos_z` | Vertical release position (feet) | 99% |
| `vx0`, `vy0`, `vz0` | Initial velocity components (ft/s) | 99% |
| `ax`, `ay`, `az` | Acceleration components (ft/s²) | 99% |
| `pfx_x`, `pfx_z` | Pitch movement (induced, feet) | 99% |
| `plate_x`, `plate_z` | Plate arrival location (feet) | 99% |
| `release_speed` | Velocity at release (mph) | 99% |
| `release_extension` | Extension toward plate (feet) | 99% |
| `release_spin_rate` | Spin rate (rpm) | 98% |
| `spin_axis` | Spin axis (degrees) | 98% |
| `pitch_type` | Pitch classification (FF, SL, CH, etc.) | 99% |
| `pitcher` | Pitcher MLB ID | 100% |
| `player_name` | Pitcher name | 100% |
| `p_throws` | Pitcher handedness (L/R) | 100% |
| `stand` | Batter handedness (L/R) | 100% |
| `description` | Outcome (called_strike, ball, swinging_strike, etc.) | 100% |

### Trajectory Computation Reference
To compute position at any point along the trajectory:

```python
def pitch_position_at_y(row, y_target):
    """Compute (x, z) position of a pitch when it reaches y_target distance from plate."""
    # Statcast coordinates: y=0 at plate, y~55 at rubber
    # release_pos_y is typically ~54-56 feet
    x0 = row['release_pos_x']
    y0 = row['release_pos_y']
    z0 = row['release_pos_z']
    vx0 = row['vx0']
    vy0 = row['vy0']
    vz0 = row['vz0']
    ax_val = row['ax']
    ay_val = row['ay']
    az_val = row['az']

    # Solve y(t) = y_target for t
    # y(t) = y0 + vy0*t + 0.5*ay*t^2 = y_target
    # 0.5*ay*t^2 + vy0*t + (y0 - y_target) = 0
    a = 0.5 * ay_val
    b = vy0
    c = y0 - y_target

    # Quadratic formula — take the positive root
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None, None, None
    t = (-b - (discriminant**0.5)) / (2*a)  # negative root (ball moving toward plate, vy0 < 0)

    x_at_target = x0 + vx0*t + 0.5*ax_val*t**2
    z_at_target = z0 + vz0*t + 0.5*az_val*t**2

    return x_at_target, z_at_target, t
```

**Decision point:** Use `y_target ≈ 23.9 feet` from plate (about 2/3 of the way from release to plate). This is approximately where research suggests the batter's commit/no-commit decision occurs.

**Plate arrival:** Use `y_target ≈ 0` (front of plate) — compare with `plate_x` and `plate_z` as validation.

## Analysis Architecture

### Step 1: Compute Per-Pitch Trajectory Points
For every pitch in 2025:
- Position at release: (release_pos_x, release_pos_y, release_pos_z)
- Position at decision point: compute from trajectory equations at y ≈ 23.9 feet
- Position at plate: (plate_x, _, plate_z)
- Time to decision point, time to plate

### Step 2: Per-Pitcher Pitch-Type Centroids
For each pitcher, for each pitch type with 30+ pitches:
- Mean release point (x, y, z)
- Mean decision-point position (x, z)
- Mean plate position (x, z)
- Standard deviation at each point

### Step 3: Pairwise Tunnel Metrics
For each pitcher, for each pair of pitch types:
- **Release separation** = Euclidean distance between mean release points
- **Decision-point separation** = distance at decision point
- **Plate separation** = distance at plate arrival
- **Tunnel ratio** = plate_separation / decision_separation (higher = more deceptive)
- **Convergence score** = how much separation grows from decision to plate

### Step 4: Aggregate Deception Score
Per pitcher:
- Weight each pitch pair by usage frequency (more common pairs count more)
- Aggregate tunnel ratios into a single **Deception Score**
- Also compute: mean release consistency (SD across all pitch types), tunnel variance

### Step 5: Validate Against Outcomes
- Correlate Deception Score with: whiff%, CSW%, xwOBA-against, K-BB%
- Partial correlations controlling for velocity and movement magnitude
- Test: does tunneling predict outcomes beyond stuff?

## Kill Criteria
- If the tunnel ratio does not correlate with ANY outcome metric (r < 0.05), the metric isn't measuring anything useful — rethink the methodology
- If release point data is too noisy (large within-pitch-type variance), the signal may be drowned out
- If the "best tunnelers" are just "the hardest throwers," the metric isn't adding information beyond velocity

## Expected Output

### Charts (for article + Explore page):
1. **League-wide tunneling leaderboard** — Top 20 / Bottom 20 deception scores
2. **Tunnel map per pitcher** — visual showing pitch positions at release, decision, and plate
3. **Pitch-pair heatmap** — which combinations tunnel best across the league
4. **Tunneling vs outcomes scatter** — does deception predict success?
5. **Year-over-year changes** — who improved/degraded tunneling in 2026?
6. **Case study visualizations** — Schlittler, Skenes, Sale, or whoever emerges as interesting

### Interactive Tool (for Explore page):
- Pitcher search → tunnel map + deception score + pitch-pair breakdown
- League percentile ranking
- Compare any two pitchers' tunnel profiles

### Article(s):
- **Flagship:** "The Pitch Tunneling Atlas: Measuring Deception Across 654 MLB Pitchers"
- **Follow-up:** "The 10 Most Deceptive Pitch Pairs in Baseball"
- **Case study:** "Why [Pitcher X]'s Stuff Plays Up: The Tunneling Advantage"
- **Myth-bust potential:** "Does Tunneling Actually Matter? (The Data Says [Yes/No/It's Complicated])"

## Methodological Notes

### Decision Point Distance
The exact decision point is debated in biomechanics research. Common estimates:
- **23-25 feet from plate** (most frequently cited)
- **150-170ms after release** (time-based rather than distance-based)
- Sensitivity analysis: compute tunnel metrics at 20, 23, 25, and 28 feet and check whether rankings change materially

### Handling Pitcher Handedness
- Compute all metrics separately for LHP and RHP
- When comparing across handedness, mirror the x-coordinates for one group
- Batter handedness interaction: compute vs-LHB and vs-RHB tunnel maps separately

### Minimum Sample Sizes
- Per pitcher: 200+ total pitches (gives 654 pitchers in 2025)
- Per pitch type: 30+ pitches (otherwise too noisy for centroid computation)
- Per pitch pair: both types must meet the 30-pitch minimum

### Weighting
- Weight pitch pairs by `min(usage_pct_A, usage_pct_B)` — a rare pitch paired with a common one shouldn't dominate the deception score
- Alternative: weight by `usage_pct_A × usage_pct_B` (frequency of encountering the pair in sequence)

### Validation Against Savant
- Cross-check plate arrival positions with Statcast `plate_x` and `plate_z`
- If the trajectory computation diverges from reported plate location by >0.5 inches on average, the physics model needs calibration

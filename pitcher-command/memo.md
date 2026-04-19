# Do Pitchers Lose Their Command? — Research Memo v2 (Final)

## Dataset
- **2025 full season**: 729,827 pitches with location data
  - All qualifying appearances (30+ pitches): 6,801
  - **True starts only: 4,892**
  - Non-starter appearances (excluded): 1,909
- **2026 early season**: 38,092 pitches
  - True starts only: ~247
- Starter identification: pitcher who threw the first pitch in the game for each half-inning side
- Source: Statcast via pybaseball (`statcast_2025.parquet`)

## Methodology
For each **starter** appearance (30+ pitches):
1. Split into first third and last third of pitches thrown
2. Compute **plate-location scatter** = sqrt(std(plate_x)^2 + std(plate_z)^2) * 12 (inches)
3. Compute **change %** = (late_scatter - early_scatter) / early_scatter * 100
4. Classify: consistent (<10% change), blow-up (>20% increase), tighten (>20% decrease)

**Metric note**: plate_x and plate_z measure where the ball crosses home plate, not the catcher's target. This captures plate-location dispersion — a proxy for command, but not command in the biomechanical sense (which requires target data). See Kusafuka et al. 2020.

**"Late" note**: Last third of the outing, not necessarily "late in the game." This is within-outing dispersion change, not proof of fatigue.

## Key Findings (all starter-only, 2025)

### 1. Average Plate-Location Scatter Is Flat
Per-pitch-count scatter:
- Pitches 1-15: 15.3" | Pitches 46-60: 15.2" | Pitches 91-105: 15.3"

Confirms r = 0.007. The population mean doesn't move.

### 2. The Distribution Is Asymmetric
4,892 true starts:
- 49.7% consistent (<10% change)
- **14.0% blow-up** (>20% increase)
- **5.2% tighten** (>20% decrease)
- Ratio: 2.7:1 blow-up to tighten

Pitch-type-adjusted check: 12.9% blow-up vs 5.9% tighten. Not a pitch-mix artifact.

### 3. Outing Length Stratification
| Outing Length | Starts | Blow-up (>20%) | Tighten (>20%) |
|--------------|--------|---------------|----------------|
| 30-59 pitches | 248 | 23.4% | 9.7% |
| 60-89 pitches | 2,371 | 14.7% | 6.0% |
| 90+ pitches | 2,273 | 12.2% | 3.9% |

Short outings have the HIGHEST blow-up rate. This likely reflects selection: pitchers pulled early are often pulled BECAUSE they're wild. True long starts (90+) have the lowest blow-up rate — the pitchers who go deep tend to maintain their scatter.

### 4. Pitcher Profiles (starter-only, 10+ true starts)

**Most scatter increase:**
| Pitcher | Starts | Avg Change | Blow-up Rate |
|---------|--------|-----------|-------------|
| Garrett Crochet | 33 | +31.5% | 12% |
| Carmen Mlodzinski | 12 | +20.7% | 58% |
| Slade Cecconi | 24 | +18.1% | 38% |
| Bowden Francis | 15 | +15.5% | 27% |
| Jacob Misiorowski | 14 | +14.1% | 36% |
| Marcus Stroman | 10 | +13.7% | 30% |
| Michael Soroka | 17 | +13.3% | 35% |
| Kumar Rocker | 15 | +13.0% | 33% |

**Most consistent (gets tighter or stays stable):**
| Pitcher | Starts | Avg Change | Blow-up Rate |
|---------|--------|-----------|-------------|
| Ben Brown | 15 | -12.0% | 0% |
| Colton Gordon | 14 | -10.8% | 0% |
| David Festa | 10 | -9.8% | 0% |
| Luis Ortiz | 17 | -8.6% | 0% |
| Max Meyer | 13 | -6.4% | 8% |
| Trevor Williams | 17 | -5.4% | 0% |
| Kodai Senga | 22 | -5.2% | 5% |
| Nick Pivetta | 33 | -4.4% | 0% |

### 5. Cross-Season Persistence Cannot Be Established Yet
With ~2 starts per pitcher in 2026, cross-season signal is noise-dominated. Mlodzinski is +20.7% in 2025 and +18.8% in 2026 (suggestive, but n=2 in 2026). Most pitchers don't replicate. Needs full 2026 season.

## Proposed Article Structure

### Title: "Do Pitchers Lose Their Command?"

1. **The Setup**: The broadcaster says "he's losing his command." Our earlier analysis said no (r = 0.007). A reader asked: what about variance?

2. **The Answer Is Nuanced**: Average scatter is flat. But the distribution is asymmetric — 14.0% of starts produce a >20% scatter spike, vs only 5.2% that tighten. The effect is heterogeneous, not universal.

3. **The Spray Charts**: Side-by-side early vs late for Crochet (worst blow-up tendency), Cecconi (high blow-up rate), and Ben Brown (most consistent, 0% blow-up across 15 starts).

4. **The Outing Length Puzzle**: Short starts have HIGHER blow-up rates. Selection bias — pitchers pulled early are often pulled because they're wild.

5. **The Pitcher Profiles**: Starter-only leaderboard. Crochet's 33 starts at +31.5% avg. Festa's 10 starts at 0% blow-up rate.

6. **What This Is Not**: This is plate-location dispersion, not command in the biomechanical sense. We don't have target data.

## Data Location
- 2025 raw data: `analysis-0405/baseball/data/raw/statcast_2025.parquet`
- Starter-only aggregates: `analysis-abs/data/pitcher_variance_2025_starters.json` (185 pitchers)

## References
- Birfer et al. 2019: Fatigue effects heterogeneous across pitchers
- Kusafuka et al. 2020: Release parameters drive pitch location; plate crossing is proxy
- Wakamiya et al. 2024: Lower release-point variability correlates with better performance

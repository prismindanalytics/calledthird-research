# CalledThird Research

Open-source baseball analytics research from [CalledThird.com](https://calledthird.com).

This repository contains the data pipelines, analysis scripts, and methodology documents behind CalledThird's published research. Each folder corresponds to one flagship article or analysis project.

## Projects

### 🎯 [Pitch Tunneling Atlas](./pitch-tunneling-atlas/)
League-wide pitch tunneling model measuring deception via trajectory physics.

- **Article:** [The Pitch Tunneling Atlas](https://calledthird.com/analysis/pitch-tunneling-atlas)
- **Physics companion:** [The Physics Behind the Tunneling Atlas](https://calledthird.com/analysis/tunneling-atlas-physics)
- **Data:** Full 2025 Statcast (739,820 pitches, 654 pitchers with 200+ pitches)
- **Approach:** Dual-agent independent analysis with cross-review
- **Key finding:** Plate separation adds +9.0% R² to whiff prediction beyond stuff; decision-point tightness adds +0.8% more (p=0.016). Both matter, but plate diversity dominates.

### 🥊 [Bench-Clearing Incidents & Umpire Behavior](./soler-lopez-brawl/)
Do umpires change the zone after bench-clearing incidents?

- **Article:** [After a Fight, the Zone Gets Cleaner](https://calledthird.com/analysis/the-zone-after-a-fight)
- **Data:** 7 Statcast-era incidents (2019-2026), umpire-specific follow-up games
- **Key finding:** No significant zone-size change (p=0.302), but accuracy improves unanimously (+2.0pp, p=0.001). Umpires get more precise, not more aggressive.

## How This Works

CalledThird runs two independent AI research agents (Claude + Codex) on the same hypothesis and data. Each produces an analysis script, a report, and charts. Agents then cross-review each other's work. A comparison memo synthesizes the results. The final published article uses the stronger methodology on each dimension.

This repo publishes the research scripts and methodology documents — **not the Statcast data itself** (available from [pybaseball](https://github.com/jldbc/pybaseball) or [Baseball Savant](https://baseballsavant.mlb.com/)).

## Reproducing a Project

Each project folder includes:
- `RESEARCH_BRIEF.md` — hypothesis, methodology, kill criteria
- `analyze_claude.py` — Claude's analysis script
- `analyze_codex.py` — Codex's analysis script
- `COMPARISON_MEMO.md` — cross-review synthesis

To reproduce:

1. Install dependencies: `pip install pybaseball pandas numpy scipy statsmodels matplotlib`
2. Pull Statcast data as specified in the project's `RESEARCH_BRIEF.md`
3. Run either analysis script; they write reports and charts to subdirectories

## Methodology Notes

- **Statistical rigor:** All claims include p-values, confidence intervals, and sample sizes
- **Physics transparency:** Trajectory models validated against Statcast ground truth; limitations documented
- **Kill criteria:** Every project specifies what result would NOT be publishable before analysis begins
- **Dual validation:** All flagship findings require agreement across two independent agents

## License

All research content is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You're welcome to reproduce, extend, or critique these analyses. Please cite:

> CalledThird (2026). "[Article Title]." CalledThird.com. https://calledthird.com/analysis/[slug]

## Contact

- Site: [calledthird.com](https://calledthird.com)
- Twitter/X: [@CalledThirdMLB](https://x.com/CalledThirdMLB)
- Bluesky: [@calledthird.com](https://bsky.app/profile/calledthird.com)
- Data inquiries: hello@calledthird.com

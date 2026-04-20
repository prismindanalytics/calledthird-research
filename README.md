# CalledThird Research

Open-source baseball analytics research from [CalledThird.com](https://calledthird.com).

This repository contains the data pipelines, analysis scripts, and methodology documents behind CalledThird's published research. Each folder corresponds to one flagship article or analysis project.

## Projects

### 🧭 [The Coaching Gap](./coaching-gap/)
Does pitcher predictability translate into hitter wOBA — and if so, *which* hitters actually extract the edge?

- **Article:** [The Coaching Gap That Lives Where Hitters Don't Chase](https://calledthird.com/analysis/coaching-gap-patience)
- **Live tracker:** [Explore → Coaching Gap](https://calledthird.com/explore#coaching-gap) (chase trajectories, live 2025→2026 scatter)
- **Data:** ~2.9M pitches across 5 seasons (2022–2026); 371-pitcher pre-registered cohort; 659 completed hitter-season transitions
- **Approach:** Six rounds of dual-agent (Claude + Codex) independent analysis with cross-review at every round; 17 hypotheses pre-registered with kill criteria
- **Key finding:** 16 of 17 hypotheses died. The one survivor: **low-chase hitters extract ~0.04 more wOBA on predictable pitches than chasers do** (pooled/FE converged across both agents; replicates every season 2022–2026 including strict 2026 holdout). Mechanism validated at the per-hitter level — reducing overall chase by 1pp cuts chase on predictable bait by ~1pp too (Spearman +0.53 across 659 transitions).

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
- **Data:** 7 Statcast-era incidents (2019–2026), umpire-specific follow-up games
- **Key finding:** No significant zone-size change (p=0.302), but accuracy improves unanimously (+2.0pp, p=0.001). Umpires get more precise, not more aggressive.

### 🔥 [The Fireman's Dilemma](./firemans-dilemma/)
How much of a reliever's inherited-runner outcome is entry situation vs individual skill?

- **Article:** [The Fireman's Dilemma](https://calledthird.com/analysis/the-firemans-dilemma)
- **Data:** 4,044 reliever entries, 6,516 inherited runners (2025), MLB play-by-play `responsiblePitcher` attribution
- **Key finding:** Outs gradient dominates — 44% strand at 0 outs, 61% at 1 out, 82% at 2 outs. League strand rate 68.3%. Cross-season skill persistence is near-zero (r=0.098) but 2026 samples are thin.

### ⚾ [The Schlittler Three-Fastball Blueprint](./schlittler-arsenal/)
How a rookie's three distinct fastballs (sinker / four-seamer / cutter) complement each other.

- **Article:** [The Schlittler Three-Fastball Blueprint](https://calledthird.com/analysis/schlittler-three-fastball-blueprint)
- **Data:** Schlittler's 2026 pitches via Statcast + Baseball Savant arsenal context
- **Key finding:** The three fastballs occupy distinct horizontal movement bands, creating a deception grid where hitters can't sit on one shape.

### 🧠 [The Count That Matters Most](./abs-count-leverage/)
Which counts deliver the highest RE288 value per ABS challenge, and how do pitchers vs hitters differ?

- **Article:** [The Count That Matters Most](https://calledthird.com/analysis/the-count-that-matters)
- **Data:** All 2026 ABS challenges with RE288 count-state linear weights
- **Key finding:** Value per challenge swings by 3×+ across counts. Hitters and pitchers have different optimal challenge counts.

### 🥎 [Catchers Are Better Challengers Than Hitters](./abs-catcher-edge/)
Why catchers succeed more often on ABS challenges than hitters do.

- **Article:** [Catchers Are Better Challengers](https://calledthird.com/analysis/catchers-are-better-challengers)
- **Data:** 2026 ABS challenges split by challenger role (pitcher / catcher / hitter)
- **Key finding:** Catchers lead in success rate by a significant margin, consistent with framing-era ball/strike perception advantage.

### 🧮 [Team Challenge IQ](./team-challenge-iq/)
Which teams challenge smartly (high success, normalized per game) vs which over-challenge?

- **Article:** [Twins vs Reds: A Tale of Two Challenge Strategies](https://calledthird.com/analysis/twins-vs-reds-abs)
- **Live tool:** [Explore → Team Strategy](https://calledthird.com/explore)
- **Data:** Every 2026 ABS challenge, normalized per game, joined with outcome
- **Key finding:** Minnesota and Cincinnati anchor opposite ends of the challenge-efficiency spectrum with similar volume.

### 🎯 [Do Pitchers Lose Their Command?](./pitcher-command/)
Within-outing plate-location scatter change from the first third to the last third of starts.

- **Article:** [Do Pitchers Lose Their Command?](https://calledthird.com/analysis/do-pitchers-lose-command)
- **Data:** 4,892 true starts in 2025 (30+ pitches), 729,827 pitches total
- **Key finding:** Population mean scatter is flat across pitch counts (r=0.007), but distribution is asymmetric — 14.0% blow up vs 5.2% tighten (2.7:1 ratio).

### 👨‍⚖️ [CB Bucknor By The Numbers](./bucknor-profile/)
A data profile of one of MLB's most-derided umpires vs the 82-umpire field.

- **Article:** [CB Bucknor By The Numbers](https://calledthird.com/analysis/cb-bucknor-by-the-numbers)
- **Data:** 2025 umpire personality dataset (83 qualified umps) + 2026 nightly cache
- **Key finding:** 3rd-worst accuracy (91.02%, p=0.0002 vs league) and #1 worst miss distance (1.34"). The miss magnitude, not the zone shape, is what makes Bucknor an outlier.

## How This Works

CalledThird runs two independent AI research agents (Claude + Codex) on the same hypothesis and data for flagship projects. Each produces an analysis script, a report, and charts. Agents cross-review each other's work. A comparison memo synthesizes the results. The final published article uses the stronger methodology on each dimension.

This repo publishes the research scripts and methodology documents — **not the Statcast data itself** (available from [pybaseball](https://github.com/jldbc/pybaseball) or [Baseball Savant](https://baseballsavant.mlb.com/)).

## Reproducing a Project

Each project folder includes a `README.md` describing the question, data, and findings. Most folders also include:
- A research brief or proposal (`RESEARCH_BRIEF.md`, `RESEARCH_PROPOSAL.md`)
- One or more analysis scripts (`analyze_*.py`, `compute_*.py`, `pull_*.py`)
- A findings memo (`memo.md`, `findings.md`, `COMPARISON_MEMO.md`)

To reproduce:

1. Install dependencies: `pip install pybaseball pandas numpy scipy statsmodels matplotlib`
2. Pull Statcast data as specified in the project's brief
3. Run the analysis script; it writes reports and charts to subdirectories

## Methodology Notes

- **Statistical rigor:** All claims include p-values, confidence intervals, and sample sizes
- **Physics transparency:** Trajectory models validated against Statcast ground truth; limitations documented
- **Kill criteria:** Every project specifies what result would NOT be publishable before analysis begins
- **Dual validation:** Flagship findings require agreement across two independent agents

## License

All research content is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You're welcome to reproduce, extend, or critique these analyses. Please cite:

> CalledThird (2026). "[Article Title]." CalledThird.com. https://calledthird.com/analysis/[slug]

## Contact

- Site: [calledthird.com](https://calledthird.com)
- Twitter/X: [@CalledThirdMLB](https://x.com/CalledThirdMLB)
- Bluesky: [@calledthird.com](https://bsky.app/profile/calledthird.com)
- Data inquiries: hello@calledthird.com

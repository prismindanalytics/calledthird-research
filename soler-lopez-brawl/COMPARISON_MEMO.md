# Comparison Memo: Independent Analysis vs Codex Analysis

**Date:** April 10, 2026
**Subject:** Methodological comparison of two independent analyses of post-brawl umpire zone behavior

---

## Bottom Line

Both analyses point to the same broad read on overall called-strike rate: there is no evidence of a large post-brawl zone shift. But the Codex version is methodologically stronger. It grades pitches against a ball-radius-adjusted rulebook zone, compares each follow-up game against the umpire's own per-game control distribution, and isolates the specific place where a signal appears: fewer called strikes off the plate in the next game. That turns a vague "maybe the zone tightens a bit" result into a sharper and more publishable claim: the overall zone does not move much, but the freebies just off the edge appear to disappear.

---

## Where the Codex Analysis Is Stronger

### 1. Zone Geometry (ball-radius expansion)

The Codex script expands the rulebook zone by one baseball radius (1.45"/12 = 0.121 ft), matching the MLB rule that a pitch only needs to clip the zone. The independent analysis originally used plate-center geometry, systematically misclassifying legal edge strikes as "outside the zone." (This has since been corrected.)

### 2. Per-Game Control Distribution

This is the single biggest structural advantage. The Codex analysis builds a game-level metric distribution for each umpire's season, then places each next game within that distribution using percentiles. The independent analysis collapses the entire season into one number per umpire, which loses game-to-game variance information and makes statistical comparisons less defensible.

### 3. Metric Decomposition

The Codex analysis breaks the zone into five distinct metrics:
- Out-of-zone strike rate (the strongest signal: -2.6pp, 6/6 below baseline)
- Shadow band strike rate (-9.3pp, suggestive)
- In-zone ball rate
- Rulebook miss rate (-2.0pp, 6/6 below baseline)
- Overall called-strike rate (no significant change)

The independent analysis uses three broader metrics (CSR, accuracy, borderline rate) that dilute the out-of-zone signal into a wider bucket.

### 4. The Edge-Distance Profile Chart

The called-strike probability curve by signed inches from the zone edge is the best single visualization across either analysis. It shows exactly where behavior changes: the -4 to 0 inch band (off the plate but close). No equivalent exists in the independent analysis.

### 5. Percentile Framing

Showing each next game's percentile rank within the umpire's own season distribution (e.g., "4th percentile for OOZ strikes") is more informative than raw deltas. The out-of-zone finding becomes visceral: every single next game is below the 50th percentile.

### 6. Soler-Lopez Watchlist Framing

The Codex analysis correctly frames the 2026 incident as an incomplete case with a dedicated "live watch" chart showing Moscoso's incident game at the 6th percentile for CSR. The independent analysis originally included it silently with N/A or zero-filled values.

---

## Where the Independent Analysis Has Something the Codex Doesn't

### 1. Zone Heatmap Scatter Plots
Pooled called-pitch scatter plots (chart 5) give spatial intuition about where strikes and balls cluster. Codex has no equivalent spatial visualization.

### 2. Trajectory Slope Chart
The baseline-to-incident-to-next-game slope chart (chart 6) shows narrative flow per incident. Useful for editorial storytelling, though the signal is noisy.

---

## What the Two Analyses Do NOT Agree On

The p-values are not comparable:
- Independent analysis originally tested incident-vs-next (paired t-test, p=0.394) — a different question from "does the next game differ from baseline?"
- Codex tested next-vs-baseline deltas (one-sample t-test, p=0.286) — the question the research brief actually asks

After correction, the independent analysis now runs the proper next-vs-baseline test and gets CSR p=0.302 (consistent with Codex's 0.286) and accuracy p=0.001 (a finding the Codex analysis supports through its miss-rate metric, p=0.001).

The out-of-zone finding (Codex: p=0.004, 6/6 consistent direction) should be characterized as the strongest exploratory signal, not a confirmed finding, given n=6 and multiple related metrics inspected.

---

## Recommended Publishable Angle

Both analyses support the same editorial framing:

**"After a Fight, the Zone Doesn't Get Bigger. It Gets Cleaner."**

The zone size stays the same. What changes is the quality of calls — specifically, fewer freebies off the plate. This is the Codex analysis's formulation, and it is the stronger and more publishable claim.

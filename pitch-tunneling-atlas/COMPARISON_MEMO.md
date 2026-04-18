# Comparison Memo: Claude vs Codex Pitch Tunneling Analyses

Prepared on April 18, 2026 from:
- [claude-reviews-codex.md](/Users/haohu/Documents/GitHub/calledthird/research/pitch-tunneling-atlas/reviews/claude-reviews-codex.md)
- [codex-reviews-claude.md](/Users/haohu/Documents/GitHub/calledthird/research/pitch-tunneling-atlas/reviews/codex-reviews-claude.md)

## Executive Summary
Both cross-reviews reached the same top-line verdict: **REVISE**. Neither analysis should be published unchanged as the definitive Pitch Tunneling Atlas.

The clearest synthesis is:
- **Claude has the stronger main “deception / whiff” metric** via divergence (`plate separation - decision separation`) and the stronger editorial insight via the R² decomposition.
- **Codex has the stronger framework coverage**: explicit physics validation, batter-handedness splits, 2026 year-over-year work, broader outcome coverage, and closer adherence to the original brief.
- The final article should use a **hybrid methodology**, not choose one analysis wholesale.

## 1. Convergent Findings: High-Confidence

These are the findings both analyses support, either directly or via both cross-reviews.

### High Confidence
- **The project is publishable only after revision.**
  Both reviews explicitly recommend `REVISE`.

- **FF-KC is the best league-wide tunnel pair family.**
  Claude ranks `FF-KC` first by divergence (`9.8"` average divergence, `n=40` pitchers).
  Codex ranks `FF-KC` first by stabilized tunnel ratio (`6.48`, `2.87"` decision separation, `13.00"` plate separation, `n=40`).
  The exact score depends on metric, but the rank-order result is robust.

- **Several pair families recur in both top groups.**
  Shared top-tier combinations:
  - `FF-KC`
  - `FF-SL`
  - `FF-ST`
  - `CH-ST`
  - `SI-SL`

- **Taylor Rogers is the clearest bottom-of-the-league anti-tunneling case.**
  Both analyses put him last or effectively last.
  Both also place `Anthony Bender` and `Joey Cantillo` in the bottom cluster.

- **The strongest outcome relationship is with whiff rate, not broader prevention metrics.**
  Claude finds divergence vs whiff `r = 0.355`, partial `r = 0.319`.
  Codex finds stabilized-ratio vs whiff `r = 0.175`, partial `r = 0.133`.
  Both analyses report weak-to-negligible CSW and xwOBA signal relative to whiff.

- **The right editorial framing is measured, not revolutionary.**
  Both reviews reject a sweeping “tunneling wins games” claim.
  The defensible story is: tunneling/deception is real, but modest, and is not the dominant driver of overall run prevention.

### Medium-High Confidence
- **Kimbrel, Pomeranz, and Megill belong in the top deception cluster.**
  The exact ranks differ, but both analyses consistently place them near the top.

- **Relievers dominate the top of the board more than starters.**
  Both analyses see this pattern, though the strength of the effect depends on the metric and weighting.

## 2. Divergent Findings: Needs Resolution

These are the questions that materially change the published leaderboard or headline numbers.

### A. Primary Metric
- **Claude:** divergence = `plate_sep - decision_sep`
- **Codex:** stabilized tunnel ratio = `plate_sep / max(decision_sep, 1 inch)`

Why this matters:
- Claude’s metric has materially stronger whiff signal.
- Codex’s metric is closer to the original brief’s “true tunneling” concept.
- The choice changes the leaderboard substantially:
  - top-20 overlap: `6/20`
  - bottom-20 overlap: `13/20`
  - overall rank Spearman: `0.59`

This is the biggest unresolved issue.

### B. Physics / Coordinate Handling
- **Claude:** uses `release_pos_y` in the decision-point solve and then bypasses trajectory-based plate coordinates by using Statcast `plate_x` / `plate_z`.
- **Codex:** calibrates to a `50 ft` trajectory reference and explicitly validates plate error, but still misses the brief’s sub-inch target.

Current plate-validation status:
- Claude-style solve: about `4.93"` mean Euclidean plate miss
- Codex calibrated solve: about `2.29"` mean Euclidean plate miss

Both fail the brief’s `0.5"` calibration target. This does not kill the project, but it means the physics section must be framed as **relative-modeling**, not literal flight-path reconstruction.

### C. Decision Distance
- **Claude:** `23.9 ft`
- **Codex:** `23.0 ft`

The brief specifies approximately `23.9 ft`, so the final rerun should use `23.9 ft` as the main setting and keep `20 / 23.9 / 25 / 28` in the sensitivity appendix.

### D. Pair Weighting
- **Claude:** `usage_a × usage_b`
- **Codex:** `min(usage_a, usage_b)`

Both are defensible and both appear in the brief. This is not a fatal disagreement, but it should be standardized before publication.

### E. Scope Coverage
- **Claude misses:** batter-handedness splits, 2026 update, xwOBA/K outcomes
- **Codex misses:** the plate-vs-decision R² decomposition that best explains why the two metrics behave differently

The final article should not accept either omission.

## 3. Recommended Methodology by Component

| Component | Recommended Source | Final Recommendation |
| --- | --- | --- |
| Main leaderboard metric | Claude | Use **divergence** for the flagship “deception” leaderboard because it has the stronger whiff signal. Do not call it pure tunneling without qualification. |
| True tunneling appendix metric | Codex-style concept | Keep a **separate tunneling metric** based on decision-point similarity vs late spread for methodology notes / appendix. This can be a stabilized ratio or a composite, but it should not replace divergence as the main article score. |
| Decision-point distance | Brief / Claude | Use **23.9 ft** as the default, with sensitivity at `20 / 25 / 28 ft`. |
| Physics validation | Codex | Use Codex’s explicit validation workflow, but rerun it under the final unified coordinate convention. State clearly that current models are valid for relative comparisons, not literal sub-inch flight reconstruction. |
| Plate vs decision decomposition | Claude | Use Claude’s **R² decomposition** as the main explanatory device in the story. |
| Regression controls | Codex | Use Codex’s broader control set: **velocity + movement + spin**, then add the decomposition variables. |
| Outcome set | Codex + Claude | Lead with **whiff**. Include CSW and xwOBA as secondary checks, but do not oversell them. |
| Batter-handedness splits (Q6) | Codex | Keep the Codex framework, but **recompute with divergence** rather than the ratio. |
| 2026 year-over-year (Q5) | Codex | Keep the Codex framework, but **recompute with divergence** and present as a small-sample supplement, not a headline table. |
| Release consistency | Codex | Use full release-point treatment and report it as background context, not as the lead result. |
| Pitch-pair heatmap | Shared | Publish the shared pair-family results because this is one of the highest-agreement sections. |

## 4. Publishable Numbers / Rankings With Confidence Levels

### Publish With High Confidence
- **Best pair family:** `FF-KC`
  - Confidence: **High**
  - Publish as: “The best league-wide tunnel family is four-seam + knuckle-curve.”
  - Avoid overcommitting to one exact score unless the final rerun is unified.

- **Other elite pair families:** `FF-SL`, `FF-ST`, `SI-SL`, `CH-ST`
  - Confidence: **High**
  - Publish as a consensus tier, not a razor-precise order.

- **Worst individual case:** `Taylor Rogers`
  - Confidence: **High**
  - Publish as the clearest anti-tunneling / convergence example.

- **Bottom cluster:** `Taylor Rogers`, `Anthony Bender`, `Joey Cantillo`
  - Confidence: **High**
  - These are supported by both analyses.

- **Outcome framing:** “Whiff is where the signal lives; CSW and xwOBA are much weaker.”
  - Confidence: **High**
  - Both analyses agree on this directionally.

### Publish With Medium Confidence
- **Top deceptive pitcher cluster:** `Craig Kimbrel`, `Drew Pomeranz`, `Trevor Megill`
  - Confidence: **Medium**
  - Publish these names as consensus top-cluster pitchers.
  - Do **not** publish a definitive exact ordering among them until the unified rerun.

- **Core decomposition stat:** plate separation explains about `8.9%` additional whiff-rate variance beyond velo+spin, while decision-point closeness explains about `1.0%` more once plate separation is included.
  - Confidence: **Medium**
  - This is Claude-only but is the strongest explanatory insight in either analysis.
  - It should be rerun in the final merged pipeline before becoming the headline stat.

- **Divergence-to-whiff relationship:** roughly `r = 0.35`
  - Confidence: **Medium**
  - Strong enough to use directionally.
  - Recompute in the merged pipeline before quoting as the final exact value.

### Publish With Low Confidence or Hold Back
- **Exact overall #1 pitcher**
  - Confidence: **Low**
  - Claude says `Kimbrel`, Codex says `Beeter`.
  - This should not be published until the final metric is standardized.

- **Any exact full top-20 order**
  - Confidence: **Low**
  - The metric choice currently drives too much of the ordering.

- **2026 movers table**
  - Confidence: **Low**
  - Keep as a supplemental note or 1-2 case studies only.

- **Exact physics-validation success claim**
  - Confidence: **Low**
  - The only high-confidence statement is that both current models fail the brief’s sub-inch target.

## 5. Recommended Editorial Packaging

### Include
- Consensus pair-family heatmap
- Consensus bottom cases (`Rogers`, `Bender`, `Cantillo`)
- Consensus top-cluster pitchers (`Kimbrel`, `Pomeranz`, `Megill`)
- Divergence-based main leaderboard after unified rerun
- Plate-vs-decision decomposition
- Batter-handedness split table, recomputed with divergence
- 2026 update as a short supplement, recomputed with divergence

### Exclude
- A single “definitive tunneling leaderboard” built from either current pipeline unchanged
- Any claim that the physics model was validated to publication-grade sub-inch precision
- Strong xwOBA or run-prevention claims
- An exact #1 pitcher claim before a unified rerun

## Final Recommendation

For the final article, use this synthesis:

1. **Main article score:** Claude-style **divergence**
2. **Main explanatory result:** Claude-style **plate-vs-decision R² decomposition**
3. **Validation / controls / splits / 2026:** Codex framework, **rerun on the divergence metric**
4. **Methodology note:** keep a separate tunneling-specific metric in the appendix so the article does not confuse “late separation” with “true tunneling”

In short:
- **Use Claude’s metric for the headline story**
- **Use Codex’s infrastructure for the fuller analysis**
- **Do not publish either analysis unchanged**

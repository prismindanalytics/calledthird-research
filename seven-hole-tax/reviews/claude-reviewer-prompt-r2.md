# Cross-Review Prompt — Claude reviews Codex (Round 2)

You previously completed Round 2 of CalledThird's "7-Hole Tax" research as Agent A (Claude). Your work is in `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/claude-analysis-r2/`. Your verdict was **comprehensive-debunk** — all four R2 hypotheses null after hierarchical shrinkage and BH-FDR correction; 0 of 78 qualifying umpires flagged.

Agent B (Codex) ran an independent ML-engineering analysis. Their work is in `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/codex-analysis-r2/`. Codex flagged **5 umpires** with q<0.10 and |effect|≥2pp — but **all 5 in the REVERSE direction** of FanSided's claim (favoring 7-hole batters with bigger zone, not penalizing them).

That's the core divergence: Claude (Bayesian shrinkage) finds 0 flagged umpires; Codex (bootstrap + BH-FDR on per-umpire counterfactuals) finds 5 reverse-direction umpires. Cross-review must resolve which finding is publishable.

## Your task

Read Codex's full Round 2 work — at minimum:
- `codex-analysis-r2/REPORT.md`
- `codex-analysis-r2/findings.json`
- `codex-analysis-r2/READY_FOR_REVIEW.md`
- The model code (`h4_per_umpire.py`, `h5_per_hitter.py`, `h6_catcher_mechanism.py`, `h7_chase_interaction.py`, `modeling_r2.py`)
- Sample chart PNGs in `codex-analysis-r2/charts/` (especially the H4 leaderboard)
- `codex-analysis-r2/charts/model_diagnostics/` for calibration

Then write `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/r2-claude-review-of-codex.md` (~800 words) as a **skeptical FanGraphs-grade peer reviewer**.

## Priority focus areas

### 1. The 5 flagged umpires (PRIORITY)
- Names, sample sizes, effect magnitudes, q-values
- Are these the same umpires that show large per-umpire posterior medians in your Bayesian fit (i.e., is your shrinkage absorbing real signal, or is Codex's bootstrap finding artifact)?
- With ~80 umpires and BH-FDR at q<0.10, what's the expected false-discovery rate? Could 5 reverse-direction umpires be exactly the FDR-tolerated false-positive count?
- **Calibration check.** Round 1 had a known issue where Codex's GBM compressed predicted probabilities into a narrow band [~0.41, ~0.64], producing artificially tight CIs that didn't reflect actual uncertainty. Is this happening again in Round 2's per-umpire model? Look at `codex-analysis-r2/charts/model_diagnostics/` for the calibration curve. If the per-umpire predicted-prob deltas are tightly clumped because the GBM is uncalibrated, the bootstrap CIs are artifacts and the q-values are not credible.
- **Bootstrap stability.** Did Codex re-bootstrap with different seeds? Do the same 5 umpires emerge across seeds?
- **All-reverse-direction is suspicious.** If umpires were genuinely heterogeneous on lineup-spot bias, you'd expect ~50/50 directional balance among the credibly-non-zero ones. All 5 in one direction either means (a) a real one-sided systematic effect (umpires are *favoring* the bottom of the order — interesting story!) or (b) a methodology artifact correlated with umpire-level features (e.g., umpire fame quartile correlated with lineup_spot in their training set, leaking signal).

### 2. H5 hitter divergence
- Codex flagged 0 hitters; you flagged 3 BH-FDR-credible (Cam Smith, Pete Crow-Armstrong, Henry Davis) — all in the *favored* direction.
- Is Codex's bootstrap simply too conservative, or is your posterior-predictive simulation too generous? Compare your residual magnitudes for those 3 hitters with whatever Codex computed for the same hitters.
- Sample sizes for Smith/Davis are thin (~40 borderline takes). Are they outliers Codex's bootstrap properly fails to flag while your posterior over-flags?

### 3. H7 directional signal
- You report low-chase spot-7-vs-3 = -1.45pp [-3.98, +1.03], P(neg)=0.87. Codex reports -0.78pp [-0.88, -0.66].
- Codex's [-0.88, -0.66] interval is suspiciously narrow given thin n. Likely the same calibration-compression artifact as Round 1.
- Check Codex's H7 implementation for: bootstrap method (paired vs full-refit), calibration of the underlying model, whether the interaction is fit via separate per-tertile models or via a single model with interaction terms.

### 4. Methodology audit
- Does Codex's model include all required controls (location, count, pitcher quality, catcher tier, umpire) per the brief?
- Is the per-umpire counterfactual computed correctly (predict same pitch with lineup_spot flipped, holding everything else)?
- Are sample-size thresholds applied (≥50 borderline calls in spots 7-9 AND in spots 1-3)?
- Is BH-FDR applied across all qualifying umpires or only a subset?

### 5. Things Codex got right
- Be honest. Where is Codex's analysis stronger than yours? (E.g., the per-umpire counterfactual structure may be more transparent than your random-slope posterior; permutation-importance baselines may catch overfitting your fits don't.)

## Format

```markdown
# Round 2 — Claude's Review of Codex

## Headline assessment
{One paragraph: do you trust Codex's "umpire-only with reverse-direction caveat" conclusion?}

## Critical issues (potential blockers)
1. {Issue with file:line reference and specific claim}
2. ...

## Methodology concerns (non-blocking)
1. ...

## The H4 divergence — 5 flagged umpires vs 0 (priority section)
{Specific reconciliation: are Codex's 5 umpires real or artifact? Convergence test against your own per-umpire posterior medians.}

## H5 divergence (3 favored-direction hitters vs 0)
{Are your hitters real? Does Codex's bootstrap miss them legitimately or over-conservatively?}

## Things Codex got right
1. ...

## Recommendation for the comparison memo
{One paragraph: what should the editorial layer publish, given the convergence/divergence pattern?}
```

Target ~800 words. Be specific, reference exact claims and file:line, pretend you're submitting to FanGraphs. Don't be polite where it would mislead.

When done, the file at `reviews/r2-claude-review-of-codex.md` is the only deliverable.

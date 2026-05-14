# Cross-Review Prompt — Claude reviews Codex (ABS Walk Spike R2)

You previously completed Round 2 of CalledThird's "ABS Walk Spike" research as Agent A (Claude). Your work is in `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/claude-analysis-r2/`. Your headline:

- **H1: +0.68pp [+0.31, +1.04]** walk-rate spike persists but narrowed from R1's +0.82pp
- **H3: −64.6% [−80.6, −49.4] zone attribution** — a SIGN-FLIP from R1's +40-50%. Interpretation: under 2025 zone applied to 2026 pitches, MORE walks would have happened than we observe (pitchers overcompensated via location shift)
- **H5: 0-0 mystery resolved with OPPOSITE direction** from R1 hypothesis. Top edge dropped MORE strikes at 0-0 (−10.48pp) than at 2-strike (−3.73pp); DiD −6.76pp credible
- Branch: `adaptation`

Agent B (Codex) ran an independent ML analysis at `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/codex-analysis-r2/`. Their headline:

- **H1: +0.66pp [+0.27, +1.04]** — virtually identical to yours
- **H3: +35.3% [+34.6, +36.0]** — *positive*, modest drift from R1's +40-50%
- Edge attribution: top edge +66.2%, bottom edge −56.1%
- H5: heart 0-0 CS delta ~0pp; top-edge 2-strike −3.17pp; SHAP supports count-dependent
- Branch: `adaptation` (same as you)

**The critical divergence:** H3 zone attribution. Claude says −64.6%. Codex says +35.3%. Same recommendation (`adaptation`), but opposite signs on the headline number. This must be resolved before publication.

## Your task

Read Codex's full Round 2 work — at minimum:
- `codex-analysis-r2/REPORT.md`
- `codex-analysis-r2/findings.json`
- `codex-analysis-r2/READY_FOR_REVIEW.md`
- Their counterfactual implementation: `codex-analysis-r2/h3_counterfactual.py`
- The model diagnostics: `codex-analysis-r2/charts/model_diagnostics/`

Then write `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/reviews/r2-claude-review-of-codex.md` (~1000 words) as a skeptical FanGraphs-grade peer reviewer.

## Priority focus: the H3 sign-flip

This is the centerpiece. Investigate WHY you got −64.6% and Codex got +35.3%. Specific things to check in `codex-analysis-r2/h3_counterfactual.py`:

1. **Counterfactual definition.** Is Codex defining the counterfactual as (a) "replace each 2026 pitch's predicted-strike probability with what the 2025 classifier would predict for that location" — same as you? Or (b) "replay the entire PA pitch-by-pitch using the 2025 classifier"? Or (c) something else?

2. **Sequencing.** Yours uses full pitch-by-pitch PA replay with the 2025 classifier's predicted probabilities driving each at-bat to its conclusion. Does Codex do the same? Or do they use a single-pitch counterfactual that doesn't propagate through count-state transitions?

3. **Aggregation.** Per-row paired-prediction (the artifact we caught in R2 7-hole-tax) gives an attribution that doesn't account for PA-level sequencing. Did Codex use refit-bagged bootstrap as required by the brief, OR per-row only? Look at `h3_counterfactual.py` for the bootstrap structure.

4. **The mean predicted CS rate.** Your finding: under 2025 classifier, mean predicted-CS on 2026 takes = 0.334, higher than both 2025 and 2026 empirical. This is the key fact — when 2026 pitchers throw to *different locations*, the 2025 classifier predicts MORE strikes overall (because the 2025 classifier was trained on the *old* location distribution). Did Codex compute this aggregate sanity check?

5. **Edge-region attribution.** Codex says top edge +66.2%, bottom edge −56.1%. Those net out toward zero. Your overall result of −64.6% is consistent with the bottom-edge channel dominating in the replay. Reconcile.

6. **Sample window.** Both should be Mar 27 – May 12 (May 13 was partial). Verify Codex's window matches yours.

7. **Calibration.** Look at `codex-analysis-r2/charts/model_diagnostics/` for the zone classifier's calibration curve. If predicted probabilities are compressed (Round 1 H2 artifact), the attribution number is unreliable.

## What you might concede to Codex

Be honest where Codex is stronger:
- If Codex's interpretation of "zone attribution" is the public-facing-correct one (R1 framing was "% of YoY spike attributable to zone change" — closer to +35% than −65%)
- If Codex's calibration is better
- If your sign-flip is actually an interpretation artifact rather than a real adaptation effect

## What you should defend

- The 0-0 H5 resolution (R-hat 1.010, ESS 552, credible DiD of −6.76pp on a clean 4-cell × 4-cell × year design)
- The pitcher-adaptation framing (both agents agree)
- The H1 narrowing of the spike

## Format

```markdown
# Round 2 — Claude's Review of Codex (ABS Walk Spike)

## Headline assessment
{One paragraph: do you trust Codex's +35.3% conclusion? Where does it agree/disagree with your -64.6%?}

## Critical issues (potential blockers)
1. {Issue with file:line reference}
2. ...

## The H3 sign-flip (priority section, ~400 words)
{Specific reconciliation: line-by-line comparison of counterfactual implementations. Whose framing is the publication-correct one?}

## Methodology concerns (non-blocking)
1. ...

## Things Codex got right
1. ...

## Recommendation for the comparison memo
{One paragraph: what should the editorial layer publish? Same branch (adaptation), but which H3 number?}
```

Target ~1000 words. Be specific. Reference file:line. The deliverable is `reviews/r2-claude-review-of-codex.md`. That's the ONLY file you write.

Do NOT modify any files in `claude-analysis-r2/` or `codex-analysis-r2/` — read-only.

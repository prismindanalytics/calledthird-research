# Cross-Review Prompt — Claude reviews Codex

You previously completed Round 1 of CalledThird's "7-Hole Tax" research as Agent A (Claude). Your work is in `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/claude-analysis/`. Agent B (Codex) ran an independent analysis with deliberately divergent ML-engineering methods (LightGBM + SHAP + counterfactual permutation + energy-distance selection probe). Their work is in `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/codex-analysis/`.

## Your task

Read Codex's full work — at minimum:
- `codex-analysis/REPORT.md`
- `codex-analysis/findings.json`
- `codex-analysis/READY_FOR_REVIEW.md`
- The model code (`challenge_model.py`, `called_pitch_model.py`, `counterfactual.py`, `selection_probe.py`)
- Sample chart PNGs in `codex-analysis/charts/`

Then write `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/claude-review-of-codex.md` (~800 words) as a **skeptical peer reviewer at FanGraphs**.

## What to flag (be specific — line numbers, exact claims)

- **Statistical errors** — bad CIs, miscalibrated counterfactuals, data leakage in CV.
- **Overclaims** — places where the data supports less than the prose says.
- **Cherry-picking** — features, hyperparameters, or thresholds that look post-hoc.
- **Methodology gaps** — checks that should have been run.
- **Reproducibility concerns** — undocumented seeds, missing preprocessing steps.

## The headline divergence to investigate (priority)

You found 7-hole batter overturn rate = **37.1%** (n=89 batter-issued challenges). Codex found **51.2%** (n=213). That's 14pp apart — they cannot both be right.

Investigate WHY. Specific things to check in `codex-analysis/`:

1. **Denominator definition.** Did Codex pool all challenges where the batter-at-plate was 7-hole (catcher + pitcher + batter initiations), while you restricted to batter-issued? Catchers overturn at ~60% per our prior `catchers-are-better-challengers` work, so pooling would mechanically inflate the 7-hole rate.
2. **Lineup-spot derivation.** Differences in pinch-hitter handling, double-switch handling, or how openers (pitcher in lineup) are assigned spots could move the n.
3. **Time window.** Did Codex pull a different date range or filter games differently?
4. **Their counterfactual scale.** Codex reports H2 effect of +0.15pp [+0.08, +0.23] — implausibly precise CIs for a 2,000-row challenge dataset. Is this a calibration artifact (poorly calibrated GBM compressing predicted probabilities)?

Your conclusion (B2/B3) and theirs (B4) need to be reconciled — but be honest. If Codex's pooled-denominator framing is a defensible interpretation of "7-hole tax," concede it. If your batter-issued framing is the more direct match for FanSided/Ringer's claim, defend it with reasoning.

## Format

```markdown
# Claude's Review of Codex's 7-Hole Tax Analysis

## Headline assessment
{One paragraph: do you trust Codex's B4 conclusion? Why or why not?}

## Critical issues (potential blockers)
1. {Issue with file:line reference and specific claim}
2. ...

## Methodology concerns (non-blocking)
1. ...

## Things Codex got right (be honest — this is peer review, not adversarial)
1. ...

## The denominator divergence (priority section)
{Specific reconciliation: who's right on H1, or are both right under different definitions?}

## Recommendation for the comparison memo
{One paragraph: what should the editorial layer do with these two reports?}
```

Target ~800 words. Be specific, reference exact claims, and pretend you're submitting to FanGraphs. Don't be polite where it would mislead.

When done, the file at `reviews/claude-review-of-codex.md` is the only deliverable.

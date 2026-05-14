# Cross-Review Prompt — Codex reviews Claude

You previously completed Round 1 of CalledThird's "7-Hole Tax" research as Agent B (Codex). Your work is in `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/codex-analysis/`. Agent A (Claude) ran an independent analysis with deliberately divergent Bayesian/interpretability methods (hierarchical Bayesian logistic GLM + Bayesian GAM + Wilson CIs). Their work is in `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/claude-analysis/`.

## Your task

Read Claude's full work — at minimum:
- `claude-analysis/REPORT.md`
- `claude-analysis/findings.json`
- `claude-analysis/READY_FOR_REVIEW.md`
- The model code (`bayes_glm_h2.py`, `bayes_gam_h3.py`, `wilson_h1.py`, `selection_probe.py`, `stratified_h3.py`)
- Sample chart PNGs in `claude-analysis/charts/`

Then write `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/codex-review-of-claude.md` (~800 words) as a **skeptical peer reviewer at FanGraphs**.

## What to flag (be specific — line numbers, exact claims)

- **Statistical errors** — convergence claims, prior misspecification, CrI misinterpretation.
- **Overclaims** — places where the posterior supports less than the prose says.
- **Cherry-picking** — pre-registered thresholds vs post-hoc hedges.
- **Methodology gaps** — what checks Claude skipped that you would have run.
- **Reproducibility concerns** — process interruptions, finalize_findings.py recovery, undocumented choices.

## The headline divergence to investigate (priority)

Claude found 7-hole batter overturn rate = **37.1%** (n=89 batter-issued challenges). You found **51.2%** (n=213). That's 14pp apart — both cannot be right.

Investigate WHY. Specific things to check in `claude-analysis/`:

1. **Denominator definition.** Claude restricts H1 to batter-issued challenges. You pooled all challenges where the batter-at-plate was 7-hole. Whose framing better matches the FanSided/Ringer claim ("7-hole batters win their challenges 30.2%")? That phrasing most naturally reads as batter-issued — but is "the umpire's bias against 7-hole batters" a property of the AB or the initiator?
2. **Sample size.** Claude has n=89 in batter-only; their Wilson CI [27.8, 47.5] is 20pp wide and contains both 30.2% and 45.2%. Does Claude's "B2" conclusion outrun the data?
3. **H3 sample.** Claude reports n=28,579 borderline pitches; you used n=2,767. Why the order-of-magnitude gap? Is Claude's borderline cutoff (±0.3 ft of rulebook edge) defined the same way you defined yours?
4. **Bayesian convergence.** Are R-hat=1.00-1.01 and ESS=717-754 sufficient given the H2 dataset has random effects on three categorical vars (pitcher/catcher/umpire) with potentially many low-count levels? Could Claude's CrI be artificially narrow due to partial-pool shrinkage?
5. **Stratified H3 reliability.** Claude flagged a child-process death during stratified H3 fits and recovered via `finalize_findings.py`. Is the reported "no stratum where the effect lives" claim auditable from the saved artifacts?

Your conclusion (B4) and theirs (B2/B3) need to be reconciled — but be honest. If Claude's batter-issued framing is the more defensible match for FanSided/Ringer's claim, concede it. If your pooled framing is the more general "umpire bias against this AB" measure, defend it.

## Format

```markdown
# Codex's Review of Claude's 7-Hole Tax Analysis

## Headline assessment
{One paragraph: do you trust Claude's B2/B3 conclusion? Why or why not?}

## Critical issues (potential blockers)
1. {Issue with file:line reference and specific claim}
2. ...

## Methodology concerns (non-blocking)
1. ...

## Things Claude got right (be honest — this is peer review, not adversarial)
1. ...

## The denominator divergence (priority section)
{Specific reconciliation: who's right on H1, or are both right under different definitions?}

## Recommendation for the comparison memo
{One paragraph: what should the editorial layer do with these two reports?}
```

Target ~800 words. Be specific, reference exact claims, and pretend you're submitting to FanGraphs. Don't be polite where it would mislead.

When done, the file at `reviews/codex-review-of-claude.md` is the only deliverable.

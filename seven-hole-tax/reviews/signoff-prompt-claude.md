# Article Sign-Off — Claude (Agent A)

You previously reviewed the draft article and gave APPROVE WITH EDITS, listing specific fixes in `reviews/article-review-claude.md`. The draft has been revised; this is your final sign-off.

## Read

1. The revised draft: `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/DRAFT.md`
2. Your prior review with the punch list: `reviews/article-review-claude.md`

## What changed (short summary, but verify against the actual draft)

The editor applied the following edits in response to your review:

1. **Numerical accuracy:**
   - Q-value language tightened to "all above 0.6 — most above 0.8" (line 60)

2. **Methodology accuracy:**
   - `fame_quartile` added to the H2 model formula (line 105)
   - Stratified H3 reframed as "supportive, not findings of record" with note on the process interruption recovery (line 105)
   - "Both pipelines fit on the full taken-pitch corpus and evaluated effects on the borderline subsample" added to clarify that 28,579 is the eval set, not the train set (line 103)
   - Codex's method renamed from "counterfactual permutation test" to "counterfactual paired prediction with bootstrapped deltas" (lines 34, 107)

3. **Mechanism overclaim:**
   - The "selection lives in counts and pitcher quality" paragraph was rewritten to remove the unsupported speculation about two-strike counts and late-inning relievers, and to explicitly say "we have not yet decomposed how much of the residual gap each piece contributes" (line 68)
   - Open-questions section #2 was tightened to match (line 95)

4. **"Isn't in the data":**
   - Replaced with "is not something we detect in this window" plus the explicit credible-interval upper bound of +1.2 pp (line 38)

5. **H2 controlled challenge model added:**
   - "What we believe" section now mentions: "A controlled hierarchical Bayesian challenge model placed the spot-7-vs-spot-3 effect at a directional negative magnitude, but its 95% credible interval crossed zero — directionally consistent with the bottom-of-order story, not decisive on its own." (line 84)

6. **Title:**
   - Changed from "Here's What Actually Explains the Number" to "Here's Why the Number Looks Bigger Than It Is" — softens the explanatory promise.

7. **Temporal heterogeneity (your optional ask):**
   - Added as Open Question #4: "The May-only subsample of borderline called pitches put the spot-7-vs-3 effect at roughly five times the full-window magnitude (still small in absolute terms, still cutting in the same direction). With six weeks of season, that could be noise; with twelve weeks it might be a trend worth investigating." (line 97)

## Deliverable

Write `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/article-signoff-claude.md` (~150-300 words) with this structure:

```markdown
# Article Sign-Off — Claude (Agent A)

## Verdict
{SIGN OFF | NEEDS MORE EDITS}

## Resolved
{Punch list items the revisions addressed.}

## Still outstanding (if any)
{Anything you flagged in the original review that wasn't addressed, OR anything new the revisions broke.}

## One-line bottom line
```

Be quick and binary. Don't re-litigate items you already approved. The goal is to confirm publication-ready or identify any remaining blocker.

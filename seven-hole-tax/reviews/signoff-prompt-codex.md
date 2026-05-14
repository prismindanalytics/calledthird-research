# Article Sign-Off — Codex (Agent B)

You previously reviewed the draft article and gave APPROVE WITH EDITS, listing specific fixes in `reviews/article-review-codex.md`. The draft has been revised; this is your final sign-off.

## Read

1. The revised draft: `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/DRAFT.md`
2. Your prior review with the punch list: `reviews/article-review-codex.md`

## What changed (short summary, but verify against the actual draft)

The editor applied the following edits in response to your review:

1. **Numerical fixes:**
   - "eight points above the FanSided figure" corrected to "6.9 points above the FanSided figure, but 8.1 points below the 45.2% league-wide batter-issued rate" (line 52)
   - "bottom third of the order wins fewer challenges than the top — about an 8-point gap" tightened to "the bottom of the order wins fewer challenges than the rest. The headline gap is roughly 8 points between the 7-hole batter-issued rate and the league-wide batter-issued rate" (line 84)
   - "catchers overturn at about 60%" softened to "the catcher framing edge we documented in prior work" with link (line 74)
   - Data-window scoping: challenges through May 4, called pitches through May 3 (lines 11, 103)

2. **Methodology fixes:**
   - `fame_quartile` added to Claude's H2 formula (line 105)
   - Your method renamed from "counterfactual permutation test" to "counterfactual paired prediction with bootstrapped deltas" / "counterfactual paired prediction" (lines 34, 107)
   - Stratified H3 described as "supportive, not findings of record" with note on the process-interruption recovery (line 105)

3. **Mechanism overclaim — your central concern:**
   - "What looks like an umpire-bias story is actually a story about which pitches the bottom of the order ends up challenging" softened to "more consistent with selection into the challenge pool than with a different called zone" (line 21)
   - "isn't in the data" changed to "is not something we detect in this window" with the credible-interval upper bound (line 38)
   - The entire mechanism paragraph rewritten — removed unsupported "two-strike counts" / "late-inning relievers" speculation, replaced with hedged language: "The remaining selection most likely lives in count and pitcher/catcher context… We have not yet decomposed how much of the residual gap each piece contributes; that is a Round 2 question. What we can say is that the data are consistent with selection into the challenge pool, not with a different called zone." (line 68)
   - "What we believe" #2 hedged: "consistent with count and pitcher/catcher context driving the deficit, not with umpire judgment" (line 84)
   - Open Question #2 hedged to match: "We've described the residual 8-point overturn deficit as consistent with count and pitcher/catcher context… A formal mediation analysis would quantify how much of the gap each piece contributes." (line 95)

4. **Underclaim:**
   - Claude's controlled hierarchical Bayesian challenge model now mentioned in "What we believe" with a directional-but-inconclusive framing (line 84)
   - All-challenger denominator framing slightly strengthened (line 76)

5. **Title:**
   - Changed from "Here's What Actually Explains the Number" to "Here's Why the Number Looks Bigger Than It Is" — softens the explanatory promise.

## Deliverable

Write `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/article-signoff-codex.md` (~150-300 words) with this structure:

```markdown
# Article Sign-Off — Codex (Agent B)

## Verdict
{SIGN OFF | NEEDS MORE EDITS}

## Resolved
{Punch list items the revisions addressed.}

## Still outstanding (if any)
{Anything you flagged in the original review that wasn't addressed, OR anything new the revisions broke.}

## One-line bottom line
```

Be quick and binary. Don't re-litigate items you already approved. The goal is to confirm publication-ready or identify any remaining blocker.

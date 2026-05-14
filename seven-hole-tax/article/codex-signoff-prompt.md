# Article Sign-Off Prompt — Codex

The CalledThird editorial layer has drafted an article integrating the Round 1 + Round 2 findings of the 7-Hole Tax research project. Your sign-off is required before publication.

## What to read

1. **The draft article:** `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/article/DRAFT.md`
2. **The comparison memos:**
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/COMPARISON_MEMO.md` (R1)
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/COMPARISON_MEMO_R2.md` (R2)
3. **The actual numbers**, to verify the article cites them correctly:
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/claude-analysis/findings.json`
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/codex-analysis/findings.json`
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/claude-analysis-r2/findings.json`
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/codex-analysis-r2/findings.json`
4. **Both R2 cross-reviews:**
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/r2-claude-review-of-codex.md`
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/r2-codex-review-of-claude.md`

## Your task

You are Agent B (Codex). You ran the ML-engineering pipeline. The article must accurately represent what your models produced — and accurately characterize where your methodology was caught in cross-review (specifically the bootstrap-CI artifact). Your sign-off says you would defend every ML-method claim in the article in front of a FanGraphs editor.

Read the draft carefully. Then write `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/article/codex-signoff.md` with one of three verdicts:

### Verdict options

**APPROVE** — The article is publication-ready as written.

**APPROVE WITH NITS** — Substantively correct but specific phrasings/numbers should change. List each change with the original quote and exact replacement text.

**REVISE** — There is a substantive error or an overclaim that must be fixed.

## What to check (priority list)

1. **Every numerical claim from your pipeline.** Verify each claim about your ML results (e.g., "−0.35pp [−0.39, −0.31]", "0 hitters credibly worse off", "−0.78pp", "energy-distance p=0.955", "5 reverse-direction umpires", etc.) against your `findings.json` files.

2. **Honest characterization of your CI artifact.** The article calls out the bootstrap-CI artifact in your H7 ("a per-pitch-residual standard error, not a model-uncertainty interval — it understates uncertainty by 2-3×") and references the same kind of issue from R1's H2. Verify this characterization is fair and accurate. If it's harsher than warranted, push back.

3. **The 5 flagged umpires.** The article doesn't name them but says they "fail Bayesian replication" because two have positive posterior medians under shrinkage. Verify the cross-method convergence-failure characterization is accurate — that the right read is "ML method flagged 5; cross-method convergence does not survive."

4. **The deep test (Test 3).** This is the central beat. Verify the table: Bayesian −0.17pp [−1.5, +1.2] on n=28,579; ML −0.35pp [−0.39, −0.31]. Verify the disclosure of why your interval is narrower (per-pitch-residual SE).

5. **The hitter analysis.** The article surfaces Cam Smith and Pete Crow-Armstrong (both visible in your residual table) and drops Henry Davis (didn't survive your bootstrap). Verify this matches your `findings.json`.

6. **The catcher mechanism (Test 6).** The article cites your energy-distance p=0.955. Verify.

7. **The chase-tertile interaction (H7 mention).** The article uses Claude's wider [-3.98, +1.03] CrI rather than your narrow [-0.88, -0.66]. Verify this is the right call given the cross-review finding about your bootstrap method.

8. **Sign-convention.** The R2 cross-review caught a sign-convention issue in Claude's narrative. Read the draft for any remaining sign issues — anywhere it implies "more strikes called for 7-hole batters" but shows a *negative* effect, that's a sign error. Verify direction throughout.

9. **Methodology section.** Verify the dual-agent process description matches what actually happened. The article references "ML with counterfactual paired prediction + SHAP + energy-distance selection probe" — confirm that's what your pipeline did.

10. **What we're not claiming section.** Verify this section honestly bounds what the data does and doesn't support. Don't let it underclaim what was actually tested.

## What you do NOT need to check

- Claude's specific Bayesian-method numbers (Claude is checking those)
- Astro/HTML formatting
- Chart components

## Format your sign-off file as:

```markdown
# Codex Sign-Off — 7-Hole Tax Article Draft

**Verdict:** APPROVE / APPROVE WITH NITS / REVISE

## Summary
{One paragraph}

## Numerical claims verified
{List each numerical claim and confirm the source. Flag any errors.}

## Required changes (for APPROVE WITH NITS or REVISE)
1. Original draft: "{exact quote}"
   Replace with: "{exact replacement text}"
   Reason: {why}
2. ...

## Editorial concerns (non-blocking)
{Things to flag but not blockers}

## Final recommendation
{One paragraph: ship/don't ship/ship after these specific changes}
```

Be a tough peer reviewer. If the article underclaims your methodology's strengths or mischaracterizes the bootstrap-CI issue, push back. If it overclaims either direction, push back.

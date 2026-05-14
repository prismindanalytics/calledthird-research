# Article Sign-Off Prompt — Claude

The CalledThird editorial layer has drafted an article integrating the Round 1 + Round 2 findings of the 7-Hole Tax research project. Your sign-off is required before publication.

## What to read

1. **The draft article:** `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/article/DRAFT.md`
2. **The comparison memos** (your synthesis input, for context):
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/COMPARISON_MEMO.md` (R1)
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/COMPARISON_MEMO_R2.md` (R2)
3. **The actual numbers**, to verify the article cites them correctly:
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/claude-analysis/findings.json` (R1 Bayesian)
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/codex-analysis/findings.json` (R1 ML)
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/claude-analysis-r2/findings.json` (R2 Bayesian)
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/codex-analysis-r2/findings.json` (R2 ML)
4. **Both R2 cross-reviews** (the most recent peer audit):
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/r2-claude-review-of-codex.md`
   - `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/r2-codex-review-of-claude.md`

## Your task

You are Agent A (Claude). You ran the Bayesian/interpretability pipeline. The article must accurately represent what your model produced. Your sign-off says you would defend every Bayesian-method claim in the article in front of a FanGraphs editor.

Read the draft carefully. Then write `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/article/claude-signoff.md` with one of three verdicts:

### Verdict options

**APPROVE** — The article is publication-ready as written. (Use this only if every claim is verified and every nuance is right.)

**APPROVE WITH NITS** — The article is substantively correct, but some specific phrasings/numbers should change before shipping. List each change with file:line of the original draft and the exact replacement text.

**REVISE** — There is a substantive error or an overclaim that must be fixed before sign-off. Specify exactly what's wrong and what would fix it.

## What to check (priority list)

1. **Every numerical claim against the source data.** For each number in the draft (e.g., "37.1% on n=89", "0 of 78 umpires", "−0.17pp [−1.5, +1.2]", etc.), find the corresponding entry in the relevant `findings.json` and verify it matches. Flag any rounding or transcription errors.

2. **Sign-convention discipline.** The R2 cross-review caught a sign-convention error in your own R2 narrative — negative effects (lower called-strike rate for spot 7) are the *favored* direction, not "pro-tax." Read the draft with that in mind. Anywhere the article describes "more strikes called" should map to *positive* spot-7 effects, and "more balls called / favored" maps to *negative*. Verify this is consistent throughout.

3. **CI / uncertainty framing.** The draft uses your Bayesian intervals (the wider, more honest ones) over Codex's bootstrap intervals throughout, per the comparison memo. Verify that wherever a number has a CI in the draft, the CI matches your Bayesian fit, not Codex's bootstrap.

4. **Random-slope SD framing.** The R2 cross-review caught your overclaim that the 0.076 logit SD meant "umpires barely differ." The draft says "the data is *consistent* with modest umpire-by-umpire variation, but no individual umpire stands out as a credibly biased actor." Verify this captures the right epistemic stance.

5. **Bottom-of-order disclosure.** The draft mentions H4 was on a bottom-of-order indicator (spots 7-9), not strict spot-7. Verify that's clear and not buried.

6. **Hitter naming discipline.** The draft surfaces Cam Smith and Pete Crow-Armstrong as residual outliers and explicitly drops Henry Davis. Verify the framing distinguishes "model-residual anomaly" from "evidence of umpire favoritism." Verify the chase-rate caveat on Crow-Armstrong is accurate (his 2025 chase rate).

7. **The deep test (Test 3).** This is the central beat of the article. Verify the table is correct: your Bayesian effect was −0.17pp [−1.5, +1.2] on n=28,579; Codex's was −0.35pp [−0.39, −0.31]. Verify the disclosure that Codex's narrower interval is a per-pitch-residual SE artifact.

8. **The methodology section.** Verify the "two independent agents, two rounds, two cross-reviews" framing matches what actually happened. Verify the open-source repository link makes sense (it'll be the actual GitHub repo path).

9. **Tone and voice consistency with prior CalledThird flagships** (Coaching Gap, Pitch Tunneling Atlas). The article should be rigorous-but-accessible, honest about uncertainty, occasionally self-deprecating, never condescending. If the voice is off anywhere, flag it.

10. **What we're not claiming section.** Verify this section accurately bounds what the data does and doesn't support. Don't let the article overclaim a debunk or underclaim what was tested.

## What you do NOT need to check

- Codex's specific numerical claims (Codex will check those in the parallel sign-off)
- Astro/HTML formatting (the draft is in markdown; the production conversion happens after sign-off)
- Chart components (those are added during the Astro wrap)
- The article's social-media plan (separate workflow)

## Format your sign-off file as:

```markdown
# Claude Sign-Off — 7-Hole Tax Article Draft

**Verdict:** APPROVE / APPROVE WITH NITS / REVISE

## Summary
{One paragraph: what's right, what needs to change}

## Numerical claims verified
{List each numerical claim and confirm the source. Flag any errors.}

## Required changes (for APPROVE WITH NITS or REVISE)
1. Original draft: "{exact quote}"
   Replace with: "{exact replacement text}"
   Reason: {why}
2. ...

## Editorial concerns (non-blocking)
{Things you'd want to flag but aren't blockers}

## Final recommendation
{One paragraph: ship/don't ship/ship after these specific changes}
```

Be a tough peer reviewer. The point of dual-agent + cross-review is that errors get caught before publication. If you find errors, find them now.

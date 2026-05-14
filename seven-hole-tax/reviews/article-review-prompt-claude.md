# Article Review — Claude (Agent A)

CalledThird has drafted the public-facing article from your dual-agent analysis. Before publication, we need both agents who did the underlying research to sign off that the article fairly represents the data.

## Read first

1. **The draft article:** `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/DRAFT.md`
2. **Your own analysis:** `claude-analysis/REPORT.md`, `claude-analysis/findings.json`, `claude-analysis/READY_FOR_REVIEW.md`
3. **Codex's analysis:** `codex-analysis/REPORT.md`, `codex-analysis/findings.json`, `codex-analysis/READY_FOR_REVIEW.md`
4. **The comparison memo:** `reviews/COMPARISON_MEMO.md`
5. **Your prior cross-review:** `reviews/claude-review-of-codex.md`
6. **Codex's cross-review of you:** `reviews/codex-review-of-claude.md`

## What you're checking

For each numerical claim and methodological characterization in the draft article, verify:

1. **Numerical accuracy.** Every number, n, percentage, CI, p-value cited in the draft must match what's actually in `findings.json` (yours or Codex's). Flag every mismatch.
2. **Methodology fidelity.** When the article describes either pipeline's method, does it accurately reflect what the code actually did? In particular: the methodology section near the bottom of the draft.
3. **Overclaim check.** Does the article state any conclusion that the data does not support at the implied confidence level? Especially places where the article asserts mechanism ("the selection lives in counts and pitcher quality") — is that demonstrated or just consistent with the data?
4. **Underclaim check.** Are there findings either of you produced that the article should mention but doesn't?
5. **Definitional splits.** The article has to handle the 37.1% vs 51.2% denominator split. Does it do so fairly to both framings, including yours?
6. **Tone / voice / brand fit.** CalledThird's voice: rigorous but accessible, honest about uncertainty, occasionally self-deprecating, never condescending. Does the draft hit it?

## Deliverable

Write `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/article-review-claude.md` (~600-800 words) with this structure:

```markdown
# Article Review — Claude (Agent A)

## Verdict
{APPROVE | APPROVE WITH EDITS | REQUEST REVISIONS}
{One paragraph on whether the draft fairly represents your analysis.}

## Numerical accuracy issues
{Every number that doesn't match findings.json — flag with line reference and the correct value.}

## Methodology accuracy issues
{Places where the article mischaracterizes what your pipeline did or what Codex's did.}

## Overclaims to fix
{Specific sentences that outrun the data, with proposed rewrites.}

## Underclaims / missing context
{Things the article should add.}

## Tone / voice notes
{Optional: places where the prose could be sharper or more in-voice.}

## Bottom line
{One paragraph: ship as-is, ship with edits, or revise. If revise, what's blocking?}
```

Be a tough editor. If the article overclaims relative to your B2-with-B3-hedge conclusion, say so. If it underclaims relative to the H3 null, say so. If it gets either pipeline's method wrong, say so. The goal is a publishable article that neither agent will privately wish had been hedged differently.

When done, that file is the only deliverable. Do NOT modify `DRAFT.md` directly.

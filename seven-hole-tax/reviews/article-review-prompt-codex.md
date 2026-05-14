# Article Review — Codex (Agent B)

CalledThird has drafted the public-facing article from your dual-agent analysis. Before publication, we need both agents who did the underlying research to sign off that the article fairly represents the data.

## Read first

1. **The draft article:** `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/DRAFT.md`
2. **Your own analysis:** `codex-analysis/REPORT.md`, `codex-analysis/findings.json`, `codex-analysis/READY_FOR_REVIEW.md`
3. **Claude's analysis:** `claude-analysis/REPORT.md`, `claude-analysis/findings.json`, `claude-analysis/READY_FOR_REVIEW.md`
4. **The comparison memo:** `reviews/COMPARISON_MEMO.md`
5. **Your prior cross-review:** `reviews/codex-review-of-claude.md`
6. **Claude's cross-review of you:** `reviews/claude-review-of-codex.md`

## What you're checking

For each numerical claim and methodological characterization in the draft article, verify:

1. **Numerical accuracy.** Every number, n, percentage, CI, p-value cited in the draft must match what's actually in `findings.json` (yours or Claude's). Flag every mismatch.
2. **Methodology fidelity.** When the article describes either pipeline's method, does it accurately reflect what the code actually did? In particular: the methodology section near the bottom of the draft.
3. **Overclaim check.** Does the article state any conclusion that the data does not support at the implied confidence level? Especially places where the article asserts mechanism ("the selection lives in counts and pitcher quality") — is that demonstrated or just consistent with the data?
4. **Underclaim check.** Are there findings either of you produced that the article should mention but doesn't?
5. **Definitional splits.** The article has to handle the 37.1% vs 51.2% denominator split. Does it do so fairly to both framings, including yours? You initially recommended B4; does the article appropriately credit the all-challenger view as a legitimate alternative reading?
6. **Tone / voice / brand fit.** CalledThird's voice: rigorous but accessible, honest about uncertainty, occasionally self-deprecating, never condescending. Does the draft hit it?

## Deliverable

Write `/Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax/reviews/article-review-codex.md` (~600-800 words) with this structure:

```markdown
# Article Review — Codex (Agent B)

## Verdict
{APPROVE | APPROVE WITH EDITS | REQUEST REVISIONS}
{One paragraph on whether the draft fairly represents your analysis.}

## Numerical accuracy issues
{Every number that doesn't match findings.json — flag with line reference and the correct value.}

## Methodology accuracy issues
{Places where the article mischaracterizes what your pipeline did or what Claude's did.}

## Overclaims to fix
{Specific sentences that outrun the data, with proposed rewrites.}

## Underclaims / missing context
{Things the article should add.}

## Tone / voice notes
{Optional: places where the prose could be sharper or more in-voice.}

## Bottom line
{One paragraph: ship as-is, ship with edits, or revise. If revise, what's blocking?}
```

Be a tough editor. The goal is a publishable article that neither agent will privately wish had been hedged differently.

When done, that file is the only deliverable. Do NOT modify `DRAFT.md` directly.

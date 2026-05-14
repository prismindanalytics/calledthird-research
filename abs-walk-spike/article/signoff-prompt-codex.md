# Article Sign-Off Prompt — Codex

The CalledThird ABS Walk Spike R3 article draft is at:
`/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/article/DRAFT.md`

You contributed Round 1, 2, and 3 of the underlying analysis as Agent B. Your work is in:
- `codex-analysis/` (R1)
- `codex-analysis-r2/` (R2)
- `codex-analysis-r3/` (R3)

The COMPARISON_MEMOs at `reviews/COMPARISON_MEMO.md`, `reviews/COMPARISON_MEMO_R2.md`, `reviews/COMPARISON_MEMO_R3.md` are the editorial spine.

## Your task

Read the draft. Verify every numerical claim against your pipeline's `findings.json` files. Verify framing matches the comparison memos. Flag any required changes.

Write `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/article/codex-signoff.md` (~800 words) with verdict: APPROVE, APPROVE WITH NITS, or REVISE.

## Required-change candidate areas

- **H1 magnitude:** Draft says "~+26%" headline with editorial CI [+0.2%, +70.1%]. Your R3 H1 point estimates: A=+28.8%, B=+27.0%, C=+24.3%. Verify.
- **H1 framing:** Draft drops Claude's Method A (−58.6%) from the headline triangulation, uses 5-method positive envelope. Confirm.
- **H3 archetype effect:** Spearman ρ −0.258 (your value), p<0.0001, SHAP interaction details. Check.
- **Named pitchers:** Finnegan +12.0pp (your value), O'Brien −6.9pp, Doval −6.4pp. Verify against your `findings.json`. Note the draft shows BOTH agents' numbers ("Bayesian says X; ML pipeline says Y") — verify the Codex numbers in those parentheticals.
- **R3 H2:** Draft says 9 ML-stable pitchers (your number) are reported as candidate YoY shifters in a footnote, not as named adapters. Verify.
- **R2 artifact disclosure:** Draft references your R1's −56% Statcast schema artifact and your R2's narrow CI artifact. Verify framing is accurate.
- **Calibration audits:** Your R3 H2 GBMs had 343/1000 OOB calibration bins >5pp. Article doesn't dive into this; verify the omission is editorially defensible.
- **FanGraphs citations:** Draft links to FanGraphs "Strike Zone Is Shrinking" and "Where Are 2026's Extra Walks Coming From?" Verify accuracy.

## Sign-convention check (critical — recurring issue)

The article discusses "command pitchers hurt" and "stuff pitchers helped." Verify the sign convention:
- HIGHER walk rate = BAD for pitcher = "hurt"
- LOWER walk rate = GOOD for pitcher = "helped"
- Finnegan (+11.4pp walks) = hurt
- O'Brien/Doval (fewer walks) = helped
- Stuff-archetype = LOW command + HIGH stuff
- Command-archetype = HIGH command + LOW stuff

Check every use of "hurt" / "helped" / "favored" / "taxed" / "smaller zone" / "larger zone" in the draft for consistency.

## Brand voice check

- The "What the dual-agent process caught this time" section should accurately reflect what your pipeline's three rounds revealed
- The methodology callout: 6 total H3 counterfactuals, 100×10 bootstrap-of-bootstrap, etc. Verify all method counts.

## Format

```markdown
# Codex Sign-Off — ABS Walk Spike R3 Article Draft

**Verdict:** APPROVE / APPROVE WITH NITS / REVISE

## Summary
{One paragraph}

## Numerical claims verified
{Table or list with file references}

## Required changes
{If any. Provide exact original quote and exact replacement text.}

## Non-blocking editorial concerns
{If any}

## Final recommendation
{One paragraph}
```

Target ~800 words. Deliverable: `article/codex-signoff.md`. Do NOT modify the draft itself.

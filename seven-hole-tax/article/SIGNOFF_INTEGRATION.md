# 7-Hole Tax Article — Sign-Off Integration Log

**Date:** 2026-05-08
**Status:** Both agents signed off; all required changes integrated. Article cleared for publication.

## Sign-off verdicts

- **Claude (Bayesian/interpretability):** APPROVE WITH NITS (3 required changes)
- **Codex (ML-engineering):** REVISE (8 required changes)

Both agents verified their pipeline's numerical claims against `findings.json` files. Claude verified all 27 numerical claims explicitly; Codex independently verified all ML-side numbers and the cross-method convergence-failure narrative.

## Changes integrated (11 total — overlapping fixes deduplicated)

| # | Section | Issue | Source | Status |
|---|---|---|---|---|
| 1 | Test 2 (line 57) | 1.27 inches and KS p=0.19 attributed to "batter-issued challenges" — actually all-challenger pooled | Claude | ✓ |
| 2 | Test 3 / CI artifact callout | "Understates uncertainty by 2-3×" sourced to wrong analysis (R2 H7, not R1 H3) | Codex #1 | ✓ |
| 3 | Test 4 disclosure (lines 86-88) | Bottom-of-order indicator vs strict spot-7 not disclosed up front | Claude #3 | ✓ |
| 4 | Test 4 (line 92) | Sign-convention error — "favoring with a larger zone" should be "smaller called-strike zone" + drop umpire names | Both Claude #1 + Codex #2 | ✓ |
| 5 | Test 5 (line 116) | Henry Davis included as named hitter; should drop per R2 memo | Codex #3 | ✓ |
| 6 | H7 paragraph (line 130) | "Directional in the FanSided direction" is wrong — negative effect is favored, not FanSided | Codex #4 | ✓ |
| 7 | "Lone shape" sentence | Same sign issue as #6 | Codex #5 | ✓ |
| 8 | Narrow-CI artifact paragraph | "Few hundred-row stratum" — actual n=804 (spot 7) and n=1,241 (spot 3) | Codex #6 | ✓ |
| 9 | "What we're not claiming" | "Smaller zone" → "larger called-strike zone" for accurate FanSided framing | Codex #7 | ✓ |
| 10 | Summary table | "3 reverse-direction outliers" → "2 cross-method outliers" | Codex #8 | ✓ |
| 11 | Methodology section | Cross-review word count "~800-1,000" → "800-1,100" (R2 reviews are longer) | Claude editorial concern (B) | ✓ |

## Non-blocking editorial concerns (NOT integrated, by design)

- **Codex H3 denominator footnote** (Codex editorial concern): The Test 3 table cites both methods but doesn't explicitly disclose Codex's tighter borderline cutoff. The narrow-CI-artifact paragraph already covers why Codex's interval is tighter. Skip.
- **Process-death recovery thread** (Claude editorial concern D): R1 stratified H3 had a process-death recovery; not load-bearing for any headline number. Skip.
- **Test 3 sample-size footnote** (Claude editorial concern A): Same coverage as above. Skip.

## Cross-method consistency verified

After integration, the article's sign-convention is consistent throughout:
- **Tax direction** = positive p7−p3 effect = MORE called strikes for 7-hole = LARGER called-strike zone the umpire is calling
- **Favored/reverse direction** = negative p7−p3 effect = FEWER called strikes for 7-hole = SMALLER called-strike zone

All references in the draft now use "called-strike zone" rather than ambiguous "zone" wording, eliminating the colloquial-vs-analytical ambiguity that triggered the original sign-convention errors.

## Article disposition

Cleared for production. Next step:
1. Convert `DRAFT.md` to Astro page at `/website/src/pages/analysis/seven-hole-tax.astro`
2. Build chart components (Test 3 table, per-umpire distribution, hitter residuals, summary table)
3. Generate OG card
4. Register in `articles.json`
5. Schedule social-media thread

---

*Both R1 and R2 cross-review documents and both sign-off documents are preserved as audit trail in `reviews/` and `article/`.*

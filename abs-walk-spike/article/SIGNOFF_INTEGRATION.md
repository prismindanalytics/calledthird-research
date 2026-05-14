# ABS Walk Spike R3 Article — Sign-Off Integration Log

**Date:** 2026-05-14
**Status:** Both agents signed off; all required changes integrated. Article cleared for publication.

## Sign-off verdicts

- **Claude (Bayesian/interpretability):** APPROVE WITH NITS (2 required fixes)
- **Codex (ML-engineering):** REVISE narrowly (3 required fixes, 1 overlapping with Claude)

Both reviewers independently verified every numerical claim against their respective pipelines' `findings.json` files. Sign-convention discipline (which we'd been flagging since the 7-hole tax piece) was clean throughout — both reviewers explicitly confirmed no inversions.

## Changes integrated (5 total)

| # | Section | Issue | Source | Status |
|---|---|---|---|---|
| 1 | Mechanism section | "29% / 20%" called-strike illustration didn't match R2 `h5_post_table` data | Claude | ✓ (replaced with 36%/26%, +10pp drop vs 4pp at 2-strike) |
| 2 | Named pitchers — O'Brien | "Stuff near top quartile, command middling" is wrong on BOTH pipelines (his command is actually weak, not middling) | Both agents | ✓ (replaced with "command in the bottom quintile" — the right story for the right reason) |
| 3 | Archetype section | FanGraphs zone-rate 50.7%→47.2% claim attributed directly to linked piece, but Codex couldn't verify in that specific article | Codex | ✓ (attribution softened to "public reporting") |
| 4 | Methodology | "FanGraphs Stuff+ leaderboard returned a 403 to our scraper" was overly specific given artifacts | Codex | ✓ (replaced with "Stuff+ was unavailable in our run") |
| 5 | Setup | "since 1950" cleaner than "1948-50" range | Claude (non-blocking) | ✓ (integrated for editorial polish) |

## Non-blocking concerns dismissed

- **Sample sizes (75,681 taken / 28,579 borderline):** Claude flagged these as not appearing in her R3 findings; Codex verified and didn't flag, confirming they're from R2 substrate carried forward correctly. Left as-is.
- **H2 footnote vs body placement:** Codex notes the H2 "nine pitchers" sentence is in body rather than literal footnote. Framed as YoY shifters not adapters, which both agents confirmed is correct. Editorial preference, not a methodology fix. Left as-is.
- **SHAP-interaction language:** Codex notes the term is weaker than permuted-label baseline; draft already uses minimal SHAP-interaction language. No change needed.
- **Finnegan "top quartile" command:** Both agents' proxies put him just below 75th (Claude 0.717) and just inside 75th (Codex 75.8). Draft framing as "top quartile" is defensible on Codex's side and matches the memo. Left as-is.

## Cross-method consistency verified

Both reviewers independently confirmed:
- H1 magnitude ~+26% with editorial CI [+0.2%, +70.1%] correctly framed
- Method A (−58.6%) correctly preserved as stress test, not headline
- H3 archetype effect: ρ = −0.258 to −0.282 (both p<0.0001) — exact match
- Named pitchers: Finnegan +11.4pp, O'Brien −8.3pp, Doval −7.5pp (Claude residuals); Finnegan +12.0pp, O'Brien −6.9pp, Doval −6.4pp (Codex residuals) — exact match to both pipelines
- Mason Miller exclusion (Codex stability 0.68 < 0.80) verified
- Within-2026 trajectory W1 9.61% → W7 8.79% verified exactly
- P(regressed)=89% verified
- 0-0 mystery DiD −6.76pp credible verified
- 46,755 PAs verified
- Sign convention: "command pitchers hurt = more walks" and "stuff pitchers helped = fewer walks" used consistently throughout

## Article disposition

Cleared for production. Next steps:
1. Convert `DRAFT.md` to Astro page at `/website/src/pages/analysis/abs-walk-spike-mid-may-update.astro`
2. Build chart components:
   - H1 triangulation (6 methods or 5-method positive envelope)
   - H3 archetype scatter (the named-pitcher centerpiece)
   - H3 archetype leaderboards (most-hurt-command + most-helped-stuff)
   - Mechanism chart (carry/adapt from R2's `h5_first_pitch_mechanism`)
   - Weekly walk rate trajectory (carry/adapt from R2)
   - Summary table
3. Generate OG card
4. Register in `articles.json`
5. Schedule social-media thread
6. Add Round 4 / All-Star break commitment to `research_queue.json`

---

*Three rounds of dual-agent. Three cross-reviews per agent. Two sign-offs on this draft. All required changes integrated. Sign convention discipline intact.*

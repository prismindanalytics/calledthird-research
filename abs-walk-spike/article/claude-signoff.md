# Claude Sign-Off — ABS Walk Spike R3 Article Draft

**Verdict:** APPROVE WITH NITS

## Summary

The draft is faithful to the dual-pipeline analysis and to COMPARISON_MEMO_R3. Every headline number maps cleanly to a `findings.json` value in `claude-analysis/`, `claude-analysis-r2/`, or `claude-analysis-r3/`, or to the cross-agent triangulation in the memo. H1 magnitude (~+26%) with editorial CI [+0.2%, +70.1%] is framed correctly, Method A's −58.6% is preserved as a stress test rather than a headline, the H3 archetype effect (ρ −0.282 mine, −0.258 Codex; Bayes slope −1.40pp; ~2.8pp full-spectrum spread) is exact, and the three named pitchers (Finnegan +11.4pp, O'Brien −8.3pp, Doval −7.5pp) match my residuals to the decimal. Sign convention holds throughout — "command pitchers hurt = more walks," "stuff pitchers helped = fewer walks" — with no inversions. Two issues require fixes: a factually wrong description of O'Brien's archetype percentiles, and a 29%/20% called-strike illustration that doesn't match our data.

## Numerical claims verified

| Claim | Source | OK |
|---|---|---|
| 46,755 PAs | R3 `panel_2026_pa_count` | yes |
| Walk rate 9.46% / 2025 same window 8.78% | R3 `rate_26_emp` = 0.0946; R2 = 0.0878 | yes |
| YoY +0.68pp; R1 was +0.82pp | R2 `yoy_delta_pp_empirical` = 0.6842; R1 = 0.8196 | yes |
| W1-W7 = 9.61/10.27/9.78/9.23/9.16/9.19/8.79% | `codex-analysis-r2/data/weekly_aggregation.csv` (exact) | yes |
| W1-3 vs W5-7 = −0.86pp; P(regressed) = 89% | R2 `prob_2026_regressed` = 0.89025 | yes |
| ~+26%, CI [+0.2%, +70.1%]; positive-envelope median +27.0%; six-method median +25.6% | R3 `triangulated`; memo §3a | yes |
| Method A stress test −58.6%; R2 artifact −64.6% | R3 method_a = −58.625; R2 `all_pitches_mean` = −64.585 | yes |
| 0-0 DiD −6.76pp credible | R2 `diff_in_diff_top_pp_mean` = −6.758 | yes |
| Traffic +0.41 / conditional +0.27pp | R2 `h2_decomp` | yes |
| ρ = −0.282 (Bayes); −0.258 (ML); slope −1.40pp/unit | R3 spearman + bayes blocks | yes |
| Finnegan +11.4/+12.0; O'Brien −8.3/−6.9; Doval −7.5/−6.4 | R3 leaderboards + Codex R3 | yes |
| Miller ML stability 0.68 < 0.80 | memo §3c | yes |
| 0 within-2026 adapters; Chandler 58.5%; 367 eligible | R3 `h2_adapter_leaderboard` | yes |
| FanGraphs zone-rate 50.7% → 47.2%; ~9.5% walk through May 8 | ROUND2_BRIEF.md citations | yes (external) |

## Required changes

### Change 1 — O'Brien archetype description is wrong on both pipelines

Draft says O'Brien's "stuff score sat near the top quartile, his command was middling." Both pipelines disagree:

- Claude proxy: `stuff_pct` = 0.447 (45th — middling, not near top quartile); `command_pct` = 0.144 (14th — very weak, not middling).
- Codex proxy: `stuff_pct` = 62.6 (above-middling but not "near top quartile"); `command_pct` = 18.1 (clearly weak, not middling).

In both pipelines, O'Brien's stuff-minus-command is positive because his command is poor, not because his stuff is elite. This is the same story the archetype model tells — he's helped by the zone — but for the right reason.

**Original:**
> O'Brien is a power reliever; his 2025 stuff score sat near the top quartile, his command was middling. The new zone hurts pitchers who need first-pitch strikes called at the top. O'Brien's game doesn't need them. He's gone from a walker into a strike-thrower as the zone has shifted.

**Replacement:**
> O'Brien is a power reliever whose 2025 profile leaned stuff-over-command — above-average whiff in our arsenal-weighted proxy and a walk rate and zone rate that put his command in the bottom quintile. The new zone hurts pitchers who need first-pitch strikes called at the top. O'Brien's game wasn't built on those. He's gone from a walker into a strike-thrower as the zone has shifted.

### Change 2 — "29% / 20%" called-strike illustration doesn't match our data

My R2 `h5_post_table` says the top-edge first-pitch called-strike rate was **36.04% in 2025, 25.55% in 2026** (`yoy_delta_pp_mean` = −10.48pp). The 29%/20% pair is not in our findings.

**Original:**
> A first-pitch on the top edge that was a strike in 2025 (29% called-strike rate) is now much more often a ball (20%). Most of that gap is concentrated on the first pitch.

**Replacement:**
> A first-pitch on the top edge that drew a called strike 36% of the time in 2025 now draws one only 26%. That 10pp drop on first pitches compares to a 4pp drop on 2-strike pitches — the asymmetry is concentrated on the first pitch.

## Non-blocking editorial concerns

- **Finnegan "top quartile" command:** my proxy = 0.717 (just under 75th); Codex = 75.8 (just inside). Defensible on Codex and matches the memo framing — leave as is.
- **"75,681 taken pitches" and "28,579 borderline pitches":** not directly in my R3 findings (my 2026 panel is 182,856 rows; R2 H3 `n_takes_replayed` = 92,322). These look like Codex's R3 substrate filtering. Author should confirm against `codex-analysis-r3/data/` before publication. Non-blocking — doesn't change any headline.
- **"1948-50" historical reference:** the brief and AP cited "highest since 1950." "Since 1950" is cleaner than the band.
- **Sign-convention slip callback** in the dual-agent-process section is correct and well-stated. Keep it.

## Final recommendation

Approve with the two required fixes. The piece's spine — magnitude muted to ~+26%, top-edge first-pitch mechanism, archetype effect at ρ ≈ −0.27, three named pitchers — is exactly what R3 produced and what the comparison memo locked. Sign convention is clean throughout: the class of error caught in 7-hole tax did not survive into the draft. The dual-agent-process callout, prior-article links, and FanGraphs references are present and accurate. After the O'Brien archetype paragraph is corrected and the 29/20 illustration is replaced with 36/26, the piece is publication-ready.

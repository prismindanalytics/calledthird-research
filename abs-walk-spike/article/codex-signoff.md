# Codex Sign-Off — ABS Walk Spike R3 Article Draft

**Verdict:** REVISE

## Summary

The article's analytical spine is correct: the R3 headline should be around +26% zone attribution with a wide positive editorial interval, Claude's negative Bernoulli replay is correctly treated as a stress test, H2 is correctly framed as no cross-method stable named adapters, and H3 is correctly made the named-pitcher centerpiece. I verified the major numerical claims against the Codex R1/R2/R3 `findings.json` files, the R3 artifact summaries, and the comparison memos. The sign convention is consistent throughout: higher walk rate is bad for pitchers, lower walk rate is good, Finnegan is hurt, O'Brien/Doval are helped, and lower called-strike rate at the top favors hitters. I would not approve as-is only because three source/precision fixes are needed before publication.

## Numerical claims verified

| Claim | Verification |
|---|---|
| H1 headline ~+26%, editorial CI [+0.2%, +70.1%] | Matches `reviews/COMPARISON_MEMO_R3.md`: all-six median +25.6%, positive-method median +27.0%, positive-method envelope [+0.2%, +70.1%]. Codex R3 methods are A +28.8% [5.7, 56.9], B +27.0% [5.3, 53.3], C +24.3% [4.6, 49.4] in `codex-analysis-r3/artifacts/h1_summary.json`. |
| Claude Method A dropped from headline | Correct. R3 memo labels -58.6% as stress-test-only; draft's six-method disclosure is faithful. |
| Walk-rate baseline | Codex R3 uses 9.46% vs 8.80%, +0.659pp; Claude/R2 memo uses 9.46% vs 8.78%, +0.68pp. Draft's +0.68 is acceptable cross-agent editorial rounding, but Codex's value is +0.66pp. |
| R1/R2 artifact disclosure | Correct. Codex R1 `findings.json` has -56.17% normalized-coordinate attribution; R3 `h1_summary.json` flags the R2 narrow CI as a seed-only artifact. |
| H3 archetype effect | Correct. Codex `h3_summary.json`: Spearman rho -0.2583, p=0.0000266, 258 qualified pitchers, 200 leaderboard bootstraps. Claude artifact: rho -0.2820, p=0.0000040, slope -1.402pp/unit, full-spectrum spread ~2.8pp. |
| Named pitchers | Correct parentheticals. Codex `h3_summary.json`: Finnegan +11.95pp, O'Brien -6.90pp, Doval -6.43pp; Miller stability 0.68 below 0.80. Claude artifact: Finnegan +11.4pp, O'Brien -8.3pp, Doval -7.5pp. |
| R3 H2 | Correct estimand framing. Codex `h2_summary.json`: 9 ML-stable 2025-vs-2026 shifters among 234 qualified pitchers; Claude `h2_adapter_leaderboard.json`: 0 named within-2026 adapters among 367 eligible pitchers, Bubba Chandler max stability 0.585. |
| Calibration audits | Omission is editorially defensible because the article does not use H2 SHAP feature-group shares as a named claim. Codex `h2_summary.json` does show 343/1000 H2 feature-importance GBMs with >5pp OOB bin deviation, which weakens only the SHAP attribution shares, not the descriptive 9-name YoY-shifter list. |
| FanGraphs citations | The "Strike Zone Is Shrinking" link supports 14 square inches and 54.3% to 40.8% on fastballs just above the top cutoff. The "Extra Walks" link supports 9.5% walk rate through May 8 and the broad decomposition, but I did not find support there for the exact 50.7% to 47.2% zone-rate claim. |

## Required changes

1. Original: "This is consistent with FanGraphs' [analysis of zone-rate shifts](https://blogs.fangraphs.com/where-are-2026s-extra-walks-coming-from/) (which they report dropped from 50.7% to 47.2% league-wide) and with industry chatter about command pitchers struggling."

   Replacement: "This is consistent with FanGraphs' broader analysis of the walk spike and with public reporting that league zone rate has dropped from 50.7% to 47.2%, as well as with industry chatter about command pitchers struggling."

2. Original: "FanGraphs' Stuff+ leaderboard returned a 403 to our scraper, so we built a defensible proxy"

   Replacement: "FanGraphs Stuff+ was unavailable in our run, so we built a defensible Statcast proxy"

3. Original: "O'Brien is a power reliever; his 2025 stuff score sat near the top quartile, his command was middling."

   Replacement: "O'Brien is a power reliever; in our proxy his stuff score sat well above his command score, and his command score was low."

## Non-blocking editorial concerns

The H2 "nine pitchers" sentence is in the body rather than a literal footnote/table, while the memo recommends footnote treatment. It is still framed as YoY shifters rather than named adapters, so I do not consider that blocking.

The draft says the ML pipeline's SHAP interaction values support the H3 direction. That is directionally fair, but Codex's own R3 review says the stronger evidence is the Spearman rank effect and the stable leaderboards; the explicit nonlinear interaction term is weaker than the permuted-label baseline. Avoid expanding the SHAP-interaction language beyond the current sentence.

## Final recommendation

Revise the three lines above, then approve. The article has the right hierarchy: H1 magnitude transparent and not overprecise, R2/R3 artifacts disclosed, H2 not overclaimed, and H3 framed with the correct hurt/helped sign convention and the correct three-name intersection. The required fixes are source precision and archetype-wording fixes, not structural objections.

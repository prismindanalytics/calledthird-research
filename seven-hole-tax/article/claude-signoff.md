# Claude Sign-Off — 7-Hole Tax Article Draft

**Verdict:** APPROVE WITH NITS

## Summary

The article is substantively correct and faithful to the Bayesian/interpretability pipeline. Every load-bearing numerical claim — the −0.17pp [−1.5, +1.2] on n=28,579, the 0/78 umpires flagged, the random-slope SD of 0.076 logit, the −1.45pp [−3.98, +1.03] low-chase signal, the −0.02 in [−0.28, +0.26] catcher-mechanism null, the 119 qualifying hitters, the cross-review table for the 5 ML-flagged umpires — checks out against the Bayesian findings.json files. The CI framing is right (Bayesian intervals throughout, with the bootstrap artifact called out by name). The dual-agent + cross-review framing is honest. Smith/Crow-Armstrong are correctly named, Davis is correctly dropped, and the "model-residual anomaly, not bias" framing is the right read.

But there are two specific sign-convention/scope errors that the cross-review specifically warned us about and that the editorial layer carried into the prose. One is a sign-convention slip in Test 4 ("larger zone" vs "smaller zone"), exactly the genre of error R2 flagged in my own narrative. One is a scope error in Test 2 (batter-issued vs all-challenger). Both must be fixed before publication. Once those are fixed, the article is ready to ship.

## Numerical claims verified

| Claim in draft | Source | Verified? |
|---|---|---|
| 2,101 ABS challenges | R1 `data.n_challenges_total` | ✓ |
| 75,681 taken pitches | R1 `data.n_taken_pitches` | ✓ |
| 28,579 borderline pitches | R1 `data.n_borderline_taken_03ft` | ✓ |
| 37.1% on n=89 (spot 7 batter-issued) | R1 `h1.batter_summary[7].rate / .n` | ✓ |
| Wilson CI [27.8, 47.5] | R1 `h1.batter_summary[7].wilson_low/high` | ✓ |
| League-wide batter-issued = 45.2% | R1 `h1.h1_verdict.league_batter_rate` | ✓ |
| 8.1pp deficit | R1 `h1.h1_verdict.deficit_vs_league_batter_pp` | ✓ |
| 51.2% on n=213 (all-challenger) | R1 `h1_overturn_rate_by_spot[7]` | ✓ |
| Spots 7-9 hover ~37% (37.1, 37.6, 36.8) | R1 `h1_batter_only_overturn_rate_by_spot` | ✓ |
| KS p = 0.19 | R1 `selection_probe.ks_challenges_vs_spot3.spot_7_vs_spot_3.pvalue` (0.194) | ✓ — but see Required Change #2 re scope |
| Mean edge distance 1.27" | R1 `selection_probe.challenge_summary` (1.270 / 1.266) | ✓ — but see Required Change #2 re scope |
| Bayesian H3: −0.17pp [−1.5, +1.2] on n=28,579 | R1 `h3_main.spot7_vs_spot3` (−0.172, −1.505, +1.175) | ✓ |
| Codex H3: −0.35pp [−0.39, −0.31] | R1 codex `h3_called_strike_counterfactual_pp` (−0.349, −0.395, −0.311) | ✓ |
| 78 umpires qualified | R2 `h4_per_umpire.n_qualifying_umpires` | ✓ |
| 0 of 78 flagged | R2 `h4_per_umpire.n_flagged` | ✓ |
| Random-slope SD = 0.076 logit | R2 `h4_per_umpire.sd_umpire_botslope_med_logit` (0.0760) | ✓ |
| ML flags O'Nora −4.47pp, Wendelstedt −2.21pp | R2 codex `h4_per_umpire.flagged_list` | ✓ |
| O'Nora and Wendelstedt have positive posterior medians | R2 cross-review table (+0.12, +0.15) | ✓ |
| 119 hitters qualified | R2 `h5_per_hitter.n_qualifying_hitters` | ✓ |
| Smith and Crow-Armstrong appear in ML residuals; Davis does not | R2 codex `h5_per_hitter.flagged_list` (only Smith and PCA) | ✓ |
| Crow-Armstrong 2025 chase rate = 0.42 | R2 `h5_per_hitter.flagged_list` (0.4168) | ✓ |
| H6 catcher: −0.02 in [−0.28, +0.26] | R2 `h6_catcher_mechanism` (−0.021, −0.277, +0.257) | ✓ |
| Codex H6 energy-distance p = 0.955 | R2 codex `h6_catcher_mechanism.p_value` | ✓ |
| H7 low-chase: −1.45pp [−3.98, +1.03] | R2 `h7_chase_interaction.results.low_chase_spot7_vs_spot3_pp` | ✓ |
| H7 P(neg) = 0.87 | R2 same record `p_neg = 0.8725` | ✓ |
| Codex H7 −0.78pp [−0.88, −0.66] | R2 codex `h7_chase_interaction` (−0.778, −0.884, −0.659) | ✓ |
| R-hat ≤ 1.01, ESS ≥ 400 | R2 `h4.rhat_max_global` 1.01, `ess_min_global` 765 (and similar across H5/H6/H7) | ✓ |

All 27 numerical claims confirmed.

## Required changes (for APPROVE WITH NITS)

### 1. Sign-convention slip in Test 4 (line 92)

**Original draft:**

> "The ML method initially flagged five umpires (all in the *reverse* direction — favoring 7-hole batters with a larger zone, not penalizing them with a smaller one)."

**Replace with:**

> "The ML method initially flagged five umpires (all in the *reverse* direction — favoring 7-hole batters with a smaller zone, calling more borderline pitches as balls, rather than penalizing them with a larger one)."

**Reason:** This is exactly the sign-convention error R2 cross-review caught in my own narrative, only applied to "zone" rather than "called-strike rate." A negative effect = lower called-strike rate for spot 7 = the umpire is calling MORE pitches as balls = a SMALLER de facto zone for the batter = the favored direction. The current draft has it reversed: it says "favoring with a larger zone," which would actually be the *taxed* direction. This is the same class of error as the one we self-disclose in the "What the dual-agent process caught" section, so getting it right in Test 4 is non-negotiable. (The self-disclosure on line 142 is correctly reasoned — it's just that we then carry the wrong version of the same logic into Test 4. Fixing line 92 makes the two passages consistent.)

### 2. Scope error in Test 2 (line 57)

**Original draft:**

> "We tested this directly. The mean absolute distance from the rulebook edge is **1.27 inches** for both 7-hole and 3-hole batter-issued challenges. A Kolmogorov-Smirnov test on the full edge-distance distribution returns **p = 0.19**. There is no signal that 7-hole batters are challenging different pitches."

**Replace with:**

> "We tested this directly. The mean absolute distance from the rulebook edge is **1.27 inches** for both 7-hole and 3-hole challenges (all challenger types pooled, n=213 / n=259). A Kolmogorov-Smirnov test on the full edge-distance distribution returns **p = 0.19**. The same direction holds when restricted to batter-issued challenges (1.32 in vs 1.29 in). There is no signal that 7-hole batters — or 7-hole at-bats more broadly — face different challenged pitches than 3-hole."

**Reason:** The 1.27 number and the KS p=0.19 are computed on the all-challenger denominator (n_x=213, n_y=259, both matching `h1_overturn_rate_by_spot` not `h1_batter_only_overturn_rate_by_spot`). The current draft attributes them to "batter-issued challenges," which is technically wrong. The corrected wording preserves the structural argument — pitch selection doesn't explain the gap — but stops misattributing the all-challenger numbers to the batter-only subset.

### 3. H4 disclosure that random slope is on bottom-of-order indicator (Test 4)

**Original draft (lines 86–88):**

> "We tested this with proper hierarchical shrinkage. **78 umpires qualified** (≥50 borderline calls each in lineup spots 7-9 *and* in spots 1-3). Each umpire's posterior distribution on the spot-7-vs-3 effect was estimated jointly with all other umpires, with partial pooling toward the league mean."

**Replace with:**

> "We tested this with proper hierarchical shrinkage. **78 umpires qualified** (≥50 borderline calls each in lineup spots 7-9 *and* in spots 1-3). Each umpire's random slope was estimated on a bottom-of-order indicator (spots 7-9 vs 1-3), not strict spot-7, to preserve statistical power; the league-mean spot-7 effect from the main GAM (the −0.17pp number above) is what we contrast against. A strict-spot-7 random-slopes sensitivity is queued for our All-Star-break re-test. Each umpire's posterior distribution was estimated jointly with all other umpires, with partial pooling toward the league mean."

**Reason:** The R2 cross-review explicitly flagged this and the comparison memo locked the article-side disclosure: "H4 is on bottom-of-order indicator, not strict spot-7. Strict spot-7 sensitivity is queued for Round 3." The methodology section does mention this at line 165, but Test 4 itself reads as if the per-umpire test was on strict spot-7. The corrected wording is short, surfaces the methodological choice in the right place, and doesn't undermine the conclusion (which is still null).

## Editorial concerns (non-blocking)

### A. Codex's n=2,767 vs Claude's n=28,579 in the Test 3 table

The Test 3 table lists both methods side by side without disclosing that Codex's counterfactual GBM ran on n=2,767 borderline pitches (a tighter borderline definition) while the Bayesian GAM ran on n=28,579. The comparison memo distinguishes them explicitly. The article's narrower-CI-is-an-artifact paragraph already explains why Codex's interval is tighter, so adding the sample-size disclosure is editorial polish rather than a correctness fix. If you want to add a footnote — "Codex's borderline definition was tighter (±2 in vs Claude's ±0.3 ft); the small-sample bootstrap CI is what the discussion below addresses" — that would close the loop, but I won't block on it.

### B. Cross-review word count ("~800-1,000 words each")

R1 reviews are 1,016 (Claude) and 806 (Codex) words. R2 reviews are 1,089 and ~1,100 words. The "~800-1,000" is fine for the R1 set; if you want strict accuracy across both rounds, "~800-1,100 words each" is more honest. Tiny. Non-blocking.

### C. Voice/tone

The voice is consistent with Coaching Gap and Pitch Tunneling Atlas — rigorous, willing to flag its own errors (the dual-agent-process callout is exactly the right note), no condescension. The "We mention this not to perform humility but because errors of this kind get into published baseball analytics constantly" line is the brand. The "What we're not claiming" section is well-bounded — it doesn't overclaim a debunk and it doesn't underclaim what was tested. The phrase "doesn't exist at any level we can credibly test" is strong; I'd defend that wording in front of a FanGraphs editor.

### D. The "process-death recovery" thread

R1 comparison memo notes the stratified H3 had a process-death recovery and the recovered evidence is partial. The article doesn't mention this, which is fine — the Test 3 numerical claim (−0.17pp) is from the main GAM, not the stratified runs, and the stratified result is "no stratum where the effect lives," which is a robustness check rather than the headline. I think the omission is correct; flagging in case anyone comes back to it.

## Final recommendation

**Ship after the three required changes above are integrated.** The piece is at the rigor level the brand requires — six tests, two methods, two cross-reviews, all null on the bias mechanism, and the residual signals (the H7 low-chase shape, the Smith/PCA hitter residuals, the random-slope SD upper bound) are reported with the right epistemic weight. The two sign/scope errors I flagged are exactly the kind of thing that, if missed, would land in the comments and become the discourse — fix them and we ship clean. I would defend every Bayesian-method claim in this draft, as edited, in front of a FanGraphs editor.

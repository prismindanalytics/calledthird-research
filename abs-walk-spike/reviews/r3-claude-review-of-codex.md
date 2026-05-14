# Round 3 — Claude's Review of Codex

## Headline assessment

Codex's R3 is the cleanest of the three rounds and I trust most of the conclusions, with two important caveats. The H1 triangulation is now genuinely defensible — three game-level refit methods land in a tight band (24.3%–28.8%) versus my own band (−58.6% to +38.3%) that is much wider because Method A is on a different estimand. The H3 archetype direction (ρ=−0.258 vs my ρ=−0.282) is convergent to the second decimal and gives us a publishable cross-method signal. Where I split from Codex is on H2: their nine "stable adapters" come from a different estimand than my zero, and the calibration audit (343/1000 H2 GBMs failing >5pp OOB) is not as decorative as their framing suggests. The headline-level disagreement is mostly methodology, not data, and the comparison memo needs to say that.

## Critical issues (potential blockers)

1. **H2 estimand mismatch is not flagged.** `codex-analysis-r3/h2_adapter_leaderboard.py:79–119` builds the shift table as 2025 full-season vs 2026 window deltas (`z25` vs `z26`, mix of 2025 mix vs 2026 mix, JSD between *years*). My `claude-analysis-r3/h2_adapter_leaderboard.py:64–112` builds the shift as 2026-week-1 vs 2026-week-last (within-2026 trajectory). Both pass the brief literally because the brief did not nail the estimand, but they answer different questions: "who looks different from his 2025 self?" (Codex) vs "who is changing during the window?" (Claude). This is the entire source of the 0-vs-9 divergence. It is not a methodology bug — it is a definitional split, and the article cannot just say "9 adapters."
2. **H2 calibration deviation is on year-shift classifiers used to derive SHAP shares**, not on the shift table itself (`h2_adapter_leaderboard.py:200–209`). So the 343/1000 OOB-bin-over-5pp does *not* invalidate the leaderboard rankings (those are descriptive: zone deltas, top-share deltas, JSD, all deterministic). It does cast doubt on the SHAP feature-group shares in `findings.json:177–184` ("arsenal/mix" ~75%, "zone-location" ~25%) because those shares come from a classifier ensemble where 34% of refits are miscalibrated. The leaderboard names are auditable; the "this pitcher changed his arsenal" interpretation is not.
3. **Method A point estimate jumped from R2's 35.3% to R3's 28.8% under the new bootstrap, but my Method A landed at −58.6%.** Codex's number is plausibly the right one — their `h1_triangulation.py:375–411` does proper game-level resample + refit. My Method A is a per-take Bernoulli replay (60 perturbation draws, not 200 game refits) and is genuinely a worse implementation of the brief's spec. Codex's H1 Method A is the better Method A; I should not be the one defending mine.

## The H2 divergence (priority — your 0 vs their 9)

After reading their code line-by-line: bootstrap N is 200 (`h2_adapter_leaderboard.py:315`), magnitude thresholds are identical (15pp zone OR 15pp top OR JSD≥0.05, `:101`), stability filter is identical at ≥80% top-15 appearances (`:342`). The methodology is faithful to the brief on every parameter. The single difference is the estimand: Codex compares 2025 to 2026, I compared within-2026 weeks.

Which is right? Both. They answer different questions. Codex's question is the one the *brief* asks ("Adapter detection — who looks different from baseline?") and his answer is more defensible. Mine asks "who is *currently* shifting?" which is the question a manager cares about but the article does not.

Codex's nine names with stability ≥0.80: Pérez (Cionel), Dollander, Mejia (Juan), Zeferjahn, King (John), Senzatela, Young (Brandon), Hancock (Emerson), Hart (Kyle). None of these appear in *my* top-25 magnitude passers in `claude-analysis-r3/findings.json:166–718`, which is itself a check: if the same pitchers appeared in both lists with different stability scores, that would be a reconciliation problem. They don't. The two estimands genuinely produce disjoint populations.

**Editorial recommendation:** the article should publish Codex's nine, framed as "year-over-year zone/mix shifters with bootstrap-stable rankings." It should not present them as "the pitchers who adapted in real time." And it must footnote the SHAP feature-group shares, not lead with them.

## The H1 cross-agent triangulation

Six estimates: −58.6% (Claude A), +14.5% (Claude C), +38.3% (Claude B), +24.3% (Codex C), +27.0% (Codex B), +28.8% (Codex A).

Sorted: −58.6, +14.5, +24.3, +27.0, +28.8, +38.3. Median = **+25.65%**.

The editorial CI is harder. Naïve min/max gives [−80.1, +70.1] which is unpublishable. The honest read: my Method A is methodologically inferior (60 draws ≠ 200 game refits) and should be dropped from the median; my Method B (kNN counterfactual) and Codex's three methods are five comparable estimates. Drop my A, take median of the remaining 5: median = **+27.0%** with editorial CI **[+5.7%, +70.1%]** (the union of Codex's narrowest defensible CI lower bound and my Method B's upper bound).

This number is consistent with R1's +40.5% and R2's +35.3% being too high — both prior rounds used fixed-model intervals. R3's honest answer is **~25–27% of the +0.66pp YoY walk-rate increase is attributable to the called-zone change**, with the rest divided between count-state mix, pitch-mix, and batter approach. That is the headline we publish.

## H3 cross-method intersection

This is where Codex really delivers. Spearman correlations agree to the third decimal: my −0.282 vs his −0.258 (Codex `findings.json:r3_h3.spearman_rho`, Claude `findings.json:720–734`). Same sign, same magnitude, same p<0.0001. This is a real archetype effect.

Named pitchers that appear in **both** stability-locked lists:

| Pitcher | Claude bucket (stab) | Codex bucket (stab) | Both stable? |
|---|---|---|---|
| Finnegan, Kyle | hurt (0.865) | hurt (0.91) | **YES** |
| Doval, Camilo | helped (0.87) | helped (0.915) | **YES** |
| O'Brien, Riley | helped (0.965) | helped (0.96) | **YES** |
| Miller, Mason | helped (0.825) | helped (0.68) | NO — Codex 0.68 < 0.80 |

**The three publication-locked names: Finnegan (command-hurt), Doval (stuff-helped), O'Brien (stuff-helped).** Miller drops off because Codex's stability score is below the 0.80 cutoff — I should not push back on this; the article should drop him from the named pitcher list and keep him only as an honorable mention if at all. Codex's additional stable hurt names (Matz, Burke Brock, Walker Ryan, Severino) do not appear in my stable list, but they are not contradicted by my analysis either; they fall just below my stability cutoff. The cleanest publication move is the three-name intersection and a one-sentence note that Codex's pipeline also surfaced four additional command-hurt names that did not clear my stability threshold.

## Methodology concerns (non-blocking)

1. **Codex's archetype proxy is similar but not identical to mine.** Both use Statcast fallback. Codex's `command` is "a blend of low walk-rate percentile and zone-rate percentile" (REPORT.md:43); mine is `mean(zone_rate_pct, -walk_rate_pct)`. These should produce nearly identical rankings but the language in the published piece should say "stuff and command proxies (not FanGraphs Stuff+)."
2. **Method C interval is reported as [4.6, 49.4]** in `findings.json:24–26`. The denominator is +0.66pp YoY. A 50% upper bound on a +0.66pp gap is +0.33pp of zone attribution, which is the same order as week-to-week noise. The 4.6% lower bound translates to +0.03pp — basically zero. Both intervals are correct, but the article should publish *percentages*, not *pp*, because the pp framing makes the effect look trivial.
3. **Bootstrap discipline:** I confirmed Codex uses game-level resample throughout (`sample_game_weights` in `h2_adapter_leaderboard.py:137, h1_triangulation.py:430`). No row-level shortcuts. This is the major lift from R2.

## Things Codex got right

1. The Method C bootstrap-of-bootstrap is faithfully 100 outer × 10 inner (`h1_triangulation.py:414`). My own Method C does the same. This is the first round where both agents implemented the brief's headline methodology.
2. The H1 GBM calibration audit was kept *separate* from the H1 attribution number, not buried. `READY_FOR_REVIEW.md:11` flags "at least one H1 bootstrap OOB calibration bin exceeded 5pp" explicitly. That is the right call.
3. The "stuff_command_interaction term has permutation importance below permuted-label baseline" finding (REPORT.md:45) is exactly the right way to handle a weak interaction: report the direction (ρ=−0.258 is real) but refuse to oversell the nonlinearity. I had the same instinct on my side.
4. The 9-name leaderboard is not framed as "the answer" — REPORT.md:33–34 explicitly says cross-method intersection is required. This is the editorial discipline that was missing in R2.

## Recommendation for the comparison memo

Publishable consensus: **H1 = ~25–27% of the +0.66pp YoY walk-rate gap is zone-attributable, editorial CI [+5.7%, +70.1%]**. **H3 = three stability-locked named pitchers (Finnegan, Doval, O'Brien) with ρ≈−0.27 archetype slope**. **H2 = one publishable null (mine: within-window adaptation is not stable enough to name) plus one publishable nine-name year-over-year list (Codex's) — and the article must distinguish these as separate questions, not gloss them together.** Codex's R3 is A-tier work; the only reason this isn't a clean publish is that the H2 estimand split was never adjudicated at brief time, and the article must reflect both answers.

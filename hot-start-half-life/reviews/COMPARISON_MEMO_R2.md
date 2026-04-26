# Hot-Start Half-Life — Comparison Memo (Round 2)

**Date:** 2026-04-25
**Inputs:** `claude-analysis/REPORT_R2.md`, `codex-analysis/REPORT_R2.md`, `reviews/claude-review-of-codex-r2.md`, `reviews/codex-review-of-claude-r2.md`, both `findings_r2.json`, prior memo `reviews/COMPARISON_MEMO.md`.

---

## 1. Headline finding (one sentence)

**The published-leaderboard names baseball is talking about (Pages, Trout, Rice, Judge, Carroll) almost entirely fail to clear two independent methodological bars for "sustainable signal" — but five names baseball is NOT talking about (Caglianone, Pereira, Barrosa, Basallo, Dingler) and three relievers (Lynch, Senzatela, King) survived both methods independently and are the article's headline picks.**

R2 confirmed R1's noise-floor finding, killed R1's stabilization-shift finding, fixed Murakami reproducibility on both sides, and surfaced a clean convergent sleeper list. Cross-review then surgically downgraded several individual claims that one method got wrong.

---

## 2. Convergent claims (publishable)

These survived four independent methodological passes (R1 Claude, R1 Codex, R2 Claude, R2 Codex) AND the R2 cross-review.

| Claim | Confidence | Evidence |
|---|---|---|
| **Pages = NOISE** | **Bulletproof** | 4-method consensus. Both R2 methods (Claude posterior q50 wOBA = .344; Codex LightGBM ROS wOBA point = .332) say his .404 wOBA is BABIP excursion. Don't even need an article paragraph defending it. |
| **5 convergent sleeper hitters: Caglianone, Pereira, Barrosa, Basallo, Dingler** | **Strong** | Both Claude (Bayesian) and Codex (LightGBM) independently surfaced all 5 in their top-decile-of-predicted-ROS-delta lists. Both verified the analog kill-gate (cosine ≥ 0.70 with ≥ 5 analogs). These are the article's named picks. |
| **3 convergent sleeper relievers: Lynch, Senzatela, King** | **Strong** | Both methods agree, both screen out 2025 saves leaders, both have analog support. |
| **Noise floor is brutal** | **Bulletproof** | Claude: 10% of 2022-2025 22-game wOBA leaders sustained ≥ 85%; median ROS decline = -0.135. Codex: 0% of 22-game BA leaders maintained ≥ 90%; 5% of OPS leaders. Same direction, same magnitude. |
| **Mason Miller K% rise is real; streak length is not credibly forecastable** | **Strong** | Both methods now agree on the K% direction (Claude posterior q50 = .500; Codex projects 34.6%). The R1 65/45/33% streak survival probabilities are killed in both pipelines. |
| **R1 "3 of 5 stabilization rates shifted" headline is DEAD** | **Bulletproof** | Claude's corrected player-season cluster bootstrap shows only wOBA still flags, and Codex's own R2 review notes that's "fragile" (110/120 draws crossed; upper CI hits the artificial 3,600-PA grid cap; lower CI is 298 vs Carleton 280). Era story doesn't survive proper methodology. |
| **Murakami pipeline now reproducible** | **Strong** | Both `r2_data_pull.py` files now call MLB Stats API as fallback when `playerid_lookup` returns empty. Murakami's MLBAM (808959) regenerates from clean checkout in both pipelines. |
| **SHAP and era counterfactual = clean kills** | **Strong** | Codex dropped both per the R1 critique. No goalpost moving. No reappearance through a side door. |

**Article spine:** *"April mostly lies — Pages's .404 won't survive, Judge's .435 mostly absorbs into his already-elite prior, and Mike Trout's Yankee Stadium burst doesn't update what we already knew. But two independent methods independently surfaced five hitters baseball isn't talking about whose underlying components project as durable. Here's the list."*

---

## 3. Divergent claims — which side is right after R2 cross-review

| Claim | Claude R2 | Codex R2 | Verdict after cross-review |
|---|---|---|---|
| **Rice and Trout** | Both flipped from R1 AMBIGUOUS → SIGNAL "driven by adding contact-quality features" | Both NOISE; LightGBM has the same features and refuses to update an elite prior | **Codex closer to right; downgrade to AMBIGUOUS for the article.** Codex's review of Claude proves the killer point: Claude's `r2_bayes_projections.py:358-387` hard-codes a 50/50 wOBA + xwOBA blend; EV p90, HardHit%, Barrel% are computed but DON'T enter the ROS wOBA estimator. The "contact-quality flip" is mislabeled — it's a hand-tuned xwOBA blend with no holdout RMSE, no calibration. Trout's R2 SIGNAL delta is +.005 to +.045 — well within prior uncertainty. The honest article call is **AMBIGUOUS** for both, with the framing: "two methods disagree; the conservative read is regression to a strong prior." |
| **Tristan Peters as #1 Codex sleeper** | (not on Claude's list) | #1 sleeper at +.145 ROS-wOBA delta | **Drop Peters from any article copy.** Claude's review of Codex caught it: Peters has `preseason_prior_woba = 0.0` (debut). His "predicted ROS wOBA" of .145 is literally his delta + 0. The 80% band is [-0.007, .302] — lower bound is sub-replacement. Five nearest analogs (Herrmann 2016, Brown 2021, Garlick 2022, Voit 2017, Kang 2015) average ROS wOBA ~.329. The pick is a definitional accident from ranking by delta-vs-zero. **Action:** Codex must filter `preseason_prior_woba > 0` before ranking; Claude's convergent picks (Caglianone, Pereira, Barrosa, Basallo, Dingler) move up correctly. |
| **Louis Varland reliever** | #1 SLEEPER (+.05 vs prior, prior anchored to late-2025 closer K%) | #2 FAKE-DOMINANT (-.046 vs 3-year-weighted prior) | **Codex's framing closer to right; drop Varland from sleeper list.** Even Claude's own model places Varland on BOTH the sleeper AND fake-dominant boards mechanically (`r2_reliever_board.py:222-233`) — the rules don't conflict but the editorial claim "Varland is a clean sleeper" is wrong. Use Codex's framing: his April K% is real but the predicted ROS K% shrinks back. Reliever sleepers reduce to convergent 3: Lynch, Senzatela, King. |
| **H2 fake hots: 0 vs 10** | Posterior absorbs prior; 0 fake hots exist | LightGBM delta-sign rule promotes 10 (Judge, Carroll, Trout, Muncy, etc.) | **Claude's framing more defensible; Codex's list is rule-mechanics, not signal.** Codex itself wrote "the right editorial wording is not 'these players will collapse'; it is 'the hot-start portion is not adding positive information beyond the baseline'" — that's an editorial cop-out for a screen that mechanically labels Aaron Judge a fake hot. **No publishable fake-hot list.** Either tighten the rule (`pred_ros < prior - 1 SD`) for a Round 3 attempt, or drop the section. |
| **H3 fake colds: 0 vs 10** | Same — Bayesian delta refuses to clear zero on bottom-decile hitters | Codex names 10 (Henry Davis, Victor Scott, Cowser, Bailey, etc.) | **Same divergence; same problem.** Codex's H3 list contains real buy-low candidates by xwOBA gap, but the predicted-positive-delta gate is too permissive. **Drop the fake-cold section** OR re-frame as "biggest xwOBA-vs-actual gaps" without the verdict-style language. |
| **Codex QRF "calibration" = 85.4%** | (audited by Claude review) | Reports calibrated 80% interval coverage = 85.4% on 2025 holdout | **Calibration framing is theatre.** Claude's review caught it: the conformal margin in `r2_qrf_coverage.py:62-67` is **0.0** for both hitters and relievers. Raw and "calibrated" coverage are byte-identical at 85.4% (hitter) and 83.0% (reliever). The QRF was already over-covering on validation; the conformal step found no positive miss to expand by. **The 85.4% is real but the framing is wrong.** Intervals are over-wide, not calibrated — every "NOISE" verdict from interval-crosses-zero sits on a too-permissive band. **Action:** Codex must drop the calibration language and own that intervals are deliberately wide, or actually shrink them on validation before re-scoring. |

---

## 4. Methodology deltas

**What both R2 implementations got right (vs R1):**
- Murakami reproducibility (both)
- Mason Miller streak survival killed (both)
- SHAP / era counterfactual cleanly retired (Codex)
- Universe scan ≥ 250 hitters and ≥ 70 relievers (both)
- Stabilization corrected with player-level resampling (Claude — see caveats below)
- QRF coverage diagnostic exists (Codex — see caveat above)

**What's still wrong after R2:**
- Claude's "Rice/Trout SIGNAL is contact-quality-driven" mislabels the actual mechanism (50/50 xwOBA blend; not all-features-blend). No holdout validation of the blend.
- Claude's "hierarchical Bayesian" claim is still partly window dressing. The shared-kappa model exists in `r2_bayes_projections.py:604-610` but the production rankings still use per-player conjugate updating; the hierarchical fit is a side-output sanity check, not the production estimator.
- Claude's stabilization repair, while substantially fixed, equal-weights players (one season per sampled player per draw) — that's a changed estimand, not a "true player-season bootstrap." Codex's review notes this (`r2_stabilization.py:163-166`, `:187-197`).
- Codex's xwOBA-gap "feature top-tier" hedge: built three variants (signed gap, abs gap, vs-prior gap) and one ranks 2 — that's having it both ways. Pick one.
- Codex's QRF calibration = 0.0 conformal margin (see §3 above).

---

## 5. What gets published

The article works — better than the R1 framing — with the cross-review corrections applied.

**Article skeleton (revised):**

1. **The lead.** April mostly lies. Among the 22-game wOBA top-5 leaders 2022-2025, only 10% sustained ≥ 85% of pace. Median decline: -0.135 wOBA. Two independent methods independently confirm. This is the spine.

2. **The named retrospective.** Pages = NOISE (bulletproof, 4-method consensus). Trout, Rice = AMBIGUOUS — Claude's contact-quality blend says signal but it's a hand-tuned 50/50 xwOBA blend without validation; Codex's LightGBM with the same features says noise; honest answer is the conservative read. Murakami = SIGNAL with caveat (only the cleanest survivor; prior is league-average proxy because no NPB translation). Mason Miller = K% rise is real (.500 posterior); streak length is not credibly forecastable, so we don't try.

3. **The headline picks (sleepers nobody is talking about).** **Five hitters surfaced by both methods:**
   - **Jac Caglianone** (LAA prospect, .312 wOBA, both methods rank top decile on contact quality)
   - **Everson Pereira** (NYY, .406 wOBA — quietly behind Rice on the leaderboard)
   - **Jorge Barrosa** (ARI/SF utility OF, .299 wOBA — discipline + contact)
   - **Samuel Basallo** (BAL prospect, .306 wOBA)
   - **Dillon Dingler** (DET catcher; xwOBA-gap +.093 on Codex's board)
   
   **Three relievers surfaced by both methods:**
   - **Daniel Lynch** (KC, +9-11 K% vs prior)
   - **Antonio Senzatela** (COL, converted starter — interesting story)
   - **John King** (TEX)

   Each named pick gets: April line + preseason prior + nearest historical analog + WHY both methods saw the signal.

4. **The methodological humility box.** *Why didn't we publish "the era stabilization shifted"?* R1 said three rates shifted; R2's corrected cluster bootstrap shows only wOBA, and even that's fragile. *Why didn't we publish a "fake hot starter" list?* Because the only credible methodology (Codex's LightGBM delta-sign rule) mechanically labels stars with high priors as fake hot — Aaron Judge running .435 instead of .500 is not actually a fake hot story. *Why aren't Tristan Peters or Louis Varland on the sleeper list?* Both got promoted by definitional accidents in one method's ranking rule and were caught on cross-review.

**What gets killed (post-cross-review):**
- The "Rice and Trout flipped to SIGNAL via contact-quality features" claim (mechanism is mislabeled; downgrade to AMBIGUOUS)
- Tristan Peters from any sleeper list (zero-prior arithmetic accident)
- Louis Varland from sleeper relievers (his own model puts him on fake-dominant)
- The fake-hot section as a confident set of "these players will regress" calls (rule too lenient; reframe as "April performance and prior wOBA, ranked by daylight" if used at all)
- Any claim that the era stabilization shifted (R1 dead; R2 only wOBA flags weakly)

---

## 6. Decision: ship now, or Round 3?

**Ship now (recommended):** The convergent picks are publishable. The article has a sharper spine after R2 than after R1. Frame Rice/Trout/Mason as AMBIGUOUS-ish honest portraits with both methods cited; the 5 sleeper hitters + 3 sleeper relievers are the headline. Article ships in 2-3 days of writing.

**Round 3 (optional):** Would add precision on the Rice/Trout flip (validate Claude's contact-quality blend on holdout; learn the right blend coefficients from 2022-2024 → 2025), tighten the fake-hot rule (`pred_ros < prior - 1 SD`), filter `preseason_prior > 0` from Codex's sleeper ranking, build NPB-translation prior for Murakami, replace Codex's 0.0 conformal margin with a real calibration. Won't change the headline picks. Adds maybe 1 day for both agents.

**My pick:** **Ship now.** Round 3 would polish edges that don't change the article. The convergent sleeper list is the differentiated content; the methodological humility box is what makes it CalledThird-distinctive vs ESPN's leaderboard.

---

## 7. Open questions for any future round

1. **Validate Claude's contact-quality blend on holdout.** Currently a hand-tuned 50/50 wOBA + xwOBA average. Should be a learned coefficient (or full-feature model) with 2025 RMSE.
2. **Tighten the "fake hot" rule** — `pred_ros < prior - 1 SD` would exclude high-prior stars and produce a cleaner editorial list.
3. **Filter `preseason_prior > 0`** in Codex's sleeper ranking to avoid the Peters-style arithmetic accident.
4. **NPB-translation prior for Murakami** — would tighten his SIGNAL projection meaningfully.
5. **Real conformal margin for Codex's QRF** — current 0.0 is over-coverage by accident, not calibration.
6. **Production-path hierarchical Bayes for Claude** — currently the shared-kappa fit is a side-output diagnostic; integrate into the universe ranking estimator.
7. **RE24-based reliever survival** for any future Mason Miller-style streak analysis.
8. **Component-stat PELT** for individual hot starters — R1 PELT was wOBA-only; component-stat change-points might find structure missed.

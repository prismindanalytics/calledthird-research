# Hot-Start Half-Life — Comparison Memo (Round 1)

**Date:** 2026-04-25
**Inputs:** `claude-analysis/REPORT.md`, `codex-analysis/REPORT.md`, `reviews/claude-review-of-codex.md`, `reviews/codex-review-of-claude.md`, both `findings.json`.

---

## 1. Headline finding (one sentence)

**The hot-start narrative is mostly noise — Andy Pages's .404 wOBA is a 94-PA BABIP excursion, the historical noise floor is brutal (≤ 10% of April wOBA leaders 2022-2025 sustained ≥ 85% of their pace), and the only April performances both methods independently let through as plausibly real are component-level signals (Rice's power/discipline, Trout's K%, Miller's K%) — *not* full-line MVP-tier durability for any of the named hot starters.**

The strong methodology-novelty story (*"the 2026 environment shifted stabilization rates by enough to invalidate Carleton 2007"*) does not survive cross-review and should not be the article framing.

---

## 2. Convergent claims (both agents agree, with independent methods)

These are the publication-ready findings.

| Claim | Claude evidence | Codex evidence | Strength |
|---|---|---|---|
| **Pages = NOISE** | Bayesian posterior: every prior inside 80% interval; q50 wOBA = .344 vs observed .404 | LightGBM ROS wOBA point = .337, QRF q50 = .329, prior = .331; nearest analog James McCann 2019 (ROS wOBA .336) | **Very strong.** Two completely different methods, same verdict. The .404 is a BABIP/luck excursion. |
| **Trout regresses to ~.370 wOBA** | Posterior q50 wOBA = .373 | LightGBM point = .375, QRF q50 = .363; nearest analog his own 2019 | **Strong.** Both find the prior *already strong*; current pace doesn't add signal. The 5-HR Yankee Stadium series is selection-bias noise. |
| **Rice regresses to ~.345-.370 wOBA** | Posterior q50 = .368 (above his .345 prior — the discipline + power components are real) | LightGBM = .343, QRF q50 = .340; nearest analogs Eric Thames 2017, Dan Vogelbach 2019 | **Strong.** Both shrink hard from the .500 current. Component differences exist (see §3). |
| **Noise floor is brutal** | 10% of 2022-2025 22-game wOBA top-5 sustained ≥ 85%; median ROS decline = -0.135 wOBA | 0% of 2022-2025 22-game BA leaders maintained ≥ 90%; 5% of OPS leaders | **Very strong.** Independent computations on the same corpus, same direction, same magnitude. |
| **K% and BB% time-invariant** | 2022-2025 stabilization PAs match Carleton 2007 within CI (BB% 122 vs 120, K% 54 vs 60) | Permutation importance ranks `whiff_per_swing_22g` and `bb_rate_22g` as informative survivors | **Moderate.** Different framings, same underlying story: plate discipline stabilizes early and remains the most-trustworthy April signal. |
| **Mason Miller K% rise is real** | Posterior K% q50 = .500 (vs .407 prior); BB% in noise band | Reliever forest agrees K% is the durable component of his start | **Moderate.** Convergent on K%, divergent on streak survival framing (§3). |

**Robust article spine:** *"Of the five names baseball is talking about, two methods independently agree the headline numbers won't survive — but each leaves behind one or two component signals worth watching. Here's which."*

---

## 3. Divergent claims (and which side is more credible)

| Claim | Claude says | Codex says | Verdict after cross-review |
|---|---|---|---|
| **Era stabilization effect** | 3 of 5 rates (wOBA, ISO, BABIP) shifted ≥ 10% vs Carleton with non-overlapping bootstrap CIs | Era counterfactual delta = -0.0012 wOBA, CI [-0.020, +0.015] → null | **Both methodologies are flawed; the article cannot lean on this.** Claude's "bootstrap CI" partitions PAs within fixed players — it is not a true sampling-uncertainty CI and ignores player-season sampling (Codex review #1 confirmed by reading `stabilization.py:214-243`). Codex's counterfactual confounds model capacity (10-yr vs 4-yr training samples) with era effect, and N=5 evaluation rows has no statistical power (Claude review #2). The honest claim ceiling: *"Carleton's 20-year-old stabilization rates may not apply to the current era, but neither agent's methodology can demonstrate the shift cleanly. We retain the empirical noise-floor finding (which doesn't depend on this) and flag stabilization as a Round 2 question."* |
| **Murakami verdict** | SIGNAL (4/5 stats real, posterior wOBA q50 = .378) | EXCLUDED (substituted Ballesteros = AMBIGUOUS) | **Claude wins on coverage; Codex wins on reproducibility critique.** Claude resolved Murakami's MLBAM via MLB Stats API; Codex only tried `pybaseball.playerid_lookup` (predictable miss for 2026 debuts). But Codex review #4 is correct: Claude's `data_pull.py:172-193` only uses `playerid_lookup` too — the Murakami parquet was manually edited, so `analyze.py` would not recreate the SIGNAL case from a clean checkout. **Action: Claude must add MLB Stats API resolution to `data_pull.py` so the Murakami pipeline is reproducible.** Until then, treat Murakami as "best-available signal with a documented caveat about prior weakness," not a clean SIGNAL claim. |
| **Mason Miller streak survival** | 65% / 45% / 33% probability streak extends ≥ 5/10/15 IP | Median expected IP to next ER ≈ 2.2 | **Codex wins.** Claude's `bayes_projections.py:524-538` models first-ER as Geometric on a Beta-mixed rate where the "observed ER" input is actually observed HR (not earned runs from baserunner sequencing). That's a toy model. Codex's RA9 proxy is admittedly rough, but it's at least directionally about runs allowed, not just home runs. Claude's 65/45/33 percentages are *numerology* and should not be published. **Action: kill the streak-extension probabilities entirely. Replace with: "Posterior K% rises to elite-tier .500; streak length itself is not credibly forecastable from 41 BF."** |
| **Pages-style hot-start mechanism** | Bayesian model finds the start "fully explained by BABIP excursion" | LightGBM uses EV/HardHit/Barrel features; reaches the same NOISE verdict | **Convergent verdict; Codex wins on mechanism credibility.** Codex review #5 is correct: Claude's projection loader (`bayes_projections.py:607-618`) reads only event-outcome fields. Claude literally cannot see the contact-quality evidence that fans care about ("his hard-hit rate is 95th percentile!"). The Pages NOISE verdict is right, but Claude's *explanation* ("BABIP-only") is incomplete. Codex's analog set (McCann 2019, Duran 2023, Bogaerts 2022) all had similar contact profiles — that's the more persuasive story. **Action: The article uses Codex's contact-quality framing for *why* Pages is noise, but cites both methods for *that* he is.** |
| **Rice components** | Posterior shows BB%, K% (negative), and ISO all genuinely shifted; wOBA shrinks to .368 | LightGBM ROS wOBA = .343; analog set is power-heavy (Thames, Vogelbach) | **Both methods agree he regresses; Claude offers richer component story.** Claude's three-of-five-stats decomposition is valuable for the article ("the power and discipline are real, the batting line isn't"). Codex's analog set independently corroborates: Eric Thames 2017 ran a similar 22-game wOBA pace and finished with .345 ROS wOBA — almost exactly Claude's q50. **Action: lead with Codex's analog ("Rice's nearest historical match is Eric Thames 2017"), back-fill with Claude's component decomposition.** |

---

## 4. Methodology deltas (where one approach clearly outperformed)

**Claude's strengths over Codex:**
- Per-starter component decomposition (BB%/K%/BABIP/ISO/wOBA each with their own posterior)
- Reliever-specific stabilization rates correctly applied to Miller
- Convergence diagnostics actually run (R-hat, ESS reported)
- Resolved Murakami's player ID (whatever the reproducibility issues)

**Codex's strengths over Claude:**
- Used contact-quality features (EV p90, HardHit, Barrel, xwOBA) — the brief explicitly required these and Claude's projection model ignored them
- True historical-analog retrieval with cosine similarity (clears the kill-gate ≥ 0.70 threshold for all four projected hitters)
- LightGBM test-set diagnostics (RMSE/MAE/R²) provide an external benchmark for projection quality
- Cleaner per-feature importance hierarchy (despite the SHAP/permutation rank disagreement)

**Both methods are weaker than they claim:**
- Claude's "hierarchical Bayesian" is one fixed-prior conjugate Beta-Binomial — the NUTS/MCMC layer adds nothing computationally and the R-hat reporting is theatre (Codex review #2). The model would be more honest as "empirical-Bayes shrinkage with conjugate update."
- Claude's "bootstrap CI" on stabilization rates is not a sampling-uncertainty CI — it's a within-player random-partition CI (Codex review #1). That alone disqualifies the "non-overlapping CIs vs Carleton" kill-gate language.
- Codex's QRF prediction intervals were never coverage-checked on the 2025 holdout (Claude review #4). With N=426 test rows you could trivially compute "what fraction of held-out 2025 ROS wOBA fall inside [q10, q90]?" Without that diagnostic, the all-noise verdict is forced by interval narrowness.
- Codex's SHAP/permutation Spearman = 0.195 vs the pre-committed 0.60 threshold (Claude review #3) — Codex moved goalposts in `tables/shap_rank_investigation.md` rather than disqualifying the feature hierarchy.

---

## 5. What gets published

The article works. The framing needs to change from *"new methodology proves the era has shifted"* to *"April lies — here's the evidence, with proper humility about which signals survive."*

**Article skeleton:**

1. **Lead with the noise floor.** Of the 20 player-seasons that led 22-game wOBA in 2022-2025, only 2 sustained ≥ 85% of that pace. Median decline: -0.135 wOBA. *Both* dual agents independently confirm. This is the spine.

2. **Pages: noise.** Two methods, two verdicts, same answer. Use Codex's analog narrative (McCann/Duran/Bogaerts) and Codex's contact-quality framing for *why*. Use Claude's posterior interval (q10-q90 = .319-.370) for the projection. This is the strongest cross-validated finding in the report.

3. **Rice: noise on the headline, signal on the components.** Lead with Eric Thames 2017 analog. Back-fill with Claude's BB%/K%/ISO decomposition showing the power and discipline are real but the batting line isn't.

4. **Trout: noise.** Both methods say his prior is already strong; the Yankee Stadium burst doesn't add signal. wOBA q50 ≈ .370 — All-Star caliber, far short of current pace.

5. **Murakami: cleanest survivor, with caveats.** This is the only April performance our analysis lets through as plausibly durable, but: (a) his prior is a 60-PA league-average proxy because no NPB-translation was used; (b) Codex's pipeline couldn't even resolve his player ID with off-the-shelf tools, which itself is a story about how 2026 debuts break standard sabermetric infrastructure. State both.

6. **Mason Miller: K% will hold; the streak prediction is not credible.** Posterior K% rises to .500 across both methods. Drop the 65/45/33% streak-extension probabilities — they came from an HR-only ER proxy that doesn't model baserunner sequencing.

7. **The methodological humility box.** *Why didn't we publish "the era has shifted"?* Because both agents tried, both agents found something, and on cross-review neither methodology held up. The honest answer is: Carleton's 2007 reference might no longer apply, but proving that requires a Round 2 with a properly bootstrapped player-season-resampling design.

**What gets killed:**
- "3 of 5 stabilization rates shifted ≥ 10%" framing
- "Era counterfactual is null" framing
- Mason Miller streak-extension probabilities (65/45/33)
- "Hierarchical Bayesian" claim — downgrade to "empirical-Bayes conjugate update"

---

## 6. Open questions for Round 2

1. **Proper stabilization bootstrap.** Resample player-seasons with replacement (not within-player PAs); cluster the bootstrap by player to handle within-season correlation. Re-test the kill-gate.
2. **Era counterfactual with matched sample sizes.** Subsample 2015-2024 to non-overlapping 4-year windows and compare predictions across windows; isolate era effect from training-set-size effect.
3. **Add contact-quality features to Claude's Bayesian projection prior.** EV p90, HardHit, Barrel, xwOBA — the brief required these and Claude's model didn't use them.
4. **Coverage-check Codex's QRF intervals.** What fraction of 2025 holdout ROS wOBA actually fall inside the [q10, q90] band?
5. **Build a proper reliever streak survival model.** RE24-based or Markov state model conditioning on baserunner state — kills the HR-only proxy.
6. **NPB-translation prior for Murakami.** Strengthens (or weakens) his SIGNAL verdict materially.
7. **Reproducibility fix for Murakami's player ID.** Add MLB Stats API resolution to `data_pull.py` so the SIGNAL claim survives a clean checkout.
8. **Pages PELT change-point limitation.** PELT was applied only to per-PA wOBA contribution; component-stat change-points (BB% / ISO / BABIP separately) might find structure the wOBA series misses.

**Recommended Round 2 scope:** Items 1, 3, 4, 7 are the minimum to make the published findings reproducible and methodologically defensible. Items 2, 5, 6, 8 are valuable but not blocking.

# Round 2 Brief — The 7-Hole Tax

**Status:** Round 2 — agents launching
**Setup:** Round 1 produced a clean league-aggregate null on the umpire-zone-bias claim and a directionally suggestive but underpowered raw deficit on batter-issued challenges. Round 2 asks whether the league-aggregate null masks a per-umpire signal, and whether specific named hitters at the bottom of the order are being personally short-changed.
**Decision criterion at end of Round 2:** Round 2 produces enough positive structure to convert the article from "we tested it and mostly no" (B+) to "here's what's actually in the data" (A). Three possible Round-2 outputs all support an A-tier piece:
1. **Some umpires exhibit the bias even though the league doesn't.** → Leaderboard piece, name names.
2. **No umpire exhibits it after shrinkage either.** → Comprehensive-debunk piece, exhaustive.
3. **A few specific hitters are personally getting short-changed.** → Named-hitter piece, surface the actual injustice.

Any combination of (1)+(3) is best. (2) alone is still an upgrade over Round 1's narrower scope.

---

## 0. What Round 1 settled (do not re-litigate)

These conclusions are locked in from Round 1; do not re-test:

- **H3 league-aggregate null on borderline called pitches.** Bayesian: −0.17pp [−1.5, +1.2], n=28,579. ML counterfactual: −0.35pp [−0.39, −0.31]. Both essentially zero. **This is the rock the article is built on. Round 2 builds *on top of* this finding, not against it.**
- **H1 raw replication is directional but underpowered.** 7-hole batter-issued overturn rate = 37.1% (n=89), Wilson CI [27.8, 47.5]. League rate 45.2%. Deficit 8.1pp, below the 10pp pre-reg gate. CI contains both 30.2% and 45.2%. Do not re-test this; it is what it is.
- **Selection probe: no edge-distance gap by lineup spot.** Mean |edge_distance| = 1.27 in for both spot 7 and spot 3, KS p = 0.19. Spot 7 batters do not challenge harder pitches.
- **Bottom-of-order pattern, not 7-specific.** Spots 7, 8, 9 cluster at ~37%; q ≥ 0.62 after BH correction. Spot 7 is statistically indistinguishable from its neighbors.
- **Denominator split is real.** Batter-issued = 37.1%; pooled all-challenger where batter-at-plate is 7-hole = 51.2%. Catcher pooling absorbs the framing edge. Both numbers are correct under their own definitions.

What's open and Round 2 will answer:
- Whether any individual umpire shows credible per-umpire lineup-spot bias once we shrink properly
- Whether any individual hitter in spots 7-9 has lost more borderline calls than their model-expected rate
- Whether the bottom-of-order pattern survives after stratifying by handedness and prior-season chase rate (the FanSided "elite pitch recognition" mechanism translated into a measurable feature)
- Whether catcher-initiated challenges in 7-hole at-bats have a different overturn pattern than batter-initiated ones (the pooled-vs-unpooled denominator anomaly)

---

## 1. The Round-2 sub-hypotheses

### H4 — Per-umpire lineup-spot bias
**Hypothesis:** With proper hierarchical shrinkage, at least one umpire with sufficient sample (≥50 borderline calls involving lineup spots 7-9) shows a credibly non-zero spot-7-vs-spot-3 effect on called-strike rate.

**Operational definition:** An umpire's effect is "credible" if its 95% credible interval (Bayesian) or 95% bootstrap CI (ML) excludes zero, AND the effect is at least 2pp in magnitude. With ~80 home-plate umpires through May 4, multiple comparisons are a serious issue: BH-FDR or hierarchical-shrinkage-based filtering required.

**Three possible outcomes:**
- Some umpires (≥1) show credible bias. → Name them.
- No umpires show credible bias after shrinkage and multiple-comparisons correction. → Comprehensive null, stronger debunk.
- A few umpires show *borderline* bias (CI just excludes zero, magnitude small). → Soft mention in the article body, but no named leaderboard.

### H5 — Named-hitter personal deficit
**Hypothesis:** Among hitters who have batted in spots 7-9 with ≥30 borderline-pitch take decisions, at least one shows a personal called-strike rate at least 3pp above their model-expected rate, with 95% CI excluding zero.

**Operational definition:** For each qualifying hitter, compute:
- Actual called-strike rate on borderline pitches faced this season
- Model-expected called-strike rate from the Round-1 controlled GAM (uses location, count, pitcher, catcher, umpire — does NOT use batter-specific features)
- Personal residual: actual − expected, with bootstrap or posterior-predictive CI

Hitters whose personal residual is significantly positive = hitters being personally short-changed beyond what the league model would predict.

**Cross-reference:** For each flagged hitter, surface their prior-season chase rate, walk rate, and contact metrics. The FanSided/Ringer mechanism implied "elite-pitch-recognition hitters in low slots are penalized." If our flagged hitters cluster at the high-discipline end of the chase-rate distribution, that's the mechanism the news pieces gestured at — surfaced rigorously.

### H6 — Catcher-initiated challenges in 7-hole at-bats
**Hypothesis:** The 51.2% pooled overturn rate for 7-hole at-bats (vs 37.1% batter-issued) reflects systematically different pitch selection by catchers when their pitcher is facing a 7-hole hitter. Specifically: catchers in 7-hole AB challenge worse calls (lower edge_distance, higher in-zone share) than they do in 3-hole ABs.

**Operational definition:** For challenges where `challenger == "catcher"`, compare the joint distribution of `(edge_distance_in, in_zone, count)` between 7-hole AB and 3-hole AB. KS or energy distance with permutation p-values.

**Why it matters:** If catchers ARE systematically picking harder fights with 7-hole at the plate, that's the mechanism that explains why pooled overturn rate (51.2%) is league-flat while batter-issued (37.1%) is deficit. Closes the loose thread Round 1 left open.

### H7 — Stratified by chase rate (the "pitch-recognition" mechanism)
**Hypothesis:** The 7-vs-3 effect on borderline called pitches differs by the batter's prior-season chase rate. Specifically: among the lowest chase-rate (most disciplined) tertile of 7-hole batters, the called-strike rate on borderline pitches exceeds the matched 3-hole rate by ≥2pp.

**Operational definition:** Stratify Round 1's H3 GAM by batter's 2025 chase-rate tertile (low/mid/high). The FanSided mechanism is most plausible within the low-chase tertile (these are the "perceived-elite-pitch-recognition" hitters). Test for an interaction effect there.

**Outcome interpretation:** If the interaction is null too, the FanSided mechanism is dead at every level we can test. If it survives in the low-chase tertile, that's a precise positive finding the article can name.

---

## 2. Editorial branches (Round 2)

Round 2 outcomes map to article framings as follows:

| Round 2 finding | Article frames as | Working title |
|---|---|---|
| **H4 + H5 both positive** (some umpires AND some hitters identified) | Leaderboard piece | *"League-Wide There's No 7-Hole Tax. Here Are the Specific Umpires and Hitters Where It Lives."* |
| **H4 positive only** (umpires identified, no specific hitters) | Umpire-leaderboard piece | *"League-Wide There's No 7-Hole Tax. These Umpires Are the Exception."* |
| **H5 positive only** (hitters identified, no specific umpires) | Hitter-victim piece | *"The 7-Hole Tax Doesn't Exist. But These Hitters Are Personally Getting Screwed."* |
| **H4 + H5 both null** but H6 positive (catcher mechanism) | Methodology piece | *"The 7-Hole Tax Is a Catcher Story, Not an Umpire Story."* |
| **All null** (H4 + H5 + H6 + H7 all return zero) | Comprehensive debunk | *"We Tested the 7-Hole Tax Six Different Ways. It Isn't There."* |

Even the "all null" branch is an A-tier piece — the comprehensive-audit framing is methodology porn the analytical community values. (Reference: Coaching Gap piece succeeded on a 16-of-17-null structure.)

---

## 3. Methodology

### 3.1 Agent A (Claude) — Bayesian / interpretability-first

**Mandate (extension of Round 1):** Hierarchical Bayesian models with random slopes. Per-umpire and per-batter posterior distributions on the lineup-spot effect, with proper shrinkage handling small-sample umpires.

Required methods:
- **H4 — Per-umpire random-slope GLM.** Extend Round 1's H3 GAM with `(0 + lineup_spot_7 | umpire)` random slope. The posterior distribution of each umpire's slope IS the per-umpire lineup-spot bias. Filter umpires with ≥50 borderline calls; report posterior median + 95% CrI for each. Compute "probability of direction" P(slope < 0) as a one-sided summary. Identify umpires whose 95% CrI fully excludes zero AND whose median magnitude ≥ 2pp.
- **H5 — Per-hitter posterior predictive residuals.** For each qualifying hitter (≥30 borderline take decisions in spots 7-9), compute their posterior-predictive expected called-strike rate from the Round-1 H3 GAM and compare to their actual rate. Bootstrap or simulate from the posterior to get a personal CI. Identify hitters whose 95% CI on their personal residual excludes zero.
- **H6 — Bayesian distributional comparison.** For catcher-initiated challenges, model `edge_distance ~ batter_lineup_spot + ...` with the same hierarchical structure. Test the spot-7-vs-3 fixed effect on edge-distance — i.e., are catchers picking systematically different pitches.
- **H7 — Interaction term.** Add `lineup_spot × chase_rate_tertile` interaction to Round 1's H3 GAM. Posterior of the interaction effect is the test.

Forbidden: black-box ML methods. SHAP, gradient boosting as primary tools (Codex's lane). LightGBM, XGBoost.

### 3.2 Agent B (Codex) — ML-engineering / model-driven

**Mandate (extension of Round 1):** Per-umpire and per-hitter model-based effect attribution via cross-validated counterfactuals.

Required methods:
- **H4 — Per-umpire counterfactual at scale.** Train a single LightGBM called-pitch model (same as Round 1, with explicit umpire×lineup_spot interaction features). For each umpire's borderline pitches, compute counterfactual called-strike-rate delta (spot 7 vs spot 3) using paired prediction. Bootstrap per-umpire CIs (N≥200). Apply BH-FDR across all umpires with ≥50 calls. Identify umpires whose corrected q-value < 0.10 AND magnitude ≥ 2pp.
- **H5 — Per-hitter expected-vs-actual deviation.** For each qualifying hitter, predict their borderline pitches' called-strike rate from a model that does NOT see batter ID. Compare to actual. Bootstrap CIs on the deviation. Identify hitters whose 95% bootstrap CI on the deviation excludes zero.
- **H6 — Energy-distance probe by challenger × spot.** Subset to catcher-initiated challenges. Joint distribution test (energy distance, multivariate KS) of `(edge_distance_in, in_zone, count, pitcher_fame_quartile)` between 7-hole and 3-hole AB. Permutation p-values.
- **H7 — Stratified counterfactual.** Re-run Round 1's counterfactual within each chase-rate tertile of 7-hole batters; compare to matched 3-hole stratum. Bootstrap CIs. Test for interaction.

Forbidden: hierarchical Bayesian inference, posterior credible intervals as primary method (Claude's lane). PyMC, bambi.

**Why the divergence still matters:** Round 2 turns the league-aggregate question into per-actor questions. Bayesian shrinkage and ML cross-validation handle multi-comparisons differently. Convergence on "this umpire shows it, this hitter shows it" via two methods locks in any positive finding. Convergence on universal-null is the comprehensive-audit framing.

---

## 4. Round 2 Deliverables per agent

Each agent must produce in `claude-analysis-r2/` or `codex-analysis-r2/`:

1. `analyze.py` — one-command reproduction entry point
2. Module scripts: `h4_per_umpire.py`, `h5_per_hitter.py`, `h6_catcher_mechanism.py`, `h7_chase_interaction.py`
3. `REPORT.md` — 1500-2500 words, structured:
   - Executive summary: which Round 2 hypotheses landed positive, which null, what the article should be
   - H4: per-umpire results with leaderboard table (umpires whose CI excludes zero AND magnitude ≥ 2pp; rest of distribution shown as histogram)
   - H5: per-hitter results with named-hitter table (hitters whose deviation CI excludes zero; cross-referenced with chase rate / discipline)
   - H6: catcher mechanism — does pitch-selection differ
   - H7: chase-rate interaction
   - Recommended editorial branch from Section 2
4. `charts/` — at minimum:
   - `h4_per_umpire_distribution.png` — every umpire's effect estimate with CIs, sorted; highlight any flagged
   - `h5_per_hitter_residuals.png` — qualifying hitters' actual vs expected, with named flags
   - `h6_catcher_mechanism.png` — joint-distribution comparison
   - `h7_chase_tertile_effect.png` — interaction effect by chase tertile
5. `findings.json` — machine-readable:
   ```json
   {
     "h4_per_umpire": {
       "n_qualifying_umpires": NN,
       "n_flagged": NN,
       "flagged_list": [{"umpire": "...", "effect_pp": X.X, "ci_low": X.X, "ci_high": X.X, "n_calls": NNN}, ...],
       "league_distribution": {"median": X.X, "iqr_low": X.X, "iqr_high": X.X}
     },
     "h5_per_hitter": {
       "n_qualifying_hitters": NN,
       "n_flagged": NN,
       "flagged_list": [{"hitter": "...", "actual": X.X, "expected": X.X, "deviation_pp": X.X, "ci_low": X.X, "ci_high": X.X, "n_borderline": NNN, "chase_rate_2025": X.X}, ...]
     },
     "h6_catcher_mechanism": {"effect_pp": X.X, "ci_low": X.X, "ci_high": X.X, "p_value": X.XX, "interpretation": "..."},
     "h7_chase_interaction": {"low_chase_effect_pp": X.X, "ci_low": X.X, "ci_high": X.X, "interaction_p": X.XX},
     "recommended_branch": "leaderboard | umpire-only | hitter-only | catcher-mechanism | comprehensive-debunk",
     "biggest_concern": "..."
   }
   ```
6. `READY_FOR_REVIEW.md` — ≤500 words

---

## 5. Inputs (REUSE Round 1 outputs)

Round 2 reuses ALL Round 1 data substrates. Do not re-pull.

- `claude-analysis/data/all_challenges_apr15_may04.json` (or codex-analysis/data/, depending on whose substrate the agent prefers)
- `claude-analysis/data/statcast_2026_apr23_may04.parquet`
- `claude-analysis/data/batter_lineup_spot.parquet`
- `claude-analysis/data/pitcher_fame_quartile.parquet`
- `claude-analysis/data/catcher_framing_tier.parquet`
- `claude-analysis/data/game_umpire.parquet`
- `team-challenge-iq/data/all_challenges_detail.json` (Mar 26 – Apr 14)
- `abs-walk-spike/data/statcast_2026_mar27_apr22.parquet`

NEW input required (both agents build their own):
- **Batter chase-rate lookup:** prior-season (2025) chase rate per batter. Pull from FanGraphs leaderboard or compute from prior-season Statcast. ≥200 PA threshold for inclusion.

---

## 6. Sample-size discipline

Round 2 is more sensitive to sample size than Round 1 because we're slicing the data finer. Hard rules:

- **Per-umpire (H4):** umpire must have ≥50 borderline-pitch calls involving lineup spots 7-9 AND ≥50 involving spots 1-3. Drop umpires below threshold; do not pool with rest.
- **Per-hitter (H5):** batter must have ≥30 borderline-pitch take decisions in spots 7-9 this season. Drop hitters below threshold.
- **Multiple comparisons (H4):** approximately 60-80 qualifying umpires; BH-FDR across the full set; report alongside raw p-values.
- **Multiple comparisons (H5):** approximately 50-100 qualifying hitters; BH-FDR across the full set.

If sample is too thin to qualify ≥10 umpires for H4 or ≥10 hitters for H5, surface that explicitly and the article reports the sample-size reality.

---

## 7. Scope fence (Round 2)

**IN scope:**
- H4-H7 as defined
- Cross-reference flagged hitters with prior-season discipline metrics
- Soft surfacing of "umpires who don't show it" as the negative class

**OUT of scope (explicit, deferred):**
- Per-team breakdown
- Pre-ABS-era comparison
- Other potential biases (handedness, batting average, prior K%) — not part of the FanSided claim
- Round 3 follow-ups (per-game, per-park, per-stadium)

---

## 8. Timeline (Round 2)

| Hour | Focus | Gate |
|------|-------|------|
| 0 | Brief approved, agents launched | Round 1 data reused, chase-rate lookup built |
| 0–6 | Both agents run H4-H7 | Both `READY_FOR_REVIEW.md` exist |
| 6–10 | Cross-review | Both review files in `reviews/` (with `r2-` prefix) |
| 10–14 | Comparison memo | `reviews/COMPARISON_MEMO_R2.md` |
| 14–24 | Article rewrite using Round 1+Round 2 evidence | Updated DRAFT.md |
| 24–48 | Final sign-off + Astro wrap + chart components | Publishable |

---

## 9. How to Run

```bash
cd /Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax

# Agents read this brief PLUS Round 1 outputs. They do NOT re-pull data.
# Each agent works in claude-analysis-r2/ or codex-analysis-r2/.

# Launch both:
# - Claude via Agent tool (background)
# - Codex via direct CLI to launch_codex.sh equivalent
```

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Per-umpire sample too thin for credible identification | Hierarchical shrinkage absorbs noise; if no umpire qualifies, that's the universal-null framing — also flagship-worthy |
| Flagged umpires are false positives | Multi-comparisons correction (BH-FDR); convergence requirement across both agents before naming |
| Per-hitter analysis identifies stars by accident (high-volume batters get longer CIs) | Wilson / posterior CIs naturally penalize low n; named hitters must clear both magnitude and CI gates |
| Chase-rate lookup data quality | If FanGraphs 403's, fall back to in-season-to-date; document the choice |
| Round 2 introduces denominator inconsistencies vs Round 1 | Reuse Round 1 substrates verbatim; new analyses build on them, don't recompute the H3 corpus |
| Agents converge on same wrong methodology | Mandates remain divergent — Bayesian random slopes vs ML per-umpire counterfactual |

# Round 3 Brief — ABS Walk Spike (A-tier elevation)

**Status:** Round 3 — agents launching
**Target deliverable:** Final input pass before article drafting. Round 3 must convert R2's "A-tier with right framing" into "A-tier unconditionally."
**Decision criterion:** Round 3 produces (1) a defensible single H3 magnitude with widened-but-honest CI, (2) a named pitcher adaptation leaderboard that survives bootstrap stability and cross-method agreement, (3) a stuff-vs-command interaction with archetype leaderboards. All three are required for the A-tier piece.

---

## 0. What Round 2 settled (DO NOT re-litigate)

These are locked from Round 2 (`reviews/COMPARISON_MEMO_R2.md`):

- **H1:** Walk rate +0.66-0.68pp YoY through May 12 (down from R1's +0.82pp). Both methods agree.
- **Within-2026 fading:** W1-3 → W5-7 = −0.86pp; P(regressed)=89%.
- **Zone shape:** Top edge −9pp, bottom edge +3pp. Consistent with R1; durable across both methods, both rounds.
- **Aggregate diagnostic:** Mean predicted CS on 2026 takes under 2025 classifier = 0.334 (vs 2025 empirical 0.327, 2026 empirical 0.325). Both pipelines independently arrive at this.
- **H5 mechanism resolved:** Top edge dropped 6-7pp MORE strikes at 0-0 than at 2-strike counts. DiD credible at −6.76pp (Bayesian). SHAP confirms count-dependence.
- **H2 decomposition:** +0.41pp traffic-driven + 0.27pp conditional. Pitchers losing the count battle earlier.
- **Editorial branch:** `adaptation`. Both agents independently agreed.

What's open and Round 3 must answer:
- Defensible H3 attribution magnitude (R2 had +35.3% [+34.6, +36.0] Codex vs −64.6% [−80.6, −49.4] Claude — same training data, two PA-replay mechanics, opposite signs)
- Which specific pitchers have credibly adapted (R2 found heterogeneity but no audit-ready named list)
- Whether the new zone differentially helps stuff-archetype pitchers and hurts command-archetype pitchers (FanGraphs gestured at this anecdotally; nobody has quantified it)

---

## 1. The Round-3 sub-hypotheses

### R3-H1 — Settle the H3 magnitude via triangulation

**Hypothesis:** The R2 H3 magnitude is bounded between Codex's +35.3% and Claude's −64.6%. A defensible point estimate with honest CI lives somewhere in this range.

**Operational definition:** Each agent implements a THIRD counterfactual method, independent of their own R2 approach. Then compute a triangulated estimate across all three of each agent's methods. The publishable headline is the cross-agent median with a CI that spans the union of the three honest intervals.

**Required methods (per agent):**

**Both agents must produce three CF estimates:**

1. **Method A (their R2 method, faithfully reproduced):**
   - Claude: per-take Bernoulli PA replay (faithful R2 reproduction, but with the `observed-outcome backstop` issue at `h3_counterfactual.py:313-316` addressed — switch to a continuation model for unresolved tails)
   - Codex: expectation-propagation PA replay (faithful R2 reproduction, but with **game-level bootstrap** N≥200 instead of the 10-seed cross-fit SD that gave the artifact-narrow CI)

2. **Method B (a deliberately different mechanism):**
   - Claude: empirical-lookup CF — for each 2026 taken pitch at (plate_x, plate_z, count), look up the empirical 2025 CS rate in the same cell + count tier. No Bayesian model. Re-replay PAs. (Claude implementation, not Codex.)
   - Codex: per-pitch SHAP-based attribution — what fraction of each pitch's predicted CS probability change is attributable to its location vs other features? Aggregate via per-pitch decomposition.

3. **Method C (the triangulation method, shared design but independent implementations):**
   - Both: **Bootstrap-of-bootstrap CI** on aggregate walk-rate attribution. Resample games (not pitches, not rows). For each game-bootstrap iteration: refit the classifier on 2025 same-window data and replay 2026 PAs. N≥100 outer × N≥10 inner = ≥1,000 total iterations. This gives a CI that includes both model uncertainty AND PA-level sequencing uncertainty.

**Output per agent:**
- Three point estimates (Methods A, B, C) with their respective CIs
- Single "triangulated" magnitude (median of the three) with the widest of the three CIs as the editorial CI
- Per-count and per-edge breakdowns under Method C only (the most rigorous)

**Editorial output:** Cross-agent triangulated estimate + ranges. The article will publish this single number with a clean CI.

### R3-H2 — Named pitcher adaptation leaderboard

**Hypothesis:** Some specific pitchers have shifted their location distribution / pitch mix / zone rate in a way that survives bootstrap stability AND cross-method agreement.

**Operational definition:** Build a top-N pitcher leaderboard. A pitcher is "named" only if:
- ≥200 pitches in 2026 Mar 27 – May 12 window (sample threshold)
- Shift magnitude meets pre-registered cutoff: |Δ zone rate| ≥ 15pp OR |Δ top-share| ≥ 15pp OR pitch-mix Jensen-Shannon divergence ≥ 0.05
- **Bootstrap stability:** appears in top-15 of ≥80% of bootstrap iterations (game-level resample, N≥200)
- **Cross-method agreement:** flagged by BOTH Bayesian and ML pipeline

**Required computation per agent:**

- **Claude:** Bayesian Beta-Binomial per-pitcher × per-week for zone rate and top-share, with bootstrap stability over games. Per-pitcher Dirichlet for pitch-mix distribution week-over-week.
- **Codex:** LightGBM per-pitcher feature-importance ensemble (10 seeds × game-bootstrap N≥100), with per-pitcher SHAP for shift attribution.

**Output:** A named, bootstrap-stable, cross-method-agreed leaderboard of pitchers who have visibly adapted. Could be 5 names, could be 25 — whatever clears the filters. If 0 names clear, that's also a publishable finding ("everyone shifted incrementally; nobody clearly adapted at scale").

### R3-H3 — Stuff vs command archetype × zone-change interaction

**Hypothesis:** Pitchers with high stuff+ and low command+ benefit from the 2026 zone change (relative to their 2025 performance). Pitchers with high command+ and low stuff+ are hurt.

**Operational definition:**
- For each pitcher with ≥200 pitches in 2026: compute their 2025 stuff+ and command+ percentile.
  - **Source priority:** FanGraphs leaderboard if available; else proxy from prior-season Statcast (stuff+ proxy: pitch arsenal whiff rate quartile; command+ proxy: 2025 BB rate quartile + zone rate)
- Compute per-pitcher 2026 walk-rate change vs their 2025 baseline (with appropriate sample-size threshold, e.g., 2025 ≥40 IP).
- Test: Is the per-pitcher walk-rate change correlated with (stuff+ percentile − command+ percentile)?
- Spearman rank correlation + Bayesian (Claude) and ML residual analysis (Codex) for the magnitude

**Output:**
- Headline: correlation coefficient + significance + magnitude of the archetype interaction
- Two leaderboards (each filtered like R3-H2):
  - "Command pitchers most hurt by the new zone" — top stuff−command differential with most-negative walk-rate change
  - "Stuff pitchers most helped" — bottom stuff−command differential with most-positive walk-rate change

If the archetype interaction is null or directional-only, that's still publishable: it kills a popular anecdotal narrative.

---

## 2. Editorial branches

Round 3 outcomes map to article framings:

| R3 finding pattern | Article frames as | Lead positioning |
|---|---|---|
| **All three positive:** Magnitude settled (~30-40%), named adapter leaderboard, archetype interaction credible | Mechanism + names | "Three weeks later, we know how the spike works AND who's adapting. Here's the named list and the archetype that lost the new zone." |
| **R3-H1 + R3-H2 positive, R3-H3 null** | Mechanism + names | Same as above, archetype claim demoted to "we tested the stuff-vs-command narrative; it doesn't survive." |
| **R3-H1 settled, R3-H2 sparse (0-5 names), R3-H3 positive** | Mechanism + archetype | "Pitcher adaptation is heterogeneous and individually noisy — but the archetype that should adapt has, and here's why." |
| **R3-H1 settled, both leaderboards sparse** | Mechanism-only piece | "We know how the spike works. Who's adapting is still hidden in the noise. Here's why and what to look for next." |

All four are publishable. The first two are A-tier ceiling. The third is A− with a name angle. The fourth is the floor we accept.

---

## 3. Data

### 3.1 Reuse (do NOT re-pull)

- All R1 substrates (R1 data folder)
- All R2 substrates (each agent's `*-analysis-r2/data/` folder)
- 2025 full-season Statcast (`research/count-distribution-abs/data/statcast_2025_full.parquet`)

### 3.2 New input — pitcher stuff+ / command+ for 2025

Each agent builds their own copy:

- **Preferred source:** FanGraphs leaderboard via API or scrape. If 403, fall back to manual data export already on disk if available.
- **Fallback (proxy):** Derive from prior-season Statcast:
  - Stuff+ proxy: weighted average of pitch-arsenal whiff rates, percentile-rank
  - Command+ proxy: walk rate percentile (lower = better command) + zone rate percentile
- **Sample threshold:** ≥40 IP in 2025 to qualify

Save as `*-analysis-r3/data/pitcher_archetype_2025.parquet` with columns: `pitcher_id`, `name`, `stuff_pct`, `command_pct`, `stuff_minus_command`, `data_source` (fangraphs | proxy).

### 3.3 Game-level bootstrap discipline

For Round 3, all bootstrap procedures resample **games** (not pitches, not rows). This is the critical fix from R2's CI artifact problem. Round 3 implementations must:
- Resample game_pk with replacement
- Per bootstrap iteration: retrain the relevant model + compute the per-iteration statistic
- N ≥ 200 outer bootstrap iterations minimum
- For the triangulation method: N ≥ 100 outer × N ≥ 10 inner

---

## 4. Methodology

### 4.1 Agent A (Claude) — Bayesian / interpretability-first

Required methods:

1. **R3-H1 Method A (faithful R2 reproduction with fix):** Per-take Bernoulli PA replay, BUT replace the observed-outcome backstop at `claude-analysis-r2/h3_counterfactual.py:313-316` with a continuation model: for unresolved PAs, sample remaining pitches from the empirical 2026 distribution of pitches in that count-state.

2. **R3-H1 Method B:** Empirical-lookup CF. For each 2026 taken pitch, look up empirical 2025 CS rate in (plate_x, plate_z, count_tier) cell using kNN smoothing (k=20). Replay PAs. No Bayesian model.

3. **R3-H1 Method C:** Bootstrap-of-bootstrap. Outer: resample game_pk with replacement, N≥100. Inner: refit Bayesian zone classifier on resampled 2025 same-window pitches, N≥10. Aggregate to walk-rate attribution. Report median + 95% percentile CI.

4. **R3-H2:** Bayesian Beta-Binomial per-pitcher × per-week for zone rate and top-share. Game-level bootstrap stability filter: pitcher appears in top-15 of ≥80% of N≥200 game-bootstrap iterations.

5. **R3-H3:** Bayesian linear/ordinal model: `walk_rate_change ~ stuff_minus_command + pitcher_random_effect`. Posterior of slope. Bootstrap stability for leaderboard membership (same filter as R3-H2).

**Forbidden:** LightGBM, XGBoost, SHAP as primary methods (Codex's lane).

### 4.2 Agent B (Codex) — ML-engineering, model-driven

Required methods:

1. **R3-H1 Method A (faithful R2 reproduction with proper CI):** Expectation-propagation PA replay (R2 design), but replace the 10-seed cross-fit SD with **game-level bootstrap** N≥200. Each iteration: resample game_pk, refit GBM, replay 2026 PAs.

2. **R3-H1 Method B:** Per-pitch SHAP attribution decomposition. For each 2026 taken pitch's predicted CS probability under the 2025 classifier, what fraction of the change is attributable to its plate_x, plate_z, count, etc.? Aggregate per-pitch contributions.

3. **R3-H1 Method C:** Bootstrap-of-bootstrap. Outer: game-bootstrap N≥100. Inner: 10-seed ensemble refit, N≥10. Same structure as Claude's Method C; independent implementation.

4. **R3-H2:** LightGBM per-pitcher feature-importance ensemble (10 seeds × game-bootstrap N≥100). SHAP per-pitcher for shift attribution. Bootstrap stability filter.

5. **R3-H3:** Train a GBM: `walk_rate_change ~ stuff_pct + command_pct + interaction + other_2025_features`. SHAP for interaction effect. Permutation importance vs permuted-label baseline. Bootstrap leaderboard.

**Critical methodology constraint (FINAL TIME):** Codex bootstrap CIs in R1 (7-hole tax), R2 (7-hole tax), and R2 (ABS walk spike) have all suffered from the fixed-model-bootstrap artifact (narrow CIs that don't reflect model uncertainty). Round 3 fix is non-negotiable:
- ALL CIs use game-level bootstrap with refit at each iteration
- 10-seed cross-fit SD is NOT a substitute
- Per-row paired bootstrap is NOT a substitute
- Calibration curve required on every GBM
- Cross-review will block on this

**Forbidden:** PyMC, bambi, hierarchical Bayesian (Claude's lane).

---

## 5. Deliverables per agent

Each agent in `claude-analysis-r3/` or `codex-analysis-r3/`:

1. `analyze.py` — entry point
2. Module scripts: `data_prep_r3.py`, `archetype_build.py`, `h1_triangulation.py`, `h2_adapter_leaderboard.py`, `h3_archetype_interaction.py`
3. `REPORT.md` — 1500-2500 words covering all three R3 hypotheses + recommended editorial branch
4. `charts/` PNGs:
   - `h1_triangulated_attribution.png` — three methods side-by-side with CIs, with R2 results for comparison
   - `h2_adapter_leaderboard.png` — named pitchers with shift magnitudes and stability scores
   - `h3_archetype_scatter.png` — per-pitcher walk-rate change vs stuff−command differential
   - `h3_archetype_leaderboards.png` — top-N most-hurt-command + top-N most-helped-stuff
   - `model_diagnostics/` (Codex): calibration curves for every GBM
   - `diagnostics/` (Claude): R-hat, ESS, traces
5. `findings.json` per the standard format
6. `READY_FOR_REVIEW.md` ≤500 words

---

## 6. Sample-size discipline

- R3-H1 game-bootstrap: N ≥ 200 outer iterations
- R3-H1 Method C bootstrap-of-bootstrap: N ≥ 100 outer × N ≥ 10 inner
- R3-H2 per-pitcher: ≥200 pitches in window
- R3-H3 per-pitcher: ≥40 2025 IP AND ≥200 2026 pitches
- Bootstrap stability filter: appears in top-15 of ≥80% of iterations

---

## 7. Scope fence

**IN scope:**
- R3-H1, R3-H2, R3-H3 as defined
- Cross-method agreement filter for naming
- Bootstrap-of-bootstrap CIs throughout

**OUT of scope (still deferred to Round 4 if needed):**
- Per-umpire breakdown
- Per-team breakdown
- Catcher framing × zone change interaction
- Pre-ABS-era comparison

---

## 8. Timeline

| Hour | Focus | Gate |
|------|-------|------|
| 0 | Brief approved, agents launched | Round 1+2 data reused, archetype lookups built |
| 0-6 | Both agents run R3-H1/H2/H3 | Both `READY_FOR_REVIEW.md` exist |
| 6-10 | Cross-review | Both reviews in `reviews/` (r3- prefix) |
| 10-14 | Comparison memo + A-tier confirmation | `reviews/COMPARISON_MEMO_R3.md` |
| 14-36 | Article draft via `calledthird-editorial` |
| 36-48 | Astro + charts + OG + ship | Live |

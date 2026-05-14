# The 7-Hole Tax — Research Brief

**Status:** Round 1 — agents launching
**Target deliverable:** Flagship CalledThird article, 24-48h Round 1 ship
**News hook expiration:** ~5-7 days (FanSided + The Ringer are the immediate competition)
**Principal risk:** Selection bias is the obvious mechanism — if our analysis can't separate "umpires call 7-hole batters differently" from "7-hole batters challenge worse pitches," we don't have a story. But that fork itself is publishable.

---

## 0. Prior CalledThird coverage

This sits squarely in our **ABS & Umpires** pillar and connects to four published pieces:

- **The Four Kinds of Zone** (Apr 18) — established that umpires have systematically different zones; reader has the framing context
- **CB Bucknor By The Numbers** (Apr 18) — set the "specific umpire's zone" template; this is the inverse: specific *batter type's* zone
- **Catchers Are Better Challengers** (Apr 22) — the prior win on challenge-outcome analysis; same data spine
- **ABS Took the High Strike** (Apr 23) — established our methodology for zone-shape analysis under controls

Reuse the analytical pipeline from `team-challenge-iq` (challenge-level data) and `abs-walk-spike` (called-pitch zone analysis).

---

## 1. The Question

**Do MLB umpires call a meaningfully different strike zone for batters in the 7-hole than for batters in the 3-hole — and if so, how much of that gap survives controls for pitch location, count, pitcher, and catcher?**

Triggered by FanSided ("MLB's ABS challenge data just proved what hitters have known for years") and The Ringer ("Accountability Culture Is Dead. ABS Is the Exception.") reporting that 7-hole batters win their ABS challenges only **30.2%** of the time, the worst rate of any lineup slot. The implied mechanism: umpires unconsciously give borderline-call benefit-of-the-doubt to hitters they perceive as having elite pitch recognition.

**Both outlets ran proportions, not controlled analyses.** The selection-effect alternative — that 7-hole batters challenge harder pitches — was not addressed. That's our opening.

---

## 2. Hypotheses

**H1 (raw replication):** The 7-hole batter ABS-challenge overturn rate is at least 10pp below the league-wide batter overturn rate (≈45%). Tests whether the FanSided/Ringer number replicates in our 970-challenge corpus extended to current date.

**H2 (controlled bias on challenges):** After controlling for `edge_distance_in` (proxy for "how close is the pitch"), count state, pitcher fame quartile, and catcher framing tier, the marginal effect of batting-order position 7 on overturn probability remains at least 5pp below position 3.

**H3 (controlled bias on all called pitches):** On taken pitches within ±0.3 ft of the rulebook edge — the borderline zone where umpires actually have judgment latitude — the called-strike rate on 7-hole batters exceeds the called-strike rate on 3-hole batters by at least 2pp, after controlling for plate location (2D spline), count, pitcher, catcher, and umpire.

H3 is the deeper question. H1 and H2 only address the challenge-outcome lens; H3 tests whether the bias exists in *every* called pitch, not just the ones that get challenged.

---

## 3. Editorial branches (replaces "kill criteria")

All four branches publish. The data picks the article framing.

| Branch | Trigger | Article framing |
|--------|---------|-----------------|
| **B1: Bias confirmed (clean)** | H1 + H2 + H3 all hold | "The 7-Hole Tax Is Real — Even After Controlling For Everything" |
| **B2: Selection artifact** | H1 holds but H2 + H3 fail | "The 7-Hole Tax Is a Mirage. Here's What's Actually Happening." |
| **B3: Real but smaller** | H1 holds, H2 holds at <full magnitude, H3 holds at <full magnitude | "The 7-Hole Tax Is Real — But Half as Big as Reported" |
| **B4: Doesn't replicate** | H1 fails (overturn rate gap <10pp in our data) | "We Tried to Replicate the 7-Hole Tax. The Number Isn't There." |

**Stretch addendum (B5, conditional):** If the bias has a specific signature — concentrated at top of the zone, on breaking pitches, with framing catchers, with star pitchers — add a "where the bias lives" subsection. Bonus, not required.

The ONLY data outcome that doesn't publish is "uninterpretable noise across the board." With our existing zone-analysis infrastructure, that's vanishingly unlikely.

---

## 4. Round structure

**Round 1 only** for this project. Single-round, news-anchored.

If Round 1 finds clean signal (B1 or B3), a Round 2 follow-up could ask:
- Is the bias asymmetric by handedness?
- Does it persist after first-pitch outcomes (i.e., is this prior-belief updating in real time)?
- Per-umpire breakdown — which umpires drive the league-aggregate effect?

But Round 2 is conditional on Round 1 finding signal worth deepening. Brief stays Round 1 only.

---

## 5. Data

### 5.1 Primary corpus

**Challenge-level data (for H1, H2):**
- `data/all_challenges_detail.json` — 970 challenges Mar 26–Apr 14 from `team-challenge-iq`. **REUSE; do not re-pull this window.**
- `data/all_challenges_2026_apr15_may03.json` — challenges Apr 15–May 3, **fresh pull required.** Use the `team-challenge-iq` fetcher script as template.

**Called-pitch data (for H3):**
- `research/abs-walk-spike/data/statcast_2026_mar27_apr22.parquet` — 106,770 rows, full Statcast schema. Reuse.
- `data/statcast_2026_apr23_may03.parquet` — fresh pull to extend through current date.

### 5.2 Critical derived input — batting order

**Statcast does NOT expose batting order directly.** Both agents must derive it.

Recommended path: MLB Stats API boxscore endpoint
```
https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
→ liveData.boxscore.teams.{home,away}.battingOrder
```

This returns an ordered list of player IDs per team per game — the starting lineup spots. Build:
- `data/batter_lineup_spot.parquet` — columns: `game_pk`, `team`, `batter_id`, `lineup_spot` (1-9), `is_pinch_hitter` (bool)

For pinch hitters / late-game subs, assign the `lineup_spot` of the position they replaced. Flag `is_pinch_hitter=True` so the analysis can include or exclude as a robustness check.

**Effort:** ~30 min of script work. Both agents can build their own copy or share. Recommend: `scripts/build_lineup_spot.py` that both can call.

### 5.3 Known data-quality issues

- `team-challenge-iq` data goes through Apr 14 only. The Apr 15–May 3 window must be pulled fresh — confirm same schema before merging.
- Lineup spots 8 and 9 in NL parks (when DH used) are real lineup slots; in 2026 every team uses DH so this is uniform.
- "7-hole" = `lineup_spot == 7`. Document any edge cases (DH games, openers leading off as a fielder, etc.).
- Pinch hitters complicate the "batting order = batter type" inference — some pinch hitters are stars subbed into the 7-hole. Flag, then run robustness with PHs excluded.

---

## 6. Features / Variables

| # | Name | Definition | Source |
|---|------|------------|--------|
| 1 | `lineup_spot` | 1-9 batting order position | MLB Stats API boxscore |
| 2 | `overturned` | 1 if ABS overturned the call | Challenge data |
| 3 | `edge_distance_in` | Distance from rulebook zone edge, inches | Challenge data |
| 4 | `in_zone` | Was the pitch in the rulebook zone | Challenge data |
| 5 | `plate_x`, `plate_z` | Pitch location at plate, ft | Statcast |
| 6 | `is_called_strike` | 1 if `description == 'called_strike'` (only for taken pitches) | Statcast |
| 7 | `count_state` | (balls, strikes), 12 distinct counts | Statcast |
| 8 | `pitcher_fame_quartile` | Pitcher's prior-season K-BB% quartile | Derived |
| 9 | `catcher_framing_tier` | Catcher's prior-season framing runs tier (top/mid/bottom) | Derived from prior framing data |
| 10 | `umpire` | Home plate umpire name | Statcast |
| 11 | `is_pinch_hitter` | True if batter is not in the starting lineup spot | Derived |

For pitcher/catcher tiers: use 2025 final-season values where available; otherwise current-season-to-date values.

---

## 7. Methodology

### 7.1 Agent A (Claude) — interpretability-first, Bayesian

**Mandate:** Hierarchical Bayesian models with random effects for umpire/pitcher/catcher, fixed effects for lineup spot, smooth splines for location. Every claim ladders down to a posterior distribution a reader can interpret.

Required methods:
- **Wilson CIs on raw overturn rate by lineup spot** (1-9), with multiple-comparisons correction. This is the H1 replication.
- **Hierarchical Bayesian logistic GLM** for H2: `overturned ~ lineup_spot + edge_distance_in + count_state + (1|pitcher) + (1|catcher) + (1|umpire)`. Use `pymc` or `bambi`. Report posterior of lineup_spot fixed effects with 95% credible intervals.
- **Hierarchical Bayesian GAM** for H3: `is_called_strike ~ lineup_spot + s(plate_x, plate_z) + count_state + (1|pitcher) + (1|catcher) + (1|umpire)`, restricted to taken pitches within ±0.3 ft of the rulebook edge. Report marginal posterior effect of `lineup_spot=7` vs `lineup_spot=3`.
- **Stratified replication** of H3 by handedness, count quadrant (hitter's vs pitcher's count), and pitch type (FF/breaking/offspeed) as robustness checks.

Forbidden: black-box ML methods. Everything reduces to a posterior + a chart a reader can replicate.

### 7.2 Agent B (Codex) — ML-engineering, model-driven

**Mandate:** Gradient-boosted classifier with explicit attribution; counterfactual via lineup-spot permutation.

Required methods:
- **Two GBM classifiers (LightGBM):** one predicting `overturned` (challenge data), one predicting `is_called_strike` (taken pitches). Features: location, count, pitcher fame quartile, catcher framing tier, umpire (one-hot or target-encoded), `lineup_spot` (one-hot 1-9).
- **SHAP analysis** to localize the marginal contribution of `lineup_spot=7` after partial-out of location and pitcher/catcher quality. Report mean |SHAP| for each lineup spot.
- **Counterfactual permutation test:** for each taken pitch on a 7-hole batter, predict `is_called_strike` under the actual `lineup_spot=7` AND under a counterfactual `lineup_spot=3`. Report mean predicted-probability delta. This is the "if everything else were identical, how would the call differ" attribution.
- **Distribution-shift check:** is the joint distribution of (location, count, pitcher quality) the same for 7-hole vs 3-hole challenges? KS or energy-distance test by feature. This is the **selection-effect probe** — directly answers H2 vs B2.

Forbidden: hand-coded Bayesian inference, GAM splines as primary method (Claude's lane). Use models, not posteriors.

**Why the divergence matters:** Claude produces interpretable posterior credible intervals — the answer in the form a statistician trusts. Codex produces a counterfactual prediction — the answer in the form a machine-learning engineer trusts. When both methods give the same lineup_spot effect direction and rough magnitude, the finding is locked. When they diverge, the comparison memo IS the story (e.g., "Bayesian says +3pp, ML counterfactual says +0.5pp, here's why the discrepancy matters").

---

## 8. Round 1 Deliverables per agent

Each agent must produce in `claude-analysis/` or `codex-analysis/`:

1. `analyze.py` — one-command reproduction entry point
2. Module scripts per methodology component
3. `REPORT.md` — 1500-2500 words covering H1, H2, H3 + which editorial branch is recommended (B1/B2/B3/B4 + optional B5)
4. `charts/` — at minimum:
   - **Overturn rate by lineup spot** (1-9) with Wilson CIs — this is the H1 replication chart
   - **Marginal lineup-spot effect on overturn probability** after controls — H2 forest plot or similar
   - **Called-strike-rate-on-borderline-pitches by lineup spot** — H3 visualization
   - **Selection-effect chart**: distribution of `edge_distance_in` (or location features) by lineup spot, to show whether 7-hole hitters actually challenge harder pitches
5. `findings.json` — machine-readable summary including:
   - `h1_overturn_rate_by_spot`: array of 9 numbers + Wilson CIs + sample sizes
   - `h2_lineup_effect_post_controls`: {effect_pp, ci_low, ci_high, baseline_spot}
   - `h3_called_strike_rate_delta`: {effect_pp, ci_low, ci_high, n_borderline_pitches}
   - `selection_effect_signal`: {ks_stat_or_similar, p_value, interpretation}
   - `recommended_branch`: "B1" | "B2" | "B3" | "B4"
6. `READY_FOR_REVIEW.md` — ≤500-word handoff: explicit answer to H1/H2/H3, recommended branch, biggest concern with own analysis

---

## 9. Timeline (Round 1)

| Hour | Focus | Gate |
|------|-------|------|
| 0 (now) | Brief approved, agents launched | Both `data/` requirements documented |
| 0–4 | Both agents extend data through May 3 (challenges + statcast), build lineup-spot lookup | Lineup-spot table exists |
| 4–10 | Both agents complete initial analysis | Both `READY_FOR_REVIEW.md` exist |
| 10–14 | Cross-review (each agent reviews the other) | Both review files in `reviews/` |
| 14–18 | Comparison memo (conversational, in main session) | `reviews/COMPARISON_MEMO.md` written |
| 18–48 | Article drafting via calledthird-editorial skill | Draft ready for human review |

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Sample too thin per lineup spot for stable overturn-rate CIs | Wilson CIs are honest about width; if any spot has n<30 challenges, flag publicly and pool with adjacent spots for the controlled analysis |
| Lineup-spot derivation introduces errors (pinch hitters, openers) | Build robustness check excluding pinch hitters; report effect sizes both ways |
| Selection effect explains 100% of the gap → null result on the bias claim | This IS branch B2 and is publishable as a debunk piece |
| Pitcher quality and lineup spot are entangled (better pitchers face better hitters who hit higher in the order) | Hierarchical pooling on pitcher random effects; stratified analysis by pitcher quartile |
| Two agents converge by accident on same wrong methodology | Mandates are deliberately divergent — Bayesian random effects vs ML SHAP/counterfactual |
| Article gets scooped by FanSided/Ringer doing their own controlled analysis | We're 24-48h to draft. They're not publishing controlled analyses. Speed advantage real. |

---

## 11. Scope fence

**IN scope:**
- League-aggregate test of H1, H2, H3
- Selection-effect probe as the central methodological move
- Stratified robustness checks (handedness, count, pitch type)

**OUT of scope (explicit):**
- Per-umpire breakdown (Round 2 if signal warrants)
- Per-team breakdown (one round at a time)
- Historical comparison to pre-ABS era (2025 data has no ABS challenge outcomes — incomparable)
- Any claim about *why* umpires would have this bias beyond reporting effect size and significance

---

## 12. How to Run

```bash
cd /Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax

# Step 0: build shared inputs (either agent can do this; do once)
# - Extend challenge data through May 3
# - Extend Statcast data through May 3
# - Build batter_lineup_spot table from MLB Stats API

# Step 1: launch both agents (Round 1)
bash /Users/haohu/.claude/skills/dual-agent-research/scripts/launch_codex.sh \
  /Users/haohu/Documents/GitHub/calledthird/research/seven-hole-tax

# Step 2 (after both READY_FOR_REVIEW.md exist): cross-review
# Step 3 (after both reviews exist): comparison memo
# Step 4: draft article via calledthird-editorial skill
```

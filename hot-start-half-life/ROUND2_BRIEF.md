# Hot-Start Half-Life — Round 2 Brief

**Status:** Draft (pre-launch), 2026-04-25
**Target deliverable:** Flagship article — *"The Sleeper Signals: Who's Actually Real in April"*
**Inputs from Round 1:** Cross-review surfaced four blocking methodology defects per agent. This round fixes them AND extends the analysis from 5 named hot starters to ~250-300 hitters and ~80 relievers.
**Principal risk:** With proper methodology, the universe scan finds zero "sleeper signal" hitters that aren't already in mainstream coverage — making this a noise-floor confirmation piece rather than a discovery piece. Null fallback specified in §6.

**Editorial context:** Round 1 confirmed Pages = NOISE and Trout/Rice regress to ~.370 wOBA, but those are the names already saturating baseball media. The sleeper-signal angle is what differentiates CalledThird from "ESPN's hot-starter tracker." The article needs to surface 2-5 names readers haven't heard yet AND 2-5 names readers have over-heard.

---

## 1. The Question

**Primary:** Across every 2026 hitter (≥ 50 PA) and reliever (≥ 25 BF) through 2026-04-25, which players have *durable* hot starts (component-level evidence persists in 2022-2025 backtests), which are *fake hot* (over-performing their underlying contact quality / discipline), and which are *fake cold* (under-performing such that bounce-back is likely)?

**Secondary:**
- Of the top-5 sleeper signals each method surfaces, how many are NOT in the published top-20 leaderboards (ESPN, FanGraphs, MLB.com)? "Sleeper" requires media obscurity.
- Among 2026 relievers, which have the largest gap between current K% and 2022-2025-anchored expected K%, with credible posteriors?
- Cross-validation: do the corrected methodologies still call Pages/Trout/Rice = NOISE? Does Murakami still SIGNAL with reproducible data pull?

---

## 2. Hypothesis

**H1 (sleeper signals exist):** ≥ 3 of the top-10 2026 hitters by predicted ROS-wOBA-vs-prior delta are NOT in the published April hot-starter top-20 lists (ESPN, MLB.com, Yahoo). These are CalledThird's "you missed these" calls.

**H2 (fake hots exist):** ≥ 3 of the top-10 published April hot starters have predicted ROS-wOBA-vs-prior delta < 0 (i.e., they will under-perform their preseason projection going forward, despite their April line). These are CalledThird's "everyone is wrong about" calls.

**H3 (fake colds exist):** ≥ 3 of the bottom-10 2026 hitters by April performance have positive predicted ROS-wOBA-vs-prior delta. These are CalledThird's "buy low" calls.

**H4 (reliever sleepers):** ≥ 2 relievers in 2026 with ≥ 25 BF show K%-rise posteriors that 2022-2025 priors would not have predicted, and they are NOT closers / known names.

**H5 (null fallback):** If the universe scan produces fewer than 2 sleeper-signal hitters whose evidence holds across both methodologies, the article becomes *"Why You Can't Beat the Hot-Starter Lists in April: 22 Games Just Doesn't Tell You Enough"* — a methodology-honest piece using the same data.

---

## 3. Data

### 3.1 Already cached
- 2022-2025 full-season Statcast (Round 1 pulled these)
- 2026 through Apr 24 (Round 1 pulled this)
- Per-season `batting_stats` / `pitching_stats` aggregates (or Statcast-derived fallback if FanGraphs 403s again)
- Player-ID lookups for the 5 named hot starters

### 3.2 Must extend
- **2026 Statcast through 2026-04-25** (one extra day) via `pybaseball.statcast`
- **All-2026-hitters universe** filtered to ≥ 50 PA through cutoff
- **All-2026-relievers universe** filtered to ≥ 25 BF and < 30 IP through cutoff (excludes starters with relief appearances)
- **Published-leaderboard reference list:** scrape or hardcode the top-20 hitters from at least one mainstream April hot-starter article (e.g., [ESPN April leaders](https://www.espn.com/mlb/story/_/id/48553796) or [MLB.com early-season stat leaders](https://www.mlb.com/news/mlb-early-season-stat-leaders-for-2026-season)). This is your "media coverage" comparison set for the sleeper-vs-known classification.

### 3.3 External joins
- Preseason projections — same fallback as R1 (3-year weighted mean) if FanGraphs API is unavailable
- For 2026 debuts (Murakami): MLB Stats API (`https://statsapi.mlb.com/api/v1/people/search?names={name}`) — **must be in `data_pull.py`, not manual.** Round 1's failure to do this was a reproducibility defect.

### 3.4 Known data-quality issues (from Round 1)
- Statcast `sz_top` / `sz_bot` 2026 schema drift — same caveat applies, do NOT use these in cross-season features
- FanGraphs leaderboard 403s — same Statcast-derived fallback applies
- Reliever earned runs not directly in Statcast — must derive via `delta_run_exp` accumulation per BF or via game logs

---

## 4. Methodology fixes (REQUIRED for both agents before new analysis)

These are non-negotiable because the Round 1 cross-review identified them as invalidating the Round 1 kill-gates.

### 4.1 Both agents

| Defect (R1) | Fix required (R2) |
|---|---|
| Murakami player-ID resolution was manual | Add MLB Stats API lookup to `data_pull.py`. Cache result. Murakami's MLBAM must regenerate from a clean checkout. |
| 5-named-hot-starter scope was too narrow | Universe = every 2026 hitter ≥ 50 PA (~250-300 players) and reliever ≥ 25 BF (~80 players). |
| No published-leaderboard comparison | Build a "media-coverage" reference set from at least one mainstream April hot-starter article. Sleeper = predicted ROS-wOBA-vs-prior in top-10 AND not in mainstream top-20. |

### 4.2 Claude (Agent A) — specific fixes from cross-review

| Defect (R1) | Fix required (R2) |
|---|---|
| `stabilization.py` "bootstrap CI" was within-player random PA partition (not sampling CI) — invalidates kill-gate language | Implement true player-season bootstrap with replacement at the player-season level. Cluster-bootstrap (resample players, then resample seasons within player) to handle within-player serial correlation. Re-run all stabilization findings. Update the findings.json with the corrected CIs. |
| `bayes_projections.py` projection prior reads only event-outcome fields (no EV/HardHit/Barrel/xwOBA) | Extend the prior to include EV p90, HardHit%, Barrel%, xwOBA, xwOBA-minus-wOBA gap as features. The Beta-Binomial likelihood is fine; the prior anchor needs a contact-quality-weighted blend. |
| `bayes_projections.py` Mason Miller uses HR-only proxy for ER (Codex review #6) | Replace with `delta_run_exp`-accumulated runs allowed per BF (Statcast field), or derive ER per BF from game logs. The HR-only proxy is dead. |
| "Hierarchical Bayesian" framing claims more than the implementation delivers (just one fixed-prior conjugate update) | Either implement an actual hierarchical model (e.g., `kappa ~ HalfNormal` partial-pooling across players within each rate stat) OR rename the section to "empirical-Bayes shrinkage with conjugate update." Honest labeling. |

### 4.3 Codex (Agent B) — specific fixes from cross-review

| Defect (R1) | Fix required (R2) |
|---|---|
| Era counterfactual confounded model capacity (10 yrs vs 4 yrs training) with era effect | Either: (a) match training-set sizes (subsample 2015-2024 to non-overlapping 4-yr windows and compare predictions across them), OR (b) drop the era counterfactual entirely from this round and replace it with a per-component persistence model (see §5.2). Recommend (b) — the era question is unfixable in this scope. |
| QRF prediction intervals never coverage-checked on 2025 holdout | Compute empirical coverage of the 80% interval on 2025 ROS holdout: what fraction of held-out player-seasons fall inside [q10, q90]? Report this. If coverage is < 70%, the intervals are mis-calibrated and verdicts based on them must be downgraded. |
| SHAP/permutation Spearman = 0.195 vs pre-committed 0.60 threshold (moved goalposts) | Either pass the threshold (tune the model or the SHAP variant), OR honestly downgrade to "feature ranking is unstable; we report permutation only." Don't move the goalpost again. |
| Murakami substitution under kill-gate was paper compliance | With MLB Stats API now in `data_pull.py` (per §4.1), Murakami should resolve cleanly. Re-run with him included. |

---

## 5. New analysis (the three integrated framings)

### 5.1 Framing A — Persistence-by-Component Atlas (the headline analysis)

Build a **2022-2025-trained "component persistence model"** answering: *"Given a player's 22-game observed values for {BB%, K%, BABIP, ISO, EV p90, HardHit%, Barrel%, xwOBA, xwOBA-wOBA gap}, what's the predicted rest-of-season wOBA delta vs preseason prior?"*

**Per agent (divergent methodology):**

- **Claude (Agent A):** Fit per-component empirical-Bayes shrinkage (or true hierarchical Bayes) → predicted ROS rate per component → wOBA reconstruction from posterior components. Rank all 2026 hitters by predicted ROS-wOBA-vs-prior delta with credible intervals.
- **Codex (Agent B):** Train LightGBM on 2022-2025 player-seasons predicting ROS wOBA delta from the 22-game component vector. QRF for prediction intervals. Rank all 2026 hitters by point + 80% interval.

**Outputs (each agent):**
- Top-10 SLEEPER signals: predicted ROS delta in top decile AND not in mainstream top-20 leaderboard
- Top-10 FAKE HOT: in mainstream top-20 leaderboard AND predicted ROS delta < 0
- Top-10 FAKE COLD: bottom decile of April performance AND predicted ROS delta > 0
- Sanity check: re-run on the 5 named R1 starters; verdicts should be roughly consistent with R1 (Pages NOISE, Trout AMBIG, etc.) but with the corrected priors

### 5.2 Framing B — Contact-Quality Mismatch Sheet (subsumed within A)

Codex specifically must include xwOBA-wOBA gap as a top-tier feature in the LightGBM model and report:
- Top-10 hitters by absolute |xwOBA - wOBA| gap (positive = over-performing, negative = under-performing)
- Cross-tab against the FAKE HOT / FAKE COLD lists from §5.1 — overlap should be high

Claude does the Bayesian version: posterior over (xwOBA - wOBA) gap with shrinkage, predicting ROS regression toward xwOBA rather than toward prior wOBA.

### 5.3 Framing C — Reliever K% True-Talent Board

For every 2026 reliever ≥ 25 BF:
- Build per-reliever K% posterior using reliever-specific stabilization rates (Carleton: K% ~ 70 BF for relievers)
- Anchor prior to 2022-2025 weighted mean K% (or league average for debuts)
- Rank by posterior K% q50 vs prior K%

**Per agent:**
- Claude: Bayesian Beta-Binomial per reliever, posterior K% q10/q50/q80/q90
- Codex: LightGBM trained on 2022-2025 reliever-seasons predicting ROS K%, QRF for intervals

**Outputs:**
- Top-5 SLEEPER reliever K% risers: posterior q50 K% materially above prior, NOT a known closer (use reference list from `pybaseball.pitching_stats(2025)` top-30 saves leaders)
- Top-5 FAKE-DOMINANT reliever K%: high April K% but posterior shrinks heavily back to prior

Mason Miller from R1 is a sanity-check case (he's a known closer with elite pre-2026 K%, so should NOT appear in sleeper list — but should appear high on K% rise board if his K% rise is real).

---

## 6. Kill Criteria (pre-committed, R2)

| Check | Threshold | Failure action |
|-------|-----------|----------------|
| **Methodology fixes** (§4) all implemented | Yes / No | If FAIL on any: cross-review will re-flag; the new findings cannot be published until fix lands |
| **Universe coverage** | ≥ 250 hitters and ≥ 70 relievers in scan | If FAIL: lower threshold from 50 PA to 40 PA hitters / 25 BF to 20 BF relievers, document |
| **Sleeper signal yield** (H1) | ≥ 3 sleeper hitters in top-10 not in mainstream top-20 | If FAIL: H5 null fallback — article framing pivots to "you can't beat the lists in April" |
| **Fake hot yield** (H2) | ≥ 3 mainstream-listed hitters with predicted ROS delta < 0 | If FAIL: drop FAKE HOT section; lead with FAKE COLD instead |
| **Cross-method agreement on sleepers** | Both methods independently surface ≥ 3 of the same names in their top-10 sleeper list | If FAIL: comparison memo treats divergent picks as ambiguous; only convergent names are publishable as "we both saw it" |
| **QRF coverage check** (Codex) | Empirical 80%-interval coverage on 2025 holdout ≥ 70% | If FAIL: Codex's interval-based verdicts (NOISE/SIGNAL) downgrade to point-estimate-with-warning |

---

## 7. Scope Fence

This is **Round 2 — universe scan + methodology fixes**.

**IN scope:**
- All §4 methodology fixes (blocking)
- §5.1 Persistence-by-Component Atlas (universe scan)
- §5.2 xwOBA-gap analysis (subsumed in A)
- §5.3 Reliever K% True-Talent Board
- Sanity-check cross-validation against R1's 5 named starters
- Sleeper / fake-hot / fake-cold name lists with component evidence per name

**OUT of scope:**
- Defensive metrics (still too noisy in 22 games)
- Park factor adjustments (use raw stats, document caveat)
- Causal mechanism for *why* a sleeper signal exists (descriptive only)
- Pitcher starter analysis (relievers only for §5.3)
- Year-effect modeling beyond 2022-2025 for projection
- ABS-zone shape effects
- Round 3 follow-ups (note in READY_FOR_REVIEW.md if interesting questions surface)

---

## 8. Deliverables per agent (R2)

Write into existing folders, prefixed with `r2_` or in `round2/` subdirectories. Do NOT overwrite Round 1 outputs.

1. `analyze_r2.py` — one-command R2 entry point (calls R1 data_pull if needed for fixes, then R2 modules)
2. New module scripts:
   - For both: `r2_universe.py` (build 2026 hitter and reliever universe with required filters)
   - For both: `r2_persistence_atlas.py` (Framing A)
   - For both: `r2_reliever_board.py` (Framing C)
   - Methodology-fix modules as needed (e.g., Claude's `r2_stabilization.py` with player-season bootstrap; Codex's `r2_qrf_coverage.py` with the coverage diagnostic)
3. `REPORT_R2.md` — ~2,500 words, structured: exec summary → methodology fixes summary → Persistence Atlas (A) findings → xwOBA-Gap (B) findings → Reliever Board (C) findings → kill-gate outcomes → open questions
4. `charts/r2/` — PNGs for: top-10 sleeper hitters, top-10 fake hot, top-10 fake cold, top-5 sleeper relievers, QRF coverage diagnostic (Codex), updated stabilization curves (Claude)
5. `findings_r2.json` — top-level keys: `methodology_fixes`, `persistence_atlas` (sleepers / fake_hot / fake_cold), `xwoba_gap`, `reliever_board`, `r1_sanity_check`
6. `READY_FOR_REVIEW_R2.md` — ≤ 500 words, lead with the headline names (sleeper picks, fake hots) followed by methodology-fix status

---

## 9. Timeline (compressed vs R1 — most data is cached)

| Day | Focus | Gate |
|------|-------|------|
| 1 | Methodology fixes + universe build | Both agents have corrected priors / models + universe parquet |
| 1-2 | Persistence Atlas + Reliever Board scans | First findings_r2.json drafts |
| 2 | Charts + REPORT writing + READY_FOR_REVIEW | Both READY_FOR_REVIEW_R2.md exist |
| 3 | Cross-review (each agent reviews the other's R2 output) | Both R2 reviews exist |
| 3 | Comparison memo (conversational) | COMPARISON_MEMO_R2.md complete |

Estimated agent runtime: 3-4 hours each (lighter than R1; data is cached).

---

## 10. Risks & Mitigations (R2-specific)

| Risk | Mitigation |
|------|------------|
| Agents skip the methodology-fix work and jump to new analysis | §4 is enforced in agent prompts as blocking before §5 work begins |
| Universe scan produces zero sleeper signals | H5 null fallback specified — article framing pivots to "you can't beat the April lists" |
| Both methods diverge wildly on sleeper picks | Cross-method agreement is a kill-criterion; only convergent names get published as confident calls |
| Mainstream leaderboard reference is stale by next week | The article publishes within 1 week of R2 completion; reference is dated as "as of YYYY-MM-DD" |
| Reliever K% rise turns out to be entirely small-sample noise after coverage check | Section §5.3 becomes a "noise floor for reliever K% in 25 BF" finding instead of a sleeper list |

---

## 11. How to Run

```bash
cd .

# Each agent runs from their own folder:
cd claude-analysis && python analyze_r2.py
cd ../codex-analysis && python analyze_r2.py

# Cross-review starts when both READY_FOR_REVIEW_R2.md exist
```

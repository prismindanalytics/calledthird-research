# The Hot-Start Half-Life — Research Brief

**Status:** Draft (pre-launch)
**Target deliverables:** Flagship article — *"The Hot-Start Half-Life: What 22 Games Actually Tells You About 2026"*
**Est. effort:** 1 round, ~3 days agent + 2 days writing
**Principal risk:** Stabilization-rate effects under the 2026 rule environment may not differ enough from classical (Carleton 2007) curves to support the "novel methodology" framing — analysis must produce a credible answer either way.

**Editorial context:** As of 2026-04-25, MLB is ~22 games into the season. The April hot-starter narrative is dominant in baseball media: Andy Pages (.409), Ben Rice (.338/.476/.800, AL BA leader), Munetaka Murakami (7 HRs through 21 games — most by a Japanese-born player to start a career), Mike Trout (5 HRs over 4 games at Yankee Stadium), Mason Miller (30⅔ consecutive scoreless IP). CalledThird's job is the rigorous "what survives" reference piece — *not* "here's a list" content.

---

## 1. The Question

**Primary:** Given 22 games of the 2026 season under a substantially shifted environment (ABS challenge era, walks at 75-year high, K rates moving), which named April 2026 hot starters represent durable signal vs noise — with proper rest-of-season prediction intervals — and do classical stabilization rates still apply or has the 2026 environment shifted them?

**Secondary:**
- For each named hot starter, what 22-game features actually predict rest-of-season performance (vs which are seductive but uninformative)?
- What is the empirical noise floor — among the top-5 BA/OPS finishers across 22 games in 2022-2025, what fraction maintained their pace through the rest of the season?
- Which historical analogs best match each 2026 hot starter through 22 games, and what happened to those analogs?

---

## 2. Hypothesis

**H1 (primary):** At least one core hitting stat's stabilization rate has shifted by ≥ 10% under the 2026 rule environment (2022-2025 data) compared to classical estimates derived from 2010-2017 data. Most likely candidates: BB% (faster stabilization due to more deterministic ABS-influenced zone), BABIP (slower stabilization due to bunched-up exit-velocity distribution), K% (faster due to environment shift).

**H2 (named hot starters):** Of the 5 named 2026 hot starters, the dual-agent analyses agree on a partition: ≥ 1 represents durable signal (rest-of-season 80% prediction interval excludes "regress fully to preseason projection mean") and ≥ 1 represents noise (interval overlaps preseason projection mean by > 70%). The rest are ambiguous.

**H3 (null-result alternative):** If 2022-2025-derived stabilization curves match classical Carleton estimates within bootstrap CI, AND all 5 named hot starters' rest-of-season projections overlap their preseason projection by > 70%, the article becomes *"April Lies — The Stats That Don't Survive May."* Publishable null with the same data; different framing.

---

## 3. Data

### 3.1 Primary corpus

- **Statcast pitch-level 2022-2026 (regular season).** 2025 full season already cached at `../pitch-tunneling-atlas/data/statcast_2025_full.parquet`. 2026-to-date through Apr 22 cached at `../abs-walk-spike/data/statcast_2026_mar27_apr22.parquet`. **You must extend the 2026 corpus through 2026-04-24** using `pybaseball.statcast(start_dt='2026-04-23', end_dt='2026-04-24')`. For 2022-2024, fetch via `pybaseball.statcast` in monthly chunks and cache to `data/`.
- **Game logs / batting splits 2022-2026.** Use `pybaseball.batting_stats(2022, 2026, qual=1)` for per-season aggregates. Use `pybaseball.batting_stats_range(start_dt, end_dt)` for windowed ROS subsets.

### 3.2 Derived inputs (you may build / cache)

- Per-player season game-sequence index (game_n within season) keyed on player + date
- Empirical league-environment-adjusted rate stats per season (BB%, K%, BABIP, ISO, wOBA)
- The classical Carleton 2007 stabilization-rate table (you'll need to embed this as a constant — it's in published baseball-research literature; standard values for reference: BB% ~120 PA, K% ~60 PA, ISO ~160 PA, BABIP ~820 PA for half-stabilization).

### 3.3 External joins

- **Preseason projections (Steamer or ZiPS).** *Best-effort.* If `pybaseball.fangraphs_projections` or similar is available, use it. If not, fall back to **prior-3-year weighted mean (5/4/3 ratio for most-recent-to-oldest)** per player as the player-specific prior. Document whichever you used.
- **Player ID crosswalk.** `pybaseball.playerid_lookup` for the named hot starters.

### 3.4 Named 2026 hot starters (must produce explicit predictions for each)

| Player | Position | Headline | What to project |
|---|---|---|---|
| Andy Pages | OF (LAD) | .409 BA through ~22 games, 95th-pctile hard-hit rate | Rest-of-season wOBA, BABIP, ISO with 80% CI |
| Ben Rice | C/1B (NYY) | .338/.476/.800, AL BA leader | Rest-of-season wOBA, OPS, ISO |
| Munetaka Murakami | 3B (NYM, debut season) | 7 HRs through 21 games | Rest-of-season HR rate, ISO, K% |
| Mike Trout | OF (LAA) | 5 HRs in 4-game Yankee Stadium series; full-season comeback | Rest-of-season wOBA, durability flag |
| Mason Miller | RP (SD) | 30⅔ consecutive scoreless IP | Rest-of-season ERA, K%, BB%, expected SD until first ER |

### 3.5 Known data-quality issues

- **Schema drift 2026:** Statcast `sz_top` / `sz_bot` are now deterministic per-batter ABS-rule values (within-batter SD = 0.000) vs per-pitch posture estimates pre-2026. Do NOT use `sz_*`-derived features in any cross-season model. Use absolute `plate_x` / `plate_z` only. (This is the same caveat as `abs-walk-spike`.)
- **Ben Rice position changes:** Confirm catcher vs 1B PA splits — defensive position can affect projection priors.
- **Murakami first-MLB-season:** No prior MLB projections exist. Use NPB-translated expectations or treat as league-average prior with documented caveat.
- **Mason Miller IP volume:** Reliever stabilization rates differ markedly from starters. Use reliever-specific stabilization estimates (BB% ~170 BF, K% ~70 BF for relievers per Carleton) — do NOT apply starter rates.

---

## 4. Features / Variables

| # | Name | Definition | Source |
|---|------|------------|--------|
| 1 | `wOBA_22g` | Through-22-game weighted on-base average per player | Statcast / batting_stats |
| 2 | `BABIP_22g` | Hits on balls in play / (AB − K − HR + SF) | Statcast |
| 3 | `BB%_22g`, `K%_22g`, `ISO_22g` | Plate-discipline / power rates | Statcast |
| 4 | `EV_p90_22g`, `HardHit%_22g`, `Barrel%_22g` | Quality-of-contact indicators (Statcast) | Statcast |
| 5 | `xwOBA_22g` minus `wOBA_22g` | Statcast luck residual | Statcast |
| 6 | Preseason prior wOBA | Steamer/ZiPS preseason projection or 3-year weighted mean fallback | FanGraphs or computed |
| 7 | League-environment rate (per season) | Season-level BB%, K%, BABIP, ISO mean | batting_stats |
| 8 | `IP_streak`, `K%_streak`, `BB%_streak` (Miller-specific) | Reliever in-streak rates | Statcast / game logs |

---

## 5. Methodology

### 5.1 Agent A (Claude) — interpretability-first

**Divergent methodological mandate:** Hierarchical Bayesian + classical stabilization theory. Specifically must use:

1. **Hierarchical Bayesian model (numpyro or PyMC)** for each rate stat (BB%, K%, BABIP, ISO, wOBA). Per-player partial pooling: `p_player ~ Beta(α + observed_successes, β + observed_failures)` with α, β fit from the prior (preseason projection or 3-year weighted mean). Posterior predictive draws give the rest-of-season distribution.
2. **Empirical-Bayes shrinkage** for individual hot-starter projections — derive shrinkage weights directly from the 2022-2025 cross-season variance decomposition.
3. **Bootstrap re-estimation of stabilization rates** on 2022-2025 (via standard split-half reliability method: take all 2022-2025 player-seasons with ≥ 200 PA, randomly split each season's PAs into two halves N times, compute correlation as N grows from 25 to 600). Report half-stabilization point with bootstrap CI for each stat. Compare to Carleton 2007 published values.
4. **PELT or Bayesian online change-point detection** on each named hot starter's per-game rolling rate to test whether the player's underlying rate has actually shifted vs preseason projection (vs being a noise excursion).

**You must NOT use:** Gradient boosting, deep learning, transformers, attention. (Those are Agent B's territory.)

### 5.2 Agent B (Codex) — ML-engineering

**Divergent methodological mandate:** Ensemble ML + counterfactual prediction. Specifically must use:

1. **Gradient-boosted regressor (LightGBM or XGBoost)** trained on 2022-2024 hitter-seasons to predict full-season wOBA from "first-N-PA features" (rolling rates, contact-quality features, plate-discipline rates, league-environment context). Train/validate/test split: 2022-2023 train, 2024 validate, 2025 test. Score 2026 hot starters.
2. **Quantile regression forests (QRF)** for prediction intervals — produce 10th/50th/80th/90th percentile rest-of-season projections for each named 2026 hot starter.
3. **Permutation feature importance + SHAP** to identify which 22-game features are actually informative for rest-of-season outcomes (vs which fans falsely believe matter).
4. **Counterfactual model comparison** — train one model on 2015-2024 (broad era), one on 2022-2025 only (current rule environment), apply each to 2026 hot starters. The delta between predictions is your evidence for whether the 2026 environment shifts the noise floor. Note: you only need 2015-2024 data for *aggregate* environment-shift evidence; the 2022-2025 cohort is sufficient for individual hot-starter projections.
5. **Historical-analog retrieval (k-NN over feature vector)** — for each 2026 hot starter, find 5 nearest 2015-2024 player-seasons by 22-game feature similarity. Report what those analogs did rest-of-season.

**You must NOT use:** Bayesian methods, MCMC, hierarchical priors, or numpyro/PyMC. (Those are Agent A's territory.)

### 5.3 Why the divergence matters

Bayesian shrinkage gives interpretable per-player posterior intervals grounded in priors. Gradient-boosted ensembles capture non-linear feature interactions and counterfactual era comparison. If both methods independently agree which hot starters are real and which are noise, the finding is robust enough to publish with high confidence. If they disagree, the comparison memo digs into the methodology delta — which itself becomes a story about the limits of either approach.

---

## 6. Kill Criteria

Pre-committed thresholds. Any failure triggers the documented fallback.

| Check | Threshold | Failure action |
|-------|-----------|----------------|
| **Stabilization-rate shift** | At least one of {BB%, K%, BABIP, ISO} 2022-2025 half-stabilization point differs from Carleton classical by ≥ 10% with non-overlapping bootstrap CI | If FAIL: drop "novel methodology" framing; pivot article to "what 22 games tells you, with proper CIs" using classical rates as-is |
| **Named hot starter coverage** | All 5 named hot starters produce a rest-of-season point estimate + 80% CI | If FAIL for any: replace with next-best 2026 hot-starter candidate from top-10 BA leaderboard, document substitution |
| **Cross-agent agreement** | For each named hot starter's rest-of-season point estimate, the two agents' 80% CIs must overlap | If FAIL: comparison memo investigates the methodology divergence; if irreconcilable, that hot starter is published as "ambiguous" with both ranges shown |
| **Sample-size sufficiency** | Each named hot starter has ≥ 50 PA (hitters) or ≥ 25 BF (Miller) before projecting | If FAIL: exclude from named list, note in methodology |
| **Historical-analog quality** | Each hot starter has ≥ 5 reasonable 2015-2024 analogs (cosine sim ≥ 0.7) | If FAIL for any: report "no good analogs" honestly — that itself is a finding |

**Null-result fallback article:** *"April Lies — The Stats That Don't Survive May."* Same data, same projections, but the framing becomes "here's the receipts on noise." Already publishable; CalledThird has a strong myth-busting brand for exactly this kind of finding.

---

## 7. Scope Fence

This is **Round 1 — full v1 analysis**.

**IN scope:**
- Hitting stabilization rates (BB%, K%, BABIP, ISO, wOBA) for 2022-2025 vs Carleton classical
- Per-named-hot-starter rest-of-season projection with 80% CI (Pages, Rice, Murakami, Trout, Miller)
- Historical-analog retrieval for each named hot starter (5 nearest 2015-2024 analogs)
- League-environment rate stats per season (so context shifts are documented)
- One reliever case (Miller) — with reliever-specific stabilization rates
- Honest reporting of the noise floor: of 2022-2025 top-5 22-game BA leaders, what fraction sustained?

**OUT of scope (do not pursue, even if tempting):**
- Defensive metric stabilization (UZR/DRS/OAA — too noisy in 22 games)
- Starting-pitcher rate stabilization beyond the Miller-specific reliever case
- Team-level standings predictions
- Park factor adjustments — use raw stats, document caveat in limitations
- Year-effect modeling beyond 2022-2025 (don't use pre-2022 data for individual-player projection — keep ball/rule consistency); 2015-2024 is allowed *only* for the Codex counterfactual era-shift comparison and for analog retrieval
- ABS-zone shape effects on hot starts (separate analysis project; reference but don't model)
- Causal claims about *why* stabilization shifted (descriptive only — "rate X has shifted by Y%" is the claim ceiling)

---

## 8. Deliverables per agent

Each agent must produce in their own folder:

1. `analyze.py` — one-command reproduction entry point
2. Module scripts (one per methodology component)
3. `REPORT.md` — ~2,000 words structured: exec summary → methods → stabilization findings → per-named-hot-starter projections → kill-gate outcomes → open questions
4. `charts/` — PNGs for: (a) stabilization curves 2022-2025 vs classical, (b) each named hot starter's projection vs prior, (c) at least one league-environment context chart
5. `findings.json` — machine-readable summary with stabilization point estimates + CIs, and per-hot-starter projection point + 10/50/80/90 quantiles
6. `READY_FOR_REVIEW.md` — ≤ 500 words

---

## 9. Timeline

| Day | Focus | Gate |
|------|-------|------|
| 1 | Data pull (extend 2026 to Apr 24, fetch 2022-2024 backfill); confirm named hot starters' PA/BF | Both agents have full corpus |
| 2 | Stabilization rate re-estimation (Agent A); ML model training (Agent B) | First findings.json drafts |
| 3 | Per-hot-starter projections + analog retrieval; charts; READY_FOR_REVIEW | Both READY_FOR_REVIEW.md exist |
| 4 | Cross-review (each agent reviews the other) | Both reviews exist |
| 5 | Comparison memo (conversational, in main session) | COMPARISON_MEMO.md complete |

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Preseason projections unavailable via API | 3-year weighted-mean fallback specified in §3.3 |
| 2026 hot-starter sample too small | Kill criteria §6 defines minimum PA / BF and substitution rule |
| Stabilization shifts are too subtle to be publishable | Null-result fallback framing already specified in §6 |
| Agents converge by accident on identical method | Mandate divergence in §5.1 / §5.2 is enforced in agent prompts |
| Codex era-shift counterfactual confounded by ball-change era | Use 2022-2025 (post-deadened-ball stabilization) as primary; 2015-2024 only for *additional* aggregate evidence with documented caveat |
| Mason Miller's streak ends mid-analysis | Article framing handles either: ongoing → "still going" sidebar; ended → "what we predicted vs what happened" sidebar. Don't condition the analysis on streak status. |

---

## 11. How to Run

```bash
cd .

# Each agent runs from their own folder:
cd claude-analysis && python analyze.py
cd ../codex-analysis && python analyze.py

# Compare via reviews/ docs after both READY_FOR_REVIEW.md exist
```

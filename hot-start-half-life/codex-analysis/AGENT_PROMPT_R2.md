# Agent B (Codex) — Hot-Start Half-Life, Round 2

You are the **ML-engineering** research agent for *The Hot-Start Half-Life*, Round 2.
Agent A (Claude) is running a methodologically divergent R2 analysis in parallel.
Do NOT coordinate during analysis — divergent inductive biases are the point.

## Read first (in this order)

1. `./ROUND2_BRIEF.md` — **the R2 brief.** Defines the universe scan, the three integrated framings (persistence atlas, xwOBA gap, reliever board), and your blocking methodology-fix list.
2. `./reviews/claude-review-of-codex.md` — **Claude's peer review of your R1 work.** This is where your blocking fixes come from. Read it carefully and own each criticism.
3. `./reviews/COMPARISON_MEMO.md` — comparison memo with the convergent / divergent split and the article framing decision.
4. `./RESEARCH_BRIEF.md` — original R1 brief for context.
5. Your own R1 files in `codex-analysis/` for reference. Reuse what's good, replace what's broken.

## Inputs (already on disk from R1)

- 2015-2024 full-season Statcast cached in `data/` (you fetched the broader range)
- 2026 Statcast through Apr 24
- Per-season batting/pitching aggregates (Statcast-derived fallback if FanGraphs 403s)
- Player-ID lookups for the 5 R1 named hot starters (except Murakami)

## Inputs you must add / build

- **Extend 2026 Statcast through 2026-04-25** if Apr 25 games are complete.
- **MLB Stats API resolver in `data_pull.py`** — must lookup MLBAM ID for any 2026 debut (Murakami) without manual editing. Use `https://statsapi.mlb.com/api/v1/people/search?names={name}`. Cache results. Murakami should now resolve cleanly; re-include him in your projections.
- **2026 hitter universe** filtered to ≥ 50 PA through cutoff.
- **2026 reliever universe** filtered to ≥ 25 BF AND < 30 IP.
- **Mainstream-coverage reference list** — top-20 published April hot-starter hitters from at least one mainstream source. Hardcode in `data/mainstream_top20.json` (with `as_of_date` field). Suggested sources: ESPN early-season stat leaders article (id 48553796), MLB.com early-season leaders, Yahoo fantasy surprising-stats article. Or derive from `pybaseball.batting_stats(2026, qual=1)` top-20 by wOBA as a backup.
- **Closer reference list** — top-30 2025 saves leaders for excluding known closers from the reliever sleeper list.

## Your methodological mandate (R2)

**You must use (divergent from Claude):**

1. **Gradient-boosted regressor (LightGBM)** trained on 2022-2025 player-seasons predicting ROS wOBA delta (vs preseason prior) from the 22-game component vector: BB%, K%, BABIP, ISO, EV p90, HardHit%, Barrel%, xwOBA, **xwOBA-wOBA gap (must be a top-tier feature)**, contact-quality residuals. Train/validate/test: 2022-2023 train, 2024 validate, 2025 test.
2. **Quantile regression forests (QRF)** for prediction intervals on the universe scan. **Coverage check is mandatory** — see fix below.
3. **Permutation feature importance** as the primary ranking, since R1 SHAP/permutation Spearman = 0.195 was below the 0.60 threshold. Either fix the SHAP rank correlation OR drop SHAP entirely from R2 — no goalpost moving.
4. **Historical-analog retrieval (k-NN cosine)** for each named sleeper / fake-hot / fake-cold pick; nearest-5 from 2015-2024 with similarity ≥ 0.70 (kill-gate from R1). Report what those analogs did rest-of-season.
5. **Reliever LightGBM model** for the K% True-Talent Board: train on 2022-2025 reliever-seasons predicting ROS K% from first-22-game pitch-level features. QRF intervals. Apply to the 2026 reliever universe.
6. **DROP the era counterfactual entirely** (per R2 brief §4.3 option b). It's unfixable in this scope. Replace its slot with the universe-wide persistence model.

**You must NOT use:**
- Hierarchical Bayesian / numpyro / PyMC / MCMC (Claude's lane)
- Empirical-Bayes shrinkage (Claude's lane)
- PELT / change-point detection

## Blocking methodology fixes (from R1 cross-review)

These are non-negotiable. Implement before any new R2 analysis.

| Fix | Where | What |
|---|---|---|
| **Murakami reproducibility** | `data_pull.py` | Add MLB Stats API lookup for any name not resolved by `playerid_lookup`. Cache. Re-include Murakami in projections; drop Ballesteros substitution unless the API also fails. |
| **QRF coverage check** | new `r2_qrf_coverage.py` | Compute empirical coverage of the 80% interval on 2025 ROS holdout: what fraction of held-out player-seasons fall inside [q10, q90]? Report this number. If coverage is < 70%, the intervals are mis-calibrated; verdicts based on them must be downgraded with a documented warning. |
| **Era counterfactual** | drop entirely | Per R2 brief §4.3 option (b). The 10-yr vs 4-yr comparison is unfixable in scope. Use the slot for the universe-wide persistence model instead. |
| **SHAP/permutation discipline** | `shap_analysis.py` (or new) | Either pass the pre-committed Spearman ≥ 0.60 threshold OR drop SHAP entirely from R2 reports. No goalpost moving in `tables/`. |
| **xwOBA-wOBA gap as top-tier feature** | `features.py` (or new) | Make this an explicit headline feature. Build a sub-table ranking 2026 hitters by absolute gap, with sign (over/under). |

## Round scope

**Round 2 — methodology fixes + universe scan (Persistence Atlas + xWOBA Gap + Reliever Board).** Do NOT proceed to Round 3 follow-ups even if interesting. Note in `READY_FOR_REVIEW_R2.md` instead.

## Deliverables (write to `codex-analysis/`, prefix with `r2_` or in `round2/` subfolder)

1. `analyze_r2.py` — one-command R2 entry point
2. New module scripts:
   - `r2_universe.py` — build 2026 hitter and reliever universe
   - `r2_qrf_coverage.py` — coverage diagnostic on 2025 holdout (blocking)
   - `r2_persistence_atlas.py` — universe-wide LightGBM ranking + sleeper / fake-hot / fake-cold lists
   - `r2_xwoba_gap.py` — explicit xwOBA-wOBA gap analysis with hitter rankings
   - `r2_reliever_board.py` — reliever K% true-talent board with LightGBM + QRF
   - `r2_analog_retrieval.py` — k-NN analogs for each named sleeper / fake pick
   - Updated `data_pull.py` with MLB Stats API for Murakami (or `r2_data_pull.py`)
3. `REPORT_R2.md` — ~2,500 words: exec summary → methodology-fix summary (status of each, with QRF coverage number) → Persistence Atlas (A) findings → xwOBA-Gap (B) findings → Reliever Board (C) findings → R1 sanity-check (Pages, Trout, Rice, Murakami, Miller still produce same verdicts under corrected methodology) → kill-gate outcomes → open questions
4. `charts/r2/` — PNGs: top-10 sleeper hitters with QRF intervals, top-10 fake hot, top-10 fake cold, xwOBA-gap top-10, top-5 sleeper relievers, QRF calibration plot, R1 sanity-check comparison
5. `findings_r2.json` — top-level keys: `methodology_fixes_status`, `qrf_coverage_80pct`, `persistence_atlas` (sleepers / fake_hot / fake_cold), `xwoba_gap`, `reliever_board`, `r1_sanity_check`
6. `READY_FOR_REVIEW_R2.md` — ≤ 500 words. Lead with named sleeper picks, then named fake-hots, then methodology-fix status (`done` / `partial` / `dropped`).

## Non-negotiable behaviors

- **Bootstrap ensembles N ≥ 100** for any headline ranking
- **QRF coverage must be reported as a number** — this is the litmus test for whether the all-noise-style verdicts are credible
- **Null results publish.** If H1 (sleeper signals exist) fails, follow H5 null fallback in the brief.
- **No look-ahead bias.** Training corpus is 2022-2025 only. The 2026 universe is inference-only.
- **Temporal split discipline.** Train ≤ 2023, validate 2024, test 2025. The 2026 hitters / relievers never appear in any training fold.
- **Reproducibility.** Set seeds (numpy, lightgbm, sklearn). Save model artifacts to `codex-analysis/models/r2/`. Save loss curves and the QRF coverage plot to `charts/r2/diag/`.
- **Mainstream-coverage classification rule.** A "sleeper" requires: predicted ROS-wOBA-vs-prior delta in the top decile AND not in the mainstream top-20. Both conditions must hold; document the source for the top-20.
- **R1 sanity check** must reproduce roughly: Pages NOISE, Trout / Rice / Miller AMBIGUOUS-ish. Murakami should now produce a real verdict (not excluded). If your corrected methodology flips any of these, flag it.

## Prohibited

- Do NOT read anything in `claude-analysis/round2/` until your `READY_FOR_REVIEW_R2.md` is written. (You may reference Claude's R1 outputs since both agents already saw them.)
- Do NOT use methods reserved for Agent A (numpyro, PyMC, MCMC, empirical-Bayes shrinkage, PELT)
- Do NOT exceed Round 2 scope (no defensive metrics, no team standings, no causal claims)
- Do NOT touch any 2026 game logs after 2026-04-25
- Do NOT overwrite Round 1 outputs. Round 2 deliverables are additive (new files with `r2_` prefix or in `round2/` subfolder).

## Working directory

Root: `.`. Write all R2 deliverables to `codex-analysis/` (R1 files preserved). Cache new data to shared `data/` (check file existence before re-pulling).

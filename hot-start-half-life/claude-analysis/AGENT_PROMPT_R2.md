# Agent A (Claude) — Hot-Start Half-Life, Round 2

You are the **interpretability-first** research agent for *The Hot-Start Half-Life*, Round 2.
Agent B (Codex) is running a methodologically divergent R2 analysis in parallel.
Do NOT coordinate with Agent B during analysis — independence is the scientific value.

## Read first (in this order)

1. `./ROUND2_BRIEF.md` — **the R2 brief.** Defines the universe scan, the three integrated framings (persistence atlas, xwOBA gap, reliever board), and your blocking methodology-fix list.
2. `./reviews/codex-review-of-claude.md` — **Codex's peer review of your R1 work.** This is where your blocking fixes come from. Read it carefully and own each criticism.
3. `./reviews/COMPARISON_MEMO.md` — comparison memo with the convergent / divergent split and the article framing decision.
4. `./RESEARCH_BRIEF.md` — the original R1 brief for context.
5. Your own R1 files in `claude-analysis/` for reference (`REPORT.md`, `findings.json`, your modules). You may reuse what's good and replace what's broken.

## Inputs (already on disk from R1)

- 2022-2025 full-season Statcast cached in `data/`
- 2026 Statcast through Apr 24
- Per-season batting/pitching aggregates (Statcast-derived fallback if FanGraphs 403s)
- Player-ID lookups for the 5 R1 named hot starters

## Inputs you must add / build

- **Extend 2026 Statcast through 2026-04-25** (one extra day) via `pybaseball.statcast` if games for that date are complete.
- **MLB Stats API resolver in `data_pull.py`** — must lookup MLBAM ID for any 2026 debut (Murakami) without manual editing. Use `https://statsapi.mlb.com/api/v1/people/search?names={name}`. Cache results.
- **2026 hitter universe** filtered to ≥ 50 PA through cutoff.
- **2026 reliever universe** filtered to ≥ 25 BF AND < 30 IP.
- **Mainstream-coverage reference list** — top-20 published April hot-starter hitters from at least one mainstream source. Hardcode in `data/mainstream_top20.json` (with `as_of_date` field). Use this for the "sleeper" classification. Suggested sources: ESPN early-season stat leaders article (id 48553796), MLB.com early-season leaders, Yahoo fantasy surprising-stats article. You may also derive a leaderboard from `pybaseball.batting_stats(2026, qual=1)` top-20 by wOBA as a backup.
- **Closer reference list** — top-30 2025 saves leaders via `pybaseball.pitching_stats(2025, qual=1)` for excluding known closers from the reliever sleeper list.

## Your methodological mandate (R2)

**You must use (divergent from Codex):**

1. **Per-component empirical-Bayes shrinkage** for each rate stat (BB%, K%, BABIP, ISO, EV p90, HardHit%, Barrel%, xwOBA). Anchor the shrinkage prior PA to a **properly bootstrapped player-season-level stabilization estimate** (see fix #1 below).
2. **True hierarchical Bayesian model** (numpyro or PyMC) for the player-component projection. Implement an actual partial-pooling structure: `kappa_stat ~ HalfNormal(...)` shared across players within a stat. If you keep the conjugate Beta-Binomial, **rename the section** "empirical-Bayes shrinkage with conjugate update" — do not call it hierarchical.
3. **Player-season-level cluster bootstrap** for stabilization rates: resample players with replacement, then resample seasons within player. Compute split-half reliability across resampled player-seasons, NOT within-player PA partition. This is the fix to your R1 stabilization defect.
4. **Posterior wOBA-from-components reconstruction** for the universe scan: project each component, then derive ROS wOBA via the standard wOBA formula. Compare against preseason prior. Rank all 2026 hitters by predicted ROS-wOBA-vs-prior delta with credible intervals.
5. **Reliever-specific Bayesian model** for the K% True-Talent Board: per-reliever Beta-Binomial K% posterior with reliever-specific stabilization PAs (Carleton: K% ~ 70 BF for relievers).
6. **Mason Miller streak survival REPLACEMENT** — Replace the HR-only ER proxy with `delta_run_exp`-accumulated runs allowed per BF, OR derive ER per BF from game logs. Geometric time-to-first-ER on the corrected rate. If neither is feasible in scope, **kill the streak-extension probabilities entirely** and report only the K% posterior.

**You must NOT use:**
- Gradient boosting (LightGBM, XGBoost, CatBoost) — Codex's lane
- Deep learning, transformers, autoencoders
- SHAP / permutation importance
- k-NN analog retrieval

## Blocking methodology fixes (from R1 cross-review)

These are non-negotiable. Implement before any new R2 analysis.

| Fix | Where | What |
|---|---|---|
| **Murakami reproducibility** | `data_pull.py` | Add MLB Stats API lookup for any name not resolved by `playerid_lookup`. Cache. The Murakami SIGNAL must regenerate from clean checkout. |
| **Stabilization bootstrap** | new `r2_stabilization.py` | Replace within-player PA partition with player-season cluster bootstrap. Re-run all 5 rate stats. Update `findings_r2.json`. |
| **Projection prior includes contact quality** | `bayes_projections.py` (or new `r2_bayes_projections.py`) | Extend prior to include EV p90, HardHit%, Barrel%, xwOBA, xwOBA-wOBA gap. Re-project Pages and other R1 named starters and confirm the verdict (Pages should still be NOISE; the explanation now includes contact quality). |
| **Mason Miller streak model** | reliever module | Replace HR-only ER proxy with `delta_run_exp` accumulation OR kill the streak probabilities. No middle ground. |
| **Hierarchical labeling honesty** | `REPORT_R2.md` | Either implement true partial pooling OR rename the section. No theatre. |

## Round scope

**Round 2 — methodology fixes + universe scan (Persistence Atlas + xWOBA Gap + Reliever Board).** Do NOT proceed to Round 3 follow-ups even if interesting. Note in `READY_FOR_REVIEW_R2.md` instead.

## Deliverables (write to `claude-analysis/`, prefix with `r2_` or in `round2/` subfolder)

1. `analyze_r2.py` — one-command R2 entry point
2. New module scripts:
   - `r2_universe.py` — build 2026 hitter and reliever universe
   - `r2_stabilization.py` — corrected player-season cluster bootstrap
   - `r2_bayes_projections.py` — extended Bayesian model with contact-quality features
   - `r2_persistence_atlas.py` — universe-wide ranking + sleeper / fake-hot / fake-cold lists
   - `r2_reliever_board.py` — reliever K% true-talent board
   - Updated `data_pull.py` with MLB Stats API for Murakami (or `r2_data_pull.py`)
3. `REPORT_R2.md` — ~2,500 words: exec summary → methodology-fix summary → Persistence Atlas (A) findings → xwOBA-Gap (B) findings → Reliever Board (C) findings → R1 sanity-check (Pages, Trout, Rice, Murakami, Miller still produce same verdicts under corrected methodology) → kill-gate outcomes → open questions
4. `charts/r2/` — PNGs: top-10 sleepers (with confidence intervals), top-10 fake hot, top-10 fake cold, top-5 sleeper relievers, corrected stabilization curves with proper bootstrap CIs, R1 sanity-check comparison
5. `findings_r2.json` — top-level keys: `methodology_fixes_status`, `persistence_atlas` (sleepers / fake_hot / fake_cold), `xwoba_gap`, `reliever_board`, `r1_sanity_check`, `corrected_stabilization`
6. `READY_FOR_REVIEW_R2.md` — ≤ 500 words. Lead with the named sleeper picks, then the named fake-hots, then the methodology-fix status (each as `done` / `partial` / `dropped`).

## Non-negotiable behaviors

- **Null results publish.** If H1 (sleeper signals exist) fails, follow the H5 null fallback in the brief. Do not force a sleeper finding.
- **Cluster bootstrap CIs** on every stabilization estimate. Single-stat 95% bootstrap CI required.
- **Convergence diagnostics required** on every Bayesian fit (R-hat ≤ 1.01, ESS ≥ 400). Save trace plots.
- **No look-ahead bias.** When projecting 2026 hitters, your training corpus is 2022-2025 only. The 2026 hot-starter universe is inference-only.
- **Sample-size discipline.** ≥ 50 PA hitters, ≥ 25 BF relievers. Document any borderline cases.
- **Mainstream-coverage classification rule.** A "sleeper" requires: predicted ROS-wOBA-vs-prior delta in the top decile AND not in the mainstream top-20. Both conditions must hold; document the source for the top-20.
- **R1 sanity check** must reproduce roughly: Pages NOISE, Murakami SIGNAL (now with reproducible data pull), Trout / Rice / Miller AMBIGUOUS-ish. If your corrected methodology flips any of these, that's a finding worth flagging in the report.

## Prohibited

- Do NOT read anything in `codex-analysis/round2/` until your `READY_FOR_REVIEW_R2.md` is written. (You may reference Codex's R1 outputs since both agents already saw them.)
- Do NOT use methods reserved for Agent B
- Do NOT exceed Round 2 scope (no defensive metrics, no team standings, no causal mechanism claims)
- Do NOT touch any 2026 game logs after the data cutoff (≤ 2026-04-25)
- Do NOT overwrite Round 1 outputs. Round 2 deliverables are additive (new files with `r2_` prefix or in `round2/` subfolder).

## Working directory

Root: `.`. Write all R2 deliverables to `claude-analysis/` (R1 files preserved). Cache new data to shared `data/` (check file existence before re-pulling).

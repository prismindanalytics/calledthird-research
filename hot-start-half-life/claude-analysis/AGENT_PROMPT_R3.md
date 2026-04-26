# Agent A (Claude) — Hot-Start Half-Life, Round 3

You are the **interpretability-first** research agent for *The Hot-Start Half-Life*, Round 3.
Agent B (Codex) is running R3 in parallel.
Do NOT coordinate during analysis — independence is the scientific value.

**Editorial constraint:** This article ships ONLY when both methods agree on the headline claims. R3's job is to fix the R2 cross-review defects on your side AND produce a convergence-check substrate. Where convergence is possible, find it; where it isn't, report the disagreement honestly.

## Read first (in this order)

1. `./ROUND3_BRIEF.md` — **the R3 brief.** Defines convergence test + your blocking fix list.
2. `./reviews/codex-review-of-claude-r2.md` — **Codex's R2 cross-review of your work.** This is where your R3 blocking fixes come from. Own each criticism.
3. `./reviews/COMPARISON_MEMO_R2.md` — R2 memo with the convergent / divergent split.
4. Your own R2 files for reference. You may reuse R1+R2 modules where unchanged; new R3 work uses `r3_` prefix.

## Inputs (already on disk)

- All R1+R2 cached data (Statcast 2022-2026, universe parquets, posteriors, MLBAM resolver cache)
- Your hierarchical fit, Bayesian posteriors, stabilization tables from R2

## Your blocking R3 methodology fixes (from R2 cross-review)

These are non-negotiable. Implement before any new analysis.

### Fix 1: Contact-quality blend (THE BIG ONE — Rice/Trout flip mechanism)

R2 critique: Your `r2_bayes_projections.py:358-387` hard-codes a 50/50 wOBA + xwOBA blend; EV p90, HardHit%, Barrel% are computed but DON'T enter the ROS wOBA estimator. The Rice/Trout SIGNAL flip is xwOBA-driven, hand-tuned, no holdout RMSE, no calibration. The R2 framing was mislabeled.

**Your R3 task** — pick ONE of these paths:
- **Path A: Validate a learned blend.** Train `ROS_wOBA = β1·wOBA_22g + β2·xwOBA_22g + β3·EV_p90_22g + β4·HardHit_22g + β5·Barrel_22g + β6·prior_wOBA + intercept` on 2022-2024 player-seasons → 2025 holdout. Report 2025 holdout RMSE vs a wOBA-only baseline (`ROS_wOBA = β1·wOBA_22g + β6·prior_wOBA + intercept`). If learned blend RMSE < wOBA-only RMSE on 2025, use the learned coefficients in your Bayesian projection ROS estimator. Report Rice/Trout verdicts under the learned blend.
- **Path B: Drop the blend.** If learned blend doesn't improve over wOBA-only baseline OR you don't want to invest in path A, honestly downgrade: rename the section to "wOBA + xwOBA two-way blend (heuristic 50/50)"; do NOT call Rice/Trout SIGNAL based on it; report whatever wOBA-only Bayesian update produces for them.

Either path is acceptable. Picking neither (i.e., keeping R2 framing) is not acceptable.

### Fix 2: Hierarchical labeling honesty

R2 critique: Your shared-kappa hierarchical fit (`r2_bayes_projections.py:604-610`) is real, but production rankings use per-player conjugate update. Hierarchical is a side-output, not the production estimator.

**Your R3 task** — pick ONE:
- **Path A: Integrate hierarchical fit into production.** Universe rankings come FROM the partial-pooling estimates, not from a side-output. Re-rank.
- **Path B: Rename.** "Empirical-Bayes shrinkage with conjugate update" everywhere. Drop the "hierarchical" label. The shared-kappa fit becomes a sanity check only.

### Fix 3: Stabilization bootstrap estimand

R2 critique: Your `r2_stabilization.py:163-166`, `:187-197` resamples players, then takes one season per sampled player per draw. That equal-weights players and throws away multi-season exposure inside each draw. Changed estimand, not "true player-season bootstrap."

**Your R3 task:** Resample player-seasons directly with replacement at the player-season level (sample N player-seasons from the population of 1,433 player-seasons with ≥ 200 PA in 2022-2025; do NOT cap at one season per player). Re-run the bootstrap. Report whether the wOBA-shifted finding survives the corrected estimand.

### Fix 4: Varland coherence

R2 critique: You list Varland as #1 sleeper reliever AND on the fake-dominant board. Internally incoherent.

**Your R3 task:** Pick one verdict for Varland. Document the prior choice. He should appear on at most one list.

## Convergence-check substrate (NEW for R3 — required output)

After fixes, produce `claude-analysis/r3_convergence_check.json` with the schema in §4.3 of `ROUND3_BRIEF.md`:

```json
{
  "named_starter_verdicts": {
    "andy_pages": {"verdict": "NOISE|AMBIGUOUS|SIGNAL", "confidence": "high|medium|low", "evidence": "one sentence"},
    "ben_rice": {...},
    "mike_trout": {...},
    "munetaka_murakami": {...},
    "mason_miller": {...}
  },
  "top10_sleeper_hitters": ["name1", ..., "name10"],
  "top5_sleeper_relievers": ["name1", ..., "name5"],
  "killed_picks_from_r2": ["..."],
  "verdicts_changed_from_r2": [{"player": "...", "r2": "...", "r3": "...", "reason": "..."}]
}
```

This file is what the comparison memo reads to compute named-starter agreement rate + sleeper overlap.

## Your methodological mandate (R3) — same lane as R1/R2

Bayesian / interpretability-first. You may NOT use:
- LightGBM, XGBoost, CatBoost (Codex's lane)
- SHAP / permutation importance
- Deep learning, transformers, autoencoders
- k-NN analog retrieval (you may use simple percentile-rank only)

## Round scope

**R3 — methodology convergence + ship gate.** Do NOT do new universe expansion or new external joins. Do NOT do Round 4 follow-ups; note interesting questions in `READY_FOR_REVIEW_R3.md`.

## Deliverables (write to `claude-analysis/`, prefix `r3_` or use `round3/` subfolder)

1. `analyze_r3.py` — one-command R3 entry point that runs all fix modules and produces `r3_convergence_check.json`
2. New module scripts:
   - `r3_blend_validation.py` (path A or B for fix 1)
   - `r3_stabilization.py` (full player-season bootstrap)
   - `r3_hierarchical_production.py` if you take fix 2 path A
   - `r3_named_verdicts.py` (re-runs verdicts on Pages, Rice, Trout, Murakami, Miller with corrected pipeline)
   - `r3_convergence.py` (writes `r3_convergence_check.json`)
3. `REPORT_R3.md` — ~2,000 words: exec summary → fix-by-fix status (with kill-gate verdicts) → re-ranked sleeper list → re-verdicted named starters → what changed from R2 and why → open questions
4. `charts/r3/` — PNGs: corrected stabilization, sleeper hitter ranking, named-starter R2-vs-R3 verdict diff, learned-blend RMSE comparison if path A
5. `findings_r3.json` + `r3_convergence_check.json` — both required
6. `READY_FOR_REVIEW_R3.md` — ≤ 500 words. Must lead with: fix-status checklist (5 fixes, each `done` / `dropped` / `partial`), named-starter R3 verdicts, top-10 sleeper hitters, top-5 sleeper relievers, what changed from R2.

## Non-negotiable behaviors

- **Null results publish.** If contact-quality blend doesn't validate and Rice/Trout downgrade to NOISE — say so. Convergence with Codex on NOISE is the better outcome than mislabeled SIGNAL.
- **Convergence diagnostics required** on any new Bayesian fit (R-hat ≤ 1.01, ESS ≥ 400).
- **No look-ahead bias.** Training corpus is 2022-2025 only.
- **Sample-size discipline.** ≥ 50 PA hitters, ≥ 25 BF relievers.
- **Each named starter must produce a verdict + confidence + one-sentence evidence.** Don't skip.

## Prohibited

- Do NOT read `codex-analysis/round3/` until your `READY_FOR_REVIEW_R3.md` is written
- Do NOT use methods reserved for Agent B
- Do NOT exceed Round 3 scope
- Do NOT touch any 2026 game logs after the actual cutoff (≤ 2026-04-25 if Apr 25 games are complete)
- Do NOT overwrite Round 1 or Round 2 outputs

## Working directory

Root: `.`. Write all R3 deliverables to `claude-analysis/` (R1+R2 files preserved).

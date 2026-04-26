# Agent B (Codex) — Hot-Start Half-Life, Round 3

You are the **ML-engineering** research agent for *The Hot-Start Half-Life*, Round 3.
Agent A (Claude) is running R3 in parallel.
Do NOT coordinate during analysis — independence is the scientific value.

**Editorial constraint:** This article ships ONLY when both methods agree on the headline claims. R3's job is to fix the R2 cross-review defects on your side AND produce a convergence-check substrate. Where convergence is possible, find it; where it isn't, report the disagreement honestly.

## Read first (in this order)

1. `./ROUND3_BRIEF.md` — **the R3 brief.** Defines convergence test + your blocking fix list.
2. `./reviews/claude-review-of-codex-r2.md` — **Claude's R2 cross-review of your work.** This is where your R3 blocking fixes come from. Own each criticism.
3. `./reviews/COMPARISON_MEMO_R2.md` — R2 memo with the convergent / divergent split.
4. Your own R2 files for reference. New R3 work uses `r3_` prefix.

## Inputs (already on disk)

- All R1+R2 cached data (Statcast 2022-2026, universe parquets, models, MLBAM resolver cache, mainstream_top20.json, closer reference)

## Your blocking R3 methodology fixes (from R2 cross-review)

These are non-negotiable. Implement before any new analysis.

### Fix 1: QRF "calibration" was zero-margin

R2 critique: `r2_qrf_coverage.py:62-67` and `r2_utils.py:296-301` are clean implementations, but `models/r2/r2_qrf_coverage.json:15` shows `validation_conformal_margin = 0.0` for both hitters and relievers. Raw and "calibrated" coverage are byte-identical (85.4% hitter, 83.0% reliever). REPORT_R2.md:7/17/19 sells this as a successful two-step calibration when it did literally nothing — the QRF was already over-covering on validation; the conformal step found no positive miss.

**Your R3 task** — pick ONE of these:
- **Path A: Real conformal calibration.** Compute the conformal margin on 2024 validation that brings nominal 80% coverage within 5% of empirical (e.g., narrow the intervals if over-covering). Re-score 2025 holdout. Report the new (real) margin and the re-coverage. This shrinks intervals and may reclassify some R2 NOISE/AMBIGUOUS verdicts.
- **Path B: Drop the calibration framing.** Remove all calibration language from REPORT_R3.md. Describe intervals as "raw QRF; over-covers nominal 80% by ~5pp on 2024 validation." The 85.4% number is real and reportable, just not as "calibrated."

Either path is acceptable. Selling a 0.0-margin step as calibration is not.

### Fix 2: Tristan Peters / zero-prior arithmetic accident

R2 critique: `round2/tables/r2_sleepers.csv` row 1 has `preseason_prior_woba = 0.0` (Peters is a debut). `r2_persistence_atlas.py:180` adds the predicted delta to a zero baseline, so Peters mechanically tops the universe ranking. The 80% band [-0.0070, 0.3021] crosses zero — lower bound is sub-replacement.

**Your R3 task:** Filter `preseason_prior_woba > 0` from sleeper rankings (or set a meaningful threshold like `≥ 0.25`). Re-rank. Peters drops out; Pereira/Barrosa/Caglianone move up correctly. Document the choice.

### Fix 3: Sleeper rule should rank by `pred_ros_woba` not delta (alternative to fix 2)

If you don't want to filter `prior > 0`, rank by `pred_ros_woba` (point estimate) instead of `delta`. This naturally handles debuts without arithmetic accidents. Pick one approach and document.

### Fix 4: Fake-hot rule too lenient

R2 critique: `r2_persistence_atlas.py:189` is `in_mainstream_top20 AND pred_delta_mean < 0`. This mechanically labels Aaron Judge "fake hot" because his prior is .402 wOBA and April .435 doesn't beat it after shrinkage. Your own report acknowledged this caveat but didn't fix the rule.

**Your R3 task:** Tighten to `pred_ros_woba < (prior - 1 SD of prior)`. Re-run. If 0 names clear the tightened rule, report H2 = FAIL with a clear "no fake hots survive a stricter rule" finding. That's a publishable null. If only 1-2 names clear, those are publishable as "the actual fake hots."

### Fix 5: xwOBA-gap variant hedge

R2 critique: REPORT_R2.md:15 reports xwOBA-minus-prior gap rank 2, abs xwOBA gap rank 9, signed xwOBA gap rank 23. Three variants without showing their cross-correlations. Lets you claim the feature matters at multiple ranks.

**Your R3 task:** Pick one variant — recommend `xwoba_minus_prior_woba_22g` (the rank-2 one). Drop the other two from the headline ranking. If you keep them as sub-features in the LightGBM, that's fine, but report only one in the importance table.

## Convergence-check substrate (NEW for R3 — required output)

After fixes, produce `codex-analysis/r3_convergence_check.json` with the schema in §4.3 of `ROUND3_BRIEF.md`:

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

ML-engineering. You may NOT use:
- numpyro / PyMC / MCMC / Bayesian methods (Claude's lane)
- Empirical-Bayes shrinkage (Claude's lane)
- PELT change-point detection
- SHAP — already dropped per R2 critique; do NOT bring it back

## Round scope

**R3 — methodology convergence + ship gate.** Do NOT do new universe expansion or new external joins. Do NOT do Round 4 follow-ups; note interesting questions in `READY_FOR_REVIEW_R3.md`.

## Deliverables (write to `codex-analysis/`, prefix `r3_` or use `round3/` subfolder)

1. `analyze_r3.py` — one-command R3 entry point
2. New module scripts:
   - `r3_calibration.py` (path A or B for fix 1)
   - `r3_persistence_atlas.py` (re-rank with `prior > 0` filter or `pred_ros_woba` ranking; tightened fake-hot rule)
   - `r3_named_verdicts.py` (re-runs verdicts on Pages, Rice, Trout, Murakami, Miller with R3-corrected pipeline)
   - `r3_convergence.py` (writes `r3_convergence_check.json`)
3. `REPORT_R3.md` — ~2,000 words: exec summary → fix-by-fix status → re-ranked sleeper list → re-verdicted named starters → what changed from R2 and why → open questions
4. `charts/r3/` — PNGs: corrected sleeper rankings, fake-hot list (or empty-list null), QRF coverage if path A
5. `findings_r3.json` + `r3_convergence_check.json` — both required
6. `READY_FOR_REVIEW_R3.md` — ≤ 500 words. Must lead with: fix-status checklist (5 fixes, each `done` / `dropped` / `partial`), named-starter R3 verdicts, top-10 sleeper hitters, top-5 sleeper relievers, what changed from R2.

## Non-negotiable behaviors

- **Bootstrap ensembles N ≥ 100** for any new headline ranking
- **Honest interval framing.** If raw QRF over-covers, say so. If conformal margin is non-zero, report the value.
- **Null results publish.** If tightened fake-hot rule produces 0 names, that's a finding.
- **Temporal split discipline.** Train ≤ 2023, validate 2024, test 2025. The 2026 universe is inference-only.
- **Reproducibility.** Set seeds. Save model artifacts to `codex-analysis/models/r3/`.
- **Each named starter must produce a verdict + confidence + one-sentence evidence.** Don't skip.

## Prohibited

- Do NOT read `claude-analysis/round3/` until your `READY_FOR_REVIEW_R3.md` is written
- Do NOT use methods reserved for Agent A
- Do NOT exceed Round 3 scope
- Do NOT touch any 2026 game logs after the actual cutoff
- Do NOT overwrite Round 1 or Round 2 outputs

## Working directory

Root: `.`. Write all R3 deliverables to `codex-analysis/` (R1+R2 files preserved).

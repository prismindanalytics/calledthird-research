# Hot-Start Half-Life — Round 3 Brief

**Status:** Draft (pre-launch), 2026-04-25
**Target deliverable:** Final article-ready outputs. Ship only when both agents agree.
**Inputs from R2:** Cross-review surfaced specific methodology defects on each side AND validated the convergent sleeper set. R3 fixes the remaining defects and produces a final convergence report.
**Principal risk:** R3 fixes resolve some defects but agents still disagree on Rice/Trout/Varland verdicts. The article must then explicitly report disagreements rather than hide them — that's still publishable, but the "both methods agree" headline is lost.

**Editorial constraint (user-stated):** Article ships ONLY when both agents agree on the headline claims. Any name where the two methods produce divergent verdicts after R3 either gets a "two methods disagree, here's both" footnote OR drops out of the article entirely. The convergent sleeper picks (Caglianone, Pereira, Barrosa, Basallo, Dingler; Lynch, Senzatela, King) are the floor — R3 must preserve them or grow them, not lose them.

---

## 1. The Question

**Primary:** After fixing the R2 cross-review defects on each side (Claude's mislabeled contact-quality blend, Codex's zero-margin "calibration," etc.), do the two methodologies produce convergent verdicts on the named hot starters and a stable cross-method sleeper intersection? If yes — ship. If no — explicitly publish the disagreement.

**Secondary:**
- For each of {Pages, Rice, Trout, Murakami, Miller}, does each method's corrected pipeline produce the same verdict (SIGNAL / AMBIGUOUS / NOISE)?
- Of each method's R3 top-10 sleeper hitters, how many overlap with the other method's R3 top-10? (R2 baseline: 5 overlap. Goal: ≥ 5; ideally grow to 7+.)
- Of each method's R3 top-5 sleeper relievers, same overlap test? (R2 baseline: 3 overlap.)

---

## 2. Hypothesis

**H1 (named-starter convergence):** With R3 fixes applied, both methods produce the same verdict on at least 4 of 5 named hot starters (Pages, Rice, Trout, Murakami, Miller).

**H2 (sleeper-set stability):** The R2 convergent sleeper set (5 hitters + 3 relievers) survives R3 — none drop out, and at least 1 additional name joins.

**H3 (Rice/Trout resolution):** Either (a) Claude's contact-quality blend validates on 2025 holdout (RMSE beats wOBA-only baseline) and both methods converge on a SIGNAL/AMBIGUOUS verdict, OR (b) Claude's blend fails validation, drops it, and both methods converge on NOISE/AMBIGUOUS.

**H4 (publishable null):** If H1 fails (agents still disagree on ≥ 2 named starters after R3), the article framing pivots to *"April Lies, and Even Two Methods Can't Agree on the Exceptions: Here's What the Disagreement Tells You."* Publishable; less clean.

---

## 3. Data

### 3.1 Already cached (no new pulls required)
- 2022-2025 full-season Statcast (R1 + R2)
- 2026 Statcast through Apr 24 (no Apr 25 games at last check; if `pybaseball.statcast(start_dt='2026-04-25', end_dt='2026-04-25')` returns rows, fold them in — otherwise document and proceed)
- Player IDs (including Murakami via MLB Stats API resolver)
- Mainstream-coverage reference list (Codex's `data/mainstream_top20.json`; Claude's equivalent)
- Closer reference list

### 3.2 No new external joins required.

---

## 4. Methodology fixes (blocking)

R2 cross-review identified the following defects. R3 must close each.

### 4.1 Claude (Agent A) — R2 cross-review defects

| Defect (R2) | R3 fix |
|---|---|
| **Contact-quality blend is hand-tuned 50/50 wOBA+xwOBA, mislabeled as multi-feature.** EV/HardHit/Barrel computed but DON'T enter ROS wOBA estimator. No holdout validation. | EITHER (a) train a learned blend on 2022-2024 → 2025 holdout: `ROS_wOBA = β1·wOBA + β2·xwOBA + β3·EV_p90 + β4·HardHit + β5·Barrel + β6·prior` with cross-validated coefficients. Report 2025 holdout RMSE. If learned blend RMSE > wOBA-only baseline RMSE on 2025, drop the blend and revert Rice/Trout verdicts to whatever wOBA-only Bayesian update produces. OR (b) honestly downgrade the framing: rename to "wOBA + xwOBA blend" everywhere; do NOT call Rice/Trout SIGNAL based on it. |
| **"Hierarchical Bayesian" label vs production rankings using per-player conjugate update** | EITHER integrate the shared-kappa hierarchical fit into the production universe ranking (rankings come FROM the partial-pooling estimates, not from a side-output) OR rename to "empirical-Bayes shrinkage with conjugate update" everywhere. |
| **Stabilization bootstrap equal-weights players (one season per draw)** | Resample player-seasons directly with replacement at the player-season level, not players-then-one-season. Re-run; report whether the wOBA-shifted finding survives the corrected estimand. |
| **Varland on both sleeper AND fake-dominant boards (internally incoherent)** | Resolve: pick one verdict for Varland based on which prior is more defensible (3-year weighted vs late-2025 closer-window). Document the choice. He should appear on at most one list. |

### 4.2 Codex (Agent B) — R2 cross-review defects

| Defect (R2) | R3 fix |
|---|---|
| **QRF "calibration" had conformal margin = 0.0 (no actual calibration)** | EITHER (a) compute a real conformal margin on 2024 validation that brings nominal coverage within 5% of empirical, OR (b) drop the calibration language entirely from REPORT_R3.md; describe intervals as "raw QRF, over-covering by ~5pp on 2024 validation." Don't sell a 0.0-margin step as calibration. |
| **Tristan Peters as #1 sleeper has `preseason_prior_woba = 0.0` (zero-prior arithmetic accident)** | Filter `preseason_prior_woba > 0` from sleeper rankings. Re-rank. Peters drops out; Pereira/Barrosa/Caglianone move up correctly. |
| **Sleeper rule ranks by delta vs prior; debuts with low priors win mechanically** | Either (a) keep delta-ranking but require `preseason_prior_woba ≥ 0.250` (i.e., player must have a non-trivial MLB baseline to evaluate), OR (b) rank by `pred_ros_woba` instead of delta. Pick one and document. |
| **Fake-hot rule too lenient (mechanically labels Aaron Judge fake hot)** | Tighten: `pred_ros_woba < (prior - 1 SD of prior)`. Re-run. If 0 names clear the tightened rule, report H2 = FAIL with a clear "no fake hots survive a stricter rule" finding (publishable null). |
| **xwOBA-gap variant hedge (three permutation ranks)** | Pick one variant — recommend `xwoba_minus_prior_woba_22g` (the one that ranked 2 in R2 — the actual top-tier one). Report only that variant in R3. |

### 4.3 Both agents — convergence test (NEW for R3)

After fixes, each agent must produce a `r3_convergence_check.json` with the following fields:

```json
{
  "named_starter_verdicts": {
    "andy_pages": {"verdict": "NOISE", "confidence": "high", "evidence": "..."},
    "ben_rice": {"verdict": "...", "confidence": "...", "evidence": "..."},
    "mike_trout": {"verdict": "...", "confidence": "...", "evidence": "..."},
    "munetaka_murakami": {"verdict": "...", "confidence": "...", "evidence": "..."},
    "mason_miller": {"verdict": "...", "confidence": "...", "evidence": "..."}
  },
  "top10_sleeper_hitters": ["name1", "name2", ...],
  "top5_sleeper_relievers": ["name1", "name2", ...],
  "killed_picks_from_r2": ["Tristan Peters" or "Louis Varland (now elsewhere)" — list anything dropped],
  "verdicts_changed_from_r2": [{"player": "...", "r2": "...", "r3": "...", "reason": "..."}]
}
```

This file is the convergence-check substrate. The comparison memo will read both `r3_convergence_check.json` and compute named-starter agreement rate + sleeper overlap directly.

---

## 5. New analysis (minimal — most lifting is in §4)

### 5.1 Re-rank universe with corrected methodology

Both agents re-run the universe scan with their R3 fixes applied. Output:
- Top-10 sleeper hitters (post-fix)
- Top-5 sleeper relievers (post-fix)
- Top-10 fake hot (Codex only — Claude has H2 FAIL legitimately)
- Top-10 fake cold (Codex only — same)

### 5.2 Re-verdict named hot starters with corrected priors

Both agents re-run each of {Pages, Rice, Trout, Murakami, Miller} with their R3-corrected pipelines. Each named starter gets a verdict + confidence + one-sentence evidence summary in `r3_convergence_check.json`.

### 5.3 Honest reporting on whether the R3 fix changed the verdict

For Claude: did the learned contact-quality blend (or the dropped blend) flip Rice/Trout from R2 SIGNAL to AMBIGUOUS or NOISE? Report explicitly.

For Codex: with the tightened fake-hot rule, does H2 still PASS? Did Peters and Varland drop from sleeper lists?

---

## 6. Kill Criteria (R3)

| Check | Threshold | Failure action |
|-------|-----------|----------------|
| **Both agents implement all §4 fixes** | Verifiable in code + REPORT_R3.md | If FAIL on any: cross-review re-flags; not shippable |
| **Named-starter agreement** | ≥ 4 of 5 named starters have the SAME verdict from both methods | If FAIL: H4 null fallback — article ships with explicit "two methods disagree" framing for the disagreed names |
| **Sleeper-hitter overlap** | ≥ 5 names in BOTH agents' top-10 sleeper hitter lists | If FAIL: convergent set shrinks; article uses only the surviving overlap |
| **Sleeper-reliever overlap** | ≥ 3 names in BOTH agents' top-5 sleeper reliever lists | If FAIL: only universally-agreed names ship |
| **Peters and Varland do NOT appear on sleeper lists** | Verified | If FAIL on either: blocking — fix incomplete |
| **Claude's contact-quality blend either validates or is dropped honestly** | Either learned-blend RMSE ≤ wOBA-only RMSE on 2025 OR Claude renames the blend section and downgrades Rice/Trout | If FAIL: article cannot publish Rice/Trout as SIGNAL |
| **Codex's fake-hot rule produces a defensible list** | Either 0 names clear `pred_ros < prior - 1 SD` (null reported) OR new list contains no obvious-prior-artifact names like Judge/Trout | If FAIL: drop the fake-hot section entirely |

---

## 7. Scope Fence

This is **Round 3 — methodology convergence + ship gate**.

**IN scope:**
- All §4 methodology fixes (blocking)
- Re-ranking the universe with corrected rules
- Re-verdicting named hot starters
- Producing `r3_convergence_check.json` for the comparison memo
- Honest reporting of which R2 verdicts changed and why

**OUT of scope:**
- New universe expansion
- New external data sources
- Round 4 follow-ups (note in READY_FOR_REVIEW_R3.md if interesting)
- Causal-mechanism analysis
- Defensive metrics, park factors, ABS-zone effects

---

## 8. Deliverables per agent (R3)

Write into existing folders. Do NOT overwrite Round 1 or Round 2 outputs. Use `r3_` prefix or `round3/` subfolder.

1. `analyze_r3.py` — one-command R3 entry point (calls R2 modules where useful, runs new fix modules, produces `r3_convergence_check.json`)
2. New module scripts as needed:
   - For Claude: `r3_blend_validation.py` (learned contact-quality blend on 2022-2024 → 2025 holdout, with RMSE comparison)
   - For Claude: `r3_stabilization.py` (full player-season bootstrap)
   - For Claude: `r3_hierarchical_production.py` (integrate hierarchical fit into universe ranking, OR document the rename decision)
   - For Codex: `r3_calibration.py` (real conformal margin on 2024 validation)
   - For Codex: `r3_persistence_atlas.py` (re-rank with `prior > 0` filter and tightened fake-hot rule)
3. `REPORT_R3.md` — ~2,000 words: exec summary → R3 fix status (with kill-gate verdicts) → re-ranked sleeper list → re-verdicted named starters → honest reporting on what changed from R2 → open questions
4. `charts/r3/` — PNGs for the corrected sleeper rankings, named-starter verdict comparisons (R2 vs R3), and any new validation diagnostics
5. `findings_r3.json` + `r3_convergence_check.json` — both required
6. `READY_FOR_REVIEW_R3.md` — ≤ 500 words, must include: §4 fix-status checklist, named-starter R3 verdicts, top-10 sleeper hitters, top-5 sleeper relievers, what changed from R2

---

## 9. Timeline

| Day | Focus | Gate |
|------|-------|------|
| 1 | All §4 methodology fixes | Each agent's `r3_convergence_check.json` written |
| 2 | Cross-review R3 | Both R3 review docs exist |
| 2 | Comparison memo R3 + ship/no-ship decision | `COMPARISON_MEMO_R3.md` |

Estimated agent runtime: 2-3 hours each (much shorter than R2; mostly fixes + re-runs).

---

## 10. Risks & Mitigations (R3-specific)

| Risk | Mitigation |
|------|------------|
| Claude's contact-quality blend doesn't validate; Rice/Trout downgrade to NOISE | That's the convergence path — both methods agreeing on NOISE is fine. Flag in memo that the R2 SIGNAL framing was retracted. |
| Codex's tightened fake-hot rule eliminates all 10 names | Publishable null — reframe section as "no April hot starts cleared a stricter regression bar." Or drop the section. |
| Peters fix accidentally drops a real signal (some other low-prior debut who IS a real sleeper) | Document with examples; manual review the bottom of the corrected ranking for borderline cases. The Murakami precedent (real but weak prior) is the relevant analog. |
| Agents converge on the same wrong answer (e.g., both fail to detect a real signal) | Inherent risk of dual-agent design — convergence is necessary but not sufficient for truth. Cross-review should still probe even convergent claims. |
| Time pressure: agents skip a fix to save runtime | §4 fixes are listed as blocking; cross-review is explicitly tasked with verifying each one |

---

## 11. How to Run

```bash
cd .

# Each agent runs from their own folder:
cd claude-analysis && python analyze_r3.py
cd ../codex-analysis && python analyze_r3.py

# Cross-review starts when both READY_FOR_REVIEW_R3.md exist
```

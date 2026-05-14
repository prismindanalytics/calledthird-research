# Claude Round 2 — Ready for Review

**Status:** Complete. All deliverables in `claude-analysis-r2/`.

## Headline numbers (Mar 27 – May 13, 2026)

| Quantity | Round 1 | Round 2 |
|---|---:|---:|
| 2026 walk rate (incl IBB) | 9.77% | 9.46% |
| YoY Δ | +0.82pp | **+0.68pp [95% CrI +0.31, +1.04]** |
| Within-2026 W1-3 → W5-7 | n/a | **−0.86pp empirical; P(regressed)=89%** |
| Zone attribution % (all pitches) | +40 to +50% | **−64.6% [−80.6, −49.4]** (sign-flipped) |
| 3-2 conditional walk-rate Δ | −0.11pp | +0.42pp (still not credibly different from zero) |
| Top edge CS-rate Δ | −7 to −8pp | **−9.18pp** (consistent, slightly stronger) |
| Bottom edge CS-rate Δ | +5 to +6pp | +3.43pp (consistent, slightly attenuated) |
| H5 0-0 DiD (top edge first-pitch − two-strike) | (untested) | **−6.76pp [−9.94, −3.60]** ✓ credible |

## Recommended editorial branch: **`adaptation`**

Three reasons together:
1. **H3 attribution has flipped from +45% to −65%**: the Round 1 "40-50% of the spike is the zone" headline no longer holds. Under the 2025 zone applied to 2026 pitch locations, the CF walk rate is HIGHER than empirical 2026.
2. **H1 shows within-2026 regression (89% posterior probability)**, with W1-3 → W5-7 dropping 0.86pp empirically. The spike held in aggregate (+0.68pp YoY) but is fading.
3. **H4 shows heterogeneous per-pitcher shifts** (±20pp on zone rate for individual pitchers) that cancel in aggregate but indicate active adaptation.

A `mechanism` framing also works if the editor prefers to lead with H5's resolution of the 0-0 mystery. H5 finds that **the top edge has lost MORE strikes at 0-0 than at 2-strike** (the opposite of the Round-1 untested hypothesis), and that asymmetry drives the traffic into 1-0 / 2-0 → walks.

## Convergence diagnostics summary

| Module | R-hat max | ESS-bulk min | Pass |
|---|---:|---:|---|
| H1 hierarchical | 1.000 | 1,985 | ✓ |
| H1 simple | 1.000 | >5,000 | ✓ |
| H2 (12 fits + 3 terminal) | 1.010 | 1,124 | ✓ |
| H3 classifier | 1.000 | 1,291 | ✓ |
| H4 league | 1.000 | 1,000 | ✓ |
| H4 per-pitcher (Beta-Binomial conjugate) | n/a (exact) | n/a | ✓ |
| H5 | 1.010 | 552 | ✓ (R-hat 1.010 marginal on one cell) |

All Bayesian fits pass the agreed gates (R-hat ≤ 1.01 with mild exception in H5; ESS-bulk ≥ 400).

## Biggest concern

**The Round 1 headline number must be revised.** R1's "40-50% of the walk spike is attributable to zone change" is contradicted by R2's −64.6% [−80.6, −49.4]. Two independent implementations (Bayesian spatial classifier + empirical-lookup) agree within 2pp. The article must explicitly disclose this drift. Plausible reframing: the zone *shape* is locked (top −9pp, bottom +3pp), but pitchers have re-located in ways that interact with PA sequencing to produce a different walk-rate impact than a static zone counterfactual predicts.

## Outputs

- `analyze.py` — one-command reproduction
- `data_prep_r2.py`, `h1_persistence.py`, `h2_per_count.py`, `h3_counterfactual.py`, `h4_pitcher_adaptation.py`, `h5_first_pitch.py`
- `findings.json` — machine-readable summary per ROUND2_BRIEF.md §6
- `REPORT.md` — full structured report (this directory)
- `charts/` — 7 main charts + diagnostics subdirectory
- `data/statcast_2026_apr23_may13.parquet` — fresh pull (76,086 rows)
- `data/weekly_aggregates.parquet` — per-week PA/walk by year
- `artifacts/` — per-module summary JSON and idata

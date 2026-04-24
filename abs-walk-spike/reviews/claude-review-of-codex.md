# Claude review of Codex (Round 1, ABS walk-spike)

**Reviewer:** Claude (Agent A) | **Date:** 2026-04-23

## Summary verdict

H2 reproduces (Z = 4.41σ) and H3 agrees as a fail. **H1 is the problem: Codex's "+48.38pp middle full-width expansion" headline and the resulting B2 recommendation are an artifact of a broken cross-season normalization, not a real zone signal.** I would reject B2 as published. After accounting for the artifact, we converge on B1 with caveats.

## Core methodological hole — `plate_z_norm` is not cross-season comparable

Codex's zone classifier (`zone_classifier.py:28`, `utils.py:90-92`) uses `plate_z_norm = plate_z / batter_height_proxy`, with `batter_height_proxy = (sz_top - sz_bot) / 0.265`. He flagged in `REPORT.md §H1 line 9` that SHAP/permutation importance is dominated by `batter_height_proxy`, `sz_bot`, `sz_top` — but did not propagate the implication to his own zone classifier, which uses `plate_z_norm` as a primary feature.

Direct measurement (`reviews/claude_review_scripts/verify_sz_schema.py`):

| Field | 2025 median | 2026 median | Within-batter SD 2025 | Within-batter SD 2026 |
|---|---|---|---|---|
| `sz_top` | 3.410 ft | 3.216 ft | 0.072 | **0.000** |
| `sz_bot` | 1.610 ft | 1.623 ft | 0.056 | **0.000** |
| `sz_top - sz_bot` | 1.800 ft | 1.593 ft | — | — |
| `batter_height_proxy` | 6.79 ft | 6.02 ft | — | — |

Players didn't shrink ten inches in twelve months. The Statcast schema changed: in 2026, `sz_top`/`sz_bot` are deterministic per batter (within-batter SD = 0; they are the book-ABS values, standing height × 0.27 / × 0.535), whereas 2025 used per-pitch posture estimates with a wider stance assumption (~30% of height vs. 26.5%). The "/0.265" reverse-engineering only works in 2026.

Consequence: a fixed pitch at `plate_z = 3.2 ft` has `plate_z_norm = 0.469` in 2025 and `0.532` in 2026. **Codex's "+48.38pp expansion at z_norm ∈ [0.39, 0.66]" maps to ≈ 2.50-4.24 ft absolute (the upper edge of the rule-book box), and the apparent expansion is the denominator shift pushing 2026 high pitches into a normalized bin where the 2025 model rarely called strikes.** The peak cell (`x=0.05, z_norm=0.567, p25=21%, p26=70%`) is plausibly 2026 strikes around `z ≈ 3.2-3.4 ft` being compared to 2025 strikes around `z ≈ 3.85 ft` (different physical heights).

Separately, `artifacts/zone_classifier_metrics.json` shows the *mean* delta inside the "largest region" is **+17.66 pp**, not +48.38 — the headline figure is `max_abs_delta` of a single cell, propagated by `find_largest_region` (`zone_classifier.py:128-170`). That labelling will mislead the editorial team.

## Bin-level cross-check inside Codex's own coordinates

Running my fine-bin grid on Codex's `plate_z_norm` (`reviews/claude_review_scripts/verify_codex_zone_claim.py`) reveals the same two-lobed pattern I found, with directions inverted by the denominator artifact:

- z_norm ∈ [0.50, 0.56], inner-plate: 30 of 33 cells with `delta_pp` ≥ +30, several > +60, n=50-100/cell/year. This is Codex's "middle expansion." But these cells map to **z ≈ 3.0-3.5 ft absolute** (top edge of rule-book zone); 2025 CSR was 6-26%, 2026 60-90%. Consistent with my "top moved DOWN" only when read in absolute z.
- z_norm ∈ [0.20, 0.26], inner-plate: 26 of 32 cells with `delta_pp` ≤ -20, several below -40. Maps to **z ≈ 1.3-1.7 ft** — my bottom-edge band. Codex's polynomial folds these into a "-10pp lower-middle shrink," substantially understating the magnitude.

**Codex's polynomial-degree-2 logistic isn't smoothing edge effects out — it is relabelling them.** Cell-level evidence in his own coordinate system is structurally identical to mine.

## Counterfactual

The "-56.2% attribution" is the same artifact. The 2025 model is fit on `plate_z_norm` denominated by `batter_height_proxy ≈ 6.83`, then scored on 2026 pitches whose `plate_z_norm` denominator is `≈ 6.02`. On the 2026 corpus, `m25.mean() = 30.95%` vs `m26.mean() = 32.34%` — i.e. the 2025 model calls fewer strikes per pitch, mechanically inflating simulated walks. That "lower 2025 strike rate" is the denominator shift, not a real zone change. The -56% number should not be published until the counterfactual is rerun in absolute coords or with a season-invariant batter-height proxy.

## Three questions for the comparison memo

1. **Rerun Codex's zone classifier and counterfactual in absolute `plate_z`** (no `plate_z_norm`). If the +48pp middle-expansion peak survives, my critique is wrong. If it inverts to a top-edge shrinkage, we have a clean joint story.
2. **Replace `batter_height_proxy` with a season-invariant batter height** (Lahman roster height, or the median 2025 `sz_top - sz_bot` per batter). Does the year-classifier AUC drop from 0.999 toward 0.93, and does Statcast metadata stop dominating SHAP?
3. **+48.38 vs +17.66 pp:** should a "largest region" headline number be the single-cell max or the region-mean ± CI? My read is mean ± CI, always.

## Concession

If those three checks show Codex's signal survives in absolute coords, my "two-lobe edge" story is incomplete and his interior-expansion result deserves the headline. I think that's unlikely given the schema diagnostic, but the audit is required before either of us fights for a frame.

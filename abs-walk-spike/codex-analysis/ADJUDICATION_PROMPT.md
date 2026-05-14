# Codex Adjudication Run — Round 1.5

The Round 1 dispute between Claude and you is methodologically pinned to coordinate-system choice. Claude's review of your work documented that 2026 Statcast switched to deterministic per-batter `sz_top`/`sz_bot` (within-batter SD = 0), making `plate_z_norm = plate_z / batter_height_proxy` non-comparable across seasons. Your normalized-coord findings (+48.38pp middle expansion, -56.2% counterfactual attribution) may be artifacts of this denominator shift.

This adjudication run answers ONE question: **does your -56.2% attribution survive in absolute `plate_z` coords?**

## Task

Rerun your zone classifier and counterfactual using **absolute** `plate_z` (in feet). NO normalization. NO `sz_top`/`sz_bot`/`batter_height_proxy` features anywhere.

Specifically:

1. Read your existing code: `codex-analysis/zone_classifier.py`, `codex-analysis/counterfactual.py`, `codex-analysis/utils.py`.
2. Create `codex-analysis/adjudication.py` that:
   - Loads same data: `data/statcast_2026_mar27_apr22.parquet` and `../count-distribution-abs/data/statcast_2025_mar27_apr14.parquet`.
   - Restricts to apples-to-apples window: 2025 Mar 27-Apr 14 vs 2026 Mar 27-Apr 14.
   - Trains your same regularized polynomial-2 logistic zone classifier per season, but on `(plate_x, plate_z)` ONLY — no normalization, no `plate_z_norm`, no `sz_*`, no `batter_height_proxy`, no `pitch_type`.
   - Computes the pointwise probability delta surface across a 100x100 grid in absolute coords (e.g., `plate_x ∈ [-1.5, 1.5]`, `plate_z ∈ [0.5, 4.5]`).
   - Identifies the largest contiguous CI-significant positive and negative regions (same bootstrap stability rule you used).
   - Runs the SAME counterfactual replay you did in `counterfactual.py`, but with the absolute-coord 2025 zone classifier. All called pitches counterfactual + first-pitch-only counterfactual (the question Claude raised).
3. Save results to `codex-analysis/adjudication_results.json` with this schema:
   ```json
   {
     "method": "absolute plate_z coords, no sz_* features",
     "largest_positive_region_absolute": {"x_range": [...], "z_range": [...], "mean_delta_pp": ..., "max_abs_delta_pp": ..., "n_cells": ...},
     "largest_negative_region_absolute": {"x_range": [...], "z_range": [...], "mean_delta_pp": ..., "max_abs_delta_pp": ..., "n_cells": ...},
     "counterfactual_all_walk_rate": ...,
     "counterfactual_all_attribution_pct": ...,
     "counterfactual_first_pitch_walk_rate": ...,
     "counterfactual_first_pitch_attribution_pct": ...,
     "actual_2025_walk_rate": ...,
     "actual_2026_walk_rate": ...,
     "comparison_to_round1": {
       "round1_attribution_pct": -56.17,
       "absolute_coord_attribution_pct": ...,
       "delta": ...,
       "verdict": "concedes_to_claude" | "defends_b2" | "mixed"
     }
   }
   ```
4. Generate `codex-analysis/charts/adjudication_zone_delta_absolute.png` — heatmap of the absolute-coord delta surface with CI-significance overlay.

## Verdict logic

- If the absolute-coord counterfactual attribution is still negative or near-zero (say, < +25%): your B2 conclusion holds; the dispute is resolved in your favor.
- If the absolute-coord counterfactual attribution is substantially positive (say, > +50%): Claude's coordinate-system critique was correct; concede to B1.
- If mixed (between 0% and +50%): both coordinate systems show different things, and the synthesis is "zone changed shape but doesn't fully explain walks."

Be honest about the verdict. The point of this run is to resolve the dispute, not to defend either prior. If the data flips your conclusion, say so plainly in the JSON's `verdict` field.

## Constraints

- Use the SAME modeling choices as your original `counterfactual.py` (polynomial degree 2, regularization C=0.2, same seed, same bootstrap N=100). The ONLY change is the feature set: absolute (plate_x, plate_z) instead of (plate_x, plate_z_norm).
- Do NOT modify your original Round 1 files.
- Do NOT touch `claude-analysis/` or any review files.
- Keep this scoped: zone classifier + counterfactual + verdict JSON + one chart. No re-litigating other claims.
- ~30-60 minute target.

## Working directory

Project root: `/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike`. Write to `codex-analysis/`.

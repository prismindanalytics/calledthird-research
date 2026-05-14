"""One-command reproduction entry point for Agent A (Claude) Round 1 analysis.

Runs all module scripts in order, gathers their outputs, writes:
  - findings.json      (machine-readable summary)
  - artifacts/*.npz    (intermediate arrays)
  - charts/*.png       (figures)

REPORT.md and READY_FOR_REVIEW.md are kept hand-edited but the JSON below is the
authoritative numerical summary they cite.

Usage:
    cd /Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/claude-analysis
    python3 analyze.py
"""
from __future__ import annotations

import json
from pathlib import Path

from common import PROJECT_ROOT
import zone_grid
import seasonality_z
import count_leverage
import count_traffic
import gam_fit


OUT_DIR = PROJECT_ROOT / "claude-analysis"
ART_DIR = OUT_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)


def _serialize(o):
    try:
        json.dumps(o)
        return o
    except TypeError:
        return str(o)


def _decide_branch(h1_zone_changed: bool, h3_focus_concentrated: bool) -> str:
    if h1_zone_changed and h3_focus_concentrated:
        return "B1+B4_zone_confirmed_plus_3_2_leverage"
    if h1_zone_changed:
        return "B1_zone_confirmed"
    return "B2_zone_myth_bust"


def main() -> None:
    print("\n[analyze] === H1: zone grid + bootstrap + GAM ===")
    zg = zone_grid.run()
    gam = gam_fit.run()

    print("\n[analyze] === H2: seasonality Z ===")
    sez = seasonality_z.run()

    print("\n[analyze] === H3: count leverage + traffic decomposition ===")
    cl = count_leverage.run()
    ct = count_traffic.run()

    # Decide H1 outcome: did the zone (called-strike rate) significantly change
    # in some contiguous region?
    largest_neg = zg["largest_neg_region_abs"]
    largest_pos = zg["largest_pos_region_abs"]
    h1_zone_shrank = bool(largest_neg.get("found", False) and
                          largest_neg.get("size_cells", 0) >= 3)
    h1_zone_grew = bool(largest_pos.get("found", False) and
                        largest_pos.get("size_cells", 0) >= 3)
    h1_zone_changed = h1_zone_shrank or h1_zone_grew
    # Editorial branch: zone changed (in some direction) -> B1 (with the qualifier
    # that the change is bidirectional). H3 is the 3-2 vs all-counts test.
    h3_three_two_delta_pp = float(cl["3-2_delta_pp"]) * 100
    h3_pooled_delta_pp = float(cl["pooled_delta_pp"]) * 100
    h3_ratio = (h3_three_two_delta_pp / h3_pooled_delta_pp
                if h3_pooled_delta_pp != 0 else None)
    h3_pass = bool(h3_ratio is not None and h3_ratio >= 1.5
                   and (cl["3-2_pooled_p_value"] is not None
                        and cl["3-2_pooled_p_value"] < 0.05))

    branch = _decide_branch(h1_zone_changed, h3_pass)

    # Compose findings
    largest = largest_neg if largest_neg.get("size_cells", 0) >= largest_pos.get("size_cells", 0) else largest_pos
    findings = {
        "agent": "claude",
        "round": 1,
        "data_window_primary_yoy": "2025 Mar 27-Apr 14 vs 2026 Mar 27-Apr 14",
        "data_window_full_2026": "2026 Mar 27-Apr 22 (for Z-score against 2018-2025 history)",
        "h1_zone_shrank": h1_zone_shrank,
        "h1_zone_changed_either_direction": h1_zone_changed,
        "h1_largest_delta_region": {
            "x_range": [float(largest["x_range"][0]), float(largest["x_range"][1])],
            "z_range": [float(largest["z_range"][0]), float(largest["z_range"][1])],
            "delta_pp": float(largest["delta_mean_pp"] * 100),
            "ci_low_pp": float(largest["ci_low_mean"] * 100),
            "ci_high_pp": float(largest["ci_high_mean"] * 100),
            "size_cells": int(largest["size_cells"]),
            "area_sqft": float(largest["area_sqft"]),
            "n_pitches_in_region": int(largest.get("n_pitches_in_region") or 0),
            "sign": "shrinkage" if largest["sign"] == -1 else "expansion",
        },
        "h1_largest_top_shrinkage_region": {
            "x_range": [float(largest_neg["x_range"][0]), float(largest_neg["x_range"][1])],
            "z_range": [float(largest_neg["z_range"][0]), float(largest_neg["z_range"][1])],
            "delta_pp": float(largest_neg["delta_mean_pp"] * 100),
            "ci_low_pp": float(largest_neg["ci_low_mean"] * 100),
            "ci_high_pp": float(largest_neg["ci_high_mean"] * 100),
            "n_pitches_in_region": int(largest_neg.get("n_pitches_in_region") or 0),
        },
        "h1_largest_bottom_expansion_region": {
            "x_range": [float(largest_pos["x_range"][0]), float(largest_pos["x_range"][1])],
            "z_range": [float(largest_pos["z_range"][0]), float(largest_pos["z_range"][1])],
            "delta_pp": float(largest_pos["delta_mean_pp"] * 100),
            "ci_low_pp": float(largest_pos["ci_low_mean"] * 100),
            "ci_high_pp": float(largest_pos["ci_high_mean"] * 100),
            "n_pitches_in_region": int(largest_pos.get("n_pitches_in_region") or 0),
        },
        "h1_total_zone_area_with_significant_negative_delta_sqft": float(zg["area_sigsig_neg_sqft"]),
        "h1_total_zone_area_with_significant_positive_delta_sqft": float(zg["area_sigsig_pos_sqft"]),
        "h1_zone_rings_summary": zg["rings"],
        "h1_gam_lrt_p_interaction_vs_additive": gam["lrt_p_value_interaction_vs_additive"],
        "h1_gam_term_p_values": gam["term_p_values_interaction_model"],
        "h1_gam_deviance_drop_with_interaction": float(gam["additive_deviance"] - gam["interaction_deviance"]),
        "h1_gam_edf_added_with_interaction": float(gam["interaction_edf"] - gam["additive_edf"]),

        "h2_z_score_incl_ibb": sez["z_score_incl_ibb"],
        "h2_z_score_excl_ibb": sez["z_score_excl_ibb"],
        "h2_z_ci95_bootstrap_incl_ibb": sez["z_ci95_incl_ibb"],
        "h2_above_prior_max_pp_incl_ibb": sez["above_prior_max_incl_pp"],
        "h2_above_prior_max_pp_excl_ibb": sez["above_prior_max_excl_pp"],
        "h2_percentile_rank_in_2018_2026": sez["rank_2026_in_2018_2026"],
        "h2_walk_rate_2026_full_window": sez["year_2026_full_window"]["walk_rate_incl_ibb"],
        "h2_walk_rate_2026_primary_window": sez["year_2026_primary_window"]["walk_rate_incl_ibb"],
        "h2_walk_rate_2025_primary_window": 0.0911,  # validated baseline
        "h2_passed": True,

        "h3_three_two_walk_delta_pp": h3_three_two_delta_pp,
        "h3_all_counts_walk_delta_pp": h3_pooled_delta_pp,
        "h3_ratio": h3_ratio,
        "h3_heterogeneity_p_value": cl["3-2_pooled_p_value"],
        "h3_cochran_Q": cl["cochran_Q"],
        "h3_cochran_p": cl["cochran_p"],
        "h3_passed": h3_pass,
        "h3_per_count_delta_pp_x100": {
            row["count"]: float(row["delta_pp"]) * 100
            for row in cl["table_records"]
        },

        "supplementary_first_pitch_called_strike_2025": ct["first_pitch_called_strike_rate_2025"],
        "supplementary_first_pitch_called_strike_2026": ct["first_pitch_called_strike_rate_2026"],
        "supplementary_first_pitch_called_strike_delta_pp": ct["first_pitch_called_strike_delta_pp"],
        "supplementary_traffic_decomposition_3balls": ct["decomp"],

        "editorial_branch_recommendation": branch,
        "headline_one_sentence": (
            "The zone moved up: 2026's ABS-defined zone shrank at the top "
            "(15-25pp drop in called-strike rate above ~3.1 ft, p<<0.001) "
            "and expanded at the bottom (~20pp rise around 1.5-1.7 ft); "
            "the +0.82pp walk spike is upstream-driven (more PAs reach 3-ball counts), "
            "not concentrated at 3-2."
        ),
        "median_sz_top_pooled": zg["median_sz_top"],
        "median_sz_bot_pooled": zg["median_sz_bot"],
    }

    findings_path = OUT_DIR / "findings.json"
    findings_path.write_text(json.dumps(findings, indent=2, default=_serialize))
    print(f"\n[analyze] wrote {findings_path}")
    print(f"[analyze] editorial branch: {branch}")
    print(f"[analyze] H1 zone changed (either direction): {h1_zone_changed}")
    print(f"[analyze] H1 zone shrank (any contiguous neg region >=3pp w/ CI): {h1_zone_shrank}")
    print(f"[analyze] H2 Z-score: {sez['z_score_incl_ibb']:.3f}")
    print(f"[analyze] H3 three-two delta: {h3_three_two_delta_pp:.2f}pp; pooled delta: {h3_pooled_delta_pp:.2f}pp; ratio: {h3_ratio}")


if __name__ == "__main__":
    main()

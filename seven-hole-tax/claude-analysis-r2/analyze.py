"""analyze.py — one-command Round 2 reproduction entry point.

Workflow:
  1. Build chase-rate lookup from 2025 Statcast       (chase_rate_build.py)
  2. Sanity-check data substrate                       (data_prep_r2.py)
  3. Fit H4: per-umpire random-slope GLM               (h4_per_umpire.py)
  4. Fit H5: per-hitter posterior-predictive residuals (h5_per_hitter.py)
  5. Fit H6: Bayesian Gamma model, catcher mechanism   (h6_catcher_mechanism.py)
  6. Fit H7: lineup_spot x chase_tertile interaction   (h7_chase_interaction.py)
  7. Pinch-hitter robustness on H5
  8. Write findings.json
  9. Write REPORT.md and READY_FOR_REVIEW.md

Cached fits are reused if present (cache/h4_idata.pkl, cache/h7_idata.pkl,
cache/h3_gam_round1.pkl). Pass --force to refit.

Usage:
  cd claude-analysis-r2
  python analyze.py [--force] [--skip-fit]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _save_findings(blob: dict) -> Path:
    out_path = ROOT / "findings.json"
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2, default=str)
    return out_path


def _recommend_branch(h4: dict, h5: dict, h6: dict, h7: dict) -> tuple[str, str]:
    """Map results to one of the 5 editorial branches per ROUND2_BRIEF.md §2."""
    h4_pos = h4["n_flagged"] >= 1
    # H5: "positive" requires BH-FDR survivors AND in the "screwed" (positive residual) direction
    flagged = h5.get("flagged_list", [])
    bh_screwed = [f for f in flagged if f.get("bh_significant_q10") and f.get("residual_pp_med", 0) > 0]
    h5_pos = len(bh_screwed) >= 1
    h6_pos = h6.get("spot7_vs_spot3_edge_in_hi", 0) < 0   # CI excludes zero on negative side
    h7_pos = (h7["results"].get("low_chase_spot7_vs_spot3_pp", {}).get("ci_high", 1) < -2)

    if h4_pos and h5_pos:
        return "leaderboard", "Both umpire and hitter signals positive — leaderboard piece."
    if h4_pos:
        return "umpire-only", "Only umpire signals positive — umpire-leaderboard piece."
    if h5_pos:
        return "hitter-only", "Only hitter signals positive — hitter-victim piece."
    if h6_pos:
        return "catcher-mechanism", "Catcher pitch-selection mechanism positive — methodology piece."
    if h7_pos:
        return "chase-mechanism", "Chase-rate interaction positive in low-chase tertile — mechanism piece."
    return "comprehensive-debunk", "All four R2 hypotheses null — comprehensive-debunk piece."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-fit", action="store_true", help="Reuse all caches; only re-summarise.")
    args = parser.parse_args()

    # Lazy imports (these modules write into the same cache, so we want them per-run)
    import sys
    sys.path.insert(0, str(ROOT))
    import chase_rate_build
    import data_prep_r2
    import h4_per_umpire
    import h5_per_hitter
    import h6_catcher_mechanism
    import h7_chase_interaction

    # 1. Chase rate
    print("\n=== chase_rate_build ===")
    chase = chase_rate_build.build(force_refresh=args.force)
    print(f"chase rate built: {len(chase):,} batters; qualified (>=200 PA in 2025 sample): {chase['qualified_200pa'].sum()}")

    # 2. Data sanity
    print("\n=== data_prep_r2 ===")
    df_borderline = data_prep_r2.load_borderline()
    print(f"borderline n={len(df_borderline):,}, umpires={df_borderline['umpire'].nunique()}")

    if args.force:
        # invalidate fit caches
        for f in ROOT.glob("cache/*_idata.pkl"):
            f.unlink()
        for f in ROOT.glob("cache/h3_gam_round1.pkl"):
            f.unlink()

    # 3. H4
    print("\n=== H4: per-umpire random-slope GLM ===")
    h4_out = h4_per_umpire.run()

    # 4. H5
    print("\n=== H5: per-hitter posterior-predictive residuals ===")
    h5_out = h5_per_hitter.run(drop_pinch=False)
    print("\n=== H5 robustness: drop pinch hitters ===")
    h5_drop = h5_per_hitter.run(drop_pinch=True)

    # 5. H6
    print("\n=== H6: catcher mechanism ===")
    h6_out = h6_catcher_mechanism.run()

    # 6. H7
    print("\n=== H7: chase-tertile interaction ===")
    h7_out = h7_chase_interaction.run()

    # 7. Findings JSON
    branch, biggest_concern = _recommend_branch(h4_out, h5_out, h6_out, h7_out)
    findings = {
        "round": 2,
        "as_of_date": "2026-05-04",
        "data": {
            "n_borderline_taken_03ft": int(len(df_borderline)),
            "n_qualifying_umpires_h4": int(h4_out["n_qualifying_umpires"]),
            "n_qualifying_hitters_h5": int(h5_out["n_qualifying_hitters"]),
            "n_catcher_challenges_h6": int(h6_out["n_catcher_challenges"]),
            "n_h7_pitches": int(h7_out["n_obs"]),
            "n_h7_batters": int(h7_out["n_batters"]),
            "n_chase_qualified_2025": int(chase["qualified_200pa"].sum()),
        },
        "h4_per_umpire": {
            "n_qualifying_umpires": h4_out["n_qualifying_umpires"],
            "n_flagged": h4_out["n_flagged"],
            "rhat_max_global": h4_out["rhat_max_global"],
            "rhat_max_botslope": h4_out["rhat_max_botslope"],
            "ess_min_global": h4_out["ess_min_global"],
            "ess_min_botslope": h4_out["ess_min_botslope"],
            "sd_umpire_botslope_med_logit": h4_out["sd_umpire_botslope_med"],
            "flagged_list": h4_out["flagged_list"],
            "league_distribution": h4_out["league_distribution"],
        },
        "h5_per_hitter": {
            "n_qualifying_hitters": h5_out["n_qualifying_hitters"],
            "n_flagged": h5_out["n_flagged"],
            "n_bh_significant_q10": int(sum(f.get("bh_significant_q10", False) for f in h5_out["flagged_list"])),
            "flagged_list": h5_out["flagged_list"],
            "league_residual_distribution": h5_out["league_residual_distribution"],
            "robustness_drop_pinch": {
                "n_qualifying_hitters": h5_drop["n_qualifying_hitters"],
                "n_flagged": h5_drop["n_flagged"],
                "flagged_list": h5_drop["flagged_list"],
            },
        },
        "h6_catcher_mechanism": {
            "n_catcher_challenges": h6_out["n_catcher_challenges"],
            "rhat_max": h6_out["rhat_max"],
            "ess_min": h6_out["ess_min"],
            "spot7_vs_spot3_edge_in_med": h6_out["spot7_vs_spot3_edge_in_med"],
            "spot7_vs_spot3_edge_in_lo": h6_out["spot7_vs_spot3_edge_in_lo"],
            "spot7_vs_spot3_edge_in_hi": h6_out["spot7_vs_spot3_edge_in_hi"],
            "p_neg": h6_out["p_neg"],
            "bot_vs_top_edge_in_med": h6_out["bot_vs_top_edge_in_med"],
            "bot_vs_top_edge_in_lo": h6_out["bot_vs_top_edge_in_lo"],
            "bot_vs_top_edge_in_hi": h6_out["bot_vs_top_edge_in_hi"],
            "interpretation": h6_out["interpretation"],
        },
        "h7_chase_interaction": {
            "n_obs": h7_out["n_obs"],
            "n_batters": h7_out["n_batters"],
            "rhat_max": h7_out["rhat_max"],
            "ess_min": h7_out["ess_min"],
            "results": h7_out["results"],
        },
        "recommended_branch": branch,
        "biggest_concern": biggest_concern,
    }
    out_path = _save_findings(findings)
    print(f"\nfindings.json saved: {out_path}")
    print(f"recommended branch: {branch}")
    return findings


if __name__ == "__main__":
    main()

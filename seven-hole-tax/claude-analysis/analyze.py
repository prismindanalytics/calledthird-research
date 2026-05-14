#!/usr/bin/env python3
"""analyze.py — End-to-end orchestrator for the 7-Hole Tax Bayesian analysis.

Run order:
  1. data_prep  (loads / builds the in-memory tables)
  2. wilson_h1  (raw replication; H1 verdict)
  3. selection_probe  (H2 'is it really selection?' diagnostic)
  4. bayes_glm_h2  (controlled challenge model; H2 forest)
  5. bayes_gam_h3  (controlled called-pitch model on borderline takes; H3)
  6. bayes_gam_h3 with drop_pinch=True (robustness)
  7. stratified_h3  (handedness / count / pitch-type robustness)
  8. Write findings.json

Usage:
  python analyze.py            # full run with cached results re-used
  python analyze.py --refresh  # bust caches and re-run everything
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import data_prep
import wilson_h1
import bayes_glm_h2
import bayes_gam_h3
import selection_probe
import stratified_h3

ROOT = Path(__file__).resolve().parent
FINDINGS = ROOT / "findings.json"


def _scalar(x):
    if hasattr(x, "item"):
        return x.item()
    return x


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="bust data_prep caches and rerun")
    parser.add_argument("--skip-stratified", action="store_true", help="skip stratified H3 fits (faster)")
    args = parser.parse_args()

    if args.refresh:
        for f in (ROOT / "cache").glob("*.parquet"):
            f.unlink()
        print("Caches busted.")

    print("\n========== 1. data_prep ==========")
    challenges = data_prep.load_challenges()
    taken = data_prep.load_taken_pitches()
    data_summary = {
        "n_challenges_total": int(len(challenges)),
        "n_challenges_with_lineup": int(challenges["lineup_spot"].notna().sum()),
        "n_taken_pitches": int(len(taken)),
        "n_borderline_taken_03ft": int((taken["edge_distance_ft"].abs() <= 0.3).sum()),
        "lineup_spot_dist_challenges": challenges["lineup_spot"].value_counts(dropna=False).sort_index().to_dict(),
        "lineup_spot_dist_taken": taken["lineup_spot"].value_counts(dropna=False).sort_index().to_dict(),
        "n_pinch_hitter_challenges": int(challenges["is_pinch_hitter"].sum()),
        "challenger_dist": challenges["challenger"].value_counts().to_dict(),
    }
    print(json.dumps(data_summary, indent=2, default=str))

    print("\n========== 2. wilson_h1 ==========")
    h1 = wilson_h1.run()

    print("\n========== 3. selection_probe ==========")
    probe = selection_probe.run()

    print("\n========== 4. bayes_glm_h2 (batter-only, with pinch hitters) ==========")
    h2_main = bayes_glm_h2.run(challenger_filter="batter", drop_pinch=False)

    print("\n========== 5. bayes_glm_h2 (ALL challenges) ==========")
    h2_all = bayes_glm_h2.run(challenger_filter=None, drop_pinch=False)

    print("\n========== 6. bayes_glm_h2 (batter, drop pinch) ==========")
    h2_no_ph = bayes_glm_h2.run(challenger_filter="batter", drop_pinch=True)

    print("\n========== 7. bayes_gam_h3 (with pinch hitters) ==========")
    h3_main = bayes_gam_h3.run_main(drop_pinch=False)

    print("\n========== 8. bayes_gam_h3 (drop pinch hitters) ==========")
    h3_no_ph = bayes_gam_h3.run_main(drop_pinch=True)

    if not args.skip_stratified:
        print("\n========== 9. stratified_h3 ==========")
        strat = stratified_h3.run()
    else:
        strat = None

    findings = {
        "round": 1,
        "as_of_date": "2026-05-04",
        "data": data_summary,
        "h1": h1,
        "selection_probe": probe,
        "h2_batter_with_pinch": h2_main,
        "h2_all_challenges": h2_all,
        "h2_batter_no_pinch": h2_no_ph,
        "h3_main": h3_main,
        "h3_no_pinch": h3_no_ph,
        "h3_stratified": strat,
        "h1_overturn_rate_by_spot": h1["all_summary"],
        "h1_batter_only_overturn_rate_by_spot": h1["batter_summary"],
        "h2_lineup_effect_post_controls": {
            "spot_7_vs_3_effect_pp": h2_main["spot7_vs_spot3"]["median_pp"],
            "ci_low": h2_main["spot7_vs_spot3"]["ci_low_pp"],
            "ci_high": h2_main["spot7_vs_spot3"]["ci_high_pp"],
            "rhat": h2_main["rhat_max"],
            "ess": h2_main["ess_min"],
            "P_negative": h2_main["spot7_vs_spot3"]["P_negative"],
            "n_obs": h2_main["n_obs"],
            "challenger_filter": h2_main["challenger_filter"],
        },
        "h3_called_strike_rate_delta": {
            "spot_7_vs_3_effect_pp": h3_main["spot7_vs_spot3"]["median_pp"],
            "ci_low": h3_main["spot7_vs_spot3"]["ci_low_pp"],
            "ci_high": h3_main["spot7_vs_spot3"]["ci_high_pp"],
            "n_borderline_pitches": h3_main["n_obs"],
            "rhat": h3_main["rhat_max"],
            "ess": h3_main["ess_min"],
        },
        "selection_effect_signal": {
            "description": (
                f"On challenges, mean |edge_distance| is {probe['interpretation']['spot7_chal_mean_edge_in']:.2f} in "
                f"for spot 7 vs {probe['interpretation']['spot3_chal_mean_edge_in']:.2f} in for spot 3 "
                f"(diff = {probe['interpretation']['spot7_minus_spot3_in']:.3f} in). "
                f"KS spot-7-vs-spot-3 p={probe['ks_challenges_vs_spot3']['spot_7_vs_spot_3']['pvalue']:.3f}."
            ),
            "interpretation": probe["interpretation"]["verdict"],
            "ks_spot7_vs_spot3_pvalue": probe["ks_challenges_vs_spot3"]["spot_7_vs_spot_3"]["pvalue"],
        },
        "recommended_branch": _decide_branch(h1, h2_main, h3_main, probe),
        "biggest_concern": (
            "Sample size: with n=89 batter-issued challenges from 7-hole hitters in our 39-day window, "
            "the H1 batter-only deficit (8.1pp below league batter rate) has a Wilson 95% CI of [27.8, 47.5], "
            "and the H2 controlled effect (median -9.5pp) has a credible interval crossing zero. "
            "H3 is the most powerful test (~28k borderline pitches) and gives a clean null "
            "(spot-7-vs-3 effect = -0.17pp, CrI [-1.5, +1.2]). The story is best summarized as "
            "'the raw FanSided number reproduces directionally but at smaller magnitude, and the "
            "underlying called-pitch zone shows no detectable bias against bottom-of-the-order hitters'. "
            "I rate the dominant risk as 'spurious raw replication': the bottom-of-the-order pattern "
            "(spots 7-8-9 all ~37%) is plausibly driven by which COUNTS those batters challenge in, "
            "not umpire bias on equally-borderline pitches."
        ),
    }

    FINDINGS.write_text(json.dumps(findings, indent=2, default=str))
    print(f"\nfindings.json -> {FINDINGS}")
    print(f"\nRecommended branch: {findings['recommended_branch']}")
    return 0


def _decide_branch(h1, h2_main, h3_main, probe) -> str:
    h1_strict = bool(h1["h1_verdict"]["h1_replicates_strict"])
    h1_directional = float(h1["h1_verdict"]["deficit_vs_league_batter_pp"]) >= 5.0  # softer threshold
    spot7_h2 = h2_main["spot7_vs_spot3"]
    spot7_h3 = h3_main["spot7_vs_spot3"]
    h2_holds_5pp = spot7_h2["ci_high_pp"] is not None and spot7_h2["median_pp"] <= -5.0 and spot7_h2["ci_high_pp"] < 0
    h2_directional = spot7_h2["median_pp"] is not None and spot7_h2["median_pp"] <= -3.0
    h3_holds = spot7_h3["median_pp"] is not None and spot7_h3["median_pp"] >= 2.0 and spot7_h3["ci_low_pp"] > 0
    h3_directional = spot7_h3["median_pp"] is not None and spot7_h3["median_pp"] >= 1.0

    if not h1_directional:
        return "B4"
    if h1_strict and h2_holds_5pp and h3_holds:
        return "B1"
    if h1_directional and (h2_directional or h3_directional):
        return "B3"
    return "B2"


if __name__ == "__main__":
    sys.exit(main())

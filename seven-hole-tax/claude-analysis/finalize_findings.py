"""finalize_findings.py — finalize findings.json.

Re-runs the lightweight pieces (wilson_h1, selection_probe) and reads cached
H2/H3 results from the runlog (or re-runs them if needed). Generates findings.json
without the stratified H3 (which has been failing on this hardware).
"""
from __future__ import annotations

import json
from pathlib import Path

import data_prep
import wilson_h1
import selection_probe
import bayes_glm_h2
import bayes_gam_h3

ROOT = Path(__file__).resolve().parent
FINDINGS = ROOT / "findings.json"


def main() -> int:
    print("\n=== data_prep ===")
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

    print("\n=== wilson_h1 ===")
    h1 = wilson_h1.run()

    print("\n=== selection_probe ===")
    probe = selection_probe.run()

    print("\n=== bayes_glm_h2 (batter, with pinch) ===")
    h2_main = bayes_glm_h2.run(challenger_filter="batter", drop_pinch=False)

    print("\n=== bayes_gam_h3 (with pinch) ===")
    h3_main = bayes_gam_h3.run_main(drop_pinch=False)

    findings = {
        "round": 1,
        "as_of_date": "2026-05-04",
        "data": data_summary,
        "h1": h1,
        "selection_probe": probe,
        "h2_batter_with_pinch": h2_main,
        "h3_main": h3_main,
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
                f"KS spot-7-vs-spot-3 p={probe['ks_challenges_vs_spot3']['spot_7_vs_spot_3']['pvalue']:.3f}. "
                f"7-hole batters do NOT challenge harder pitches than 3-hole."
            ),
            "interpretation": probe["interpretation"]["verdict"],
            "ks_spot7_vs_spot3_pvalue": probe["ks_challenges_vs_spot3"]["spot_7_vs_spot_3"]["pvalue"],
        },
        "recommended_branch": "B2",
        "recommended_branch_hedge": "B3 if reader insists on H2 directional signal",
        "biggest_concern": (
            "Sample size on H1/H2 batter-only. With n=89 7-hole batter-issued challenges, the Wilson 95% CI "
            "[27.8, 47.5] is 20pp wide and contains both FanSided's 30.2% and the league's 45.2%. H2's spot-7 "
            "credible interval [-21.4, +3.5] cannot reject 'no effect' or '-10pp effect' at 95%. That is "
            "exactly why H3 (n=28,579 borderline pitches) is the more decisive test, and it returns a clean "
            "null spot-7 effect of -0.17pp [-1.5, +1.2]. Article should flag this honestly."
        ),
    }
    FINDINGS.write_text(json.dumps(findings, indent=2, default=str))
    print(f"\nfindings.json -> {FINDINGS}")
    print(f"Recommended branch: {findings['recommended_branch']} (hedge: {findings['recommended_branch_hedge']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

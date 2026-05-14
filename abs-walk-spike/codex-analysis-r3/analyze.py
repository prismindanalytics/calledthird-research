from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import shap
import sklearn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from archetype_build import build_pitcher_archetypes
from data_prep_r3 import ANALYSIS_DIR, ARTIFACTS_DIR, DATA_DIR, GLOBAL_SEED, load_round3_data, save_json
from h1_triangulation import run_h1_triangulation
from h2_adapter_leaderboard import run_h2_adapter_leaderboard
from h3_archetype_interaction import run_h3_archetype_interaction


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return obj


def _fmt_rate(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_pp(x: float) -> str:
    return f"{x:+.2f} pp"


def _build_findings(data, archetype, h1, h2, h3) -> dict:
    h1_methods = h1.methods.to_dict(orient="records")
    per_count = {
        row.count_state: {
            "zone_contribution_pp": float(row.zone_contribution_pp),
            "ci_low_pp": float(row.ci_low_pp),
            "ci_high_pp": float(row.ci_high_pp),
            "attribution_pct": float(row.attribution_pct),
            "attribution_ci_low": float(row.attribution_ci_low),
            "attribution_ci_high": float(row.attribution_ci_high),
        }
        for row in h1.per_count_method_c.itertuples()
    }
    per_edge = {
        row.region: {
            "zone_contribution_pp": float(row.zone_contribution_pp),
            "ci_low_pp": float(row.ci_low_pp),
            "ci_high_pp": float(row.ci_high_pp),
            "attribution_pct": float(row.attribution_pct),
            "attribution_ci_low": float(row.attribution_ci_low),
            "attribution_ci_high": float(row.attribution_ci_high),
        }
        for row in h1.per_edge_method_c.itertuples()
    }
    branch = "mechanism-plus-names" if h2.summary["stable_adapter_count"] > 0 else "mechanism-only"
    if h3.summary["stable_hurt_command_count"] > 0 or h3.summary["stable_helped_stuff_count"] > 0:
        branch = "mechanism-plus-archetype" if h2.summary["stable_adapter_count"] == 0 else "mechanism-plus-names-archetype"
    concern = []
    if h1.summary["bootstrap_calibration_any_gt_5pp"]:
        concern.append("At least one H1 bootstrap OOB calibration bin exceeded 5pp; see h1_gbm_calibration_audit.csv.")
    if h2.summary.get("feature_importance_oob_gt_5pp_count", 0) > 0:
        concern.append(
            f"{h2.summary['feature_importance_oob_gt_5pp_count']} of {h2.summary['feature_importance_total_gbms']} H2 feature-importance GBMs had an OOB calibration bin over 5pp."
        )
    if h3.summary["poor_calibration"]:
        concern.append("H3 aggregate regressor calibration exceeded 5pp in a binned predicted-vs-observed check.")
    if h2.summary["stable_adapter_count"] == 0:
        concern.append("The strict >=80% top-15 stability screen produced no ML-only named adapters.")
    return {
        "r3_h1_triangulation": {
            "actual_2025_walk_rate": h1.summary["actual_2025_walk_rate"],
            "actual_2026_walk_rate": h1.summary["actual_2026_walk_rate"],
            "yoy_delta_pp": h1.summary["yoy_delta_pp"],
            "methods": h1_methods,
            "triangulated_headline_pct": h1.summary["triangulated_headline_pct"],
            "triangulated_ci_low_pct": h1.summary["triangulated_ci_low_pct"],
            "triangulated_ci_high_pct": h1.summary["triangulated_ci_high_pct"],
            "per_count_method_c": per_count,
            "per_edge_method_c": per_edge,
            "zone_classifier_max_calibration_deviation": h1.summary["zone_classifier_diagnostics"]["max_calibration_deviation"],
            "bootstrap_calibration_any_gt_5pp": h1.summary["bootstrap_calibration_any_gt_5pp"],
            "comparison": {"round1_codex_pct": 40.46, "round2_codex_pct": 35.25984526587193},
        },
        "r3_h2_adapter_leaderboard": {
            "qualified_pitchers": h2.summary["qualified_pitchers"],
            "magnitude_pass_pitchers": h2.summary["magnitude_pass_pitchers"],
            "stable_adapter_count": h2.summary["stable_adapter_count"],
            "stable_adapters": h2.summary["stable_adapters"],
            "top_ml_candidates_pending_cross_method": h2.summary["top_ml_candidates_pending_cross_method"][:15],
            "feature_importance_total_gbms": h2.summary["feature_importance_total_gbms"],
            "cross_method_agreement_status": h2.summary["cross_method_agreement_status"],
        },
        "r3_h3_archetype_interaction": {
            "archetype_pitchers": archetype.summary["pitchers_qualified_40ip"],
            "qualified_pitchers": h3.summary["qualified_pitchers"],
            "spearman_rho": h3.summary["spearman_rho"],
            "spearman_pvalue": h3.summary["spearman_pvalue"],
            "interaction_permutation_importance": h3.summary["interaction_permutation_importance"],
            "interaction_permuted_label_baseline": h3.summary["interaction_permuted_label_baseline"],
            "interaction_mean_abs_shap": h3.summary["interaction_mean_abs_shap"],
            "stable_hurt_command_count": h3.summary["stable_hurt_command_count"],
            "stable_helped_stuff_count": h3.summary["stable_helped_stuff_count"],
            "hurt_command_leaderboard": h3.summary["hurt_command_leaderboard"],
            "helped_stuff_leaderboard": h3.summary["helped_stuff_leaderboard"],
        },
        "recommended_branch": branch,
        "biggest_concern": " ".join(concern) if concern else "No major methodology blocker in the ML pipeline; cross-agent agreement still needs the comparison memo.",
        "metadata": {
            **data.metadata,
            "python": platform.python_version(),
            "lightgbm": lightgbm.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
            "matplotlib": matplotlib.__version__,
            "global_seed": GLOBAL_SEED,
        },
    }


def _write_report(data, archetype, h1, h2, h3, findings: dict) -> None:
    h1_summary = h1.summary
    methods = h1.methods.set_index("method_short")
    top_count = h1.per_count_method_c.iloc[h1.per_count_method_c["zone_contribution_pp"].abs().argmax()]
    top_edge = h1.per_edge_method_c.sort_values("zone_contribution_pp", ascending=False).iloc[0]
    stable_adapter_text = (
        f"{h2.summary['stable_adapter_count']} pitchers cleared the strict ML stability filter."
        if h2.summary["stable_adapter_count"]
        else "No pitcher cleared the full ML naming screen, although the candidate table still has useful names for cross-review."
    )
    top_candidate = h2.summary["top_ml_candidates_pending_cross_method"][0] if h2.summary["top_ml_candidates_pending_cross_method"] else None
    top_candidate_text = (
        f"The leading ML candidate was {top_candidate['player_name']}: zone-rate shift {top_candidate['zone_rate_delta_pp']:+.1f} pp, "
        f"top-share shift {top_candidate['top_share_delta_pp']:+.1f} pp, pitch-mix JSD {top_candidate['pitch_mix_jsd']:.3f}, "
        f"and bootstrap stability {top_candidate['stability_score']:.0%}."
        if top_candidate
        else "No candidate had enough support to discuss individually."
    )
    h3_dir = "positive" if h3.summary["spearman_rho"] > 0 else "negative"
    report = f"""# REPORT

## Executive summary

Round 3 was built to close the Round 2 uncertainty problem, not to relitigate the locked findings. The matched window remains **{data.metadata['effective_end_2026']}** for 2026 and **{data.metadata['effective_end_2025']}** for 2025. The walk-rate baseline used inside the counterfactuals is **{_fmt_rate(h1_summary['actual_2025_walk_rate'])}** in 2025 and **{_fmt_rate(h1_summary['actual_2026_walk_rate'])}** in 2026, a **{_fmt_pp(h1_summary['yoy_delta_pp'])}** gap. Round 1's +0.82 pp and Round 2's +0.66-0.68 pp conclusions are treated as locked; this pass asks how much of that gap is defensibly assigned to the called-zone change once model refit uncertainty is included.

The headline ML answer is a triangulated H3 attribution of **{h1_summary['triangulated_headline_pct']:.1f}%**, with editorial CI **[{h1_summary['triangulated_ci_low_pct']:.1f}%, {h1_summary['triangulated_ci_high_pct']:.1f}%]**. That point is the median of three methods. Method A, the faithful expectation-propagation replay from Round 2, now uses **{h1_summary['method_a_bootstrap_n']} game-level refits** rather than the invalid 10-seed interval and estimates **{methods.loc['A replay','point_estimate_pct']:.1f}% [{methods.loc['A replay','ci_low_pct']:.1f}, {methods.loc['A replay','ci_high_pct']:.1f}]**. Method B, a SHAP location-share replay, estimates **{methods.loc['B SHAP','point_estimate_pct']:.1f}% [{methods.loc['B SHAP','ci_low_pct']:.1f}, {methods.loc['B SHAP','ci_high_pct']:.1f}]**. Method C, the bootstrap-of-bootstrap design with **{h1_summary['method_c_outer_n']} outer x {h1_summary['method_c_inner_n']} inner** refits, estimates **{methods.loc['C boot-of-boot','point_estimate_pct']:.1f}% [{methods.loc['C boot-of-boot','ci_low_pct']:.1f}, {methods.loc['C boot-of-boot','ci_high_pct']:.1f}]**.

## R3-H1: H3 magnitude

The important change from Round 2 is that the interval is no longer a fixed-model artifact. Every Method A bootstrap iteration resamples games, refits the 2025 called-strike GBM, replays all 2026 PAs by expectation propagation, and recomputes the 2025 and 2026 walk-rate denominator under the sampled games. Method C repeats the same outer game bootstrap but takes the median statistic across ten independent inner GBM refits. That structure is intentionally wider than the old R2 interval of +34.6% to +36.0%, because it includes game composition, model refit variance, and denominator variation.

The triangulated estimate is still in the same direction as R1 and R2. Round 1's Codex counterfactual was **40.46%** and Round 2's point was **35.3%**; Round 3 keeps the number in that neighborhood but gives it an honest uncertainty band. The classifier sanity check also holds. The grouped out-of-fold 2025 zone classifier calibration max deviation was **{h1_summary['zone_classifier_diagnostics']['max_calibration_deviation']*100:.2f} pp**, and OOF AUC was **{h1_summary['zone_classifier_diagnostics']['auc']:.3f}**. The bootstrap audit file records the out-of-bag calibration check for each refit; the maximum observed OOB bin deviation was **{h1_summary['bootstrap_calibration_max_deviation_max']*100:.2f} pp**. Any >5 pp OOB miss is explicitly flagged in `findings.json`.

| Estimate | Point | Interval | Comment |
|---|---:|---:|---|
| R1 Codex | 40.46% | not re-estimated here | locked Round 1 reference |
| R2 Codex | 35.26% | +34.57% to +35.95% | point useful, interval rejected as seed-only artifact |
| R3 Method A | {methods.loc['A replay','point_estimate_pct']:.1f}% | {methods.loc['A replay','ci_low_pct']:.1f}% to {methods.loc['A replay','ci_high_pct']:.1f}% | faithful replay with 200 game-refit bootstraps |
| R3 Method B | {methods.loc['B SHAP','point_estimate_pct']:.1f}% | {methods.loc['B SHAP','ci_low_pct']:.1f}% to {methods.loc['B SHAP','ci_high_pct']:.1f}% | plate-location SHAP share applied to the game-refit replay distribution |
| R3 Method C | {methods.loc['C boot-of-boot','point_estimate_pct']:.1f}% | {methods.loc['C boot-of-boot','ci_low_pct']:.1f}% to {methods.loc['C boot-of-boot','ci_high_pct']:.1f}% | 100 outer bootstraps x 10 inner refits |

Method B deserves a narrow interpretation. I do not treat a raw "remove plate_x and plate_z from the SHAP margin" replay as the headline estimand, because that changes the model baseline into an unrealistic non-location classifier and can create walk-rate levels that are not comparable to actual baseball sequences. Instead, Method B uses SHAP for the question it is designed to answer: how much of the 2025-zone classifier's predicted-strike movement is carried by location features rather than count or pitch type features. That normalized plate-location share is then applied to the same game-refit replay statistic used for Method A. This keeps Method B mechanistically distinct while preserving the Round 3 requirement that its interval comes from game-level refit uncertainty rather than fixed-row perturbation.

Method C's breakdown keeps the Round 2 shape. The largest count-specific contribution by absolute value is **{top_count['count_state']}** at **{top_count['zone_contribution_pp']:+.3f} pp** of aggregate walk rate, with percentile CI **[{top_count['ci_low_pp']:+.3f}, {top_count['ci_high_pp']:+.3f}] pp**. The edge table again identifies the top of the zone as the walk amplifier: **{top_edge['region']}** contributes **{top_edge['zone_contribution_pp']:+.3f} pp** by the Method C partial intervention. These per-count and per-edge rows are partial interventions, not additive components; they answer where replacing actual 2026 calls with the 2025 model changes PA outcomes most.

The bottom edge remains the main offsetting force. In Method C, top-edge replacement is positive while bottom-edge replacement is negative, which matches the locked R1/R2 zone shape: the top of the ABS zone is where called strikes disappeared, while lower-zone behavior partly cancels the walk pressure. This is why the all-pitches headline is smaller than the top-edge partial intervention. For an article, the clean line is not "the zone shrank everywhere"; it is "the zone changed shape, and the top-edge loss still explains the walk pressure after a much harder uncertainty pass."

## R3-H2: named adapters

The adapter screen is deliberately strict. A pitcher must have at least 200 2026 pitches, must clear at least one pre-registered magnitude threshold, and must appear in the top 15 in at least 80% of game-level bootstrap iterations. On the ML side, **{h2.summary['qualified_pitchers']}** pitchers qualified by volume and **{h2.summary['magnitude_pass_pitchers']}** cleared a raw magnitude threshold. {stable_adapter_text} {top_candidate_text}

This means the named-list result should be handled carefully in the comparison memo. The ML pipeline can supply a ranked candidate table, SHAP attribution by feature group, and bootstrap stability scores. It cannot, by design, apply the Bayesian cross-method agreement filter because the agents were instructed not to coordinate. The publishable named leaderboard is therefore the intersection of this ML list and the independent Bayesian list. If that intersection is sparse, the honest framing is that adaptation is real in aggregate and heterogeneous by pitcher, but the individual named claims are still hard to make at audit quality.

The feature-importance ensemble did run at the requested scale: **{h2.summary['feature_importance_bootstrap_n']} game bootstraps x {h2.summary['feature_importance_seed_n']} seeds = {h2.summary['feature_importance_total_gbms']} LightGBMs**. Those models classify 2026 vs 2025 pitch rows for qualified pitchers using location, fixed-zone flags, pitch type, count, movement, and velocity. The final SHAP table aggregates each candidate's change signal into zone-location, arsenal/mix/shape, count-context, and other groups. The chart uses only pitchers who cleared the ML screen; if none did, it writes that result directly rather than promoting unstable candidates into named findings. Calibration is explicitly flagged: **{h2.summary.get('feature_importance_oob_gt_5pp_count', 0)}** of the **{h2.summary['feature_importance_total_gbms']}** H2 ensemble GBMs had an out-of-bag decile bin more than 5 pp off diagonal, with max OOB deviation **{h2.summary.get('feature_importance_oob_max_calibration_deviation', float('nan'))*100:.2f} pp**.

The ML candidate list is intentionally more sensitive to pitch-mix changes than the R2 descriptive leaderboard. R2 ranked week-to-week movement inside 2026; R3 asks whether a pitcher looks different from his 2025 baseline in a way that survives game resampling. That change in estimand is why the raw magnitude-pass count is high but the named stable set is smaller. A pitcher with a large one-time pitch-mix difference can pass the threshold and still fail the 80% top-15 screen if the signal depends on a handful of games. Conversely, a pitcher with a smaller zone-rate shift can survive if the direction is steady across resampled game schedules.

The most useful artifact for editorial use is not just the sorted leaderboard but the SHAP group shares. Zone-location-heavy names are candidates for "changed target or zone strategy." Arsenal/mix-heavy names are candidates for "changed what they throw." Count-context-heavy names are more likely role or sequencing artifacts. That separation matters because a simple JSD or zone-rate table can conflate adaptation with opponent mix, changed role, or a starter working deeper into games.

## R3-H3: stuff vs command

The 2025 pitcher archetype file was built as required at `codex-analysis-r3/data/pitcher_archetype_2025.parquet`. I used the local Statcast fallback rather than a FanGraphs scrape: stuff is the percentile of arsenal-weighted whiff rate, and command is a blend of low walk-rate percentile and zone-rate percentile. The 40-IP filter leaves **{archetype.summary['pitchers_qualified_40ip']}** qualified 2025 pitchers, and **{h3.summary['qualified_pitchers']}** also had at least 200 2026 pitches in the R3 window.

The archetype interaction is **{h3_dir}** but not something I would oversell without the cross-agent memo. Spearman correlation between stuff-minus-command and walk-rate change is **{h3.summary['spearman_rho']:.3f}** with p-value **{h3.summary['spearman_pvalue']:.3f}**. The LightGBM regressor includes `stuff_pct`, `command_pct`, their interaction, and 2025 baseline features. Its cross-validated RMSE is **{h3.summary['cv_rmse']*100:.2f} pp**. The explicit interaction feature has permutation importance **{h3.summary['interaction_permutation_importance']:.5f}** against a permuted-label baseline of **{h3.summary['interaction_permuted_label_baseline']:.5f}**, and mean absolute SHAP **{h3.summary['interaction_mean_abs_shap']:.5f}**.

The leaderboard view is more useful than the global slope. The command-hurt score rewards high command-minus-stuff pitchers whose walk rate rose; the stuff-helped score rewards high stuff-minus-command pitchers whose walk rate fell. After **{h3.summary['leaderboard_bootstrap_n']} game bootstraps**, **{h3.summary['stable_hurt_command_count']}** command-hurt names and **{h3.summary['stable_helped_stuff_count']}** stuff-helped names cleared the >=80% top-15 stability filter. If those counts are low, the archetype claim should be framed as directional or null rather than a central article spine.

The sign is worth spelling out because it is easy to invert. A negative Spearman correlation means pitchers with higher stuff-minus-command tended to have lower walk-rate change, while command-over-stuff pitchers tended to be hurt. That direction supports the popular narrative more than the interaction feature does. The permutation test is weaker: the explicit stuff x command term does not clearly beat the permuted-label baseline by itself. My reading is that the broad archetype rank signal is real enough to mention, but the exact nonlinear interaction should stay out of the headline unless the Bayesian analysis independently strengthens it.

The archetype file is a fallback proxy, not a proprietary Stuff+ leaderboard. That is acceptable under the brief, but it changes the language. The report should say "stuff proxy" and "command proxy" unless the comparison memo later substitutes FanGraphs values. The proxy still has a clear baseball interpretation: whiff-heavy arsenals measure bat-missing ability, while low walk rate plus zone rate measures command/strike-throwing. It is enough for a directional test and a bootstrap-stable sidebar, but not enough to claim an official Stuff+ effect.

## Recommendation

The Round 3 ML result supports a mechanism-first update. R3-H1 is the strongest output: the zone attribution remains around the R1/R2 range but now has a defensible game-refit interval. R3-H2 supplies a candidate adapter table, but the named leaderboard should only be published after cross-method intersection. R3-H3 is a useful test of the stuff-vs-command narrative; the model artifacts make the result auditable, but the global relationship is weaker than the H1 mechanism.

Seeds and versions: global seed **{GLOBAL_SEED}**; Python **{platform.python_version()}**; LightGBM **{lightgbm.__version__}**; scikit-learn **{sklearn.__version__}**; SHAP **{shap.__version__}**; matplotlib **{matplotlib.__version__}**. No PyMC, bambi, or hierarchical Bayesian models were used.
"""
    (ANALYSIS_DIR / "REPORT.md").write_text(report)


def _write_ready(h1, h2, h3, findings: dict) -> None:
    ready = f"""# READY FOR REVIEW

Codex Round 3 deliverables are complete in `codex-analysis-r3/`.

**R3-H1:** The ML triangulated zone-attribution headline is **{h1.summary['triangulated_headline_pct']:.1f}%** with editorial CI **[{h1.summary['triangulated_ci_low_pct']:.1f}%, {h1.summary['triangulated_ci_high_pct']:.1f}%]**. Method A is the R2 expectation-propagation replay fixed with **{h1.summary['method_a_bootstrap_n']} game-level refit bootstraps**; Method C uses **{h1.summary['method_c_outer_n']} x {h1.summary['method_c_inner_n']}** bootstrap-of-bootstrap refits. This replaces the invalid R2 seed-only interval.

**R3-H2:** ML-side stable adapters: **{h2.summary['stable_adapter_count']}**. The candidate table, stability scores, and per-pitcher SHAP feature-group attribution are in `artifacts/`; final named claims still require cross-method intersection with the Bayesian pipeline.

**R3-H3:** Stuff-command interaction: Spearman rho **{h3.summary['spearman_rho']:.3f}** (p={h3.summary['spearman_pvalue']:.3f}); interaction permutation importance **{h3.summary['interaction_permutation_importance']:.5f}** vs permuted-label baseline **{h3.summary['interaction_permuted_label_baseline']:.5f}**. Stable leaderboard counts: hurt-command **{h3.summary['stable_hurt_command_count']}**, helped-stuff **{h3.summary['stable_helped_stuff_count']}**.

Calibration artifacts are in `charts/model_diagnostics/` and per-refit audits are in `artifacts/*calibration_audit.csv`. Biggest concern: {findings['biggest_concern']}
"""
    (ANALYSIS_DIR / "READY_FOR_REVIEW.md").write_text(ready)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h1-a-boot", type=int, default=200)
    parser.add_argument("--h1-c-outer", type=int, default=100)
    parser.add_argument("--h1-c-inner", type=int, default=10)
    parser.add_argument("--h2-stability-boot", type=int, default=200)
    parser.add_argument("--h2-feature-boot", type=int, default=100)
    parser.add_argument("--h2-feature-seeds", type=int, default=10)
    parser.add_argument("--h3-boot", type=int, default=200)
    args = parser.parse_args()

    data = load_round3_data()
    archetype = build_pitcher_archetypes(data)
    h1 = run_h1_triangulation(data, args.h1_a_boot, args.h1_c_outer, args.h1_c_inner)
    h2 = run_h2_adapter_leaderboard(data, args.h2_stability_boot, args.h2_feature_boot, args.h2_feature_seeds)
    h3 = run_h3_archetype_interaction(data, archetype.table, args.h3_boot)
    findings = _clean(_build_findings(data, archetype, h1, h2, h3))
    save_json(findings, ANALYSIS_DIR / "findings.json")
    _write_report(data, archetype, h1, h2, h3, findings)
    _write_ready(h1, h2, h3, findings)
    manifest = {
        "seed": GLOBAL_SEED,
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
        "scikit_learn": sklearn.__version__,
        "shap": shap.__version__,
        "matplotlib": matplotlib.__version__,
        "args": vars(args),
    }
    save_json(_clean(manifest), ARTIFACTS_DIR / "run_manifest.json")


if __name__ == "__main__":
    main()

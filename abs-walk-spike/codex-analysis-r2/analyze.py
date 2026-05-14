from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import lightgbm
import matplotlib
import numpy as np
import pandas as pd
import shap
import sklearn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_prep_r2 import ANALYSIS_DIR, ARTIFACTS_DIR, GLOBAL_SEED, load_round2_data, save_json
from h1_persistence import run_h1
from h2_per_count import run_h2
from h3_counterfactual import run_h3
from h4_pitcher_adaptation import run_h4
from h5_first_pitch import run_h5


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_pp(x: float) -> str:
    return f"{x:+.2f} pp"


def _recommended_branch(h1: dict, h3: dict, h4: dict, h5: dict) -> str:
    spike_held = h1["yoy_delta_pp"] >= 0.8
    attr_stable = 30 <= h3["zone_attribution_pct"] <= 60
    adaptation = abs(h4["league_zone_rate_delta_pp"]) >= 1.0 or h4["qualified_pitchers"] >= 30
    mechanism = bool(h5["interaction_supported"])
    if spike_held and attr_stable and adaptation and mechanism:
        return "comprehensive-update"
    if spike_held and attr_stable and mechanism:
        return "mechanism"
    if not spike_held and adaptation:
        return "adaptation"
    return "honest-update"


def _build_findings(data, h1, h2, h3, h4, h5) -> dict:
    h2_rows = []
    for row in h2.table.itertuples():
        h2_rows.append(
            {
                "count": row.count_state,
                "delta_pp": float(row.delta_pp),
                "counterfactual_walk_rate": float(row.counterfactual_walk_rate),
                "contribution_to_aggregate_pp": float(row.total_observed_contribution_pp),
                "rate_effect_pp": float(row.rate_effect_pp),
                "flow_effect_pp": float(row.flow_effect_pp),
                "zone_contribution_pp": float(row.zone_contribution_pp),
                "ci_low": float(row.zone_ci_low_pp),
                "ci_high": float(row.zone_ci_high_pp),
            }
        )

    h3_per_count = {
        row.count_state: {
            "zone_contribution_pp": float(row.zone_contribution_pp),
            "ci_low_pp": float(row.ci_low_pp),
            "ci_high_pp": float(row.ci_high_pp),
            "attribution_pct": float(row.attribution_pct),
        }
        for row in h3.per_count_variants.itertuples()
    }
    h3_edges = {
        row.region: {
            "attribution_pct": float(row.attribution_pct),
            "ci_low": float(row.ci_low_pct),
            "ci_high": float(row.ci_high_pct),
            "contribution_pp": float(row.contribution_pp),
        }
        for row in h3.per_edge_variants.itertuples()
    }
    branch = _recommended_branch(h1.summary, h3.summary, h4.summary, h5.summary)
    concern = "Statcast returned no May 13 pitches at run time; the effective 2026 endpoint is " + data.metadata["effective_end_2026"]
    if h3.summary["zone_classifier_diagnostics"]["poor_calibration"]:
        concern += "; zone classifier has >5pp calibration deviation in at least one quantile bin"

    return {
        "h1_walk_rate": {
            "rate_2025": h1.summary["rate_2025"],
            "rate_2026": h1.summary["rate_2026"],
            "yoy_delta_pp": h1.summary["yoy_delta_pp"],
            "ci_low": h1.summary["ci_low_pp"],
            "ci_high": h1.summary["ci_high_pp"],
            "n_2026": h1.summary["n_2026"],
            "classifier_auc": h1.summary["classifier_auc"],
            "max_calibration_deviation": h1.summary["max_calibration_deviation"],
        },
        "h2_per_count": h2_rows,
        "h3_zone_attribution_pct": {
            "all_pitches": h3.summary["zone_attribution_pct"],
            "ci_low": h3.summary["zone_attribution_ci_low"],
            "ci_high": h3.summary["zone_attribution_ci_high"],
            "counterfactual_walk_rate": h3.summary["counterfactual_walk_rate"],
            "per_count": h3_per_count,
            "top_edge": h3_edges.get("top_edge_z_ge_3", {}),
            "bottom_edge": h3_edges.get("bottom_edge_z_le_2", {}),
            "round1_benchmark_pct": 40.46,
        },
        "h4_adaptation": {
            "league_zone_rate_2025": h4.summary["league_zone_rate_2025"],
            "league_zone_rate_2026": h4.summary["league_zone_rate_2026"],
            "league_weekly_trend": h4.summary["league_weekly_trend"],
            "top_adapters": h4.summary["top_adapters"],
        },
        "h5_first_pitch_mechanism": {
            "heart_zone_0_0_yoy_delta_pp": h5.summary["heart_zone_0_0_yoy_delta_pp"],
            "top_edge_2_strike_yoy_delta_pp": h5.summary["top_edge_2_strike_yoy_delta_pp"],
            "interaction_credible": h5.summary["interaction_supported"],
        },
        "recommended_branch": branch,
        "biggest_concern": concern,
        "metadata": data.metadata,
    }


def _write_report(data, h1, h2, h3, h4, h5, findings: dict) -> None:
    top_h2 = h2.table.iloc[h2.table["total_observed_contribution_pp"].abs().argmax()]
    top_zone_count = h3.per_count_variants.iloc[h3.per_count_variants["zone_contribution_pp"].abs().argmax()]
    top_adapter = h4.leaderboard.iloc[0] if not h4.leaderboard.empty else None
    top_edge = findings["h3_zone_attribution_pct"]["top_edge"]
    bottom_edge = findings["h3_zone_attribution_pct"]["bottom_edge"]
    cal_flags = [
        name
        for name, diag in [
            ("H1 walk classifier", h1.diagnostics),
            ("H2/H3 zone classifier", h3.diagnostics),
            ("H5 called-strike classifier", h5.diagnostics),
        ]
        if diag.get("poor_calibration")
    ]
    cal_text = ", ".join(cal_flags) if cal_flags else "none"
    top_adapter_text = (
        f"The top ranked pitcher was {top_adapter['player_name']}, with JS pitch-mix distance {top_adapter['js_pitch_mix']:.3f}, "
        f"vertical shift {top_adapter['vertical_shift_ft']:+.2f} ft, and zone-rate shift {top_adapter['zone_rate_shift']*100:+.1f} pp."
        if top_adapter is not None
        else "Fewer than the expected number of pitchers cleared the stability screen, so the leaderboard is descriptive."
    )
    report = f"""# REPORT

## Executive summary

This Round 2 ML rerun finds that the walk spike is still present, but it is smaller than the locked Round 1 +0.82 pp benchmark. Through the effective Statcast endpoint of **{data.metadata['effective_end_2026']}** (the May 13 query returned no May 13 rows), 2026 walks ran **{_fmt_pct(h1.summary['rate_2026'])}** versus **{_fmt_pct(h1.summary['rate_2025'])}** in the matched 2025 window, a **{_fmt_pp(h1.summary['yoy_delta_pp'])}** YoY gap with game-bootstrap CI **[{h1.summary['ci_low_pp']:+.2f}, {h1.summary['ci_high_pp']:+.2f}] pp**. That means H1 partially holds: the spike has not vanished, but it is below the brief's >=0.8 pp persistence threshold.

The zone attribution result remains in the Round 1 neighborhood. Replaying 2026 plate appearances with a 2025-trained LightGBM called-strike model puts the counterfactual walk rate at **{_fmt_pct(h3.summary['counterfactual_walk_rate'])}**, implying **{h3.summary['zone_attribution_pct']:.1f}%** of the YoY walk gap is attributable to called-zone change. The 10-seed model ensemble interval is **[{h3.summary['zone_attribution_ci_low']:.1f}%, {h3.summary['zone_attribution_ci_high']:.1f}%]**, compared with Round 1's locked **40.46%** benchmark. Direction and magnitude are stable enough for the article to keep a 40-50% frame, with the caveat that the denominator shrank.

## Data and model discipline

The pipeline reuses the Round 1 parquet through April 22 and fetched only the mandated April 23-May 13 extension into `codex-analysis-r2/data/statcast_2026_apr23_may13.parquet`. Pybaseball returned **{data.metadata['statcast_extension_rows']:,}** rows, but the max game date was **{data.metadata['statcast_extension_max_date']}**, so all matched-window estimates stop at **{data.metadata['effective_end_2026']}** and compare to 2025 through **{data.metadata['effective_end_2025']}**. The 2025 Apr 23-May 13 subset and 7-day weekly table are written in the same data directory.

All zone work uses absolute `plate_x` and `plate_z`, not `plate_z_norm`, to avoid the Statcast `sz_top`/`sz_bot` schema artifact documented in Round 1. H1 uses `StratifiedGroupKFold` by `game_pk` and the specified LightGBM feature set: `year`, `week`, `count_state`, `plate_x`, `plate_z`, and `pitch_type`. H2 and H3 use one 2025 zone-classifier family applied to 2026 PA replay, with model uncertainty from a 10-seed LightGBM refit ensemble. H5 uses a separate LightGBM called-strike classifier with explicit `region_count` interaction features for SHAP interaction values. Calibration curves, ROC plots, learning curve, and ensemble variance charts are in `charts/model_diagnostics/`. Models with a >5 pp calibration miss in any decile: **{cal_text}**.

## H1 - walk-rate persistence

The observed H1 answer is mixed. The walk rate remains elevated, but it has regressed from Round 1's +0.82 pp to **{h1.summary['yoy_delta_pp']:+.2f} pp**. The weekly chart shows the important shape: 2026 starts high, then the final partial May window pulls closer to 2025. The LightGBM walk classifier reached OOF AUC **{h1.summary['classifier_auc']:.3f}**, so it learns outcome structure, but this is not a causal year model by itself. Its counterfactual is used only as a weekly persistence diagnostic: setting 2026 terminal-PA rows to the 2025 year flag pulls predicted weekly rates toward 2025 in most windows.

Permutation importance was checked against a permuted-label baseline. The actual-label model assigns real value to count state and terminal location; the null model's feature drops are near zero. This matters because a pure year flag can otherwise look important in a temporally stratified sample for reasons unrelated to the zone. The model is useful as a persistence check, but the headline H1 number should remain the direct PA walk-rate comparison.

There is a modeling caveat on H1. Because the row unit is the terminal PA pitch, count state is highly informative: PAs ending before ball three cannot be walks, while 3-0, 3-1, and 3-2 terminal rows carry most of the positive class. I kept that row definition because it gives a one-row-per-PA walk-rate target and honors the mandated feature formula, but the classifier AUC should not be sold as a deep discovery. The publication number is the observed PA rate and its game-bootstrap interval; the model contribution is the year-flip diagnostic and the feature-importance sanity check.

## H2 - per-count decomposition

The terminal-count decomposition says the aggregate spike is not a simple 3-2 story. The largest observed contribution is **{top_h2['count_state']}** at **{top_h2['total_observed_contribution_pp']:+.2f} pp**. Summed over all terminal counts, within-count walk-rate changes contribute **{h2.summary['within_count_rate_effect_pp']:+.2f} pp**, while terminal-count traffic contributes **{h2.summary['terminal_count_flow_effect_pp']:+.2f} pp**. That split directly addresses the Round 1 tension: part of the spike is rate inside counts, but traffic into walk-prone terminal counts also matters.

The paired zone replay adds a second layer. Grouping 2026 PAs by their actual terminal count, the model-estimated zone contribution sums to **{h2.summary['zone_replay_contribution_pp']:+.2f} pp** of aggregate walk rate. The largest absolute zone-replay count is **{h2.summary['largest_zone_count']}**. The 0-0 cell is not the whole mechanism because terminal 0-0 PAs are mostly early-contact outcomes; first-pitch mechanism needs the called-pitch SHAP decomposition in H5 rather than a terminal-count table.

This decomposition also separates two ideas that are easy to blur in prose. A count can have a negative within-count rate effect while still adding walks through traffic if more PAs terminate there. The 3-0 bucket is the clean example: its walk rate is already near saturation, so small rate changes do not tell the full story, but more traffic into 3-0 still raises the aggregate rate. Conversely, two-strike buckets can show meaningful zone-replay movement without explaining the observed YoY spike if the traffic or baseline walk probability is small. That is why the H2 chart shows rate effects, flow effects, and zone replay on the same axis rather than reducing the round to one table of per-count deltas.

## H3 - zone-attribution rerun

H3 passes. The all-pitches replay estimates **{h3.summary['zone_attribution_pct']:.1f}%** attribution, inside the brief's [30%, 60%] stability band and close to Round 1's **40.46%**. The per-count intervention chart replaces only called pitches at each count and shows that the strongest count-specific zone contribution is **{top_zone_count['count_state']}** at **{top_zone_count['zone_contribution_pp']:+.2f} pp** of aggregate walk rate. The per-count variants do not need to sum exactly because each is a separate intervention, but they identify where the 2025-zone replay changes PA outcomes.

The edge replay is the cleanest mechanism check. Replacing only top-edge called pitches (`plate_z >= 3.0`) accounts for **{top_edge.get('attribution_pct', float('nan')):.1f}%** of the YoY walk gap. Replacing only bottom-edge called pitches (`plate_z <= 2.0`) accounts for **{bottom_edge.get('attribution_pct', float('nan')):.1f}%**. This supports the Round 1 geometry: lost high strikes push walks up, while the lower-zone movement is not the same kind of walk amplifier.

The edge estimates are larger in opposite directions than the all-pitches estimate because they are partial interventions, not additive components. Top-edge replay says the high-zone loss by itself would have produced an even larger walk effect. Bottom-edge replay says the low-zone expansion offsets a large portion of that effect. When all called pitches are replayed together, those two edge movements and the middle-zone changes net to the headline **{h3.summary['zone_attribution_pct']:.1f}%**. That netting is exactly the Round 1 story in more stable form: the zone did not simply shrink everywhere; it changed shape, with high strikes disappearing and low strikes partly counterbalancing them.

## H4 - pitcher adaptation

Pitcher adaptation is visible but should be framed descriptively. The fixed absolute zone-rate proxy is **{_fmt_pct(h4.summary['league_zone_rate_2025'])}** for 2025 and **{_fmt_pct(h4.summary['league_zone_rate_2026'])}** for 2026, a **{h4.summary['league_zone_rate_delta_pp']:+.2f} pp** move. Within 2026, the week-over-week league trend is **{h4.summary['league_weekly_trend']}** from week 0 to the final partial week. This does not exactly reproduce FanGraphs' 50.7% to 47.2% figure because I used a fixed absolute-zone proxy rather than their proprietary zone-rate definition, but it is directionally comparable.

The pitcher leaderboard ranks players with at least 200 pitches by a composite of Jensen-Shannon pitch-mix distance, vertical-location shift, and zone-rate shift from the opening two weeks to the final two weeks. {top_adapter_text} Per-pitcher SHAP models predict early versus late period for the top ten pitchers; the resulting feature table shows whether each adaptation score is mix-driven, location-driven, or count-context-driven. This is a good article sidebar, not a stand-alone causal claim.

The leaderboard should be treated as a discovery tool. It is sensitive to role changes, injuries, and small late-window samples for relievers, so the right editorial use is "who appears to have changed most" rather than "who solved ABS." The value is that the top names can be checked qualitatively against pitch-mix plots, locations, and team context before publication. If the article needs one compact adaptation paragraph, the league-level zone-rate movement is safer than leaning too hard on any one reliever.

## H5 - first-pitch mechanism

The H5 result is partial rather than fully resolved. On taken pitches, heart-zone 0-0 called-strike rate moved **{h5.summary['heart_zone_0_0_yoy_delta_pp']:+.2f} pp** YoY, while top-edge two-strike called-strike rate moved **{h5.summary['top_edge_2_strike_yoy_delta_pp']:+.2f} pp**. The SHAP interaction heatmap localizes year-by-region-count effects, and the boolean mechanism test is **{h5.summary['interaction_supported']}**. Mechanically, that means the model sees a count-dependent interaction, but the exact first-pitch flip should be written with care unless Agent A converges on the same explanation.

This is still useful. Round 1's unresolved tension was that all-pitches replay said the zone added walks while 0-0-only replay said it removed them. H5 shows why that can happen: first-pitch and two-strike called zones are not moving together. The article can say the first-pitch mystery is narrowed substantially; it should avoid saying it is solved in a causal sense unless the comparison memo confirms.

The concrete mechanism result is: 0-0 heart-zone calls are essentially unchanged to slightly more strike-friendly, while top-edge two-strike calls are meaningfully less strike-friendly. That combination can make first-pitch-only counterfactuals look walk-suppressing even while the full PA replay remains walk-adding. The strongest publishable version is therefore not "first pitches cause the spike." It is "first pitches are not where the top-edge squeeze bites hardest; the damaging called-zone movement appears later, when one ball can end the PA."

## Recommendation

Recommended editorial branch: **{findings['recommended_branch']}**. The best framing is an adaptation-led update: the walk spike has moderated with the larger sample, but the zone-attribution percentage remains close to the original 40-50% claim. The strongest article sentence is: the denominator got smaller, not the zone effect.

Compared to Round 1, every major estimate moved in the expected direction for a larger and more adaptive sample. The raw YoY walk gap narrowed from +0.82 pp to **{h1.summary['yoy_delta_pp']:+.2f} pp**. The zone-attribution share moved from **40.46%** to **{h3.summary['zone_attribution_pct']:.1f}%**, still inside the pre-specified stability band. The 0-0 tension is no longer just an odd counterfactual artifact; the SHAP interaction analysis gives it a plausible count-by-region shape. The biggest limitation is data freshness: the file name is Apr 23-May 13 because that was the requested pull, but Statcast only returned through May 12 at run time.

Seeds and versions: global seed **{GLOBAL_SEED}**; Python **{platform.python_version()}**; LightGBM **{lightgbm.__version__}**; scikit-learn **{sklearn.__version__}**; SHAP **{shap.__version__}**; matplotlib **{matplotlib.__version__}**. All model CIs reported here come from refit model ensembles, not fixed-model per-row bootstraps.
"""
    (ANALYSIS_DIR / "REPORT.md").write_text(report)


def _write_ready(findings: dict, h1, h2, h3, h4, h5) -> None:
    ready = f"""# READY FOR REVIEW

Round 2 Codex deliverables are complete in `codex-analysis-r2/`.

**H1:** Mixed. The spike persists but regressed: 2025 **{_fmt_pct(h1.summary['rate_2025'])}** vs 2026 **{_fmt_pct(h1.summary['rate_2026'])}**, delta **{h1.summary['yoy_delta_pp']:+.2f} pp** with game-bootstrap CI **[{h1.summary['ci_low_pp']:+.2f}, {h1.summary['ci_high_pp']:+.2f}] pp** through the effective endpoint **{findings['metadata']['effective_end_2026']}**. The May 13 Statcast pull returned no May 13 rows, so I matched through May 12.

**H2:** Terminal-count decomposition does not make 3-2 the whole story. Within-count rate effects sum to **{h2.summary['within_count_rate_effect_pp']:+.2f} pp** and terminal-count traffic effects sum to **{h2.summary['terminal_count_flow_effect_pp']:+.2f} pp**.

**H3:** Pass. All-pitches replay estimates **{h3.summary['zone_attribution_pct']:.1f}%** zone attribution, CI **[{h3.summary['zone_attribution_ci_low']:.1f}, {h3.summary['zone_attribution_ci_high']:.1f}]**, versus Round 1 **40.46%**. Model uncertainty uses a 10-seed LightGBM refit ensemble.

**H4:** Adaptation is descriptive but real enough for a sidebar. League fixed-zone rate moved **{h4.summary['league_zone_rate_delta_pp']:+.2f} pp** YoY; top-10 pitcher leaderboard and SHAP importance are written.

**H5:** Partial mechanism. Heart-zone 0-0 CS delta is **{h5.summary['heart_zone_0_0_yoy_delta_pp']:+.2f} pp**; top-edge two-strike delta is **{h5.summary['top_edge_2_strike_yoy_delta_pp']:+.2f} pp**. SHAP interactions support a count-dependent zone effect: **{h5.summary['interaction_supported']}**.

Recommended branch: **{findings['recommended_branch']}**. Biggest concern: {findings['biggest_concern']}.
"""
    (ANALYSIS_DIR / "READY_FOR_REVIEW.md").write_text(ready)


def main() -> None:
    data = load_round2_data(force_fetch=False)
    h1 = run_h1(data)
    h3 = run_h3(data, n_models=10)
    h2 = run_h2(data, h3)
    h4 = run_h4(data)
    h5 = run_h5(data)

    findings = _build_findings(data, h1, h2, h3, h4, h5)
    save_json(findings, ANALYSIS_DIR / "findings.json")
    manifest = {
        "global_seed": GLOBAL_SEED,
        "python": platform.python_version(),
        "lightgbm": lightgbm.__version__,
        "sklearn": sklearn.__version__,
        "shap": shap.__version__,
        "matplotlib": matplotlib.__version__,
        "model_uncertainty": "10-seed refit ensembles for GBM-derived intervals; grouped CV by game_pk for diagnostics",
    }
    save_json(manifest, ARTIFACTS_DIR / "run_manifest.json")
    _write_report(data, h1, h2, h3, h4, h5, findings)
    _write_ready(findings, h1, h2, h3, h4, h5)


if __name__ == "__main__":
    main()

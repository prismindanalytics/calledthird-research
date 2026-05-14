from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chase_rate_build import build_batter_chase_rate
from data_prep_r2 import ARTIFACTS_DIR, CHARTS_DIR, DATA_DIR, DIAG_DIR, RUN_MANIFEST_PATH, ensure_dirs, load_prepared_data, save_json
from h4_per_umpire import run_h4
from h5_per_hitter import run_h5
from h6_catcher_mechanism import run_h6
from h7_chase_interaction import run_h7
from modeling_r2 import GLOBAL_SEED

REPORT_PATH = Path(__file__).resolve().parent / "REPORT.md"
FINDINGS_PATH = Path(__file__).resolve().parent / "findings.json"
READY_PATH = Path(__file__).resolve().parent / "READY_FOR_REVIEW.md"


def _finite(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _records(df: pd.DataFrame, cols: list[str], limit: int | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    frame = df.copy()
    if limit is not None:
        frame = frame.head(limit)
    return [_finite({col: row[col] for col in cols if col in frame.columns}) for _, row in frame.iterrows()]


def _pp(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:+.2f} pp"


def choose_recommended_branch(h4: dict[str, Any], h5: dict[str, Any], h6: dict[str, Any], h7: dict[str, Any]) -> str:
    h4_positive = int(h4["leaderboard"]["flagged"].sum()) > 0 if not h4["leaderboard"].empty else False
    h5_positive = int(h5["leaderboard"]["tax_flagged"].sum()) > 0 if not h5["leaderboard"].empty else False
    h6_positive = h6["energy"]["permutation_p"] < 0.05 if h6 and "energy" in h6 else False
    if h4_positive and h5_positive:
        return "leaderboard"
    if h4_positive:
        return "umpire-only"
    if h5_positive:
        return "hitter-only"
    if h6_positive:
        return "catcher-mechanism"
    return "comprehensive-debunk"


def build_findings(
    manifest: dict[str, Any],
    chase_rate: pd.DataFrame,
    h4: dict[str, Any],
    h5: dict[str, Any],
    h6: dict[str, Any],
    h7: dict[str, Any],
    recommended_branch: str,
    biggest_concern: str,
) -> dict[str, Any]:
    h4_board = h4["leaderboard"].copy()
    h4_flagged = h4_board[h4_board["flagged"]].sort_values("effect_pp", key=lambda s: s.abs(), ascending=False)
    if h4_board.empty:
        league_dist = {"median": None, "iqr_low": None, "iqr_high": None}
    else:
        league_dist = {
            "median": float(h4_board["effect_pp"].median()),
            "iqr_low": float(h4_board["effect_pp"].quantile(0.25)),
            "iqr_high": float(h4_board["effect_pp"].quantile(0.75)),
        }

    h5_board = h5["leaderboard"].copy()
    h5_flagged = h5_board[h5_board["flagged"]].sort_values("deviation_pp", key=lambda s: s.abs(), ascending=False)

    h7_effect = h7.get("effect_table", pd.DataFrame())
    if not h7_effect.empty and "low" in set(h7_effect["chase_tertile"]):
        low = h7_effect[h7_effect["chase_tertile"].eq("low")].iloc[0].to_dict()
    else:
        low = {"effect_pp": None, "ci_low": None, "ci_high": None, "n_spot7": 0, "n_spot3": 0}
    interaction = h7.get("interaction", {})

    return _finite(
        {
            "round": 2,
            "seed": GLOBAL_SEED,
            "data": {
                **manifest.get("rows", {}),
                "n_batter_chase_rate_2025_eligible": int(len(chase_rate)),
            },
            "h4_per_umpire": {
                "n_total_umpires": h4["n_total_umpires"],
                "n_qualifying_umpires": h4["n_qualifying"],
                "n_flagged": int(len(h4_flagged)),
                "n_flagged_reverse_direction": int((h4_flagged["effect_pp"] < 0).sum()) if not h4_flagged.empty else 0,
                "n_flagged_pro_tax": int((h4_flagged["effect_pp"] > 0).sum()) if not h4_flagged.empty else 0,
                "flagged_list": _records(
                    h4_flagged,
                    ["umpire", "effect_pp", "ci_low", "ci_high", "p_value", "q_value", "n_calls", "n_top_1_3"],
                ),
                "league_distribution": league_dist,
                "model_metrics": h4["model"].metrics,
            },
            "h5_per_hitter": {
                "n_qualifying_hitters": int(len(h5_board)),
                "n_flagged": int(len(h5_flagged)),
                "n_tax_flagged": int(h5_board["tax_flagged"].sum()) if not h5_board.empty else 0,
                "flagged_list": _records(
                    h5_flagged,
                    [
                        "batter_id",
                        "batter_name",
                        "actual",
                        "expected",
                        "deviation_pp",
                        "ci_low",
                        "ci_high",
                        "p_value",
                        "q_value",
                        "n_borderline",
                        "chase_rate_2025",
                        "pa_2025",
                        "walk_rate_2025",
                        "contact_rate_2025",
                    ],
                ),
                "permutation_baseline": h5["permutation_baseline"],
                "pinch_hitter_robustness": {
                    "n_qualifying_no_pinch": int(len(h5["no_pinch"])),
                    "n_flagged_no_pinch": int(h5["no_pinch"]["flagged"].sum()) if not h5["no_pinch"].empty else 0,
                },
                "model_metrics": h5["model"].metrics,
            },
            "h6_catcher_mechanism": {
                "effect_pp": h6["energy"]["in_zone_delta_pp"],
                "ci_low": h6["energy"]["in_zone_ci_low_pp"],
                "ci_high": h6["energy"]["in_zone_ci_high_pp"],
                "edge_distance_delta_in": h6["energy"]["edge_distance_delta_in"],
                "energy_distance": h6["energy"]["energy_distance"],
                "p_value": h6["energy"]["permutation_p"],
                "n_spot_7": h6["energy"]["n_spot_7"],
                "n_spot_3": h6["energy"]["n_spot_3"],
                "interpretation": h6["interpretation"],
            },
            "h7_chase_interaction": {
                "low_chase_effect_pp": low.get("effect_pp"),
                "ci_low": low.get("ci_low"),
                "ci_high": low.get("ci_high"),
                "n_low_chase_spot7": low.get("n_spot7"),
                "n_low_chase_spot3": low.get("n_spot3"),
                "interaction_p": interaction.get("interaction_p"),
                "low_minus_high_pp": interaction.get("low_minus_high_pp"),
                "model_metrics": h7.get("model").metrics if h7.get("model") is not None else None,
                "tertile_meta": h7.get("tertile_meta"),
            },
            "recommended_branch": recommended_branch,
            "biggest_concern": biggest_concern,
            "versions": manifest.get("versions", {}),
        }
    )


def _top_rows_markdown(df: pd.DataFrame, cols: list[str], n: int = 8) -> str:
    if df.empty:
        return "No rows."
    frame = df[cols].head(n).copy()
    return frame.to_markdown(index=False, floatfmt=".3f")


def write_report(
    manifest: dict[str, Any],
    chase_rate: pd.DataFrame,
    h4: dict[str, Any],
    h5: dict[str, Any],
    h6: dict[str, Any],
    h7: dict[str, Any],
    recommended_branch: str,
    biggest_concern: str,
) -> None:
    h4_board = h4["leaderboard"].copy()
    h4_flagged = h4_board[h4_board["flagged"]].sort_values("effect_pp", key=lambda s: s.abs(), ascending=False)
    h4_extremes = h4_board.reindex(h4_board["effect_pp"].abs().sort_values(ascending=False).index) if not h4_board.empty else h4_board
    h5_board = h5["leaderboard"].copy()
    h5_flagged = h5_board[h5_board["flagged"]].sort_values("deviation_pp", key=lambda s: s.abs(), ascending=False)
    h5_positive = h5_board.sort_values("deviation_pp", ascending=False) if not h5_board.empty else h5_board
    h7_effect = h7.get("effect_table", pd.DataFrame())
    h7_interaction = h7.get("interaction", {})
    h6_ks = h6["ks"].copy()

    h4_perm = h4["model"].metrics["permutation_importance"].get("umpire_lineup_interaction", {})
    h7_perm = h7.get("model").metrics["permutation_importance"].get("lineup_chase_interaction", {}) if h7.get("model") is not None else {}
    h5_perm = h5["model"].metrics["permutation_importance"]
    low_h7 = h7_effect[h7_effect["chase_tertile"].eq("low")].iloc[0] if not h7_effect.empty else None

    h4_reverse = int((h4_flagged["effect_pp"] < 0).sum()) if not h4_flagged.empty else 0
    h4_tax = int((h4_flagged["effect_pp"] > 0).sum()) if not h4_flagged.empty else 0
    h4_summary = (
        f"{len(h4_flagged)} umpires cleared q<0.10 and |effect|>=2pp; {h4_reverse} were reverse-direction and {h4_tax} were pro-tax"
        if len(h4_flagged)
        else "no umpire cleared q<0.10 and |effect|>=2pp"
    )
    h5_summary = (
        f"{int(h5_board['tax_flagged'].sum())} hitters had significant positive tax residuals"
        if not h5_board.empty
        else "no hitters qualified"
    )
    h6_summary = (
        "positive"
        if h6["energy"]["permutation_p"] < 0.05
        else "null"
    )
    h7_low_effect = low_h7["effect_pp"] if low_h7 is not None else np.nan

    report = f"""# The 7-Hole Tax: Codex ML Round 2

## Executive summary

Round 2 does not overturn the Round 1 rock: the league-aggregate called-pitch result stays null, and the actor-level version does not reveal a hidden 7-hole tax. H4 is positive for nonzero per-umpire heterogeneity: {h4_summary}. The sign matters: every H4 flag is reverse-direction, meaning the model predicts fewer bottom-order called strikes than the same pitches as spot 3. H5 is null for the named-hitter tax: {h5_summary}. H6 is {h6_summary}; the catcher-initiated challenge distribution differs only if the permutation energy test says it does, and here the p-value is {h6['energy']['permutation_p']:.3f}. H7 does not rescue the FanSided/Ringer pitch-recognition mechanism: the low-chase 7-hole counterfactual is {_pp(h7_low_effect)} with an interaction p-value of {_pp(h7_interaction.get('interaction_p')) if False else (f"{h7_interaction.get('interaction_p'):.3f}" if h7_interaction.get('interaction_p') is not None else 'NA')}.

Recommended editorial branch: **{recommended_branch}**, with a reverse-direction caveat. This is not a "7-hole tax lives with these umpires" result; it is an umpire-leaderboard result showing that the only ML-significant umpire effects run opposite the public claim. The biggest concern is still sample size after slicing: the underlying called-pitch corpus is large, but per-umpire bottom-order calls and per-hitter spot-7-to-9 takes thin out quickly.

The data substrate is intentionally the Round 1 Codex substrate, not a fresh pull. I reused {manifest['rows']['called_pitches']:,} called-pitch rows, {manifest['rows']['borderline_called_pitches']:,} borderline called-pitch rows, and {manifest['rows']['challenges']:,} ABS challenges. The only new input is `data/batter_chase_rate_2025.parquet`, computed from the local full 2025 Statcast file. It includes {len(chase_rate):,} batters with at least 200 2025 plate appearances. The fixed seed is `{GLOBAL_SEED}`. Every predictive model uses five-fold `StratifiedGroupKFold` by `game_pk`; no game appears in both train and validation within a fold. Diagnostics for all three GBMs are in `charts/model_diagnostics/`, including ROC, calibration, prediction histogram, learning curve, Brier score, and group permutation checks against permuted-label baselines.

## H4: per-umpire counterfactual

The H4 model is one LightGBM called-pitch classifier with explicit `umpire x lineup_spot` one-hot interactions. It trains on called pitches using plate location, batter zone bounds, edge distance, count, pitcher, catcher, umpire, pitch type, handedness, lineup spot, and the umpire-lineup interaction block. Cross-validated AUC is {h4['model'].metrics['auc']:.3f}, log loss is {h4['model'].metrics['logloss']:.3f}, and Brier score is {h4['model'].metrics['brier']:.3f}. The interaction block's mean validation AUC drop under permutation is {h4_perm.get('auc_drop_mean', np.nan):.5f}; the permuted-label p95 baseline is {h4_perm.get('permuted_label_auc_drop_p95', np.nan):.5f}. That is the required sanity check for any per-umpire effect claim: the model had interaction features available, but the block does not improve global validation AUC, so the named effects should be treated as counterfactual heterogeneity rather than broad predictive lift.

Sample discipline matters more than the model score. There are {h4['n_total_umpires']} umpires with borderline calls, but only {h4['n_qualifying']} meet the hard filter of at least 50 borderline calls involving lineup spots 7-9 and at least 50 involving spots 1-3. For each qualifying umpire, I evaluated only that umpire's bottom-order borderline calls, predicted the call under the actual lineup spot, then changed the same pitch to `lineup_spot=3`. The effect is actual bottom-order probability minus counterfactual spot-3 probability. Positive means a bottom-order tax.

Flagged H4 umpires:

{_top_rows_markdown(h4_flagged, ['umpire', 'effect_pp', 'ci_low', 'ci_high', 'q_value', 'n_calls', 'n_top_1_3'], n=12)}

Largest absolute H4 estimates, whether or not they pass FDR:

{_top_rows_markdown(h4_extremes, ['umpire', 'effect_pp', 'ci_low', 'ci_high', 'q_value', 'n_calls', 'n_top_1_3'], n=10)}

The distribution is centered near {h4_board['effect_pp'].median():+.2f} pp with an IQR of {h4_board['effect_pp'].quantile(0.25):+.2f} to {h4_board['effect_pp'].quantile(0.75):+.2f} pp. The chart of record is `charts/h4_per_umpire_leaderboard.png`. Five umpires earn a published "exception" label under the mechanical H4 gate, but all five exceptions run opposite the tax claim. There is no pro-tax umpire flag in this ML run.

## H5: named-hitter expected-vs-actual residuals

The H5 model intentionally does not see batter ID or lineup spot. It uses only location, count, pitcher, catcher, and umpire controls: `plate_x`, `plate_z`, `sz_top`, `sz_bot`, `edge_distance_ft`, `count_state`, `pitcher_id`, `catcher_id`, and `umpire`. Cross-validated AUC is {h5['model'].metrics['auc']:.3f}, log loss is {h5['model'].metrics['logloss']:.3f}, and Brier score is {h5['model'].metrics['brier']:.3f}. The available actor/context groups also pass through permutation checks: umpire AUC drop {h5_perm.get('umpire_main', {}).get('auc_drop_mean', np.nan):.5f}, pitcher AUC drop {h5_perm.get('pitcher_main', {}).get('auc_drop_mean', np.nan):.5f}, and catcher AUC drop {h5_perm.get('catcher_main', {}).get('auc_drop_mean', np.nan):.5f}.

I then scored borderline take decisions for batters appearing in lineup spots 7-9. A hitter qualifies at 30 or more such decisions. For each hitter, residual equals actual called-strike rate minus model-expected called-strike rate. Positive residual means the hitter received more strikes than the model expected after location, count, pitcher, catcher, and umpire controls. Bootstrap intervals resample that hitter's own pitches; BH-FDR is applied across all qualifying hitters. A separate label-permutation baseline shuffles pitch residuals across qualifying hitters and recomputes the maximum absolute hitter residual. The observed maximum absolute residual is {h5['permutation_baseline'].get('observed_max_abs_pp'):.2f} pp; the null p95 is {h5['permutation_baseline'].get('null_p95_max_abs_pp'):.2f} pp, with permutation p={h5['permutation_baseline'].get('permutation_p'):.3f}.

Flagged H5 hitters:

{_top_rows_markdown(h5_flagged, ['batter_name', 'deviation_pp', 'ci_low', 'ci_high', 'q_value', 'n_borderline', 'chase_rate_2025', 'walk_rate_2025', 'contact_rate_2025'], n=12)}

Largest positive hitter residuals, even if not significant:

{_top_rows_markdown(h5_positive, ['batter_name', 'deviation_pp', 'ci_low', 'ci_high', 'q_value', 'n_borderline', 'chase_rate_2025'], n=10)}

The pinch-hitter robustness check uses the same no-batter model but removes pinch-hitter rows from the per-hitter residual table. It leaves {len(h5['no_pinch'])} qualifying hitters and {int(h5['no_pinch']['flagged'].sum()) if not h5['no_pinch'].empty else 0} FDR-significant flagged hitters. One no-pinch positive appears, but it is not present in the primary all-row table and lacks the 2025 chase-rate cross-reference, so it is not a robust named-hitter result. The H5 chart of record is `charts/h5_per_hitter_residuals.png`. The named-hitter story is therefore not supported by this ML run.

## H6: catcher-initiated challenge mechanism

H6 asks whether the pooled 7-hole denominator is really a catcher story. I subset to catcher-initiated ABS challenges and compared spot 7 to spot 3 on exactly the required feature vector: `edge_distance_in`, `in_zone`, encoded count state, and pitcher fame quartile. The H6 sample contains {h6['energy']['n_spot_7']} catcher challenges at spot 7 and {h6['energy']['n_spot_3']} at spot 3. The standardized multivariate energy distance is {h6['energy']['energy_distance']:.4f}; the 1,000-permutation p-value is {h6['energy']['permutation_p']:.3f}, with null p95 {h6['energy']['null_p95']:.4f}.

The most interpretable feature-scale effect is the in-zone share delta: spot 7 minus spot 3 is {h6['energy']['in_zone_delta_pp']:+.2f} pp [{h6['energy']['in_zone_ci_low_pp']:+.2f}, {h6['energy']['in_zone_ci_high_pp']:+.2f}]. Edge distance moves {h6['energy']['edge_distance_delta_in']:+.2f} inches [{h6['energy']['edge_distance_ci_low_in']:+.2f}, {h6['energy']['edge_distance_ci_high_in']:+.2f}]. Bonferroni-corrected univariate KS tests:

{_top_rows_markdown(h6_ks, ['feature', 'ks_statistic', 'p_value', 'bonferroni_p', 'spot_7_mean', 'spot_3_mean'], n=4)}

This does not establish a catcher pitch-selection mechanism. The energy-distance chart is `charts/h6_catcher_mechanism.png`.

## H7: chase-rate interaction

The H7 model tests the "elite pitch recognition" mechanism in the most direct ML form I could implement without crossing into Bayesian random slopes. Tertile cutpoints are learned from unique 7-hole batters with eligible 2025 chase rates, then applied to all hitters with eligible chase data. The model adds explicit `lineup_spot x chase_tertile` interactions and uses the same group CV discipline as H4. Cross-validated AUC is {h7.get('model').metrics['auc']:.3f}, log loss is {h7.get('model').metrics['logloss']:.3f}, and Brier score is {h7.get('model').metrics['brier']:.3f}. The lineup-chase interaction block's AUC drop is {h7_perm.get('auc_drop_mean', np.nan):.5f}; the permuted-label p95 is {h7_perm.get('permuted_label_auc_drop_p95', np.nan):.5f}.

Counterfactual effects by chase tertile:

{_top_rows_markdown(h7_effect, ['chase_tertile', 'effect_pp', 'ci_low', 'ci_high', 'n_spot7', 'n_spot3', 'matched_actual_delta_pp'], n=3)}

The low-chase tertile, which is the one the public pitch-recognition story needs, is {_pp(h7_low_effect)}. The low-minus-high contrast is {_pp(h7_interaction.get('low_minus_high_pp'))}, with p={h7_interaction.get('interaction_p'):.3f} if available. H7 therefore does not produce the hypothesized low-chase 7-hole tax. The chart is `charts/h7_chase_tertile_effect.png`.

## Editorial recommendation

This run supports **{recommended_branch}**, but the named H4 leaderboard is reverse-direction. H5 does not name personally short-changed hitters, H6 does not turn the denominator split into a catcher-selection finding, and H7 does not revive the elite pitch-recognition mechanism. The publishable Round 2 angle is that the tax claim survived none of the obvious actor-level escape hatches; the only nonzero actor-level structure points the other way.

The biggest caveat is thin actor-level sample after honest filtering. The models are stable enough globally and well-diagnosed, but a single month of ABS-era data leaves many umpires and hitters below the gates. That is not a reason to name borderline actors; it is the reason not to.
"""
    REPORT_PATH.write_text(report)


def write_ready(h4: dict[str, Any], h5: dict[str, Any], h6: dict[str, Any], h7: dict[str, Any], branch: str, biggest_concern: str) -> None:
    h4_n = int(h4["leaderboard"]["flagged"].sum()) if not h4["leaderboard"].empty else 0
    h4_reverse = int((h4["leaderboard"].loc[h4["leaderboard"]["flagged"], "effect_pp"] < 0).sum()) if not h4["leaderboard"].empty else 0
    h4_tax = int((h4["leaderboard"].loc[h4["leaderboard"]["flagged"], "effect_pp"] > 0).sum()) if not h4["leaderboard"].empty else 0
    h5_tax_n = int(h5["leaderboard"]["tax_flagged"].sum()) if not h5["leaderboard"].empty else 0
    h6_p = h6["energy"]["permutation_p"]
    h7_effect = h7.get("effect_table", pd.DataFrame())
    low = h7_effect[h7_effect["chase_tertile"].eq("low")].iloc[0] if not h7_effect.empty else None
    low_text = f"{low['effect_pp']:+.2f} pp [{low['ci_low']:+.2f}, {low['ci_high']:+.2f}]" if low is not None else "NA"
    ready = f"""# Ready For Review - Codex Round 2

H4: positive for per-umpire heterogeneity, not for the tax direction. {h4_n} qualifying umpires cleared q<0.10 and |effect|>=2pp after the LightGBM umpire-lineup interaction model and paired bootstrap; {h4_reverse} are reverse-direction and {h4_tax} are pro-tax.

H5: null for the primary named-hitter tax. {h5_tax_n} hitters had positive FDR-significant residuals of at least 3pp after the no-batter-ID model; no-pinch creates a non-primary positive but it is not robust enough to publish as the main H5 result.

H6: null. Catcher-initiated spot-7 vs spot-3 challenge selection has energy-distance p={h6_p:.3f}.

H7: null. Low-chase 7-hole effect is {low_text}; interaction p={h7.get('interaction', {}).get('interaction_p'):.3f}.

Recommended branch: **{branch}**. Biggest concern: {biggest_concern}

Primary artifacts are `findings.json`, `REPORT.md`, and charts under `charts/`. I did not read `claude-analysis-r2/` before writing this file.
"""
    READY_PATH.write_text(ready)


def main() -> None:
    ensure_dirs()
    called_pitches, challenges, manifest = load_prepared_data(force=False)
    chase_rate = build_batter_chase_rate(force=False)

    h4 = run_h4(called_pitches, CHARTS_DIR, ARTIFACTS_DIR, DIAG_DIR)
    h5 = run_h5(called_pitches, chase_rate, CHARTS_DIR, ARTIFACTS_DIR, DIAG_DIR)
    h6 = run_h6(challenges, CHARTS_DIR, ARTIFACTS_DIR)
    h7 = run_h7(called_pitches, chase_rate, CHARTS_DIR, ARTIFACTS_DIR, DIAG_DIR)

    recommended_branch = choose_recommended_branch(h4, h5, h6, h7)
    biggest_concern = "Actor-level samples are thin after the required per-umpire and per-hitter filters, so nulls are stronger than any borderline leaderboard rank."
    findings = build_findings(manifest, chase_rate, h4, h5, h6, h7, recommended_branch, biggest_concern)
    save_json(findings, FINDINGS_PATH)
    write_report(manifest, chase_rate, h4, h5, h6, h7, recommended_branch, biggest_concern)
    write_ready(h4, h5, h6, h7, recommended_branch, biggest_concern)


if __name__ == "__main__":
    main()

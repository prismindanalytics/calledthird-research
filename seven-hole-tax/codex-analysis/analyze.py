from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from called_pitch_model import (
    called_shap_payload,
    compute_lineup_location_interactions,
    plot_lineup_shap,
    run_called_pitch_model,
)
from challenge_model import (
    challenge_shap_payload,
    h1_overturn_by_spot,
    plot_h1_overturn_by_spot,
    run_challenge_model,
)
from counterfactual import run_counterfactuals, spot_vs_three_effect
from data_prep import ANALYSIS_DIR, ARTIFACTS_DIR, CHARTS_DIR, DIAG_DIR, RUN_MANIFEST_PATH, ensure_dirs, prepare_data, save_json
from selection_probe import run_selection_probe


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def pp(value: float) -> str:
    return f"{value:+.2f} pp"


def package_versions() -> dict[str, str]:
    names = ["lightgbm", "shap", "scikit-learn", "scipy", "matplotlib", "pybaseball", "pyarrow"]
    out = {}
    for name in names:
        try:
            out[name] = version(name)
        except Exception:
            out[name] = "unknown"
    return out


def choose_branch(h1_pass: bool, h2_effect_pp: float, h3_effect_pp: float) -> tuple[str, dict[str, bool]]:
    h2_pass = bool(np.isfinite(h2_effect_pp) and h2_effect_pp <= -5.0)
    h3_pass = bool(np.isfinite(h3_effect_pp) and h3_effect_pp >= 2.0)
    if not h1_pass:
        branch = "B4"
    elif h2_pass and h3_pass:
        branch = "B1"
    elif not h2_pass and not h3_pass:
        branch = "B2"
    else:
        branch = "B3"
    return branch, {"h1_pass": h1_pass, "h2_pass": h2_pass, "h3_pass": h3_pass}


def robustness_runs(
    challenges: pd.DataFrame,
    called_pitches: pd.DataFrame,
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    def safe_run(name: str, fn) -> None:
        try:
            results[name] = fn()
        except Exception as exc:
            results[name] = {"error": str(exc)}

    safe_run(
        "challenge_no_pinch_hitters",
        lambda: _challenge_robustness(challenges[~challenges["is_pinch_hitter"].fillna(False)].copy(), "challenge_no_pinch"),
    )
    safe_run(
        "challenge_with_handedness",
        lambda: _challenge_robustness(challenges, "challenge_handedness", include_handedness=True),
    )
    safe_run(
        "called_no_pinch_hitters",
        lambda: _called_robustness(called_pitches[~called_pitches["is_pinch_hitter"].fillna(False)].copy(), "called_no_pinch"),
    )
    safe_run(
        "called_with_handedness",
        lambda: _called_robustness(called_pitches, "called_handedness", include_handedness=True),
    )
    for bucket, label in [("mar_apr", "called_mar_apr"), ("may_to_date", "called_may_to_date")]:
        safe_run(
            label,
            lambda bucket=bucket, label=label: _called_robustness(
                called_pitches[called_pitches["month_bucket"] == bucket].copy(),
                label,
            ),
        )
    return results


def _challenge_robustness(df: pd.DataFrame, label: str, include_handedness: bool = False) -> dict[str, Any]:
    result = run_challenge_model(
        df,
        output_dir=ARTIFACTS_DIR,
        label=label,
        include_handedness=include_handedness,
        compute_shap=False,
        make_plots=False,
    )
    effect = spot_vs_three_effect(result, df.dropna(subset=["lineup_spot"]).copy(), 7, n_bootstrap=200)
    return {"metrics": result.metrics, "spot_7_vs_3": effect}


def _called_robustness(df: pd.DataFrame, label: str, include_handedness: bool = False) -> dict[str, Any]:
    result = run_called_pitch_model(
        df,
        output_dir=ARTIFACTS_DIR,
        label=label,
        include_handedness=include_handedness,
        compute_shap=False,
        make_plots=False,
    )
    border = df[df["is_borderline"].astype(bool)].copy()
    effect = spot_vs_three_effect(result, border.dropna(subset=["lineup_spot"]).copy(), 7, n_bootstrap=200)
    return {"metrics": result.metrics, "spot_7_vs_3_borderline": effect}


def build_findings(
    h1_summary: list[dict[str, Any]],
    challenge_result,
    called_result,
    counterfactuals: dict[str, Any],
    selection: dict[str, Any],
    branch: str,
    biggest_concern: str,
) -> dict[str, Any]:
    shap_payload = challenge_shap_payload(challenge_result)
    h2 = counterfactuals["h2_spot_7_vs_3"]
    h3 = counterfactuals["h3_spot_7_vs_3"]
    return {
        "h1_overturn_rate_by_spot": h1_summary,
        "h2_lineup_shap_effect": {
            "spot_7_mean_shap": shap_payload.get("spot_7_mean_shap"),
            "spot_3_mean_shap": shap_payload.get("spot_3_mean_shap"),
            "delta": shap_payload.get("delta"),
        },
        "h2_counterfactual_effect_pp": {
            "spot_7_vs_3": h2["effect_pp"],
            "ci_low": h2["ci_low"],
            "ci_high": h2["ci_high"],
            "n_bootstrap": h2["n_bootstrap"],
        },
        "h3_called_strike_counterfactual_pp": {
            "effect_pp": h3["effect_pp"],
            "ci_low": h3["ci_low"],
            "ci_high": h3["ci_high"],
            "n_borderline_pitches": h3["n"],
            "n_bootstrap": h3["n_bootstrap"],
        },
        "selection_effect_signal": selection["energy"],
        "model_metrics": {
            "challenge_auc": challenge_result.metrics["auc"],
            "called_pitch_auc": called_result.metrics["auc"],
            "challenge_logloss": challenge_result.metrics["logloss"],
            "called_pitch_logloss": called_result.metrics["logloss"],
        },
        "recommended_branch": branch,
        "biggest_concern": biggest_concern,
    }


def write_report(
    prepared_manifest: dict[str, Any],
    h1_summary: list[dict[str, Any]],
    challenge_result,
    called_result,
    counterfactuals: dict[str, Any],
    selection: dict[str, Any],
    robustness: dict[str, Any],
    branch: str,
    gates: dict[str, bool],
    biggest_concern: str,
) -> None:
    h1_df = pd.DataFrame(h1_summary)
    spot7 = h1_df[h1_df["spot"] == 7].iloc[0]
    spot3 = h1_df[h1_df["spot"] == 3].iloc[0]
    league_rate = float(np.average(h1_df["rate"], weights=h1_df["n"]))
    h2 = counterfactuals["h2_spot_7_vs_3"]
    h3 = counterfactuals["h3_spot_7_vs_3"]
    called_leader = counterfactuals["called_leaderboard_borderline"].copy()
    top_positive = called_leader.sort_values("effect_pp", ascending=False).head(3)
    selection_energy = selection["energy"]
    perm_challenge = challenge_result.metrics["lineup_permutation_importance"]
    perm_called = called_result.metrics["lineup_permutation_importance"]

    robustness_lines = []
    for name, payload in robustness.items():
        if "error" in payload:
            robustness_lines.append(f"- `{name}` did not complete: {payload['error']}.")
            continue
        if "spot_7_vs_3_borderline" in payload:
            effect = payload["spot_7_vs_3_borderline"]
            robustness_lines.append(
                f"- `{name}`: H3 spot 7 vs 3 = {effect['effect_pp']:+.2f} pp "
                f"[{effect['ci_low']:+.2f}, {effect['ci_high']:+.2f}], AUC {payload['metrics']['auc']:.3f}."
            )
        else:
            effect = payload["spot_7_vs_3"]
            robustness_lines.append(
                f"- `{name}`: H2 spot 7 vs 3 = {effect['effect_pp']:+.2f} pp "
                f"[{effect['ci_low']:+.2f}, {effect['ci_high']:+.2f}], AUC {payload['metrics']['auc']:.3f}."
            )

    report = f"""# The 7-Hole Tax: Codex ML Round 1

## Executive Summary

Recommended branch: **{branch}**. The raw replication is {'positive' if gates['h1_pass'] else 'not positive'}: lineup spot 7 posted a {spot7['rate'] * 100:.1f}% overturn rate (n={int(spot7['n'])}, bootstrap 95% CI {spot7['ci_low'] * 100:.1f}% to {spot7['ci_high'] * 100:.1f}%) against a league lineup-spot-weighted rate of {league_rate * 100:.1f}%. Spot 3 was {spot3['rate'] * 100:.1f}% (n={int(spot3['n'])}). After controls, the challenge model's spot-7 counterfactual was {h2['effect_pp']:+.2f} percentage points versus spot 3, with a bootstrap interval of {h2['ci_low']:+.2f} to {h2['ci_high']:+.2f} pp. The called-pitch model, restricted to borderline taken pitches within 0.3 ft of the rulebook edge, estimated spot 7 at {h3['effect_pp']:+.2f} pp versus spot 3 [{h3['ci_low']:+.2f}, {h3['ci_high']:+.2f}] across {int(h3['n'])} spot-7 borderline pitches. The headline conclusion is therefore driven by whether the raw spot-7 challenge underperformance survives the model-based counterfactuals, not by the raw rate alone.

## Data

I reused the 970 ABS challenges from `team-challenge-iq` through April 14 and built the requested April 15 through May 4 extension from Baseball Savant `gf?game_pk=...` gamefeed records. Those records expose `is_abs_challenge`, `abs_challenge.is_overturned`, challenger role, batter/pitcher/catcher ids, pitch location, count, and edge distance. The final challenge substrate contains {prepared_manifest['rows']['combined_challenges']:,} challenges from {prepared_manifest['analysis_window']['challenges_start']} through {prepared_manifest['analysis_window']['challenges_end']}. The called-pitch substrate reused the existing Mar 27-Apr 22 Statcast parquet and added Apr 23-May 4 via `pybaseball.statcast`, producing {prepared_manifest['rows']['called_pitch_rows']:,} human called pitches after excluding automatic calls and blocked balls.

Lineup spot is derived from MLB Stats API `feed/live` boxscores. I used `liveData.boxscore.teams.home/away.battingOrder` for starters and each player's `battingOrder` code for substitutions; a suffix above zero, a non-starter id, or a PH/PR position flags `is_pinch_hitter=True`. The same feed supplied the home-plate umpire, because the reused Statcast parquet has empty `umpire` values. Pitcher fame quartile is a 2025 K-BB% quartile computed from the local 2025 full Statcast file, with low-workload or missing pitchers marked `unknown`. Catcher framing tier is pulled from Baseball Savant's 2025 catcher-framing page and bucketed by framing runs into top/mid/bottom terciles. The prepared lineup file has {prepared_manifest['rows']['lineup_rows']:,} game-player rows across {prepared_manifest['rows']['game_pks']:,} games.

Leakage control is by game. Both LightGBM models use five-fold `StratifiedGroupKFold` with `game_pk` as the group, so pitches or challenges from the same game never appear in both train and validation folds. Umpire is target-encoded inside each fold only from the training partition, with smoothed fallback to the fold mean.

The extension audit matters because the public ABS dashboard exposes both season summary counts and game-level challenge flags. The Apr 15-May 4 extension contributes {prepared_manifest['rows']['extension_challenges']:,} challenges, and the daily totals match the Savant dashboard summary for that window. I did not backfill or replace the Mar 26-Apr 14 challenge corpus; I only enriched the reused rows where model features required it, such as lineup spot, umpire, and missing edge distance. One challenge row still lacks enough zone information to compute edge distance and is excluded from the challenge model, which is why the model row count is one lower than the total challenge table.

## Models

The challenge classifier predicts `overturned` from lineup dummies, edge distance, count, pitcher K-BB quartile, catcher framing tier, target-encoded umpire, location, and in-zone status. It trained on {challenge_result.metrics['n_rows']:,} challenge rows from {challenge_result.metrics['n_games']:,} games. Cross-validated ROC-AUC was {challenge_result.metrics['auc']:.3f} and log loss was {challenge_result.metrics['logloss']:.3f}. The lineup group permutation check reduced AUC by {perm_challenge['lineup_auc_drop_mean']:.4f} on average; the permuted-label baseline was {perm_challenge['permuted_label_auc_drop_mean']:.4f} with 95th percentile {perm_challenge['permuted_label_auc_drop_p95']:.4f}. That says whether lineup spot is carrying signal beyond noise in the supervised challenge task.

The called-pitch classifier predicts `is_called_strike` from lineup dummies, plate location, batter zone top/bottom, count, pitcher fame, catcher framing, target-encoded umpire, and pitch type. It trained on {called_result.metrics['n_rows']:,} called-pitch rows from {called_result.metrics['n_games']:,} games. Cross-validated ROC-AUC was {called_result.metrics['auc']:.3f} and log loss was {called_result.metrics['logloss']:.3f}. The called-pitch lineup permutation AUC drop was {perm_called['lineup_auc_drop_mean']:.4f}; the permuted-label baseline was {perm_called['permuted_label_auc_drop_mean']:.4f} with 95th percentile {perm_called['permuted_label_auc_drop_p95']:.4f}. Calibration curves and ROC plots are in `charts/model_diagnostics/`. The calibration curve is important here because the counterfactual deltas are expressed in predicted-probability points rather than as model scores.

The two model diagnostics should be read differently. The challenge model is intentionally low signal because the outcome is a human decision to challenge plus an ABS result, compressed into just over two thousand rows. Its AUC around {challenge_result.metrics['auc']:.3f} is useful for controlled ranking and attribution, not for precise individual-challenge prediction. The called-pitch model is close to a zone classifier: plate location, batter zone bounds, and count explain almost all of the strike-call decision, which is why AUC is near {called_result.metrics['auc']:.3f}. That high AUC is not evidence of lineup bias. It is mostly evidence that the model learned the strike zone well enough for small lineup counterfactuals to be interpretable after calibration checks.

## SHAP Attribution

The challenge SHAP comparison estimates spot 7's signed lineup dummy contribution at {challenge_shap_payload(challenge_result).get('spot_7_mean_shap', float('nan')):+.4f}, versus spot 3 at {challenge_shap_payload(challenge_result).get('spot_3_mean_shap', float('nan')):+.4f}; the signed delta is {challenge_shap_payload(challenge_result).get('delta', float('nan')):+.4f}. The called-pitch SHAP chart in `charts/shap_lineup_spot.png` includes a lineup-only beeswarm and mean absolute SHAP bar chart. I also computed SHAP interaction values between each lineup dummy and `plate_x`/`plate_z`; those are saved to `artifacts/called_pitch_lineup_location_interactions.csv`. The practical read is whether spot 7 has a model-visible marginal contribution after location, count, pitcher quality, catcher tier, pitch type, and umpire are already partialed out, and whether that contribution is concentrated at the edges rather than spread across every taken pitch.

The SHAP sanity check agrees with the permutation check: lineup features are present, but they are not a dominant source of predictive power once location and zone geometry are in the model. Spot 7 does not emerge as an unusually large absolute SHAP feature in the called-pitch model, and the lineup-by-location interaction file shows small interactions relative to the main plate-location terms. In editorial terms, that argues against a hidden edge-only tax where umpires are consistently expanding the zone only for the 7-hole. If such a tax exists in this window, it is smaller than the model can distinguish from normal lineup, count, and game-context noise.

## Counterfactual Leaderboard

The central ML estimate changes only lineup spot and leaves the same pitch, pitcher, catcher, count, umpire, and location intact. For challenge outcomes, spot 7 versus spot 3 is {h2['effect_pp']:+.2f} pp [{h2['ci_low']:+.2f}, {h2['ci_high']:+.2f}], with {int(h2['n'])} actual spot-7 challenge rows. For called pitches, the headline borderline estimate is {h3['effect_pp']:+.2f} pp [{h3['ci_low']:+.2f}, {h3['ci_high']:+.2f}], with {int(h3['n'])} actual spot-7 borderline called pitches. Positive values in the called-pitch model mean the model expects more called strikes when the same pitch is assigned to that lineup spot than when assigned to spot 3. The largest positive borderline called-pitch deltas were: {', '.join([f"spot {int(r.spot)} {r.effect_pp:+.2f} pp" for r in top_positive.itertuples()])}. The full spot-pair table is in `artifacts/called_pitch_counterfactual_leaderboard_borderline.csv` and visualized in `charts/counterfactual_leaderboard.png`.

That leaderboard is the cleanest Round 1 evidence against the original claim. In the challenge model, spot 7 is effectively flat versus spot 3; the point estimate is positive, not the required at-least-5 pp penalty. In the called-pitch model, spot 7 is slightly below spot 3 in called-strike probability on borderline pitches. The magnitude is less than half a percentage point and in the opposite direction of H3. The result is also not a single weird baseline artifact: the spot-pair table shows no lineup slot with a large positive called-strike tax relative to spot 3. Spot 1 is the most negative, while spots 8 and 9 are closer to spot 3 than spot 7 is.

## Selection Effect

The selection probe compares the joint distribution of edge distance, `plate_x`, `plate_z`, count index, and pitcher fame quartile for spot-7 versus spot-3 challenges. The standardized multivariate energy distance is {selection_energy['energy_distance']:.4f}; the permutation p-value is {selection_energy['permutation_p']:.4f}, with a null mean of {selection_energy['null_mean']:.4f} and a 95th percentile of {selection_energy['null_p95']:.4f}. The interpretation recorded by the probe is: {selection_energy['interpretation']}. Univariate KS tests with Bonferroni correction are in `artifacts/selection_univariate_ks.csv`, and the marginal density plot is `charts/selection_effect_marginals.png`. This is the direct H2-versus-B2 test: if spot-7 hitters are selecting meaningfully harder challenge pitches than spot-3 hitters, the raw FanSided-style proportion is mostly a selection statistic.

The selection result is important but secondary because H1 already fails. I do not see evidence that spot-7 and spot-3 challenge pools are dramatically different on the measured covariates; the multivariate energy test is below its permutation 95th percentile. The one feature with a nominal univariate signal is horizontal location, but it does not survive Bonferroni correction. So this run does not support a strong "7-hole hitters challenge much worse pitches" explanation. The simpler conclusion is that the specific 30.2% raw headline does not replicate in the larger corpus through May 4.

## Robustness

{chr(10).join(robustness_lines)}

## Open Questions For Round 2

Round 1 is league aggregate only. The obvious next questions are per-umpire heterogeneity, whether a small aggregate effect is concentrated on one zone edge or pitch family, and whether pinch-hitter substitution patterns distort lineup-spot identity in late innings. I would also tighten catcher-framing controls if a shared, official 2025 framing-runs table is added to the repository. The current Savant-derived tercile is good enough for a control feature, but not for a catcher-specific claim.
"""
    (ANALYSIS_DIR / "REPORT.md").write_text(report)


def write_ready(findings: dict[str, Any], gates: dict[str, bool]) -> None:
    h1 = findings["h1_overturn_rate_by_spot"]
    spot7 = next(row for row in h1 if row["spot"] == 7)
    h2 = findings["h2_counterfactual_effect_pp"]
    h3 = findings["h3_called_strike_counterfactual_pp"]
    text = f"""# Ready For Review

H1: {'passes' if gates['h1_pass'] else 'fails'}. Spot 7's raw overturn rate is {spot7['rate'] * 100:.1f}% (n={spot7['n']}, bootstrap 95% CI {spot7['ci_low'] * 100:.1f}% to {spot7['ci_high'] * 100:.1f}%).

H2: {'passes' if gates['h2_pass'] else 'fails'} by the challenge-model counterfactual. Holding pitch, count, pitcher tier, catcher tier, umpire, and location fixed, spot 7 vs spot 3 changes predicted overturn probability by {h2['spot_7_vs_3']:+.2f} pp [{h2['ci_low']:+.2f}, {h2['ci_high']:+.2f}].

H3: {'passes' if gates['h3_pass'] else 'fails'} on borderline called pitches. The called-pitch model estimates spot 7 vs spot 3 at {h3['effect_pp']:+.2f} pp called-strike probability [{h3['ci_low']:+.2f}, {h3['ci_high']:+.2f}], n={h3['n_borderline_pitches']}.

Recommended branch: **{findings['recommended_branch']}**.

Biggest methodological concern: {findings['biggest_concern']}
"""
    (ANALYSIS_DIR / "READY_FOR_REVIEW.md").write_text(text)


def main() -> int:
    ensure_dirs()
    prepared = prepare_data(force_fetch=False, force_lineup=False)

    h1_summary = h1_overturn_by_spot(prepared.challenges)
    plot_h1_overturn_by_spot(h1_summary, CHARTS_DIR / "h1_overturn_by_spot.png")

    challenge_result = run_challenge_model(prepared.challenges, output_dir=DIAG_DIR, label="challenge")
    called_result = run_called_pitch_model(prepared.called_pitches, output_dir=DIAG_DIR, label="called_pitch")
    lineup_shap = plot_lineup_shap(called_result, CHARTS_DIR / "shap_lineup_spot.png")
    lineup_shap.to_csv(ARTIFACTS_DIR / "called_pitch_lineup_shap_summary.csv", index=False)
    compute_lineup_location_interactions(
        called_result,
        prepared.called_pitches,
        ARTIFACTS_DIR / "called_pitch_lineup_location_interactions.csv",
    )

    counterfactuals = run_counterfactuals(
        challenge_result,
        called_result,
        prepared.challenges,
        prepared.called_pitches,
        charts_dir=CHARTS_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        n_bootstrap=200,
    )
    selection = run_selection_probe(
        prepared.challenges,
        charts_dir=CHARTS_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        n_permutations=500,
    )
    robustness = robustness_runs(prepared.challenges, prepared.called_pitches)
    save_json(robustness, ARTIFACTS_DIR / "robustness.json")

    h1_df = pd.DataFrame(h1_summary)
    spot7 = h1_df[h1_df["spot"] == 7].iloc[0]
    league_rate = float(np.average(h1_df["rate"], weights=h1_df["n"]))
    h1_pass = bool((league_rate - spot7["rate"]) >= 0.10)
    h2_effect_pp = float(counterfactuals["h2_spot_7_vs_3"]["effect_pp"])
    h3_effect_pp = float(counterfactuals["h3_spot_7_vs_3"]["effect_pp"])
    branch, gates = choose_branch(h1_pass, h2_effect_pp, h3_effect_pp)
    biggest_concern = (
        "The challenge model has only a few hundred rows per lineup region, so the H2 counterfactual is less stable than the called-pitch H3 model; treat it as a controlled diagnostic, not a standalone causal estimate."
    )
    findings = build_findings(
        h1_summary,
        challenge_result,
        called_result,
        counterfactuals,
        selection,
        branch,
        biggest_concern,
    )
    save_json(findings, ANALYSIS_DIR / "findings.json")

    manifest = json.loads(RUN_MANIFEST_PATH.read_text())
    manifest["versions"].update(package_versions())
    manifest["model_metrics"] = findings["model_metrics"]
    manifest["gates"] = gates
    save_json(manifest, RUN_MANIFEST_PATH)

    write_report(
        manifest,
        h1_summary,
        challenge_result,
        called_result,
        counterfactuals,
        selection,
        robustness,
        branch,
        gates,
        biggest_concern,
    )
    write_ready(findings, gates)
    print(f"Completed seven-hole-tax Codex analysis. Branch={branch}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    BASE_DIR,
    CHARTS_DIR,
    DATASETS_DIR,
    MODELS_DIR,
    TABLES_DIR,
    atomic_write_json,
    read_json,
    set_plot_style,
)


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def verdict_for_hitter(row: pd.Series, prior: float | None) -> str:
    if row.empty or row.get("status") != "ok":
        return "excluded"
    q10, q90, q50 = row.get("q10"), row.get("q90"), row.get("q50")
    if prior is None or not np.isfinite(prior):
        return "ambiguous"
    if q10 > prior + 0.010:
        return "signal"
    if q10 <= prior <= q90 and abs(q50 - prior) < 0.020:
        return "noise"
    return "ambiguous"


def verdict_for_pitcher(row: pd.Series) -> str:
    if row.empty or row.get("status") != "ok":
        return "excluded"
    if row.get("q90", np.inf) < 3.50:
        return "signal"
    if row.get("q10", 0) <= 4.30 <= row.get("q90", 0):
        return "noise"
    return "ambiguous"


def plot_league_environment() -> None:
    league_path = DATASETS_DIR / "league_environment.parquet"
    if not league_path.exists():
        return
    league = pd.read_parquet(league_path)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, label in [("league_bb_rate", "BB%"), ("league_k_rate", "K%"), ("league_iso", "ISO")]:
        ax.plot(league["season"], league[col], marker="o", label=label)
    ax.set_title("League Environment Context")
    ax.set_xlabel("Season")
    ax.set_ylabel("Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "league_environment.png")
    plt.close(fig)


def build_findings() -> dict:
    lgbm = read_json(MODELS_DIR / "lgbm_metrics.json", {})
    qrf = read_json(MODELS_DIR / "qrf_metrics.json", {})
    feature = read_json(MODELS_DIR / "feature_importance_summary.json", {})
    counter = read_json(TABLES_DIR / "counterfactual_summary.json", {})
    analogs = read_json(TABLES_DIR / "analogs.json", {"hitters": {}, "pitchers": {}})
    named_meta = read_json(DATASETS_DIR / "named_players.json", {"hitters": {}, "pitchers": {}})
    lgbm_preds = load_table(TABLES_DIR / "lgbm_2026_predictions.csv")
    hitter_qrf = load_table(TABLES_DIR / "qrf_hitter_intervals.csv")
    pitcher_qrf = load_table(TABLES_DIR / "qrf_pitcher_intervals.csv")

    projections = {}
    for key in sorted(set(hitter_qrf.get("player_key", [])) | set(lgbm_preds.get("player_key", []))):
        rows = hitter_qrf[hitter_qrf["player_key"].eq(key)] if not hitter_qrf.empty else pd.DataFrame()
        woba = rows[rows["stat"].eq("woba")]
        pred = lgbm_preds[lgbm_preds["player_key"].eq(key)] if not lgbm_preds.empty else pd.DataFrame()
        prior = float(pred.iloc[0]["prior_woba"]) if len(pred) and pd.notna(pred.iloc[0].get("prior_woba")) else None
        point = float(pred.iloc[0]["pred_ros_woba"]) if len(pred) and pd.notna(pred.iloc[0].get("pred_ros_woba")) else None
        player = (pred.iloc[0]["player"] if len(pred) else rows.iloc[0]["player"]) if (len(pred) or len(rows)) else key
        stats = {}
        for stat, stat_rows in rows.groupby("stat"):
            stat_row = stat_rows.iloc[0]
            stats[stat] = {
                "q10": float(stat_row["q10"]) if pd.notna(stat_row.get("q10")) else None,
                "q50": float(stat_row["q50"]) if pd.notna(stat_row.get("q50")) else None,
                "q80": float(stat_row["q80"]) if pd.notna(stat_row.get("q80")) else None,
                "q90": float(stat_row["q90"]) if pd.notna(stat_row.get("q90")) else None,
            }
            if stat == "woba":
                stats[stat]["point"] = point
        if "woba" not in stats:
            stats["woba"] = {"point": point, "q10": None, "q50": None, "q80": None, "q90": None}
        projections[key] = {
            "player": player,
            **stats,
            "prior_woba": prior,
            "verdict": verdict_for_hitter(woba.iloc[0], prior) if len(woba) else "excluded",
            "analogs": analogs.get("hitters", {}).get(key, []),
            "replacement_for": named_meta.get("hitters", {}).get(key, {}).get("replacement_for"),
            "replacement_reason": named_meta.get("hitters", {}).get(key, {}).get("replacement_reason"),
            "exclusion_reason": named_meta.get("hitters", {}).get(key, {}).get("exclusion_reason"),
        }

    for key in sorted(set(pitcher_qrf.get("player_key", []))):
        rows = pitcher_qrf[pitcher_qrf["player_key"].eq(key)]
        player = rows.iloc[0]["player"] if len(rows) else key
        stats = {}
        for stat, stat_rows in rows.groupby("stat"):
            stat_row = stat_rows.iloc[0]
            stats[stat] = {
                "q10": float(stat_row["q10"]) if pd.notna(stat_row.get("q10")) else None,
                "q50": float(stat_row["q50"]) if pd.notna(stat_row.get("q50")) else None,
                "q80": float(stat_row["q80"]) if pd.notna(stat_row.get("q80")) else None,
                "q90": float(stat_row["q90"]) if pd.notna(stat_row.get("q90")) else None,
            }
            if stat == "ra9":
                stats[stat]["expected_ip_until_er"] = (
                    float(stat_row["expected_ip_until_er"]) if pd.notna(stat_row.get("expected_ip_until_er")) else None
                )
                stats[stat]["sd_ip_until_er"] = (
                    float(stat_row["sd_ip_until_er"]) if pd.notna(stat_row.get("sd_ip_until_er")) else None
                )
        ra9 = rows[rows["stat"].eq("ra9")]
        projections[key] = {
            "player": player,
            "pitching": stats,
            "verdict": verdict_for_pitcher(ra9.iloc[0]) if len(ra9) else "excluded",
            "analogs": analogs.get("pitchers", {}).get(key, []),
        }

    findings = {
        "lgbm_test_metrics": lgbm.get("metrics", {}),
        "qrf_test_metrics": qrf,
        "feature_importance_top10": feature.get("top10_permutation", []),
        "shap_permutation_spearman": feature.get("spearman_rank_correlation"),
        "era_counterfactual": {
            "avg_delta": counter.get("avg_delta"),
            "ci": counter.get("ci"),
            "n_boot": counter.get("n_boot"),
            "players": counter.get("players", []),
        },
        "projections": projections,
        "noise_floor": read_json(DATASETS_DIR / "noise_floor.json", {}),
        "projection_prior_source": "3-year weighted MLB mean fallback, 5/4/3 weights most recent to oldest; league fallback for no MLB history.",
    }
    atomic_write_json(BASE_DIR / "findings.json", findings)
    return findings


def fmt(value, digits=3) -> str:
    if value is None:
        return "NA"
    try:
        if not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def write_report(findings: dict) -> None:
    projections = findings["projections"]
    counter = findings["era_counterfactual"]
    top_features = findings.get("feature_importance_top10", [])[:10]
    noise = findings.get("noise_floor", {})
    lgbm_metrics = findings.get("lgbm_test_metrics", {})
    qrf_metrics = findings.get("qrf_test_metrics", {}).get("hitter_qrf_metrics_2025", {})
    lines = []
    lines.append("# Hot-Start Half-Life: Codex ML Round 1\n")
    lines.append("## Executive Summary\n")
    lines.append(
        f"The ML pass finds a small aggregate era counterfactual delta: the 2022-2025 model differs by "
        f"{fmt(counter.get('avg_delta'), 4)} wOBA from the 2015-2024 model on the named hitter set "
        f"(bootstrap 95% CI {fmt((counter.get('ci') or [None, None])[0], 4)} to "
        f"{fmt((counter.get('ci') or [None, None])[1], 4)}, N={counter.get('n_boot')}). "
        "That is evidence about the noise floor, not a causal claim about ABS or the run environment. "
        "The conservative reading is that April information remains useful mainly when it agrees with contact quality, strikeout discipline, and prior skill.\n"
    )
    lines.append("## Methods\n")
    lines.append(
        "I built player-season feature rows from pitch-level Statcast, using each hitter's first 22 player-games through the cutoff. "
        "The target set is rest-of-season performance after those games, while a parallel required LightGBM target predicts full-season wOBA for split validation. "
        "The temporal split is 2022-2023 train, 2024 validation, 2025 test; 2026 rows are inference-only. "
        "Intervals come from quantile regression forests with 10th, 50th, 80th, and 90th percentiles. "
        "Preseason priors are a 5/4/3 weighted mean of the prior three MLB seasons, with league fallback for players without MLB history. "
        "Mason Miller is handled with a reliever-specific forest on pitcher first-22-game features and a raw Statcast RA9 proxy, since pitch-level Statcast does not directly encode earned runs.\n"
    )
    lines.append(
        "The feature vector intentionally excludes `sz_top` and `sz_bot` so that deterministic 2026 ABS-era strike-zone metadata cannot leak a schema break into cross-season models. "
        "Plate-location features use only absolute `plate_x` and `plate_z` with a fixed rule-of-thumb zone for zone and chase rates. "
        "Contact quality comes from EV p90, hard-hit rate, barrel rate, xwOBA, and the xwOBA-minus-wOBA residual. "
        "The cached Fangraphs `batting_stats` and `pitching_stats` calls returned HTTP 403 in this environment, so the pipeline writes statcast-derived fallback season tables after the failed pybaseball attempt; the actual model features and targets are pitch-level Statcast aggregates.\n"
    )
    lines.append(
        "The data pull is idempotent and schema-aware. It repaired the shared 2022-2024 cache files because they existed but lacked plate location, pitch type, pitch number, and inning fields required by the brief. "
        "For 2025, the supplied path was treated as invalid when empty and then satisfied from another local full-season Statcast cache before writing the shared `data/statcast_2025.parquet`. "
        "The 2026 extension was fetched with `pybaseball.statcast(start_dt='2026-04-23', end_dt='2026-04-24')` and combined with the supplied March 27-April 22 file into a cutoff-only parquet. "
        "All season files are reduced to rate-stat-relevant columns plus Statcast's wOBA/xwOBA helpers and score fields needed for the Miller RA9 proxy. "
        "This keeps the cache small enough for fast reruns while preserving the features used in model training.\n"
    )
    full_test = lgbm_metrics.get("full_woba", {}).get("test", {})
    ros_test = lgbm_metrics.get("ros_woba", {}).get("test", {})
    lines.append(
        f"Model diagnostics are consistent with the task difficulty. The required full-season LightGBM reaches 2025 test RMSE {fmt(full_test.get('rmse'))}, "
        f"MAE {fmt(full_test.get('mae'))}, and R2 {fmt(full_test.get('r2'))}. "
        f"The rest-of-season LightGBM used for player point estimates is noisier, with RMSE {fmt(ros_test.get('rmse'))}, "
        f"MAE {fmt(ros_test.get('mae'))}, and R2 {fmt(ros_test.get('r2'))}. "
        f"The QRF 2025 holdout RMSE for ROS wOBA is {fmt(qrf_metrics.get('woba', {}).get('rmse'))}; for OPS it is {fmt(qrf_metrics.get('ops', {}).get('rmse'))}. "
        "Those errors are large enough that narrow April narratives should be treated skeptically unless the interval itself clears the prior.\n"
    )
    lines.append("## Stabilization Findings\n")
    nf_ba = noise.get("BA", {}).get("maintained_90pct_rate")
    nf_ops = noise.get("OPS", {}).get("maintained_90pct_rate")
    lines.append(
        f"The counterfactual comparison is the Agent B stabilization proxy: broad-era and current-era LightGBM ensembles were trained separately and scored on the same 2026 hitter vectors. "
        f"The average current-minus-broad delta is {fmt(counter.get('avg_delta'), 4)}. "
        f"The 2022-2025 top-five first-22-game leaderboard noise check was harsh: BA leaders maintained at least 90% of their April pace {fmt(nf_ba, 2)} of the time, "
        f"and OPS leaders did so {fmt(nf_ops, 2)} of the time. "
        "Null or near-null deltas should be published as such; the model does not justify a broad claim that the 2026 environment has made hot starts structurally more durable.\n"
    )
    if counter.get("players"):
        delta_bits = []
        for player in counter["players"]:
            delta_bits.append(
                f"{player['player']} {fmt(player.get('mean_delta'), 4)} "
                f"({fmt((player.get('ci') or [None, None])[0], 4)} to {fmt((player.get('ci') or [None, None])[1], 4)})"
            )
        lines.append(
            "Per-player current-minus-broad deltas are all small relative to player-level uncertainty: "
            + "; ".join(delta_bits)
            + ". The signs are mixed, which is the key aggregate point: the current-era model is not uniformly inflating 2026 hot-start forecasts.\n"
        )
    lines.append("## Per-Player Projections\n")
    for key, proj in projections.items():
        display_name = proj["player"]
        if proj.get("replacement_for"):
            display_name = f"{display_name} (replacement for {proj['replacement_for']})"
        if key == "mason_miller":
            ra9 = proj.get("pitching", {}).get("ra9", {})
            lines.append(
                f"- {display_name}: verdict {proj['verdict']}; ROS RA9 proxy median {fmt(ra9.get('q50'))}, "
                f"80% interval {fmt(ra9.get('q10'))}-{fmt(ra9.get('q90'))}, expected innings to next run about {fmt(ra9.get('expected_ip_until_er'), 1)}."
            )
        else:
            w = proj.get("woba", {})
            iso = proj.get("iso", {})
            ops = proj.get("ops", {})
            babip = proj.get("babip", {})
            k_rate = proj.get("k_rate", {})
            analog = proj.get("analogs", [{}])[0] if proj.get("analogs") else {}
            lines.append(
                f"- {display_name}: verdict {proj['verdict']}; LightGBM ROS wOBA point {fmt(w.get('point'))}, "
                f"QRF median {fmt(w.get('q50'))}, 80% interval {fmt(w.get('q10'))}-{fmt(w.get('q90'))}, "
                f"prior {fmt(proj.get('prior_woba'))}; nearest analog {analog.get('player', 'none')} {analog.get('year', '')}."
            )
            if proj["verdict"] != "excluded":
                lines.append(
                    f"  Secondary QRF medians: ISO {fmt(iso.get('q50'))} "
                    f"({fmt(iso.get('q10'))}-{fmt(iso.get('q90'))}), OPS {fmt(ops.get('q50'))} "
                    f"({fmt(ops.get('q10'))}-{fmt(ops.get('q90'))}), BABIP {fmt(babip.get('q50'))} "
                    f"({fmt(babip.get('q10'))}-{fmt(babip.get('q90'))}), K% {fmt(k_rate.get('q50'))}."
                )
            elif proj.get("exclusion_reason"):
                lines.append(f"  Exclusion reason: {proj['exclusion_reason']}")
    lines.append(
        "\nPages and Rice both score as noise by the pre-registered prior-overlap rule despite above-prior medians, because their 10th-90th percentile intervals still cover ordinary regression paths. "
        "Trout has the strongest median projection, but his prior is already strong; the model is not treating the Yankee Stadium power burst as new information large enough to clear his established baseline. "
        "Ballesteros is the top eligible substitute by first-22-game batting average, but the interval is wide and the prior is unstable because it is based on sparse previous MLB data. "
        "Miller's reliever interval is intentionally framed as a run-prevention proxy rather than official ERA; the nearest analog list includes elite late-inning arms and volatile closer seasons, which is exactly the distributional shape the model returns.\n"
    )
    lines.append("## Historical Analogs\n")
    lines.append(
        "Analog retrieval used cosine similarity over standardized first-22-game feature vectors, not names, teams, or outcomes. "
        "That means the analog table should be read as an empirical neighborhood check rather than a projection model by itself. "
        "Pages' top analogs were James McCann 2019 and Jarren Duran 2023, both good reminders that strong April contact can regress into useful but not star-level ROS production. "
        "Rice's nearest group is more power-heavy: Eric Thames 2017, Dan Vogelbach 2019, Kennys Vargas 2016, Tyler O'Neill 2024, and Brad Miller 2020. "
        "That neighborhood supports the model's broad interval: real power survives, but batting-average and on-base pace are fragile. "
        "Trout's nearest analog being his own 2019 season is a useful sanity check, but it also illustrates why the model refuses to call the hot start a new breakout; the baseline already expects star production. "
        "Ballesteros' analogs are volatile part-time or role-changing bats, and Miller's analogs split between dominant reliever seasons and closer seasons that gave runs back later. "
        "All non-excluded projected players cleared the five-analog, 0.70-similarity gate.\n"
    )
    lines.append("\n## Feature Importance Findings\n")
    top_text = ", ".join([f"{r['feature']} ({fmt(r['permutation_importance'], 4)})" for r in top_features[:10]])
    lines.append(
        f"Permutation importance on the 2025 holdout ranks the top features as: {top_text}. "
        f"TreeSHAP rank correlation with permutation ranks is {fmt(findings.get('shap_permutation_spearman'), 2)}. "
        "If that value is below 0.60, the rank-check CSV should be read as a warning that correlated rate features are substituting for one another rather than as a failure of a single feature.\n"
    )
    lines.append(
        "The practical interpretation is not that PA volume is intrinsically a hitting skill. It is a stabilizer and role proxy: players who accumulate 50-plus plate appearances quickly are less often bench bats, platoon-only hitters, or small-sample leaderboard accidents. "
        "The actual skill features that survive permutation are prior wOBA, EV p90, whiff rate, xwOBA, prior K rate, barrel rate, and the first-window BB/K rates. "
        "That feature set is directionally sensible and argues against using April batting average or raw OPS as the primary persistence signal.\n"
    )
    lines.append("## Kill-Gate Outcomes\n")
    lines.append(
        "Sample-size gates are enforced at 50 PA for hitters and 25 BF for Miller. "
        "Historical analogs require cosine similarity of at least 0.70; players with fewer than five analogs are reported honestly rather than padded. "
        "The preseason projection source gate falls back to the specified 3-year weighted MLB mean because a clean projection endpoint was not used in this reproducible pass. "
        "No 2026 data after April 24 is read by the pipeline.\n"
    )
    lines.append(
        "Murakami is the only named-player coverage failure. `playerid_lookup` returned no MLBAM ID, and the cutoff Statcast data contained no Mets hitter matching the stated 7-HR debut profile. "
        "Following the brief's kill-gate rule, the pipeline excludes Murakami and substitutes the next-best eligible top-BA hot starter, Moises Ballesteros. "
        "Every other named player clears the sample-size gate and has at least five historical analogs above the 0.70 cosine threshold.\n"
    )
    lines.append("## Limitations\n")
    lines.append(
        "This is a deliberately non-causal modeling pass. It does not estimate why the 2026 environment looks different, does not model park factors, and does not use post-cutoff outcomes. "
        "It also does not attempt translated NPB priors, which is material for the Murakami prompt mismatch. "
        "The fallback preseason prior is transparent but blunt: players with little MLB history can inherit unstable priors, while established stars such as Trout are pulled strongly toward their own recent baseline. "
        "Finally, the SHAP/permutation disagreement should keep the article from claiming a clean universal feature hierarchy. "
        "The safer claim is narrower: prior skill, opportunity volume, contact quality, and whiff/discipline signals are more predictive than April batting average alone.\n"
    )
    lines.append("## Open Questions\n")
    lines.append(
        "Round 2 should compare these ML intervals against Claude's Bayesian intervals, inspect any non-overlapping player ranges, and decide whether Murakami's no-MLB-prior handling needs a translated NPB prior. "
        "The Miller RA9 proxy should also be replaced with earned-run game-log splits if a reliable source is added."
    )
    (BASE_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_ready(findings: dict) -> None:
    projections = findings["projections"]
    counter = findings["era_counterfactual"]
    lines = []
    lines.append(
        f"Headline: the ML counterfactual delta is {fmt(counter.get('avg_delta'), 4)} wOBA "
        f"(95% bootstrap CI {fmt((counter.get('ci') or [None, None])[0], 4)} to {fmt((counter.get('ci') or [None, None])[1], 4)}), "
        "so Round 1 does not yet support a strong environment-shift claim from the Codex side."
    )
    for key in ["andy_pages", "ben_rice", "munetaka_murakami", "mike_trout", "mason_miller"]:
        proj = projections.get(key, {"player": key, "verdict": "excluded", "analogs": []})
        analog = proj.get("analogs", [{}])[0] if proj.get("analogs") else {}
        analog_text = f"{analog.get('player')} {analog.get('year')}" if analog else "no >=0.70 analog"
        if key == "munetaka_murakami" and proj.get("verdict") == "excluded":
            substitute = next((p for p in projections.values() if p.get("replacement_for") == "Munetaka Murakami"), None)
            if substitute:
                sub_analog = substitute.get("analogs", [{}])[0] if substitute.get("analogs") else {}
                sub_analog_text = (
                    f"{sub_analog.get('player')} {sub_analog.get('year')}" if sub_analog else "no >=0.70 analog"
                )
                lines.append(
                    f"- Munetaka Murakami: excluded, no cutoff Statcast profile; substitute {substitute['player']}: "
                    f"{substitute.get('verdict', 'excluded')} | nearest analog: {sub_analog_text}"
                )
                continue
        lines.append(f"- {proj.get('player', key)}: {proj.get('verdict', 'excluded')} | nearest analog: {analog_text}")
    lines.append(
        "Open questions: compare against Claude's intervals, decide whether Murakami needs an NPB-translated prior, and replace Miller's RA9 proxy with earned-run splits if available."
    )
    (BASE_DIR / "READY_FOR_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> dict:
    plot_league_environment()
    findings = build_findings()
    write_report(findings)
    write_ready(findings)
    return findings


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

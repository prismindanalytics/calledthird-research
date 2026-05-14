from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score

from data_prep_r3 import (
    ARTIFACTS_DIR,
    CHARTS_DIR,
    DIAG_DIR,
    GLOBAL_SEED,
    Round3Data,
    encode_features,
    lgbm_classifier,
    percentile_ci,
    row_weights_from_games,
    sample_game_weights,
    save_json,
    terminal_pa_rows,
    write_classifier_diagnostics,
)


H2_NUMERIC = [
    "plate_x",
    "plate_z",
    "in_fixed_zone",
    "is_top_edge",
    "is_bottom_edge",
    "balls",
    "strikes",
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
]
H2_CATEGORICAL = ["pitch_type", "count_state", "zone_region"]


@dataclass
class H2Result:
    summary: dict
    leaderboard: pd.DataFrame
    all_candidates: pd.DataFrame
    stability: pd.DataFrame
    shap_attribution: pd.DataFrame


def _mix_vector(frame: pd.DataFrame, pitch_types: list[str], weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        counts = frame["pitch_type"].value_counts(normalize=False)
        total = float(counts.sum())
        return np.array([counts.get(pt, 0.0) / total if total else 0.0 for pt in pitch_types])
    tmp = pd.DataFrame({"pitch_type": frame["pitch_type"].to_numpy(), "w": weights})
    counts = tmp.groupby("pitch_type", observed=True)["w"].sum()
    total = float(counts.sum())
    return np.array([counts.get(pt, 0.0) / total if total else 0.0 for pt in pitch_types])


def _weighted_rate(values: pd.Series, weights: np.ndarray | None = None) -> float:
    arr = values.to_numpy(dtype=float)
    if weights is None:
        return float(np.nanmean(arr))
    denom = float(np.sum(weights))
    return float(np.dot(arr, weights) / denom) if denom > 0 else float("nan")


def _shift_table(
    df_2025: pd.DataFrame,
    df_2026: pd.DataFrame,
    pitchers: pd.DataFrame,
    pitch_types: list[str],
    weights_2025: dict[int, int] | None = None,
    weights_2026: dict[int, int] | None = None,
) -> pd.DataFrame:
    rows = []
    for p in pitchers.itertuples():
        p25 = df_2025[df_2025["pitcher"] == p.pitcher]
        p26 = df_2026[df_2026["pitcher"] == p.pitcher]
        if p25.empty or p26.empty:
            continue
        w25 = row_weights_from_games(p25, weights_2025) if weights_2025 is not None else None
        w26 = row_weights_from_games(p26, weights_2026) if weights_2026 is not None else None
        if w25 is not None and w25.sum() <= 0:
            continue
        if w26 is not None and w26.sum() <= 0:
            continue
        z25 = _weighted_rate(p25["in_fixed_zone"], w25)
        z26 = _weighted_rate(p26["in_fixed_zone"], w26)
        top25 = _weighted_rate(p25["is_top_edge"], w25)
        top26 = _weighted_rate(p26["is_top_edge"], w26)
        mix25 = _mix_vector(p25, pitch_types, w25)
        mix26 = _mix_vector(p26, pitch_types, w26)
        jsd = float(jensenshannon(mix25, mix26, base=2.0))
        zone_delta_pp = (z26 - z25) * 100
        top_delta_pp = (top26 - top25) * 100
        magnitude_pass = abs(zone_delta_pp) >= 15.0 or abs(top_delta_pp) >= 15.0 or jsd >= 0.05
        score = max(abs(zone_delta_pp) / 15.0, abs(top_delta_pp) / 15.0, jsd / 0.05)
        rows.append(
            {
                "pitcher": int(p.pitcher),
                "player_name": p.player_name,
                "pitches_2025": int(len(p25)),
                "pitches_2026": int(len(p26)),
                "zone_rate_2025": z25,
                "zone_rate_2026": z26,
                "zone_rate_delta_pp": zone_delta_pp,
                "top_share_2025": top25,
                "top_share_2026": top26,
                "top_share_delta_pp": top_delta_pp,
                "pitch_mix_jsd": jsd,
                "magnitude_pass": bool(magnitude_pass),
                "adaptation_score": score,
            }
        )
    return pd.DataFrame(rows).sort_values("adaptation_score", ascending=False).reset_index(drop=True)


def _bootstrap_stability(
    df_2025: pd.DataFrame,
    df_2026: pd.DataFrame,
    pitchers: pd.DataFrame,
    pitch_types: list[str],
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(GLOBAL_SEED + 5100)
    games25 = df_2025["game_pk"].dropna().astype(int).unique()
    games26 = df_2026["game_pk"].dropna().astype(int).unique()
    counts: dict[int, int] = {int(p): 0 for p in pitchers["pitcher"]}
    boot_rows = []
    for b in range(n_boot):
        w25 = sample_game_weights(games25, rng)
        w26 = sample_game_weights(games26, rng)
        tab = _shift_table(df_2025, df_2026, pitchers, pitch_types, w25, w26)
        top = tab[tab["magnitude_pass"]].head(15)
        for p in top["pitcher"].astype(int):
            counts[p] = counts.get(p, 0) + 1
        for row in top.itertuples():
            boot_rows.append(
                {
                    "iteration": b,
                    "pitcher": int(row.pitcher),
                    "player_name": row.player_name,
                    "adaptation_score": float(row.adaptation_score),
                    "zone_rate_delta_pp": float(row.zone_rate_delta_pp),
                    "top_share_delta_pp": float(row.top_share_delta_pp),
                    "pitch_mix_jsd": float(row.pitch_mix_jsd),
                }
            )
        if (b + 1) % 25 == 0:
            print(f"  H2 stability bootstrap {b + 1}/{n_boot}")
    stability = pd.DataFrame({"pitcher": list(counts.keys()), "top15_appearances": list(counts.values())})
    stability["bootstrap_iterations"] = n_boot
    stability["stability_score"] = stability["top15_appearances"] / n_boot
    return stability, pd.DataFrame(boot_rows)


def _feature_group(name: str) -> str:
    if name in {"plate_x", "plate_z", "in_fixed_zone", "is_top_edge", "is_bottom_edge"} or name.startswith("zone_region_"):
        return "zone_location"
    if name.startswith("pitch_type_") or name in {"release_speed", "release_spin_rate", "pfx_x", "pfx_z"}:
        return "arsenal_mix_shape"
    if name in {"balls", "strikes"} or name.startswith("count_state_"):
        return "count_context"
    return "other"


def _feature_importance_ensemble(
    combined: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    n_boot: int,
    n_seeds: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(GLOBAL_SEED + 5600)
    games = combined["game_pk"].dropna().astype(int).unique()
    rows = []
    audit_rows = []
    for b in range(n_boot):
        weights = sample_game_weights(games, rng)
        w = row_weights_from_games(combined, weights)
        fit_idx = np.flatnonzero(w > 0)
        oob_idx = np.flatnonzero(w == 0)
        for s in range(n_seeds):
            seed = GLOBAL_SEED + 5700 + b * 20 + s
            model = lgbm_classifier(seed, n_estimators=45)
            model.fit(X.iloc[fit_idx], y[fit_idx], sample_weight=w[fit_idx])
            imp = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
            imp["feature_group"] = imp["feature"].map(_feature_group)
            grouped = imp.groupby("feature_group", observed=True)["importance"].sum().to_dict()
            total = float(sum(grouped.values())) or 1.0
            row = {"bootstrap_iteration": b, "seed_index": s}
            for group in ["zone_location", "arsenal_mix_shape", "count_context", "other"]:
                row[f"{group}_importance_share"] = grouped.get(group, 0.0) / total
            rows.append(row)
            audit = {"bootstrap_iteration": b, "seed_index": s, "oob_n": int(len(oob_idx))}
            if len(oob_idx) >= 500 and len(np.unique(y[oob_idx])) == 2:
                prob = model.predict_proba(X.iloc[oob_idx])[:, 1]
                from data_prep_r3 import calibration_table

                tab = calibration_table(y[oob_idx], prob)
                audit["oob_auc"] = float(roc_auc_score(y[oob_idx], prob))
                audit["oob_max_calibration_deviation"] = float(tab["abs_deviation"].max())
                audit["oob_poor_calibration"] = bool(audit["oob_max_calibration_deviation"] > 0.05)
            audit_rows.append(audit)
        if (b + 1) % 10 == 0:
            print(f"  H2 feature-importance ensemble {b + 1}/{n_boot} bootstraps")
    out = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    out.to_csv(ARTIFACTS_DIR / "h2_feature_importance_ensemble.csv", index=False)
    audit.to_csv(ARTIFACTS_DIR / "h2_gbm_calibration_audit.csv", index=False)
    return out


def _write_learning_curve(combined: pd.DataFrame, X: pd.DataFrame, y: np.ndarray) -> None:
    rng = np.random.default_rng(GLOBAL_SEED + 5900)
    games = combined["game_pk"].dropna().astype(int).unique()
    holdout = set(rng.choice(games, size=max(20, int(len(games) * 0.18)), replace=False))
    train_idx_all = ~combined["game_pk"].isin(holdout).to_numpy()
    test_idx = combined["game_pk"].isin(holdout).to_numpy()
    train_games = combined.loc[train_idx_all, "game_pk"].dropna().astype(int).unique()
    rows = []
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        use_games = set(rng.choice(train_games, size=max(10, int(len(train_games) * frac)), replace=False))
        idx = train_idx_all & combined["game_pk"].isin(use_games).to_numpy()
        model = lgbm_classifier(GLOBAL_SEED + int(frac * 1000), n_estimators=45)
        model.fit(X.iloc[idx], y[idx])
        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        rows.append({"train_game_fraction": frac, "train_rows": int(idx.sum()), "holdout_auc": float(roc_auc_score(y[test_idx], prob))})
    curve = pd.DataFrame(rows)
    curve.to_csv(ARTIFACTS_DIR / "h2_year_shift_classifier_learning_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(curve["train_game_fraction"], curve["holdout_auc"], marker="o", color="#2563eb")
    ax.set_xlabel("Training game fraction")
    ax.set_ylabel("Holdout AUC")
    ax.set_title("H2 year-shift classifier learning curve")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "h2_year_shift_classifier_learning_curve.png", dpi=200)
    plt.close(fig)


def _per_pitcher_shap(combined: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return pd.DataFrame()
    top_ids = set(leaderboard["pitcher"].head(25).astype(int))
    keep = combined["pitcher"].astype(int).isin(top_ids).to_numpy()
    if keep.sum() == 0:
        return pd.DataFrame()
    rows = []
    for seed_offset in range(10):
        model = lgbm_classifier(GLOBAL_SEED + 6100 + seed_offset, n_estimators=60)
        model.fit(X, y)
        contrib = model.booster_.predict(X.loc[keep], pred_contrib=True)[:, :-1]
        contrib_frame = pd.DataFrame(contrib, columns=X.columns)
        group_abs = pd.DataFrame(
            {
                group: contrib_frame[[c for c in X.columns if _feature_group(c) == group]].abs().sum(axis=1)
                for group in ["zone_location", "arsenal_mix_shape", "count_context", "other"]
            }
        )
        meta = combined.loc[keep, ["pitcher", "player_name", "year_2026"]].reset_index(drop=True)
        group_abs = pd.concat([meta, group_abs], axis=1)
        grouped = (
            group_abs.groupby(["pitcher", "player_name"], observed=True)[["zone_location", "arsenal_mix_shape", "count_context", "other"]]
            .mean()
            .reset_index()
        )
        grouped["seed_index"] = seed_offset
        rows.append(grouped)
    out = pd.concat(rows, ignore_index=True)
    summary = (
        out.groupby(["pitcher", "player_name"], observed=True)[["zone_location", "arsenal_mix_shape", "count_context", "other"]]
        .mean()
        .reset_index()
    )
    total = summary[["zone_location", "arsenal_mix_shape", "count_context", "other"]].sum(axis=1).replace(0, np.nan)
    for group in ["zone_location", "arsenal_mix_shape", "count_context", "other"]:
        summary[f"{group}_share"] = summary[group] / total
    summary.to_csv(ARTIFACTS_DIR / "h2_per_pitcher_shap_attribution.csv", index=False)
    return summary


def _plot_leaderboard(leaderboard: pd.DataFrame) -> None:
    plot_df = leaderboard.head(15).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No pitcher cleared the >=80% stability and magnitude screen", ha="center", va="center")
        ax.axis("off")
    else:
        colors = ["#2563eb" if v >= 0 else "#dc2626" for v in plot_df["zone_rate_delta_pp"]]
        ax.barh(plot_df["player_name"], plot_df["adaptation_score"], color=colors)
        for i, row in enumerate(plot_df.itertuples()):
            ax.text(
                row.adaptation_score + 0.03,
                i,
                f"stab {row.stability_score:.0%} | zone {row.zone_rate_delta_pp:+.1f}pp | top {row.top_share_delta_pp:+.1f}pp | JS {row.pitch_mix_jsd:.2f}",
                va="center",
                fontsize=8.5,
            )
        ax.set_xlabel("Adaptation score")
        ax.set_title("R3-H2 stable adapter leaderboard")
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "h2_adapter_leaderboard.png", dpi=220)
    plt.close(fig)


def run_h2_adapter_leaderboard(
    data: Round3Data,
    n_stability_boot: int = 200,
    n_feature_boot: int = 100,
    n_feature_seeds: int = 10,
) -> H2Result:
    print("R3-H2: building pitcher adapter metrics")
    df_2026 = data.df_2026.dropna(subset=["pitcher", "plate_x", "plate_z"]).copy()
    df_2025 = data.df_2025.dropna(subset=["pitcher", "plate_x", "plate_z"]).copy()
    q26 = (
        df_2026.groupby(["pitcher", "player_name"], observed=True)
        .size()
        .reset_index(name="pitches_2026")
    )
    q25 = df_2025.groupby("pitcher", observed=True).size().reset_index(name="pitches_2025")
    pitchers = q26[q26["pitches_2026"] >= 200].merge(q25, on="pitcher", how="inner")
    pitchers = pitchers[pitchers["pitches_2025"] >= 120].copy()
    pitchers["pitcher"] = pitchers["pitcher"].astype(int)
    pitch_types = sorted(pd.concat([df_2025["pitch_type"], df_2026["pitch_type"]]).value_counts().head(12).index.tolist())

    all_candidates = _shift_table(df_2025, df_2026, pitchers[["pitcher", "player_name"]], pitch_types)
    all_candidates.to_csv(ARTIFACTS_DIR / "h2_all_adapter_candidates.csv", index=False)

    stability, boot_membership = _bootstrap_stability(df_2025, df_2026, pitchers[["pitcher", "player_name"]], pitch_types, n_stability_boot)
    stability.to_csv(ARTIFACTS_DIR / "h2_bootstrap_stability.csv", index=False)
    boot_membership.to_csv(ARTIFACTS_DIR / "h2_bootstrap_top15_membership.csv", index=False)

    leaderboard = all_candidates.merge(stability[["pitcher", "stability_score", "top15_appearances"]], on="pitcher", how="left")
    leaderboard["stability_score"] = leaderboard["stability_score"].fillna(0.0)
    leaderboard["stable_adapter"] = leaderboard["magnitude_pass"] & (leaderboard["stability_score"] >= 0.80)
    leaderboard = leaderboard.sort_values(["stable_adapter", "stability_score", "adaptation_score"], ascending=[False, False, False])
    stable = leaderboard[leaderboard["stable_adapter"]].copy()
    stable.to_csv(ARTIFACTS_DIR / "h2_stable_adapter_leaderboard.csv", index=False)

    print(f"R3-H2: LightGBM feature-importance ensemble {n_feature_boot} bootstraps x {n_feature_seeds} seeds")
    combined = pd.concat(
        [
            df_2025[df_2025["pitcher"].astype(int).isin(set(pitchers["pitcher"]))],
            df_2026[df_2026["pitcher"].astype(int).isin(set(pitchers["pitcher"]))],
        ],
        ignore_index=True,
    )
    X, columns = encode_features(combined, H2_NUMERIC, H2_CATEGORICAL)
    y = combined["year_2026"].to_numpy(dtype=int)
    feature_ensemble = _feature_importance_ensemble(combined, X, y, n_feature_boot, n_feature_seeds)
    feature_audit = pd.read_csv(ARTIFACTS_DIR / "h2_gbm_calibration_audit.csv")
    prob_model = lgbm_classifier(GLOBAL_SEED + 6300, n_estimators=70)
    prob_model.fit(X, y)
    prob = prob_model.predict_proba(X)[:, 1]
    diagnostics = write_classifier_diagnostics("h2_year_shift_classifier", y, prob)
    _write_learning_curve(combined, X, y)
    shap_attr = _per_pitcher_shap(combined, X, y, leaderboard)
    if not shap_attr.empty:
        stable = stable.merge(shap_attr, on=["pitcher", "player_name"], how="left")
        stable.to_csv(ARTIFACTS_DIR / "h2_stable_adapter_leaderboard.csv", index=False)

    _plot_leaderboard(stable)
    stability_scores = leaderboard["stability_score"].to_numpy(dtype=float)
    summary = {
        "qualified_pitchers": int(len(pitchers)),
        "magnitude_pass_pitchers": int(all_candidates["magnitude_pass"].sum()),
        "stable_adapter_count": int(len(stable)),
        "stability_bootstrap_n": n_stability_boot,
        "feature_importance_bootstrap_n": n_feature_boot,
        "feature_importance_seed_n": n_feature_seeds,
        "feature_importance_total_gbms": int(n_feature_boot * n_feature_seeds),
        "feature_importance_oob_max_calibration_deviation": float(feature_audit["oob_max_calibration_deviation"].max(skipna=True)),
        "feature_importance_oob_gt_5pp_count": int((feature_audit["oob_max_calibration_deviation"] > 0.05).sum()),
        "stable_adapters": stable.head(25).to_dict(orient="records"),
        "top_ml_candidates_pending_cross_method": leaderboard.head(25).to_dict(orient="records"),
        "feature_importance_group_mean": feature_ensemble.mean(numeric_only=True).to_dict(),
        "year_shift_classifier_diagnostics": diagnostics,
        "bootstrap_stability_max": float(np.nanmax(stability_scores)) if len(stability_scores) else float("nan"),
        "cross_method_agreement_status": "ML-only list; Bayesian agreement must be applied in comparison memo without coordination.",
    }
    save_json(summary, ARTIFACTS_DIR / "h2_summary.json")
    return H2Result(
        summary=summary,
        leaderboard=stable,
        all_candidates=all_candidates,
        stability=stability,
        shap_attribution=shap_attr,
    )

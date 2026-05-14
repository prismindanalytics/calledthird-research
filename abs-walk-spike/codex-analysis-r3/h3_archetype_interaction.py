from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from data_prep_r3 import (
    ARTIFACTS_DIR,
    CHARTS_DIR,
    DIAG_DIR,
    GLOBAL_SEED,
    Round3Data,
    encode_features,
    lgbm_regressor,
    percentile_ci,
    row_weights_from_games,
    sample_game_weights,
    save_json,
    terminal_pa_rows,
    weighted_mean,
)


H3_NUMERIC = [
    "stuff_pct",
    "command_pct",
    "stuff_minus_command",
    "stuff_command_interaction",
    "walk_rate_2025_full",
    "zone_rate_2025_full",
    "top_share_2025_full",
    "arsenal_whiff_rate",
    "mean_release_speed_2025",
    "mean_plate_z_2025",
    "pitches_2026_window",
]
H3_CATEGORICAL: list[str] = []


@dataclass
class H3Result:
    summary: dict
    model_table: pd.DataFrame
    hurt_command: pd.DataFrame
    helped_stuff: pd.DataFrame
    bootstrap_stability: pd.DataFrame


def _pitcher_window_metrics(df: pd.DataFrame, year_label: str, weights: dict[int, int] | None = None) -> pd.DataFrame:
    terminal = terminal_pa_rows(df)
    rows = []
    for pitcher, group in terminal.groupby("pitcher", observed=True):
        if pd.isna(pitcher):
            continue
        w = row_weights_from_games(group, weights) if weights is not None else np.ones(len(group), dtype=float)
        if w.sum() <= 0:
            continue
        rows.append(
            {
                "pitcher_id": int(pitcher),
                f"pa_{year_label}": float(w.sum()),
                f"walk_rate_{year_label}": weighted_mean(group["walk_event"].to_numpy(dtype=float), w),
            }
        )
    pa = pd.DataFrame(rows)
    pitch_rows = []
    for pitcher, group in df.groupby("pitcher", observed=True):
        if pd.isna(pitcher):
            continue
        w = row_weights_from_games(group, weights) if weights is not None else np.ones(len(group), dtype=float)
        if w.sum() <= 0:
            continue
        pitch_rows.append(
            {
                "pitcher_id": int(pitcher),
                f"pitches_{year_label}": float(w.sum()),
                f"zone_rate_{year_label}": weighted_mean(group["in_fixed_zone"].to_numpy(dtype=float), w),
                f"top_share_{year_label}": weighted_mean(group["is_top_edge"].to_numpy(dtype=float), w),
            }
        )
    pitch = pd.DataFrame(pitch_rows)
    return pa.merge(pitch, on="pitcher_id", how="outer")


def _model_input(data: Round3Data, archetypes: pd.DataFrame) -> pd.DataFrame:
    m26 = _pitcher_window_metrics(data.df_2026, "2026_window")
    model = archetypes.merge(m26, on="pitcher_id", how="inner")
    model = model[model["pitches_2026_window"] >= 200].copy()
    model["walk_rate_change"] = model["walk_rate_2026_window"] - model["walk_rate_2025_full"]
    model["walk_rate_change_pp"] = model["walk_rate_change"] * 100
    model["command_minus_stuff"] = model["command_pct"] - model["stuff_pct"]
    model["stuff_command_interaction"] = ((model["stuff_pct"] - 50.0) / 50.0) * ((model["command_pct"] - 50.0) / 50.0)
    return model.replace([np.inf, -np.inf], np.nan).dropna(subset=H3_NUMERIC + ["walk_rate_change"])


def _cross_validated_predictions(model_table: pd.DataFrame) -> tuple[np.ndarray, dict]:
    X, columns = encode_features(model_table, H3_NUMERIC, H3_CATEGORICAL)
    y = model_table["walk_rate_change"].to_numpy(dtype=float)
    preds = np.zeros(len(model_table), dtype=float)
    k = min(5, max(2, len(model_table) // 12))
    cv = KFold(n_splits=k, shuffle=True, random_state=GLOBAL_SEED + 7100)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        reg = lgbm_regressor(GLOBAL_SEED + 7200 + fold, n_estimators=80)
        reg.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = reg.predict(X.iloc[test_idx])
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    diagnostics = _write_regression_diagnostics("h3_archetype_gbm", y, preds)
    diagnostics["cv_rmse"] = rmse
    diagnostics["cv_folds"] = k
    return preds, diagnostics


def _write_regression_diagnostics(name: str, y: np.ndarray, pred: np.ndarray) -> dict:
    frame = pd.DataFrame({"y": y, "pred": pred})
    frame["bin"] = pd.qcut(frame["pred"], q=min(10, max(3, len(frame) // 8)), duplicates="drop")
    cal = (
        frame.groupby("bin", observed=True)
        .agg(n=("y", "size"), predicted=("pred", "mean"), empirical=("y", "mean"))
        .reset_index(drop=True)
    )
    cal["abs_deviation"] = (cal["empirical"] - cal["predicted"]).abs()
    cal.to_csv(ARTIFACTS_DIR / f"{name}_calibration.csv", index=False)
    max_dev = float(cal["abs_deviation"].max()) if len(cal) else float("nan")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lo = float(min(frame["y"].min(), frame["pred"].min()))
    hi = float(max(frame["y"].max(), frame["pred"].max()))
    ax.plot([lo, hi], [lo, hi], color="#6b7280", linestyle="--", linewidth=1)
    ax.plot(cal["predicted"], cal["empirical"], marker="o", color="#7c3aed")
    ax.set_xlabel("Mean predicted walk-rate change")
    ax.set_ylabel("Empirical walk-rate change")
    ax.set_title(f"{name} calibration")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / f"{name}_calibration.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(pred, y - pred, s=28, color="#2563eb", alpha=0.75)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xlabel("Predicted walk-rate change")
    ax.set_ylabel("Residual")
    ax.set_title(f"{name} residuals")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / f"{name}_residuals.png", dpi=200)
    plt.close(fig)
    return {"max_calibration_deviation": max_dev, "poor_calibration": bool(max_dev > 0.05)}


def _permutation_tables(model_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    X, columns = encode_features(model_table, H3_NUMERIC, H3_CATEGORICAL)
    y = model_table["walk_rate_change"].to_numpy(dtype=float)
    model = lgbm_regressor(GLOBAL_SEED + 7300, n_estimators=100)
    model.fit(X, y)
    actual = permutation_importance(model, X, y, n_repeats=40, random_state=GLOBAL_SEED + 7310, scoring="neg_root_mean_squared_error")
    rng = np.random.default_rng(GLOBAL_SEED + 7320)
    y_perm = rng.permutation(y)
    null_model = lgbm_regressor(GLOBAL_SEED + 7330, n_estimators=100)
    null_model.fit(X, y_perm)
    null = permutation_importance(null_model, X, y_perm, n_repeats=40, random_state=GLOBAL_SEED + 7340, scoring="neg_root_mean_squared_error")
    actual_tab = pd.DataFrame({"feature": columns, "importance_mean": actual.importances_mean, "importance_std": actual.importances_std})
    null_tab = pd.DataFrame({"feature": columns, "null_importance_mean": null.importances_mean, "null_importance_std": null.importances_std})
    actual_tab.to_csv(ARTIFACTS_DIR / "h3_permutation_importance.csv", index=False)
    null_tab.to_csv(ARTIFACTS_DIR / "h3_permutation_importance_permuted_label_baseline.csv", index=False)
    comparison = actual_tab.merge(null_tab, on="feature", how="left")
    comparison["actual_minus_null"] = comparison["importance_mean"] - comparison["null_importance_mean"]
    comparison.to_csv(ARTIFACTS_DIR / "h3_permutation_vs_null.csv", index=False)

    contrib = model.booster_.predict(X, pred_contrib=True)[:, :-1]
    shap = pd.DataFrame(contrib, columns=columns)
    shap_summary = pd.DataFrame(
        {
            "feature": columns,
            "mean_abs_shap": np.abs(contrib).mean(axis=0),
            "mean_shap": contrib.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_summary.to_csv(ARTIFACTS_DIR / "h3_shap_summary.csv", index=False)
    interaction_shap = shap["stuff_command_interaction"] if "stuff_command_interaction" in shap.columns else pd.Series(dtype=float)
    meta = model_table[["pitcher_id", "name", "stuff_pct", "command_pct", "stuff_minus_command", "walk_rate_change_pp"]].copy()
    meta["interaction_shap"] = interaction_shap.to_numpy() if len(interaction_shap) else np.nan
    meta.to_csv(ARTIFACTS_DIR / "h3_interaction_shap_by_pitcher.csv", index=False)
    info = {
        "interaction_permutation_importance": float(comparison.loc[comparison["feature"] == "stuff_command_interaction", "importance_mean"].iloc[0]),
        "interaction_null_importance": float(comparison.loc[comparison["feature"] == "stuff_command_interaction", "null_importance_mean"].iloc[0]),
        "interaction_mean_abs_shap": float(shap_summary.loc[shap_summary["feature"] == "stuff_command_interaction", "mean_abs_shap"].iloc[0]),
    }
    return comparison, shap_summary, info


def _bootstrap_leaderboards(data: Round3Data, model_table: pd.DataFrame, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(GLOBAL_SEED + 7600)
    games26 = data.df_2026["game_pk"].dropna().astype(int).unique()
    counts_hurt = {int(p): 0 for p in model_table["pitcher_id"]}
    counts_help = {int(p): 0 for p in model_table["pitcher_id"]}
    rows = []
    base = model_table.set_index("pitcher_id")
    for b in range(n_boot):
        w26 = sample_game_weights(games26, rng)
        m26 = _pitcher_window_metrics(data.df_2026, "2026_window", w26)
        tab = base.drop(columns=[c for c in base.columns if c.endswith("_2026_window") or c in {"walk_rate_change", "walk_rate_change_pp"}], errors="ignore")
        tab = tab.reset_index().merge(m26, on="pitcher_id", how="inner").set_index("pitcher_id")
        tab["walk_rate_change"] = tab["walk_rate_2026_window"] - tab["walk_rate_2025_full"]
        tab["walk_rate_change_pp"] = tab["walk_rate_change"] * 100
        tab["command_minus_stuff"] = tab["command_pct"] - tab["stuff_pct"]
        tab["hurt_command_score"] = np.maximum(tab["command_minus_stuff"], 0) / 100.0 * np.maximum(tab["walk_rate_change_pp"], 0)
        tab["helped_stuff_score"] = np.maximum(tab["stuff_minus_command"], 0) / 100.0 * np.maximum(-tab["walk_rate_change_pp"], 0)
        hurt = tab.sort_values("hurt_command_score", ascending=False).head(15)
        helped = tab.sort_values("helped_stuff_score", ascending=False).head(15)
        for p in hurt.index.astype(int):
            counts_hurt[p] = counts_hurt.get(p, 0) + 1
        for p in helped.index.astype(int):
            counts_help[p] = counts_help.get(p, 0) + 1
        for p, row in hurt.iterrows():
            rows.append({"iteration": b, "leaderboard": "hurt_command", "pitcher_id": int(p), "score": float(row.hurt_command_score)})
        for p, row in helped.iterrows():
            rows.append({"iteration": b, "leaderboard": "helped_stuff", "pitcher_id": int(p), "score": float(row.helped_stuff_score)})
        if (b + 1) % 25 == 0:
            print(f"  H3 leaderboard bootstrap {b + 1}/{n_boot}")
    stability = pd.DataFrame(
        [
            {"pitcher_id": p, "leaderboard": "hurt_command", "top15_appearances": c, "stability_score": c / n_boot}
            for p, c in counts_hurt.items()
        ]
        + [
            {"pitcher_id": p, "leaderboard": "helped_stuff", "top15_appearances": c, "stability_score": c / n_boot}
            for p, c in counts_help.items()
        ]
    )
    stability.to_csv(ARTIFACTS_DIR / "h3_bootstrap_leaderboard_stability.csv", index=False)
    pd.DataFrame(rows).to_csv(ARTIFACTS_DIR / "h3_bootstrap_leaderboard_membership.csv", index=False)

    tab = model_table.copy()
    tab["command_minus_stuff"] = tab["command_pct"] - tab["stuff_pct"]
    tab["hurt_command_score"] = np.maximum(tab["command_minus_stuff"], 0) / 100.0 * np.maximum(tab["walk_rate_change_pp"], 0)
    tab["helped_stuff_score"] = np.maximum(tab["stuff_minus_command"], 0) / 100.0 * np.maximum(-tab["walk_rate_change_pp"], 0)
    hurt = tab.merge(stability[stability["leaderboard"] == "hurt_command"][["pitcher_id", "stability_score"]], on="pitcher_id", how="left")
    helped = tab.merge(stability[stability["leaderboard"] == "helped_stuff"][["pitcher_id", "stability_score"]], on="pitcher_id", how="left")
    hurt["stable"] = hurt["stability_score"].fillna(0) >= 0.80
    helped["stable"] = helped["stability_score"].fillna(0) >= 0.80
    hurt = hurt.sort_values(["stable", "hurt_command_score"], ascending=[False, False])
    helped = helped.sort_values(["stable", "helped_stuff_score"], ascending=[False, False])
    hurt.to_csv(ARTIFACTS_DIR / "h3_hurt_command_leaderboard.csv", index=False)
    helped.to_csv(ARTIFACTS_DIR / "h3_helped_stuff_leaderboard.csv", index=False)
    return stability, hurt, helped


def _plot_scatter(model_table: pd.DataFrame, rho: float, pvalue: float) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    colors = np.where(model_table["stuff_minus_command"] >= 0, "#2563eb", "#dc2626")
    ax.scatter(model_table["stuff_minus_command"], model_table["walk_rate_change_pp"], s=42, c=colors, alpha=0.78)
    if len(model_table) > 2:
        m, b = np.polyfit(model_table["stuff_minus_command"], model_table["walk_rate_change_pp"], 1)
        xs = np.linspace(model_table["stuff_minus_command"].min(), model_table["stuff_minus_command"].max(), 100)
        ax.plot(xs, m * xs + b, color="#111827", linewidth=1.2)
    ax.axhline(0, color="#6b7280", linestyle="--", linewidth=1)
    ax.axvline(0, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_xlabel("Stuff percentile minus command percentile")
    ax.set_ylabel("2026 walk-rate change vs 2025 baseline (pp)")
    ax.set_title(f"R3-H3 archetype interaction (Spearman rho={rho:.2f}, p={pvalue:.3f})")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "h3_archetype_scatter.png", dpi=220)
    plt.close(fig)


def _plot_leaderboards(hurt: pd.DataFrame, helped: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
    panels = [
        (axes[0], hurt.head(10).iloc[::-1], "Command pitchers most hurt", "hurt_command_score", "#dc2626"),
        (axes[1], helped.head(10).iloc[::-1], "Stuff pitchers most helped", "helped_stuff_score", "#2563eb"),
    ]
    for ax, frame, title, score_col, color in panels:
        if frame.empty:
            ax.text(0.5, 0.5, "No qualifiers", ha="center", va="center")
            ax.axis("off")
            continue
        ax.barh(frame["name"], frame[score_col], color=color)
        for i, row in enumerate(frame.itertuples()):
            ax.text(
                getattr(row, score_col) + 0.02,
                i,
                f"{row.walk_rate_change_pp:+.1f}pp | S-C {row.stuff_minus_command:+.0f} | stab {getattr(row, 'stability_score', 0):.0%}",
                va="center",
                fontsize=8.2,
            )
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "h3_archetype_leaderboards.png", dpi=220)
    plt.close(fig)


def run_h3_archetype_interaction(data: Round3Data, archetypes: pd.DataFrame, n_boot: int = 200) -> H3Result:
    print("R3-H3: modeling stuff/command archetype interaction")
    model_table = _model_input(data, archetypes)
    model_table.to_csv(ARTIFACTS_DIR / "h3_model_table.csv", index=False)
    rho, pvalue = spearmanr(model_table["stuff_minus_command"], model_table["walk_rate_change_pp"])
    preds, diagnostics = _cross_validated_predictions(model_table)
    model_table["cv_pred_walk_rate_change"] = preds
    perm, shap_summary, perm_info = _permutation_tables(model_table)
    stability, hurt, helped = _bootstrap_leaderboards(data, model_table, n_boot)
    _plot_scatter(model_table, float(rho), float(pvalue))
    _plot_leaderboards(hurt, helped)

    stable_hurt = hurt[hurt["stable"]].copy()
    stable_helped = helped[helped["stable"]].copy()
    summary = {
        "qualified_pitchers": int(len(model_table)),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pvalue),
        "cv_rmse": diagnostics["cv_rmse"],
        "max_calibration_deviation": diagnostics["max_calibration_deviation"],
        "poor_calibration": diagnostics["poor_calibration"],
        "interaction_permutation_importance": perm_info["interaction_permutation_importance"],
        "interaction_permuted_label_baseline": perm_info["interaction_null_importance"],
        "interaction_mean_abs_shap": perm_info["interaction_mean_abs_shap"],
        "leaderboard_bootstrap_n": n_boot,
        "stable_hurt_command_count": int(len(stable_hurt)),
        "stable_helped_stuff_count": int(len(stable_helped)),
        "hurt_command_leaderboard": hurt.head(15).to_dict(orient="records"),
        "helped_stuff_leaderboard": helped.head(15).to_dict(orient="records"),
        "stable_hurt_command": stable_hurt.head(15).to_dict(orient="records"),
        "stable_helped_stuff": stable_helped.head(15).to_dict(orient="records"),
        "archetype_source": "proxy",
    }
    save_json(summary, ARTIFACTS_DIR / "h3_summary.json")
    return H3Result(summary=summary, model_table=model_table, hurt_command=hurt, helped_stuff=helped, bootstrap_stability=stability)

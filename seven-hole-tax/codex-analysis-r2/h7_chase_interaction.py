from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from modeling_r2 import (
    GLOBAL_SEED,
    ModelSpec,
    bootstrap_mean_ci,
    fit_lgbm_classifier,
    predict_as_lineup_spot,
    predict_with_model,
)

N_BOOTSTRAP = 500


def assign_chase_tertiles(called_pitches: pd.DataFrame, chase_rate: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = called_pitches.merge(
        chase_rate[["batter_id", "chase_rate_2025", "pa_2025", "walk_rate_2025", "contact_rate_2025"]],
        on="batter_id",
        how="left",
    )
    spot7_batters = (
        df[df["lineup_spot"].eq(7) & df["chase_rate_2025"].notna()][["batter_id", "chase_rate_2025"]]
        .drop_duplicates("batter_id")
        .copy()
    )
    if len(spot7_batters) < 3:
        df["chase_tertile"] = pd.NA
        return df, {"error": "fewer than three 7-hole batters with eligible 2025 chase rate"}

    labels = ["low", "mid", "high"]
    _, bins = pd.qcut(spot7_batters["chase_rate_2025"], q=3, labels=labels, retbins=True, duplicates="drop")
    if len(bins) < 4:
        df["chase_tertile"] = pd.NA
        return df, {"error": "could not form three chase-rate tertiles"}
    bins = np.asarray(bins, dtype=float)
    bins[0] = -np.inf
    bins[-1] = np.inf
    df["chase_tertile"] = pd.cut(df["chase_rate_2025"], bins=bins, labels=labels, include_lowest=True).astype("object")
    meta = {
        "tertile_edges": {
            "low_high": float(bins[1]),
            "mid_high": float(bins[2]),
        },
        "n_unique_spot7_batters_with_chase": int(len(spot7_batters)),
        "spot7_chase_rate_min": float(spot7_batters["chase_rate_2025"].min()),
        "spot7_chase_rate_max": float(spot7_batters["chase_rate_2025"].max()),
    }
    return df, meta


def _h7_model_frame(called_with_chase: pd.DataFrame) -> pd.DataFrame:
    required = [
        "is_called_strike",
        "game_pk",
        "lineup_spot",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
        "edge_distance_ft",
        "count_state",
        "pitcher_id",
        "catcher_id",
        "umpire",
        "pitch_type",
        "stand",
        "p_throws",
        "chase_tertile",
    ]
    df = called_with_chase.dropna(subset=required).copy()
    df["lineup_spot"] = pd.to_numeric(df["lineup_spot"], errors="coerce").astype(int)
    df = df[df["lineup_spot"].between(1, 9)].copy()
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    df["catcher_id"] = df["catcher_id"].astype(str)
    df["chase_tertile"] = df["chase_tertile"].astype(str)
    return df.reset_index(drop=True)


def train_h7_model(called_with_chase: pd.DataFrame, diag_dir: Path):
    df = _h7_model_frame(called_with_chase)
    spec = ModelSpec(
        name="h7_called_pitch_chase_interaction",
        target="is_called_strike",
        numeric_features=["plate_x", "plate_z", "sz_top", "sz_bot", "edge_distance_ft"],
        categorical_features=[
            "count_state",
            "pitcher_id",
            "catcher_id",
            "umpire",
            "pitch_type",
            "stand",
            "p_throws",
            "chase_tertile",
        ],
        include_lineup=True,
        interaction_specs=["lineup_chase_tertile"],
        permutation_groups=["lineup", "lineup_chase_interaction"],
        min_category_count=20,
        n_estimators=780,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
    )
    return fit_lgbm_classifier(df, spec, output_dir=diag_dir)


def _effect_by_tertile(model_result, model_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    boot_by_tertile: dict[str, np.ndarray] = {}
    for idx, tertile in enumerate(["low", "mid", "high"]):
        spot7 = model_frame[
            model_frame["is_borderline"].astype(bool)
            & model_frame["lineup_spot"].eq(7)
            & model_frame["chase_tertile"].eq(tertile)
        ].copy()
        spot3 = model_frame[
            model_frame["is_borderline"].astype(bool)
            & model_frame["lineup_spot"].eq(3)
            & model_frame["chase_tertile"].eq(tertile)
        ].copy()
        if spot7.empty:
            rows.append(
                {
                    "chase_tertile": tertile,
                    "effect_pp": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "n_spot7": 0,
                    "n_spot3": int(len(spot3)),
                    "actual_spot7": np.nan,
                    "actual_spot3": float(spot3["is_called_strike"].mean()) if len(spot3) else np.nan,
                }
            )
            continue
        pred_actual = predict_with_model(model_result, spot7)
        pred_three = predict_as_lineup_spot(model_result, spot7, 3)
        diff = pred_actual - pred_three
        low, high, boots = bootstrap_mean_ci(diff, n_bootstrap=N_BOOTSTRAP, seed_offset=7000 + idx)
        boot_by_tertile[tertile] = boots * 100.0
        rows.append(
            {
                "chase_tertile": tertile,
                "effect_pp": float(diff.mean() * 100.0),
                "ci_low": low * 100.0,
                "ci_high": high * 100.0,
                "n_spot7": int(len(spot7)),
                "n_spot3": int(len(spot3)),
                "actual_spot7": float(spot7["is_called_strike"].mean()),
                "actual_spot3": float(spot3["is_called_strike"].mean()) if len(spot3) else np.nan,
                "matched_actual_delta_pp": (float(spot7["is_called_strike"].mean()) - float(spot3["is_called_strike"].mean())) * 100.0
                if len(spot3)
                else np.nan,
                "n_bootstrap": N_BOOTSTRAP,
            }
        )
    return pd.DataFrame(rows), boot_by_tertile


def _interaction_test(boot_by_tertile: dict[str, np.ndarray], effect_table: pd.DataFrame) -> dict[str, float | None]:
    if "low" not in boot_by_tertile or "high" not in boot_by_tertile:
        return {"low_minus_high_pp": None, "ci_low": None, "ci_high": None, "interaction_p": None}
    rng = np.random.default_rng(GLOBAL_SEED + 770)
    low_boot = boot_by_tertile["low"]
    high_boot = boot_by_tertile["high"]
    size = min(len(low_boot), len(high_boot))
    low_sample = rng.choice(low_boot, size=size, replace=True)
    high_sample = rng.choice(high_boot, size=size, replace=True)
    contrast = low_sample - high_sample
    low_effect = float(effect_table.loc[effect_table["chase_tertile"].eq("low"), "effect_pp"].iloc[0])
    high_effect = float(effect_table.loc[effect_table["chase_tertile"].eq("high"), "effect_pp"].iloc[0])
    obs = low_effect - high_effect
    se = float(np.std(contrast, ddof=1))
    p_value = float(2 * (1 - stats.norm.cdf(abs(obs / se)))) if se > 0 else (0.0 if obs != 0 else 1.0)
    ci_low, ci_high = np.quantile(contrast, [0.025, 0.975]).tolist()
    return {
        "low_minus_high_pp": obs,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "interaction_p": p_value,
    }


def plot_h7_effects(effect_table: pd.DataFrame, path: Path) -> None:
    labels = ["low", "mid", "high"]
    df = effect_table.set_index("chase_tertile").reindex(labels).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    x = np.arange(len(df))
    y = df["effect_pp"].astype(float)
    yerr = np.vstack([(y - df["ci_low"]).to_numpy(), (df["ci_high"] - y).to_numpy()])
    ax.bar(x, y, color=["#2E86AB", "#68707A", "#D1495B"], alpha=0.86)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#222222", capsize=4, linewidth=1.1)
    ax.axhline(0, color="#111111", linewidth=1.0)
    ax.axhline(2, color="#BBBBBB", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t.title()} chase\nn7={int(n7)}, n3={int(n3)}" for t, n7, n3 in zip(df["chase_tertile"], df["n_spot7"], df["n_spot3"])])
    ax.set_ylabel("Predicted called-strike delta vs spot 3 (pp)")
    ax.set_title("H7 Spot-7 Counterfactual by 2025 Chase-Rate Tertile")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_h7(
    called_pitches: pd.DataFrame,
    chase_rate: pd.DataFrame,
    charts_dir: Path,
    artifacts_dir: Path,
    diag_dir: Path,
) -> dict[str, Any]:
    called_with_chase, tertile_meta = assign_chase_tertiles(called_pitches, chase_rate)
    if "error" in tertile_meta:
        return {"error": tertile_meta["error"], "tertile_meta": tertile_meta}
    model_result = train_h7_model(called_with_chase, diag_dir)
    model_frame = _h7_model_frame(called_with_chase)
    effect_table, boot_by_tertile = _effect_by_tertile(model_result, model_frame)
    interaction = _interaction_test(boot_by_tertile, effect_table)
    effect_table.to_csv(artifacts_dir / "h7_chase_tertile_effects.csv", index=False)
    pd.DataFrame([tertile_meta | interaction]).to_csv(artifacts_dir / "h7_chase_tertile_meta.csv", index=False)
    plot_h7_effects(effect_table, charts_dir / "h7_chase_tertile_effect.png")
    return {
        "model": model_result,
        "effect_table": effect_table,
        "interaction": interaction,
        "tertile_meta": tertile_meta,
    }

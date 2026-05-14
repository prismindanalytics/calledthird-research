from __future__ import annotations

from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from modeling import GLOBAL_SEED, ModelResult, ModelSpec, encode_frame, fit_lgbm_classifier, lineup_shap_effect


def _model_frame(
    called_pitches: pd.DataFrame,
    exclude_pinch_hitters: bool = False,
    month_bucket: str | None = None,
) -> pd.DataFrame:
    df = called_pitches.copy()
    if exclude_pinch_hitters:
        df = df[~df["is_pinch_hitter"].fillna(False)].copy()
    if month_bucket is not None:
        df = df[df["month_bucket"] == month_bucket].copy()
    required = [
        "is_called_strike",
        "lineup_spot",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
        "count_state",
        "pitcher_fame_quartile",
        "catcher_framing_tier",
        "umpire",
        "pitch_type",
        "game_pk",
    ]
    df = df.dropna(subset=required).copy()
    df["lineup_spot"] = df["lineup_spot"].astype(int)
    return df[df["lineup_spot"].between(1, 9)].reset_index(drop=True)


def run_called_pitch_model(
    called_pitches: pd.DataFrame,
    output_dir: Path,
    label: str = "called_pitch",
    include_handedness: bool = False,
    exclude_pinch_hitters: bool = False,
    month_bucket: str | None = None,
    compute_shap: bool = True,
    make_plots: bool = True,
) -> ModelResult:
    df = _model_frame(
        called_pitches,
        exclude_pinch_hitters=exclude_pinch_hitters,
        month_bucket=month_bucket,
    )
    categorical = ["count_state", "pitcher_fame_quartile", "catcher_framing_tier", "pitch_type"]
    if include_handedness:
        categorical.extend(["stand", "p_throws"])
    spec = ModelSpec(
        name=label,
        target="is_called_strike",
        numeric_features=["plate_x", "plate_z", "sz_top", "sz_bot"],
        categorical_features=categorical,
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
        min_category_count=30,
    )
    return fit_lgbm_classifier(
        df,
        spec,
        output_dir=output_dir,
        compute_shap=compute_shap,
        make_plots=make_plots,
        shap_sample_size=5000,
    )


def called_shap_payload(result: ModelResult) -> dict[str, float]:
    return lineup_shap_effect(result, 7, 3)


def plot_lineup_shap(result: ModelResult, path: Path) -> pd.DataFrame:
    if result.shap_values is None or result.shap_matrix is None:
        raise RuntimeError("SHAP values are not available for lineup SHAP plot")
    lineup_cols = [col for col in result.shap_matrix.columns if col.startswith("lineup_spot_")]
    rows = []
    for col in lineup_cols:
        idx = result.shap_matrix.columns.get_loc(col)
        rows.append(
            {
                "feature": col,
                "spot": int(col.rsplit("_", 1)[-1]),
                "mean_abs_shap": float(np.abs(result.shap_values[:, idx]).mean()),
                "mean_shap": float(result.shap_values[:, idx].mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.55, 1.0]})
    rng = np.random.default_rng(GLOBAL_SEED)
    for y_pos, feature in enumerate(summary["feature"]):
        idx = result.shap_matrix.columns.get_loc(feature)
        values = result.shap_values[:, idx]
        active = result.shap_matrix[feature].to_numpy()
        jitter = rng.normal(0, 0.055, size=len(values))
        axes[0].scatter(
            values,
            np.full(len(values), y_pos) + jitter,
            c=active,
            cmap="coolwarm",
            s=10,
            alpha=0.35,
            linewidths=0,
        )
    axes[0].axvline(0, color="#222222", linewidth=0.9)
    axes[0].set_yticks(range(len(summary)))
    axes[0].set_yticklabels([f"Spot {spot}" for spot in summary["spot"]])
    axes[0].set_xlabel("SHAP value")
    axes[0].set_title("Called-Pitch Lineup-Spot SHAP Beeswarm")
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].barh([f"Spot {spot}" for spot in summary["spot"]], summary["mean_abs_shap"], color="#2E86AB")
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_title("Lineup-Spot Attribution")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return summary.sort_values("spot").reset_index(drop=True)


def compute_lineup_location_interactions(
    result: ModelResult,
    called_pitches: pd.DataFrame,
    output_path: Path,
    sample_size: int = 1500,
) -> pd.DataFrame:
    df = _model_frame(called_pitches)
    sample = df.sample(min(sample_size, len(df)), random_state=GLOBAL_SEED).copy()
    matrix = encode_frame(sample, result.transformer)
    explainer = shap.TreeExplainer(result.model)
    interactions = explainer.shap_interaction_values(matrix)
    if isinstance(interactions, list):
        interactions = interactions[-1]
    interactions = np.asarray(interactions)
    rows: list[dict[str, Any]] = []
    for spot in range(1, 10):
        spot_col = f"lineup_spot_{spot}"
        if spot_col not in matrix.columns:
            continue
        spot_idx = matrix.columns.get_loc(spot_col)
        for loc_col in ["plate_x", "plate_z"]:
            loc_idx = matrix.columns.get_loc(loc_col)
            values = interactions[:, spot_idx, loc_idx]
            rows.append(
                {
                    "spot": spot,
                    "location_feature": loc_col,
                    "mean_abs_interaction": float(np.abs(values).mean()),
                    "mean_interaction": float(values.mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False)
    return out


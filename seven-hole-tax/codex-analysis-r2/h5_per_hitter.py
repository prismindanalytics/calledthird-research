from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling_r2 import (
    GLOBAL_SEED,
    ModelSpec,
    bh_fdr,
    bootstrap_mean_ci,
    bootstrap_two_sided_p,
    fit_lgbm_classifier,
    predict_with_model,
)

N_BOOTSTRAP = 500
N_PERMUTATIONS = 1000


def _h5_model_frame(called_pitches: pd.DataFrame) -> pd.DataFrame:
    required = [
        "is_called_strike",
        "game_pk",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
        "edge_distance_ft",
        "count_state",
        "pitcher_id",
        "catcher_id",
        "umpire",
    ]
    df = called_pitches.dropna(subset=required).copy()
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    df["catcher_id"] = df["catcher_id"].astype(str)
    return df.reset_index(drop=True)


def train_h5_model(called_pitches: pd.DataFrame, diag_dir: Path):
    df = _h5_model_frame(called_pitches)
    spec = ModelSpec(
        name="h5_no_batter_called_pitch",
        target="is_called_strike",
        numeric_features=["plate_x", "plate_z", "sz_top", "sz_bot", "edge_distance_ft"],
        categorical_features=["count_state", "pitcher_id", "catcher_id", "umpire"],
        include_lineup=False,
        interaction_specs=[],
        permutation_groups=["umpire_main", "pitcher_main", "catcher_main"],
        min_category_count=20,
        n_estimators=750,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=85,
    )
    return fit_lgbm_classifier(df, spec, output_dir=diag_dir)


def _qualified_hitter_residuals(
    eval_frame: pd.DataFrame,
    predictions: np.ndarray,
    chase_rate: pd.DataFrame,
    min_n: int = 30,
    exclude_pinch_hitters: bool = False,
) -> pd.DataFrame:
    df = eval_frame.copy()
    df["expected"] = predictions
    if exclude_pinch_hitters:
        df = df[~df["is_pinch_hitter"].fillna(False)].copy()
    df = df[df["is_borderline"].astype(bool) & df["lineup_spot"].isin([7, 8, 9])].copy()
    rows: list[dict[str, Any]] = []
    for idx, (batter_id, group) in enumerate(df.groupby("batter_id", observed=True)):
        if len(group) < min_n:
            continue
        residuals = group["is_called_strike"].astype(float).to_numpy() - group["expected"].astype(float).to_numpy()
        deviation = float(residuals.mean())
        low, high, boots = bootstrap_mean_ci(residuals, n_bootstrap=N_BOOTSTRAP, seed_offset=2000 + idx)
        rows.append(
            {
                "batter_id": int(batter_id),
                "batter_name": str(group["batter_name"].dropna().iloc[0]) if group["batter_name"].notna().any() else f"batter_{batter_id}",
                "n_borderline": int(len(group)),
                "actual": float(group["is_called_strike"].mean()),
                "expected": float(group["expected"].mean()),
                "deviation": deviation,
                "deviation_pp": deviation * 100.0,
                "ci_low": low * 100.0,
                "ci_high": high * 100.0,
                "p_value": bootstrap_two_sided_p(deviation, boots),
                "n_pinch_hitter_rows": int(group["is_pinch_hitter"].fillna(False).sum()),
                "n_bootstrap": N_BOOTSTRAP,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value"] = bh_fdr(out["p_value"])
    out["ci_excludes_zero"] = (out["ci_low"] > 0) | (out["ci_high"] < 0)
    out["flagged"] = out["q_value"].lt(0.10) & out["deviation_pp"].abs().ge(3.0) & out["ci_excludes_zero"]
    out["tax_flagged"] = out["flagged"] & out["deviation_pp"].gt(0)
    chase_cols = [
        "batter_id",
        "chase_rate_2025",
        "pa_2025",
        "walk_rate_2025",
        "strikeout_rate_2025",
        "contact_rate_2025",
    ]
    out = out.merge(chase_rate[chase_cols], on="batter_id", how="left")
    return out.sort_values("deviation_pp", ascending=False).reset_index(drop=True)


def hitter_residual_permutation_baseline(leaderboard: pd.DataFrame, eval_frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, Any]:
    if leaderboard.empty:
        return {"n_permutations": N_PERMUTATIONS, "observed_max_abs_pp": None, "permutation_p": None}
    df = eval_frame.copy()
    df["expected"] = predictions
    df = df[df["is_borderline"].astype(bool) & df["lineup_spot"].isin([7, 8, 9])].copy()
    keep = set(leaderboard["batter_id"].astype(int))
    df = df[df["batter_id"].astype(int).isin(keep)].copy()
    residuals = df["is_called_strike"].astype(float).to_numpy() - df["expected"].astype(float).to_numpy()
    labels = df["batter_id"].astype(int).to_numpy()
    unique = leaderboard["batter_id"].astype(int).tolist()
    observed_max = float(leaderboard["deviation_pp"].abs().max())
    rng = np.random.default_rng(GLOBAL_SEED + 517)
    null = np.empty(N_PERMUTATIONS, dtype=float)
    for perm_idx in range(N_PERMUTATIONS):
        shuffled = rng.permutation(residuals)
        max_abs = 0.0
        for batter_id in unique:
            mean = float(shuffled[labels == batter_id].mean()) * 100.0
            max_abs = max(max_abs, abs(mean))
        null[perm_idx] = max_abs
    p_value = float((np.sum(null >= observed_max) + 1) / (N_PERMUTATIONS + 1))
    return {
        "n_permutations": N_PERMUTATIONS,
        "observed_max_abs_pp": observed_max,
        "null_mean_max_abs_pp": float(null.mean()),
        "null_p95_max_abs_pp": float(np.quantile(null, 0.95)),
        "permutation_p": p_value,
    }


def plot_h5_residuals(leaderboard: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, max(6.0, min(15.0, 0.34 * max(len(leaderboard), 1) + 1.5))))
    if leaderboard.empty:
        ax.text(0.5, 0.5, "No qualifying hitters", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        df = leaderboard.sort_values("deviation_pp").reset_index(drop=True)
        y = np.arange(len(df))
        colors = np.where(df["tax_flagged"], "#D1495B", np.where(df["flagged"], "#2E86AB", "#68707A"))
        ax.errorbar(
            df["deviation_pp"],
            y,
            xerr=np.vstack([(df["deviation_pp"] - df["ci_low"]).to_numpy(), (df["ci_high"] - df["deviation_pp"]).to_numpy()]),
            fmt="none",
            ecolor="#222222",
            capsize=2.5,
            linewidth=0.9,
        )
        ax.scatter(df["deviation_pp"], y, c=colors, s=34, zorder=2)
        ax.axvline(0, color="#111111", linewidth=1.0)
        ax.axvline(3, color="#BBBBBB", linestyle="--", linewidth=0.9)
        ax.axvline(-3, color="#BBBBBB", linestyle="--", linewidth=0.9)
        labels = [f"{name} ({n})" for name, n in zip(df["batter_name"], df["n_borderline"])]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Actual minus expected called-strike rate (pp)")
        ax.set_title("H5 Per-Hitter Borderline Residuals, Spots 7-9")
        ax.grid(axis="x", alpha=0.2)
        for row_idx, row in df.iterrows():
            if row["flagged"] or row_idx in {0, len(df) - 1}:
                ax.text(
                    row["ci_high"] + 0.12,
                    row_idx,
                    f"q={row['q_value']:.2f}",
                    va="center",
                    ha="left",
                    fontsize=7,
                    color="#333333",
                )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_h5(
    called_pitches: pd.DataFrame,
    chase_rate: pd.DataFrame,
    charts_dir: Path,
    artifacts_dir: Path,
    diag_dir: Path,
) -> dict[str, Any]:
    model_result = train_h5_model(called_pitches, diag_dir)
    eval_frame = _h5_model_frame(called_pitches)
    for col in ["lineup_spot", "is_borderline", "batter_id", "batter_name", "is_pinch_hitter"]:
        if col not in eval_frame.columns and col in called_pitches.columns:
            eval_frame[col] = called_pitches.loc[eval_frame.index, col].to_numpy()
    predictions = predict_with_model(model_result, eval_frame)

    leaderboard = _qualified_hitter_residuals(eval_frame, predictions, chase_rate, min_n=30, exclude_pinch_hitters=False)
    no_pinch = _qualified_hitter_residuals(eval_frame, predictions, chase_rate, min_n=30, exclude_pinch_hitters=True)
    perm = hitter_residual_permutation_baseline(leaderboard, eval_frame, predictions)

    leaderboard.to_csv(artifacts_dir / "h5_per_hitter_residuals.csv", index=False)
    no_pinch.to_csv(artifacts_dir / "h5_per_hitter_residuals_no_pinch.csv", index=False)
    pd.DataFrame([perm]).to_csv(artifacts_dir / "h5_hitter_residual_permutation_baseline.csv", index=False)
    plot_h5_residuals(leaderboard, charts_dir / "h5_per_hitter_residuals.png")
    return {
        "model": model_result,
        "leaderboard": leaderboard,
        "no_pinch": no_pinch,
        "permutation_baseline": perm,
        "n_qualifying": int(len(leaderboard)),
    }

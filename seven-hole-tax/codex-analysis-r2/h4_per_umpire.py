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
    predict_as_lineup_spot,
    predict_with_model,
)

N_BOOTSTRAP = 500


def _called_pitch_model_frame(called_pitches: pd.DataFrame) -> pd.DataFrame:
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
    ]
    df = called_pitches.dropna(subset=required).copy()
    df["lineup_spot"] = pd.to_numeric(df["lineup_spot"], errors="coerce").astype(int)
    df = df[df["lineup_spot"].between(1, 9)].copy()
    df["pitcher_id"] = df["pitcher_id"].astype(str)
    df["catcher_id"] = df["catcher_id"].astype(str)
    return df.reset_index(drop=True)


def train_h4_model(called_pitches: pd.DataFrame, diag_dir: Path):
    df = _called_pitch_model_frame(called_pitches)
    spec = ModelSpec(
        name="h4_called_pitch_umpire_interaction",
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
        ],
        include_lineup=True,
        interaction_specs=["umpire_lineup"],
        permutation_groups=["lineup", "umpire_lineup_interaction"],
        min_category_count=20,
        n_estimators=800,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
    )
    return fit_lgbm_classifier(df, spec, output_dir=diag_dir)


def _h4_qualifying_counts(borderline: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for umpire, group in borderline.groupby("umpire", observed=True):
        lineup = pd.to_numeric(group["lineup_spot"], errors="coerce").astype(int)
        n_bottom = int(lineup.isin([7, 8, 9]).sum())
        n_top = int(lineup.isin([1, 2, 3]).sum())
        rows.append({"umpire": str(umpire), "n_bottom_7_9": n_bottom, "n_top_1_3": n_top, "n_borderline": int(len(group))})
    out = pd.DataFrame(rows)
    out["qualifies"] = out["n_bottom_7_9"].ge(50) & out["n_top_1_3"].ge(50)
    return out.sort_values(["qualifies", "n_bottom_7_9"], ascending=[False, False]).reset_index(drop=True)


def per_umpire_counterfactuals(model_result, called_pitches: pd.DataFrame, artifacts_dir: Path, charts_dir: Path) -> dict[str, Any]:
    base = _called_pitch_model_frame(called_pitches)
    borderline = base[base["is_borderline"].astype(bool)].copy()
    counts = _h4_qualifying_counts(borderline)
    counts.to_csv(artifacts_dir / "h4_umpire_sample_counts.csv", index=False)
    qualifying_umpires = counts[counts["qualifies"]]["umpire"].tolist()

    rows: list[dict[str, Any]] = []
    for idx, umpire in enumerate(qualifying_umpires):
        subset = borderline[(borderline["umpire"].astype(str) == umpire) & (borderline["lineup_spot"].isin([7, 8, 9]))].copy()
        if subset.empty:
            continue
        pred_actual = predict_with_model(model_result, subset)
        pred_three = predict_as_lineup_spot(model_result, subset, 3)
        diff = pred_actual - pred_three
        effect = float(diff.mean())
        low, high, boots = bootstrap_mean_ci(diff, n_bootstrap=N_BOOTSTRAP, seed_offset=1000 + idx)
        rows.append(
            {
                "umpire": umpire,
                "effect": effect,
                "effect_pp": effect * 100.0,
                "ci_low": low * 100.0,
                "ci_high": high * 100.0,
                "p_value": bootstrap_two_sided_p(effect, boots),
                "n_calls": int(len(subset)),
                "n_bottom_7_9": int(len(subset)),
                "n_top_1_3": int(counts.loc[counts["umpire"] == umpire, "n_top_1_3"].iloc[0]),
                "n_bootstrap": N_BOOTSTRAP,
            }
        )

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        leaderboard = pd.DataFrame(
            columns=[
                "umpire",
                "effect",
                "effect_pp",
                "ci_low",
                "ci_high",
                "p_value",
                "q_value",
                "n_calls",
                "n_bottom_7_9",
                "n_top_1_3",
                "n_bootstrap",
                "flagged",
            ]
        )
    else:
        leaderboard["q_value"] = bh_fdr(leaderboard["p_value"])
        leaderboard["ci_excludes_zero"] = (leaderboard["ci_low"] > 0) | (leaderboard["ci_high"] < 0)
        leaderboard["flagged"] = leaderboard["q_value"].lt(0.10) & leaderboard["effect_pp"].abs().ge(2.0)
        leaderboard = leaderboard.sort_values("effect_pp").reset_index(drop=True)

    leaderboard.to_csv(artifacts_dir / "h4_per_umpire_leaderboard.csv", index=False)
    plot_h4_leaderboard(leaderboard, charts_dir / "h4_per_umpire_leaderboard.png")
    return {
        "model": model_result,
        "leaderboard": leaderboard,
        "counts": counts,
        "n_qualifying": int(len(leaderboard)),
        "n_total_umpires": int(counts["umpire"].nunique()) if not counts.empty else 0,
    }


def plot_h4_leaderboard(leaderboard: pd.DataFrame, path: Path) -> None:
    fig_height = max(6.0, min(18.0, 0.3 * max(len(leaderboard), 1) + 1.5))
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    if leaderboard.empty:
        ax.text(0.5, 0.5, "No qualifying umpires", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
    else:
        df = leaderboard.sort_values("effect_pp").reset_index(drop=True)
        y = np.arange(len(df))
        colors = np.where(df["flagged"], "#D1495B", "#68707A")
        ax.errorbar(
            df["effect_pp"],
            y,
            xerr=np.vstack([(df["effect_pp"] - df["ci_low"]).to_numpy(), (df["ci_high"] - df["effect_pp"]).to_numpy()]),
            fmt="none",
            ecolor="#222222",
            capsize=2.5,
            linewidth=0.9,
            zorder=1,
        )
        ax.scatter(df["effect_pp"], y, c=colors, s=34, zorder=2)
        ax.axvline(0, color="#111111", linewidth=1.0)
        ax.axvline(2, color="#BBBBBB", linestyle="--", linewidth=0.9)
        ax.axvline(-2, color="#BBBBBB", linestyle="--", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(df["umpire"], fontsize=8)
        ax.set_xlabel("Predicted called-strike delta: actual bottom spot vs counterfactual spot 3 (pp)")
        ax.set_title("H4 Per-Umpire Bottom-Order Counterfactuals")
        ax.grid(axis="x", alpha=0.2)
        for row_idx, row in df.iterrows():
            ax.text(
                row["ci_high"] + 0.05,
                row_idx,
                f"q={row['q_value']:.2f}, n={int(row['n_calls'])}",
                va="center",
                ha="left",
                fontsize=7,
                color="#333333",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_h4(called_pitches: pd.DataFrame, charts_dir: Path, artifacts_dir: Path, diag_dir: Path) -> dict[str, Any]:
    model_result = train_h4_model(called_pitches, diag_dir)
    return per_umpire_counterfactuals(model_result, called_pitches, artifacts_dir, charts_dir)

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling import GLOBAL_SEED, ModelResult, predict_with_model


def _predict_as_spot(result: ModelResult, df: pd.DataFrame, spot: int) -> np.ndarray:
    modified = df.copy()
    modified["lineup_spot"] = int(spot)
    return predict_with_model(result, modified)


def bootstrap_delta(values_a: np.ndarray, values_b: np.ndarray, n_bootstrap: int = 200) -> tuple[float, float]:
    if len(values_a) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(GLOBAL_SEED)
    deltas = np.empty(n_bootstrap)
    idx = np.arange(len(values_a))
    diff = values_a - values_b
    for boot_idx in range(n_bootstrap):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        deltas[boot_idx] = diff[sample_idx].mean()
    return tuple(np.quantile(deltas, [0.025, 0.975]).tolist())


def spot_vs_three_effect(
    result: ModelResult,
    df: pd.DataFrame,
    actual_spot: int,
    n_bootstrap: int = 200,
) -> dict[str, Any]:
    subset = df[df["lineup_spot"].astype(int) == actual_spot].copy()
    if subset.empty:
        return {
            "spot": actual_spot,
            "comparison": f"{actual_spot}v3",
            "effect": float("nan"),
            "effect_pp": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
            "n_bootstrap": n_bootstrap,
        }
    pred_actual = _predict_as_spot(result, subset, actual_spot)
    pred_three = _predict_as_spot(result, subset, 3)
    effect = float(np.mean(pred_actual - pred_three))
    low, high = bootstrap_delta(pred_actual, pred_three, n_bootstrap=n_bootstrap)
    return {
        "spot": actual_spot,
        "comparison": f"{actual_spot}v3",
        "effect": effect,
        "effect_pp": effect * 100.0,
        "ci_low": float(low) * 100.0,
        "ci_high": float(high) * 100.0,
        "n": int(len(subset)),
        "n_bootstrap": n_bootstrap,
    }


def counterfactual_leaderboard(
    result: ModelResult,
    df: pd.DataFrame,
    n_bootstrap: int = 200,
) -> pd.DataFrame:
    clean = df.dropna(subset=["lineup_spot"]).copy()
    clean["lineup_spot"] = clean["lineup_spot"].astype(int)
    rows = [spot_vs_three_effect(result, clean, spot, n_bootstrap=n_bootstrap) for spot in range(1, 10)]
    return pd.DataFrame(rows)


def plot_counterfactual_leaderboard(leaderboard: pd.DataFrame, path: Path, title: str) -> None:
    df = leaderboard.copy().sort_values("effect_pp")
    colors = ["#D1495B" if spot == 7 else "#2E86AB" if spot == 3 else "#6C757D" for spot in df["spot"]]
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.barh(df["comparison"], df["effect_pp"], color=colors)
    ax.errorbar(
        df["effect_pp"],
        np.arange(len(df)),
        xerr=np.vstack([(df["effect_pp"] - df["ci_low"]).to_numpy(), (df["ci_high"] - df["effect_pp"]).to_numpy()]),
        fmt="none",
        ecolor="#222222",
        capsize=3,
        linewidth=1.0,
    )
    ax.axvline(0, color="#111111", linewidth=1.0)
    ax.set_xlabel("Predicted-probability delta vs spot 3 (pp)")
    ax.set_ylabel("Actual lineup spot")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    for y_pos, row in enumerate(df.itertuples(index=False)):
        ax.text(
            row.effect_pp + (0.03 if row.effect_pp >= 0 else -0.03),
            y_pos,
            f"n={int(row.n)}",
            va="center",
            ha="left" if row.effect_pp >= 0 else "right",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_counterfactuals(
    challenge_result: ModelResult,
    called_result: ModelResult,
    challenges: pd.DataFrame,
    called_pitches: pd.DataFrame,
    charts_dir: Path,
    artifacts_dir: Path,
    n_bootstrap: int = 200,
) -> dict[str, Any]:
    challenge_clean = challenges.dropna(subset=["lineup_spot", "edge_distance_in", "plate_x", "plate_z"]).copy()
    challenge_leaderboard = counterfactual_leaderboard(challenge_result, challenge_clean, n_bootstrap=n_bootstrap)

    called_clean = called_pitches.dropna(subset=["lineup_spot", "plate_x", "plate_z", "sz_top", "sz_bot"]).copy()
    called_all = counterfactual_leaderboard(called_result, called_clean, n_bootstrap=n_bootstrap)
    called_borderline = counterfactual_leaderboard(
        called_result,
        called_clean[called_clean["is_borderline"].astype(bool)].copy(),
        n_bootstrap=n_bootstrap,
    )

    challenge_leaderboard.to_csv(artifacts_dir / "challenge_counterfactual_leaderboard.csv", index=False)
    called_all.to_csv(artifacts_dir / "called_pitch_counterfactual_leaderboard_all.csv", index=False)
    called_borderline.to_csv(artifacts_dir / "called_pitch_counterfactual_leaderboard_borderline.csv", index=False)
    plot_counterfactual_leaderboard(
        called_borderline,
        charts_dir / "counterfactual_leaderboard.png",
        "Borderline Called-Pitch Counterfactual: Each Spot vs Spot 3",
    )

    h2 = challenge_leaderboard[challenge_leaderboard["spot"] == 7].iloc[0].to_dict()
    h3 = called_borderline[called_borderline["spot"] == 7].iloc[0].to_dict()
    return {
        "challenge_leaderboard": challenge_leaderboard,
        "called_leaderboard_all": called_all,
        "called_leaderboard_borderline": called_borderline,
        "h2_spot_7_vs_3": h2,
        "h3_spot_7_vs_3": h3,
    }


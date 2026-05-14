from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling import GLOBAL_SEED, ModelResult, ModelSpec, fit_lgbm_classifier, lineup_shap_effect


def bootstrap_rate(values: np.ndarray, n_bootstrap: int = 1000) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(GLOBAL_SEED)
    rates = np.empty(n_bootstrap)
    for idx in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        rates[idx] = sample.mean()
    return tuple(np.quantile(rates, [0.025, 0.975]).tolist())


def h1_overturn_by_spot(challenges: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    clean = challenges.dropna(subset=["lineup_spot", "overturned"]).copy()
    clean["lineup_spot"] = clean["lineup_spot"].astype(int)
    for spot in range(1, 10):
        subset = clean[clean["lineup_spot"] == spot]["overturned"].astype(int).to_numpy()
        rate = float(subset.mean()) if len(subset) else float("nan")
        low, high = bootstrap_rate(subset)
        rows.append(
            {
                "spot": spot,
                "rate": rate,
                "ci_low": float(low),
                "ci_high": float(high),
                "n": int(len(subset)),
            }
        )
    return rows


def plot_h1_overturn_by_spot(summary: list[dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame(summary)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    yerr = np.vstack([(df["rate"] - df["ci_low"]).to_numpy(), (df["ci_high"] - df["rate"]).to_numpy()])
    colors = ["#D1495B" if spot == 7 else "#2E86AB" if spot == 3 else "#6C757D" for spot in df["spot"]]
    ax.bar(df["spot"].astype(str), df["rate"] * 100, color=colors, alpha=0.9)
    ax.errorbar(
        np.arange(len(df)),
        df["rate"] * 100,
        yerr=yerr * 100,
        fmt="none",
        ecolor="#222222",
        capsize=4,
        linewidth=1.2,
    )
    league = np.average(df["rate"], weights=df["n"])
    ax.axhline(league * 100, color="#111111", linestyle="--", linewidth=1.1, label=f"League {league * 100:.1f}%")
    ax.set_xlabel("Lineup spot")
    ax.set_ylabel("ABS challenge overturn rate")
    ax.set_title("Observed ABS Challenge Overturn Rate by Batter Lineup Spot")
    ax.set_ylim(0, max(75, float(df["ci_high"].max() * 115)))
    for idx, row in df.iterrows():
        ax.text(idx, max(1.5, row["rate"] * 100 - 6), f"n={int(row['n'])}", ha="center", va="center", color="white", fontsize=8)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _model_frame(
    challenges: pd.DataFrame,
    exclude_pinch_hitters: bool = False,
) -> pd.DataFrame:
    df = challenges.copy()
    if exclude_pinch_hitters:
        df = df[~df["is_pinch_hitter"].fillna(False)].copy()
    required = [
        "overturned",
        "lineup_spot",
        "edge_distance_in",
        "plate_x",
        "plate_z",
        "in_zone",
        "count_state",
        "pitcher_fame_quartile",
        "catcher_framing_tier",
        "umpire",
        "game_pk",
    ]
    df = df.dropna(subset=required).copy()
    df["lineup_spot"] = df["lineup_spot"].astype(int)
    return df[df["lineup_spot"].between(1, 9)].reset_index(drop=True)


def run_challenge_model(
    challenges: pd.DataFrame,
    output_dir: Path,
    label: str = "challenge",
    include_handedness: bool = False,
    exclude_pinch_hitters: bool = False,
    compute_shap: bool = True,
    make_plots: bool = True,
) -> ModelResult:
    df = _model_frame(challenges, exclude_pinch_hitters=exclude_pinch_hitters)
    categorical = ["count_state", "pitcher_fame_quartile", "catcher_framing_tier"]
    if include_handedness:
        categorical.extend(["stand", "p_throws"])
    spec = ModelSpec(
        name=label,
        target="overturned",
        numeric_features=["edge_distance_in", "plate_x", "plate_z", "in_zone"],
        categorical_features=categorical,
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=25,
        min_category_count=5,
    )
    return fit_lgbm_classifier(
        df,
        spec,
        output_dir=output_dir,
        compute_shap=compute_shap,
        make_plots=make_plots,
        shap_sample_size=2500,
    )


def challenge_shap_payload(result: ModelResult) -> dict[str, float]:
    return lineup_shap_effect(result, 7, 3)


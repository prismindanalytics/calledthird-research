from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, pdist

from modeling import GLOBAL_SEED


FEATURES = ["edge_distance_in", "plate_x", "plate_z", "count_index", "pitcher_fame_numeric"]


def _count_index(count_state: pd.Series) -> pd.Series:
    order = {
        "0-0": 0,
        "1-0": 1,
        "0-1": 2,
        "2-0": 3,
        "1-1": 4,
        "0-2": 5,
        "3-0": 6,
        "2-1": 7,
        "1-2": 8,
        "3-1": 9,
        "2-2": 10,
        "3-2": 11,
    }
    return count_state.map(order).fillna(-1)


def _pitcher_fame_numeric(series: pd.Series) -> pd.Series:
    mapping = {"Q1_low": 1, "Q2": 2, "Q3": 3, "Q4_high": 4, "unknown": 2.5}
    return series.fillna("unknown").map(mapping).fillna(2.5)


def _standardized_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    spots = df[df["lineup_spot"].isin([3, 7])].copy()
    matrix = spots[FEATURES].astype(float)
    matrix = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, 1)
    a = matrix[spots["lineup_spot"] == 7].to_numpy(dtype=float)
    b = matrix[spots["lineup_spot"] == 3].to_numpy(dtype=float)
    return a, b


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(2.0 * cdist(a, b).mean() - pdist(a).mean() - pdist(b).mean())


def energy_permutation_test(df: pd.DataFrame, n_permutations: int = 500) -> dict[str, float]:
    a, b = _standardized_arrays(df)
    observed = energy_distance(a, b)
    combined = np.vstack([a, b])
    n_a = len(a)
    rng = np.random.default_rng(GLOBAL_SEED)
    null = np.empty(n_permutations)
    for idx in range(n_permutations):
        order = rng.permutation(len(combined))
        null[idx] = energy_distance(combined[order[:n_a]], combined[order[n_a:]])
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return {
        "energy_distance": float(observed),
        "permutation_p": p_value,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "n_spot_7": int(n_a),
        "n_spot_3": int(len(b)),
        "n_permutations": n_permutations,
    }


def univariate_ks(df: pd.DataFrame) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    spot7 = df[df["lineup_spot"] == 7]
    spot3 = df[df["lineup_spot"] == 3]
    for feature in FEATURES:
        a = spot7[feature].dropna().astype(float)
        b = spot3[feature].dropna().astype(float)
        if len(a) < 2 or len(b) < 2:
            stat = pvalue = float("nan")
        else:
            test = stats.ks_2samp(a, b)
            stat = float(test.statistic)
            pvalue = float(test.pvalue)
        rows.append(
            {
                "feature": feature,
                "ks_statistic": stat,
                "p_value": pvalue,
                "bonferroni_p": min(1.0, pvalue * len(FEATURES)) if np.isfinite(pvalue) else float("nan"),
                "spot_7_mean": float(a.mean()) if len(a) else float("nan"),
                "spot_3_mean": float(b.mean()) if len(b) else float("nan"),
            }
        )
    return rows


def _plot_density(ax: plt.Axes, a: pd.Series, b: pd.Series, title: str) -> None:
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    values = pd.concat([a, b])
    if len(values) < 5 or values.nunique() < 3:
        bins = min(10, max(3, int(values.nunique())))
        ax.hist(b, bins=bins, density=True, alpha=0.45, label="Spot 3", color="#2E86AB")
        ax.hist(a, bins=bins, density=True, alpha=0.45, label="Spot 7", color="#D1495B")
    else:
        x_grid = np.linspace(values.quantile(0.01), values.quantile(0.99), 200)
        for series, color, label in [(b, "#2E86AB", "Spot 3"), (a, "#D1495B", "Spot 7")]:
            if len(series) > 3 and series.nunique() > 2:
                kde = stats.gaussian_kde(series)
                ax.plot(x_grid, kde(x_grid), color=color, linewidth=2, label=label)
                ax.fill_between(x_grid, kde(x_grid), color=color, alpha=0.16)
    ax.set_title(title)
    ax.grid(alpha=0.2)


def plot_selection_marginals(df: pd.DataFrame, path: Path) -> None:
    spot7 = df[df["lineup_spot"] == 7]
    spot3 = df[df["lineup_spot"] == 3]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    _plot_density(axes[0, 0], spot7["edge_distance_in"], spot3["edge_distance_in"], "Edge distance (inches)")
    _plot_density(axes[0, 1], spot7["plate_x"], spot3["plate_x"], "Horizontal location")
    _plot_density(axes[1, 0], spot7["plate_z"], spot3["plate_z"], "Vertical location")
    _plot_density(axes[1, 1], spot7["pitcher_fame_numeric"], spot3["pitcher_fame_numeric"], "Pitcher K-BB quartile")
    for ax in axes.ravel():
        ax.legend(frameon=False)
    fig.suptitle("Challenge Selection: Spot 7 vs Spot 3 Marginal Distributions", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_selection_probe(
    challenges: pd.DataFrame,
    charts_dir: Path,
    artifacts_dir: Path,
    n_permutations: int = 500,
) -> dict[str, Any]:
    df = challenges.dropna(subset=["lineup_spot", "edge_distance_in", "plate_x", "plate_z", "count_state"]).copy()
    df["lineup_spot"] = df["lineup_spot"].astype(int)
    df = df[df["lineup_spot"].isin([3, 7])].copy()
    df["count_index"] = _count_index(df["count_state"])
    df["pitcher_fame_numeric"] = _pitcher_fame_numeric(df["pitcher_fame_quartile"])
    df = df.dropna(subset=FEATURES)

    energy = energy_permutation_test(df, n_permutations=n_permutations)
    ks_rows = univariate_ks(df)
    pd.DataFrame(ks_rows).to_csv(artifacts_dir / "selection_univariate_ks.csv", index=False)
    pd.DataFrame([energy]).to_csv(artifacts_dir / "selection_energy_distance.csv", index=False)
    plot_selection_marginals(df, charts_dir / "selection_effect_marginals.png")
    interpretation = (
        "large selection difference; raw challenge rates are likely selection-contaminated"
        if energy["permutation_p"] < 0.01 and energy["energy_distance"] > energy["null_p95"]
        else "no decisive multivariate selection shift between spot 7 and spot 3 challenges"
    )
    energy["interpretation"] = interpretation
    return {"energy": energy, "ks": ks_rows}


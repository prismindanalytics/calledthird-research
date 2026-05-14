from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, pdist

from modeling_r2 import GLOBAL_SEED, bootstrap_mean_ci, count_state_index, pitcher_fame_numeric

FEATURES = ["edge_distance_in", "in_zone", "count_state_encoded", "pitcher_fame_numeric"]
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000


def _h6_frame(challenges: pd.DataFrame) -> pd.DataFrame:
    df = challenges.copy()
    df = df[df["challenger"].astype(str).str.lower().eq("catcher")].copy()
    df = df[df["lineup_spot"].isin([3, 7])].copy()
    df["lineup_spot"] = df["lineup_spot"].astype(int)
    df["count_state_encoded"] = count_state_index(df["count_state"])
    df["pitcher_fame_numeric"] = pitcher_fame_numeric(df["pitcher_fame_quartile"])
    for col in ["edge_distance_in", "in_zone"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=FEATURES + ["lineup_spot"]).reset_index(drop=True)


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(2.0 * cdist(a, b).mean() - pdist(a).mean() - pdist(b).mean())


def energy_test(df: pd.DataFrame) -> dict[str, Any]:
    matrix = df[FEATURES].astype(float)
    matrix = (matrix - matrix.mean()) / matrix.std(ddof=0).replace(0, 1)
    a = matrix[df["lineup_spot"].eq(7)].to_numpy(dtype=float)
    b = matrix[df["lineup_spot"].eq(3)].to_numpy(dtype=float)
    observed = energy_distance(a, b)
    combined = np.vstack([a, b])
    n_a = len(a)
    rng = np.random.default_rng(GLOBAL_SEED + 610)
    null = np.empty(N_PERMUTATIONS, dtype=float)
    for idx in range(N_PERMUTATIONS):
        order = rng.permutation(len(combined))
        null[idx] = energy_distance(combined[order[:n_a]], combined[order[n_a:]])
    p_value = float((np.sum(null >= observed) + 1) / (N_PERMUTATIONS + 1))
    return {
        "energy_distance": float(observed),
        "permutation_p": p_value,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "null_values": null,
        "n_spot_7": int(n_a),
        "n_spot_3": int(len(b)),
        "n_permutations": N_PERMUTATIONS,
    }


def univariate_ks(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    spot7 = df[df["lineup_spot"].eq(7)]
    spot3 = df[df["lineup_spot"].eq(3)]
    for feature in FEATURES:
        a = spot7[feature].dropna().astype(float)
        b = spot3[feature].dropna().astype(float)
        if len(a) < 2 or len(b) < 2:
            stat = pvalue = float("nan")
        else:
            test = stats.ks_2samp(a, b)
            stat, pvalue = float(test.statistic), float(test.pvalue)
        rows.append(
            {
                "feature": feature,
                "ks_statistic": stat,
                "p_value": pvalue,
                "bonferroni_p": min(1.0, pvalue * len(FEATURES)) if np.isfinite(pvalue) else float("nan"),
                "spot_7_mean": float(a.mean()) if len(a) else float("nan"),
                "spot_3_mean": float(b.mean()) if len(b) else float("nan"),
                "spot_7_n": int(len(a)),
                "spot_3_n": int(len(b)),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_feature_delta(df: pd.DataFrame, feature: str) -> tuple[float, float, float]:
    spot7 = df[df["lineup_spot"].eq(7)][feature].astype(float).to_numpy()
    spot3 = df[df["lineup_spot"].eq(3)][feature].astype(float).to_numpy()
    rng = np.random.default_rng(GLOBAL_SEED + 640 + len(feature))
    boots = np.empty(N_BOOTSTRAP, dtype=float)
    for idx in range(N_BOOTSTRAP):
        a = rng.choice(spot7, size=len(spot7), replace=True)
        b = rng.choice(spot3, size=len(spot3), replace=True)
        boots[idx] = float(a.mean() - b.mean())
    obs = float(spot7.mean() - spot3.mean())
    low, high = np.quantile(boots, [0.025, 0.975]).tolist()
    return obs, float(low), float(high)


def plot_h6_energy(payload: dict[str, Any], path: Path) -> None:
    null = payload["null_values"]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.hist(null, bins=32, color="#2E86AB", alpha=0.82, label="Permuted spot labels")
    ax.axvline(payload["energy_distance"], color="#D1495B", linewidth=2.2, label="Observed")
    ax.axvline(payload["null_p95"], color="#555555", linestyle="--", linewidth=1.2, label="Null p95")
    ax.set_xlabel("Standardized energy distance")
    ax.set_ylabel("Permutations")
    ax.set_title("H6 Catcher-Initiated Challenge Selection: Spot 7 vs Spot 3")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_h6(challenges: pd.DataFrame, charts_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    df = _h6_frame(challenges)
    energy = energy_test(df)
    ks = univariate_ks(df)
    in_zone_delta, in_zone_low, in_zone_high = _bootstrap_feature_delta(df, "in_zone")
    edge_delta, edge_low, edge_high = _bootstrap_feature_delta(df, "edge_distance_in")

    energy_for_save = {k: v for k, v in energy.items() if k != "null_values"}
    summary = {
        **energy_for_save,
        "in_zone_delta_pp": in_zone_delta * 100.0,
        "in_zone_ci_low_pp": in_zone_low * 100.0,
        "in_zone_ci_high_pp": in_zone_high * 100.0,
        "edge_distance_delta_in": edge_delta,
        "edge_distance_ci_low_in": edge_low,
        "edge_distance_ci_high_in": edge_high,
    }
    pd.DataFrame([summary]).to_csv(artifacts_dir / "h6_catcher_energy_distance.csv", index=False)
    ks.to_csv(artifacts_dir / "h6_catcher_univariate_ks.csv", index=False)
    plot_h6_energy(energy, charts_dir / "h6_catcher_mechanism.png")
    interpretation = (
        "positive catcher-selection shift"
        if energy["permutation_p"] < 0.05
        else "no decisive catcher pitch-selection shift"
    )
    return {
        "frame": df,
        "energy": summary,
        "ks": ks,
        "interpretation": interpretation,
    }

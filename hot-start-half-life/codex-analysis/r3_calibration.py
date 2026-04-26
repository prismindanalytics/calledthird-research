from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATASETS_DIR, atomic_write_json, set_plot_style
from r2_utils import (
    add_pitcher_prior_columns,
    add_r2_hitter_columns,
    conformal_interval_margin,
    fit_leaf_qrf,
    metric_dict,
    predict_qrf_frame,
    write_joblib,
)
from r3_utils import (
    R3_DIAG_DIR,
    R3_HITTER_FEATURE_COLS,
    R3_MODELS_DIR,
    R3_RELIEVER_FEATURE_COLS,
    R3_TABLES_DIR,
    interval_coverage,
)


def plot_raw_coverage(frame: pd.DataFrame, target: str, path, title: str) -> None:
    ordered = frame.sort_values("q50").reset_index(drop=True)
    x = np.arange(len(ordered))
    covered = (ordered[target] >= ordered["q10"]) & (ordered[target] <= ordered["q90"])
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.fill_between(x, ordered["q10"], ordered["q90"], color="#8ecae6", alpha=0.35, label="raw 80% QRF interval")
    ax.plot(x, ordered["q50"], color="#264653", lw=1.3, label="median")
    ax.scatter(x[covered], ordered.loc[covered, target], s=13, color="#2a9d8f", label="covered", zorder=3)
    ax.scatter(x[~covered], ordered.loc[~covered, target], s=18, color="#d62828", label="miss", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Player-seasons sorted by QRF median")
    ax.set_ylabel(target)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def hitter_raw_qrf() -> dict:
    hist = pd.read_parquet(DATASETS_DIR / "hitter_features.parquet")
    hist = add_r2_hitter_columns(hist)
    data = hist[
        hist["season"].between(2022, 2025)
        & hist["pa_22g"].ge(50)
        & hist["pa_ros"].ge(50)
        & hist["ros_woba_delta_vs_prior"].notna()
    ].copy()
    train = data[data["season"].isin([2022, 2023])].copy()
    valid = data[data["season"].eq(2024)].copy()
    test = data[data["season"].eq(2025)].copy()
    target = "ros_woba_delta_vs_prior"

    model, imputer = fit_leaf_qrf(train, R3_HITTER_FEATURE_COLS, target, seed_offset=311)
    valid_raw = predict_qrf_frame(model, imputer, valid, R3_HITTER_FEATURE_COLS, margin=0.0)
    test_raw = predict_qrf_frame(model, imputer, test, R3_HITTER_FEATURE_COLS, margin=0.0)
    nonnegative_margin = conformal_interval_margin(
        valid[target].to_numpy(),
        valid_raw["q10"].to_numpy(),
        valid_raw["q90"].to_numpy(),
    )

    valid_out = pd.concat(
        [valid[["season", "batter", "pa_22g", target]].reset_index(drop=True), valid_raw.reset_index(drop=True)],
        axis=1,
    )
    test_out = pd.concat(
        [test[["season", "batter", "pa_22g", target]].reset_index(drop=True), test_raw.reset_index(drop=True)],
        axis=1,
    )
    valid_out.to_csv(R3_TABLES_DIR / "r3_qrf_hitter_raw_coverage_2024.csv", index=False)
    test_out.to_csv(R3_TABLES_DIR / "r3_qrf_hitter_raw_coverage_2025.csv", index=False)
    plot_raw_coverage(
        test_out,
        target,
        R3_DIAG_DIR / "r3_qrf_raw_coverage_hitter_2025.png",
        "R3 Hitter Raw QRF 80% Interval Coverage on 2025 Holdout",
    )
    write_joblib({"model": model, "imputer": imputer, "margin_used": 0.0}, R3_MODELS_DIR / "r3_qrf_hitter_delta_raw_diagnostic.joblib")

    return {
        "target": target,
        "train_n": int(len(train)),
        "valid_n": int(len(valid)),
        "test_n": int(len(test)),
        "raw_coverage_80pct_2024_validation": float(interval_coverage(valid[target], valid_raw["q10"], valid_raw["q90"])),
        "raw_coverage_80pct_2025_test": float(interval_coverage(test[target], test_raw["q10"], test_raw["q90"])),
        "nonnegative_conformal_margin_on_2024": float(nonnegative_margin),
        "median_metrics_2025": metric_dict(test[target], test_raw["q50"]),
    }


def reliever_raw_qrf() -> dict:
    hist = pd.read_parquet(DATASETS_DIR / "pitcher_features.parquet")
    hist = add_pitcher_prior_columns(hist)
    starts_path = DATASETS_DIR / "r2_pitcher_start_counts_2022_2026.parquet"
    if starts_path.exists():
        starts = pd.read_parquet(starts_path)
        hist = hist.merge(starts[starts["season"].between(2022, 2025)], on=["season", "pitcher"], how="left")
        hist["starts_full"] = hist["starts_full"].fillna(0)
    else:
        hist["starts_full"] = 0
    data = hist[
        hist["season"].between(2022, 2025)
        & hist["bf_22g"].ge(25)
        & hist["bf_ros"].ge(25)
        & hist["ip_22g"].lt(30)
        & hist["ip_full"].lt(95)
        & hist["starts_full"].eq(0)
        & hist["ros_k_rate"].notna()
    ].copy()
    train = data[data["season"].isin([2022, 2023])].copy()
    valid = data[data["season"].eq(2024)].copy()
    test = data[data["season"].eq(2025)].copy()
    target = "ros_k_rate"

    model, imputer = fit_leaf_qrf(train, R3_RELIEVER_FEATURE_COLS, target, seed_offset=331)
    valid_raw = predict_qrf_frame(model, imputer, valid, R3_RELIEVER_FEATURE_COLS, margin=0.0)
    test_raw = predict_qrf_frame(model, imputer, test, R3_RELIEVER_FEATURE_COLS, margin=0.0)
    nonnegative_margin = conformal_interval_margin(
        valid[target].to_numpy(),
        valid_raw["q10"].to_numpy(),
        valid_raw["q90"].to_numpy(),
    )

    valid_out = pd.concat(
        [valid[["season", "pitcher", "bf_22g", target]].reset_index(drop=True), valid_raw.reset_index(drop=True)],
        axis=1,
    )
    test_out = pd.concat(
        [test[["season", "pitcher", "bf_22g", target]].reset_index(drop=True), test_raw.reset_index(drop=True)],
        axis=1,
    )
    valid_out.to_csv(R3_TABLES_DIR / "r3_qrf_reliever_raw_coverage_2024.csv", index=False)
    test_out.to_csv(R3_TABLES_DIR / "r3_qrf_reliever_raw_coverage_2025.csv", index=False)
    plot_raw_coverage(
        test_out,
        target,
        R3_DIAG_DIR / "r3_qrf_raw_coverage_reliever_2025.png",
        "R3 Reliever Raw QRF 80% K% Interval Coverage on 2025 Holdout",
    )
    write_joblib({"model": model, "imputer": imputer, "margin_used": 0.0}, R3_MODELS_DIR / "r3_qrf_reliever_k_rate_raw_diagnostic.joblib")

    return {
        "target": target,
        "train_n": int(len(train)),
        "valid_n": int(len(valid)),
        "test_n": int(len(test)),
        "raw_coverage_80pct_2024_validation": float(interval_coverage(valid[target], valid_raw["q10"], valid_raw["q90"])),
        "raw_coverage_80pct_2025_test": float(interval_coverage(test[target], test_raw["q10"], test_raw["q90"])),
        "nonnegative_conformal_margin_on_2024": float(nonnegative_margin),
        "median_metrics_2025": metric_dict(test[target], test_raw["q50"]),
    }


def main() -> dict:
    payload = {
        "calibration_path": "B",
        "decision": "No R3 calibrated interval is claimed. Outputs are raw QRF intervals; the nonnegative conformal margin is reported only to document that the R2 step did not alter widths.",
        "hitter": hitter_raw_qrf(),
        "reliever": reliever_raw_qrf(),
    }
    atomic_write_json(R3_MODELS_DIR / "r3_qrf_raw_coverage.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

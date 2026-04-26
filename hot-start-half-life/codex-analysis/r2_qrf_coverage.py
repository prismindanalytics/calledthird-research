from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATASETS_DIR, SEED, atomic_write_json, set_plot_style
from r2_utils import (
    R2_DIAG_DIR,
    R2_HITTER_FEATURE_COLS,
    R2_MODELS_DIR,
    R2_RELIEVER_FEATURE_COLS,
    R2_TABLES_DIR,
    add_pitcher_prior_columns,
    add_r2_hitter_columns,
    conformal_interval_margin,
    fit_leaf_qrf,
    interval_coverage,
    metric_dict,
    predict_qrf_frame,
    write_joblib,
)


def plot_coverage(test: pd.DataFrame, pred: pd.DataFrame, target: str, path, title: str) -> None:
    frame = pd.concat([test[[target]].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    frame = frame.sort_values("q50").reset_index(drop=True)
    x = np.arange(len(frame))
    covered = (frame[target] >= frame["q10"]) & (frame[target] <= frame["q90"])
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.fill_between(x, frame["q10"], frame["q90"], color="#86bbd8", alpha=0.35, label="80% interval")
    ax.plot(x, frame["q50"], color="#2f4858", lw=1.5, label="median")
    ax.scatter(x[covered], frame.loc[covered, target], s=14, color="#2a9d8f", label="covered", zorder=3)
    ax.scatter(x[~covered], frame.loc[~covered, target], s=18, color="#d62828", label="miss", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("2025 holdout player-seasons sorted by QRF median")
    ax.set_ylabel(target)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def hitter_coverage() -> dict:
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

    model, imputer = fit_leaf_qrf(train, R2_HITTER_FEATURE_COLS, target, seed_offset=11)
    valid_raw = predict_qrf_frame(model, imputer, valid, R2_HITTER_FEATURE_COLS, margin=0.0)
    margin = conformal_interval_margin(valid[target].to_numpy(), valid_raw["q10"].to_numpy(), valid_raw["q90"].to_numpy())
    test_raw = predict_qrf_frame(model, imputer, test, R2_HITTER_FEATURE_COLS, margin=0.0)
    test_cal = predict_qrf_frame(model, imputer, test, R2_HITTER_FEATURE_COLS, margin=margin)
    raw_coverage = interval_coverage(test[target], test_raw["q10"], test_raw["q90"])
    calibrated_coverage = interval_coverage(test[target], test_cal["q10"], test_cal["q90"])
    metrics = metric_dict(test[target], test_cal["q50"])

    out = pd.concat(
        [
            test[["season", "batter", "pa_22g", target]].reset_index(drop=True),
            test_cal.reset_index(drop=True).add_prefix("cal_"),
            test_raw.reset_index(drop=True).add_prefix("raw_"),
        ],
        axis=1,
    )
    out.to_csv(R2_TABLES_DIR / "r2_qrf_hitter_coverage_2025.csv", index=False)
    plot_coverage(test, test_cal, target, R2_DIAG_DIR / "r2_qrf_coverage_hitter.png", "Hitter QRF 80% Interval Coverage on 2025 Holdout")
    write_joblib({"model": model, "imputer": imputer, "margin": margin}, R2_MODELS_DIR / "r2_qrf_hitter_delta_diagnostic.joblib")

    return {
        "target": target,
        "train_n": int(len(train)),
        "valid_n": int(len(valid)),
        "test_n": int(len(test)),
        "raw_coverage_80pct_2025": float(raw_coverage),
        "calibrated_coverage_80pct_2025": float(calibrated_coverage),
        "validation_conformal_margin": float(margin),
        "median_metrics_2025": metrics,
        "warning": "coverage below 70%; interval verdicts must be downgraded" if calibrated_coverage < 0.70 else None,
    }


def reliever_coverage() -> dict:
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
    model, imputer = fit_leaf_qrf(train, R2_RELIEVER_FEATURE_COLS, target, seed_offset=31)
    valid_raw = predict_qrf_frame(model, imputer, valid, R2_RELIEVER_FEATURE_COLS, margin=0.0)
    margin = conformal_interval_margin(valid[target].to_numpy(), valid_raw["q10"].to_numpy(), valid_raw["q90"].to_numpy())
    test_raw = predict_qrf_frame(model, imputer, test, R2_RELIEVER_FEATURE_COLS, margin=0.0)
    test_cal = predict_qrf_frame(model, imputer, test, R2_RELIEVER_FEATURE_COLS, margin=margin)
    raw_coverage = interval_coverage(test[target], test_raw["q10"], test_raw["q90"])
    calibrated_coverage = interval_coverage(test[target], test_cal["q10"], test_cal["q90"])
    out = pd.concat(
        [
            test[["season", "pitcher", "bf_22g", target]].reset_index(drop=True),
            test_cal.reset_index(drop=True).add_prefix("cal_"),
            test_raw.reset_index(drop=True).add_prefix("raw_"),
        ],
        axis=1,
    )
    out.to_csv(R2_TABLES_DIR / "r2_qrf_reliever_coverage_2025.csv", index=False)
    plot_coverage(test, test_cal, target, R2_DIAG_DIR / "r2_qrf_coverage_reliever.png", "Reliever QRF 80% K% Interval Coverage on 2025 Holdout")
    write_joblib({"model": model, "imputer": imputer, "margin": margin}, R2_MODELS_DIR / "r2_qrf_reliever_k_rate_diagnostic.joblib")
    return {
        "target": target,
        "train_n": int(len(train)),
        "valid_n": int(len(valid)),
        "test_n": int(len(test)),
        "raw_coverage_80pct_2025": float(raw_coverage),
        "calibrated_coverage_80pct_2025": float(calibrated_coverage),
        "validation_conformal_margin": float(margin),
        "median_metrics_2025": metric_dict(test[target], test_cal["q50"]),
        "warning": "coverage below 70%; reliever interval verdicts must be downgraded" if calibrated_coverage < 0.70 else None,
    }


def main() -> dict:
    payload = {
        "seed": SEED,
        "hitter": hitter_coverage(),
        "reliever": reliever_coverage(),
    }
    payload["qrf_coverage_80pct"] = payload["hitter"]["calibrated_coverage_80pct_2025"]
    atomic_write_json(R2_MODELS_DIR / "r2_qrf_coverage.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from common import (
    CHARTS_DIR,
    DATASETS_DIR,
    HITTER_FEATURE_COLS,
    MODELS_DIR,
    PITCHER_FEATURE_COLS,
    SEED,
    TABLES_DIR,
    atomic_write_json,
    read_json,
    safe_divide,
    set_plot_style,
)


QUANTILES = [0.10, 0.50, 0.80, 0.90]
HITTER_TARGETS = {
    "woba": "ros_woba",
    "iso": "ros_iso",
    "babip": "ros_babip",
    "ops": "ros_ops",
    "hr_rate": "ros_hr_rate",
    "k_rate": "ros_k_rate",
}
PITCHER_TARGETS = {"ra9": "ros_ra9", "k_rate": "ros_k_rate", "bb_rate": "ros_bb_rate"}


def fit_qrf(df: pd.DataFrame, feature_cols: list[str], target: str):
    train = df[df[target].notna()].copy()
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(train[feature_cols])
    y = train[target].to_numpy()
    model = RandomForestQuantileRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x, y)
    return model, imputer, train


def predict_quantiles(model, imputer, row: pd.DataFrame, feature_cols: list[str]) -> dict:
    pred = np.asarray(model.predict(imputer.transform(row[feature_cols]), quantiles=QUANTILES))[0]
    return {"q10": float(pred[0]), "q50": float(pred[1]), "q80": float(pred[2]), "q90": float(pred[3])}


def qrf_metrics(model, imputer, test: pd.DataFrame, feature_cols: list[str], target: str) -> dict:
    test = test[test[target].notna()].copy()
    if test.empty:
        return {}
    pred = np.asarray(model.predict(imputer.transform(test[feature_cols]), quantiles=[0.5])).reshape(-1)
    return {
        "rmse": float(np.sqrt(mean_squared_error(test[target], pred))),
        "mae": float(mean_absolute_error(test[target], pred)),
        "n": int(len(test)),
    }


def run_hitter_qrf() -> tuple[pd.DataFrame, dict]:
    hitters = pd.read_parquet(DATASETS_DIR / "hitter_features.parquet")
    hitters_2026 = pd.read_parquet(DATASETS_DIR / "hitter_features_2026.parquet")
    named = read_json(DATASETS_DIR / "named_players.json", {})["hitters"]
    train_pool = hitters[
        hitters["season"].between(2022, 2024) & hitters["pa_22g"].ge(50) & hitters["pa_ros"].ge(50)
    ].copy()
    test_pool = hitters[hitters["season"].eq(2025) & hitters["pa_22g"].ge(50) & hitters["pa_ros"].ge(50)].copy()
    rows = []
    metrics = {}
    for stat, target in HITTER_TARGETS.items():
        model, imputer, fitted_train = fit_qrf(train_pool, HITTER_FEATURE_COLS, target)
        joblib.dump(model, MODELS_DIR / f"qrf_hitter_{stat}.joblib")
        joblib.dump(imputer, MODELS_DIR / f"qrf_hitter_{stat}_imputer.joblib")
        metrics[stat] = qrf_metrics(model, imputer, test_pool, HITTER_FEATURE_COLS, target)
        for key, info in named.items():
            mlbam = info.get("mlbam")
            if mlbam is None:
                rows.append({"player_key": key, "player": info["name"], "stat": stat, "status": "missing_id"})
                continue
            row = hitters_2026[hitters_2026["batter"].eq(mlbam)]
            if row.empty:
                rows.append({"player_key": key, "player": info["name"], "mlbam": mlbam, "stat": stat, "status": "missing_statcast"})
                continue
            qs = predict_quantiles(model, imputer, row, HITTER_FEATURE_COLS)
            rec = row.iloc[0]
            rows.append(
                {
                    "player_key": key,
                    "player": info["name"],
                    "mlbam": int(mlbam),
                    "stat": stat,
                    "pa_22g": float(rec.get("pa_22g", np.nan)),
                    "prior_woba": float(rec.get("preseason_prior_woba", np.nan)),
                    "observed_22g": float(rec.get(f"{stat}_22g", np.nan)) if f"{stat}_22g" in row else np.nan,
                    "status": "ok" if rec.get("pa_22g", 0) >= 50 else "below_pa_threshold",
                    **qs,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "qrf_hitter_intervals.csv", index=False)
    return out, metrics


def run_pitcher_qrf() -> tuple[pd.DataFrame, dict]:
    pitchers = pd.read_parquet(DATASETS_DIR / "pitcher_features.parquet")
    pitchers_2026 = pd.read_parquet(DATASETS_DIR / "pitcher_features_2026.parquet")
    named = read_json(DATASETS_DIR / "named_players.json", {})["pitchers"]
    pool = pitchers[
        pitchers["season"].between(2022, 2024)
        & pitchers["bf_22g"].ge(25)
        & pitchers["bf_ros"].ge(25)
        & pitchers["ip_full"].le(95)
    ].copy()
    test_pool = pitchers[
        pitchers["season"].eq(2025) & pitchers["bf_22g"].ge(25) & pitchers["bf_ros"].ge(25) & pitchers["ip_full"].le(95)
    ].copy()
    rows = []
    metrics = {}
    for stat, target in PITCHER_TARGETS.items():
        model, imputer, _ = fit_qrf(pool, PITCHER_FEATURE_COLS, target)
        joblib.dump(model, MODELS_DIR / f"qrf_pitcher_{stat}.joblib")
        joblib.dump(imputer, MODELS_DIR / f"qrf_pitcher_{stat}_imputer.joblib")
        metrics[stat] = qrf_metrics(model, imputer, test_pool, PITCHER_FEATURE_COLS, target)
        for key, info in named.items():
            mlbam = info.get("mlbam")
            row = pitchers_2026[pitchers_2026["pitcher"].eq(mlbam)]
            if row.empty:
                rows.append({"player_key": key, "player": info["name"], "mlbam": mlbam, "stat": stat, "status": "missing_statcast"})
                continue
            qs = predict_quantiles(model, imputer, row, PITCHER_FEATURE_COLS)
            rec = row.iloc[0]
            expected_ip_until_er = None
            sd_ip_until_er = None
            if stat == "ra9" and np.isfinite(qs["q50"]) and qs["q50"] > 0:
                expected_ip_until_er = float(9.0 / qs["q50"])
                sd_ip_until_er = expected_ip_until_er
            rows.append(
                {
                    "player_key": key,
                    "player": info["name"],
                    "mlbam": int(mlbam),
                    "stat": stat,
                    "bf_22g": float(rec.get("bf_22g", np.nan)),
                    "observed_22g": float(rec.get(f"{stat}_22g", np.nan)) if f"{stat}_22g" in row else np.nan,
                    "status": "ok" if rec.get("bf_22g", 0) >= 25 else "below_bf_threshold",
                    "expected_ip_until_er": expected_ip_until_er,
                    "sd_ip_until_er": sd_ip_until_er,
                    **qs,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "qrf_pitcher_intervals.csv", index=False)
    return out, metrics


def plot_hitter_intervals(intervals: pd.DataFrame) -> None:
    woba = intervals[(intervals["stat"].eq("woba")) & (intervals["status"].eq("ok"))].copy()
    if woba.empty:
        return
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(woba))
    ax.hlines(y, woba["q10"], woba["q90"], color="#86bbd8", lw=5, label="10th-90th")
    ax.scatter(woba["q50"], y, color="#2f4858", zorder=3, label="median")
    ax.scatter(woba["prior_woba"], y, color="#f26419", marker="x", zorder=4, label="3-year prior")
    ax.set_yticks(y)
    ax.set_yticklabels(woba["player"])
    ax.set_xlabel("Rest-of-season wOBA")
    ax.set_title("QRF Rest-of-Season wOBA Intervals")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "qrf_hitter_woba_intervals.png")
    plt.close(fig)


def plot_pitcher_intervals(intervals: pd.DataFrame) -> None:
    ra9 = intervals[(intervals["stat"].eq("ra9")) & (intervals["status"].eq("ok"))].copy()
    if ra9.empty:
        return
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hlines([0], ra9["q10"], ra9["q90"], color="#86bbd8", lw=6)
    ax.scatter(ra9["q50"], [0], color="#2f4858", zorder=3)
    ax.set_yticks([0])
    ax.set_yticklabels(ra9["player"])
    ax.set_xlabel("Rest-of-season RA9 proxy")
    ax.set_title("Mason Miller QRF Run-Prevention Interval")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "qrf_mason_miller_ra9_interval.png")
    plt.close(fig)


def main() -> dict:
    hitter_intervals, hitter_metrics = run_hitter_qrf()
    pitcher_intervals, pitcher_metrics = run_pitcher_qrf()
    plot_hitter_intervals(hitter_intervals)
    plot_pitcher_intervals(pitcher_intervals)
    payload = {"hitter_qrf_metrics_2025": hitter_metrics, "pitcher_qrf_metrics_2025": pitcher_metrics}
    atomic_write_json(MODELS_DIR / "qrf_metrics.json", payload)
    return payload


if __name__ == "__main__":
    main()

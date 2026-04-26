from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import (
    CHARTS_DIR,
    DATASETS_DIR,
    DIAG_DIR,
    HITTER_FEATURE_COLS,
    MODELS_DIR,
    SEED,
    TABLES_DIR,
    atomic_write_json,
    read_json,
    set_plot_style,
)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def train_one(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, target: str, feature_cols: list[str]):
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[feature_cols])
    x_valid = imputer.transform(valid[feature_cols])
    x_test = imputer.transform(test[feature_cols])
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1200,
        learning_rate=0.025,
        num_leaves=23,
        min_child_samples=18,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.25,
        random_state=SEED,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(
        x_train,
        train[target].to_numpy(),
        eval_set=[(x_train, train[target].to_numpy()), (x_valid, valid[target].to_numpy())],
        eval_names=["train", "valid"],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    pred_valid = model.predict(x_valid)
    pred_test = model.predict(x_test)
    metrics = {
        "valid": metric_dict(valid[target].to_numpy(), pred_valid),
        "test": metric_dict(test[target].to_numpy(), pred_test),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }
    return model, imputer, metrics


def plot_loss(model: lgb.LGBMRegressor, target: str) -> None:
    evals = model.evals_result_
    if not evals:
        return
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, vals in evals.items():
        metric = next(iter(vals))
        ax.plot(vals[metric], label=split)
    ax.set_title(f"LightGBM Loss Curve: {target}")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DIAG_DIR / f"lgbm_loss_{target}.png")
    plt.close(fig)


def plot_feature_importance(model: lgb.LGBMRegressor, feature_cols: list[str]) -> None:
    gain = model.booster_.feature_importance(importance_type="gain")
    imp = pd.DataFrame({"feature": feature_cols, "gain": gain}).sort_values("gain", ascending=False).head(20)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp["feature"][::-1], imp["gain"][::-1], color="#33658a")
    ax.set_title("LightGBM Gain Importance (ROS wOBA)")
    ax.set_xlabel("Gain")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "lgbm_feature_importance.png")
    plt.close(fig)
    imp.to_csv(TABLES_DIR / "lgbm_gain_importance.csv", index=False)


def score_named(model, imputer, feature_cols: list[str]) -> pd.DataFrame:
    hitters_2026 = pd.read_parquet(DATASETS_DIR / "hitter_features_2026.parquet")
    named = read_json(DATASETS_DIR / "named_players.json", {})["hitters"]
    rows = []
    for key, info in named.items():
        mlbam = info.get("mlbam")
        if mlbam is None:
            rows.append({"player_key": key, "player": info["name"], "status": "missing_id"})
            continue
        row = hitters_2026[hitters_2026["batter"].eq(mlbam)]
        if row.empty:
            rows.append({"player_key": key, "player": info["name"], "mlbam": mlbam, "status": "missing_statcast"})
            continue
        rec = row.iloc[0].to_dict()
        x = imputer.transform(row[feature_cols])
        rows.append(
            {
                "player_key": key,
                "player": info["name"],
                "mlbam": int(mlbam),
                "pa_22g": float(rec.get("pa_22g", np.nan)),
                "prior_woba": float(rec.get("preseason_prior_woba", np.nan)),
                "woba_22g": float(rec.get("woba_22g", np.nan)),
                "pred_ros_woba": float(model.predict(x)[0]),
                "status": "ok" if rec.get("pa_22g", 0) >= 50 else "below_pa_threshold",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES_DIR / "lgbm_2026_predictions.csv", index=False)
    return out


def main() -> dict:
    hitters = pd.read_parquet(DATASETS_DIR / "hitter_features.parquet")
    feature_cols = HITTER_FEATURE_COLS
    data = hitters[
        hitters["season"].between(2022, 2025)
        & hitters["pa_22g"].ge(50)
        & hitters["pa_ros"].ge(50)
        & hitters["ros_woba"].notna()
        & hitters["full_woba"].notna()
    ].copy()
    train = data[data["season"].isin([2022, 2023])]
    valid = data[data["season"].eq(2024)]
    test = data[data["season"].eq(2025)]
    metrics: dict[str, dict] = {}
    fitted: dict[str, tuple] = {}
    for target in ["full_woba", "ros_woba"]:
        model, imputer, target_metrics = train_one(train, valid, test, target, feature_cols)
        metrics[target] = target_metrics
        fitted[target] = (model, imputer)
        joblib.dump(model, MODELS_DIR / f"lgbm_{target}.joblib")
        joblib.dump(imputer, MODELS_DIR / f"lgbm_{target}_imputer.joblib")
        plot_loss(model, target)
    ros_model, ros_imputer = fitted["ros_woba"]
    plot_feature_importance(ros_model, feature_cols)
    score_named(ros_model, ros_imputer, feature_cols)
    payload = {"feature_cols": feature_cols, "metrics": metrics}
    atomic_write_json(MODELS_DIR / "lgbm_metrics.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

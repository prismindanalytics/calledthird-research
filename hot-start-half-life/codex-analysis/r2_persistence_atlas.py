from __future__ import annotations

import json
import warnings

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

from common import DATASETS_DIR, SEED, atomic_write_json, atomic_write_parquet, set_plot_style
from r2_utils import (
    NAMED_HITTER_KEYS,
    R2_CHARTS_DIR,
    R2_DIAG_DIR,
    R2_HITTER_FEATURE_COLS,
    R2_MODELS_DIR,
    R2_TABLES_DIR,
    add_player_names,
    add_r2_hitter_columns,
    conformal_interval_margin,
    fit_leaf_qrf,
    interval_coverage,
    metric_dict,
    predict_qrf_frame,
    write_joblib,
)


TARGET = "ros_woba_delta_vs_prior"
BOOTSTRAP_N = 100
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def lgbm_params(seed: int, n_estimators: int = 700) -> dict:
    return {
        "objective": "regression",
        "n_estimators": n_estimators,
        "learning_rate": 0.035,
        "num_leaves": 15,
        "min_child_samples": 22,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.08,
        "reg_lambda": 0.35,
        "random_state": seed,
        "n_jobs": -1,
        "force_col_wise": True,
        "verbosity": -1,
    }


def load_training_data() -> pd.DataFrame:
    hist = pd.read_parquet(DATASETS_DIR / "hitter_features.parquet")
    hist = add_r2_hitter_columns(hist)
    data = hist[
        hist["season"].between(2022, 2025)
        & hist["pa_22g"].ge(50)
        & hist["pa_ros"].ge(50)
        & hist[TARGET].notna()
    ].copy()
    return data


def train_eval_model(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[R2_HITTER_FEATURE_COLS])
    x_valid = imputer.transform(valid[R2_HITTER_FEATURE_COLS])
    x_test = imputer.transform(test[R2_HITTER_FEATURE_COLS])
    model = lgb.LGBMRegressor(**lgbm_params(SEED))
    model.fit(
        x_train,
        train[TARGET].to_numpy(),
        eval_set=[(x_train, train[TARGET].to_numpy()), (x_valid, valid[TARGET].to_numpy())],
        eval_names=["train", "valid"],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)],
    )
    pred_valid = model.predict(x_valid)
    pred_test = model.predict(x_test)
    metrics = {
        "valid": metric_dict(valid[TARGET], pred_valid),
        "test": metric_dict(test[TARGET], pred_test),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }
    return model, imputer, metrics, x_test


def plot_loss(model) -> None:
    evals = model.evals_result_
    if not evals:
        return
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, vals in evals.items():
        metric = next(iter(vals))
        ax.plot(vals[metric], label=split)
    ax.set_title("R2 LightGBM Loss: ROS wOBA Delta vs Prior")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(R2_DIAG_DIR / "r2_lgbm_loss_hitter_delta.png")
    plt.close(fig)


def compute_permutation(model, x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="neg_mean_squared_error",
        n_repeats=30,
        random_state=SEED,
        n_jobs=-1,
    )
    imp = (
        pd.DataFrame(
            {
                "feature": R2_HITTER_FEATURE_COLS,
                "permutation_importance": result.importances_mean,
                "permutation_std": result.importances_std,
            }
        )
        .sort_values("permutation_importance", ascending=False)
        .reset_index(drop=True)
    )
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp.to_csv(R2_TABLES_DIR / "r2_permutation_importance_hitter.csv", index=False)
    set_plot_style()
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["permutation_importance"], xerr=top["permutation_std"], color="#33658a")
    ax.set_title("R2 Permutation Importance on 2025 Holdout")
    ax.set_xlabel("Increase in MSE")
    fig.tight_layout()
    fig.savefig(R2_CHARTS_DIR / "r2_permutation_importance_top15.png")
    plt.close(fig)
    return imp


def train_bootstrap_ensemble(data: pd.DataFrame, imputer: SimpleImputer, best_iteration: int) -> list:
    rng = np.random.default_rng(SEED)
    x_all = imputer.transform(data[R2_HITTER_FEATURE_COLS])
    y_all = data[TARGET].to_numpy()
    models = []
    n_estimators = max(80, int(best_iteration))
    for i in range(BOOTSTRAP_N):
        sample_idx = rng.integers(0, len(data), len(data))
        model = lgb.LGBMRegressor(**lgbm_params(SEED + 1000 + i, n_estimators=n_estimators))
        model.fit(x_all[sample_idx], y_all[sample_idx])
        models.append(model)
    write_joblib({"models": models, "feature_cols": R2_HITTER_FEATURE_COLS, "n_bootstrap": BOOTSTRAP_N}, R2_MODELS_DIR / "r2_hitter_lgbm_bootstrap_ensemble.joblib")
    return models


def fit_production_qrf(data: pd.DataFrame):
    train = data[data["season"].between(2022, 2024)].copy()
    calibrate = data[data["season"].eq(2025)].copy()
    model, imputer = fit_leaf_qrf(train, R2_HITTER_FEATURE_COLS, TARGET, seed_offset=101)
    raw = predict_qrf_frame(model, imputer, calibrate, R2_HITTER_FEATURE_COLS, margin=0.0)
    margin = conformal_interval_margin(calibrate[TARGET].to_numpy(), raw["q10"].to_numpy(), raw["q90"].to_numpy())
    coverage = interval_coverage(calibrate[TARGET], raw["q10"] - margin, raw["q90"] + margin)
    write_joblib({"model": model, "imputer": imputer, "margin": margin}, R2_MODELS_DIR / "r2_hitter_qrf_delta_production.joblib")
    return model, imputer, margin, coverage


def predict_universe(ensemble: list, lgbm_imputer: SimpleImputer, qrf_model, qrf_imputer, qrf_margin: float) -> pd.DataFrame:
    universe = pd.read_parquet(DATASETS_DIR / "r2_hitter_universe.parquet").copy()
    x = lgbm_imputer.transform(universe[R2_HITTER_FEATURE_COLS])
    preds = np.column_stack([model.predict(x) for model in ensemble])
    universe["pred_delta_mean"] = preds.mean(axis=1)
    universe["pred_delta_bootstrap_sd"] = preds.std(axis=1, ddof=1)
    qrf = predict_qrf_frame(qrf_model, qrf_imputer, universe, R2_HITTER_FEATURE_COLS, margin=qrf_margin)
    universe = pd.concat([universe.reset_index(drop=True), qrf.reset_index(drop=True).add_prefix("delta_")], axis=1)
    universe["pred_ros_woba"] = universe["preseason_prior_woba"] + universe["pred_delta_mean"]
    universe["q10_ros_woba"] = universe["preseason_prior_woba"] + universe["delta_q10"]
    universe["q50_ros_woba"] = universe["preseason_prior_woba"] + universe["delta_q50"]
    universe["q90_ros_woba"] = universe["preseason_prior_woba"] + universe["delta_q90"]
    top_decile = universe["pred_delta_mean"].quantile(0.90)
    bottom_april = universe["woba_cutoff"].quantile(0.10)
    universe["pred_delta_top_decile"] = universe["pred_delta_mean"].ge(top_decile)
    universe["april_woba_bottom_decile"] = universe["woba_cutoff"].le(bottom_april)
    universe["sleeper_candidate"] = universe["pred_delta_top_decile"] & ~universe["in_mainstream_top20"]
    universe["fake_hot_candidate"] = universe["in_mainstream_top20"] & universe["pred_delta_mean"].lt(0)
    universe["fake_cold_candidate"] = universe["april_woba_bottom_decile"] & universe["pred_delta_mean"].gt(0)
    universe["verdict"] = "middle"
    universe.loc[universe["sleeper_candidate"], "verdict"] = "sleeper"
    universe.loc[universe["fake_hot_candidate"], "verdict"] = "fake_hot"
    universe.loc[universe["fake_cold_candidate"], "verdict"] = "fake_cold"
    return universe.sort_values("pred_delta_mean", ascending=False).reset_index(drop=True)


def verdict_from_row(row: pd.Series) -> str:
    if pd.isna(row.get("pred_delta_mean")):
        return "EXCLUDED"
    q10 = row.get("delta_q10", np.nan)
    q90 = row.get("delta_q90", np.nan)
    mean = row.get("pred_delta_mean", np.nan)
    if np.isfinite(q10) and q10 > 0.010:
        return "SIGNAL"
    if np.isfinite(q10) and np.isfinite(q90) and q10 <= 0 <= q90 and abs(mean) < 0.020:
        return "NOISE"
    return "AMBIGUOUS"


def select_lists(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sleepers = pred[pred["sleeper_candidate"]].sort_values("pred_delta_mean", ascending=False).head(10)
    fake_hot = pred[pred["fake_hot_candidate"]].sort_values("pred_delta_mean", ascending=True).head(10)
    fake_cold = pred[pred["fake_cold_candidate"]].sort_values("pred_delta_mean", ascending=False).head(10)
    sanity = pred[pred["batter"].isin(NAMED_HITTER_KEYS.values())].copy()
    sanity["r2_verdict"] = sanity.apply(verdict_from_row, axis=1)
    return sleepers, fake_hot, fake_cold, sanity


def plot_interval_list(df: pd.DataFrame, path, title: str) -> None:
    if df.empty:
        return
    frame = df.sort_values("pred_delta_mean").copy()
    y = np.arange(len(frame))
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, max(3.8, 0.42 * len(frame) + 1.5)))
    ax.hlines(y, frame["delta_q10"], frame["delta_q90"], color="#86bbd8", lw=5, label="QRF 80% interval")
    ax.scatter(frame["pred_delta_mean"], y, color="#2f4858", zorder=3, label="LGBM bootstrap mean")
    ax.axvline(0, color="#d62828", lw=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["player"])
    ax.set_xlabel("Predicted ROS wOBA delta vs preseason prior")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_r1_sanity(sanity: pd.DataFrame) -> None:
    if sanity.empty:
        return
    frame = sanity.sort_values("pred_delta_mean").copy()
    y = np.arange(len(frame))
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hlines(y, frame["q10_ros_woba"], frame["q90_ros_woba"], color="#86bbd8", lw=5, label="QRF 80% ROS wOBA")
    ax.scatter(frame["pred_ros_woba"], y, color="#2f4858", zorder=3, label="prediction")
    ax.scatter(frame["preseason_prior_woba"], y, color="#f26419", marker="x", zorder=4, label="prior")
    ax.scatter(frame["woba_cutoff"], y, color="#6a4c93", marker=".", zorder=4, label="April")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["player"])
    ax.set_xlabel("wOBA")
    ax.set_title("R1 Hitter Sanity Check Under R2 Methodology")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(R2_CHARTS_DIR / "r2_r1_sanity_check_hitters.png")
    plt.close(fig)


def compact_records(df: pd.DataFrame) -> list[dict]:
    cols = [
        "player",
        "batter",
        "team",
        "pa_cutoff",
        "woba_cutoff",
        "preseason_prior_woba",
        "pred_delta_mean",
        "delta_q10",
        "delta_q50",
        "delta_q90",
        "pred_ros_woba",
        "q10_ros_woba",
        "q90_ros_woba",
        "xwoba_minus_woba_22g",
        "in_mainstream_top20",
    ]
    out = []
    for _, row in df.iterrows():
        rec = {}
        for col in cols:
            val = row.get(col)
            if pd.isna(val):
                rec[col] = None
            elif isinstance(val, (np.integer, int)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                rec[col] = float(val)
            elif isinstance(val, (np.bool_, bool)):
                rec[col] = bool(val)
            else:
                rec[col] = val
        out.append(rec)
    return out


def main() -> dict:
    data = load_training_data()
    train = data[data["season"].isin([2022, 2023])].copy()
    valid = data[data["season"].eq(2024)].copy()
    test = data[data["season"].eq(2025)].copy()
    eval_model, eval_imputer, metrics, x_test = train_eval_model(train, valid, test)
    plot_loss(eval_model)
    write_joblib({"model": eval_model, "imputer": eval_imputer, "feature_cols": R2_HITTER_FEATURE_COLS}, R2_MODELS_DIR / "r2_hitter_lgbm_eval.joblib")
    perm = compute_permutation(eval_model, x_test, test[TARGET].to_numpy())

    ensemble = train_bootstrap_ensemble(data, eval_imputer, metrics["best_iteration"])
    qrf_model, qrf_imputer, qrf_margin, qrf_prod_cal_coverage = fit_production_qrf(data)
    pred = predict_universe(ensemble, eval_imputer, qrf_model, qrf_imputer, qrf_margin)
    pred = add_player_names(pred, "batter", "player")
    sleepers, fake_hot, fake_cold, sanity = select_lists(pred)
    pred.to_csv(R2_TABLES_DIR / "r2_persistence_predictions.csv", index=False)
    sleepers.to_csv(R2_TABLES_DIR / "r2_sleepers.csv", index=False)
    fake_hot.to_csv(R2_TABLES_DIR / "r2_fake_hot.csv", index=False)
    fake_cold.to_csv(R2_TABLES_DIR / "r2_fake_cold.csv", index=False)
    sanity.to_csv(R2_TABLES_DIR / "r2_r1_sanity_hitters.csv", index=False)
    atomic_write_parquet(pred, DATASETS_DIR / "r2_persistence_predictions.parquet")

    plot_interval_list(sleepers, R2_CHARTS_DIR / "r2_top10_sleeper_hitters.png", "Top Sleeper Hitter Signals")
    plot_interval_list(fake_hot, R2_CHARTS_DIR / "r2_top10_fake_hot_hitters.png", "Mainstream Hot Starters With Negative ROS Delta")
    plot_interval_list(fake_cold, R2_CHARTS_DIR / "r2_top10_fake_cold_hitters.png", "Fake-Cold / Buy-Low Hitter Signals")
    plot_r1_sanity(sanity)

    xwoba_gap_rank = int(perm.loc[perm["feature"].eq("xwoba_minus_woba_22g"), "rank"].iloc[0]) if (perm["feature"].eq("xwoba_minus_woba_22g")).any() else None
    payload = {
        "target": TARGET,
        "feature_cols": R2_HITTER_FEATURE_COLS,
        "lgbm_metrics": metrics,
        "bootstrap_ensemble_n": BOOTSTRAP_N,
        "qrf_production_calibration_coverage_on_2025": float(qrf_prod_cal_coverage),
        "qrf_production_margin": float(qrf_margin),
        "permutation_top10": perm.head(10).to_dict(orient="records"),
        "xwoba_gap_permutation_rank": xwoba_gap_rank,
        "sleepers": compact_records(sleepers),
        "fake_hot": compact_records(fake_hot),
        "fake_cold": compact_records(fake_cold),
        "r1_sanity_check": [
            {**rec, "r2_verdict": sanity.loc[sanity["batter"].eq(rec["batter"]), "r2_verdict"].iloc[0]}
            for rec in compact_records(sanity)
        ],
        "hypothesis_counts": {
            "sleepers_in_top10": int(len(sleepers)),
            "fake_hot_negative_delta": int(len(fake_hot)),
            "fake_cold_positive_delta": int(len(fake_cold)),
        },
        "shap_status": "dropped from Round 2 after Round 1 SHAP/permutation Spearman failed the pre-committed 0.60 threshold; permutation importance is the sole feature ranking.",
    }
    atomic_write_json(R2_MODELS_DIR / "r2_persistence_atlas.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

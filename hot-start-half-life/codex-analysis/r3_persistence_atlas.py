from __future__ import annotations

import json
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

from common import DATASETS_DIR, SEED, atomic_write_json, atomic_write_parquet, set_plot_style
from r2_utils import (
    add_player_names,
    fit_leaf_qrf,
    metric_dict,
    predict_qrf_frame,
    write_joblib,
)
from r3_utils import (
    NAMED_HITTER_KEYS,
    R3_BOOTSTRAP_N,
    R3_CHARTS_DIR,
    R3_DIAG_DIR,
    R3_HITTER_FEATURE_COLS,
    R3_MIN_HITTER_PRIOR_WOBA,
    R3_MODELS_DIR,
    R3_REPORTED_GAP_FEATURE,
    R3_TABLES_DIR,
    R3_UNREPORTED_GAP_FEATURES,
    add_2026_prior_woba_sd,
    load_hitter_training_data,
    records_for_json,
    validation_prior_woba_sd,
)


TARGET = "ros_woba_delta_vs_prior"
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


def train_eval_model(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[R3_HITTER_FEATURE_COLS])
    x_valid = imputer.transform(valid[R3_HITTER_FEATURE_COLS])
    x_test = imputer.transform(test[R3_HITTER_FEATURE_COLS])
    model = lgb.LGBMRegressor(**lgbm_params(SEED + 300))
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
    ax.set_title("R3 LightGBM Loss: ROS wOBA Delta vs Prior")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(R3_DIAG_DIR / "r3_lgbm_loss_hitter_delta.png")
    plt.close(fig)


def compute_permutation(model, x_test: np.ndarray, y_test: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="neg_mean_squared_error",
        n_repeats=30,
        random_state=SEED + 301,
        n_jobs=-1,
    )
    imp = (
        pd.DataFrame(
            {
                "feature": R3_HITTER_FEATURE_COLS,
                "permutation_importance": result.importances_mean,
                "permutation_std": result.importances_std,
            }
        )
        .sort_values("permutation_importance", ascending=False)
        .reset_index(drop=True)
    )
    imp["full_rank"] = np.arange(1, len(imp) + 1)
    reported = imp[~imp["feature"].isin(R3_UNREPORTED_GAP_FEATURES)].copy().reset_index(drop=True)
    reported["reported_rank"] = np.arange(1, len(reported) + 1)
    imp.to_csv(R3_TABLES_DIR / "r3_permutation_importance_hitter_full.csv", index=False)
    reported.to_csv(R3_TABLES_DIR / "r3_permutation_importance_hitter_reported.csv", index=False)

    set_plot_style()
    top = reported.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["permutation_importance"], xerr=top["permutation_std"], color="#33658a")
    ax.set_title("R3 Reported Permutation Importance on 2025 Holdout")
    ax.set_xlabel("Increase in MSE")
    fig.tight_layout()
    fig.savefig(R3_CHARTS_DIR / "r3_permutation_importance_top15_reported.png")
    plt.close(fig)
    return imp, reported


def train_bootstrap_ensemble(data: pd.DataFrame, imputer: SimpleImputer, best_iteration: int) -> list:
    rng = np.random.default_rng(SEED + 302)
    x_all = imputer.transform(data[R3_HITTER_FEATURE_COLS])
    y_all = data[TARGET].to_numpy()
    models = []
    n_estimators = max(80, int(best_iteration))
    for i in range(R3_BOOTSTRAP_N):
        sample_idx = rng.integers(0, len(data), len(data))
        model = lgb.LGBMRegressor(**lgbm_params(SEED + 3000 + i, n_estimators=n_estimators))
        model.fit(x_all[sample_idx], y_all[sample_idx])
        models.append(model)
    write_joblib(
        {"models": models, "feature_cols": R3_HITTER_FEATURE_COLS, "n_bootstrap": R3_BOOTSTRAP_N},
        R3_MODELS_DIR / "r3_hitter_lgbm_bootstrap_ensemble.joblib",
    )
    return models


def fit_production_qrf(data: pd.DataFrame):
    train = data[data["season"].isin([2022, 2023])].copy()
    test = data[data["season"].eq(2025)].copy()
    model, imputer = fit_leaf_qrf(train, R3_HITTER_FEATURE_COLS, TARGET, seed_offset=301)
    raw = predict_qrf_frame(model, imputer, test, R3_HITTER_FEATURE_COLS, margin=0.0)
    coverage = float(((test[TARGET].to_numpy() >= raw["q10"].to_numpy()) & (test[TARGET].to_numpy() <= raw["q90"].to_numpy())).mean())
    write_joblib({"model": model, "imputer": imputer, "margin_used": 0.0}, R3_MODELS_DIR / "r3_hitter_qrf_delta_raw_production.joblib")
    return model, imputer, coverage


def predict_universe(ensemble: list, lgbm_imputer: SimpleImputer, qrf_model, qrf_imputer, prior_sd_floor: float) -> pd.DataFrame:
    universe = pd.read_parquet(DATASETS_DIR / "r2_hitter_universe.parquet").copy()
    x = lgbm_imputer.transform(universe[R3_HITTER_FEATURE_COLS])
    preds = np.column_stack([model.predict(x) for model in ensemble])
    universe["pred_delta_mean"] = preds.mean(axis=1)
    universe["pred_delta_bootstrap_sd"] = preds.std(axis=1, ddof=1)
    qrf = predict_qrf_frame(qrf_model, qrf_imputer, universe, R3_HITTER_FEATURE_COLS, margin=0.0)
    universe = pd.concat([universe.reset_index(drop=True), qrf.reset_index(drop=True).add_prefix("delta_")], axis=1)
    universe["pred_ros_woba"] = universe["preseason_prior_woba"] + universe["pred_delta_mean"]
    universe["q10_ros_woba"] = universe["preseason_prior_woba"] + universe["delta_q10"]
    universe["q50_ros_woba"] = universe["preseason_prior_woba"] + universe["delta_q50"]
    universe["q90_ros_woba"] = universe["preseason_prior_woba"] + universe["delta_q90"]
    universe = add_2026_prior_woba_sd(universe, prior_sd_floor)
    universe["fake_hot_threshold_woba"] = universe["preseason_prior_woba"] - universe["preseason_prior_woba_sd"]
    universe["fake_hot_margin_woba"] = universe["pred_ros_woba"] - universe["fake_hot_threshold_woba"]
    universe["fake_hot_shortfall_sd"] = (universe["preseason_prior_woba"] - universe["pred_ros_woba"]) / universe["preseason_prior_woba_sd"]

    top_decile = universe["pred_delta_mean"].quantile(0.90)
    bottom_april = universe["woba_cutoff"].quantile(0.10)
    universe["pred_delta_top_decile"] = universe["pred_delta_mean"].ge(top_decile)
    universe["april_woba_bottom_decile"] = universe["woba_cutoff"].le(bottom_april)
    universe["sleeper_candidate"] = (
        universe["pred_delta_top_decile"]
        & ~universe["in_mainstream_top20"].astype(bool)
        & universe["preseason_prior_woba"].gt(R3_MIN_HITTER_PRIOR_WOBA)
    )
    universe["fake_hot_candidate"] = (
        universe["in_mainstream_top20"].astype(bool)
        & universe["preseason_prior_woba"].gt(R3_MIN_HITTER_PRIOR_WOBA)
        & universe["pred_ros_woba"].lt(universe["fake_hot_threshold_woba"])
    )
    universe["fake_cold_candidate"] = universe["april_woba_bottom_decile"] & universe["pred_delta_mean"].gt(0)
    universe["verdict"] = "middle"
    universe.loc[universe["sleeper_candidate"], "verdict"] = "sleeper"
    universe.loc[universe["fake_hot_candidate"], "verdict"] = "fake_hot"
    universe.loc[universe["fake_cold_candidate"], "verdict"] = "fake_cold"
    return universe.sort_values("pred_delta_mean", ascending=False).reset_index(drop=True)


def verdict_from_row(row: pd.Series) -> str:
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
    fake_hot = pred[pred["fake_hot_candidate"]].sort_values("fake_hot_margin_woba", ascending=True).head(10)
    fake_cold = pred[pred["fake_cold_candidate"]].sort_values("pred_delta_mean", ascending=False).head(10)
    sanity = pred[pred["batter"].isin(NAMED_HITTER_KEYS.values())].copy()
    sanity["r3_verdict"] = sanity.apply(verdict_from_row, axis=1)
    return sleepers, fake_hot, fake_cold, sanity


def plot_interval_list(df: pd.DataFrame, path, title: str, x_label: str = "Predicted ROS wOBA delta vs preseason prior") -> None:
    set_plot_style()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.axis("off")
        ax.text(0.5, 0.55, "No players cleared the R3 rule", ha="center", va="center", fontsize=13)
        ax.text(0.5, 0.35, title, ha="center", va="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return
    frame = df.sort_values("pred_delta_mean").copy()
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8, max(3.8, 0.42 * len(frame) + 1.5)))
    ax.hlines(y, frame["delta_q10"], frame["delta_q90"], color="#86bbd8", lw=5, label="raw QRF 80% interval")
    ax.scatter(frame["pred_delta_mean"], y, color="#2f4858", zorder=3, label="LGBM bootstrap mean")
    ax.axvline(0, color="#d62828", lw=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["player"])
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_r3_sanity(sanity: pd.DataFrame) -> None:
    if sanity.empty:
        return
    frame = sanity.sort_values("pred_delta_mean").copy()
    y = np.arange(len(frame))
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hlines(y, frame["q10_ros_woba"], frame["q90_ros_woba"], color="#86bbd8", lw=5, label="raw QRF 80% ROS wOBA")
    ax.scatter(frame["pred_ros_woba"], y, color="#2f4858", zorder=3, label="prediction")
    ax.scatter(frame["preseason_prior_woba"], y, color="#f26419", marker="x", zorder=4, label="prior")
    ax.scatter(frame["woba_cutoff"], y, color="#6a4c93", marker=".", zorder=4, label="April")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["player"])
    ax.set_xlabel("wOBA")
    ax.set_title("Named Hitter Verdict Inputs Under R3 Methodology")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(R3_CHARTS_DIR / "r3_named_hitter_verdict_inputs.png")
    plt.close(fig)


def compact_records(df: pd.DataFrame) -> list[dict]:
    cols = [
        "player",
        "batter",
        "team",
        "pa_cutoff",
        "woba_cutoff",
        "xwoba_22g",
        "preseason_prior_woba",
        "preseason_prior_woba_sd",
        "pred_delta_mean",
        "pred_delta_bootstrap_sd",
        "delta_q10",
        "delta_q50",
        "delta_q90",
        "pred_ros_woba",
        "q10_ros_woba",
        "q90_ros_woba",
        "xwoba_minus_prior_woba_22g",
        "xwoba_minus_woba_22g",
        "in_mainstream_top20",
        "fake_hot_threshold_woba",
        "fake_hot_shortfall_sd",
    ]
    return records_for_json(df, cols)


def main() -> dict:
    data = load_hitter_training_data()
    train = data[data["season"].isin([2022, 2023])].copy()
    valid = data[data["season"].eq(2024)].copy()
    test = data[data["season"].eq(2025)].copy()
    eval_model, eval_imputer, metrics, x_test = train_eval_model(train, valid, test)
    plot_loss(eval_model)
    write_joblib(
        {"model": eval_model, "imputer": eval_imputer, "feature_cols": R3_HITTER_FEATURE_COLS},
        R3_MODELS_DIR / "r3_hitter_lgbm_eval.joblib",
    )
    perm, reported_perm = compute_permutation(eval_model, x_test, test[TARGET].to_numpy())

    ensemble = train_bootstrap_ensemble(train, eval_imputer, metrics["best_iteration"])
    qrf_model, qrf_imputer, qrf_prod_raw_coverage = fit_production_qrf(data)
    prior_sd_floor = validation_prior_woba_sd(data)
    pred = predict_universe(ensemble, eval_imputer, qrf_model, qrf_imputer, prior_sd_floor)
    pred = add_player_names(pred, "batter", "player")
    sleepers, fake_hot, fake_cold, sanity = select_lists(pred)

    pred.to_csv(R3_TABLES_DIR / "r3_persistence_predictions.csv", index=False)
    sleepers.to_csv(R3_TABLES_DIR / "r3_sleepers.csv", index=False)
    fake_hot.to_csv(R3_TABLES_DIR / "r3_fake_hot.csv", index=False)
    fake_cold.to_csv(R3_TABLES_DIR / "r3_fake_cold.csv", index=False)
    sanity.to_csv(R3_TABLES_DIR / "r3_named_hitter_verdict_inputs.csv", index=False)
    atomic_write_parquet(pred, DATASETS_DIR / "r3_persistence_predictions.parquet")

    plot_interval_list(sleepers, R3_CHARTS_DIR / "r3_top10_sleeper_hitters.png", "R3 Top Sleeper Hitter Signals")
    plot_interval_list(fake_hot, R3_CHARTS_DIR / "r3_top10_fake_hot_hitters.png", "R3 Strict Fake-Hot Hitter List")
    plot_interval_list(fake_cold, R3_CHARTS_DIR / "r3_top10_fake_cold_hitters.png", "R3 Fake-Cold / Buy-Low Hitter Signals")
    plot_r3_sanity(sanity)

    chosen_row = perm[perm["feature"].eq(R3_REPORTED_GAP_FEATURE)]
    chosen_report_row = reported_perm[reported_perm["feature"].eq(R3_REPORTED_GAP_FEATURE)]
    payload = {
        "target": TARGET,
        "feature_cols": R3_HITTER_FEATURE_COLS,
        "lgbm_metrics": metrics,
        "bootstrap_ensemble_n": R3_BOOTSTRAP_N,
        "qrf_interval_framing": "raw QRF; no calibrated interval claimed in R3",
        "qrf_production_raw_coverage_on_2025": float(qrf_prod_raw_coverage),
        "qrf_production_margin_used": 0.0,
        "sleeper_rule": "non-mainstream, top-decile predicted ROS wOBA delta, preseason_prior_woba > 0; ranked by predicted delta",
        "fake_hot_rule": "in mainstream top 20 and pred_ros_woba < preseason_prior_woba - preseason_prior_woba_sd",
        "preseason_prior_woba_sd_floor_source": "2024 validation SD of ros_woba_delta_vs_prior",
        "preseason_prior_woba_sd_floor": float(prior_sd_floor),
        "reported_gap_feature": R3_REPORTED_GAP_FEATURE,
        "unreported_gap_features_kept_as_model_controls": sorted(R3_UNREPORTED_GAP_FEATURES),
        "permutation_top10_reported": reported_perm.head(10).to_dict(orient="records"),
        "xwoba_minus_prior_full_rank": int(chosen_row["full_rank"].iloc[0]) if len(chosen_row) else None,
        "xwoba_minus_prior_reported_rank": int(chosen_report_row["reported_rank"].iloc[0]) if len(chosen_report_row) else None,
        "sleepers": compact_records(sleepers),
        "fake_hot": compact_records(fake_hot),
        "fake_cold": compact_records(fake_cold),
        "named_hitter_verdict_inputs": [
            {**rec, "r3_verdict": sanity.loc[sanity["batter"].eq(rec["batter"]), "r3_verdict"].iloc[0]}
            for rec in compact_records(sanity)
        ],
        "hypothesis_counts": {
            "sleepers_in_top10": int(len(sleepers)),
            "strict_fake_hot_count": int(len(fake_hot)),
            "fake_cold_positive_delta": int(len(fake_cold)),
        },
        "fix_notes": {
            "zero_prior_sleepers": "Peters-style zero preseason priors are excluded by preseason_prior_woba > 0.",
            "qrf_calibration": "R3 uses Path B: drop calibration framing and report raw QRF coverage.",
            "xwoba_gap": "Only xwoba_minus_prior_woba_22g is reported in the feature-importance table.",
        },
    }
    atomic_write_json(R3_MODELS_DIR / "r3_persistence_atlas.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

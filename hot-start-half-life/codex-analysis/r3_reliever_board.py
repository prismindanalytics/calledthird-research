from __future__ import annotations

import json
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from common import DATASETS_DIR, SEED, atomic_write_json, atomic_write_parquet, set_plot_style
from r2_utils import (
    add_pitcher_prior_columns,
    fit_leaf_qrf,
    metric_dict,
    predict_qrf_frame,
    write_joblib,
)
from r3_utils import (
    NAMED_PITCHER_KEYS,
    R3_BOOTSTRAP_N,
    R3_CHARTS_DIR,
    R3_DIAG_DIR,
    R3_MIN_RELIEVER_PRIOR_K_RATE,
    R3_MODELS_DIR,
    R3_RELIEVER_FEATURE_COLS,
    R3_TABLES_DIR,
    records_for_json,
)


TARGET = "ros_k_rate"
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def lgbm_params(seed: int, n_estimators: int = 650) -> dict:
    return {
        "objective": "regression",
        "n_estimators": n_estimators,
        "learning_rate": 0.035,
        "num_leaves": 13,
        "min_child_samples": 18,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.05,
        "reg_lambda": 0.30,
        "random_state": seed,
        "n_jobs": -1,
        "force_col_wise": True,
        "verbosity": -1,
    }


def load_training_data() -> pd.DataFrame:
    hist = pd.read_parquet(DATASETS_DIR / "pitcher_features.parquet")
    hist = add_pitcher_prior_columns(hist)
    starts_path = DATASETS_DIR / "r2_pitcher_start_counts_2022_2026.parquet"
    if starts_path.exists():
        starts = pd.read_parquet(starts_path)
        hist = hist.merge(starts[starts["season"].between(2022, 2025)], on=["season", "pitcher"], how="left")
        hist["starts_full"] = hist["starts_full"].fillna(0)
    else:
        hist["starts_full"] = 0
    return hist[
        hist["season"].between(2022, 2025)
        & hist["bf_22g"].ge(25)
        & hist["bf_ros"].ge(25)
        & hist["ip_22g"].lt(30)
        & hist["ip_full"].lt(95)
        & hist["starts_full"].eq(0)
        & hist[TARGET].notna()
    ].copy()


def train_eval_model(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[R3_RELIEVER_FEATURE_COLS])
    x_valid = imputer.transform(valid[R3_RELIEVER_FEATURE_COLS])
    x_test = imputer.transform(test[R3_RELIEVER_FEATURE_COLS])
    model = lgb.LGBMRegressor(**lgbm_params(SEED + 371))
    model.fit(
        x_train,
        train[TARGET].to_numpy(),
        eval_set=[(x_train, train[TARGET].to_numpy()), (x_valid, valid[TARGET].to_numpy())],
        eval_names=["train", "valid"],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    metrics = {
        "valid": metric_dict(valid[TARGET], model.predict(x_valid)),
        "test": metric_dict(test[TARGET], model.predict(x_test)),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }
    return model, imputer, metrics


def plot_loss(model) -> None:
    evals = model.evals_result_
    if not evals:
        return
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, vals in evals.items():
        metric = next(iter(vals))
        ax.plot(vals[metric], label=split)
    ax.set_title("R3 LightGBM Loss: Reliever ROS K%")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(R3_DIAG_DIR / "r3_lgbm_loss_reliever_k_rate.png")
    plt.close(fig)


def train_bootstrap_ensemble(data: pd.DataFrame, imputer: SimpleImputer, best_iteration: int) -> list:
    rng = np.random.default_rng(SEED + 372)
    x_all = imputer.transform(data[R3_RELIEVER_FEATURE_COLS])
    y_all = data[TARGET].to_numpy()
    models = []
    n_estimators = max(80, int(best_iteration))
    for i in range(R3_BOOTSTRAP_N):
        sample_idx = rng.integers(0, len(data), len(data))
        model = lgb.LGBMRegressor(**lgbm_params(SEED + 4000 + i, n_estimators=n_estimators))
        model.fit(x_all[sample_idx], y_all[sample_idx])
        models.append(model)
    write_joblib(
        {"models": models, "feature_cols": R3_RELIEVER_FEATURE_COLS, "n_bootstrap": R3_BOOTSTRAP_N},
        R3_MODELS_DIR / "r3_reliever_lgbm_bootstrap_ensemble.joblib",
    )
    return models


def fit_production_qrf(data: pd.DataFrame):
    train = data[data["season"].isin([2022, 2023])].copy()
    test = data[data["season"].eq(2025)].copy()
    model, imputer = fit_leaf_qrf(train, R3_RELIEVER_FEATURE_COLS, TARGET, seed_offset=371)
    raw = predict_qrf_frame(model, imputer, test, R3_RELIEVER_FEATURE_COLS, margin=0.0)
    coverage = float(((test[TARGET].to_numpy() >= raw["q10"].to_numpy()) & (test[TARGET].to_numpy() <= raw["q90"].to_numpy())).mean())
    write_joblib({"model": model, "imputer": imputer, "margin_used": 0.0}, R3_MODELS_DIR / "r3_reliever_qrf_k_rate_raw_production.joblib")
    return model, imputer, coverage


def predict_universe(ensemble: list, imputer: SimpleImputer, qrf_model, qrf_imputer) -> pd.DataFrame:
    universe = pd.read_parquet(DATASETS_DIR / "r2_reliever_universe.parquet").copy()
    x = imputer.transform(universe[R3_RELIEVER_FEATURE_COLS])
    preds = np.column_stack([model.predict(x) for model in ensemble])
    universe["pred_k_rate_mean"] = preds.mean(axis=1)
    universe["pred_k_rate_bootstrap_sd"] = preds.std(axis=1, ddof=1)
    qrf = predict_qrf_frame(qrf_model, qrf_imputer, universe, R3_RELIEVER_FEATURE_COLS, margin=0.0)
    universe = pd.concat([universe.reset_index(drop=True), qrf.reset_index(drop=True).add_prefix("k_rate_")], axis=1)
    universe["pred_k_delta_vs_prior"] = universe["pred_k_rate_mean"] - universe["preseason_prior_k_rate"]
    universe["april_minus_pred_k_rate"] = universe["k_rate_cutoff"] - universe["pred_k_rate_mean"]
    universe["known_closer_or_named_miller"] = universe["known_2025_closer"] | universe["pitcher"].eq(NAMED_PITCHER_KEYS["mason_miller"])
    high_april_cut = universe["k_rate_cutoff"].quantile(0.75)
    universe["fake_dominant_candidate"] = universe["k_rate_cutoff"].ge(high_april_cut) & universe["april_minus_pred_k_rate"].ge(0.100)
    universe["sleeper_reliever_candidate"] = (
        ~universe["known_closer_or_named_miller"].astype(bool)
        & ~universe["fake_dominant_candidate"].astype(bool)
        & universe["preseason_prior_k_rate"].ge(R3_MIN_RELIEVER_PRIOR_K_RATE)
        & universe["pred_k_delta_vs_prior"].ge(0.025)
        & universe["k_rate_q50"].gt(universe["preseason_prior_k_rate"])
    )
    return universe.sort_values("pred_k_delta_vs_prior", ascending=False).reset_index(drop=True)


def reliever_verdict(row: pd.Series) -> str:
    prior = row.get("preseason_prior_k_rate", np.nan)
    q10 = row.get("k_rate_q10", np.nan)
    q90 = row.get("k_rate_q90", np.nan)
    mean = row.get("pred_k_rate_mean", np.nan)
    if np.isfinite(q10) and np.isfinite(prior) and q10 > prior + 0.020:
        return "SIGNAL"
    if np.isfinite(q10) and np.isfinite(q90) and q10 <= prior <= q90:
        return "AMBIGUOUS"
    if np.isfinite(mean) and np.isfinite(prior) and abs(mean - prior) < 0.015:
        return "NOISE"
    return "AMBIGUOUS"


def plot_reliever_intervals(df: pd.DataFrame, path, title: str) -> None:
    set_plot_style()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.axis("off")
        ax.text(0.5, 0.55, "No relievers cleared the R3 rule", ha="center", va="center", fontsize=13)
        ax.text(0.5, 0.35, title, ha="center", va="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return
    frame = df.sort_values("pred_k_delta_vs_prior")
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.48 * len(frame) + 1.4)))
    ax.hlines(y, frame["k_rate_q10"], frame["k_rate_q90"], color="#86bbd8", lw=5, label="raw QRF 80% interval")
    ax.scatter(frame["pred_k_rate_mean"], y, color="#2f4858", zorder=3, label="LGBM mean")
    ax.scatter(frame["preseason_prior_k_rate"], y, color="#f26419", marker="x", zorder=4, label="prior K%")
    ax.set_yticks(y)
    ax.set_yticklabels(frame["player"])
    ax.set_xlabel("Rest-of-season K%")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def compact_records(df: pd.DataFrame) -> list[dict]:
    cols = [
        "player",
        "pitcher",
        "bf_cutoff",
        "ip_cutoff",
        "k_rate_cutoff",
        "preseason_prior_k_rate",
        "pred_k_rate_mean",
        "pred_k_rate_bootstrap_sd",
        "k_rate_q10",
        "k_rate_q50",
        "k_rate_q90",
        "pred_k_delta_vs_prior",
        "known_2025_closer",
        "known_closer_or_named_miller",
        "sleeper_reliever_candidate",
        "fake_dominant_candidate",
    ]
    return records_for_json(df, cols)


def main() -> dict:
    data = load_training_data()
    train = data[data["season"].isin([2022, 2023])].copy()
    valid = data[data["season"].eq(2024)].copy()
    test = data[data["season"].eq(2025)].copy()
    eval_model, eval_imputer, metrics = train_eval_model(train, valid, test)
    plot_loss(eval_model)
    write_joblib(
        {"model": eval_model, "imputer": eval_imputer, "feature_cols": R3_RELIEVER_FEATURE_COLS},
        R3_MODELS_DIR / "r3_reliever_lgbm_eval.joblib",
    )

    ensemble = train_bootstrap_ensemble(train, eval_imputer, metrics["best_iteration"])
    qrf_model, qrf_imputer, qrf_raw_coverage = fit_production_qrf(data)
    pred = predict_universe(ensemble, eval_imputer, qrf_model, qrf_imputer)
    sleepers = pred[pred["sleeper_reliever_candidate"]].sort_values("pred_k_delta_vs_prior", ascending=False).head(5)
    fake_dom = pred[pred["fake_dominant_candidate"]].sort_values("april_minus_pred_k_rate", ascending=False).head(5)
    sanity = pred[pred["pitcher"].eq(NAMED_PITCHER_KEYS["mason_miller"])].copy()
    if len(sanity):
        sanity["r3_verdict"] = sanity.apply(reliever_verdict, axis=1)

    pred.to_csv(R3_TABLES_DIR / "r3_reliever_board.csv", index=False)
    sleepers.to_csv(R3_TABLES_DIR / "r3_reliever_sleepers.csv", index=False)
    fake_dom.to_csv(R3_TABLES_DIR / "r3_reliever_fake_dominant.csv", index=False)
    sanity.to_csv(R3_TABLES_DIR / "r3_named_reliever_verdict_inputs.csv", index=False)
    atomic_write_parquet(pred, DATASETS_DIR / "r3_reliever_board.parquet")

    plot_reliever_intervals(sleepers, R3_CHARTS_DIR / "r3_top5_sleeper_relievers.png", "R3 Top Sleeper Reliever K% Risers")
    plot_reliever_intervals(fake_dom, R3_CHARTS_DIR / "r3_top5_fake_dominant_relievers.png", "R3 High-April K% Relievers Most Likely To Shrink")
    plot_reliever_intervals(sanity, R3_CHARTS_DIR / "r3_named_reliever_verdict_inputs.png", "Mason Miller R3 K% Verdict Inputs")

    payload = {
        "target": TARGET,
        "lgbm_metrics": metrics,
        "bootstrap_ensemble_n": R3_BOOTSTRAP_N,
        "qrf_interval_framing": "raw QRF; no calibrated interval claimed in R3",
        "qrf_production_raw_coverage_on_2025": float(qrf_raw_coverage),
        "minimum_prior_k_rate_for_sleepers": float(R3_MIN_RELIEVER_PRIOR_K_RATE),
        "fake_dominant_rule": "top-quartile April K% and April K% at least 10 percentage points above predicted ROS K%; excluded from sleeper list membership",
        "reliever_universe_n": int(len(pred)),
        "sleepers": compact_records(sleepers),
        "fake_dominant": compact_records(fake_dom),
        "mason_miller": [
            {**rec, "r3_verdict": sanity.iloc[i].get("r3_verdict") if i < len(sanity) else None}
            for i, rec in enumerate(compact_records(sanity))
        ],
    }
    atomic_write_json(R3_MODELS_DIR / "r3_reliever_board.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

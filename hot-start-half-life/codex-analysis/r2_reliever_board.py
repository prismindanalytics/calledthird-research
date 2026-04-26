from __future__ import annotations

import json
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from common import DATASETS_DIR, SEED, atomic_write_json, set_plot_style
from r2_utils import (
    NAMED_PITCHER_KEYS,
    R2_CHARTS_DIR,
    R2_DIAG_DIR,
    R2_MODELS_DIR,
    R2_RELIEVER_FEATURE_COLS,
    R2_TABLES_DIR,
    add_pitcher_prior_columns,
    conformal_interval_margin,
    fit_leaf_qrf,
    metric_dict,
    predict_qrf_frame,
    write_joblib,
)


TARGET = "ros_k_rate"
BOOTSTRAP_N = 100
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
    data = hist[
        hist["season"].between(2022, 2025)
        & hist["bf_22g"].ge(25)
        & hist["bf_ros"].ge(25)
        & hist["ip_22g"].lt(30)
        & hist["ip_full"].lt(95)
        & hist["starts_full"].eq(0)
        & hist[TARGET].notna()
    ].copy()
    return data


def train_eval_model(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[R2_RELIEVER_FEATURE_COLS])
    x_valid = imputer.transform(valid[R2_RELIEVER_FEATURE_COLS])
    x_test = imputer.transform(test[R2_RELIEVER_FEATURE_COLS])
    model = lgb.LGBMRegressor(**lgbm_params(SEED + 71))
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
    ax.set_title("R2 LightGBM Loss: Reliever ROS K%")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(R2_DIAG_DIR / "r2_lgbm_loss_reliever_k_rate.png")
    plt.close(fig)


def train_bootstrap_ensemble(data: pd.DataFrame, imputer: SimpleImputer, best_iteration: int) -> list:
    rng = np.random.default_rng(SEED + 72)
    x_all = imputer.transform(data[R2_RELIEVER_FEATURE_COLS])
    y_all = data[TARGET].to_numpy()
    models = []
    n_estimators = max(80, int(best_iteration))
    for i in range(BOOTSTRAP_N):
        sample_idx = rng.integers(0, len(data), len(data))
        model = lgb.LGBMRegressor(**lgbm_params(SEED + 2000 + i, n_estimators=n_estimators))
        model.fit(x_all[sample_idx], y_all[sample_idx])
        models.append(model)
    write_joblib({"models": models, "feature_cols": R2_RELIEVER_FEATURE_COLS, "n_bootstrap": BOOTSTRAP_N}, R2_MODELS_DIR / "r2_reliever_lgbm_bootstrap_ensemble.joblib")
    return models


def fit_production_qrf(data: pd.DataFrame):
    train = data[data["season"].between(2022, 2024)].copy()
    calibrate = data[data["season"].eq(2025)].copy()
    model, imputer = fit_leaf_qrf(train, R2_RELIEVER_FEATURE_COLS, TARGET, seed_offset=171)
    raw = predict_qrf_frame(model, imputer, calibrate, R2_RELIEVER_FEATURE_COLS, margin=0.0)
    margin = conformal_interval_margin(calibrate[TARGET].to_numpy(), raw["q10"].to_numpy(), raw["q90"].to_numpy())
    write_joblib({"model": model, "imputer": imputer, "margin": margin}, R2_MODELS_DIR / "r2_reliever_qrf_k_rate_production.joblib")
    return model, imputer, margin


def predict_universe(ensemble: list, imputer: SimpleImputer, qrf_model, qrf_imputer, qrf_margin: float) -> pd.DataFrame:
    universe = pd.read_parquet(DATASETS_DIR / "r2_reliever_universe.parquet").copy()
    x = imputer.transform(universe[R2_RELIEVER_FEATURE_COLS])
    preds = np.column_stack([model.predict(x) for model in ensemble])
    universe["pred_k_rate_mean"] = preds.mean(axis=1)
    universe["pred_k_rate_bootstrap_sd"] = preds.std(axis=1, ddof=1)
    qrf = predict_qrf_frame(qrf_model, qrf_imputer, universe, R2_RELIEVER_FEATURE_COLS, margin=qrf_margin)
    universe = pd.concat([universe.reset_index(drop=True), qrf.reset_index(drop=True).add_prefix("k_rate_")], axis=1)
    universe["pred_k_delta_vs_prior"] = universe["pred_k_rate_mean"] - universe["preseason_prior_k_rate"]
    universe["april_minus_pred_k_rate"] = universe["k_rate_cutoff"] - universe["pred_k_rate_mean"]
    universe["known_closer_or_named_miller"] = universe["known_2025_closer"] | universe["pitcher"].eq(NAMED_PITCHER_KEYS["mason_miller"])
    universe["sleeper_reliever_candidate"] = (
        ~universe["known_closer_or_named_miller"]
        & universe["pred_k_delta_vs_prior"].ge(0.025)
        & universe["k_rate_q50"].gt(universe["preseason_prior_k_rate"])
    )
    high_april_cut = universe["k_rate_cutoff"].quantile(0.75)
    universe["fake_dominant_candidate"] = universe["k_rate_cutoff"].ge(high_april_cut) & universe["april_minus_pred_k_rate"].ge(0.075)
    return universe.sort_values("pred_k_delta_vs_prior", ascending=False).reset_index(drop=True)


def plot_reliever_intervals(df: pd.DataFrame, path, title: str) -> None:
    if df.empty:
        return
    frame = df.sort_values("pred_k_delta_vs_prior")
    y = np.arange(len(frame))
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.48 * len(frame) + 1.4)))
    ax.hlines(y, frame["k_rate_q10"], frame["k_rate_q90"], color="#86bbd8", lw=5, label="QRF 80% interval")
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


def compact_records(df: pd.DataFrame) -> list[dict]:
    cols = [
        "player",
        "pitcher",
        "bf_cutoff",
        "ip_cutoff",
        "k_rate_cutoff",
        "preseason_prior_k_rate",
        "pred_k_rate_mean",
        "k_rate_q10",
        "k_rate_q50",
        "k_rate_q90",
        "pred_k_delta_vs_prior",
        "known_2025_closer",
        "known_closer_or_named_miller",
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
    eval_model, eval_imputer, metrics = train_eval_model(train, valid, test)
    plot_loss(eval_model)
    write_joblib({"model": eval_model, "imputer": eval_imputer, "feature_cols": R2_RELIEVER_FEATURE_COLS}, R2_MODELS_DIR / "r2_reliever_lgbm_eval.joblib")

    ensemble = train_bootstrap_ensemble(data, eval_imputer, metrics["best_iteration"])
    qrf_model, qrf_imputer, qrf_margin = fit_production_qrf(data)
    pred = predict_universe(ensemble, eval_imputer, qrf_model, qrf_imputer, qrf_margin)
    sleepers = pred[pred["sleeper_reliever_candidate"]].sort_values("pred_k_delta_vs_prior", ascending=False).head(5)
    fake_dom = pred[pred["fake_dominant_candidate"]].sort_values("april_minus_pred_k_rate", ascending=False).head(5)
    sanity = pred[pred["pitcher"].eq(NAMED_PITCHER_KEYS["mason_miller"])].copy()
    if len(sanity):
        sanity["r2_verdict"] = sanity.apply(reliever_verdict, axis=1)

    pred.to_csv(R2_TABLES_DIR / "r2_reliever_board.csv", index=False)
    sleepers.to_csv(R2_TABLES_DIR / "r2_reliever_sleepers.csv", index=False)
    fake_dom.to_csv(R2_TABLES_DIR / "r2_reliever_fake_dominant.csv", index=False)
    sanity.to_csv(R2_TABLES_DIR / "r2_r1_sanity_reliever.csv", index=False)
    atomic_write_json(R2_MODELS_DIR / "r2_reliever_board_table_manifest.json", {"rows": int(len(pred))})

    plot_reliever_intervals(sleepers, R2_CHARTS_DIR / "r2_top5_sleeper_relievers.png", "Top Sleeper Reliever K% Risers")
    plot_reliever_intervals(fake_dom, R2_CHARTS_DIR / "r2_top5_fake_dominant_relievers.png", "High-April K% Relievers Most Likely To Shrink")
    plot_reliever_intervals(sanity, R2_CHARTS_DIR / "r2_r1_sanity_check_reliever.png", "Mason Miller R2 K% Sanity Check")

    payload = {
        "target": TARGET,
        "lgbm_metrics": metrics,
        "bootstrap_ensemble_n": BOOTSTRAP_N,
        "qrf_production_margin": float(qrf_margin),
        "reliever_universe_n": int(len(pred)),
        "sleepers": compact_records(sleepers),
        "fake_dominant": compact_records(fake_dom),
        "r1_sanity_check": [
            {**rec, "r2_verdict": sanity.iloc[i].get("r2_verdict") if i < len(sanity) else None}
            for i, rec in enumerate(compact_records(sanity))
        ],
    }
    atomic_write_json(R2_MODELS_DIR / "r2_reliever_board.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

GLOBAL_SEED = 20260505
LINEUP_SPOTS = list(range(1, 10))


@dataclass
class ModelSpec:
    name: str
    numeric_features: list[str]
    categorical_features: list[str]
    target: str
    group: str = "game_pk"
    include_lineup: bool = True
    target_encode_col: str = "umpire"
    min_category_count: int = 25
    n_estimators: int = 700
    learning_rate: float = 0.035
    num_leaves: int = 31
    min_child_samples: int = 60


@dataclass
class EncodedTransformer:
    spec: ModelSpec
    medians: dict[str, float]
    categories: dict[str, list[str]]
    umpire_map: dict[str, float]
    global_target_mean: float
    feature_names: list[str]


@dataclass
class ModelResult:
    name: str
    model: lgb.LGBMClassifier
    transformer: EncodedTransformer
    metrics: dict[str, Any]
    oof_pred: np.ndarray
    oof_index: np.ndarray
    feature_names: list[str]
    shap_values: np.ndarray | None = None
    shap_matrix: pd.DataFrame | None = None
    shap_summary: pd.DataFrame | None = None


def lgbm_params(spec: ModelSpec, random_state: int, n_estimators: int | None = None) -> dict[str, Any]:
    return {
        "objective": "binary",
        "n_estimators": int(n_estimators or spec.n_estimators),
        "learning_rate": spec.learning_rate,
        "num_leaves": spec.num_leaves,
        "min_child_samples": spec.min_child_samples,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.3,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }


def _clean_category(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown", "<NA>": "unknown"})


def build_global_categories(df: pd.DataFrame, spec: ModelSpec) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for col in spec.categorical_features:
        cleaned = _clean_category(df[col] if col in df.columns else pd.Series(["unknown"] * len(df), index=df.index))
        counts = cleaned.value_counts()
        keep = counts[counts >= spec.min_category_count].index.tolist()
        if "unknown" not in keep:
            keep.append("unknown")
        if "OTHER" not in keep:
            keep.append("OTHER")
        categories[col] = sorted(set(map(str, keep)))
    return categories


def make_transformer(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    spec: ModelSpec,
    categories: dict[str, list[str]],
) -> EncodedTransformer:
    medians = {
        col: float(pd.to_numeric(train_df[col], errors="coerce").median())
        for col in spec.numeric_features
    }
    global_mean = float(np.mean(y_train))
    if spec.target_encode_col in train_df.columns:
        enc_frame = pd.DataFrame(
            {
                "key": _clean_category(train_df[spec.target_encode_col]).to_numpy(),
                "target": y_train,
            }
        )
        stats = enc_frame.groupby("key")["target"].agg(["sum", "count"])
        smooth = 20.0
        umpire_map = ((stats["sum"] + smooth * global_mean) / (stats["count"] + smooth)).to_dict()
    else:
        umpire_map = {}
    transformer = EncodedTransformer(
        spec=spec,
        medians=medians,
        categories=categories,
        umpire_map={str(k): float(v) for k, v in umpire_map.items()},
        global_target_mean=global_mean,
        feature_names=[],
    )
    encoded = encode_frame(train_df, transformer)
    transformer.feature_names = list(encoded.columns)
    return transformer


def encode_frame(df: pd.DataFrame, transformer: EncodedTransformer) -> pd.DataFrame:
    spec = transformer.spec
    pieces: list[pd.DataFrame] = []

    numeric = pd.DataFrame(index=df.index)
    for col in spec.numeric_features:
        numeric[col] = pd.to_numeric(df[col], errors="coerce").fillna(transformer.medians.get(col, 0.0))
    pieces.append(numeric.reset_index(drop=True))

    if spec.include_lineup:
        lineup = pd.to_numeric(df["lineup_spot"], errors="coerce").fillna(0).astype(int)
        lineup_frame = pd.DataFrame(index=df.index)
        for spot in LINEUP_SPOTS:
            lineup_frame[f"lineup_spot_{spot}"] = (lineup == spot).astype(float)
        pieces.append(lineup_frame.reset_index(drop=True))

    for col in spec.categorical_features:
        values = _clean_category(df[col] if col in df.columns else pd.Series(["unknown"] * len(df), index=df.index))
        cats = transformer.categories.get(col, ["unknown", "OTHER"])
        values = values.where(values.isin(cats), "OTHER")
        cat = pd.Categorical(values, categories=cats)
        pieces.append(pd.get_dummies(cat, prefix=col, dtype=float).reset_index(drop=True))

    if spec.target_encode_col:
        keys = _clean_category(df[spec.target_encode_col] if spec.target_encode_col in df.columns else pd.Series(["unknown"] * len(df), index=df.index))
        te = keys.map(transformer.umpire_map).fillna(transformer.global_target_mean).astype(float)
        pieces.append(pd.DataFrame({f"{spec.target_encode_col}_target": te.to_numpy()}))

    encoded = pd.concat(pieces, axis=1)
    if transformer.feature_names:
        for col in transformer.feature_names:
            if col not in encoded.columns:
                encoded[col] = 0.0
        encoded = encoded[transformer.feature_names]
    return encoded.astype(float)


def _lineup_group_permutation_importance(
    model: lgb.LGBMClassifier,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    rng: np.random.Generator,
    n_repeats: int = 10,
) -> dict[str, float]:
    pred = model.predict_proba(x_val)[:, 1]
    baseline_auc = roc_auc_score(y_val, pred) if len(np.unique(y_val)) > 1 else np.nan
    baseline_loss = log_loss(y_val, pred, labels=[0, 1])
    lineup_cols = [col for col in x_val.columns if col.startswith("lineup_spot_")]
    auc_drops: list[float] = []
    loss_increases: list[float] = []
    null_auc_drops: list[float] = []
    for _ in range(n_repeats):
        permuted = x_val.copy()
        order = rng.permutation(len(permuted))
        permuted.loc[:, lineup_cols] = permuted[lineup_cols].to_numpy()[order]
        perm_pred = model.predict_proba(permuted)[:, 1]
        if len(np.unique(y_val)) > 1:
            auc_drops.append(float(baseline_auc - roc_auc_score(y_val, perm_pred)))
        loss_increases.append(float(log_loss(y_val, perm_pred, labels=[0, 1]) - baseline_loss))

        shuffled_y = rng.permutation(y_val)
        if len(np.unique(shuffled_y)) > 1:
            base_null = roc_auc_score(shuffled_y, pred)
            perm_null = roc_auc_score(shuffled_y, perm_pred)
            null_auc_drops.append(float(base_null - perm_null))

    return {
        "lineup_auc_drop_mean": float(np.nanmean(auc_drops)) if auc_drops else float("nan"),
        "lineup_auc_drop_std": float(np.nanstd(auc_drops)) if auc_drops else float("nan"),
        "lineup_logloss_increase_mean": float(np.mean(loss_increases)),
        "permuted_label_auc_drop_mean": float(np.nanmean(null_auc_drops)) if null_auc_drops else float("nan"),
        "permuted_label_auc_drop_p95": float(np.nanpercentile(null_auc_drops, 95)) if null_auc_drops else float("nan"),
    }


def _plot_roc(y_true: np.ndarray, pred: np.ndarray, auc: float, path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, pred)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(fpr, tpr, linewidth=2.0, label=f"AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_calibration(y_true: np.ndarray, pred: np.ndarray, path: Path, title: str) -> dict[str, list[float]]:
    prob_true, prob_pred = calibration_curve(y_true, pred, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed rate")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return {"prob_pred": [float(x) for x in prob_pred], "prob_true": [float(x) for x in prob_true]}


def _plot_learning_curves(curves: list[dict[str, list[float]]], path: Path, title: str) -> None:
    if not curves:
        return
    max_len = max(len(curve["training"]) for curve in curves)
    train = []
    val = []
    for curve in curves:
        tr = curve["training"] + [curve["training"][-1]] * (max_len - len(curve["training"]))
        va = curve["validation"] + [curve["validation"][-1]] * (max_len - len(curve["validation"]))
        train.append(tr)
        val.append(va)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    rounds = np.arange(1, max_len + 1)
    ax.plot(rounds, np.mean(train, axis=0), label="Train log loss", linewidth=2)
    ax.plot(rounds, np.mean(val, axis=0), label="Validation log loss", linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Binary log loss")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fit_lgbm_classifier(
    df: pd.DataFrame,
    spec: ModelSpec,
    output_dir: Path,
    compute_shap: bool = True,
    make_plots: bool = True,
    shap_sample_size: int = 4000,
    n_splits: int = 5,
) -> ModelResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = df.dropna(subset=[spec.target, spec.group]).copy().reset_index(drop=True)
    y = clean[spec.target].astype(int).to_numpy()
    groups = clean[spec.group].astype(str).to_numpy()
    categories = build_global_categories(clean, spec)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)
    oof_pred = np.zeros(len(clean), dtype=float)
    oof_index = np.arange(len(clean))
    fold_metrics: list[dict[str, float]] = []
    best_iterations: list[int] = []
    permutation_payloads: list[dict[str, float]] = []
    curves: list[dict[str, list[float]]] = []
    rng = np.random.default_rng(GLOBAL_SEED)

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(clean, y, groups)):
        train_df = clean.iloc[train_idx].copy()
        val_df = clean.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]
        transformer = make_transformer(train_df, y_train, spec, categories)
        x_train = encode_frame(train_df, transformer)
        x_val = encode_frame(val_df, transformer)
        model = lgb.LGBMClassifier(**lgbm_params(spec, GLOBAL_SEED + fold_idx))
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = model.predict_proba(x_val)[:, 1]
        oof_pred[val_idx] = pred
        fold_auc = roc_auc_score(y_val, pred) if len(np.unique(y_val)) > 1 else np.nan
        fold_loss = log_loss(y_val, pred, labels=[0, 1])
        fold_metrics.append({"auc": float(fold_auc), "logloss": float(fold_loss)})
        best_iterations.append(int(model.best_iteration_ or spec.n_estimators))
        permutation_payloads.append(_lineup_group_permutation_importance(model, x_val, y_val, rng))
        curves.append(
            {
                "training": model.evals_result_["training"]["binary_logloss"],
                "validation": model.evals_result_["valid_1"]["binary_logloss"],
            }
        )

    auc = float(roc_auc_score(y, oof_pred)) if len(np.unique(y)) > 1 else float("nan")
    loss = float(log_loss(y, oof_pred, labels=[0, 1]))
    calibration = None
    if make_plots:
        _plot_roc(y, oof_pred, auc, output_dir / f"{spec.name}_roc.png", f"{spec.name.replace('_', ' ').title()} ROC")
        calibration = _plot_calibration(
            y,
            oof_pred,
            output_dir / f"{spec.name}_calibration.png",
            f"{spec.name.replace('_', ' ').title()} Calibration",
        )
        _plot_learning_curves(
            curves,
            output_dir / f"{spec.name}_learning_curve.png",
            f"{spec.name.replace('_', ' ').title()} Learning Curve",
        )

    final_estimators = max(50, int(np.nanmedian(best_iterations)))
    final_transformer = make_transformer(clean, y, spec, categories)
    x_full = encode_frame(clean, final_transformer)
    final_model = lgb.LGBMClassifier(**lgbm_params(spec, GLOBAL_SEED + 333, n_estimators=final_estimators))
    final_model.fit(x_full, y)

    shap_values = None
    shap_matrix = None
    shap_summary = None
    if compute_shap:
        sample = clean.sample(min(shap_sample_size, len(clean)), random_state=GLOBAL_SEED).copy()
        shap_matrix = encode_frame(sample, final_transformer)
        explainer = shap.TreeExplainer(final_model)
        values = explainer.shap_values(shap_matrix)
        if isinstance(values, list):
            values = values[-1]
        shap_values = np.asarray(values)
        shap_summary = (
            pd.DataFrame(
                {
                    "feature": shap_matrix.columns,
                    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                    "mean_shap": shap_values.mean(axis=0),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_summary.to_csv(output_dir / f"{spec.name}_shap_importance.csv", index=False)

    perm_df = pd.DataFrame(permutation_payloads)
    metrics = {
        "auc": auc,
        "logloss": loss,
        "folds": fold_metrics,
        "best_iterations": best_iterations,
        "n_rows": int(len(clean)),
        "n_games": int(pd.Series(groups).nunique()),
        "positive_rate": float(y.mean()),
        "calibration": calibration,
        "lineup_permutation_importance": {
            key: float(perm_df[key].mean()) for key in perm_df.columns
        },
    }

    return ModelResult(
        name=spec.name,
        model=final_model,
        transformer=final_transformer,
        metrics=metrics,
        oof_pred=oof_pred,
        oof_index=oof_index,
        feature_names=list(x_full.columns),
        shap_values=shap_values,
        shap_matrix=shap_matrix,
        shap_summary=shap_summary,
    )


def predict_with_model(result: ModelResult, df: pd.DataFrame) -> np.ndarray:
    x = encode_frame(df, result.transformer)
    return result.model.predict_proba(x)[:, 1]


def lineup_shap_effect(result: ModelResult, spot_a: int = 7, spot_b: int = 3) -> dict[str, float]:
    if result.shap_values is None or result.shap_matrix is None:
        return {"spot_7_mean_shap": float("nan"), "spot_3_mean_shap": float("nan"), "delta": float("nan")}
    payload: dict[str, float] = {}
    for spot in [spot_a, spot_b]:
        col = f"lineup_spot_{spot}"
        if col not in result.shap_matrix.columns:
            payload[f"spot_{spot}_mean_shap"] = float("nan")
            continue
        idx = result.shap_matrix.columns.get_loc(col)
        mask = result.shap_matrix[col].to_numpy() == 1
        payload[f"spot_{spot}_mean_shap"] = float(result.shap_values[mask, idx].mean()) if mask.any() else float("nan")
        payload[f"spot_{spot}_mean_abs_shap"] = float(np.abs(result.shap_values[:, idx]).mean())
    payload["delta"] = payload.get(f"spot_{spot_a}_mean_shap", np.nan) - payload.get(f"spot_{spot_b}_mean_shap", np.nan)
    return payload


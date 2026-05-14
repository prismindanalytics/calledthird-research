from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

GLOBAL_SEED = 20260506
LINEUP_SPOTS = list(range(1, 10))
warnings.simplefilter("ignore", PerformanceWarning)


@dataclass
class ModelSpec:
    name: str
    target: str
    numeric_features: list[str]
    categorical_features: list[str]
    group: str = "game_pk"
    include_lineup: bool = True
    interaction_specs: list[str] = field(default_factory=list)
    permutation_groups: list[str] = field(default_factory=list)
    min_category_count: int = 25
    n_estimators: int = 700
    learning_rate: float = 0.035
    num_leaves: int = 31
    min_child_samples: int = 70


@dataclass
class EncodedTransformer:
    spec: ModelSpec
    medians: dict[str, float]
    categories: dict[str, list[str]]
    feature_names: list[str]


@dataclass
class ModelResult:
    name: str
    model: lgb.LGBMClassifier
    transformer: EncodedTransformer
    metrics: dict[str, Any]
    oof_pred: np.ndarray
    training_frame: pd.DataFrame
    feature_names: list[str]


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


def clean_category(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown", "<NA>": "unknown"})


def safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return token or "unknown"


def build_global_categories(df: pd.DataFrame, spec: ModelSpec) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    cat_cols = set(spec.categorical_features)
    if "umpire_lineup" in spec.interaction_specs:
        cat_cols.add("umpire")
    if "lineup_chase_tertile" in spec.interaction_specs:
        cat_cols.add("chase_tertile")
    for col in sorted(cat_cols):
        values = clean_category(df[col] if col in df.columns else pd.Series(["unknown"] * len(df), index=df.index))
        counts = values.value_counts()
        keep = counts[counts >= spec.min_category_count].index.astype(str).tolist()
        if col == "chase_tertile":
            keep = [x for x in ["low", "mid", "high"] if x in set(values)]
        if "unknown" not in keep:
            keep.append("unknown")
        if "OTHER" not in keep:
            keep.append("OTHER")
        categories[col] = sorted(set(keep))
    return categories


def make_transformer(train_df: pd.DataFrame, spec: ModelSpec, categories: dict[str, list[str]]) -> EncodedTransformer:
    medians: dict[str, float] = {}
    for col in spec.numeric_features:
        if col in train_df.columns:
            val = pd.to_numeric(train_df[col], errors="coerce").median()
            medians[col] = 0.0 if pd.isna(val) else float(val)
        else:
            medians[col] = 0.0
    transformer = EncodedTransformer(spec=spec, medians=medians, categories=categories, feature_names=[])
    encoded = encode_frame(train_df, transformer)
    transformer.feature_names = list(encoded.columns)
    return transformer


def _lineup_frame(df: pd.DataFrame) -> pd.DataFrame:
    lineup = pd.to_numeric(df["lineup_spot"], errors="coerce").fillna(0).astype(int)
    out = pd.DataFrame(index=df.index)
    for spot in LINEUP_SPOTS:
        out[f"lineup_spot_{spot}"] = (lineup == spot).astype(np.float32)
    return out


def _category_dummies(df: pd.DataFrame, col: str, categories: list[str]) -> pd.DataFrame:
    values = clean_category(df[col] if col in df.columns else pd.Series(["unknown"] * len(df), index=df.index))
    values = values.where(values.isin(categories), "OTHER")
    cat = pd.Categorical(values, categories=categories)
    return pd.get_dummies(cat, prefix=col, dtype=np.float32)


def _interaction_umpire_lineup(df: pd.DataFrame, transformer: EncodedTransformer) -> pd.DataFrame:
    cats = transformer.categories.get("umpire", ["unknown", "OTHER"])
    umpire = clean_category(df["umpire"] if "umpire" in df.columns else pd.Series(["unknown"] * len(df), index=df.index))
    umpire = umpire.where(umpire.isin(cats), "OTHER")
    lineup = pd.to_numeric(df["lineup_spot"], errors="coerce").fillna(0).astype(int)
    data: dict[str, np.ndarray] = {}
    for ump in cats:
        token = safe_token(ump)
        ump_mask = umpire.eq(ump).to_numpy()
        for spot in LINEUP_SPOTS:
            data[f"inter_umpire_lineup__{token}__spot_{spot}"] = (ump_mask & lineup.eq(spot).to_numpy()).astype(np.float32)
    return pd.DataFrame(data, index=df.index)


def _interaction_lineup_chase(df: pd.DataFrame, transformer: EncodedTransformer) -> pd.DataFrame:
    cats = [c for c in transformer.categories.get("chase_tertile", ["low", "mid", "high"]) if c not in {"unknown", "OTHER"}]
    chase = clean_category(df["chase_tertile"] if "chase_tertile" in df.columns else pd.Series(["unknown"] * len(df), index=df.index))
    chase = chase.where(chase.isin(cats), "unknown")
    lineup = pd.to_numeric(df["lineup_spot"], errors="coerce").fillna(0).astype(int)
    data: dict[str, np.ndarray] = {}
    for tertile in cats:
        chase_mask = chase.eq(tertile).to_numpy()
        for spot in LINEUP_SPOTS:
            data[f"inter_lineup_chase__spot_{spot}__{tertile}"] = (chase_mask & lineup.eq(spot).to_numpy()).astype(np.float32)
    return pd.DataFrame(data, index=df.index)


def encode_frame(df: pd.DataFrame, transformer: EncodedTransformer) -> pd.DataFrame:
    spec = transformer.spec
    pieces: list[pd.DataFrame] = []
    numeric = pd.DataFrame(index=df.index)
    for col in spec.numeric_features:
        numeric[col] = pd.to_numeric(df[col], errors="coerce").fillna(transformer.medians.get(col, 0.0)).astype(np.float32)
    pieces.append(numeric)

    if spec.include_lineup:
        pieces.append(_lineup_frame(df))

    for col in spec.categorical_features:
        pieces.append(_category_dummies(df, col, transformer.categories.get(col, ["unknown", "OTHER"])))

    if "umpire_lineup" in spec.interaction_specs:
        pieces.append(_interaction_umpire_lineup(df, transformer))
    if "lineup_chase_tertile" in spec.interaction_specs:
        pieces.append(_interaction_lineup_chase(df, transformer))

    encoded = pd.concat([p.reset_index(drop=True) for p in pieces], axis=1)
    if transformer.feature_names:
        missing_cols = [col for col in transformer.feature_names if col not in encoded.columns]
        for col in missing_cols:
            encoded[col] = np.float32(0.0)
        encoded = encoded[transformer.feature_names]
    return encoded.astype(np.float32)


def feature_group_columns(columns: list[str], group_name: str) -> list[str]:
    if group_name == "lineup":
        return [c for c in columns if c.startswith("lineup_spot_")]
    if group_name == "umpire_lineup_interaction":
        return [c for c in columns if c.startswith("inter_umpire_lineup__")]
    if group_name == "lineup_chase_interaction":
        return [c for c in columns if c.startswith("inter_lineup_chase__")]
    if group_name == "umpire_main":
        return [c for c in columns if c.startswith("umpire_")]
    if group_name == "pitcher_main":
        return [c for c in columns if c.startswith("pitcher_id_") or c.startswith("pitcher_fame_quartile_")]
    if group_name == "catcher_main":
        return [c for c in columns if c.startswith("catcher_id_") or c.startswith("catcher_framing_tier_")]
    return [c for c in columns if c.startswith(f"{group_name}_")]


def permutation_importance_groups(
    model: lgb.LGBMClassifier,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    groups: list[str],
    rng: np.random.Generator,
    n_repeats: int = 8,
) -> dict[str, dict[str, float]]:
    pred = model.predict_proba(x_val)[:, 1]
    has_both = len(np.unique(y_val)) > 1
    baseline_auc = roc_auc_score(y_val, pred) if has_both else np.nan
    baseline_loss = log_loss(y_val, pred, labels=[0, 1])
    out: dict[str, dict[str, float]] = {}
    for group_name in groups:
        cols = feature_group_columns(list(x_val.columns), group_name)
        if not cols:
            out[group_name] = {
                "n_columns": 0,
                "auc_drop_mean": float("nan"),
                "auc_drop_std": float("nan"),
                "logloss_increase_mean": float("nan"),
                "permuted_label_auc_drop_mean": float("nan"),
                "permuted_label_auc_drop_p95": float("nan"),
            }
            continue
        auc_drops: list[float] = []
        loss_increases: list[float] = []
        null_auc_drops: list[float] = []
        for _ in range(n_repeats):
            permuted = x_val.copy()
            order = rng.permutation(len(permuted))
            permuted.loc[:, cols] = permuted[cols].to_numpy()[order]
            perm_pred = model.predict_proba(permuted)[:, 1]
            if has_both:
                auc_drops.append(float(baseline_auc - roc_auc_score(y_val, perm_pred)))
            loss_increases.append(float(log_loss(y_val, perm_pred, labels=[0, 1]) - baseline_loss))

            shuffled_y = rng.permutation(y_val)
            if len(np.unique(shuffled_y)) > 1:
                null_auc_drops.append(float(roc_auc_score(shuffled_y, pred) - roc_auc_score(shuffled_y, perm_pred)))
        out[group_name] = {
            "n_columns": int(len(cols)),
            "auc_drop_mean": float(np.nanmean(auc_drops)) if auc_drops else float("nan"),
            "auc_drop_std": float(np.nanstd(auc_drops)) if auc_drops else float("nan"),
            "logloss_increase_mean": float(np.mean(loss_increases)),
            "permuted_label_auc_drop_mean": float(np.nanmean(null_auc_drops)) if null_auc_drops else float("nan"),
            "permuted_label_auc_drop_p95": float(np.nanpercentile(null_auc_drops, 95)) if null_auc_drops else float("nan"),
        }
    return out


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


def _plot_prediction_histogram(pred: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.hist(pred, bins=30, color="#2E86AB", alpha=0.85)
    ax.set_xlabel("Predicted called-strike probability")
    ax.set_ylabel("Rows")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


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
    rounds = np.arange(1, max_len + 1)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
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
    n_splits: int = 5,
) -> ModelResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    required = [spec.target, spec.group]
    clean = df.dropna(subset=required).copy().reset_index(drop=True)
    if len(clean) < n_splits:
        raise ValueError(f"{spec.name}: not enough rows for {n_splits}-fold CV")
    y = clean[spec.target].astype(int).to_numpy()
    groups = clean[spec.group].astype(str).to_numpy()
    categories = build_global_categories(clean, spec)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)
    oof_pred = np.zeros(len(clean), dtype=float)
    fold_metrics: list[dict[str, float]] = []
    best_iterations: list[int] = []
    permutation_payloads: dict[str, list[dict[str, float]]] = {g: [] for g in spec.permutation_groups}
    curves: list[dict[str, list[float]]] = []
    rng = np.random.default_rng(GLOBAL_SEED)

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(clean, y, groups)):
        train_df = clean.iloc[train_idx].copy()
        val_df = clean.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]
        transformer = make_transformer(train_df, spec, categories)
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
        fold_metrics.append(
            {
                "auc": float(roc_auc_score(y_val, pred)) if len(np.unique(y_val)) > 1 else float("nan"),
                "logloss": float(log_loss(y_val, pred, labels=[0, 1])),
                "brier": float(brier_score_loss(y_val, pred)),
            }
        )
        best_iterations.append(int(model.best_iteration_ or spec.n_estimators))
        group_payload = permutation_importance_groups(model, x_val, y_val, spec.permutation_groups, rng)
        for group_name, payload in group_payload.items():
            permutation_payloads[group_name].append(payload)
        curves.append(
            {
                "training": model.evals_result_["training"]["binary_logloss"],
                "validation": model.evals_result_["valid_1"]["binary_logloss"],
            }
        )

    auc = float(roc_auc_score(y, oof_pred)) if len(np.unique(y)) > 1 else float("nan")
    loss = float(log_loss(y, oof_pred, labels=[0, 1]))
    brier = float(brier_score_loss(y, oof_pred))
    title = spec.name.replace("_", " ").title()
    calibration = _plot_calibration(y, oof_pred, output_dir / f"{spec.name}_calibration.png", f"{title} Calibration")
    _plot_roc(y, oof_pred, auc, output_dir / f"{spec.name}_roc.png", f"{title} ROC")
    _plot_prediction_histogram(oof_pred, output_dir / f"{spec.name}_prediction_hist.png", f"{title} Prediction Histogram")
    _plot_learning_curves(curves, output_dir / f"{spec.name}_learning_curve.png", f"{title} Learning Curve")

    final_estimators = max(50, int(np.nanmedian(best_iterations)))
    final_transformer = make_transformer(clean, spec, categories)
    x_full = encode_frame(clean, final_transformer)
    final_model = lgb.LGBMClassifier(**lgbm_params(spec, GLOBAL_SEED + 333, n_estimators=final_estimators))
    final_model.fit(x_full, y)

    perm_summary: dict[str, Any] = {}
    for group_name, payloads in permutation_payloads.items():
        frame = pd.DataFrame(payloads)
        if frame.empty:
            continue
        perm_summary[group_name] = {col: float(frame[col].mean()) for col in frame.columns if col != "n_columns"}
        perm_summary[group_name]["n_columns"] = int(frame["n_columns"].max())

    metrics = {
        "auc": auc,
        "logloss": loss,
        "brier": brier,
        "folds": fold_metrics,
        "best_iterations": best_iterations,
        "n_rows": int(len(clean)),
        "n_games": int(pd.Series(groups).nunique()),
        "positive_rate": float(y.mean()),
        "calibration": calibration,
        "permutation_importance": perm_summary,
    }
    return ModelResult(
        name=spec.name,
        model=final_model,
        transformer=final_transformer,
        metrics=metrics,
        oof_pred=oof_pred,
        training_frame=clean,
        feature_names=list(x_full.columns),
    )


def predict_with_model(result: ModelResult, df: pd.DataFrame) -> np.ndarray:
    x = encode_frame(df, result.transformer)
    return result.model.predict_proba(x)[:, 1]


def predict_as_lineup_spot(result: ModelResult, df: pd.DataFrame, spot: int) -> np.ndarray:
    modified = df.copy()
    modified["lineup_spot"] = int(spot)
    return predict_with_model(result, modified)


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int = 500, seed_offset: int = 0) -> tuple[float, float, np.ndarray]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return float("nan"), float("nan"), np.array([])
    rng = np.random.default_rng(GLOBAL_SEED + seed_offset)
    idx = np.arange(len(clean))
    boots = np.empty(n_bootstrap, dtype=float)
    for boot_idx in range(n_bootstrap):
        sample = rng.choice(idx, size=len(idx), replace=True)
        boots[boot_idx] = clean[sample].mean()
    low, high = np.quantile(boots, [0.025, 0.975]).tolist()
    return float(low), float(high), boots


def bootstrap_two_sided_p(effect: float, boot_values: np.ndarray) -> float:
    boot_values = np.asarray(boot_values, dtype=float)
    boot_values = boot_values[np.isfinite(boot_values)]
    if len(boot_values) < 5:
        return float("nan")
    se = float(np.std(boot_values, ddof=1))
    if se <= 0:
        return 0.0 if effect != 0 else 1.0
    z = abs(float(effect) / se)
    return float(2.0 * (1.0 - stats.norm.cdf(z)))


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out
    q = valid * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q.iloc[::-1]).iloc[::-1].clip(upper=1.0)
    out.loc[q.index] = q
    return out


def count_state_index(series: pd.Series) -> pd.Series:
    order = {
        "0-0": 0,
        "1-0": 1,
        "0-1": 2,
        "2-0": 3,
        "1-1": 4,
        "0-2": 5,
        "3-0": 6,
        "2-1": 7,
        "1-2": 8,
        "3-1": 9,
        "2-2": 10,
        "3-2": 11,
    }
    return series.astype(str).map(order).fillna(-1).astype(float)


def pitcher_fame_numeric(series: pd.Series) -> pd.Series:
    mapping = {"Q1_low": 1.0, "Q2": 2.0, "Q3": 3.0, "Q4_high": 4.0, "unknown": 2.5}
    return series.fillna("unknown").astype(str).map(mapping).fillna(2.5).astype(float)

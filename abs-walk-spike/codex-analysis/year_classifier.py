from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from utils import ABS_ZONE_BOTTOM, ABS_ZONE_TOP, ARTIFACTS_DIR, CHARTS_DIR, GLOBAL_SEED, ensure_output_dirs, save_json

NUMERIC_FEATURES = ["plate_x", "plate_z", "plate_z_norm", "sz_top", "sz_bot", "batter_height_proxy"]
LOCATION_ONLY_FEATURES = ["plate_x", "plate_z", "plate_z_norm"]


@dataclass
class YearClassifierResult:
    cv_auc: float
    location_only_cv_auc: float
    fold_aucs: list[float]
    best_iterations: list[int]
    pitch_categories: list[str]
    model: lgb.LGBMClassifier
    shap_importance: pd.DataFrame
    permutation_importance: pd.DataFrame
    pd_surface: np.ndarray
    x_grid: np.ndarray
    z_grid: np.ndarray
    top_region: dict[str, list[float]]
    top_pitch_types: list[str]
    holdout_auc: float
    diagnostics: dict


def compress_pitch_types(series: pd.Series, min_count: int = 100) -> tuple[pd.Series, list[str]]:
    cleaned = series.fillna("UNK").astype(str)
    counts = cleaned.value_counts()
    keep = counts[counts >= min_count].index.tolist()
    reduced = cleaned.where(cleaned.isin(keep), "OTHER")
    categories = sorted(set(keep + ["OTHER"]))
    return reduced, categories


def encode_year_matrix(
    df: pd.DataFrame,
    pitch_categories: list[str],
    numeric_features: list[str] | None = None,
) -> pd.DataFrame:
    numeric_features = numeric_features or NUMERIC_FEATURES
    encoded = df.copy()
    encoded["pitch_type_reduced"] = encoded["pitch_type"].fillna("UNK").astype(str)
    encoded["pitch_type_reduced"] = encoded["pitch_type_reduced"].where(
        encoded["pitch_type_reduced"].isin(pitch_categories),
        "OTHER",
    )
    pitch_type = pd.Categorical(encoded["pitch_type_reduced"], categories=pitch_categories)
    dummies = pd.get_dummies(pitch_type, prefix="pitch_type", dtype=float)
    return pd.concat([encoded[numeric_features].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def _lgbm_params(random_state: int, n_estimators: int = 400) -> dict:
    return {
        "objective": "binary",
        "n_estimators": n_estimators,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 80,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }


def _connected_component_bbox(mask: np.ndarray, x_grid: np.ndarray, z_grid: np.ndarray, score: np.ndarray) -> dict[str, list[float]]:
    visited = np.zeros_like(mask, dtype=bool)
    best_cells: list[tuple[int, int]] = []
    best_score = -np.inf
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i, j] or visited[i, j]:
                continue
            queue = [(i, j)]
            visited[i, j] = True
            cells: list[tuple[int, int]] = []
            component_score = -np.inf
            while queue:
                r, c = queue.pop()
                cells.append((r, c))
                component_score = max(component_score, score[r, c])
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            if component_score > best_score:
                best_score = component_score
                best_cells = cells

    if not best_cells:
        max_idx = np.unravel_index(np.argmax(score), score.shape)
        best_cells = [max_idx]

    rows = [cell[0] for cell in best_cells]
    cols = [cell[1] for cell in best_cells]
    return {
        "x_range": [float(x_grid[min(cols)]), float(x_grid[max(cols)])],
        "z_range": [float(z_grid[min(rows)]), float(z_grid[max(rows)])],
    }


def _plot_learning_curves(fold_curves: list[dict[str, list[float]]], path: Path) -> None:
    max_len = max(len(curve["training"]) for curve in fold_curves)
    train_pad = []
    val_pad = []
    for curve in fold_curves:
        train = curve["training"] + [curve["training"][-1]] * (max_len - len(curve["training"]))
        val = curve["validation"] + [curve["validation"][-1]] * (max_len - len(curve["validation"]))
        train_pad.append(train)
        val_pad.append(val)

    train_mean = np.mean(np.asarray(train_pad), axis=0)
    val_mean = np.mean(np.asarray(val_pad), axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    rounds = np.arange(1, max_len + 1)
    ax.plot(rounds, train_mean, label="Train log loss", linewidth=2)
    ax.plot(rounds, val_mean, label="Validation log loss", linewidth=2)
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Binary log loss")
    ax.set_title("Year Classifier Learning Curves")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _compute_partial_dependence(
    model: lgb.LGBMClassifier,
    reference_df: pd.DataFrame,
    pitch_categories: list[str],
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    sample_size: int = 1500,
    numeric_features: list[str] | None = None,
) -> np.ndarray:
    numeric_features = numeric_features or NUMERIC_FEATURES
    reference = reference_df.sample(min(sample_size, len(reference_df)), random_state=GLOBAL_SEED).copy()
    encoded = encode_year_matrix(reference, pitch_categories, numeric_features=numeric_features)
    base = encoded.to_numpy(dtype=float)
    feature_order = list(encoded.columns)
    idx_x = feature_order.index("plate_x")
    idx_z = feature_order.index("plate_z")
    idx_z_norm = feature_order.index("plate_z_norm")
    height_proxy = reference["batter_height_proxy"].to_numpy()
    n_ref = len(reference)

    surface = np.zeros((len(z_grid), len(x_grid)))
    for zi, z_value in enumerate(z_grid):
        z_absolute = z_value * height_proxy
        batch = np.tile(base, (len(x_grid), 1))
        batch[:, idx_z_norm] = z_value
        batch[:, idx_z] = np.tile(z_absolute, len(x_grid))
        batch[:, idx_x] = np.concatenate([np.full(n_ref, x_value) for x_value in x_grid])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            preds = model.predict_proba(batch)[:, 1].reshape(len(x_grid), n_ref)
        surface[zi, :] = preds.mean(axis=1)
    return surface


def _plot_partial_dependence(
    surface: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    top_region: dict[str, list[float]],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    extent = [x_grid.min(), x_grid.max(), z_grid.min(), z_grid.max()]
    image = ax.imshow(surface, origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax.axvline(-17.0 / 24.0, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.axvline(17.0 / 24.0, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.axhline(ABS_ZONE_BOTTOM, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.axhline(ABS_ZONE_TOP, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
    x0, x1 = top_region["x_range"]
    z0, z1 = top_region["z_range"]
    ax.add_patch(
        plt.Rectangle((x0, z0), max(x1 - x0, 0.03), max(z1 - z0, 0.03), fill=False, edgecolor="#7CFC00", linewidth=2.0)
    )
    ax.set_xlabel("plate_x (ft)")
    ax.set_ylabel("plate_z_norm")
    ax.set_title("Year Classifier Partial Dependence: P(2026) Across the Plate")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Predicted probability of 2026")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_year_classifier(called_2025: pd.DataFrame, called_2026: pd.DataFrame) -> YearClassifierResult:
    ensure_output_dirs()
    combined = pd.concat([called_2025.copy(), called_2026.copy()], ignore_index=True)
    reduced_pitch, pitch_categories = compress_pitch_types(combined["pitch_type"])
    combined["pitch_type"] = reduced_pitch

    combined["season_label"] = (combined["season"] == 2026).astype(int)
    groups = combined["season"].astype(str) + "_" + combined["game_pk"].astype(str)

    design_matrix = encode_year_matrix(combined, pitch_categories)
    location_design = encode_year_matrix(combined, pitch_categories, numeric_features=LOCATION_ONLY_FEATURES)
    target = combined["season_label"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    oof_pred = np.zeros(len(combined))
    fold_aucs: list[float] = []
    best_iterations: list[int] = []
    fold_curves: list[dict[str, list[float]]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(design_matrix, target, groups)):
        x_train = design_matrix.iloc[train_idx]
        y_train = target[train_idx]
        x_val = design_matrix.iloc[val_idx]
        y_val = target[val_idx]

        model = lgb.LGBMClassifier(**_lgbm_params(GLOBAL_SEED + fold_idx))
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        pred = model.predict_proba(x_val)[:, 1]
        oof_pred[val_idx] = pred
        fold_auc = roc_auc_score(y_val, pred)
        fold_aucs.append(float(fold_auc))
        best_iterations.append(int(model.best_iteration_ or model.n_estimators))
        fold_curves.append(
            {
                "training": model.evals_result_["training"]["binary_logloss"],
                "validation": model.evals_result_["valid_1"]["binary_logloss"],
            }
        )

    cv_auc = float(roc_auc_score(target, oof_pred))

    location_oof = np.zeros(len(combined))
    location_best_iterations: list[int] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(location_design, target, groups)):
        x_train = location_design.iloc[train_idx]
        y_train = target[train_idx]
        x_val = location_design.iloc[val_idx]
        y_val = target[val_idx]
        location_model = lgb.LGBMClassifier(**_lgbm_params(GLOBAL_SEED + 500 + fold_idx))
        location_model.fit(
            x_train,
            y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        location_oof[val_idx] = location_model.predict_proba(x_val)[:, 1]
        location_best_iterations.append(int(location_model.best_iteration_ or location_model.n_estimators))

    location_only_cv_auc = float(roc_auc_score(target, location_oof))

    # Independent holdout for permutation-importance sanity checking.
    holdout_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED + 17)
    train_idx, holdout_idx = next(holdout_splitter.split(design_matrix, target, groups))
    x_train = design_matrix.iloc[train_idx]
    y_train = target[train_idx]
    x_holdout = design_matrix.iloc[holdout_idx]
    y_holdout = target[holdout_idx]
    diagnostic_model = lgb.LGBMClassifier(**_lgbm_params(GLOBAL_SEED + 101))
    diagnostic_model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_holdout, y_holdout)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    holdout_pred = diagnostic_model.predict_proba(x_holdout)[:, 1]
    holdout_auc = float(roc_auc_score(y_holdout, holdout_pred))
    perm = permutation_importance(
        diagnostic_model,
        x_holdout,
        y_holdout,
        n_repeats=12,
        scoring="roc_auc",
        random_state=GLOBAL_SEED,
        n_jobs=-1,
    )
    permutation_df = (
        pd.DataFrame(
            {
                "feature": x_holdout.columns,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    final_estimators = int(np.median(best_iterations))
    final_model = lgb.LGBMClassifier(**_lgbm_params(GLOBAL_SEED + 333, n_estimators=final_estimators))
    final_model.fit(design_matrix, target)
    location_final_model = lgb.LGBMClassifier(
        **_lgbm_params(GLOBAL_SEED + 444, n_estimators=int(np.median(location_best_iterations)))
    )
    location_final_model.fit(location_design, target)

    shap_sample = combined.sample(min(len(combined), 5000), random_state=GLOBAL_SEED).copy()
    shap_matrix = encode_year_matrix(shap_sample, pitch_categories)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(shap_matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    shap.summary_plot(shap_values, shap_matrix, show=False, max_display=15, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "year_classifier_shap.png", dpi=220, bbox_inches="tight")
    plt.close()

    shap_importance = (
        pd.DataFrame({"feature": shap_matrix.columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    x_grid = np.linspace(float(combined["plate_x"].quantile(0.02)), float(combined["plate_x"].quantile(0.98)), 60)
    z_grid = np.linspace(float(combined["plate_z_norm"].quantile(0.02)), float(combined["plate_z_norm"].quantile(0.98)), 60)
    pd_surface = _compute_partial_dependence(
        location_final_model,
        combined,
        pitch_categories,
        x_grid,
        z_grid,
        numeric_features=LOCATION_ONLY_FEATURES,
    )
    threshold = np.quantile(pd_surface, 0.95)
    top_region = _connected_component_bbox(pd_surface >= threshold, x_grid, z_grid, pd_surface)
    _plot_partial_dependence(pd_surface, x_grid, z_grid, top_region, CHARTS_DIR / "year_classifier_partial_dependence.png")
    _plot_learning_curves(fold_curves, CHARTS_DIR / "year_classifier_learning_curve.png")

    top_pitch_types = [
        feature.replace("pitch_type_", "")
        for feature in shap_importance["feature"].tolist()
        if feature.startswith("pitch_type_")
    ][:3]

    diagnostics = {
        "cv_auc": cv_auc,
        "location_only_cv_auc": location_only_cv_auc,
        "holdout_auc": holdout_auc,
        "fold_aucs": fold_aucs,
        "best_iterations": best_iterations,
        "top_shap_features": shap_importance.head(10).to_dict(orient="records"),
        "top_permutation_features": permutation_df.head(10).to_dict(orient="records"),
    }
    save_json(diagnostics, ARTIFACTS_DIR / "year_classifier_metrics.json")
    shap_importance.to_csv(ARTIFACTS_DIR / "year_classifier_shap_importance.csv", index=False)
    permutation_df.to_csv(ARTIFACTS_DIR / "year_classifier_permutation_importance.csv", index=False)

    return YearClassifierResult(
        cv_auc=cv_auc,
        location_only_cv_auc=location_only_cv_auc,
        fold_aucs=fold_aucs,
        best_iterations=best_iterations,
        pitch_categories=pitch_categories,
        model=final_model,
        shap_importance=shap_importance,
        permutation_importance=permutation_df,
        pd_surface=pd_surface,
        x_grid=x_grid,
        z_grid=z_grid,
        top_region=top_region,
        top_pitch_types=top_pitch_types,
        holdout_auc=holdout_auc,
        diagnostics=diagnostics,
    )

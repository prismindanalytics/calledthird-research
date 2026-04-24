from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from utils import (
    ABS_ZONE_BOTTOM,
    ABS_ZONE_TOP,
    ARTIFACTS_DIR,
    CHARTS_DIR,
    GLOBAL_SEED,
    PLATE_HALF_WIDTH_FT,
    bootstrap_interval,
    ensure_output_dirs,
    save_json,
)

ZONE_FEATURES = ["plate_x", "plate_z_norm"]


@dataclass
class ZoneAnalysisResult:
    model_2025: Any
    model_2026: Any
    x_grid: np.ndarray
    z_grid: np.ndarray
    delta_surface: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    bootstrap_surfaces: np.ndarray
    largest_region: dict
    shrink_region: dict | None
    h1_pass: bool
    stability: dict
    supplementary: dict
    distribution_shift: dict


def _zone_model(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=0.2,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=4000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_zone_model(df: pd.DataFrame, random_state: int) -> Pipeline:
    model = _zone_model(random_state)
    model.fit(df[ZONE_FEATURES].to_numpy(dtype=float), df["is_called_strike"].to_numpy(dtype=int))
    return model


def build_zone_grid(called_2025: pd.DataFrame, called_2026: pd.DataFrame, points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    combined = pd.concat([called_2025[ZONE_FEATURES], called_2026[ZONE_FEATURES]], ignore_index=True)
    x_low = max(-1.6, float(combined["plate_x"].quantile(0.01)))
    x_high = min(1.6, float(combined["plate_x"].quantile(0.99)))
    z_low = max(0.12, float(combined["plate_z_norm"].quantile(0.01)))
    z_high = min(0.68, float(combined["plate_z_norm"].quantile(0.99)))
    return np.linspace(x_low, x_high, points), np.linspace(z_low, z_high, points)


def predict_surface(model: Pipeline, x_grid: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    xx, zz = np.meshgrid(x_grid, z_grid)
    grid = np.column_stack([xx.ravel(), zz.ravel()])
    preds = model.predict_proba(grid)[:, 1]
    return preds.reshape(len(z_grid), len(x_grid))


def _connected_components(mask: np.ndarray, signed_surface: np.ndarray) -> list[dict]:
    visited = np.zeros_like(mask, dtype=bool)
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    components: list[dict] = []

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i, j] or visited[i, j]:
                continue
            queue = [(i, j)]
            visited[i, j] = True
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.pop()
                cells.append((r, c))
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            values = np.array([signed_surface[r, c] for r, c in cells])
            components.append(
                {
                    "cells": cells,
                    "mean_delta": float(values.mean()),
                    "max_abs_delta": float(np.abs(values).max()),
                    "size": int(len(cells)),
                }
            )

    return components


def find_largest_region(
    delta_surface: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    threshold_pp: float = 3.0,
    direction: str | None = None,
) -> tuple[dict, bool]:
    threshold = threshold_pp / 100.0
    if direction == "negative":
        significant = (ci_high < 0) & (delta_surface <= -threshold)
    elif direction == "positive":
        significant = (ci_low > 0) & (delta_surface >= threshold)
    else:
        significant = ((ci_low > 0) | (ci_high < 0)) & (np.abs(delta_surface) >= threshold)
    components = _connected_components(significant, delta_surface)
    if not components:
        surface = -delta_surface if direction == "negative" else delta_surface if direction == "positive" else np.abs(delta_surface)
        max_idx = np.unravel_index(np.argmax(surface), surface.shape)
        i, j = max_idx
        region = {
            "x_range": [float(x_grid[j]), float(x_grid[j])],
            "z_range": [float(z_grid[i]), float(z_grid[i])],
            "delta_pp": float(delta_surface[i, j] * 100),
            "mean_delta_pp": float(delta_surface[i, j] * 100),
            "size": 1,
        }
        return region, False

    components = sorted(components, key=lambda item: (item["max_abs_delta"], item["size"]), reverse=True)
    best = components[0]
    rows = [cell[0] for cell in best["cells"]]
    cols = [cell[1] for cell in best["cells"]]
    span_ok = len(set(rows)) > 1 and len(set(cols)) > 1
    region = {
        "x_range": [float(x_grid[min(cols)]), float(x_grid[max(cols)])],
        "z_range": [float(z_grid[min(rows)]), float(z_grid[max(rows)])],
        "delta_pp": float(best["max_abs_delta"] * 100 * np.sign(best["mean_delta"])),
        "mean_delta_pp": float(best["mean_delta"] * 100),
        "size": int(best["size"]),
    }
    return region, span_ok


def _energy_distance_2d(frame_a: pd.DataFrame, frame_b: pd.DataFrame, sample_size: int = 3000) -> float:
    rng = np.random.default_rng(GLOBAL_SEED)
    a = frame_a[["plate_x", "plate_z_norm"]].to_numpy(dtype=float)
    b = frame_b[["plate_x", "plate_z_norm"]].to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if len(a) > sample_size:
        a = a[rng.choice(len(a), size=sample_size, replace=False)]
    if len(b) > sample_size:
        b = b[rng.choice(len(b), size=sample_size, replace=False)]
    cross = cdist(a, b).mean()
    within_a = pdist(a).mean()
    within_b = pdist(b).mean()
    return float(2 * cross - within_a - within_b)


def compute_distribution_shift(called_2025: pd.DataFrame, called_2026: pd.DataFrame) -> dict:
    called_strikes_2025 = called_2025[called_2025["description"] == "called_strike"].copy()
    called_strikes_2026 = called_2026[called_2026["description"] == "called_strike"].copy()

    stats_payload = {
        "energy_distance_overall": _energy_distance_2d(called_strikes_2025, called_strikes_2026),
        "ks_plate_x": {
            "statistic": float(stats.ks_2samp(called_strikes_2025["plate_x"], called_strikes_2026["plate_x"]).statistic),
            "pvalue": float(stats.ks_2samp(called_strikes_2025["plate_x"], called_strikes_2026["plate_x"]).pvalue),
        },
        "ks_plate_z_norm": {
            "statistic": float(
                stats.ks_2samp(called_strikes_2025["plate_z_norm"], called_strikes_2026["plate_z_norm"]).statistic
            ),
            "pvalue": float(
                stats.ks_2samp(called_strikes_2025["plate_z_norm"], called_strikes_2026["plate_z_norm"]).pvalue
            ),
        },
    }

    top_2025 = called_strikes_2025[called_strikes_2025["plate_z_norm"] >= ABS_ZONE_TOP - 0.02]
    top_2026 = called_strikes_2026[called_strikes_2026["plate_z_norm"] >= ABS_ZONE_TOP - 0.02]
    bottom_2025 = called_strikes_2025[called_strikes_2025["plate_z_norm"] <= ABS_ZONE_BOTTOM + 0.02]
    bottom_2026 = called_strikes_2026[called_strikes_2026["plate_z_norm"] <= ABS_ZONE_BOTTOM + 0.02]
    stats_payload["energy_distance_top_band"] = _energy_distance_2d(top_2025, top_2026, sample_size=1500)
    stats_payload["energy_distance_bottom_band"] = _energy_distance_2d(bottom_2025, bottom_2026, sample_size=1500)
    stats_payload["top_band_share_2025"] = float((called_strikes_2025["plate_z_norm"] >= ABS_ZONE_TOP - 0.02).mean())
    stats_payload["top_band_share_2026"] = float((called_strikes_2026["plate_z_norm"] >= ABS_ZONE_TOP - 0.02).mean())
    stats_payload["bottom_band_share_2025"] = float((called_strikes_2025["plate_z_norm"] <= ABS_ZONE_BOTTOM + 0.02).mean())
    stats_payload["bottom_band_share_2026"] = float((called_strikes_2026["plate_z_norm"] <= ABS_ZONE_BOTTOM + 0.02).mean())
    return stats_payload


def _plot_zone_delta(
    delta_surface: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    largest_region: dict,
    path: Path,
) -> None:
    significant = (ci_low > 0) | (ci_high < 0)
    extent = [x_grid.min(), x_grid.max(), z_grid.min(), z_grid.max()]
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    image = ax.imshow(delta_surface * 100, origin="lower", extent=extent, cmap="coolwarm", aspect="auto", vmin=-8, vmax=8)
    if significant.any() and not significant.all():
        ax.contour(x_grid, z_grid, significant.astype(int), levels=[0.5], colors="black", linewidths=0.9)
    ax.axvline(-PLATE_HALF_WIDTH_FT, linestyle="--", color="black", linewidth=1.0, alpha=0.8)
    ax.axvline(PLATE_HALF_WIDTH_FT, linestyle="--", color="black", linewidth=1.0, alpha=0.8)
    ax.axhline(ABS_ZONE_BOTTOM, linestyle="--", color="black", linewidth=1.0, alpha=0.8)
    ax.axhline(ABS_ZONE_TOP, linestyle="--", color="black", linewidth=1.0, alpha=0.8)
    x0, x1 = largest_region["x_range"]
    z0, z1 = largest_region["z_range"]
    ax.add_patch(
        plt.Rectangle((x0, z0), max(x1 - x0, 0.03), max(z1 - z0, 0.03), fill=False, edgecolor="#FFD700", linewidth=2.0)
    )
    ax.set_xlabel("plate_x (ft)")
    ax.set_ylabel("plate_z_norm")
    ax.set_title("Two-Zone Classifier Delta: 2026 minus 2025 called-strike probability")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Delta called-strike probability (pp)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_zone_analysis(
    called_2025: pd.DataFrame,
    called_2026: pd.DataFrame,
    called_2025_full: pd.DataFrame,
    called_2026_full: pd.DataFrame,
    n_bootstrap: int = 100,
) -> ZoneAnalysisResult:
    ensure_output_dirs()
    x_grid, z_grid = build_zone_grid(called_2025, called_2026)
    model_2025 = fit_zone_model(called_2025, GLOBAL_SEED)
    model_2026 = fit_zone_model(called_2026, GLOBAL_SEED + 1)
    surface_2025 = predict_surface(model_2025, x_grid, z_grid)
    surface_2026 = predict_surface(model_2026, x_grid, z_grid)
    delta_surface = surface_2026 - surface_2025

    bootstrap_surfaces = np.zeros((n_bootstrap, len(z_grid), len(x_grid)), dtype=np.float32)
    rng = np.random.default_rng(GLOBAL_SEED)

    for bootstrap_idx in range(n_bootstrap):
        sample_2025 = called_2025.iloc[rng.choice(len(called_2025), size=len(called_2025), replace=True)].copy()
        sample_2026 = called_2026.iloc[rng.choice(len(called_2026), size=len(called_2026), replace=True)].copy()
        boot_2025 = fit_zone_model(sample_2025, GLOBAL_SEED + 1000 + bootstrap_idx)
        boot_2026 = fit_zone_model(sample_2026, GLOBAL_SEED + 2000 + bootstrap_idx)
        bootstrap_surfaces[bootstrap_idx] = predict_surface(boot_2026, x_grid, z_grid) - predict_surface(boot_2025, x_grid, z_grid)

    ci_low = np.quantile(bootstrap_surfaces, 0.025, axis=0)
    ci_high = np.quantile(bootstrap_surfaces, 0.975, axis=0)
    largest_region, contiguous = find_largest_region(delta_surface, ci_low, ci_high, x_grid, z_grid)
    shrink_region, shrink_contiguous = find_largest_region(
        delta_surface,
        ci_low,
        ci_high,
        x_grid,
        z_grid,
        direction="negative",
    )

    x0, x1 = largest_region["x_range"]
    z0, z1 = largest_region["z_range"]
    bbox_mask = (x_grid >= x0) & (x_grid <= x1)
    z_mask = (z_grid >= z0) & (z_grid <= z1)
    region_values = bootstrap_surfaces[:, z_mask, :][:, :, bbox_mask]
    region_mean = region_values.mean(axis=(1, 2))
    stability = {
        "region_mean_delta_ci_pp": [float(value * 100) for value in bootstrap_interval(region_mean)],
        "region_sign_agreement": float(np.mean(np.sign(region_mean) == np.sign(largest_region["mean_delta_pp"]))),
        "bootstrap_max_abs_delta_ci_pp": [float(value * 100) for value in bootstrap_interval(np.abs(bootstrap_surfaces).max(axis=(1, 2)))],
    }

    supplementary_model_2025 = fit_zone_model(called_2025_full, GLOBAL_SEED + 7)
    supplementary_model_2026 = fit_zone_model(called_2026_full, GLOBAL_SEED + 8)
    supplementary_surface = predict_surface(supplementary_model_2026, x_grid, z_grid) - predict_surface(
        supplementary_model_2025, x_grid, z_grid
    )
    supplementary_region, supplementary_contiguous = find_largest_region(
        supplementary_surface,
        supplementary_surface,
        supplementary_surface,
        x_grid,
        z_grid,
    )
    supplementary = {
        "largest_delta_pp": supplementary_region["delta_pp"],
        "largest_region": supplementary_region,
        "contiguous": supplementary_contiguous,
        "surface_correlation_with_primary": float(np.corrcoef(delta_surface.ravel(), supplementary_surface.ravel())[0, 1]),
    }

    distribution_shift = compute_distribution_shift(called_2025, called_2026)
    _plot_zone_delta(delta_surface, ci_low, ci_high, x_grid, z_grid, largest_region, CHARTS_DIR / "zone_classifier_delta_heatmap.png")

    diagnostics = {
        "largest_region": largest_region,
        "shrink_region": shrink_region,
        "stability": stability,
        "distribution_shift": distribution_shift,
        "supplementary": supplementary,
    }
    save_json(diagnostics, ARTIFACTS_DIR / "zone_classifier_metrics.json")

    return ZoneAnalysisResult(
        model_2025=model_2025,
        model_2026=model_2026,
        x_grid=x_grid,
        z_grid=z_grid,
        delta_surface=delta_surface,
        ci_low=ci_low,
        ci_high=ci_high,
        bootstrap_surfaces=bootstrap_surfaces,
        largest_region=largest_region,
        shrink_region=shrink_region if shrink_contiguous else None,
        h1_pass=shrink_contiguous,
        stability=stability,
        supplementary=supplementary,
        distribution_shift=distribution_shift,
    )


def compute_count_location_deltas(called_2026: pd.DataFrame, model_2025: Any, model_2026: Any) -> pd.DataFrame:
    scored = called_2026.copy()
    features = scored[ZONE_FEATURES].to_numpy(dtype=float)
    scored["p_2025"] = model_2025.predict_proba(features)[:, 1]
    scored["p_2026"] = model_2026.predict_proba(features)[:, 1]
    scored["delta"] = scored["p_2026"] - scored["p_2025"]
    summary = (
        scored.groupby("count_state", observed=True)
        .agg(
            pitches=("count_state", "size"),
            mean_called_strike_delta=("delta", "mean"),
            mean_2025=("p_2025", "mean"),
            mean_2026=("p_2026", "mean"),
        )
        .reset_index()
    )
    ordering = {count: idx for idx, count in enumerate(["0-0", "1-0", "0-1", "2-0", "1-1", "0-2", "3-0", "2-1", "1-2", "3-1", "2-2", "3-2"])}
    summary["sort_key"] = summary["count_state"].map(ordering).fillna(999)
    return summary.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)

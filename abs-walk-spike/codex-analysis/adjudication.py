from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import (  # noqa: E402
    BALL_LIKE_DESCRIPTIONS,
    BUNT_FOUL_DESCRIPTIONS,
    CALLED_DESCRIPTIONS,
    CHARTS_DIR,
    FOUL_DESCRIPTIONS,
    FOUL_TIP_DESCRIPTIONS,
    GLOBAL_SEED,
    IN_PLAY_DESCRIPTIONS,
    STRIKE_LIKE_DESCRIPTIONS,
    WALK_EVENTS,
    bootstrap_interval,
    ensure_output_dirs,
    save_json,
)

DATA_2026_PATH = SCRIPT_DIR.parent / "data" / "statcast_2026_mar27_apr22.parquet"
DATA_2025_PATH = SCRIPT_DIR.parent / ".." / "count-distribution-abs" / "data" / "statcast_2025_mar27_apr14.parquet"
RESULTS_PATH = SCRIPT_DIR / "adjudication_results.json"
CHART_PATH = CHARTS_DIR / "adjudication_zone_delta_absolute.png"

ROUND1_ATTRIBUTION_PCT = -56.17
GRID_POINTS = 100
GRID_X_RANGE = (-1.5, 1.5)
GRID_Z_RANGE = (0.5, 4.5)
ZONE_FEATURES = ["plate_x", "plate_z"]
PRIMARY_START_2025 = pd.Timestamp("2025-03-27")
PRIMARY_END_2025 = pd.Timestamp("2025-04-14")
PRIMARY_START_2026 = pd.Timestamp("2026-03-27")
PRIMARY_END_2026 = pd.Timestamp("2026-04-14")
REQUIRED_COLUMNS = [
    "game_date",
    "description",
    "events",
    "plate_x",
    "plate_z",
    "balls",
    "strikes",
    "game_pk",
    "at_bat_number",
    "pitch_number",
]


@dataclass
class PitchStep:
    description: str
    event: str | None
    called_idx: int


@dataclass
class PASequence:
    pa_id: str
    actual_walk: int
    actual_counts: tuple[str, ...]
    steps: tuple[PitchStep, ...]


def _load_window(path: Path, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=REQUIRED_COLUMNS).copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    frame = frame[(frame["game_date"] >= start_date) & (frame["game_date"] <= end_date)].copy()
    frame["count_state"] = frame["balls"].astype(int).astype(str) + "-" + frame["strikes"].astype(int).astype(str)
    return frame


def _prepare_pitch_frame(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["season"] = season
    prepared["is_called_strike"] = (prepared["description"] == "called_strike").astype(int)
    prepared["pa_id"] = (
        str(season)
        + "_"
        + prepared["game_pk"].astype(int).astype(str)
        + "_"
        + prepared["at_bat_number"].astype(int).astype(str)
    )
    return prepared


def prepare_called_pitches(frame: pd.DataFrame) -> pd.DataFrame:
    called = frame[frame["description"].isin(CALLED_DESCRIPTIONS)].copy()
    return called.dropna(subset=ZONE_FEATURES)


def compute_walk_rate(frame: pd.DataFrame) -> float:
    terminal = frame[frame["events"].notna() & (frame["events"] != "")].copy()
    return float(terminal["events"].isin(WALK_EVENTS).mean())


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


def fit_zone_model(frame: pd.DataFrame, random_state: int) -> Pipeline:
    model = _zone_model(random_state)
    model.fit(frame[ZONE_FEATURES].to_numpy(dtype=float), frame["is_called_strike"].to_numpy(dtype=int))
    return model


def build_zone_grid(points: int = GRID_POINTS) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.linspace(GRID_X_RANGE[0], GRID_X_RANGE[1], points),
        np.linspace(GRID_Z_RANGE[0], GRID_Z_RANGE[1], points),
    )


def predict_surface(model: Pipeline, x_grid: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    xx, zz = np.meshgrid(x_grid, z_grid)
    grid = np.column_stack([xx.ravel(), zz.ravel()])
    return model.predict_proba(grid)[:, 1].reshape(len(z_grid), len(x_grid))


def _connected_components(mask: np.ndarray, signed_surface: np.ndarray) -> list[dict[str, Any]]:
    visited = np.zeros_like(mask, dtype=bool)
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    components: list[dict[str, Any]] = []

    for row_idx in range(mask.shape[0]):
        for col_idx in range(mask.shape[1]):
            if not mask[row_idx, col_idx] or visited[row_idx, col_idx]:
                continue
            queue = [(row_idx, col_idx)]
            visited[row_idx, col_idx] = True
            cells: list[tuple[int, int]] = []
            while queue:
                row, col = queue.pop()
                cells.append((row, col))
                for delta_row, delta_col in offsets:
                    next_row = row + delta_row
                    next_col = col + delta_col
                    if (
                        0 <= next_row < mask.shape[0]
                        and 0 <= next_col < mask.shape[1]
                        and mask[next_row, next_col]
                        and not visited[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))

            values = np.array([signed_surface[row, col] for row, col in cells], dtype=float)
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
    direction: str,
    threshold_pp: float = 3.0,
) -> dict[str, Any]:
    threshold = threshold_pp / 100.0
    if direction == "positive":
        significant = (ci_low > 0) & (delta_surface >= threshold)
    elif direction == "negative":
        significant = (ci_high < 0) & (delta_surface <= -threshold)
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    components = _connected_components(significant, delta_surface)
    if not components:
        return {
            "x_range": [],
            "z_range": [],
            "mean_delta_pp": float("nan"),
            "max_abs_delta_pp": float("nan"),
            "n_cells": 0,
        }

    components = sorted(components, key=lambda item: (item["max_abs_delta"], item["size"]), reverse=True)
    best = components[0]
    rows = [cell[0] for cell in best["cells"]]
    cols = [cell[1] for cell in best["cells"]]
    return {
        "x_range": [float(x_grid[min(cols)]), float(x_grid[max(cols)])],
        "z_range": [float(z_grid[min(rows)]), float(z_grid[max(rows)])],
        "mean_delta_pp": float(best["mean_delta"] * 100),
        "max_abs_delta_pp": float(best["max_abs_delta"] * 100),
        "n_cells": int(best["size"]),
    }


def bootstrap_zone_delta(
    called_2025: pd.DataFrame,
    called_2026: pd.DataFrame,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    n_bootstrap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_2025 = fit_zone_model(called_2025, GLOBAL_SEED)
    model_2026 = fit_zone_model(called_2026, GLOBAL_SEED + 1)
    delta_surface = predict_surface(model_2026, x_grid, z_grid) - predict_surface(model_2025, x_grid, z_grid)

    rng = np.random.default_rng(GLOBAL_SEED)
    boot_surfaces = np.zeros((n_bootstrap, len(z_grid), len(x_grid)), dtype=np.float32)
    for bootstrap_idx in range(n_bootstrap):
        sample_2025 = called_2025.iloc[rng.choice(len(called_2025), size=len(called_2025), replace=True)].copy()
        sample_2026 = called_2026.iloc[rng.choice(len(called_2026), size=len(called_2026), replace=True)].copy()
        boot_model_2025 = fit_zone_model(sample_2025, GLOBAL_SEED + 1000 + bootstrap_idx)
        boot_model_2026 = fit_zone_model(sample_2026, GLOBAL_SEED + 2000 + bootstrap_idx)
        boot_surfaces[bootstrap_idx] = predict_surface(boot_model_2026, x_grid, z_grid) - predict_surface(
            boot_model_2025,
            x_grid,
            z_grid,
        )

    ci_low = np.quantile(boot_surfaces, 0.025, axis=0)
    ci_high = np.quantile(boot_surfaces, 0.975, axis=0)
    return delta_surface, ci_low, ci_high


def _plot_delta_surface(
    delta_surface: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    positive_region: dict[str, Any],
    negative_region: dict[str, Any],
    path: Path,
) -> None:
    significant = (ci_low > 0) | (ci_high < 0)
    extent = [x_grid.min(), x_grid.max(), z_grid.min(), z_grid.max()]
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    image = ax.imshow(
        delta_surface * 100,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        aspect="auto",
        vmin=-12,
        vmax=12,
    )
    if significant.any():
        ax.contour(x_grid, z_grid, significant.astype(int), levels=[0.5], colors="black", linewidths=0.9)

    for region, color, label in (
        (positive_region, "#f59e0b", "Largest positive"),
        (negative_region, "#10b981", "Largest negative"),
    ):
        if region["n_cells"] == 0:
            continue
        x0, x1 = region["x_range"]
        z0, z1 = region["z_range"]
        ax.add_patch(
            plt.Rectangle(
                (x0, z0),
                max(x1 - x0, 0.03),
                max(z1 - z0, 0.03),
                fill=False,
                edgecolor=color,
                linewidth=2.0,
                label=label,
            )
        )

    ax.set_xlabel("plate_x (ft)")
    ax.set_ylabel("plate_z (ft)")
    ax.set_title("Absolute-Coordinate Zone Delta: 2026 minus 2025 called-strike probability")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Delta called-strike probability (pp)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _actual_count_rates(frame: pd.DataFrame) -> pd.DataFrame:
    terminal_ids = set(frame.loc[frame["events"].notna() & (frame["events"] != ""), "pa_id"])
    filtered = frame[frame["pa_id"].isin(terminal_ids)].copy()
    return (
        filtered[["count_state", "pa_id"]]
        .drop_duplicates()
        .merge(
            filtered[["pa_id", "events"]]
            .dropna()
            .drop_duplicates(subset=["pa_id"], keep="last")
            .assign(actual_walk=lambda value: value["events"].isin(WALK_EVENTS).astype(int))[["pa_id", "actual_walk"]],
            on="pa_id",
            how="left",
        )
        .groupby("count_state", observed=True)
        .agg(pas=("pa_id", "size"), walks=("actual_walk", "sum"))
        .assign(walk_rate=lambda value: value["walks"] / value["pas"])
        .reset_index()
    )


def build_pa_sequences(
    frame: pd.DataFrame,
    include_called_pitch: Callable[[pd.DataFrame], pd.Series],
) -> tuple[list[PASequence], pd.DataFrame]:
    ordered = frame.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True).copy()
    terminal = ordered[ordered["events"].notna() & (ordered["events"] != "")].copy()
    outcomes = terminal.drop_duplicates(subset=["pa_id"], keep="last").set_index("pa_id")
    valid_pa_ids = set(outcomes.index)
    ordered = ordered[ordered["pa_id"].isin(valid_pa_ids)].reset_index(drop=True).copy()

    called_mask = ordered["description"].isin(CALLED_DESCRIPTIONS) & ordered[ZONE_FEATURES].notna().all(axis=1)
    active_mask = called_mask & include_called_pitch(ordered)
    ordered["called_idx"] = -1
    ordered.loc[active_mask, "called_idx"] = np.arange(int(active_mask.sum()))

    sequences: list[PASequence] = []
    for pa_id, group in ordered.groupby("pa_id", sort=False):
        steps = tuple(
            PitchStep(
                description=row.description,
                event=None if pd.isna(row.events) or row.events == "" else str(row.events),
                called_idx=int(row.called_idx),
            )
            for row in group.itertuples()
        )
        sequences.append(
            PASequence(
                pa_id=str(pa_id),
                actual_walk=int(outcomes.loc[pa_id, "events"] in WALK_EVENTS),
                actual_counts=tuple(group["count_state"].drop_duplicates().tolist()),
                steps=steps,
            )
        )

    called_frame = ordered.loc[active_mask, ["pa_id", "count_state", "description", "plate_x", "plate_z"]].reset_index(drop=True)
    return sequences, called_frame


def _advance_actual_pitch(description: str, event: str | None, balls: int, strikes: int) -> tuple[bool, float, int, int, bool]:
    if event in WALK_EVENTS:
        return True, 1.0, balls, strikes, False
    if description in IN_PLAY_DESCRIPTIONS or description == "hit_by_pitch" or (event is not None and event not in WALK_EVENTS):
        return True, 0.0, balls, strikes, False
    if description in BALL_LIKE_DESCRIPTIONS:
        if balls == 3:
            return True, 1.0, 4, strikes, False
        return False, 0.0, balls + 1, strikes, False
    if description in STRIKE_LIKE_DESCRIPTIONS:
        if strikes == 2:
            return True, 0.0, balls, 3, False
        return False, 0.0, balls, strikes + 1, False
    if description in FOUL_TIP_DESCRIPTIONS:
        if strikes == 2:
            return True, 0.0, balls, 3, False
        return False, 0.0, balls, strikes + 1, False
    if description in BUNT_FOUL_DESCRIPTIONS:
        if strikes == 2:
            return True, 0.0, balls, 3, False
        return False, 0.0, balls, strikes + 1, False
    if description in FOUL_DESCRIPTIONS:
        if strikes < 2:
            return False, 0.0, balls, strikes + 1, False
        return False, 0.0, balls, strikes, False
    raise ValueError(f"Unhandled description in replay: {description}")


def simulate_pa(
    sequence: PASequence,
    strike_probs: np.ndarray,
    continuation_probs: dict[str, float],
    rng: np.random.Generator,
) -> tuple[float, bool]:
    balls = 0
    strikes = 0

    for step in sequence.steps:
        if step.called_idx >= 0:
            is_strike = bool(rng.random() < strike_probs[step.called_idx])
            if is_strike:
                if strikes == 2:
                    return 0.0, False
                strikes += 1
            else:
                if balls == 3:
                    return 1.0, False
                balls += 1
            continue

        resolved, walk_value, balls, strikes, unresolved = _advance_actual_pitch(step.description, step.event, balls, strikes)
        if resolved:
            return walk_value, unresolved

    tail_state = f"{balls}-{strikes}"
    return float(continuation_probs.get(tail_state, continuation_probs["0-0"])), True


def simulate_dataset(
    sequences: list[PASequence],
    strike_probs: np.ndarray,
    continuation_probs: dict[str, float],
    draws: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    outcomes = np.zeros(len(sequences), dtype=float)
    unresolved = np.zeros(len(sequences), dtype=float)

    for draw_idx in range(draws):
        rng = np.random.default_rng(seed + draw_idx)
        for pa_idx, sequence in enumerate(sequences):
            walk_value, unresolved_flag = simulate_pa(sequence, strike_probs, continuation_probs, rng)
            outcomes[pa_idx] += walk_value
            unresolved[pa_idx] += float(unresolved_flag)

    outcomes /= draws
    unresolved /= draws
    return outcomes, float(unresolved.mean())


def run_counterfactual(
    df_2025: pd.DataFrame,
    df_2026: pd.DataFrame,
    called_2025: pd.DataFrame,
    actual_2025_walk_rate: float,
    actual_2026_walk_rate: float,
    include_called_pitch: Callable[[pd.DataFrame], pd.Series],
    n_bootstrap: int = 100,
    draws_per_bootstrap: int = 12,
) -> dict[str, Any]:
    continuation = _actual_count_rates(df_2026)
    continuation_probs = dict(zip(continuation["count_state"], continuation["walk_rate"], strict=True))
    continuation_probs.setdefault("0-0", actual_2026_walk_rate)

    sequences, called_frame_2026 = build_pa_sequences(df_2026, include_called_pitch)
    cf_features = called_frame_2026[ZONE_FEATURES].to_numpy(dtype=float)
    point_model = fit_zone_model(called_2025, GLOBAL_SEED + 500)
    point_probs = point_model.predict_proba(cf_features)[:, 1] if len(cf_features) else np.array([], dtype=float)
    pa_walk_probs, unresolved_share_mean = simulate_dataset(
        sequences,
        point_probs,
        continuation_probs,
        draws=32,
        seed=GLOBAL_SEED + 700,
    )

    counterfactual_walk_rate = float(pa_walk_probs.mean())
    denominator = actual_2026_walk_rate - actual_2025_walk_rate
    attribution_pct = float(((actual_2026_walk_rate - counterfactual_walk_rate) / denominator) * 100.0)

    rng = np.random.default_rng(GLOBAL_SEED + 900)
    bootstrap_walk_rates = np.zeros(n_bootstrap, dtype=float)
    bootstrap_attribution = np.zeros(n_bootstrap, dtype=float)
    bootstrap_unresolved = np.zeros(n_bootstrap, dtype=float)

    for bootstrap_idx in range(n_bootstrap):
        sample_2025 = called_2025.iloc[rng.choice(len(called_2025), size=len(called_2025), replace=True)].copy()
        boot_model = fit_zone_model(sample_2025, GLOBAL_SEED + 10000 + bootstrap_idx)
        boot_probs = boot_model.predict_proba(cf_features)[:, 1] if len(cf_features) else np.array([], dtype=float)
        boot_pa_probs, boot_unresolved = simulate_dataset(
            sequences,
            boot_probs,
            continuation_probs,
            draws=draws_per_bootstrap,
            seed=GLOBAL_SEED + 20000 + bootstrap_idx * 13,
        )
        bootstrap_walk_rates[bootstrap_idx] = float(boot_pa_probs.mean())
        bootstrap_unresolved[bootstrap_idx] = boot_unresolved
        bootstrap_attribution[bootstrap_idx] = float(
            ((actual_2026_walk_rate - bootstrap_walk_rates[bootstrap_idx]) / denominator) * 100.0
        )

    return {
        "counterfactual_walk_rate": counterfactual_walk_rate,
        "attribution_pct": attribution_pct,
        "bootstrap_walk_rate_ci": [float(value) for value in bootstrap_interval(bootstrap_walk_rates)],
        "bootstrap_attribution_ci": [float(value) for value in bootstrap_interval(bootstrap_attribution)],
        "unresolved_share_mean": unresolved_share_mean,
        "bootstrap_unresolved_share_mean": float(bootstrap_unresolved.mean()),
        "n_counterfactual_called_pitches": int(len(called_frame_2026)),
    }


def _all_called_pitch_mask(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=frame.index)


def _first_pitch_only_mask(frame: pd.DataFrame) -> pd.Series:
    # Claude's critique is specifically about the first pitch of the PA at 0-0.
    return frame["count_state"] == "0-0"


def choose_verdict(counterfactual_all_attr: float, counterfactual_first_pitch_attr: float) -> str:
    if counterfactual_all_attr > 50.0:
        return "concedes_to_claude"
    if counterfactual_all_attr < 25.0 and counterfactual_first_pitch_attr < 25.0:
        return "defends_b2"
    return "mixed"


def main() -> None:
    ensure_output_dirs()

    raw_2025 = _load_window(DATA_2025_PATH, PRIMARY_START_2025, PRIMARY_END_2025)
    raw_2026 = _load_window(DATA_2026_PATH, PRIMARY_START_2026, PRIMARY_END_2026)

    df_2025 = _prepare_pitch_frame(raw_2025, 2025)
    df_2026 = _prepare_pitch_frame(raw_2026, 2026)
    called_2025 = prepare_called_pitches(df_2025)
    called_2026 = prepare_called_pitches(df_2026)

    actual_2025_walk_rate = compute_walk_rate(df_2025)
    actual_2026_walk_rate = compute_walk_rate(df_2026)

    x_grid, z_grid = build_zone_grid()
    delta_surface, ci_low, ci_high = bootstrap_zone_delta(called_2025, called_2026, x_grid, z_grid, n_bootstrap=100)
    positive_region = find_largest_region(delta_surface, ci_low, ci_high, x_grid, z_grid, direction="positive")
    negative_region = find_largest_region(delta_surface, ci_low, ci_high, x_grid, z_grid, direction="negative")

    counterfactual_all = run_counterfactual(
        df_2025=df_2025,
        df_2026=df_2026,
        called_2025=called_2025,
        actual_2025_walk_rate=actual_2025_walk_rate,
        actual_2026_walk_rate=actual_2026_walk_rate,
        include_called_pitch=_all_called_pitch_mask,
        n_bootstrap=100,
    )
    counterfactual_first_pitch = run_counterfactual(
        df_2025=df_2025,
        df_2026=df_2026,
        called_2025=called_2025,
        actual_2025_walk_rate=actual_2025_walk_rate,
        actual_2026_walk_rate=actual_2026_walk_rate,
        include_called_pitch=_first_pitch_only_mask,
        n_bootstrap=100,
    )

    _plot_delta_surface(delta_surface, ci_low, ci_high, x_grid, z_grid, positive_region, negative_region, CHART_PATH)

    absolute_attr = counterfactual_all["attribution_pct"]
    verdict = choose_verdict(absolute_attr, counterfactual_first_pitch["attribution_pct"])
    results = {
        "method": "absolute plate_z coords, no sz_* features",
        "largest_positive_region_absolute": positive_region,
        "largest_negative_region_absolute": negative_region,
        "counterfactual_all_walk_rate": round(counterfactual_all["counterfactual_walk_rate"], 6),
        "counterfactual_all_attribution_pct": round(counterfactual_all["attribution_pct"], 2),
        "counterfactual_first_pitch_walk_rate": round(counterfactual_first_pitch["counterfactual_walk_rate"], 6),
        "counterfactual_first_pitch_attribution_pct": round(counterfactual_first_pitch["attribution_pct"], 2),
        "actual_2025_walk_rate": round(actual_2025_walk_rate, 6),
        "actual_2026_walk_rate": round(actual_2026_walk_rate, 6),
        "comparison_to_round1": {
            "round1_attribution_pct": ROUND1_ATTRIBUTION_PCT,
            "absolute_coord_attribution_pct": round(absolute_attr, 2),
            "delta": round(absolute_attr - ROUND1_ATTRIBUTION_PCT, 2),
            "verdict": verdict,
        },
    }
    save_json(results, RESULTS_PATH)


if __name__ == "__main__":
    main()

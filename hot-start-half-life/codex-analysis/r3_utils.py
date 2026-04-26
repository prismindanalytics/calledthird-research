from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import BASE_DIR, CHARTS_DIR, DATASETS_DIR, MODELS_DIR, SEED, atomic_write_json
from r2_utils import (
    NAMED_HITTER_KEYS,
    NAMED_PITCHER_KEYS,
    R2_HITTER_FEATURE_COLS,
    R2_RELIEVER_FEATURE_COLS,
    add_player_names,
    add_r2_hitter_columns,
    interval_coverage,
)


ROUND3_DIR = BASE_DIR / "round3"
R3_MODELS_DIR = MODELS_DIR / "r3"
R3_CHARTS_DIR = CHARTS_DIR / "r3"
R3_DIAG_DIR = R3_CHARTS_DIR / "diag"
R3_TABLES_DIR = ROUND3_DIR / "tables"

for directory in [ROUND3_DIR, R3_MODELS_DIR, R3_CHARTS_DIR, R3_DIAG_DIR, R3_TABLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


R3_HITTER_FEATURE_COLS = list(R2_HITTER_FEATURE_COLS)
R3_RELIEVER_FEATURE_COLS = list(R2_RELIEVER_FEATURE_COLS)
R3_REPORTED_GAP_FEATURE = "xwoba_minus_prior_woba_22g"
R3_UNREPORTED_GAP_FEATURES = {"xwoba_minus_woba_22g", "abs_xwoba_minus_woba_22g"}
R3_BOOTSTRAP_N = 100
R3_MIN_HITTER_PRIOR_WOBA = 0.0
R3_MIN_RELIEVER_PRIOR_K_RATE = 0.10


def finite_or_none(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def records_for_json(df: pd.DataFrame, cols: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append({col: finite_or_none(row.get(col)) for col in cols})
    return out


def load_hitter_training_data() -> pd.DataFrame:
    hist = pd.read_parquet(DATASETS_DIR / "hitter_features.parquet")
    hist = add_r2_hitter_columns(hist)
    return hist[
        hist["season"].between(2022, 2025)
        & hist["pa_22g"].ge(50)
        & hist["pa_ros"].ge(50)
        & hist["ros_woba_delta_vs_prior"].notna()
    ].copy()


def validation_prior_woba_sd(data: pd.DataFrame) -> float:
    valid = data[data["season"].eq(2024)].copy()
    sd = valid["ros_woba_delta_vs_prior"].std(ddof=1)
    return float(sd) if pd.notna(sd) and np.isfinite(sd) else 0.05


def weighted_history_sd(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return float("nan")
    weights = np.array([5.0, 4.0, 3.0], dtype=float)[: len(clean)]
    mean = float(np.average(clean, weights=weights))
    return float(np.sqrt(np.average((clean - mean) ** 2, weights=weights)))


def add_2026_prior_woba_sd(universe: pd.DataFrame, floor_sd: float) -> pd.DataFrame:
    hist = pd.read_parquet(DATASETS_DIR / "hitter_features.parquet")
    rows = []
    for batter, group in hist[hist["season"].between(2023, 2025)].groupby("batter"):
        vals = group.sort_values("season", ascending=False).head(3)["woba_full"]
        rows.append({"batter": int(batter), "player_prior_woba_sd": weighted_history_sd(vals)})
    sd_lookup = pd.DataFrame(rows)
    out = universe.merge(sd_lookup, on="batter", how="left")
    out["preseason_prior_woba_sd_floor"] = float(floor_sd)
    player_sd = pd.to_numeric(out["player_prior_woba_sd"], errors="coerce")
    out["preseason_prior_woba_sd"] = np.sqrt(float(floor_sd) ** 2 + player_sd.fillna(0.0) ** 2)
    return out


def load_r3_hitter_predictions() -> pd.DataFrame:
    path = DATASETS_DIR / "r3_persistence_predictions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_csv(R3_TABLES_DIR / "r3_persistence_predictions.csv")


def write_findings(payload: dict[str, Any]) -> None:
    atomic_write_json(BASE_DIR / "findings_r3.json", payload)


__all__ = [
    "NAMED_HITTER_KEYS",
    "NAMED_PITCHER_KEYS",
    "R3_BOOTSTRAP_N",
    "R3_CHARTS_DIR",
    "R3_DIAG_DIR",
    "R3_HITTER_FEATURE_COLS",
    "R3_MIN_HITTER_PRIOR_WOBA",
    "R3_MIN_RELIEVER_PRIOR_K_RATE",
    "R3_MODELS_DIR",
    "R3_RELIEVER_FEATURE_COLS",
    "R3_REPORTED_GAP_FEATURE",
    "R3_TABLES_DIR",
    "R3_UNREPORTED_GAP_FEATURES",
    "SEED",
    "add_2026_prior_woba_sd",
    "add_player_names",
    "finite_or_none",
    "interval_coverage",
    "load_hitter_training_data",
    "load_r3_hitter_predictions",
    "records_for_json",
    "validation_prior_woba_sd",
    "write_findings",
]

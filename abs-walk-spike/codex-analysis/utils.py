from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "codex-analysis"
CHARTS_DIR = ANALYSIS_DIR / "charts"
ARTIFACTS_DIR = ANALYSIS_DIR / "artifacts"

DATA_2026_PATH = ROOT / "data" / "statcast_2026_mar27_apr22.parquet"
DATA_2025_PRIMARY_PATH = ROOT / ".." / "count-distribution-abs" / "data" / "statcast_2025_mar27_apr14.parquet"
DATA_2025_FULL_PATH = ROOT / ".." / "count-distribution-abs" / "data" / "statcast_2025_full.parquet"
APRIL_HISTORY_PATH = ROOT / "data" / "april_walk_history.csv"
SUBSTRATE_SUMMARY_PATH = ROOT / "data" / "substrate_summary.json"

PRIMARY_END_2025 = pd.Timestamp("2025-04-14")
PRIMARY_END_2026 = pd.Timestamp("2026-04-14")
FULL_END_2025 = pd.Timestamp("2025-04-22")
FULL_END_2026 = pd.Timestamp("2026-04-22")
FULL_START_2025 = pd.Timestamp("2025-03-27")

CALLED_DESCRIPTIONS = {"called_strike", "ball"}
AUTO_DESCRIPTIONS = {"automatic_ball", "automatic_strike"}
WALK_EVENTS = {"walk", "intent_walk"}

BALL_LIKE_DESCRIPTIONS = {"ball", "blocked_ball", "pitchout", "automatic_ball"}
STRIKE_LIKE_DESCRIPTIONS = {
    "called_strike",
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
    "automatic_strike",
}
FOUL_DESCRIPTIONS = {"foul"}
FOUL_TIP_DESCRIPTIONS = {"foul_tip"}
BUNT_FOUL_DESCRIPTIONS = {"foul_bunt", "bunt_foul_tip"}
IN_PLAY_DESCRIPTIONS = {"hit_into_play", "hit_into_play_score", "hit_into_play_no_out"}

COUNT_ORDER = ["0-0", "1-0", "0-1", "2-0", "1-1", "0-2", "3-0", "2-1", "1-2", "3-1", "2-2", "3-2"]
ABS_VERTICAL_SHARE = 0.535 - 0.27
ABS_ZONE_BOTTOM = 0.27
ABS_ZONE_TOP = 0.535
PLATE_HALF_WIDTH_FT = 17.0 / 24.0
GLOBAL_SEED = 20260423


def ensure_output_dirs() -> None:
    for path in (ANALYSIS_DIR, CHARTS_DIR, ARTIFACTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def load_substrate_summary() -> dict[str, Any]:
    return json.loads(SUBSTRATE_SUMMARY_PATH.read_text())


def load_pitch_data() -> dict[str, pd.DataFrame]:
    df_2026 = pd.read_parquet(DATA_2026_PATH)
    df_2025_primary = pd.read_parquet(DATA_2025_PRIMARY_PATH)
    df_2025_full = pd.read_parquet(DATA_2025_FULL_PATH)

    for frame in (df_2026, df_2025_primary, df_2025_full):
        frame["game_date"] = pd.to_datetime(frame["game_date"])

    return {
        "2025_primary": df_2025_primary[df_2025_primary["game_date"] <= PRIMARY_END_2025].copy(),
        "2026_primary": df_2026[df_2026["game_date"] <= PRIMARY_END_2026].copy(),
        "2025_full_window": df_2025_full[
            (df_2025_full["game_date"] >= FULL_START_2025) & (df_2025_full["game_date"] <= FULL_END_2025)
        ].copy(),
        "2026_full_window": df_2026[df_2026["game_date"] <= FULL_END_2026].copy(),
    }


def load_april_history() -> pd.DataFrame:
    return pd.read_csv(APRIL_HISTORY_PATH)


def add_derived_columns(df: pd.DataFrame, season: int) -> pd.DataFrame:
    enriched = df.copy()
    enriched["season"] = season
    enriched["zone_height"] = enriched["sz_top"] - enriched["sz_bot"]
    enriched["batter_height_proxy"] = enriched["zone_height"] / ABS_VERTICAL_SHARE
    enriched["plate_z_norm"] = enriched["plate_z"] / enriched["batter_height_proxy"]
    enriched["pitch_type"] = enriched["pitch_type"].fillna("UNK").astype(str)
    enriched["count_state"] = enriched["balls"].astype(int).astype(str) + "-" + enriched["strikes"].astype(int).astype(str)
    enriched["is_called_strike"] = (enriched["description"] == "called_strike").astype(int)
    enriched["pa_id"] = (
        str(season)
        + "_"
        + enriched["game_pk"].astype(int).astype(str)
        + "_"
        + enriched["at_bat_number"].astype(int).astype(str)
    )
    return enriched


def prepare_called_pitches(df: pd.DataFrame, season: int) -> pd.DataFrame:
    called = add_derived_columns(df, season)
    called = called[called["description"].isin(CALLED_DESCRIPTIONS)].copy()
    called = called.dropna(subset=["plate_x", "plate_z", "plate_z_norm", "sz_top", "sz_bot", "batter_height_proxy"])
    return called


def sort_pitch_sequence(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def compute_walk_rate(df: pd.DataFrame) -> dict[str, float]:
    terminal = df[df["events"].notna() & (df["events"] != "")].copy()
    walk_rate = float(terminal["events"].isin(WALK_EVENTS).mean())
    return {
        "pas": int(len(terminal)),
        "walks": int(terminal["events"].isin(WALK_EVENTS).sum()),
        "walk_rate": walk_rate,
    }


def build_pa_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    enriched = sort_pitch_sequence(df)
    terminal = enriched[enriched["events"].notna() & (enriched["events"] != "")].copy()
    terminal = terminal.drop_duplicates(subset=["pa_id"], keep="last")
    terminal["actual_walk"] = terminal["events"].isin(WALK_EVENTS).astype(int)
    return terminal[["pa_id", "events", "actual_walk"]]


def compute_count_walk_rates(df: pd.DataFrame) -> pd.DataFrame:
    reached = sort_pitch_sequence(df)[["pa_id", "count_state"]].drop_duplicates()
    outcomes = build_pa_outcomes(df)[["pa_id", "actual_walk"]]
    merged = reached.merge(outcomes, on="pa_id", how="left", validate="many_to_one")
    summary = (
        merged.groupby("count_state", observed=True)
        .agg(pas=("pa_id", "size"), walks=("actual_walk", "sum"))
        .reset_index()
    )
    summary["walk_rate"] = summary["walks"] / summary["pas"]
    summary["ci_low"], summary["ci_high"] = zip(
        *[wilson_interval(int(w), int(n)) for w, n in zip(summary["walks"], summary["pas"], strict=True)]
    )
    ordering = {count: idx for idx, count in enumerate(COUNT_ORDER)}
    summary["sort_key"] = summary["count_state"].map(ordering).fillna(999)
    return summary.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials == 0:
        return (np.nan, np.nan)
    p_hat = successes / trials
    denom = 1.0 + (z**2 / trials)
    center = (p_hat + (z**2 / (2 * trials))) / denom
    margin = z * np.sqrt((p_hat * (1.0 - p_hat) / trials) + (z**2 / (4 * trials**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_interval(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return tuple(np.quantile(values, [alpha / 2, 1 - alpha / 2]).tolist())


def region_label(x_range: tuple[float, float], z_range: tuple[float, float]) -> str:
    x_low, x_high = x_range
    z_low, z_high = z_range
    vertical = "middle"
    horizontal = "center"

    if z_low >= ABS_ZONE_TOP - 0.015:
        vertical = "top"
    elif z_high <= ABS_ZONE_BOTTOM + 0.015:
        vertical = "bottom"
    elif z_low < ABS_ZONE_BOTTOM and z_high > ABS_ZONE_TOP:
        vertical = "full-height"
    elif z_high < (ABS_ZONE_BOTTOM + ABS_ZONE_TOP) / 2:
        vertical = "lower-middle"
    elif z_low > (ABS_ZONE_BOTTOM + ABS_ZONE_TOP) / 2:
        vertical = "upper-middle"

    if x_high <= -0.45:
        horizontal = "glove-side edge"
    elif x_low >= 0.45:
        horizontal = "arm-side edge"
    elif x_low < -0.45 and x_high > 0.45:
        horizontal = "full-width"

    return f"{vertical} {horizontal}".strip()


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.2f} pp"

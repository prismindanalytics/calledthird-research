"""Shared loaders and helpers for the Claude (Agent A) Round 2 ABS walk-spike analysis.

All paths resolved relative to the project root
`/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike`.

Round 2 adds:
  - Apr 23 – May 13 extension (data/statcast_2026_apr23_may13.parquet)
  - 2025 same-window subset (Apr 23 – May 13)
  - Weekly aggregates (7-day windows starting Mar 27)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike")
DATA_DIR = PROJECT_ROOT / "data"
SHARED_2025 = PROJECT_ROOT.parent / "count-distribution-abs" / "data"
R2_DIR = PROJECT_ROOT / "claude-analysis-r2"
R2_DATA = R2_DIR / "data"
R2_CHARTS = R2_DIR / "charts"
R2_DIAG = R2_CHARTS / "diagnostics"
R2_ARTIFACTS = R2_DIR / "artifacts"

PARQUET_2026_R1 = DATA_DIR / "statcast_2026_mar27_apr22.parquet"
PARQUET_2026_R2 = R2_DATA / "statcast_2026_apr23_may13.parquet"
PARQUET_2025_FULL = SHARED_2025 / "statcast_2025_full.parquet"

WINDOW_START = pd.Timestamp("2026-03-27")
WINDOW_END = pd.Timestamp("2026-05-13")
SAFE_WINDOW_END_DAYS = 1  # exclude the last day if very recent (handled in data_prep)

WALK_EVENTS = {"walk", "intent_walk"}
AUTO_DESCRIPTIONS = {"automatic_ball", "automatic_strike"}
CALLED_DESCRIPTIONS = {"called_strike", "ball"}

ALL_COUNTS: Iterable[str] = [
    "0-0", "0-1", "0-2",
    "1-0", "1-1", "1-2",
    "2-0", "2-1", "2-2",
    "3-0", "3-1", "3-2",
]


def load_2026_full() -> pd.DataFrame:
    """Round 1 + Round 2: Mar 27 – May 13, 2026 (full window)."""
    r1 = pd.read_parquet(PARQUET_2026_R1)
    r2 = pd.read_parquet(PARQUET_2026_R2)
    common = sorted(set(r1.columns) & set(r2.columns))
    df = pd.concat([r1[common], r2[common]], ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = 2026
    df = df.sort_values("game_date").reset_index(drop=True)
    return df


def load_2025_full() -> pd.DataFrame:
    """Full 2025 season (used for fitting zone classifier in H3)."""
    df = pd.read_parquet(PARQUET_2025_FULL)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = 2025
    return df


def load_2025_samewindow() -> pd.DataFrame:
    """2025 Mar 27 – May 13 subset (YoY apples-to-apples for H1)."""
    df = load_2025_full()
    mask = (df["game_date"] >= "2025-03-27") & (df["game_date"] <= "2025-05-13")
    out = df.loc[mask].copy()
    return out


def harmonize_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce two frames to their intersection of columns (Round 1 schema-safety rule)."""
    common = sorted(set(df_a.columns) & set(df_b.columns))
    return df_a[common].copy(), df_b[common].copy()


def called_pitches_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Human called pitches: description in {'called_strike','ball'} and not auto_*.

    Auto calls (automatic_ball / automatic_strike from ABS challenges) never have
    description == 'called_strike' or 'ball' anyway, but we explicitly exclude to be
    safe if pybaseball schema changes.
    """
    mask = df["description"].isin(CALLED_DESCRIPTIONS) & ~df["description"].isin(
        AUTO_DESCRIPTIONS
    )
    out = df.loc[mask & df["plate_x"].notna() & df["plate_z"].notna()].copy()
    out["is_called_strike"] = (out["description"] == "called_strike").astype(int)
    return out


def plate_appearance_mask(df: pd.DataFrame) -> pd.Series:
    """True on PA-terminating rows: events is non-null and non-empty."""
    ev = df["events"]
    return ev.notna() & (ev.astype(str).str.len() > 0) & (ev.astype(str) != "None")


def plate_appearances(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[plate_appearance_mask(df)].copy()


def walk_flag(pa_df: pd.DataFrame) -> pd.Series:
    return pa_df["events"].isin(WALK_EVENTS).astype(int)


def walk_rate(pa_df: pd.DataFrame) -> float:
    if len(pa_df) == 0:
        return float("nan")
    return walk_flag(pa_df).mean()


def count_state(df: pd.DataFrame) -> pd.Series:
    """12-cell count grid as (balls, strikes) tuple strings like '2-1'."""
    b = df["balls"].fillna(-1).astype(int)
    s = df["strikes"].fillna(-1).astype(int)
    return b.astype(str) + "-" + s.astype(str)


def week_index_2026(dates: pd.Series, *, anchor: pd.Timestamp = WINDOW_START) -> pd.Series:
    """7-day windows starting Mar 27 of the season's year.

    Week 1 = Mar 27 – Apr 2, Week 2 = Apr 3 – Apr 9, ..., Week 7 = May 8 – May 14, ...
    """
    d = pd.to_datetime(dates)
    delta_days = (d - anchor).dt.days
    return (delta_days // 7 + 1).astype(int)


def zone_region(plate_x: pd.Series, plate_z: pd.Series) -> pd.Series:
    """Classify a pitch into one of: heart, top_edge, bottom_edge, in_off, far_off.

    Geometry is the same as Round 1 absolute-coord adjudication:
      - heart: |plate_x| <= 0.7 ft, 2.0 <= plate_z <= 3.2 ft
      - top_edge: |plate_x| <= 1.0 ft, 3.2 < plate_z <= 3.9 ft
      - bottom_edge: |plate_x| <= 1.0 ft, 1.0 <= plate_z < 2.0 ft
      - in_off: |plate_x| > 1.0 or plate_z < 1.0 or plate_z > 3.9 ft
    """
    px = plate_x.astype(float)
    pz = plate_z.astype(float)
    ax = px.abs()
    region = pd.Series("in_off", index=px.index, dtype="object")
    heart_mask = (ax <= 0.7) & (pz >= 2.0) & (pz <= 3.2)
    top_mask = (ax <= 1.0) & (pz > 3.2) & (pz <= 3.9)
    bot_mask = (ax <= 1.0) & (pz >= 1.0) & (pz < 2.0)
    region.loc[heart_mask] = "heart"
    region.loc[top_mask] = "top_edge"
    region.loc[bot_mask] = "bottom_edge"
    return region


def rulebook_zone_flag(plate_x: pd.Series, plate_z: pd.Series) -> pd.Series:
    """Approximate rulebook zone: |x| <= 0.83 ft (half plate + half ball ~ 0.83) and 1.5 <= z <= 3.6 ft.

    Used as a stable per-season zone-rate metric in H4.
    """
    px = plate_x.astype(float)
    pz = plate_z.astype(float)
    return ((px.abs() <= 0.83) & (pz >= 1.5) & (pz <= 3.6)).astype(int)


def pretty_pct(p: float) -> str:
    return f"{100.0 * p:.2f}%"


def ensure_dirs() -> None:
    R2_DATA.mkdir(parents=True, exist_ok=True)
    R2_CHARTS.mkdir(parents=True, exist_ok=True)
    R2_DIAG.mkdir(parents=True, exist_ok=True)
    R2_ARTIFACTS.mkdir(parents=True, exist_ok=True)

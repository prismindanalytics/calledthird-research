"""Shared data loaders and helpers for the Claude (Agent A) ABS walk-spike analysis.

All paths are resolved relative to the project root
`/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike")
DATA_DIR = PROJECT_ROOT / "data"
SHARED_2025 = PROJECT_ROOT.parent / "count-distribution-abs" / "data"

PARQUET_2026 = DATA_DIR / "statcast_2026_mar27_apr22.parquet"
PARQUET_2025_SAMEWIN = SHARED_2025 / "statcast_2025_mar27_apr14.parquet"
PARQUET_2025_FULL = SHARED_2025 / "statcast_2025_full.parquet"
APRIL_HISTORY_CSV = DATA_DIR / "april_walk_history.csv"

WALK_EVENTS = {"walk", "intent_walk"}
AUTO_DESCRIPTIONS = {"automatic_ball", "automatic_strike"}
CALLED_DESCRIPTIONS = {"called_strike", "ball"}


def load_2026() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_2026)
    df["season"] = 2026
    return df


def load_2025_samewin() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_2025_SAMEWIN)
    df["season"] = 2025
    return df


def load_2025_extended_to_apr22() -> pd.DataFrame:
    """Full 2025 filtered to Mar 27 - Apr 22 for supplementary window extension."""
    df = pd.read_parquet(PARQUET_2025_FULL)
    df["game_date"] = pd.to_datetime(df["game_date"])
    mask = (df["game_date"] >= "2025-03-27") & (df["game_date"] <= "2025-04-22")
    out = df.loc[mask].copy()
    out["season"] = 2025
    return out


def restrict_to_primary_window(df: pd.DataFrame) -> pd.DataFrame:
    """Truncate a 2026 frame at Apr 14 so both years use Mar 27 - Apr 14."""
    d = pd.to_datetime(df["game_date"])
    mo_day = d.dt.strftime("%m-%d")
    return df.loc[(mo_day >= "03-27") & (mo_day <= "04-14")].copy()


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


def safe_plate_z_norm(df: pd.DataFrame) -> pd.Series:
    """Height-normalized plate_z using each row's sz_top / sz_bot.

    (plate_z - sz_bot) / (sz_top - sz_bot); 0 = bottom of human zone, 1 = top.
    """
    sz_top = df["sz_top"].astype(float)
    sz_bot = df["sz_bot"].astype(float)
    denom = sz_top - sz_bot
    norm = (df["plate_z"].astype(float) - sz_bot) / denom
    norm[(denom <= 0.0) | denom.isna()] = np.nan
    return norm


def count_state(df: pd.DataFrame) -> pd.Series:
    """12-cell count grid as (balls, strikes) tuple strings like '2-1'."""
    b = df["balls"].fillna(-1).astype(int)
    s = df["strikes"].fillna(-1).astype(int)
    return b.astype(str) + "-" + s.astype(str)


ALL_COUNTS: Iterable[str] = [
    "0-0", "0-1", "0-2",
    "1-0", "1-1", "1-2",
    "2-0", "2-1", "2-2",
    "3-0", "3-1", "3-2",
]


def pretty_pct(p: float) -> str:
    return f"{100.0 * p:.2f}%"

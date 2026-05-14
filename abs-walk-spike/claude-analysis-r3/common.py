"""Shared loaders and helpers for the Claude (Agent A) Round 3 ABS walk-spike analysis.

Round 3 is the A-tier elevation pass: triangulation of H3 magnitude, named adapter
leaderboard with bootstrap stability, and stuff vs command archetype interaction.

All paths resolved relative to the project root
`/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike`.

R3 reuses R1 + R2 data substrates and 2025 full-season Statcast. No re-pulls.
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
R2_ARTIFACTS = R2_DIR / "artifacts"

R3_DIR = PROJECT_ROOT / "claude-analysis-r3"
R3_DATA = R3_DIR / "data"
R3_CHARTS = R3_DIR / "charts"
R3_DIAG = R3_CHARTS / "diagnostics"
R3_ARTIFACTS = R3_DIR / "artifacts"

PARQUET_2026_R1 = DATA_DIR / "statcast_2026_mar27_apr22.parquet"
PARQUET_2026_R2 = R2_DATA / "statcast_2026_apr23_may13.parquet"
PARQUET_2025_FULL = SHARED_2025 / "statcast_2025_full.parquet"

WINDOW_START = pd.Timestamp("2026-03-27")
WINDOW_END = pd.Timestamp("2026-05-12")  # data caps May 12

WALK_EVENTS = {"walk", "intent_walk"}
AUTO_DESCRIPTIONS = {"automatic_ball", "automatic_strike"}
CALLED_DESCRIPTIONS = {"called_strike", "ball"}

ALL_COUNTS: list[str] = [
    "0-0", "0-1", "0-2",
    "1-0", "1-1", "1-2",
    "2-0", "2-1", "2-2",
    "3-0", "3-1", "3-2",
]

# Count tier — collapsed states for classifier conditioning
COUNT_TIER = {
    "0-0": "early", "1-0": "early", "0-1": "early",
    "1-1": "middle", "2-0": "middle", "0-2": "middle",
    "1-2": "two_strike", "2-2": "two_strike", "3-2": "two_strike",
    "2-1": "middle",
    "3-0": "three_ball", "3-1": "three_ball",
}
COUNT_TIERS_LIST = ["early", "middle", "two_strike", "three_ball"]


def load_2026_full() -> pd.DataFrame:
    """Round 1 + Round 2 concatenation: Mar 27 – May 12, 2026."""
    r1 = pd.read_parquet(PARQUET_2026_R1)
    r2 = pd.read_parquet(PARQUET_2026_R2)
    common = sorted(set(r1.columns) & set(r2.columns))
    df = pd.concat([r1[common], r2[common]], ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = 2026
    return df.sort_values("game_date").reset_index(drop=True)


def load_2025_full() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_2025_FULL)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = 2025
    return df


def load_2025_samewindow() -> pd.DataFrame:
    df = load_2025_full()
    mask = (df["game_date"] >= "2025-03-27") & (df["game_date"] <= "2025-05-12")
    return df.loc[mask].copy()


def called_pitches_subset(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["description"].isin(CALLED_DESCRIPTIONS) & ~df["description"].isin(
        AUTO_DESCRIPTIONS
    )
    out = df.loc[mask & df["plate_x"].notna() & df["plate_z"].notna()].copy()
    out["is_called_strike"] = (out["description"] == "called_strike").astype(int)
    return out


def plate_appearance_mask(df: pd.DataFrame) -> pd.Series:
    ev = df["events"]
    return ev.notna() & (ev.astype(str).str.len() > 0) & (ev.astype(str) != "None")


def count_state(df: pd.DataFrame) -> pd.Series:
    b = df["balls"].fillna(-1).astype(int)
    s = df["strikes"].fillna(-1).astype(int)
    return b.astype(str) + "-" + s.astype(str)


def week_index_2026(dates: pd.Series, *, anchor: pd.Timestamp = WINDOW_START) -> pd.Series:
    d = pd.to_datetime(dates)
    delta_days = (d - anchor).dt.days
    return (delta_days // 7 + 1).astype(int)


def zone_region(plate_x: pd.Series, plate_z: pd.Series) -> pd.Series:
    """heart / top_edge / bottom_edge / in_off / far_off."""
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
    px = plate_x.astype(float)
    pz = plate_z.astype(float)
    return ((px.abs() <= 0.83) & (pz >= 1.5) & (pz <= 3.6)).astype(int)


def ensure_dirs() -> None:
    R3_DATA.mkdir(parents=True, exist_ok=True)
    R3_CHARTS.mkdir(parents=True, exist_ok=True)
    R3_DIAG.mkdir(parents=True, exist_ok=True)
    R3_ARTIFACTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pitch grouping (for pitch-mix Dirichlet / JSD)
# ---------------------------------------------------------------------------

PITCH_GROUPS = {
    "fastball": ["FF", "FT", "SI", "FC"],
    "breaking": ["SL", "CU", "KC", "ST", "SV", "SC"],
    "offspeed": ["CH", "FS", "FO"],
}
PITCH_GROUP_NAMES = ["fastball", "breaking", "offspeed", "other"]


def assign_pitch_group(pitch_type: str) -> str:
    if pd.isna(pitch_type):
        return "other"
    for g, l in PITCH_GROUPS.items():
        if pitch_type in l:
            return g
    return "other"


# ---------------------------------------------------------------------------
# Game-level bootstrap helpers
# ---------------------------------------------------------------------------

def game_bootstrap_indices(
    df: pd.DataFrame,
    *,
    n_iter: int,
    seed: int = 2026,
) -> list[np.ndarray]:
    """For each iteration, produce row indices for a resample of game_pk.

    Resample is *with replacement* over unique game_pks (N=#games), then collect
    all row indices belonging to the resampled games. Row counts vary slightly
    per iteration; this is the standard game-cluster bootstrap.
    """
    rng = np.random.default_rng(seed)
    games = df["game_pk"].astype("int64").values
    unique_games = np.unique(games)
    n_games = len(unique_games)
    # Pre-build game_pk -> row indices map (vectorized)
    game_to_rows: dict[int, np.ndarray] = {}
    sort_order = np.argsort(games, kind="stable")
    sorted_games = games[sort_order]
    edges = np.searchsorted(sorted_games, unique_games)
    edges = np.r_[edges, len(games)]
    for i, g in enumerate(unique_games):
        game_to_rows[int(g)] = sort_order[edges[i]:edges[i + 1]]
    out = []
    for _ in range(n_iter):
        sampled = rng.choice(unique_games, size=n_games, replace=True)
        idxs = np.concatenate([game_to_rows[int(g)] for g in sampled])
        out.append(idxs)
    return out


def pretty_pct(p: float) -> str:
    return f"{100.0 * p:.2f}%"

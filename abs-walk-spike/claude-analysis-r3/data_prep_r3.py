"""data_prep_r3 — build R3 panels (reusing R1+R2 substrates, no re-pull).

Outputs (only built once, then cached):
  - R3_DATA / "panel_2026.parquet"  : pitch-level panel for 2026 Mar 27 – May 12
                                       with pa_id, count_state, zone_region,
                                       is_take, is_called_strike, is_walk,
                                       pa_terminating, pitch_group, week
  - R3_DATA / "panel_2025_samewindow.parquet" : 2025 same-window matched panel
  - R3_DATA / "pa_2026.parquet" : one row per 2026 PA with terminal outcome
  - R3_DATA / "pa_2025_samewindow.parquet" : 2025 same-window per-PA
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    R3_DATA,
    WALK_EVENTS,
    WINDOW_START,
    WINDOW_END,
    assign_pitch_group,
    count_state,
    ensure_dirs,
    load_2025_samewindow,
    load_2026_full,
    plate_appearance_mask,
    rulebook_zone_flag,
    week_index_2026,
    zone_region,
)


PANEL_2026_PATH = R3_DATA / "panel_2026.parquet"
PANEL_2025_PATH = R3_DATA / "panel_2025_samewindow.parquet"
PA_2026_PATH = R3_DATA / "pa_2026.parquet"
PA_2025_PATH = R3_DATA / "pa_2025_samewindow.parquet"
META_PATH = R3_DATA / "panel_meta.json"


def _build_panel(df_in: pd.DataFrame, season_label: int) -> pd.DataFrame:
    df = df_in.copy()
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    df["pa_id"] = (
        df["game_pk"].astype("Int64").astype(str)
        + "_"
        + df["at_bat_number"].astype("Int64").astype(str)
    )
    df["count_state"] = count_state(df)
    df["is_take"] = df["description"].isin(["called_strike", "ball"]).astype(int)
    df["is_called_strike"] = (df["description"] == "called_strike").astype(int)
    df["pa_terminating"] = plate_appearance_mask(df).astype(int)
    df["is_walk"] = df["events"].isin(WALK_EVENTS).astype(int)
    df["zone_region"] = zone_region(df["plate_x"], df["plate_z"])
    df["rulebook_zone"] = rulebook_zone_flag(df["plate_x"], df["plate_z"])
    df["pitch_group"] = df["pitch_type"].map(assign_pitch_group)
    df["above_3"] = (df["plate_z"].astype(float) > 3.0).astype(int)
    if season_label == 2026:
        df["week"] = week_index_2026(df["game_date"])
    else:
        # Mirror the same anchor structure on 2025 calendar
        anchor25 = pd.Timestamp("2025-03-27")
        d = pd.to_datetime(df["game_date"])
        df["week"] = ((d - anchor25).dt.days // 7 + 1).astype(int)
    return df


def _build_pa_table(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per PA with starting count, ending outcome, terminal counts."""
    term = panel.loc[panel["pa_terminating"] == 1].copy()
    # Take last row per PA (terminating row IS the last in correctly-ordered panel)
    pa = term.groupby("pa_id", as_index=False).agg({
        "game_pk": "first",
        "pitcher": "first",
        "batter": "first",
        "game_date": "first",
        "events": "last",
        "is_walk": "last",
        "balls": "last",
        "strikes": "last",
        "count_state": "last",
        "week": "first",
    })
    pa = pa.rename(columns={"count_state": "terminal_count_state"})
    # Starting count (first pitch of PA)
    first_pitch = panel.groupby("pa_id", as_index=False).agg({
        "count_state": "first",
        "zone_region": "first",
    })
    first_pitch = first_pitch.rename(columns={
        "count_state": "starting_count_state",
        "zone_region": "first_pitch_region",
    })
    pa = pa.merge(first_pitch, on="pa_id", how="left")
    # PA length
    pa_len = panel.groupby("pa_id").size().rename("n_pitches")
    pa = pa.merge(pa_len.reset_index(), on="pa_id", how="left")
    # Pa reached 3-x flag
    reached_3 = panel.groupby("pa_id").apply(
        lambda g: int((g["balls"].astype(int) >= 3).any()),
        include_groups=False,
    ).rename("reached_3_ball")
    pa = pa.merge(reached_3.reset_index(), on="pa_id", how="left")
    return pa


def build_all(force: bool = False) -> dict:
    ensure_dirs()
    if (not force) and PANEL_2026_PATH.exists() and PA_2026_PATH.exists() and PANEL_2025_PATH.exists():
        meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
        print(f"[data_prep_r3] cached panels found; reusing (panel_meta={meta})")
        return meta

    print("[data_prep_r3] building 2026 panel...")
    df_2026 = load_2026_full()
    d = pd.to_datetime(df_2026["game_date"]).dt.normalize()
    df_2026 = df_2026.loc[(d >= WINDOW_START) & (d <= WINDOW_END)].copy()
    panel_2026 = _build_panel(df_2026, season_label=2026)
    panel_2026.to_parquet(PANEL_2026_PATH, index=False)
    pa_2026 = _build_pa_table(panel_2026)
    pa_2026.to_parquet(PA_2026_PATH, index=False)

    print("[data_prep_r3] building 2025 same-window panel...")
    df_2025 = load_2025_samewindow()
    panel_2025 = _build_panel(df_2025, season_label=2025)
    panel_2025.to_parquet(PANEL_2025_PATH, index=False)
    pa_2025 = _build_pa_table(panel_2025)
    pa_2025.to_parquet(PA_2025_PATH, index=False)

    meta = {
        "panel_2026_rows": int(len(panel_2026)),
        "panel_2026_pa_count": int(len(pa_2026)),
        "panel_2026_games": int(panel_2026["game_pk"].nunique()),
        "panel_2026_pitchers": int(panel_2026["pitcher"].nunique()),
        "panel_2025_rows": int(len(panel_2025)),
        "panel_2025_pa_count": int(len(pa_2025)),
        "panel_2025_games": int(panel_2025["game_pk"].nunique()),
        "panel_2025_pitchers": int(panel_2025["pitcher"].nunique()),
        "window_start": str(WINDOW_START.date()),
        "window_end": str(WINDOW_END.date()),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[data_prep_r3] meta: {meta}")
    return meta


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def get_panel_2026() -> pd.DataFrame:
    if not PANEL_2026_PATH.exists():
        build_all()
    return pd.read_parquet(PANEL_2026_PATH)


def get_panel_2025() -> pd.DataFrame:
    if not PANEL_2025_PATH.exists():
        build_all()
    return pd.read_parquet(PANEL_2025_PATH)


def get_pa_2026() -> pd.DataFrame:
    if not PA_2026_PATH.exists():
        build_all()
    return pd.read_parquet(PA_2026_PATH)


def get_pa_2025() -> pd.DataFrame:
    if not PA_2025_PATH.exists():
        build_all()
    return pd.read_parquet(PA_2025_PATH)


if __name__ == "__main__":
    build_all(force=False)

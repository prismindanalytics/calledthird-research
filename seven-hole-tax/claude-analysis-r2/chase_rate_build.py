"""chase_rate_build.py — build prior-season (2025) chase rate per batter.

Definition (FanGraphs-style):
  chase_rate = swings on out-of-zone pitches / total out-of-zone pitches faced

A "swing" is any pitch where description != ball / called_strike / blocked_ball /
hit_by_pitch / pitchout / automatic_ball / automatic_strike. We approximate as any
non-take outcome.

Out-of-zone uses Statcast `zone` column. Statcast `zone` values 11-14 are "outside
the strike zone" (the four shadow regions). Values 1-9 are inside the zone.

Inputs:
  /Users/haohu/Documents/GitHub/calledthird/research/pitch-tunneling-atlas/data/statcast_2025_full.parquet
    (740k-row season-wide sample of 2025 Statcast)

Outputs:
  data/batter_chase_rate_2025.parquet  columns: [batter, n_oz_pitches, n_oz_swings, chase_rate, n_total_pa, qualified_200pa]

Threshold: only mark `qualified_200pa = True` for batters with >=200 plate
appearances in the 2025 sample. Round 2 H7 uses qualified batters only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

SC_2025 = Path("/Users/haohu/Documents/GitHub/calledthird/research/pitch-tunneling-atlas/data/statcast_2025_full.parquet")
OUT = DATA / "batter_chase_rate_2025.parquet"


# Pitch outcomes that count as a swing (anything that's not a take or a HBP/automatic call)
SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "foul_bunt",
    "missed_bunt", "hit_into_play", "bunt_foul_tip",
}
TAKE_DESCS = {"ball", "called_strike", "blocked_ball", "hit_by_pitch", "pitchout", "automatic_ball", "automatic_strike"}


def build(force_refresh: bool = False) -> pd.DataFrame:
    if OUT.exists() and not force_refresh:
        return pd.read_parquet(OUT)

    if not SC_2025.exists():
        raise FileNotFoundError(f"2025 Statcast parquet not found at {SC_2025}")

    print(f"Loading 2025 Statcast from {SC_2025} ...")
    df = pd.read_parquet(SC_2025, columns=[
        "game_pk", "at_bat_number", "pitch_number", "batter", "description", "zone", "events",
    ])
    n_raw = len(df)
    print(f"  raw rows: {n_raw:,}")

    df = df.dropna(subset=["batter", "zone"]).copy()
    df["batter"] = df["batter"].astype(int)
    df["zone"] = df["zone"].astype(int)
    df["out_of_zone"] = df["zone"].between(11, 14).astype(int)
    df["swung"] = df["description"].isin(SWING_DESCS).astype(int)

    # Plate appearance count from at_bat_number distinct (game_pk, at_bat_number)
    pa = (
        df.dropna(subset=["events"])  # events column has the AB result on the final pitch only
          .drop_duplicates(subset=["game_pk", "at_bat_number"])
          .groupby("batter").size().rename("n_total_pa")
    )

    # Per-batter chase numbers
    batter_grp = df.groupby("batter")
    n_oz_pitches = batter_grp["out_of_zone"].sum().rename("n_oz_pitches")
    n_oz_swings = batter_grp.apply(lambda g: int(((g["out_of_zone"] == 1) & (g["swung"] == 1)).sum()), include_groups=False).rename("n_oz_swings")

    chase = pd.concat([n_oz_pitches, n_oz_swings, pa], axis=1).reset_index()
    chase["n_total_pa"] = chase["n_total_pa"].fillna(0).astype(int)
    chase["chase_rate"] = np.where(chase["n_oz_pitches"] > 0, chase["n_oz_swings"] / chase["n_oz_pitches"], np.nan)
    chase["qualified_200pa"] = chase["n_total_pa"] >= 200

    print(f"  unique batters: {len(chase):,}")
    print(f"  qualified (>=200 PA in 2025 sample): {chase['qualified_200pa'].sum():,}")
    print(f"  median chase rate (qualified): {chase.loc[chase['qualified_200pa'], 'chase_rate'].median():.3f}")

    chase.to_parquet(OUT, index=False)
    return chase


if __name__ == "__main__":
    chase = build(force_refresh=True)
    print(chase.describe())

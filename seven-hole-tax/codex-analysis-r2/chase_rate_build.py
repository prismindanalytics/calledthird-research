from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_prep_r2 import CHASE_RATE_PATH, STATCAST_2025_FULL_PATH, ensure_dirs

PLATE_HALF_WIDTH_FT = 17.0 / 24.0

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "bunt_foul_tip",
    "missed_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

CONTACT_DESCRIPTIONS = {
    "foul",
    "foul_tip",
    "foul_bunt",
    "bunt_foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}


def _in_rulebook_zone(df: pd.DataFrame) -> pd.Series:
    x = pd.to_numeric(df["plate_x"], errors="coerce").astype(float)
    z = pd.to_numeric(df["plate_z"], errors="coerce").astype(float)
    top = pd.to_numeric(df["sz_top"], errors="coerce").astype(float)
    bot = pd.to_numeric(df["sz_bot"], errors="coerce").astype(float)
    return x.abs().le(PLATE_HALF_WIDTH_FT) & z.ge(bot) & z.le(top)


def build_batter_chase_rate(force: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if CHASE_RATE_PATH.exists() and not force:
        return pd.read_parquet(CHASE_RATE_PATH)
    if not STATCAST_2025_FULL_PATH.exists():
        raise FileNotFoundError(f"Missing local 2025 Statcast file: {STATCAST_2025_FULL_PATH}")

    use_cols = [
        "batter",
        "events",
        "description",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
    ]
    df = pd.read_parquet(STATCAST_2025_FULL_PATH, columns=use_cols).copy()
    df = df.dropna(subset=["batter", "description", "plate_x", "plate_z", "sz_top", "sz_bot"])
    df["batter_id"] = pd.to_numeric(df["batter"], errors="coerce").astype("Int64")
    df = df[df["batter_id"].notna()].copy()
    df["description"] = df["description"].astype(str)
    df["events"] = df["events"].fillna("").astype(str)
    df["is_pa"] = df["events"].ne("")
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)
    df["is_contact"] = df["description"].isin(CONTACT_DESCRIPTIONS)
    df["in_zone_rulebook"] = _in_rulebook_zone(df)
    df["is_out_zone"] = ~df["in_zone_rulebook"]
    df["is_chase"] = df["is_out_zone"] & df["is_swing"]
    df["is_walk"] = df["events"].isin(["walk", "intent_walk"])
    df["is_strikeout"] = df["events"].str.contains("strikeout", case=False, na=False)

    grouped = (
        df.groupby("batter_id", observed=True)
        .agg(
            pa_2025=("is_pa", "sum"),
            pitches_seen_2025=("description", "size"),
            out_zone_pitches_2025=("is_out_zone", "sum"),
            chases_2025=("is_chase", "sum"),
            swings_2025=("is_swing", "sum"),
            contacts_2025=("is_contact", "sum"),
            walks_2025=("is_walk", "sum"),
            strikeouts_2025=("is_strikeout", "sum"),
        )
        .reset_index()
    )
    grouped["chase_rate_2025"] = grouped["chases_2025"] / grouped["out_zone_pitches_2025"].replace(0, np.nan)
    grouped["walk_rate_2025"] = grouped["walks_2025"] / grouped["pa_2025"].replace(0, np.nan)
    grouped["strikeout_rate_2025"] = grouped["strikeouts_2025"] / grouped["pa_2025"].replace(0, np.nan)
    grouped["contact_rate_2025"] = grouped["contacts_2025"] / grouped["swings_2025"].replace(0, np.nan)
    grouped["eligible_200_pa"] = grouped["pa_2025"].ge(200)
    out = grouped[grouped["eligible_200_pa"]].copy()
    out = out.sort_values("chase_rate_2025").reset_index(drop=True)
    out.to_parquet(CHASE_RATE_PATH, index=False)
    return out


if __name__ == "__main__":
    built = build_batter_chase_rate(force=False)
    print(f"Wrote {len(built)} eligible batter chase-rate rows to {Path(CHASE_RATE_PATH)}")

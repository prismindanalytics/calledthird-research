"""data_prep.py — consolidate all inputs into the two analysis tables we need.

Produces (in-memory; cached to parquet on disk):

  challenges_full
    one row per ABS challenge, full corpus Mar 26 - May 4, 2026.
    Joined to lineup_spot, pitcher_fame_quartile, catcher_framing_tier, umpire (from statcast).

  taken_pitches
    one row per TAKEN pitch (description in {'called_strike', 'ball'}) Mar 27 - May 3.
    Joined to the same controls + edge_distance_ft (recomputed from sz_top/sz_bot/plate_x/plate_z).

This module exposes:
  - load_challenges() -> pd.DataFrame
  - load_taken_pitches() -> pd.DataFrame
  - count_quadrant(balls, strikes) -> str
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

CHALLENGES_OLD = Path("/Users/haohu/Documents/GitHub/calledthird/research/team-challenge-iq/data/all_challenges_detail.json")
CHALLENGES_NEW = DATA / "all_challenges_apr15_may04.json"
SC_OLD = Path("/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/data/statcast_2026_mar27_apr22.parquet")
SC_NEW = DATA / "statcast_2026_apr23_may04.parquet"
LINEUP = DATA / "batter_lineup_spot.parquet"
PITCHER_TIER = DATA / "pitcher_fame_quartile.parquet"
CATCHER_TIER = DATA / "catcher_framing_tier.parquet"

CHALLENGES_CACHE = CACHE / "challenges_full.parquet"
TAKEN_CACHE = CACHE / "taken_pitches.parquet"


# --- Helpers ---------------------------------------------------------------

def count_quadrant(balls: int | float, strikes: int | float) -> str:
    """Map (balls, strikes) -> {'hitter','pitcher','even'} buckets used in H3 stratification."""
    if pd.isna(balls) or pd.isna(strikes):
        return "even"
    b, s = int(balls), int(strikes)
    if (b, s) in {(1, 0), (2, 0), (3, 0), (2, 1), (3, 1)}:
        return "hitter"
    if (b, s) in {(0, 1), (0, 2), (1, 2)}:
        return "pitcher"
    return "even"


def pitch_group(pt: str | None) -> str:
    if pt in {"FF", "FT", "FA", "SI", "FC"}:
        return "fastball"
    if pt in {"CU", "KC", "SL", "ST", "SV", "CS"}:
        return "breaking"
    if pt in {"CH", "FS", "FO", "EP", "SC"}:
        return "offspeed"
    return "other"


def _edge_distance_ft(plate_x: pd.Series, plate_z: pd.Series, sz_top: pd.Series, sz_bot: pd.Series) -> np.ndarray:
    """Signed distance from the rulebook zone edge in feet.

    Positive = outside the zone (and the value is the distance to the nearest edge,
    i.e. how far away from the zone the pitch is).
    Negative = inside the zone (and the value is the distance from the nearest edge,
    so a pitch right on the edge is 0, and the heart of the zone is most negative).

    A pitch is "borderline" if |edge_distance_ft| <= some threshold (e.g. 0.3 ft).
    """
    half_w = 0.83
    # signed coordinates relative to each boundary; >0 means outside that boundary
    out_left = -half_w - plate_x.values            # >0 if x < -half_w
    out_right = plate_x.values - half_w            # >0 if x > +half_w
    out_top = plate_z.values - sz_top.values       # >0 if above sz_top
    out_bot = sz_bot.values - plate_z.values       # >0 if below sz_bot
    stacked = np.stack([out_left, out_right, out_top, out_bot], axis=1)
    max_breach = stacked.max(axis=1)               # >0 outside any boundary; <=0 inside zone
    # If outside (max_breach > 0): edge_distance = max_breach (distance from nearest violated boundary)
    # If inside (max_breach <= 0): edge_distance = max_breach (already negative; nearest edge dist)
    return max_breach


# --- Loaders ---------------------------------------------------------------

def load_challenges(force_refresh: bool = False) -> pd.DataFrame:
    if CHALLENGES_CACHE.exists() and not force_refresh:
        return pd.read_parquet(CHALLENGES_CACHE)

    old_rows: list[dict] = []
    if CHALLENGES_OLD.exists():
        with open(CHALLENGES_OLD) as f:
            old_rows = json.load(f)
    new_rows: list[dict] = []
    if CHALLENGES_NEW.exists():
        with open(CHALLENGES_NEW) as f:
            new_rows = json.load(f)
    df = pd.DataFrame(old_rows + new_rows)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df = df[df["game_date"].between(pd.to_datetime("2026-03-26").date(), pd.to_datetime("2026-05-04").date())].copy()
    # If umpire/home_team missing, fill from statcast
    sc_meta = _statcast_meta()
    df = df.merge(sc_meta, on="game_pk", how="left", suffixes=("", "_sc"))
    df["umpire"] = df["umpire"].where(df["umpire"].notna(), df["umpire_sc"])
    df["home_team"] = df["home_team"].where(df["home_team"].notna(), df["home_team_sc"])
    df["away_team"] = df["away_team"].where(df["away_team"].notna(), df["away_team_sc"])
    df = df.drop(columns=[c for c in ("umpire_sc", "home_team_sc", "away_team_sc") if c in df.columns])

    # If edge_distance_in is null, recompute from joining the corresponding statcast row
    sc_pitch = _statcast_pitch_keys()
    df = df.merge(
        sc_pitch[["game_pk", "at_bat_number", "pitch_number", "plate_x", "plate_z", "sz_top", "sz_bot"]]
            .rename(columns={"at_bat_number": "ab_number", "plate_x": "plate_x_sc", "plate_z": "plate_z_sc"}),
        on=["game_pk", "ab_number", "pitch_number"], how="left",
    )
    edge_dist_ft = _edge_distance_ft(
        df["plate_x_sc"].fillna(df["plate_x"]),
        df["plate_z_sc"].fillna(df["plate_z"]),
        df["sz_top"], df["sz_bot"],
    )
    df["edge_distance_ft_calc"] = edge_dist_ft
    # The challenge feed reports the ABSOLUTE distance to the closest edge in inches —
    # so it should match |edge_distance_ft_calc|*12. Use the feed value where available,
    # else fall back to the |calculated| value in inches.
    df["edge_distance_in"] = pd.to_numeric(df["edge_distance_in"], errors="coerce")
    df["edge_distance_in_calc"] = (df["edge_distance_ft_calc"] * 12).abs()
    df["edge_distance_in_final"] = df["edge_distance_in"].where(df["edge_distance_in"].notna(), df["edge_distance_in_calc"])

    # Lineup spot
    if LINEUP.exists():
        ls = pd.read_parquet(LINEUP)
        df = df.merge(ls.rename(columns={"team": "team_lineup"}), on=["game_pk", "batter_id"], how="left")
    else:
        df["lineup_spot"] = np.nan
        df["is_pinch_hitter"] = False

    # Pitcher fame and catcher framing tiers
    if PITCHER_TIER.exists():
        pt = pd.read_parquet(PITCHER_TIER)[["pitcher_id", "fame_quartile"]]
        df = df.merge(pt, on="pitcher_id", how="left")
    if CATCHER_TIER.exists():
        ct = pd.read_parquet(CATCHER_TIER)[["catcher_id", "framing_tier"]]
        df = df.merge(ct, on="catcher_id", how="left")

    # Count_state and quadrant
    df["count_state"] = df["balls"].astype("Int64").astype(str) + "-" + df["strikes"].astype("Int64").astype(str)
    df["count_quadrant"] = [count_quadrant(b, s) for b, s in zip(df["balls"], df["strikes"])]
    df["pitch_group"] = df["pitch_type"].apply(pitch_group)
    df["overturned"] = pd.to_numeric(df["overturned"], errors="coerce").fillna(0).astype(int)
    df["in_zone"] = pd.to_numeric(df["in_zone"], errors="coerce").fillna(0).astype(int)

    df["lineup_spot"] = pd.to_numeric(df["lineup_spot"], errors="coerce")
    df["is_pinch_hitter"] = df["is_pinch_hitter"].fillna(False).astype(bool)
    df["fame_quartile"] = pd.to_numeric(df["fame_quartile"], errors="coerce")
    df["framing_tier"] = df["framing_tier"].fillna("unknown")

    df.to_parquet(CHALLENGES_CACHE, index=False)
    return df


def load_taken_pitches(force_refresh: bool = False) -> pd.DataFrame:
    if TAKEN_CACHE.exists() and not force_refresh:
        return pd.read_parquet(TAKEN_CACHE)

    cols = [
        "game_pk", "game_date", "at_bat_number", "pitch_number", "batter", "pitcher",
        "fielder_2", "stand", "p_throws", "balls", "strikes", "plate_x", "plate_z",
        "sz_top", "sz_bot", "description", "pitch_type", "home_team", "away_team",
    ]
    frames = []
    for p in (SC_OLD, SC_NEW):
        if p.exists():
            df = pd.read_parquet(p, columns=cols)
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["description"].isin(["called_strike", "ball"])].copy()
    df = df.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot", "batter"])
    df["batter"] = df["batter"].astype(int)
    df["pitcher"] = df["pitcher"].astype(int)
    df["fielder_2"] = df["fielder_2"].astype(int)
    df["game_pk"] = df["game_pk"].astype(int)
    df["called_strike"] = (df["description"] == "called_strike").astype(int)
    df["edge_distance_ft"] = _edge_distance_ft(df["plate_x"], df["plate_z"], df["sz_top"], df["sz_bot"])

    # Umpire from boxscore
    meta = _statcast_meta()[["game_pk", "umpire"]]
    df = df.merge(meta, on="game_pk", how="left")

    # Lineup spot
    if LINEUP.exists():
        ls = pd.read_parquet(LINEUP).rename(columns={"batter_id": "batter"})
        df = df.merge(ls.drop(columns=["team"]), on=["game_pk", "batter"], how="left")
    else:
        df["lineup_spot"] = np.nan
        df["is_pinch_hitter"] = False

    if PITCHER_TIER.exists():
        pt = pd.read_parquet(PITCHER_TIER)[["pitcher_id", "fame_quartile"]].rename(columns={"pitcher_id": "pitcher"})
        df = df.merge(pt, on="pitcher", how="left")
    if CATCHER_TIER.exists():
        ct = pd.read_parquet(CATCHER_TIER)[["catcher_id", "framing_tier"]].rename(columns={"catcher_id": "fielder_2"})
        df = df.merge(ct, on="fielder_2", how="left")

    df["count_state"] = df["balls"].astype("Int64").astype(str) + "-" + df["strikes"].astype("Int64").astype(str)
    df["count_quadrant"] = [count_quadrant(b, s) for b, s in zip(df["balls"], df["strikes"])]
    df["pitch_group"] = df["pitch_type"].apply(pitch_group)
    df["lineup_spot"] = pd.to_numeric(df["lineup_spot"], errors="coerce")
    df["is_pinch_hitter"] = df["is_pinch_hitter"].fillna(False).astype(bool)
    df["fame_quartile"] = pd.to_numeric(df["fame_quartile"], errors="coerce")
    df["framing_tier"] = df["framing_tier"].fillna("unknown")
    # Drop unmapped lineup spots and umpire NaNs (small fraction)
    df = df.dropna(subset=["lineup_spot", "umpire"]).copy()
    df["lineup_spot"] = df["lineup_spot"].astype(int)

    df.to_parquet(TAKEN_CACHE, index=False)
    return df


def _statcast_meta() -> pd.DataFrame:
    """Per-game metadata: umpire (from MLB boxscore), home_team, away_team (from statcast)."""
    cache = CACHE / "statcast_meta.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for p in (SC_OLD, SC_NEW):
        if p.exists():
            df = pd.read_parquet(p, columns=["game_pk", "home_team", "away_team"])
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["game_pk"]).drop_duplicates(subset=["game_pk"])
    df["game_pk"] = df["game_pk"].astype(int)
    # Statcast 'umpire' column is empty in pybaseball pulls — pull umpire from cached boxscores
    ump_path = DATA / "game_umpire.parquet"
    if ump_path.exists():
        ump = pd.read_parquet(ump_path)
        ump["game_pk"] = ump["game_pk"].astype(int)
        df = df.merge(ump, on="game_pk", how="left")
    else:
        df["umpire"] = None
    df.to_parquet(cache, index=False)
    return df


def _statcast_pitch_keys() -> pd.DataFrame:
    cache = CACHE / "statcast_pitch_keys.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    for p in (SC_OLD, SC_NEW):
        if p.exists():
            df = pd.read_parquet(p, columns=["game_pk", "at_bat_number", "pitch_number", "plate_x", "plate_z", "sz_top", "sz_bot"])
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["game_pk", "at_bat_number", "pitch_number"])
    df["game_pk"] = df["game_pk"].astype(int)
    df["at_bat_number"] = df["at_bat_number"].astype(int)
    df["pitch_number"] = df["pitch_number"].astype(int)
    df.to_parquet(cache, index=False)
    return df


if __name__ == "__main__":
    ch = load_challenges(force_refresh=True)
    print(f"challenges_full: {len(ch):,} rows, {ch['lineup_spot'].notna().sum():,} have lineup_spot")
    print(ch[["lineup_spot", "is_pinch_hitter", "overturned", "fame_quartile", "framing_tier"]].head())
    print("Lineup spot distribution among challenges:")
    print(ch["lineup_spot"].value_counts(dropna=False).sort_index())
    tk = load_taken_pitches(force_refresh=True)
    print(f"taken_pitches: {len(tk):,}")
    print("Borderline (|edge|<0.3 ft):", (tk["edge_distance_ft"].abs() < 0.3).sum())

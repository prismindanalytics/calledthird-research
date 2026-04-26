"""r2_data_pull.py — R2 data extension + reproducibility fixes.

Fixes from R1 cross-review:
  1. MLB Stats API resolver for any 2026 debut not in playerid_lookup (Murakami).
     Cached to data/mlbam_resolver_cache.json.
  2. Extends 2026 statcast through 2026-04-24 (one extra day vs R1 which stopped
     at 2026-04-23). Per the brief we may try 2026-04-25 too; only pull if games
     for that date are complete.
  3. Builds a single unified 2026 PA-level parquet via deduplication.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
DATA.mkdir(parents=True, exist_ok=True)

MLBAM_CACHE = DATA / "mlbam_resolver_cache.json"

STATCAST_COLS = [
    "game_date", "game_pk", "game_year", "game_type",
    "batter", "pitcher", "player_name", "stand", "p_throws",
    "events", "description", "type", "bb_type",
    "balls", "strikes", "at_bat_number",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "estimated_woba_using_speedangle", "estimated_slg_using_speedangle",
    "iso_value", "babip_value", "woba_value", "woba_denom",
    "home_team", "away_team", "inning_topbot", "inning",
]


def _load_cache() -> dict:
    if MLBAM_CACHE.exists():
        try:
            return json.load(open(MLBAM_CACHE))
        except Exception:
            return {}
    return {}


def _save_cache(d: dict) -> None:
    MLBAM_CACHE.write_text(json.dumps(d, indent=2, sort_keys=True))


def resolve_mlbam(name: str, *, prefer_active: bool = True) -> Optional[int]:
    """Resolve a full player name to MLBAM id using MLB Stats API.

    Tries pybaseball.playerid_lookup first; on miss, falls back to
    https://statsapi.mlb.com/api/v1/people/search?names={name}.
    Cache results in data/mlbam_resolver_cache.json so a clean checkout will
    regenerate Murakami's id without manual editing.
    """
    cache = _load_cache()
    if name in cache and cache[name].get("mlbam"):
        return int(cache[name]["mlbam"])

    # First: pybaseball
    parts = name.strip().split()
    if len(parts) >= 2:
        first, last = parts[0], " ".join(parts[1:])
        try:
            from pybaseball import playerid_lookup
            r = playerid_lookup(last, first)
            if r is not None and len(r) > 0:
                r = r.copy()
                r["mlb_played_last"] = pd.to_numeric(r["mlb_played_last"], errors="coerce")
                r = r.sort_values("mlb_played_last", ascending=False).iloc[0]
                if pd.notna(r["key_mlbam"]):
                    mlbam = int(r["key_mlbam"])
                    cache[name] = {"mlbam": mlbam, "source": "pybaseball"}
                    _save_cache(cache)
                    return mlbam
        except Exception as e:
            print(f"  [pybaseball-fail] {name}: {e}")

    # Second: MLB Stats API
    try:
        url = "https://statsapi.mlb.com/api/v1/people/search"
        r = requests.get(url, params={"names": name}, timeout=15)
        r.raise_for_status()
        data = r.json()
        people = data.get("people", [])
        if people:
            chosen = None
            if prefer_active:
                for p in people:
                    if p.get("active"):
                        chosen = p
                        break
            if chosen is None:
                chosen = people[0]
            mlbam = int(chosen["id"])
            cache[name] = {"mlbam": mlbam, "source": "mlb_stats_api",
                           "fullName": chosen.get("fullName"),
                           "primaryPosition": chosen.get("primaryPosition", {}).get("abbreviation")}
            _save_cache(cache)
            print(f"  [mlb-api] {name} -> {mlbam} ({chosen.get('fullName')})")
            return mlbam
    except Exception as e:
        print(f"  [mlb-api-fail] {name}: {e}")

    cache[name] = {"mlbam": None, "source": "miss"}
    _save_cache(cache)
    return None


def _slim(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in STATCAST_COLS if c in df.columns]
    return df[keep].copy()


def extend_2026_to_apr25() -> Path:
    """Try to extend 2026 statcast through 2026-04-25.

    If 2026-04-25 not available, returns the path with through_apr24 contents.
    """
    out_25 = DATA / "statcast_2026_apr24_25.parquet"
    if not out_25.exists():
        from pybaseball import statcast
        try:
            df = statcast(start_dt="2026-04-24", end_dt="2026-04-25")
            df = _slim(df)
            df.to_parquet(out_25, index=False)
            print(f"[saved] 2026-04-24/25: {out_25.name} ({len(df):,} rows, dates {sorted(df['game_date'].astype(str).unique())})")
        except Exception as e:
            print(f"[fail] extend_2026_to_apr25: {e}")
            return DATA / "statcast_2026_through_apr24.parquet"

    # Build unified through_apr25 file
    out_full = DATA / "statcast_2026_through_apr25.parquet"
    if not out_full.exists():
        # Read existing through_apr24 and combine with new
        through_24 = pd.read_parquet(DATA / "statcast_2026_through_apr24.parquet")
        new = pd.read_parquet(out_25)
        # Align cols
        cols_common = [c for c in STATCAST_COLS if c in through_24.columns and c in new.columns]
        # Add cols missing from one side as NaN
        all_cols = sorted(set(through_24.columns) | set(new.columns))
        for c in all_cols:
            if c not in through_24.columns:
                through_24[c] = pd.NA
            if c not in new.columns:
                new[c] = pd.NA
        # Coerce game_date to common dtype
        for d in (through_24, new):
            if "game_date" in d.columns:
                d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce")
        # Subset to columns common to both
        cols_both = [c for c in STATCAST_COLS if c in through_24.columns and c in new.columns]
        df = pd.concat([through_24[cols_both], new[cols_both]], ignore_index=True)
        # Dedupe
        if all(c in df.columns for c in ["game_pk", "batter", "pitcher", "at_bat_number", "balls", "strikes"]):
            df = df.drop_duplicates(subset=["game_pk", "batter", "pitcher", "at_bat_number", "balls", "strikes"]).reset_index(drop=True)
        df.to_parquet(out_full, index=False)
        print(f"[saved] {out_full.name}: {len(df):,} rows, dates {sorted(df['game_date'].astype(str).unique())}")
    return out_full


def resolve_named() -> pd.DataFrame:
    """Resolve named hot starters with the new MLB Stats API fallback.

    Always writes data/named_hot_starters_r2.parquet so a clean checkout works.
    """
    out = DATA / "named_hot_starters_r2.parquet"
    rows = []
    for slug, first, last, role in [
        ("andy_pages", "Andy", "Pages", "hitter"),
        ("ben_rice", "Ben", "Rice", "hitter"),
        ("munetaka_murakami", "Munetaka", "Murakami", "hitter"),
        ("mike_trout", "Mike", "Trout", "hitter"),
        ("mason_miller", "Mason", "Miller", "reliever"),
    ]:
        full = f"{first} {last}"
        mlbam = resolve_mlbam(full)
        rows.append({
            "slug": slug, "first": first, "last": last, "role": role, "mlbam": mlbam,
            "name": full,
        })
        print(f"[lookup] {full} -> mlbam {mlbam}")
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    return df


def main():
    print("[r2_data_pull] starting")
    extend_2026_to_apr25()
    resolve_named()
    print("[r2_data_pull] done")


if __name__ == "__main__":
    main()

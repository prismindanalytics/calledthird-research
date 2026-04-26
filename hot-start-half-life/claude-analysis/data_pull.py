"""data_pull.py — fetch + cache all data needed for the Hot-Start Half-Life analysis.

Idempotent: every output file is only fetched if not already on disk.

Caches under research/hot-start-half-life/data/:
  - statcast_2026_apr23_24.parquet            (extends 2026 to Apr 24)
  - statcast_2022.parquet, statcast_2023.parquet, statcast_2024.parquet
  - batting_stats_<season>.parquet  for 2015-2025
  - pitching_stats_<season>.parquet for 2015-2025
  - playerid_lookup.parquet                  (resolved IDs for 5 named hot starters)
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path
from datetime import date, timedelta

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path("<home>/Documents/GitHub/calledthird")
DATA_DIR = REPO_ROOT / "research/hot-start-half-life/data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Existing cached corpora — symlinks live in pitch-tunneling-atlas/data
EXISTING_2025 = REPO_ROOT / "research/pitch-tunneling-atlas/data/statcast_2025_full.parquet"
EXISTING_2026 = REPO_ROOT / "research/abs-walk-spike/data/statcast_2026_mar27_apr22.parquet"

# Slim Statcast columns we keep — keeps each season cache near 1-2 GB
STATCAST_COLS = [
    "game_date", "game_pk", "game_year", "game_type",
    "batter", "pitcher", "player_name", "stand", "p_throws",
    "events", "description", "type", "bb_type",
    "balls", "strikes", "at_bat_number",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "estimated_woba_using_speedangle",
    "iso_value", "babip_value", "woba_value", "woba_denom",
    "home_team", "away_team", "inning_topbot",
]


def _save(df: pd.DataFrame, out: Path, msg: str) -> None:
    df.to_parquet(out, index=False)
    print(f"[saved] {msg}: {out.name}  ({len(df):,} rows, {df.memory_usage(deep=True).sum()/1e6:.1f} MB)")


def _slim(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in STATCAST_COLS if c in df.columns]
    return df[keep].copy()


# ---------------- Statcast extension / backfill ----------------

def extend_2026() -> Path:
    out = DATA_DIR / "statcast_2026_apr23_24.parquet"
    if out.exists():
        print(f"[skip] {out.name} already cached")
        return out
    from pybaseball import statcast
    print("[fetch] Statcast 2026-04-23 to 2026-04-24")
    df = statcast(start_dt="2026-04-23", end_dt="2026-04-24")
    df = _slim(df)
    _save(df, out, "Statcast 2026 Apr 23-24")
    return out


def backfill_season(year: int) -> Path:
    out = DATA_DIR / f"statcast_{year}.parquet"
    if out.exists():
        print(f"[skip] {out.name} already cached")
        return out

    from pybaseball import statcast
    print(f"[fetch] Statcast {year} (monthly chunks)")
    chunks = []
    # Regular season approx Mar 28 - Oct 5
    months = [
        (f"{year}-03-28", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-10-05"),
    ]
    for s, e in months:
        attempts = 0
        while attempts < 3:
            try:
                t0 = time.time()
                d = statcast(start_dt=s, end_dt=e)
                if d is not None and len(d) > 0:
                    d = _slim(d)
                    chunks.append(d)
                    print(f"  [chunk] {s} to {e}: {len(d):,} rows  ({time.time()-t0:.1f}s)")
                else:
                    print(f"  [chunk] {s} to {e}: empty")
                break
            except Exception as ex:
                attempts += 1
                print(f"  [retry {attempts}] {s} to {e}: {ex}")
                time.sleep(5)
        else:
            print(f"  [give up] {s} to {e}")
    if not chunks:
        raise RuntimeError(f"Statcast {year}: no data fetched")
    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "balls", "strikes", "pitcher", "batter"]).reset_index(drop=True)
    _save(df, out, f"Statcast {year}")
    return out


# ---------------- batting/pitching stats ----------------

def fetch_batting_stats(year: int) -> Path | None:
    out = DATA_DIR / f"batting_stats_{year}.parquet"
    if out.exists():
        print(f"[skip] {out.name} already cached")
        return out
    from pybaseball import batting_stats
    try:
        print(f"[fetch] batting_stats {year}")
        df = batting_stats(year, qual=1)
        df["season"] = year
        _save(df, out, f"batting_stats {year}")
        return out
    except Exception as ex:
        print(f"[warn] batting_stats {year} failed ({type(ex).__name__}): "
              f"{str(ex)[:120]} — continuing; we'll use Statcast aggregates instead")
        return None


def fetch_pitching_stats(year: int) -> Path | None:
    out = DATA_DIR / f"pitching_stats_{year}.parquet"
    if out.exists():
        print(f"[skip] {out.name} already cached")
        return out
    from pybaseball import pitching_stats
    try:
        print(f"[fetch] pitching_stats {year}")
        df = pitching_stats(year, qual=1)
        df["season"] = year
        _save(df, out, f"pitching_stats {year}")
        return out
    except Exception as ex:
        print(f"[warn] pitching_stats {year} failed ({type(ex).__name__}): "
              f"{str(ex)[:120]} — continuing")
        return None


# ---------------- Player ID lookups ----------------

NAMED_HOT_STARTERS = [
    {"slug": "andy_pages",     "first": "Andy",      "last": "Pages",     "role": "hitter"},
    {"slug": "ben_rice",       "first": "Ben",       "last": "Rice",      "role": "hitter"},
    {"slug": "munetaka_murakami","first": "Munetaka","last": "Murakami",  "role": "hitter"},
    {"slug": "mike_trout",     "first": "Mike",      "last": "Trout",     "role": "hitter"},
    {"slug": "mason_miller",   "first": "Mason",     "last": "Miller",    "role": "reliever"},
]


def lookup_named() -> Path:
    out = DATA_DIR / "named_hot_starters.parquet"
    if out.exists():
        print(f"[skip] {out.name} already cached")
        return out
    from pybaseball import playerid_lookup
    rows = []
    for p in NAMED_HOT_STARTERS:
        try:
            r = playerid_lookup(p["last"], p["first"])
        except Exception as ex:
            print(f"[lookup-fail] {p['first']} {p['last']}: {ex}")
            continue
        if r is None or len(r) == 0:
            print(f"[lookup-empty] {p['first']} {p['last']}")
            continue
        # Take the active row (most-recent mlb_played_last)
        r = r.copy()
        r["mlb_played_last"] = pd.to_numeric(r["mlb_played_last"], errors="coerce")
        r = r.sort_values("mlb_played_last", ascending=False).iloc[0]
        rows.append({
            "slug": p["slug"], "first": p["first"], "last": p["last"], "role": p["role"],
            "mlbam": int(r["key_mlbam"]) if pd.notna(r["key_mlbam"]) else None,
            "fangraphs": str(r["key_fangraphs"]) if pd.notna(r["key_fangraphs"]) else None,
            "bbref": str(r["key_bbref"]) if pd.notna(r["key_bbref"]) else None,
        })
        print(f"[lookup] {p['first']} {p['last']} -> mlbam {rows[-1]['mlbam']}")
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    return out


# ---------------- Driver ----------------

def main():
    print(f"DATA_DIR = {DATA_DIR}")

    # Pre-existing corpora — symlink into data/ for one place
    for src, name in [
        (EXISTING_2025, "statcast_2025_full.parquet"),
        (EXISTING_2026, "statcast_2026_mar27_apr22.parquet"),
    ]:
        link = DATA_DIR / name
        if not link.exists():
            link.symlink_to(src)
            print(f"[link] {name} -> {src}")

    extend_2026()

    for y in (2022, 2023, 2024):
        backfill_season(y)

    for y in range(2015, 2026):  # 2015..2025
        fetch_batting_stats(y)
        fetch_pitching_stats(y)

    lookup_named()

    print("[done] data_pull complete")


if __name__ == "__main__":
    main()

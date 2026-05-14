#!/usr/bin/env python3
"""Pull Statcast pitch-level data for Apr 23 - May 4, 2026 via pybaseball.

Output: data/statcast_2026_apr23_may04.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pybaseball import statcast

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = OUT_DIR / "statcast_2026_apr23_may04.parquet"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching Statcast 2026-04-23 .. 2026-05-04 ...")
    df = statcast(start_dt="2026-04-23", end_dt="2026-05-04")
    if df is None or df.empty:
        print("ERROR: pybaseball returned no data", file=sys.stderr)
        return 1
    print(f"  rows={len(df):,}  games={df['game_pk'].nunique()}  date range={df['game_date'].min()}..{df['game_date'].max()}")
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

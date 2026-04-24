"""Build a single 2026 parquet for Mar 27 – Apr 22.

Sources:
  - Daily files Mar 27 – Apr 5 from research/count-distribution-abs/data/
  - Aggregate Apr 6 – Apr 14 from research/count-distribution-abs/data/
  - Pull from pybaseball: Apr 15 – Apr 22

Output: research/abs-walk-spike/data/statcast_2026_mar27_apr22.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pybaseball import statcast

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "research" / "count-distribution-abs" / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT = OUT_DIR / "statcast_2026_mar27_apr22.parquet"


def load_existing() -> pd.DataFrame:
    frames = []
    for date in pd.date_range("2026-03-27", "2026-04-05"):
        f = SRC_DIR / f"{date.date()}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
        else:
            print(f"  WARN: missing {f.name}", file=sys.stderr)
    agg = SRC_DIR / "statcast_2026_apr06_14.parquet"
    if agg.exists():
        frames.append(pd.read_parquet(agg))
    else:
        print(f"  WARN: missing {agg.name}", file=sys.stderr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_recent() -> pd.DataFrame:
    print("Fetching 2026-04-15 to 2026-04-22 from Statcast...")
    df = statcast(start_dt="2026-04-15", end_dt="2026-04-22")
    print(f"  pulled {len(df):,} rows")
    return df


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_existing()
    print(f"Existing rows (Mar 27 – Apr 14): {len(existing):,}")
    if existing.empty:
        print("ERROR: no existing 2026 data — aborting", file=sys.stderr)
        return 1

    recent = fetch_recent()
    if recent.empty:
        print("ERROR: pybaseball returned no rows for Apr 15–22", file=sys.stderr)
        return 1

    common = sorted(set(existing.columns) & set(recent.columns))
    if len(common) < 50:
        print(f"ERROR: schema diverged ({len(common)} common cols)", file=sys.stderr)
        return 1
    full = pd.concat([existing[common], recent[common]], ignore_index=True)
    full["game_date"] = pd.to_datetime(full["game_date"])
    full = full.sort_values("game_date").reset_index(drop=True)
    print(
        f"\nFinal: {len(full):,} rows, {full['game_date'].min().date()} → "
        f"{full['game_date'].max().date()}, {len(common)} columns"
    )
    full.to_parquet(OUT, index=False)
    print(f"WROTE {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

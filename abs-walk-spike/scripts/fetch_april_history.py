"""Compute Mar 27 – Apr 22 plate-appearance and walk counts per year, 2018–2025.

Apples-to-apples with the 2026 data window we have (Mar 27 – Apr 22).
Walks include intentional walks to match ESPN/MLB headline definition.

Used to build the seasonality baseline (H2). Pulls Statcast pitch-by-pitch
per year, immediately reduces to PA-level events, aggregates, drops
the raw data. Keeps memory bounded.

Output: research/abs-walk-spike/data/april_walk_history.csv
        Columns: year, season_pas, season_walks_incl_ibb, season_walks_excl_ibb,
                 walk_rate_incl_ibb, walk_rate_excl_ibb
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pybaseball import statcast

YEARS = list(range(2018, 2026))  # 2018..2025
OUT_DIR = Path(__file__).resolve().parents[1] / "data"
OUT = OUT_DIR / "april_walk_history.csv"


def april_rate(year: int) -> dict | None:
    print(f"\n=== {year} Mar 27 – Apr 22 ===")
    try:
        df = statcast(start_dt=f"{year}-03-27", end_dt=f"{year}-04-22")
    except Exception as e:
        print(f"  ERROR fetching {year}: {e}", file=sys.stderr)
        return None
    if df.empty:
        print(f"  no rows for {year}", file=sys.stderr)
        return None

    # Each plate-appearance terminates with one pitch where `events` is non-null.
    pa_terminating = df[df["events"].notna() & (df["events"] != "")]
    n_pa = len(pa_terminating)
    n_bb_excl = int((pa_terminating["events"] == "walk").sum())
    n_bb_incl = int(pa_terminating["events"].isin(["walk", "intent_walk"]).sum())
    rate_excl = n_bb_excl / n_pa if n_pa else float("nan")
    rate_incl = n_bb_incl / n_pa if n_pa else float("nan")
    print(
        f"  PA={n_pa:,}  BB(excl IBB)={n_bb_excl:,} ({rate_excl*100:.2f}%)  "
        f"BB(incl IBB)={n_bb_incl:,} ({rate_incl*100:.2f}%)"
    )
    return {
        "year": year,
        "season_pas": n_pa,
        "season_walks_excl_ibb": n_bb_excl,
        "season_walks_incl_ibb": n_bb_incl,
        "walk_rate_excl_ibb": rate_excl,
        "walk_rate_incl_ibb": rate_incl,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in YEARS:
        result = april_rate(year)
        if result is None:
            print(f"  skipping {year}", file=sys.stderr)
            continue
        rows.append(result)

    if not rows:
        print("ERROR: no years pulled", file=sys.stderr)
        return 1

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nWROTE {OUT}")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

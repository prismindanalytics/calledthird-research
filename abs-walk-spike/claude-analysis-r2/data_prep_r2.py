"""Round 2 data prep.

1. Pull Statcast Apr 23 – May 13, 2026 via pybaseball and save as
   `claude-analysis-r2/data/statcast_2026_apr23_may13.parquet`.
2. Use the `set(2025_full.columns) & set(R1.columns) & set(new.columns)` rule to
   ensure cross-season schema compatibility.
3. Build a weekly aggregates table for the full Mar 27 – May 13 window for both
   2025 (same window) and 2026.

Run this once; the parquet is reused by H1–H5.

The Statcast pipeline backs out to Apr 23 (one day after the Round 1 cutoff of Apr 22).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from common import (
    PARQUET_2026_R1,
    PARQUET_2026_R2,
    PARQUET_2025_FULL,
    R2_DATA,
    SAFE_WINDOW_END_DAYS,
    WALK_EVENTS,
    WINDOW_END,
    WINDOW_START,
    count_state,
    ensure_dirs,
    plate_appearance_mask,
    week_index_2026,
)


def fetch_2026_apr23_may13() -> pd.DataFrame:
    from pybaseball import statcast

    end = WINDOW_END.strftime("%Y-%m-%d")
    print(f"[data_prep_r2] Fetching Statcast 2026-04-23 .. {end} ...", flush=True)
    df = statcast(start_dt="2026-04-23", end_dt=end)
    print(f"[data_prep_r2] pulled {len(df):,} rows", flush=True)
    return df


def prune_recent_partial(df: pd.DataFrame) -> pd.DataFrame:
    """Statcast has a publishing lag for the most recent day.

    Drop the last day if its PA count is below 30 percent of the median of the
    prior week to avoid dragging walk rate.
    """
    if df.empty:
        return df
    d = pd.to_datetime(df["game_date"]).dt.normalize()
    by_day = df.assign(date=d).groupby("date").size()
    if len(by_day) < 3:
        return df
    median_prior = by_day.iloc[:-1].median()
    last_day = by_day.index[-1]
    last_count = by_day.iloc[-1]
    if last_count < 0.3 * median_prior:
        print(
            f"[data_prep_r2] dropping {last_day.date()} as partial (n={last_count} vs median {median_prior:.0f})",
            flush=True,
        )
        return df.loc[d != last_day].copy()
    return df


def build_weekly_aggregates() -> pd.DataFrame:
    """Per-week PA / walk counts for 2025 (Apr-May same window) and 2026 full window."""
    from common import load_2025_samewindow, load_2026_full

    rows = []
    for year, df_loader in [(2025, load_2025_samewindow), (2026, load_2026_full)]:
        df = df_loader()
        pa = df.loc[plate_appearance_mask(df)].copy()
        # Restrict to Mar 27 – May 13 windows
        d = pd.to_datetime(pa["game_date"]).dt.normalize()
        if year == 2025:
            mask = (d >= "2025-03-27") & (d <= "2025-05-13")
            anchor = pd.Timestamp("2025-03-27")
        else:
            mask = (d >= "2026-03-27") & (d <= "2026-05-13")
            anchor = WINDOW_START
        pa = pa.loc[mask].copy()
        pa["week"] = week_index_2026(pa["game_date"], anchor=anchor)
        pa["is_walk"] = pa["events"].isin(WALK_EVENTS).astype(int)
        agg = (
            pa.groupby("week")
            .agg(n_pa=("is_walk", "size"), n_walk=("is_walk", "sum"))
            .reset_index()
        )
        agg["year"] = year
        agg["walk_rate"] = agg["n_walk"] / agg["n_pa"]
        rows.append(agg)
    weekly = pd.concat(rows, ignore_index=True)[
        ["year", "week", "n_pa", "n_walk", "walk_rate"]
    ]
    return weekly


def main() -> int:
    ensure_dirs()

    if PARQUET_2026_R2.exists():
        print(f"[data_prep_r2] {PARQUET_2026_R2.name} exists; skipping fetch", flush=True)
        new = pd.read_parquet(PARQUET_2026_R2)
        print(f"[data_prep_r2] loaded {len(new):,} rows from cache", flush=True)
    else:
        new = fetch_2026_apr23_may13()
        if new.empty:
            print("[data_prep_r2] ERROR: pybaseball returned no rows", file=sys.stderr)
            return 1
        new = prune_recent_partial(new)

        r1 = pd.read_parquet(PARQUET_2026_R1)
        full_2025 = pd.read_parquet(PARQUET_2025_FULL)
        common = sorted(set(r1.columns) & set(new.columns) & set(full_2025.columns))
        print(f"[data_prep_r2] schema intersection: {len(common)} cols", flush=True)
        if len(common) < 50:
            print("[data_prep_r2] schema diverged too much; aborting", file=sys.stderr)
            return 1
        new = new[common]
        new["game_date"] = pd.to_datetime(new["game_date"])
        new = new.sort_values("game_date").reset_index(drop=True)
        new.to_parquet(PARQUET_2026_R2, index=False)
        print(
            f"[data_prep_r2] wrote {PARQUET_2026_R2} ({len(new):,} rows, "
            f"{new['game_date'].min().date()} → {new['game_date'].max().date()})",
            flush=True,
        )

    # Build weekly aggregates
    weekly = build_weekly_aggregates()
    out_weekly = R2_DATA / "weekly_aggregates.parquet"
    weekly.to_parquet(out_weekly, index=False)
    print(f"[data_prep_r2] wrote weekly aggregates: {out_weekly}", flush=True)
    print(weekly.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

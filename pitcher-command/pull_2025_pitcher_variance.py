#!/usr/bin/env python3
"""Pull 2025 full-season Statcast data and compute pitcher command variance.

For each pitcher outing (30+ pitches), computes:
- Scatter (combined std of plate_x, plate_z) for first/middle/last third
- Extreme miss rates (P90, P95, P99 of distance from pitcher's own mean)
- Per-pitch-type location variance
- Overall distribution stats

Outputs per-outing and per-pitcher aggregate data.

Usage:
    python pull_2025_pitcher_variance.py
"""

import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pybaseball import statcast

OUTPUT_DIR = Path(__file__).parent / "data"


def compute_outing_variance(group: pd.DataFrame, name: str, game_pk: int) -> dict:
    """Compute variance metrics for a single pitcher outing."""
    n = len(group)
    if n < 30:
        return None

    third = n // 3
    early = group.iloc[:third]
    late = group.iloc[-third:]

    # Overall scatter
    early_scatter = np.sqrt(early['plate_x'].std()**2 + early['plate_z'].std()**2) * 12
    late_scatter = np.sqrt(late['plate_x'].std()**2 + late['plate_z'].std()**2) * 12
    change = late_scatter - early_scatter
    change_pct = (change / early_scatter * 100) if early_scatter > 0 else 0

    # Distance from own mean (per pitch type)
    dists = []
    for pt, pt_group in group.groupby('pitch_type'):
        if len(pt_group) < 3:
            continue
        mx = pt_group['plate_x'].mean()
        mz = pt_group['plate_z'].mean()
        for _, row in pt_group.iterrows():
            d = math.sqrt((row['plate_x'] - mx)**2 + (row['plate_z'] - mz)**2) * 12
            dists.append((row.name, d))

    if not dists:
        return None

    # Sort by original order to split early/late
    dists.sort(key=lambda x: x[0])
    all_d = [d for _, d in dists]
    early_d = all_d[:len(all_d)//3]
    late_d = all_d[-(len(all_d)//3):]

    return {
        'name': name,
        'game_pk': int(game_pk),
        'pitches': n,
        'early_scatter': round(early_scatter, 2),
        'late_scatter': round(late_scatter, 2),
        'change': round(change, 2),
        'change_pct': round(change_pct, 1),
        'early_p90': round(np.percentile(early_d, 90), 2) if early_d else 0,
        'late_p90': round(np.percentile(late_d, 90), 2) if late_d else 0,
        'early_p95': round(np.percentile(early_d, 95), 2) if early_d else 0,
        'late_p95': round(np.percentile(late_d, 95), 2) if late_d else 0,
        'overall_std': round(np.std(all_d), 2),
        'pct_gt_6in': round(sum(1 for d in all_d if d > 6) / len(all_d) * 100, 1),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_outings = []
    total_pitches = 0

    # Pull 2025 in 2-week chunks (April - September)
    start = date(2025, 3, 27)
    end = date(2025, 9, 30)
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=13), end)
        print(f"Pulling {current} to {chunk_end}...")

        try:
            df = statcast(
                start_dt=current.strftime("%Y-%m-%d"),
                end_dt=chunk_end.strftime("%Y-%m-%d"),
            )
        except Exception as e:
            print(f"  Error: {e}")
            current = chunk_end + timedelta(days=1)
            continue

        if df is None or len(df) == 0:
            current = chunk_end + timedelta(days=1)
            continue

        # Filter to pitches with location data
        df = df[df['plate_x'].notna() & df['plate_z'].notna()].copy()
        print(f"  {len(df):,} pitches with location")
        total_pitches += len(df)

        # Sort for proper ordering
        df = df.sort_values(['game_pk', 'pitcher', 'at_bat_number', 'pitch_number'])
        df['pitcher_pitch_num'] = df.groupby(['game_pk', 'pitcher']).cumcount() + 1

        # Process each outing
        for (game_pk, pitcher), group in df.groupby(['game_pk', 'pitcher']):
            name = group['player_name'].iloc[0] if 'player_name' in group.columns else str(pitcher)
            result = compute_outing_variance(group, name, game_pk)
            if result:
                all_outings.append(result)

        current = chunk_end + timedelta(days=1)

    print(f"\n=== RESULTS ===")
    print(f"Total pitches processed: {total_pitches:,}")
    print(f"Outings (30+ pitches): {len(all_outings):,}")

    # Save raw outings
    out_path = OUTPUT_DIR / "pitcher_variance_2025.json"
    out_path.write_text(json.dumps(all_outings, indent=2))
    print(f"Saved to {out_path}")

    # Compute aggregates
    df = pd.DataFrame(all_outings)

    print(f"\n=== DISTRIBUTION ===")
    print(f"Mean change: {df['change_pct'].mean():+.1f}%")
    print(f"Median change: {df['change_pct'].median():+.1f}%")
    consistent = (df['change_pct'].abs() <= 10).mean() * 100
    big_increase = (df['change_pct'] > 20).mean() * 100
    big_decrease = (df['change_pct'] < -20).mean() * 100
    print(f"Consistent (<10% change): {consistent:.0f}%")
    print(f"Big increase (>20%): {big_increase:.0f}%")
    print(f"Big decrease (>20%): {big_decrease:.0f}%")

    # Per-pitcher aggregates (10+ outings)
    pitcher_agg = df.groupby('name').agg(
        outings=('game_pk', 'count'),
        avg_change=('change', 'mean'),
        avg_change_pct=('change_pct', 'mean'),
        avg_early=('early_scatter', 'mean'),
        avg_late=('late_scatter', 'mean'),
        blowup_rate=('change_pct', lambda x: (x > 20).mean() * 100),
    ).reset_index()
    pitcher_agg = pitcher_agg[pitcher_agg['outings'] >= 10].sort_values('avg_change_pct', ascending=False)

    agg_path = OUTPUT_DIR / "pitcher_variance_2025_aggregates.json"
    pitcher_agg.to_json(agg_path, orient='records', indent=2)
    print(f"\nPitchers with 10+ outings: {len(pitcher_agg)}")
    print(f"Saved aggregates to {agg_path}")

    print(f"\nTop 10 biggest scatter increase:")
    for _, r in pitcher_agg.head(10).iterrows():
        print(f"  {r['name']:25s} {r['outings']:3.0f} starts | Avg: {r['avg_change_pct']:+.1f}% | Blowup rate: {r['blowup_rate']:.0f}%")

    print(f"\nMost consistent (tightest late):")
    for _, r in pitcher_agg.tail(10).iterrows():
        print(f"  {r['name']:25s} {r['outings']:3.0f} starts | Avg: {r['avg_change_pct']:+.1f}% | Blowup rate: {r['blowup_rate']:.0f}%")


if __name__ == "__main__":
    main()

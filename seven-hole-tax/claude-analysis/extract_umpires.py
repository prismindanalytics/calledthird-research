#!/usr/bin/env python3
"""Extract per-game home-plate umpire from cached MLB live boxscore JSON.

Output: data/game_umpire.parquet  columns: game_pk, umpire
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache" / "boxscores"
OUT = ROOT / "data" / "game_umpire.parquet"


def main() -> int:
    rows: list[dict] = []
    for p in sorted(CACHE.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        game_pk = int(p.stem)
        # Officials are at liveData.boxscore.officials -> [{"officialType": "Home Plate", "official": {...}}]
        live = data.get("liveData") or {}
        boxscore = live.get("boxscore") or {}
        officials = boxscore.get("officials") or []
        umpire = None
        for o in officials:
            if o.get("officialType", "").lower().startswith("home"):
                umpire = (o.get("official") or {}).get("fullName")
                break
        rows.append({"game_pk": game_pk, "umpire": umpire})
    df = pd.DataFrame(rows)
    print(f"Extracted {len(df)} games; umpire populated for {df['umpire'].notna().sum()}")
    df.to_parquet(OUT, index=False)
    print(f"Saved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

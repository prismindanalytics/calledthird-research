#!/usr/bin/env python3
"""Build per-batter lineup-spot lookup from MLB Stats API live boxscore.

Strategy:
  - Discover all game_pks present in challenges + statcast.
  - For each game, fetch
        https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
    and read `liveData.boxscore.teams.{home,away}.battingOrder` (1-9 player IDs)
    plus `players` to find substitutions / pinch hitters.
  - Each batter is assigned the lineup_spot of the position they replaced.

Output: data/batter_lineup_spot.parquet  with columns
  game_pk, team, batter_id, lineup_spot, is_pinch_hitter

Polite: 0.1s sleep between game requests.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache" / "boxscores"
OUT_PATH = DATA / "batter_lineup_spot.parquet"

LIVE_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
HEADERS = {"User-Agent": "Mozilla/5.0 (CalledThird Analytics; seven-hole-tax)"}


def _gather_game_pks() -> list[int]:
    pks: set[int] = set()
    # From challenges (existing + new)
    for p in [
        Path("/Users/haohu/Documents/GitHub/calledthird/research/team-challenge-iq/data/all_challenges_detail.json"),
        DATA / "all_challenges_apr15_may04.json",
    ]:
        if p.exists():
            with open(p) as f:
                rows = json.load(f)
            for r in rows:
                if r.get("game_pk"):
                    pks.add(int(r["game_pk"]))
    # From statcast (existing + new)
    for p in [
        Path("/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/data/statcast_2026_mar27_apr22.parquet"),
        DATA / "statcast_2026_apr23_may04.parquet",
    ]:
        if p.exists():
            df = pd.read_parquet(p, columns=["game_pk"])
            pks.update(int(x) for x in df["game_pk"].unique() if pd.notna(x))
    return sorted(pks)


def _fetch_live(game_pk: int) -> dict | None:
    cache_path = CACHE / f"{game_pk}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    url = LIVE_URL.format(game_pk=game_pk)
    try:
        resp = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  WARN game {game_pk}: {e}", file=sys.stderr)
        return None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    return data


def _parse_team(team_label: str, team_data: dict, game_pk: int) -> list[dict]:
    """Extract per-batter lineup spot for one team.

    Strategy:
      1. battingOrder gives the 9 STARTERS. lineup_spot = 1..9.
      2. Substitutions: for each player in `players`, look at their `battingOrder`
         attribute. MLB returns this as e.g. "100" (slot 1), "201" (sub 1 of slot 2),
         "302" (sub 2 of slot 3), etc. The leading digit(s) are the lineup spot.
    """
    rows: list[dict] = []
    abbr = (team_data.get("team", {}) or {}).get("abbreviation") or team_label
    starters = team_data.get("battingOrder", []) or []
    starter_set = set(int(pid) for pid in starters)

    players = team_data.get("players", {}) or {}
    for pid_key, pdata in players.items():
        person = pdata.get("person", {}) or {}
        pid = person.get("id")
        if pid is None:
            continue
        bo = pdata.get("battingOrder")
        if not bo:
            continue
        try:
            bo_num = int(bo)
        except Exception:
            continue
        # MLB encodes battingOrder as "X00" for starter at slot X, "X01"/"X02" for
        # subsequent subs in slot X.
        slot = bo_num // 100
        sub_index = bo_num % 100
        if slot < 1 or slot > 9:
            continue
        is_pinch = sub_index > 0 or pid not in starter_set
        rows.append({
            "game_pk": game_pk,
            "team": abbr,
            "batter_id": int(pid),
            "lineup_spot": int(slot),
            "is_pinch_hitter": bool(is_pinch),
        })
    return rows


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    game_pks = _gather_game_pks()
    print(f"Discovered {len(game_pks)} unique game_pks across all data sources")

    all_rows: list[dict] = []
    failures: list[int] = []
    for i, pk in enumerate(game_pks):
        data = _fetch_live(pk)
        if not data:
            failures.append(pk)
            continue
        teams = (data.get("liveData") or {}).get("boxscore", {}).get("teams", {}) or {}
        for label in ("home", "away"):
            t = teams.get(label, {}) or {}
            all_rows.extend(_parse_team(label, t, pk))
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(game_pks)} games processed, {len(all_rows)} rows")
        time.sleep(0.05)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("ERROR: no rows extracted", file=sys.stderr)
        return 1
    # dedupe (some games may have a player listed multiple times)
    df = df.drop_duplicates(subset=["game_pk", "batter_id", "lineup_spot", "is_pinch_hitter"])
    print(f"Writing {len(df):,} rows ({df['game_pk'].nunique()} games, {df['batter_id'].nunique()} unique batters)")
    print(f"  is_pinch_hitter: {df['is_pinch_hitter'].sum()} ({df['is_pinch_hitter'].mean():.2%})")
    print(f"  lineup_spot dist:\n{df['lineup_spot'].value_counts().sort_index()}")
    if failures:
        print(f"  failures: {len(failures)} games (e.g. {failures[:5]})")
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

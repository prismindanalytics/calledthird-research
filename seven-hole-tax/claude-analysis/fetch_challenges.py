#!/usr/bin/env python3
"""Fetch ABS challenges Apr 15 - May 4, 2026 from Baseball Savant gamefeed API.

Schema is intentionally aligned with team-challenge-iq/data/all_challenges_detail.json
so the two corpora can be concatenated.

Usage:
    python fetch_challenges.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = OUT_DIR / "all_challenges_apr15_may04.json"

GF_URL = "https://baseballsavant.mlb.com/gf"
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
HEADERS = {"User-Agent": "Mozilla/5.0 (CalledThird Analytics; seven-hole-tax)"}


def fetch_schedule(d: date) -> list[dict]:
    """Return schedule entries (dict with gamePk, official_date, etc.) for a date."""
    resp = httpx.get(
        SCHEDULE_URL,
        params={"date": d.strftime("%Y-%m-%d"), "sportId": 1},
        timeout=30,
    )
    resp.raise_for_status()
    games = []
    for gd in resp.json().get("dates", []):
        for g in gd.get("games", []):
            games.append(g)
    return games


def fetch_game_abs(game_pk: int, fallback_date: str | None = None) -> list[dict]:
    """Fetch ABS challenges for a single game; returns rows compatible with
    team-challenge-iq's all_challenges_detail.json schema.
    """
    resp = httpx.get(
        GF_URL,
        params={"game_pk": game_pk},
        headers=HEADERS,
        timeout=45,
        follow_redirects=True,
    )
    if resp.status_code != 200:
        print(f"    WARN gamefeed {resp.status_code} for {game_pk}", file=sys.stderr)
        return []

    try:
        data = resp.json()
    except Exception as e:
        print(f"    WARN gamefeed JSON parse failed for {game_pk}: {e}", file=sys.stderr)
        return []

    if not data.get("hasAbs"):
        return []

    home_team = data.get("home_team_abbrev") or data.get("home_team", {}).get("abbreviation")
    away_team = data.get("away_team_abbrev") or data.get("away_team", {}).get("abbreviation")
    game_date = (data.get("game_date") or fallback_date or "").split("T")[0]

    # Umpire: try multiple keys (Savant has been inconsistent)
    umpire = (
        data.get("home_plate_umpire")
        or (data.get("umpires", {}) or {}).get("home_plate")
        or data.get("hp_umpire")
    )
    if isinstance(umpire, dict):
        umpire = umpire.get("name") or umpire.get("fullName")

    challenges: list[dict] = []
    all_pitches = list(data.get("team_home", []) or []) + list(data.get("team_away", []) or [])
    for p in all_pitches:
        if not p.get("is_abs_challenge"):
            continue
        abs_info = p.get("abs_challenge", {}) or {}
        is_overturned = bool(abs_info.get("is_overturned", False))
        challenger_type = abs_info.get("challenging_player_type", "")
        is_batter = abs_info.get("is_batter", False)
        edge_dist_ft = abs_info.get("edge_distance")
        # convert to inches to match all_challenges_detail.json schema
        edge_dist_in = (edge_dist_ft * 12.0) if edge_dist_ft is not None else None

        final_call = p.get("call_name", "") or ""
        final_strike = final_call.lower() == "strike"
        if is_overturned:
            initial_call = "Ball" if final_strike else "Strike"
        else:
            initial_call = final_call

        challenger = challenger_type if challenger_type else ("batter" if is_batter else "catcher")
        in_zone = bool(p.get("savantIsInZone", False))

        challenges.append({
            "id": None,
            "game_pk": game_pk,
            "game_date": game_date,
            "play_id": p.get("play_id"),
            "inning": p.get("inning"),
            "ab_number": p.get("ab_number"),
            "pitch_number": p.get("pitch_number"),
            "team_batting": p.get("team_batting"),
            "team_fielding": p.get("team_fielding"),
            "batter_id": p.get("batter"),
            "batter_name": p.get("batter_name"),
            "pitcher_id": p.get("pitcher"),
            "pitcher_name": p.get("pitcher_name"),
            "catcher_id": p.get("catcher"),
            "catcher_name": p.get("catcher_name"),
            "stand": p.get("stand"),
            "p_throws": p.get("p_throws"),
            "initial_call": initial_call,
            "final_call": final_call,
            "overturned": int(is_overturned),
            "challenger": challenger,
            "plate_x": p.get("px"),
            "plate_z": p.get("pz"),
            "zone": p.get("zone"),
            "in_zone": int(in_zone),
            "pitch_type": p.get("pitch_type"),
            "pitch_name": p.get("pitch_name"),
            "start_speed": p.get("start_speed"),
            "balls": p.get("balls"),
            "strikes": p.get("strikes"),
            "outs": p.get("outs"),
            "created_at": None,
            "challenge_value": None,
            "edge_distance_in": edge_dist_in,
            "home_team": home_team,
            "away_team": away_team,
            "umpire": umpire,
        })

    return challenges


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = date(2026, 4, 15)
    end = date(2026, 5, 4)
    print(f"Fetching ABS challenges {start} .. {end}")
    all_challenges: list[dict] = []
    daily_progress: dict[str, int] = {}
    d = start
    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        try:
            games = fetch_schedule(d)
        except Exception as e:
            print(f"  {date_str} schedule failed: {e}", file=sys.stderr)
            d += timedelta(days=1)
            continue
        # Only regular-season MLB games (gameType=R)
        games = [g for g in games if g.get("gameType") == "R" and g.get("status", {}).get("statusCode") in {"F", "FR", "FT"}]
        n_games = len(games)
        date_total = 0
        for i, g in enumerate(games):
            pk = g["gamePk"]
            ch = fetch_game_abs(pk, fallback_date=date_str)
            date_total += len(ch)
            all_challenges.extend(ch)
            time.sleep(0.05)
        print(f"  {date_str}: {n_games:>2} games, {date_total:>3} challenges")
        daily_progress[date_str] = date_total
        d += timedelta(days=1)

    print(f"Total fetched: {len(all_challenges)} challenges across {sum(1 for v in daily_progress.values() if v > 0)} active days")
    OUT_PATH.write_text(json.dumps(all_challenges, indent=2))
    print(f"Saved -> {OUT_PATH}")

    # Quick QA stats
    if all_challenges:
        by_date: dict[str, int] = {}
        for r in all_challenges:
            by_date[r["game_date"]] = by_date.get(r["game_date"], 0) + 1
        print("Per-date challenge counts:")
        for d_, n in sorted(by_date.items()):
            print(f"  {d_}: {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

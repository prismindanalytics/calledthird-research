#!/usr/bin/env python3
"""Compute per-entry inherited runner stats from MLB play-by-play API.

Uses the responsiblePitcher field to get official scorer attribution
at the per-event level, combined with entry context (inning, outs).

This is the authoritative per-entry analysis that the boxscore+Statcast
join could not provide.

Usage:
    python compute_ir_from_playbyplay.py --start 2026-03-27 --end 2026-04-05
    python compute_ir_from_playbyplay.py --season 2025
"""

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

import httpx

OUTPUT_DIR = Path(__file__).parent / "data"


def analyze_game(game_pk: int) -> list[dict]:
    """Analyze inherited runners from play-by-play for one game."""
    try:
        resp = httpx.get(
            f"https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay",
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
    except Exception:
        return []

    plays = data.get("allPlays", [])
    if not plays:
        return []

    # Track pitcher transitions
    entries = {}  # pitcher_id -> entry info
    current_pitcher = None

    for play in plays:
        pitcher = play.get("matchup", {}).get("pitcher", {})
        pid = pitcher.get("id")
        if not pid:
            continue

        if current_pitcher and pid != current_pitcher and pid not in entries:
            about = play.get("about", {})
            # Get outs from the first pitch event's count
            play_events = play.get("playEvents", [])
            entry_outs = play_events[0].get("count", {}).get("outs", 0) if play_events else 0
            inning = about.get("inning", 0)
            entries[pid] = {
                "game_pk": game_pk,
                "pitcher_id": pid,
                "pitcher_name": pitcher.get("fullName", ""),
                "inning": inning,
                "half": about.get("halfInning", ""),
                "outs": entry_outs,
                "is_extra_inning": inning >= 10,
                "inherited_runners": 0,
                "inherited_scored": 0,
            }

        current_pitcher = pid

        # Track inherited runs via responsiblePitcher
        for runner in play.get("runners", []):
            movement = runner.get("movement", {})
            details = runner.get("details", {})

            if movement.get("end") == "score":
                resp_pid = details.get("responsiblePitcher", {}).get("id")
                mound_pid = play.get("matchup", {}).get("pitcher", {}).get("id")

                if resp_pid and mound_pid and resp_pid != mound_pid:
                    if mound_pid in entries:
                        entries[mound_pid]["inherited_scored"] += 1

    # Get official IR counts from boxscore
    try:
        box_resp = httpx.get(
            f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore",
            timeout=30,
        )
        box = box_resp.json()
        for team_key in ["home", "away"]:
            team_abbr = box.get("teams", {}).get(team_key, {}).get("team", {}).get("abbreviation", "")
            players = box.get("teams", {}).get(team_key, {}).get("players", {})
            for _, pdata in players.items():
                stats = pdata.get("stats", {}).get("pitching", {})
                if not stats:
                    continue
                ir = stats.get("inheritedRunners", 0)
                if ir == 0:
                    continue
                pid = pdata.get("person", {}).get("id")
                if pid in entries:
                    entries[pid]["inherited_runners"] = ir
                    entries[pid]["team"] = team_abbr
    except Exception:
        pass

    # Return only entries with inherited runners
    return [e for e in entries.values() if e["inherited_runners"] > 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2026-03-27")
    parser.add_argument("--end", type=str, default="2026-04-05")
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()

    if args.season:
        start_dt = date(args.season, 3, 27)
        end_dt = date(args.season, 9, 28)
    else:
        start_dt = date.fromisoformat(args.start)
        end_dt = date.fromisoformat(args.end)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_entries = []
    total_games = 0
    current = start_dt

    while current <= end_dt:
        # Get games for date
        try:
            resp = httpx.get(
                "https://statsapi.mlb.com/api/v1/schedule",
                params={"date": current.strftime("%Y-%m-%d"), "sportId": 1},
                timeout=30,
            )
            games = []
            for gd in resp.json().get("dates", []):
                for g in gd.get("games", []):
                    if g.get("status", {}).get("detailedState") == "Final":
                        games.append(g["gamePk"])
        except Exception:
            games = []

        for gpk in games:
            entries = analyze_game(gpk)
            for e in entries:
                e["game_date"] = current.strftime("%Y-%m-%d")
            all_entries.extend(entries)
            total_games += 1

        if total_games % 100 == 0 and total_games > 0:
            print(f"  {current}: {total_games} games, {len(all_entries)} entries")

        current += timedelta(days=1)
        time.sleep(0.05)

    # Save
    label = str(args.season) if args.season else f"{args.start}_to_{args.end}"
    out_path = OUTPUT_DIR / f"ir_playbyplay_{label}.json"
    out_path.write_text(json.dumps(all_entries, indent=2))

    total_ir = sum(e["inherited_runners"] for e in all_entries)
    total_irs = sum(e["inherited_scored"] for e in all_entries)
    strand = (total_ir - total_irs) / total_ir * 100 if total_ir > 0 else 0

    print(f"\n=== RESULTS ({label}) ===")
    print(f"Games: {total_games}")
    print(f"Entries with IR: {len(all_entries)}")
    print(f"Total IR: {total_ir}, IRS: {total_irs}")
    print(f"Strand rate: {strand:.1f}%")

    # By outs (PER-ENTRY, authoritative)
    print(f"\nBy outs at entry:")
    for outs_val in [0, 1, 2]:
        subset = [e for e in all_entries if e["outs"] == outs_val and not e["is_extra_inning"]]
        if not subset:
            continue
        ti = sum(e["inherited_runners"] for e in subset)
        ts = sum(e["inherited_scored"] for e in subset)
        print(f"  {outs_val} outs: {len(subset)} entries, {ti} IR, strand={(ti-ts)/ti*100:.1f}%")

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

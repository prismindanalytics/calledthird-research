#!/usr/bin/env python3
"""Pull 2025 full-season inherited runner data from MLB boxscore API.

Fetches every 2025 regular season game's boxscore and extracts
inheritedRunners / inheritedRunnersScored per pitcher.

~2,400 games × 1 API call each = ~2,400 requests.

Usage:
    python pull_2025_inherited_runners.py
"""

import json
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

OUTPUT_DIR = Path(__file__).parent / "data"


def fetch_games_for_date(dt: date) -> list[int]:
    """Get game PKs for a date."""
    resp = httpx.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"date": dt.strftime("%Y-%m-%d"), "sportId": 1},
        timeout=30,
    )
    if resp.status_code != 200:
        return []
    data = resp.json()
    pks = []
    for gd in data.get("dates", []):
        for g in gd.get("games", []):
            if g.get("status", {}).get("detailedState") == "Final":
                pks.append(g["gamePk"])
    return pks


def fetch_ir_from_boxscore(game_pk: int) -> list[dict]:
    """Extract IR/IRS from a game's boxscore."""
    try:
        resp = httpx.get(
            f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore",
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        box = resp.json()
    except Exception:
        return []

    entries = []
    for team_key in ["home", "away"]:
        team_data = box.get("teams", {}).get(team_key, {})
        team_abbr = team_data.get("team", {}).get("abbreviation", "")
        players = team_data.get("players", {})

        for _, pdata in players.items():
            stats = pdata.get("stats", {}).get("pitching", {})
            if not stats:
                continue
            ir = stats.get("inheritedRunners", 0)
            if ir == 0:
                continue
            irs = stats.get("inheritedRunnersScored", 0)
            person = pdata.get("person", {})

            entries.append({
                "game_pk": game_pk,
                "pitcher_id": person.get("id", 0),
                "pitcher_name": person.get("fullName", ""),
                "team": team_abbr,
                "inherited_runners": ir,
                "inherited_scored": irs,
                "innings_pitched": stats.get("inningsPitched", "0"),
            })

    return entries


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2025 regular season: March 27 - September 28
    start = date(2025, 3, 27)
    end = date(2025, 9, 28)

    all_entries = []
    total_games = 0
    current = start

    while current <= end:
        game_pks = fetch_games_for_date(current)
        if game_pks:
            for pk in game_pks:
                entries = fetch_ir_from_boxscore(pk)
                all_entries.extend(entries)
                total_games += 1

            if total_games % 100 == 0:
                print(f"  {current}: {total_games} games, {len(all_entries)} IR entries so far")

        current += timedelta(days=1)
        # Be nice to the API
        time.sleep(0.1)

    # Save raw entries
    out_path = OUTPUT_DIR / "inherited_runners_2025_raw.json"
    out_path.write_text(json.dumps(all_entries, indent=2))

    # Aggregate per pitcher
    from collections import defaultdict
    pitcher = defaultdict(lambda: {"name": "", "team": "", "entries": 0, "ir": 0, "irs": 0})
    for r in all_entries:
        pid = r["pitcher_id"]
        pitcher[pid]["name"] = r["pitcher_name"]
        pitcher[pid]["team"] = r["team"]
        pitcher[pid]["entries"] += 1
        pitcher[pid]["ir"] += r["inherited_runners"]
        pitcher[pid]["irs"] += r["inherited_scored"]

    agg = []
    for pid, s in pitcher.items():
        if s["ir"] < 1:
            continue
        sr = (s["ir"] - s["irs"]) / s["ir"] * 100
        agg.append({
            "pitcher_id": pid,
            "name": s["name"],
            "team": s["team"],
            "entries": s["entries"],
            "inherited": s["ir"],
            "scored": s["irs"],
            "strand_rate": round(sr, 1),
        })
    agg.sort(key=lambda x: x["inherited"], reverse=True)

    agg_path = OUTPUT_DIR / "inherited_runners_2025_agg.json"
    agg_path.write_text(json.dumps(agg, indent=2))

    total_ir = sum(r["inherited_runners"] for r in all_entries)
    total_irs = sum(r["inherited_scored"] for r in all_entries)
    strand = (total_ir - total_irs) / total_ir * 100 if total_ir > 0 else 0

    print(f"\n=== 2025 FULL SEASON RESULTS ===")
    print(f"Games: {total_games}")
    print(f"IR entries: {len(all_entries)}")
    print(f"Total IR: {total_ir}, IRS: {total_irs}")
    print(f"Strand rate: {strand:.1f}%")
    print(f"Pitchers with IR: {len(agg)}")
    print(f"Saved raw -> {out_path}")
    print(f"Saved agg -> {agg_path}")

    # Top/bottom
    qualified = [p for p in agg if p["inherited"] >= 15]
    print(f"\nQualified (15+ IR): {len(qualified)}")
    print("\nBest stranders:")
    for p in sorted(qualified, key=lambda x: x["strand_rate"], reverse=True)[:10]:
        print(f"  {p['name']:25s} {p['team']:>4s} | {p['inherited']:3d} IR | {p['scored']:3d} IRS | {p['strand_rate']:5.1f}%")
    print("\nWorst stranders:")
    for p in sorted(qualified, key=lambda x: x["strand_rate"])[:10]:
        print(f"  {p['name']:25s} {p['team']:>4s} | {p['inherited']:3d} IR | {p['scored']:3d} IRS | {p['strand_rate']:5.1f}%")


if __name__ == "__main__":
    main()

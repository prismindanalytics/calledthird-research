from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from common import DATASETS_DIR, DATA_DIR, atomic_write_json, atomic_write_parquet, clean_statcast_frame
from data_pull import resolve_mlbam_id
from features import (
    aggregate_hitter_window,
    aggregate_pitcher_window,
    assign_game_number,
    batting_team,
    load_statcast,
    make_2026_hitter_features,
    make_2026_pitcher_features,
)
from r2_utils import (
    NAMED_HITTER_KEYS,
    NAMED_PITCHER_KEYS,
    R2_TABLES_DIR,
    add_2026_pitcher_prior_columns,
    add_player_names,
    add_r2_hitter_columns,
    load_name_map,
)


ESPN_OPS_TOP20_AS_OF_2026_04_25 = [
    {"rank": 1, "name": "Yordan Alvarez", "team": "HOU", "metric": "OPS", "value": 1.247},
    {"rank": 2, "name": "Ben Rice", "team": "NYY", "metric": "OPS", "value": 1.197},
    {"rank": 3, "name": "Munetaka Murakami", "team": "CHW", "metric": "OPS", "value": 1.020},
    {"rank": 4, "name": "James Wood", "team": "WSH", "metric": "OPS", "value": 1.005},
    {"rank": 5, "name": "Corbin Carroll", "team": "ARI", "metric": "OPS", "value": 0.976},
    {"rank": 6, "name": "Max Muncy", "team": "LAD", "metric": "OPS", "value": 0.972},
    {"rank": 7, "name": "Sal Stewart", "team": "CIN", "metric": "OPS", "value": 0.970},
    {"rank": 8, "name": "Mike Trout", "team": "LAA", "metric": "OPS", "value": 0.969},
    {"rank": 9, "name": "Michael Harris II", "team": "ATL", "metric": "OPS", "value": 0.944},
    {"rank": 9, "name": "Matt Olson", "team": "ATL", "metric": "OPS", "value": 0.944},
    {"rank": 9, "name": "Ryan O'Hearn", "team": "PIT", "metric": "OPS", "value": 0.944},
    {"rank": 12, "name": "CJ Abrams", "team": "WSH", "metric": "OPS", "value": 0.936},
    {"rank": 13, "name": "Aaron Judge", "team": "NYY", "metric": "OPS", "value": 0.931},
    {"rank": 13, "name": "Andy Pages", "team": "LAD", "metric": "OPS", "value": 0.931},
    {"rank": 15, "name": "Shea Langeliers", "team": "ATH", "metric": "OPS", "value": 0.926},
    {"rank": 16, "name": "Jordan Walker", "team": "STL", "metric": "OPS", "value": 0.924},
    {"rank": 17, "name": "Carter Jensen", "team": "KC", "metric": "OPS", "value": 0.923},
    {"rank": 19, "name": "Kevin McGonigle", "team": "DET", "metric": "OPS", "value": 0.911},
    {"rank": 20, "name": "Xavier Edwards", "team": "MIA", "metric": "OPS", "value": 0.908},
    {"rank": 21, "name": "Drake Baldwin", "team": "ATL", "metric": "OPS", "value": 0.903},
]


def latest_2026_statcast_path() -> Path:
    path = DATA_DIR / "statcast_2026_through_apr25.parquet"
    if path.exists() and path.stat().st_size:
        return path
    return DATA_DIR / "statcast_2026_through_apr24.parquet"


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def people_names_by_id(ids: list[int]) -> dict[int, str]:
    ids = sorted({int(i) for i in ids if pd.notna(i)})
    out: dict[int, str] = {}
    for start in range(0, len(ids), 100):
        chunk = ids[start : start + 100]
        url = "https://statsapi.mlb.com/api/v1/people"
        try:
            response = requests.get(url, params={"personIds": ",".join(map(str, chunk))}, timeout=20)
            response.raise_for_status()
            for person in response.json().get("people", []):
                if person.get("id") and person.get("fullName"):
                    out[int(person["id"])] = str(person["fullName"])
        except Exception:
            continue
    return out


def pitcher_start_counts(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    first = df[df["inning"].eq(1)].copy()
    if first.empty:
        return pd.DataFrame(columns=["season", "pitcher", col_name])
    first["fielding_team"] = np.where(
        first["inning_topbot"].astype(str).str.lower().eq("top"),
        first["home_team"],
        first["away_team"],
    )
    starts = (
        first.sort_values(["season", "game_pk", "fielding_team", "at_bat_number", "pitch_number"])
        .drop_duplicates(["season", "game_pk", "fielding_team"])
        .groupby(["season", "pitcher"])
        .size()
        .reset_index(name=col_name)
    )
    return starts


def build_historical_pitcher_starts() -> pd.DataFrame:
    out = DATASETS_DIR / "r2_pitcher_start_counts_2022_2026.parquet"
    rows = []
    for season in range(2022, 2026):
        path = DATA_DIR / f"statcast_{season}.parquet"
        df = load_statcast(path)
        df["season"] = season
        rows.append(pitcher_start_counts(df, "starts_full"))
    current = load_statcast(latest_2026_statcast_path())
    current["season"] = 2026
    rows.append(pitcher_start_counts(current, "starts_cutoff").rename(columns={"starts_cutoff": "starts_full"}))
    starts = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["season", "pitcher", "starts_full"])
    atomic_write_parquet(starts, out)
    starts.to_csv(R2_TABLES_DIR / "r2_pitcher_start_counts_2022_2026.csv", index=False)
    return starts


def refresh_name_map(extra_ids: list[int]) -> None:
    existing = load_name_map()
    missing = [int(i) for i in extra_ids if int(i) not in existing or existing[int(i)].startswith("MLBAM ")]
    fetched = people_names_by_id(missing)
    rows = [{"key_mlbam": key, "name": value} for key, value in {**existing, **fetched}.items()]
    for key, value in {**NAMED_HITTER_KEYS, **NAMED_PITCHER_KEYS}.items():
        rows.append({"key_mlbam": value, "name": key.replace("_", " ").title()})
    pd.DataFrame(rows).drop_duplicates("key_mlbam", keep="last").to_csv(DATASETS_DIR / "name_map.csv", index=False)


def write_mainstream_reference() -> dict:
    path = DATA_DIR / "mainstream_top20.json"
    players = []
    for rec in ESPN_OPS_TOP20_AS_OF_2026_04_25:
        mlbam = resolve_mlbam_id(rec["name"])
        players.append({**rec, "mlbam": mlbam, "normalized_name": normalize_name(rec["name"])})
    payload = {
        "as_of_date": "2026-04-25",
        "source_name": "ESPN MLB player batting stat leaders",
        "source_url": "https://www.espn.com/mlb/stats/player/_/season/2026/table/batting/sort/OPS/dir/desc",
        "selection_rule": "Top 20 rows on ESPN batting OPS leaderboard captured for Round 2. This is the mainstream-coverage proxy required for sleeper classification.",
        "players": players,
    }
    atomic_write_json(path, payload)
    return payload


def write_closer_reference() -> dict:
    path = DATA_DIR / "closer_reference_top30_2025.json"
    if path.exists() and path.stat().st_size:
        return json.loads(path.read_text(encoding="utf-8"))
    url = "https://statsapi.mlb.com/api/v1/stats/leaders"
    params = {"leaderCategories": "saves", "season": "2025", "statGroup": "pitching", "limit": 30}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    leaders = response.json().get("leagueLeaders", [{}])[0].get("leaders", [])
    players = []
    for item in leaders:
        person = item.get("person", {})
        team = item.get("team", {})
        players.append(
            {
                "rank": int(item.get("rank")),
                "saves": int(item.get("value")),
                "mlbam": int(person.get("id")),
                "name": person.get("fullName"),
                "team": team.get("name"),
            }
        )
    payload = {
        "as_of_date": "2026-04-25",
        "source_name": "MLB Stats API 2025 saves leaders",
        "source_url": response.url,
        "selection_rule": "Top 30 2025 saves leaders excluded from sleeper-reliever list as known closers.",
        "players": players,
    }
    atomic_write_json(path, payload)
    return payload


def build_2026_hitter_universe() -> pd.DataFrame:
    historical = pd.read_parquet(DATASETS_DIR / "hitter_features.parquet")
    league = pd.read_parquet(DATASETS_DIR / "league_environment.parquet")
    first22, _ = make_2026_hitter_features(league, historical)
    first22 = add_r2_hitter_columns(first22, reference=None)

    statcast = load_statcast(latest_2026_statcast_path())
    statcast["season"] = 2026
    statcast["batting_team"] = batting_team(statcast)
    cutoff = aggregate_hitter_window(statcast, "cutoff")
    teams = (
        statcast[statcast["events"].notna()]
        .groupby(["season", "batter"])["batting_team"]
        .agg(lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else pd.NA)
        .reset_index(name="team_cutoff")
    )
    universe = first22.merge(cutoff[["season", "batter", "pa_cutoff", "woba_cutoff", "ops_cutoff"]], on=["season", "batter"], how="left")
    universe = universe.merge(teams, on=["season", "batter"], how="left")
    universe["team"] = universe.get("team").fillna(universe.get("team_cutoff"))
    universe = universe[universe["pa_cutoff"].ge(50)].copy()
    refresh_name_map(universe["batter"].dropna().astype(int).tolist())
    universe = add_player_names(universe, "batter", "player")

    mainstream = write_mainstream_reference()
    mainstream_ids = {int(p["mlbam"]) for p in mainstream["players"] if p.get("mlbam") is not None}
    mainstream_names = {p["normalized_name"] for p in mainstream["players"]}
    universe["normalized_name"] = universe["player"].map(normalize_name)
    universe["in_mainstream_top20"] = universe["batter"].map(lambda x: int(x) in mainstream_ids) | universe["normalized_name"].isin(mainstream_names)
    universe["named_r1_case"] = universe["batter"].isin(NAMED_HITTER_KEYS.values())
    out = universe.sort_values(["woba_cutoff", "pa_cutoff"], ascending=False).reset_index(drop=True)
    atomic_write_parquet(out, DATASETS_DIR / "r2_hitter_universe.parquet")
    out.to_csv(R2_TABLES_DIR / "r2_hitter_universe.csv", index=False)
    return out


def build_2026_reliever_universe() -> pd.DataFrame:
    historical = pd.read_parquet(DATASETS_DIR / "pitcher_features.parquet")
    league = pd.read_parquet(DATASETS_DIR / "league_pitching_environment.parquet")
    first22 = make_2026_pitcher_features(league)
    first22 = add_2026_pitcher_prior_columns(first22, historical)

    statcast = load_statcast(latest_2026_statcast_path())
    statcast["season"] = 2026
    cutoff = aggregate_pitcher_window(statcast, "cutoff")
    starts = pitcher_start_counts(statcast, "starts_cutoff")
    universe = first22.merge(cutoff[["season", "pitcher", "bf_cutoff", "ip_cutoff", "k_rate_cutoff"]], on=["season", "pitcher"], how="left")
    universe = universe.merge(starts, on=["season", "pitcher"], how="left")
    universe["starts_cutoff"] = universe["starts_cutoff"].fillna(0).astype(int)
    universe = universe[universe["bf_cutoff"].ge(25) & universe["ip_cutoff"].lt(30) & universe["starts_cutoff"].eq(0)].copy()
    refresh_name_map(universe["pitcher"].dropna().astype(int).tolist())
    universe = add_player_names(universe, "pitcher", "player")
    closers = write_closer_reference()
    closer_ids = {int(p["mlbam"]) for p in closers.get("players", []) if p.get("mlbam") is not None}
    universe["known_2025_closer"] = universe["pitcher"].map(lambda x: int(x) in closer_ids)
    universe["named_r1_case"] = universe["pitcher"].isin(NAMED_PITCHER_KEYS.values())
    out = universe.sort_values(["k_rate_cutoff", "bf_cutoff"], ascending=False).reset_index(drop=True)
    atomic_write_parquet(out, DATASETS_DIR / "r2_reliever_universe.parquet")
    out.to_csv(R2_TABLES_DIR / "r2_reliever_universe.csv", index=False)
    return out


def main() -> dict:
    build_historical_pitcher_starts()
    hitters = build_2026_hitter_universe()
    relievers = build_2026_reliever_universe()
    manifest = json.loads((DATA_DIR / "statcast_2026_r2_cutoff_manifest.json").read_text(encoding="utf-8")) if (DATA_DIR / "statcast_2026_r2_cutoff_manifest.json").exists() else {}
    payload = {
        "requested_cutoff": "2026-04-25",
        "actual_max_game_date": manifest.get("actual_max_game_date"),
        "hitter_universe_n": int(len(hitters)),
        "reliever_universe_n": int(len(relievers)),
        "hitter_threshold": "pa_cutoff >= 50",
        "reliever_threshold": "bf_cutoff >= 25 and ip_cutoff < 30",
        "mainstream_reference": "data/mainstream_top20.json",
        "closer_reference": "data/closer_reference_top30_2025.json",
    }
    atomic_write_json(DATASETS_DIR / "r2_universe_manifest.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

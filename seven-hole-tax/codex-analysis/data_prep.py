from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from pybaseball import statcast

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "codex-analysis"
DATA_DIR = ANALYSIS_DIR / "data"
ARTIFACTS_DIR = ANALYSIS_DIR / "artifacts"
CHARTS_DIR = ANALYSIS_DIR / "charts"
DIAG_DIR = CHARTS_DIR / "model_diagnostics"

TEAM_CHALLENGE_PATH = ROOT.parent / "team-challenge-iq" / "data" / "all_challenges_detail.json"
STATCAST_BASE_PATH = ROOT.parent / "abs-walk-spike" / "data" / "statcast_2026_mar27_apr22.parquet"
STATCAST_2025_FULL_PATH = ROOT.parent / "count-distribution-abs" / "data" / "statcast_2025_full.parquet"

CHALLENGE_EXTENSION_PATH = DATA_DIR / "all_challenges_apr15_may04.json"
STATCAST_EXTENSION_PATH = DATA_DIR / "statcast_2026_apr23_may04.parquet"
LINEUP_PATH = DATA_DIR / "batter_lineup_spot.parquet"
UMPIRE_PATH = DATA_DIR / "game_umpires.parquet"
PITCHER_FAME_PATH = DATA_DIR / "pitcher_fame_quartile.parquet"
CATCHER_FRAMING_PATH = DATA_DIR / "catcher_framing_tier.parquet"

PREPARED_CHALLENGES_PATH = ARTIFACTS_DIR / "challenges_full.parquet"
PREPARED_CALLED_PATH = ARTIFACTS_DIR / "called_pitches_full.parquet"
RUN_MANIFEST_PATH = ARTIFACTS_DIR / "run_manifest.json"

USER_AGENT = "CalledThird research bot; contact: research@calledthird.local"
PLATE_HALF_WIDTH_FT = 17.0 / 24.0
MAX_ANALYSIS_DATE = pd.Timestamp("2026-05-04")


@dataclass
class PreparedData:
    challenges: pd.DataFrame
    called_pitches: pd.DataFrame
    manifest: dict[str, Any]


def ensure_dirs() -> None:
    for path in [DATA_DIR, ARTIFACTS_DIR, CHARTS_DIR, DIAG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(payload: Any, path: Path) -> None:
    def default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        return str(obj)

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def request_json(url: str, timeout: int = 30) -> Any:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def schedule_game_pks(start_date: str, end_date: str) -> list[int]:
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&gameTypes=R&startDate={start_date}&endDate={end_date}"
    )
    payload = request_json(url)
    game_pks: list[int] = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            if game.get("gameType") == "R":
                game_pks.append(int(game["gamePk"]))
    return sorted(set(game_pks))


def _home_plate_umpire_from_gf(gf: dict[str, Any]) -> str | None:
    for official in gf.get("boxscore", {}).get("officials", []):
        if official.get("officialType") == "Home Plate":
            return official.get("official", {}).get("fullName")
    return None


def _challenge_row_from_gf_pitch(pitch: dict[str, Any], gf: dict[str, Any]) -> dict[str, Any] | None:
    challenge = pitch.get("abs_challenge") or {}
    if not pitch.get("is_abs_challenge") or not challenge:
        return None

    initial_call = "Strike" if str(pitch.get("call", "")).upper() == "S" else "Ball"
    overturned = int(bool(challenge.get("is_overturned")))
    final_call = initial_call
    if overturned:
        final_call = "Ball" if initial_call == "Strike" else "Strike"

    home_team = gf.get("home_team_data", {}).get("abbreviation")
    away_team = gf.get("away_team_data", {}).get("abbreviation")
    edge_ft = challenge.get("edge_distance")
    if edge_ft is None:
        edge_ft = challenge.get("edge_distance_calc")
    edge_in = None if edge_ft is None else abs(float(edge_ft)) * 12.0

    return {
        "id": None,
        "game_pk": int(pitch["game_pk"]),
        "game_date": pd.to_datetime(gf.get("game_date") or gf.get("gameDate")).strftime("%Y-%m-%d"),
        "play_id": pitch.get("play_id"),
        "inning": pitch.get("inning"),
        "ab_number": pitch.get("ab_number"),
        "pitch_number": pitch.get("pitch_number"),
        "team_batting": pitch.get("team_batting"),
        "team_fielding": pitch.get("team_fielding"),
        "batter_id": pitch.get("batter"),
        "batter_name": pitch.get("batter_name"),
        "pitcher_id": pitch.get("pitcher"),
        "pitcher_name": pitch.get("pitcher_name"),
        "catcher_id": pitch.get("catcher"),
        "catcher_name": pitch.get("catcher_name"),
        "stand": pitch.get("stand"),
        "p_throws": pitch.get("p_throws"),
        "initial_call": initial_call,
        "final_call": final_call,
        "overturned": overturned,
        "challenger": challenge.get("challenging_player_type"),
        "plate_x": pitch.get("plate_x") if pitch.get("plate_x") is not None else pitch.get("px"),
        "plate_z": pitch.get("plate_z") if pitch.get("plate_z") is not None else pitch.get("pz"),
        "zone": pitch.get("zone"),
        "in_zone": int(bool(pitch.get("isInZone", pitch.get("savantIsInZone", False)))),
        "pitch_type": pitch.get("pitch_type"),
        "pitch_name": pitch.get("pitch_name"),
        "start_speed": pitch.get("start_speed"),
        "balls": pitch.get("pre_balls", pitch.get("balls")),
        "strikes": pitch.get("pre_strikes", pitch.get("strikes")),
        "outs": pitch.get("outs"),
        "created_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "challenge_value": None,
        "edge_distance_in": edge_in,
        "home_team": home_team,
        "away_team": away_team,
        "umpire": _home_plate_umpire_from_gf(gf),
    }


def fetch_abs_challenge_extension(force: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if CHALLENGE_EXTENSION_PATH.exists() and not force:
        return pd.DataFrame(load_json(CHALLENGE_EXTENSION_PATH))

    rows: list[dict[str, Any]] = []
    game_pks = schedule_game_pks("2026-04-15", "2026-05-04")
    for idx, game_pk in enumerate(game_pks, start=1):
        url = f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}"
        try:
            gf = request_json(url, timeout=45)
        except Exception as exc:
            print(f"WARN: gf fetch failed for {game_pk}: {exc}", file=sys.stderr)
            continue
        if not gf.get("hasAbs", False):
            time.sleep(0.1)
            continue
        for pitch in gf.get("team_home", []) + gf.get("team_away", []):
            row = _challenge_row_from_gf_pitch(pitch, gf)
            if row is not None:
                rows.append(row)
        if idx % 25 == 0:
            print(f"  challenge gf fetch {idx}/{len(game_pks)} games, {len(rows)} rows")
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
        df = df[(df["game_date"] >= "2026-04-15") & (df["game_date"] <= MAX_ANALYSIS_DATE)].copy()
        df = df.sort_values(["game_date", "game_pk", "inning", "ab_number", "pitch_number"]).reset_index(drop=True)
        df["game_date"] = df["game_date"].dt.strftime("%Y-%m-%d")
        df["id"] = np.arange(200000, 200000 + len(df))
    save_json(df.to_dict(orient="records"), CHALLENGE_EXTENSION_PATH)
    return df


def fetch_statcast_extension(force: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if STATCAST_EXTENSION_PATH.exists() and not force:
        return pd.read_parquet(STATCAST_EXTENSION_PATH)
    print("Fetching Statcast 2026-04-23 to 2026-05-04 via pybaseball...")
    df = statcast(start_dt="2026-04-23", end_dt="2026-05-04")
    if df.empty:
        raise RuntimeError("pybaseball returned no Statcast rows for 2026-04-23 to 2026-05-04")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date"] <= MAX_ANALYSIS_DATE].copy()
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    df.to_parquet(STATCAST_EXTENSION_PATH, index=False)
    return df


def _team_abbr_from_feed(feed: dict[str, Any], side: str) -> str:
    return feed["gameData"]["teams"][side]["abbreviation"]


def _home_plate_umpire_from_feed(feed: dict[str, Any]) -> tuple[int | None, str | None]:
    officials = feed.get("liveData", {}).get("boxscore", {}).get("officials", [])
    for official in officials:
        if official.get("officialType") == "Home Plate":
            ump = official.get("official", {})
            return ump.get("id"), ump.get("fullName")
    return None, None


def build_lineup_lookup(game_pks: list[int], force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    if LINEUP_PATH.exists() and UMPIRE_PATH.exists() and not force:
        return pd.read_parquet(LINEUP_PATH), pd.read_parquet(UMPIRE_PATH)

    lineup_rows: list[dict[str, Any]] = []
    umpire_rows: list[dict[str, Any]] = []
    for idx, game_pk in enumerate(sorted(set(map(int, game_pks))), start=1):
        try:
            feed = request_json(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live", timeout=45)
        except Exception as exc:
            print(f"WARN: feed/live failed for {game_pk}: {exc}", file=sys.stderr)
            continue
        game_date = pd.to_datetime(feed["gameData"]["datetime"]["officialDate"])
        hp_id, hp_name = _home_plate_umpire_from_feed(feed)
        home_abbr = _team_abbr_from_feed(feed, "home")
        away_abbr = _team_abbr_from_feed(feed, "away")
        umpire_rows.append(
            {
                "game_pk": int(game_pk),
                "game_date": game_date,
                "home_team": home_abbr,
                "away_team": away_abbr,
                "umpire_id": hp_id,
                "umpire": hp_name,
            }
        )

        for side, abbr in [("home", home_abbr), ("away", away_abbr)]:
            team = feed["liveData"]["boxscore"]["teams"][side]
            starters = [int(pid) for pid in team.get("battingOrder", [])]
            starter_spot = {pid: pos + 1 for pos, pid in enumerate(starters)}
            for player in team.get("players", {}).values():
                player_id = int(player["person"]["id"])
                batting_order = player.get("battingOrder")
                if batting_order is None and player_id not in starter_spot:
                    continue
                if batting_order is not None:
                    order_int = int(batting_order)
                    lineup_spot = order_int // 100
                    suffix = order_int % 100
                else:
                    lineup_spot = starter_spot[player_id]
                    suffix = 0
                if not 1 <= lineup_spot <= 9:
                    continue
                all_positions = player.get("allPositions", []) or []
                position_codes = {pos.get("abbreviation") for pos in all_positions}
                is_pinch = (player_id not in starter_spot) or suffix > 0 or bool(position_codes & {"PH", "PR"})
                lineup_rows.append(
                    {
                        "game_pk": int(game_pk),
                        "team": abbr,
                        "batter_id": player_id,
                        "lineup_spot": int(lineup_spot),
                        "is_pinch_hitter": bool(is_pinch),
                    }
                )
        if idx % 50 == 0:
            print(f"  lineup feed fetch {idx}/{len(set(game_pks))} games")
        time.sleep(0.1)

    lineup = pd.DataFrame(lineup_rows).drop_duplicates(["game_pk", "team", "batter_id"], keep="last")
    umpires = pd.DataFrame(umpire_rows).drop_duplicates(["game_pk"], keep="last")
    lineup.to_parquet(LINEUP_PATH, index=False)
    umpires.to_parquet(UMPIRE_PATH, index=False)
    return lineup, umpires


def build_pitcher_fame(force: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if PITCHER_FAME_PATH.exists() and not force:
        return pd.read_parquet(PITCHER_FAME_PATH)
    df = pd.read_parquet(STATCAST_2025_FULL_PATH)
    terminal = df[df["events"].notna() & (df["events"] != "")].copy()
    terminal["is_k"] = terminal["events"].astype(str).str.contains("strikeout", case=False, na=False)
    terminal["is_bb"] = terminal["events"].isin(["walk", "intent_walk"])
    summary = (
        terminal.groupby("pitcher", observed=True)
        .agg(pa=("events", "size"), strikeouts=("is_k", "sum"), walks=("is_bb", "sum"))
        .reset_index()
        .rename(columns={"pitcher": "pitcher_id"})
    )
    summary["k_minus_bb_rate"] = (summary["strikeouts"] - summary["walks"]) / summary["pa"]
    eligible = summary[summary["pa"] >= 20].copy()
    eligible["pitcher_fame_quartile"] = pd.qcut(
        eligible["k_minus_bb_rate"],
        q=4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        duplicates="drop",
    ).astype(str)
    out = summary.merge(
        eligible[["pitcher_id", "pitcher_fame_quartile"]],
        on="pitcher_id",
        how="left",
    )
    out["pitcher_fame_quartile"] = out["pitcher_fame_quartile"].fillna("unknown")
    out.to_parquet(PITCHER_FAME_PATH, index=False)
    return out


def build_catcher_framing(force: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if CATCHER_FRAMING_PATH.exists() and not force:
        return pd.read_parquet(CATCHER_FRAMING_PATH)
    url = "https://baseballsavant.mlb.com/catcher_framing?year=2025&team=&min=100&sort=4,1"
    html = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45).text
    match = re.search(r"const\s+data\s*=\s*(\[.*?\]);", html, re.S)
    if not match:
        raise RuntimeError("Could not parse Baseball Savant catcher framing data")
    data = json.loads(match.group(1))
    df = pd.DataFrame(data)
    id_col = "fielder_2" if "fielder_2" in df.columns else "id"
    name_col = "f2_name_display_first_last" if "f2_name_display_first_last" in df.columns else "name"
    out = df[[id_col, name_col, "rv_tot", "pitches_shadow"]].copy()
    out = out.rename(
        columns={
            id_col: "catcher_id",
            name_col: "catcher_name",
            "rv_tot": "framing_runs",
        }
    )
    out["catcher_id"] = pd.to_numeric(out["catcher_id"], errors="coerce").astype("Int64")
    out = out[out["catcher_id"].notna()].copy()
    out["framing_runs"] = pd.to_numeric(out["framing_runs"], errors="coerce")
    low, high = out["framing_runs"].quantile([1 / 3, 2 / 3]).tolist()
    out["catcher_framing_tier"] = np.where(
        out["framing_runs"] >= high,
        "top",
        np.where(out["framing_runs"] <= low, "bottom", "mid"),
    )
    out.to_parquet(CATCHER_FRAMING_PATH, index=False)
    return out


def load_statcast_full(force_fetch: bool = False) -> pd.DataFrame:
    base = pd.read_parquet(STATCAST_BASE_PATH)
    ext = fetch_statcast_extension(force=force_fetch)
    common = sorted(set(base.columns) & set(ext.columns))
    full = pd.concat([base[common], ext[common]], ignore_index=True)
    full["game_date"] = pd.to_datetime(full["game_date"])
    full = full[full["game_date"] <= MAX_ANALYSIS_DATE].copy()
    full = full.drop_duplicates(["game_pk", "at_bat_number", "pitch_number"], keep="last")
    return full.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def compute_edge_distance_ft(
    plate_x: pd.Series,
    plate_z: pd.Series,
    sz_top: pd.Series,
    sz_bot: pd.Series,
) -> pd.Series:
    x = pd.to_numeric(plate_x, errors="coerce").astype(float)
    z = pd.to_numeric(plate_z, errors="coerce").astype(float)
    top = pd.to_numeric(sz_top, errors="coerce").astype(float)
    bot = pd.to_numeric(sz_bot, errors="coerce").astype(float)
    outside_x = np.maximum(np.abs(x) - PLATE_HALF_WIDTH_FT, 0.0)
    outside_z = np.maximum.reduce([bot - z, z - top, np.zeros(len(x))])
    outside = np.hypot(outside_x, outside_z)
    inside = np.minimum.reduce([PLATE_HALF_WIDTH_FT - np.abs(x), z - bot, top - z])
    is_inside = (np.abs(x) <= PLATE_HALF_WIDTH_FT) & (z >= bot) & (z <= top)
    return pd.Series(np.where(is_inside, np.maximum(inside, 0.0), outside), index=plate_x.index)


def _batting_team_from_statcast(df: pd.DataFrame) -> pd.Series:
    return np.where(df["inning_topbot"].astype(str).str.lower().eq("top"), df["away_team"], df["home_team"])


def enrich_challenges(
    challenges: pd.DataFrame,
    statcast_full: pd.DataFrame,
    lineup: pd.DataFrame,
    umpires: pd.DataFrame,
    pitcher_fame: pd.DataFrame,
    catcher_framing: pd.DataFrame,
) -> pd.DataFrame:
    df = challenges.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    for col in ["game_pk", "batter_id", "pitcher_id", "catcher_id", "ab_number", "pitch_number", "balls", "strikes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    stat_cols = [
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "sz_top",
        "sz_bot",
        "release_speed",
        "pitch_type",
        "pitch_name",
        "stand",
        "p_throws",
    ]
    available = [col for col in stat_cols if col in statcast_full.columns]
    stat_small = statcast_full[available].drop_duplicates(["game_pk", "at_bat_number", "pitch_number"], keep="last")
    df = df.merge(
        stat_small,
        left_on=["game_pk", "ab_number", "pitch_number"],
        right_on=["game_pk", "at_bat_number", "pitch_number"],
        how="left",
        suffixes=("", "_statcast"),
    )
    for col in ["pitch_type", "pitch_name", "stand", "p_throws"]:
        stat_col = f"{col}_statcast"
        if stat_col in df.columns:
            df[col] = df[col].where(df[col].notna(), df[stat_col])
            df = df.drop(columns=[stat_col])
    if "release_speed" in df.columns and "start_speed" in df.columns:
        df["start_speed"] = df["start_speed"].where(df["start_speed"].notna(), df["release_speed"])

    missing_edge = df["edge_distance_in"].isna() if "edge_distance_in" in df.columns else pd.Series(True, index=df.index)
    if {"plate_x", "plate_z", "sz_top", "sz_bot"}.issubset(df.columns):
        computed = compute_edge_distance_ft(df["plate_x"], df["plate_z"], df["sz_top"], df["sz_bot"]) * 12.0
        df.loc[missing_edge, "edge_distance_in"] = computed.loc[missing_edge]

    lineup_key = lineup.rename(columns={"team": "team_batting"})
    df = df.merge(
        lineup_key,
        on=["game_pk", "team_batting", "batter_id"],
        how="left",
        validate="many_to_one",
    )
    df = df.merge(umpires[["game_pk", "umpire"]], on="game_pk", how="left", suffixes=("", "_feed"))
    df["umpire"] = df["umpire"].astype("object").where(df["umpire"].notna(), df["umpire_feed"].astype("object"))
    df = df.drop(columns=[col for col in ["umpire_feed", "at_bat_number"] if col in df.columns])

    df = df.merge(
        pitcher_fame[["pitcher_id", "pitcher_fame_quartile"]],
        on="pitcher_id",
        how="left",
    )
    df = df.merge(
        catcher_framing[["catcher_id", "catcher_framing_tier", "framing_runs"]],
        on="catcher_id",
        how="left",
    )
    df["pitcher_fame_quartile"] = df["pitcher_fame_quartile"].fillna("unknown")
    df["catcher_framing_tier"] = df["catcher_framing_tier"].fillna("unknown")
    df["count_state"] = df["balls"].astype("Int64").astype(str) + "-" + df["strikes"].astype("Int64").astype(str)
    df["overturned"] = pd.to_numeric(df["overturned"], errors="coerce").fillna(0).astype(int)
    df["in_zone"] = pd.to_numeric(df["in_zone"], errors="coerce").fillna(0).astype(int)
    df["is_pinch_hitter"] = df["is_pinch_hitter"].fillna(False).astype(bool)
    df["month_bucket"] = np.where(df["game_date"] < "2026-05-01", "mar_apr", "may_to_date")
    return df


def enrich_called_pitches(
    statcast_full: pd.DataFrame,
    lineup: pd.DataFrame,
    umpires: pd.DataFrame,
    pitcher_fame: pd.DataFrame,
    catcher_framing: pd.DataFrame,
) -> pd.DataFrame:
    df = statcast_full.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["description"].isin(["called_strike", "ball"])].copy()
    df = df.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot", "batter", "pitcher", "fielder_2"])
    df["team_batting"] = _batting_team_from_statcast(df)
    df["batter_id"] = pd.to_numeric(df["batter"], errors="coerce").astype("Int64")
    df["pitcher_id"] = pd.to_numeric(df["pitcher"], errors="coerce").astype("Int64")
    df["catcher_id"] = pd.to_numeric(df["fielder_2"], errors="coerce").astype("Int64")
    df["balls"] = pd.to_numeric(df["balls"], errors="coerce").astype("Int64")
    df["strikes"] = pd.to_numeric(df["strikes"], errors="coerce").astype("Int64")
    df["count_state"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    df["is_called_strike"] = (df["description"] == "called_strike").astype(int)
    df["edge_distance_ft"] = compute_edge_distance_ft(df["plate_x"], df["plate_z"], df["sz_top"], df["sz_bot"])
    df["is_borderline"] = df["edge_distance_ft"].le(0.3)
    df = df.merge(
        lineup.rename(columns={"team": "team_batting"}),
        on=["game_pk", "team_batting", "batter_id"],
        how="left",
        validate="many_to_one",
    )
    df = df.merge(umpires[["game_pk", "umpire"]], on="game_pk", how="left", suffixes=("", "_feed"))
    if "umpire_feed" in df.columns:
        df["umpire"] = df["umpire"].astype("object").where(df["umpire"].notna(), df["umpire_feed"].astype("object"))
        df = df.drop(columns=["umpire_feed"])
    df = df.merge(
        pitcher_fame[["pitcher_id", "pitcher_fame_quartile"]],
        on="pitcher_id",
        how="left",
    )
    df = df.merge(
        catcher_framing[["catcher_id", "catcher_framing_tier", "framing_runs"]],
        on="catcher_id",
        how="left",
    )
    df["pitcher_fame_quartile"] = df["pitcher_fame_quartile"].fillna("unknown")
    df["catcher_framing_tier"] = df["catcher_framing_tier"].fillna("unknown")
    df["pitch_type"] = df["pitch_type"].fillna("unknown").astype(str)
    df["stand"] = df["stand"].fillna("unknown").astype(str)
    df["p_throws"] = df["p_throws"].fillna("unknown").astype(str)
    df["is_pinch_hitter"] = df["is_pinch_hitter"].fillna(False).astype(bool)
    df["month_bucket"] = np.where(df["game_date"] < "2026-05-01", "mar_apr", "may_to_date")
    return df


def prepare_data(force_fetch: bool = False, force_lineup: bool = False) -> PreparedData:
    ensure_dirs()
    existing_challenges = pd.DataFrame(load_json(TEAM_CHALLENGE_PATH))
    extension_challenges = fetch_abs_challenge_extension(force=force_fetch)
    statcast_full = load_statcast_full(force_fetch=force_fetch)
    combined_challenges = pd.concat([existing_challenges, extension_challenges], ignore_index=True)
    combined_challenges["game_date"] = pd.to_datetime(combined_challenges["game_date"])
    combined_challenges = combined_challenges[combined_challenges["game_date"] <= MAX_ANALYSIS_DATE].copy()
    combined_challenges = combined_challenges.drop_duplicates(["game_pk", "play_id"], keep="last")

    game_pks = sorted(
        set(statcast_full["game_pk"].dropna().astype(int).tolist())
        | set(combined_challenges["game_pk"].dropna().astype(int).tolist())
    )
    lineup, umpires = build_lineup_lookup(game_pks, force=force_lineup)
    pitcher_fame = build_pitcher_fame(force=force_fetch)
    catcher_framing = build_catcher_framing(force=force_fetch)

    challenges = enrich_challenges(combined_challenges, statcast_full, lineup, umpires, pitcher_fame, catcher_framing)
    called_pitches = enrich_called_pitches(statcast_full, lineup, umpires, pitcher_fame, catcher_framing)
    challenges.to_parquet(PREPARED_CHALLENGES_PATH, index=False)
    called_pitches.to_parquet(PREPARED_CALLED_PATH, index=False)

    manifest = {
        "seed": 20260505,
        "analysis_window": {
            "challenges_start": str(challenges["game_date"].min().date()),
            "challenges_end": str(challenges["game_date"].max().date()),
            "called_pitches_start": str(called_pitches["game_date"].min().date()),
            "called_pitches_end": str(called_pitches["game_date"].max().date()),
            "max_allowed_date": str(MAX_ANALYSIS_DATE.date()),
        },
        "rows": {
            "existing_challenges": int(len(existing_challenges)),
            "extension_challenges": int(len(extension_challenges)),
            "combined_challenges": int(len(challenges)),
            "statcast_rows": int(len(statcast_full)),
            "called_pitch_rows": int(len(called_pitches)),
            "borderline_called_pitch_rows": int(called_pitches["is_borderline"].sum()),
            "lineup_rows": int(len(lineup)),
            "game_pks": int(len(game_pks)),
        },
        "missingness": {
            "challenge_lineup_missing": int(challenges["lineup_spot"].isna().sum()),
            "called_lineup_missing": int(called_pitches["lineup_spot"].isna().sum()),
            "challenge_edge_missing": int(challenges["edge_distance_in"].isna().sum()),
            "called_umpire_missing": int(called_pitches["umpire"].isna().sum()),
        },
        "sources": {
            "existing_challenges": str(TEAM_CHALLENGE_PATH),
            "challenge_extension": str(CHALLENGE_EXTENSION_PATH),
            "statcast_base": str(STATCAST_BASE_PATH),
            "statcast_extension": str(STATCAST_EXTENSION_PATH),
            "lineup_lookup": str(LINEUP_PATH),
            "pitcher_fame": str(PITCHER_FAME_PATH),
            "catcher_framing": str(CATCHER_FRAMING_PATH),
        },
        "versions": {
            "python": sys.version.replace("\n", " "),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    save_json(manifest, RUN_MANIFEST_PATH)
    return PreparedData(challenges=challenges, called_pitches=called_pitches, manifest=manifest)

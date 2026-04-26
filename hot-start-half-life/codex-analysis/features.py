from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    DATA_DIR,
    DATASETS_DIR,
    HIT_EVENTS,
    HITTER_ANALOG_COLS,
    HITTER_FEATURE_COLS,
    HBP_EVENTS,
    K_EVENTS,
    KNOWN_HITTERS,
    KNOWN_PITCHERS,
    NON_PA_EVENTS,
    PITCHER_ANALOG_COLS,
    PITCHER_FEATURE_COLS,
    SF_EVENTS,
    SWING_DESCRIPTIONS,
    WALK_EVENTS,
    WHIFF_DESCRIPTIONS,
    atomic_write_json,
    atomic_write_parquet,
    clean_statcast_frame,
    safe_divide,
    slugify_player,
)


def load_statcast(path: Path) -> pd.DataFrame:
    return clean_statcast_frame(pd.read_parquet(path))


def batting_team(df: pd.DataFrame) -> pd.Series:
    return np.where(df["inning_topbot"].astype(str).str.lower().eq("top"), df["away_team"], df["home_team"])


def assign_game_number(df: pd.DataFrame, player_col: str) -> pd.DataFrame:
    games = (
        df[[player_col, "game_date", "game_pk"]]
        .dropna(subset=[player_col, "game_date", "game_pk"])
        .drop_duplicates()
        .sort_values([player_col, "game_date", "game_pk"])
    )
    games["game_n"] = games.groupby(player_col).cumcount() + 1
    return df.merge(games[[player_col, "game_pk", "game_n"]], on=[player_col, "game_pk"], how="left")


def add_event_flags(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    ev = out["events"].astype(str)
    out["is_pa"] = out["events"].notna() & ~ev.isin(NON_PA_EVENTS)
    out["is_bb"] = ev.isin(WALK_EVENTS)
    out["is_hbp"] = ev.isin(HBP_EVENTS)
    out["is_k"] = ev.isin(K_EVENTS)
    out["is_sf"] = ev.isin(SF_EVENTS)
    out["is_hit"] = ev.isin(HIT_EVENTS)
    out["is_1b"] = ev.eq("single")
    out["is_2b"] = ev.eq("double")
    out["is_3b"] = ev.eq("triple")
    out["is_hr"] = ev.eq("home_run")
    out["is_sac_bunt"] = ev.eq("sac_bunt")
    out["is_ci"] = ev.eq("catcher_interf")
    out["is_ab"] = out["is_pa"] & ~(out["is_bb"] | out["is_hbp"] | out["is_sf"] | out["is_sac_bunt"] | out["is_ci"])
    return out


def aggregate_hitter_window(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    events = add_event_flags(df[df["events"].notna()].copy())
    keys = ["season", "batter"]
    if events.empty:
        return pd.DataFrame(columns=keys)

    grouped = events.groupby(keys)
    stats = grouped.agg(
        pa=("is_pa", "sum"),
        ab=("is_ab", "sum"),
        bb=("is_bb", "sum"),
        hbp=("is_hbp", "sum"),
        k=("is_k", "sum"),
        sf=("is_sf", "sum"),
        hits=("is_hit", "sum"),
        singles=("is_1b", "sum"),
        doubles=("is_2b", "sum"),
        triples=("is_3b", "sum"),
        hr=("is_hr", "sum"),
        woba_num=("woba_value", "sum"),
        woba_den=("woba_denom", "sum"),
        xwoba=("estimated_woba_using_speedangle", "mean"),
    ).reset_index()
    total_bases = stats["singles"] + 2 * stats["doubles"] + 3 * stats["triples"] + 4 * stats["hr"]
    bip_den = stats["ab"] - stats["k"] - stats["hr"] + stats["sf"]
    stats["avg"] = safe_divide(stats["hits"], stats["ab"])
    stats["obp"] = safe_divide(stats["hits"] + stats["bb"] + stats["hbp"], stats["ab"] + stats["bb"] + stats["hbp"] + stats["sf"])
    stats["slg"] = safe_divide(total_bases, stats["ab"])
    stats["ops"] = stats["obp"] + stats["slg"]
    stats["babip"] = safe_divide(stats["hits"] - stats["hr"], bip_den)
    stats["bb_rate"] = safe_divide(stats["bb"], stats["pa"])
    stats["k_rate"] = safe_divide(stats["k"], stats["pa"])
    stats["hr_rate"] = safe_divide(stats["hr"], stats["pa"])
    stats["iso"] = stats["slg"] - stats["avg"]
    stats["woba"] = safe_divide(stats["woba_num"], stats["woba_den"])

    pitches = df.copy()
    pitches["is_swing"] = pitches["description"].isin(SWING_DESCRIPTIONS)
    pitches["is_whiff"] = pitches["description"].isin(WHIFF_DESCRIPTIONS)
    pitches["is_called_strike"] = pitches["description"].eq("called_strike")
    pitches["in_zone"] = (pitches["plate_x"].abs() <= 0.83) & pitches["plate_z"].between(1.5, 3.5)
    pitch_stats = pitches.groupby(keys).agg(
        pitch_count=("description", "size"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        called_strikes=("is_called_strike", "sum"),
        in_zone=("in_zone", "sum"),
    ).reset_index()
    pitch_stats["swing_rate"] = safe_divide(pitch_stats["swings"], pitch_stats["pitch_count"])
    pitch_stats["whiff_per_swing"] = safe_divide(pitch_stats["whiffs"], pitch_stats["swings"])
    pitch_stats["called_strike_rate"] = safe_divide(pitch_stats["called_strikes"], pitch_stats["pitch_count"])
    pitch_stats["zone_rate"] = safe_divide(pitch_stats["in_zone"], pitch_stats["pitch_count"])

    outside = pitches[~pitches["in_zone"].fillna(False)]
    chase = outside.groupby(keys).agg(outside_pitches=("description", "size"), outside_swings=("is_swing", "sum")).reset_index()
    chase["chase_rate"] = safe_divide(chase["outside_swings"], chase["outside_pitches"])

    bbe = df[df["launch_speed"].notna()].copy()
    if len(bbe):
        bbe["hardhit"] = bbe["launch_speed"] >= 95
        bbe["barrel"] = pd.to_numeric(bbe["launch_speed_angle"], errors="coerce").eq(6)
        bbe_stats = bbe.groupby(keys).agg(
            bbe=("launch_speed", "size"),
            ev_mean=("launch_speed", "mean"),
            ev_p90=("launch_speed", lambda s: float(np.nanpercentile(s, 90))),
            la_mean=("launch_angle", "mean"),
            hardhit=("hardhit", "sum"),
            barrel=("barrel", "sum"),
        ).reset_index()
        bbe_stats["hardhit_rate"] = safe_divide(bbe_stats["hardhit"], bbe_stats["bbe"])
        bbe_stats["barrel_rate"] = safe_divide(bbe_stats["barrel"], bbe_stats["bbe"])
    else:
        bbe_stats = pd.DataFrame(columns=keys)

    out = stats.merge(pitch_stats, on=keys, how="left").merge(chase[[*keys, "chase_rate"]], on=keys, how="left").merge(
        bbe_stats, on=keys, how="left"
    )
    out["pitches_per_pa"] = safe_divide(out["pitch_count"], out["pa"])
    out["xwoba_minus_woba"] = out["xwoba"] - out["woba"]
    keep = keys + [
        "pa",
        "ab",
        "bb",
        "k",
        "hits",
        "hr",
        "woba",
        "avg",
        "obp",
        "slg",
        "ops",
        "babip",
        "bb_rate",
        "k_rate",
        "hr_rate",
        "iso",
        "xwoba",
        "xwoba_minus_woba",
        "pitches_per_pa",
        "swing_rate",
        "whiff_per_swing",
        "called_strike_rate",
        "zone_rate",
        "chase_rate",
        "bbe",
        "ev_mean",
        "ev_p90",
        "la_mean",
        "hardhit_rate",
        "barrel_rate",
    ]
    out = out[[col for col in keep if col in out.columns]]
    rename = {col: f"{col}_{prefix}" for col in out.columns if col not in keys}
    return out.rename(columns=rename)


def add_prior_features(df: pd.DataFrame, league: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["batter", "season"]).copy()
    league_lookup = league.set_index("season")
    for col in ["preseason_prior_woba", "preseason_prior_iso", "preseason_prior_k_rate"]:
        df[col] = np.nan
    weights = {1: 5.0, 2: 4.0, 3: 3.0}
    full_lookup = df.set_index(["batter", "season"])
    for idx, row in df.iterrows():
        batter = row["batter"]
        season = int(row["season"])
        vals: dict[str, list[tuple[float, float]]] = {"woba": [], "iso": [], "k_rate": []}
        for lag, weight in weights.items():
            key = (batter, season - lag)
            if key in full_lookup.index:
                prior = full_lookup.loc[key]
                vals["woba"].append((float(prior.get("woba_full", np.nan)), weight))
                vals["iso"].append((float(prior.get("iso_full", np.nan)), weight))
                vals["k_rate"].append((float(prior.get("k_rate_full", np.nan)), weight))
        for stat, target in [
            ("woba", "preseason_prior_woba"),
            ("iso", "preseason_prior_iso"),
            ("k_rate", "preseason_prior_k_rate"),
        ]:
            clean = [(v, w) for v, w in vals[stat] if np.isfinite(v)]
            if clean:
                df.at[idx, target] = sum(v * w for v, w in clean) / sum(w for _, w in clean)
            elif season in league_lookup.index:
                fallback = "league_woba" if stat == "woba" else "league_iso" if stat == "iso" else "league_k_rate"
                df.at[idx, target] = league_lookup.loc[season, fallback]
    return df


def make_hitter_features(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_frames = []
    league_frames = []
    for season in seasons:
        path = DATA_DIR / f"statcast_{season}.parquet"
        df = load_statcast(path)
        df["season"] = season
        df["batting_team"] = batting_team(df)
        df = assign_game_number(df, "batter")
        first = aggregate_hitter_window(df[df["game_n"].le(22)], "22g")
        full = aggregate_hitter_window(df, "full")
        ros = aggregate_hitter_window(df[df["game_n"].gt(22)], "ros")
        row = first.merge(full, on=["season", "batter"], how="left").merge(ros, on=["season", "batter"], how="left")
        for stat in ["woba", "avg", "obp", "slg", "ops", "babip", "bb_rate", "k_rate", "hr_rate", "iso"]:
            if f"{stat}_full" in row.columns:
                row[f"full_{stat}"] = row[f"{stat}_full"]
            if f"{stat}_ros" in row.columns:
                row[f"ros_{stat}"] = row[f"{stat}_ros"]
        teams = (
            df[df["events"].notna()]
            .groupby(["season", "batter"])["batting_team"]
            .agg(lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else pd.NA)
            .reset_index(name="team")
        )
        row = row.merge(teams, on=["season", "batter"], how="left")
        season_frames.append(row)
        lg = full.assign(season=season)
        totals = {
            "season": season,
            "league_woba": safe_divide(lg["woba_num_full"].sum(), lg["woba_den_full"].sum()) if "woba_num_full" in lg else lg["woba_full"].mean(),
            "league_bb_rate": safe_divide(lg["bb_full"].sum(), lg["pa_full"].sum()),
            "league_k_rate": safe_divide(lg["k_full"].sum(), lg["pa_full"].sum()),
            "league_babip": safe_divide((lg["hits_full"] - lg["hr_full"]).sum(), (lg["ab_full"] - lg["k_full"] - lg["hr_full"]).sum()),
            "league_iso": lg["iso_full"].mean(),
        }
        league_frames.append(totals)
    hitters = pd.concat(season_frames, ignore_index=True)
    league = pd.DataFrame(league_frames)
    hitters = hitters.merge(league, on="season", how="left")
    hitters = add_prior_features(hitters, league)
    for col in HITTER_FEATURE_COLS + HITTER_ANALOG_COLS + [
        "ros_woba",
        "full_woba",
        "ros_iso",
        "ros_babip",
        "ros_ops",
        "ros_hr_rate",
        "ros_k_rate",
    ]:
        if col not in hitters.columns:
            hitters[col] = np.nan
    return hitters, league


def infer_murakami_id(hitters_2026: pd.DataFrame) -> int | None:
    try:
        from data_pull import resolve_mlbam_id

        resolved = resolve_mlbam_id("Munetaka Murakami")
        if resolved is not None and hitters_2026["batter"].eq(resolved).any():
            return int(resolved)
    except Exception:
        pass

    candidates = hitters_2026.copy()
    if "team" in candidates.columns:
        candidates = candidates[candidates["team"].isin(["NYM", "CWS"])]
    candidates = candidates[~candidates["batter"].isin([v["mlbam"] for v in KNOWN_HITTERS.values() if v["mlbam"]])]
    candidates = candidates.sort_values(["hr_22g", "iso_22g", "pa_22g"], ascending=False)
    if len(candidates) and candidates.iloc[0].get("hr_22g", 0) >= 4:
        return int(candidates.iloc[0]["batter"])
    return None


def lookup_player_name(mlbam: int) -> str:
    try:
        names = reverse_name_map([mlbam])
        if len(names):
            return str(names.iloc[0]["name"]).strip().title()
    except Exception:
        pass
    return f"MLBAM {mlbam}"


def pick_hot_starter_substitute(hitters_2026: pd.DataFrame, excluded_ids: set[int]) -> dict | None:
    pool = hitters_2026[hitters_2026["pa_22g"].ge(50) & ~hitters_2026["batter"].isin(excluded_ids)].copy()
    if pool.empty:
        return None
    candidate = pool.sort_values(["avg_22g", "woba_22g", "pa_22g"], ascending=False).iloc[0]
    mlbam = int(candidate["batter"])
    name = lookup_player_name(mlbam)
    return {
        "key": slugify_player(name),
        "name": name,
        "mlbam": mlbam,
        "replacement_for": "Munetaka Murakami",
        "replacement_reason": "Murakami had no playerid_lookup result and no matching 2026 Statcast cutoff profile; selected top eligible BA leaderboard substitute.",
    }


def make_2026_hitter_features(league: pd.DataFrame, historical: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    path = DATA_DIR / "statcast_2026_through_apr25.parquet"
    if not path.exists():
        path = DATA_DIR / "statcast_2026_through_apr24.parquet"
    df = load_statcast(path)
    df["season"] = 2026
    df["batting_team"] = batting_team(df)
    df = assign_game_number(df, "batter")
    first = aggregate_hitter_window(df[df["game_n"].le(22)], "22g")
    teams = (
        df[df["events"].notna()]
        .groupby(["season", "batter"])["batting_team"]
        .agg(lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else pd.NA)
        .reset_index(name="team")
    )
    first = first.merge(teams, on=["season", "batter"], how="left")
    prev = historical[historical["season"].between(2023, 2025)].copy()
    league_2025 = league[league["season"].eq(2025)].iloc[0].to_dict()
    first["league_woba"] = league_2025["league_woba"]
    first["league_bb_rate"] = league_2025["league_bb_rate"]
    first["league_k_rate"] = league_2025["league_k_rate"]
    first["league_babip"] = league_2025["league_babip"]
    first["league_iso"] = league_2025["league_iso"]
    first["preseason_prior_woba"] = np.nan
    first["preseason_prior_iso"] = np.nan
    first["preseason_prior_k_rate"] = np.nan
    for idx, row in first.iterrows():
        batter = row["batter"]
        vals = prev[prev["batter"].eq(batter)].sort_values("season", ascending=False).head(3)
        weights = np.array([5, 4, 3], dtype=float)[: len(vals)]
        if len(vals):
            first.at[idx, "preseason_prior_woba"] = np.average(vals["woba_full"], weights=weights)
            first.at[idx, "preseason_prior_iso"] = np.average(vals["iso_full"], weights=weights)
            first.at[idx, "preseason_prior_k_rate"] = np.average(vals["k_rate_full"], weights=weights)
        else:
            first.at[idx, "preseason_prior_woba"] = league_2025["league_woba"]
            first.at[idx, "preseason_prior_iso"] = league_2025["league_iso"]
            first.at[idx, "preseason_prior_k_rate"] = league_2025["league_k_rate"]
    named = {key: dict(value) for key, value in KNOWN_HITTERS.items()}
    murakami_id = infer_murakami_id(first)
    named["munetaka_murakami"]["mlbam"] = murakami_id
    if murakami_id is None:
        named["munetaka_murakami"]["status"] = "excluded_missing_statcast_id"
        named["munetaka_murakami"][
            "exclusion_reason"
        ] = "No playerid_lookup result and no NYM hitter in the cutoff Statcast data matched the 7-HR Murakami profile."
        excluded_ids = {int(v["mlbam"]) for v in named.values() if v.get("mlbam") is not None}
        substitute = pick_hot_starter_substitute(first, excluded_ids)
        if substitute is not None:
            named[substitute.pop("key")] = substitute
    for col in HITTER_FEATURE_COLS + HITTER_ANALOG_COLS:
        if col not in first.columns:
            first[col] = np.nan
    return first, named


def outs_on_play(events: pd.Series) -> pd.Series:
    ev = events.astype(str)
    outs = np.zeros(len(ev), dtype=float)
    outs += ev.isin(["field_out", "force_out", "fielders_choice_out", "strikeout", "sac_bunt", "sac_fly"]).astype(float)
    outs += ev.isin(["grounded_into_double_play", "double_play", "strikeout_double_play", "sac_fly_double_play"]).astype(float) * 2
    outs += ev.eq("triple_play").astype(float) * 3
    return pd.Series(outs, index=events.index)


def aggregate_pitcher_window(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    events = add_event_flags(df[df["events"].notna()].copy())
    keys = ["season", "pitcher"]
    if events.empty:
        return pd.DataFrame(columns=keys)
    events["runs_allowed"] = (events["post_bat_score"] - events["bat_score"]).clip(lower=0).fillna(0)
    events["outs_recorded"] = outs_on_play(events["events"])
    grouped = events.groupby(keys)
    stats = grouped.agg(
        bf=("is_pa", "sum"),
        k=("is_k", "sum"),
        bb=("is_bb", "sum"),
        hr=("is_hr", "sum"),
        runs_allowed=("runs_allowed", "sum"),
        outs=("outs_recorded", "sum"),
    ).reset_index()
    stats["ip"] = stats["outs"] / 3.0
    stats["k_rate"] = safe_divide(stats["k"], stats["bf"])
    stats["bb_rate"] = safe_divide(stats["bb"], stats["bf"])
    stats["hr_rate"] = safe_divide(stats["hr"], stats["bf"])
    stats["ra9"] = safe_divide(stats["runs_allowed"] * 9, stats["ip"])

    pitches = df.copy()
    pitches["is_swing"] = pitches["description"].isin(SWING_DESCRIPTIONS)
    pitches["is_whiff"] = pitches["description"].isin(WHIFF_DESCRIPTIONS)
    pitches["in_zone"] = (pitches["plate_x"].abs() <= 0.83) & pitches["plate_z"].between(1.5, 3.5)
    pitch_stats = pitches.groupby(keys).agg(
        pitch_count=("description", "size"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        in_zone=("in_zone", "sum"),
    ).reset_index()
    pitch_stats["swing_rate"] = safe_divide(pitch_stats["swings"], pitch_stats["pitch_count"])
    pitch_stats["whiff_per_swing"] = safe_divide(pitch_stats["whiffs"], pitch_stats["swings"])
    pitch_stats["zone_rate"] = safe_divide(pitch_stats["in_zone"], pitch_stats["pitch_count"])

    bbe = df[df["launch_speed"].notna()].copy()
    if len(bbe):
        bbe["hardhit"] = bbe["launch_speed"] >= 95
        bbe_stats = bbe.groupby(keys).agg(
            bbe=("launch_speed", "size"),
            ev_p90_allowed=("launch_speed", lambda s: float(np.nanpercentile(s, 90))),
            hardhit_allowed=("hardhit", "sum"),
        ).reset_index()
        bbe_stats["hardhit_allowed_rate"] = safe_divide(bbe_stats["hardhit_allowed"], bbe_stats["bbe"])
    else:
        bbe_stats = pd.DataFrame(columns=keys)
    out = stats.merge(pitch_stats, on=keys, how="left").merge(bbe_stats, on=keys, how="left")
    out["pitches_per_bf"] = safe_divide(out["pitch_count"], out["bf"])
    rename = {col: f"{col}_{prefix}" for col in out.columns if col not in keys}
    return out.rename(columns=rename)


def make_pitcher_features(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    league_rows = []
    for season in seasons:
        df = load_statcast(DATA_DIR / f"statcast_{season}.parquet")
        df["season"] = season
        df = assign_game_number(df, "pitcher")
        first = aggregate_pitcher_window(df[df["game_n"].le(22)], "22g")
        full = aggregate_pitcher_window(df, "full")
        ros = aggregate_pitcher_window(df[df["game_n"].gt(22)], "ros")
        row = first.merge(full, on=["season", "pitcher"], how="left").merge(ros, on=["season", "pitcher"], how="left")
        for stat in ["ra9", "k_rate", "bb_rate", "hr_rate", "bf", "ip"]:
            if f"{stat}_full" in row.columns:
                row[f"full_{stat}"] = row[f"{stat}_full"]
            if f"{stat}_ros" in row.columns:
                row[f"ros_{stat}"] = row[f"{stat}_ros"]
        rows.append(row)
        league_rows.append(
            {
                "season": season,
                "league_pitcher_k_rate": safe_divide(full["k_full"].sum(), full["bf_full"].sum()),
                "league_pitcher_bb_rate": safe_divide(full["bb_full"].sum(), full["bf_full"].sum()),
                "league_pitcher_ra9": safe_divide(full["runs_allowed_full"].sum() * 9, full["ip_full"].sum()),
            }
        )
    pitchers = pd.concat(rows, ignore_index=True)
    league = pd.DataFrame(league_rows)
    pitchers = pitchers.merge(league, on="season", how="left")
    for col in PITCHER_FEATURE_COLS + PITCHER_ANALOG_COLS + ["ros_ra9", "ros_k_rate", "ros_bb_rate"]:
        if col not in pitchers.columns:
            pitchers[col] = np.nan
    return pitchers, league


def make_2026_pitcher_features(league_pitching: pd.DataFrame) -> pd.DataFrame:
    path = DATA_DIR / "statcast_2026_through_apr25.parquet"
    if not path.exists():
        path = DATA_DIR / "statcast_2026_through_apr24.parquet"
    df = load_statcast(path)
    df["season"] = 2026
    df = assign_game_number(df, "pitcher")
    first = aggregate_pitcher_window(df[df["game_n"].le(22)], "22g")
    lg = league_pitching[league_pitching["season"].eq(2025)].iloc[0]
    first["league_pitcher_k_rate"] = lg["league_pitcher_k_rate"]
    first["league_pitcher_bb_rate"] = lg["league_pitcher_bb_rate"]
    first["league_pitcher_ra9"] = lg["league_pitcher_ra9"]
    for col in PITCHER_FEATURE_COLS + PITCHER_ANALOG_COLS:
        if col not in first.columns:
            first[col] = np.nan
    return first


def reverse_name_map(ids: list[int]) -> pd.DataFrame:
    from pybaseball import playerid_reverse_lookup

    ids = sorted({int(i) for i in ids if pd.notna(i)})
    frames = []
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        try:
            frames.append(playerid_reverse_lookup(chunk, key_type="mlbam"))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame(columns=["key_mlbam", "name"])
    out = pd.concat(frames, ignore_index=True)
    out["name"] = out["name_first"].fillna("") + " " + out["name_last"].fillna("")
    return out[["key_mlbam", "name"]].drop_duplicates()


def compute_noise_floor(hitters: pd.DataFrame) -> dict:
    years = [2022, 2023, 2024, 2025]
    out: dict[str, dict] = {}
    for stat, col in [("BA", "avg"), ("OPS", "ops")]:
        records = []
        for season in years:
            pool = hitters[(hitters["season"].eq(season)) & (hitters["pa_22g"].ge(50))].copy()
            top = pool.nlargest(5, f"{col}_22g")
            for _, row in top.iterrows():
                first = row.get(f"{col}_22g")
                ros = row.get(f"{col}_ros")
                records.append(
                    {
                        "season": int(season),
                        "batter": int(row["batter"]),
                        "first22": float(first) if pd.notna(first) else None,
                        "ros": float(ros) if pd.notna(ros) else None,
                        "maintained_90pct": bool(pd.notna(first) and pd.notna(ros) and ros >= 0.9 * first),
                    }
                )
        out[stat] = {
            "n": len(records),
            "maintained_90pct_count": int(sum(r["maintained_90pct"] for r in records)),
            "maintained_90pct_rate": float(np.mean([r["maintained_90pct"] for r in records])) if records else None,
            "records": records,
        }
    return out


def main() -> None:
    seasons = list(range(2015, 2026))
    hitters, league = make_hitter_features(seasons)
    hitters = hitters[hitters["pa_22g"].notna()].copy()
    pitchers, league_pitching = make_pitcher_features(seasons)

    hitters_2026, named_hitters = make_2026_hitter_features(league, hitters)
    pitchers_2026 = make_2026_pitcher_features(league_pitching)
    named_pitchers = {key: dict(value) for key, value in KNOWN_PITCHERS.items()}

    ids = list(hitters["batter"].dropna().unique()) + list(pitchers["pitcher"].dropna().unique())
    ids += list(hitters_2026["batter"].dropna().unique()) + list(pitchers_2026["pitcher"].dropna().unique())
    names = reverse_name_map(ids)
    manual_rows = []
    for value in [*named_hitters.values(), *named_pitchers.values()]:
        if value.get("mlbam") is not None:
            manual_rows.append({"key_mlbam": int(value["mlbam"]), "name": value["name"]})
    names = pd.concat([names, pd.DataFrame(manual_rows)], ignore_index=True).drop_duplicates("key_mlbam", keep="last")

    atomic_write_parquet(hitters, DATASETS_DIR / "hitter_features.parquet")
    atomic_write_parquet(hitters_2026, DATASETS_DIR / "hitter_features_2026.parquet")
    atomic_write_parquet(pitchers, DATASETS_DIR / "pitcher_features.parquet")
    atomic_write_parquet(pitchers_2026, DATASETS_DIR / "pitcher_features_2026.parquet")
    atomic_write_parquet(league, DATASETS_DIR / "league_environment.parquet")
    atomic_write_parquet(league_pitching, DATASETS_DIR / "league_pitching_environment.parquet")
    names.to_csv(DATASETS_DIR / "name_map.csv", index=False)
    atomic_write_json(DATASETS_DIR / "named_players.json", {"hitters": named_hitters, "pitchers": named_pitchers})
    atomic_write_json(DATASETS_DIR / "noise_floor.json", compute_noise_floor(hitters))


if __name__ == "__main__":
    main()

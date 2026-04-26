from __future__ import annotations

import calendar
import sys
from urllib.parse import quote
from pathlib import Path

import pandas as pd
import requests

from common import (
    BASE_DIR,
    DATA_DIR,
    KNOWN_HITTERS,
    KNOWN_PITCHERS,
    STATCAST_COLUMNS,
    atomic_write_json,
    atomic_write_parquet,
    clean_statcast_frame,
    statcast_cache_valid,
)


LOCAL_SEASON_SOURCES = {
    2022: [
        Path("../coaching-gap/data/harmonized/raw/statcast_2022.parquet")
    ],
    2023: [
        Path("../coaching-gap/data/harmonized/raw/statcast_2023.parquet")
    ],
    2024: [
        Path("../coaching-gap/data/harmonized/raw/statcast_2024.parquet")
    ],
    2025: [
        Path("../pitch-tunneling-atlas/data/statcast_2025_full.parquet"),
        Path("../schlittler-arsenal/data/statcast_2025_full.parquet"),
        Path("../count-distribution-abs/data/statcast_2025_full.parquet"),
    ],
}

SOURCE_2026_BASE = Path(
    "../abs-walk-spike/data/statcast_2026_mar27_apr22.parquet"
)

MLB_STATS_CACHE = DATA_DIR / "mlb_stats_api_player_cache.json"


def month_chunks(season: int) -> list[tuple[str, str]]:
    chunks = []
    for month in range(3, 11):
        start = f"{season}-{month:02d}-01"
        end = f"{season}-{month:02d}-{calendar.monthrange(season, month)[1]:02d}"
        chunks.append((start, end))
    return chunks


def read_local_source(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        print(f"[data_pull] local source unreadable {path}: {exc}", file=sys.stderr)
        return None
    missing = [col for col in ["pitch_type", "plate_x", "plate_z", "pitch_number", "inning"] if col not in df.columns]
    if missing:
        print(f"[data_pull] local source lacks {missing}: {path}", file=sys.stderr)
        return None
    return clean_statcast_frame(df)


def fetch_statcast_range(start_dt: str, end_dt: str) -> pd.DataFrame:
    from pybaseball import statcast

    print(f"[data_pull] fetching Statcast {start_dt}..{end_dt}", flush=True)
    df = statcast(start_dt=start_dt, end_dt=end_dt)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=STATCAST_COLUMNS)
    return clean_statcast_frame(df)


def _read_mlb_stats_cache() -> dict:
    if not MLB_STATS_CACHE.exists() or MLB_STATS_CACHE.stat().st_size == 0:
        return {}
    try:
        import json

        return json.loads(MLB_STATS_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_mlb_stats_cache(cache: dict) -> None:
    atomic_write_json(MLB_STATS_CACHE, cache)


def resolve_mlbam_id_via_stats_api(name: str) -> dict:
    """Resolve a player name through MLB Stats API and cache the raw match.

    `pybaseball.playerid_lookup` uses a Chadwick/Lahman-style crosswalk and can
    lag MLB debuts. The Stats API search endpoint is the reproducible fallback
    required for 2026 debut cases such as Munetaka Murakami.
    """

    cache = _read_mlb_stats_cache()
    key = name.strip().lower()
    if key in cache:
        return cache[key]

    url = f"https://statsapi.mlb.com/api/v1/people/search?names={quote(name)}"
    record = {"requested_name": name, "lookup_status": "not_found", "url": url, "id": None}
    try:
        response = requests.get(url, timeout=20)
        record["status_code"] = response.status_code
        response.raise_for_status()
        payload = response.json()
        people = payload.get("people", [])
        if people:
            exact = [
                person
                for person in people
                if str(person.get("fullName", "")).strip().lower() == key or str(person.get("nameFirstLast", "")).strip().lower() == key
            ]
            person = exact[0] if exact else people[0]
            record.update(
                {
                    "lookup_status": "ok",
                    "id": person.get("id"),
                    "fullName": person.get("fullName") or person.get("nameFirstLast"),
                    "active": person.get("active"),
                    "mlbDebutDate": person.get("mlbDebutDate"),
                    "currentTeam": (person.get("currentTeam") or {}).get("name"),
                    "primaryPosition": (person.get("primaryPosition") or {}).get("abbreviation"),
                }
            )
    except Exception as exc:
        record["lookup_status"] = f"error:{exc}"

    cache[key] = record
    _write_mlb_stats_cache(cache)
    return record


def resolve_mlbam_id(name: str, fallback_mlbam: int | None = None) -> int | None:
    resolved = resolve_mlbam_id_via_stats_api(name)
    if resolved.get("id") is not None:
        return int(resolved["id"])
    return fallback_mlbam


def ensure_season_statcast(season: int) -> Path:
    out = DATA_DIR / f"statcast_{season}.parquet"
    valid, reason = statcast_cache_valid(out, min_rows=50000 if season != 2020 else 15000)
    if valid:
        print(f"[data_pull] valid season cache {out}")
        return out
    print(f"[data_pull] repairing season cache {out}: {reason}")

    for source in LOCAL_SEASON_SOURCES.get(season, []):
        df = read_local_source(source)
        if df is not None and len(df) > 0:
            atomic_write_parquet(df, out)
            print(f"[data_pull] wrote {out} from local source {source} rows={len(df):,}")
            return out

    frames = []
    for start, end in month_chunks(season):
        try:
            frame = fetch_statcast_range(start, end)
        except Exception as exc:
            print(f"[data_pull] fetch failed {start}..{end}: {exc}", file=sys.stderr)
            continue
        if len(frame):
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No Statcast data fetched for {season}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"], keep="first"
    )
    atomic_write_parquet(df, out)
    print(f"[data_pull] wrote fetched season cache {out} rows={len(df):,}")
    return out


def ensure_2026_extension() -> Path:
    out = DATA_DIR / "statcast_2026_apr23_24.parquet"
    valid, reason = statcast_cache_valid(out, min_rows=1000)
    if valid:
        print(f"[data_pull] valid 2026 extension {out}")
        return out
    print(f"[data_pull] fetching 2026 extension because cache is {reason}")
    df = fetch_statcast_range("2026-04-23", "2026-04-24")
    if len(df) == 0:
        raise RuntimeError("pybaseball returned no rows for 2026-04-23..2026-04-24")
    atomic_write_parquet(df, out)
    print(f"[data_pull] wrote 2026 extension rows={len(df):,}")
    return out


def ensure_2026_extension_through_apr25() -> tuple[Path, str | None]:
    """Fetch only Apr 24-25 incrementally and record the actual max game date.

    Apr 25 games may not be complete when the pipeline runs. We still query
    through Apr 25, but if MLBAM has no Apr 25 rows yet, the combined R2 file
    truthfully records Apr 24 as the latest available game date.
    """

    out = DATA_DIR / "statcast_2026_apr24_25.parquet"
    valid, reason = statcast_cache_valid(out, min_rows=1000)
    if valid:
        df = pd.read_parquet(out, columns=["game_date"])
        max_date = pd.to_datetime(df["game_date"], errors="coerce").max()
        return out, max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None

    print(f"[data_pull] fetching Apr 24-25 extension because cache is {reason}")
    df = fetch_statcast_range("2026-04-24", "2026-04-25")
    if len(df) == 0:
        atomic_write_parquet(df, out)
        print("[data_pull] Apr 24-25 extension returned no rows")
        return out, None
    atomic_write_parquet(df, out)
    max_date = pd.to_datetime(df["game_date"], errors="coerce").max()
    actual = max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None
    print(f"[data_pull] wrote Apr 24-25 extension rows={len(df):,}; max game_date={actual}")
    return out, actual


def ensure_2026_combined() -> Path:
    extension = ensure_2026_extension()
    out_base = DATA_DIR / "statcast_2026_mar27_apr22.parquet"
    out = DATA_DIR / "statcast_2026_through_apr24.parquet"

    valid, reason = statcast_cache_valid(out, min_rows=50000)
    if valid:
        print(f"[data_pull] valid 2026 combined cache {out}")
        return out

    if not SOURCE_2026_BASE.exists() or SOURCE_2026_BASE.stat().st_size == 0:
        raise FileNotFoundError(f"Missing supplied 2026 source: {SOURCE_2026_BASE}")
    base = clean_statcast_frame(pd.read_parquet(SOURCE_2026_BASE))
    atomic_write_parquet(base, out_base)
    ext = clean_statcast_frame(pd.read_parquet(extension))
    df = pd.concat([base, ext], ignore_index=True).drop_duplicates(
        subset=["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"], keep="first"
    )
    atomic_write_parquet(df, out)
    print(f"[data_pull] wrote 2026 combined cache {out} rows={len(df):,}; prior reason={reason}")
    return out


def ensure_2026_combined_through_apr25() -> Path:
    extension_23_24 = ensure_2026_extension()
    extension_24_25, actual_extension_cutoff = ensure_2026_extension_through_apr25()
    out_base = DATA_DIR / "statcast_2026_mar27_apr22.parquet"
    out = DATA_DIR / "statcast_2026_through_apr25.parquet"

    valid, reason = statcast_cache_valid(out, min_rows=50000)
    if valid:
        print(f"[data_pull] valid 2026 R2 combined cache {out}")
        return out

    if not SOURCE_2026_BASE.exists() or SOURCE_2026_BASE.stat().st_size == 0:
        raise FileNotFoundError(f"Missing supplied 2026 source: {SOURCE_2026_BASE}")
    base = clean_statcast_frame(pd.read_parquet(SOURCE_2026_BASE))
    atomic_write_parquet(base, out_base)
    frames = [base]
    for path in [extension_23_24, extension_24_25]:
        frame = clean_statcast_frame(pd.read_parquet(path))
        if len(frame):
            frames.append(frame)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"], keep="first"
    )
    atomic_write_parquet(df, out)
    max_date = pd.to_datetime(df["game_date"], errors="coerce").max()
    actual_cutoff = max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else actual_extension_cutoff
    atomic_write_json(
        DATA_DIR / "statcast_2026_r2_cutoff_manifest.json",
        {
            "requested_cutoff": "2026-04-25",
            "actual_max_game_date": actual_cutoff,
            "apr25_rows_available": bool(actual_cutoff == "2026-04-25"),
            "rows": int(len(df)),
            "prior_reason_if_rebuilt": reason,
        },
    )
    print(f"[data_pull] wrote 2026 R2 combined cache {out} rows={len(df):,}; max game_date={actual_cutoff}")
    return out


def ensure_fg_stat_tables() -> None:
    from pybaseball import batting_stats, pitching_stats

    for season in range(2015, 2026):
        batting_out = DATA_DIR / f"batting_stats_{season}.parquet"
        if not batting_out.exists() or batting_out.stat().st_size == 0:
            try:
                print(f"[data_pull] fetching batting_stats {season}")
                atomic_write_parquet(batting_stats(season, qual=1), batting_out)
            except Exception as exc:
                print(f"[data_pull] batting_stats failed {season}: {exc}", file=sys.stderr)
                atomic_write_parquet(fallback_batting_stats(season), batting_out)
        pitching_out = DATA_DIR / f"pitching_stats_{season}.parquet"
        if not pitching_out.exists() or pitching_out.stat().st_size == 0:
            try:
                print(f"[data_pull] fetching pitching_stats {season}")
                atomic_write_parquet(pitching_stats(season, qual=1), pitching_out)
            except Exception as exc:
                print(f"[data_pull] pitching_stats failed {season}: {exc}", file=sys.stderr)
                atomic_write_parquet(fallback_pitching_stats(season), pitching_out)


def fallback_batting_stats(season: int) -> pd.DataFrame:
    from features import aggregate_hitter_window, batting_team

    df = clean_statcast_frame(pd.read_parquet(DATA_DIR / f"statcast_{season}.parquet"))
    df["season"] = season
    df["batting_team"] = batting_team(df)
    stats = aggregate_hitter_window(df, "full")
    stats["Season"] = season
    stats["IDfg"] = pd.NA
    stats["MLBAMID"] = stats["batter"]
    stats["Name"] = "MLBAM " + stats["batter"].astype("Int64").astype(str)
    stats["source"] = "statcast_fallback_after_pybaseball_403"
    return stats


def fallback_pitching_stats(season: int) -> pd.DataFrame:
    from features import aggregate_pitcher_window

    df = clean_statcast_frame(pd.read_parquet(DATA_DIR / f"statcast_{season}.parquet"))
    df["season"] = season
    stats = aggregate_pitcher_window(df, "full")
    stats["Season"] = season
    stats["IDfg"] = pd.NA
    stats["MLBAMID"] = stats["pitcher"]
    stats["Name"] = "MLBAM " + stats["pitcher"].astype("Int64").astype(str)
    stats["source"] = "statcast_fallback_after_pybaseball_403"
    return stats


def ensure_player_lookups() -> None:
    from pybaseball import playerid_lookup

    rows = []
    for info in [*KNOWN_HITTERS.values(), *KNOWN_PITCHERS.values()]:
        first, last = info["name"].split(" ", 1)
        try:
            lookup = playerid_lookup(last, first)
            if len(lookup):
                record = lookup.iloc[0].to_dict()
                record["requested_name"] = info["name"]
                record["lookup_status"] = "pybaseball_ok"
                rows.append(record)
            else:
                stats_record = resolve_mlbam_id_via_stats_api(info["name"])
                rows.append(
                    {
                        "requested_name": info["name"],
                        "key_mlbam": stats_record.get("id") or info.get("mlbam"),
                        "lookup_status": "stats_api_" + str(stats_record.get("lookup_status")),
                        "stats_api_full_name": stats_record.get("fullName"),
                        "stats_api_url": stats_record.get("url"),
                    }
                )
        except Exception as exc:
            stats_record = resolve_mlbam_id_via_stats_api(info["name"])
            rows.append(
                {
                    "requested_name": info["name"],
                    "key_mlbam": stats_record.get("id") or info.get("mlbam"),
                    "lookup_status": f"pybaseball_error:{exc};stats_api_{stats_record.get('lookup_status')}",
                    "stats_api_full_name": stats_record.get("fullName"),
                    "stats_api_url": stats_record.get("url"),
                }
            )
    pd.DataFrame(rows).to_csv(DATA_DIR / "player_id_lookups.csv", index=False)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for season in range(2015, 2026):
        ensure_season_statcast(season)
    ensure_2026_combined()
    ensure_2026_combined_through_apr25()
    ensure_fg_stat_tables()
    ensure_player_lookups()
    atomic_write_json(
        BASE_DIR / "data_pull_manifest.json",
        {
            "statcast_columns": STATCAST_COLUMNS,
            "seasons_cached": list(range(2015, 2026)),
            "cutoff_2026": "requested 2026-04-25; see data/statcast_2026_r2_cutoff_manifest.json for actual max game_date",
            "preseason_projection_source": "3-year weighted MLB mean fallback; pybaseball projection endpoint not used",
        },
    )


if __name__ == "__main__":
    main()

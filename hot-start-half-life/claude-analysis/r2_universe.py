"""r2_universe.py — build 2026 hitter and reliever universe with required filters.

Hitters: every batter with >= 50 PA in 2026 through cutoff.
Relievers: every pitcher with >= 25 BF in 2026 AND < 30 IP (excludes starters).
  - IP estimate: 1 IP = 3 outs. Outs ≈ ABs - hits + DPs + sac flies + sac bunts ≈
    PA - on_base - HBP. Approximation; exact IP requires play-by-play game logs.
    For "starter exclusion" we use BF >= 60 with > 4.0 BF/appearance avg as a
    signal that the player has any starts. Or, simpler, average BF per game.

Reliever filter logic:
  - Each pitcher's distinct (game_pk) appearances counted.
  - BF/appearance > 18 -> classified as starter.
  - Anything BF/appearance <= 12 -> reliever.
  - 12-18 -> ambiguous (excluded by default).

Outputs:
  data/r2_hitter_universe.parquet
  data/r2_reliever_universe.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import numpy as np

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"

sys.path.insert(0, str(CLAUDE))
from stabilization import PA_EVENTS

CUTOFF = "2026-04-25"


def load_2026_pa(extended: bool = True) -> pd.DataFrame:
    """Load 2026 PA-level statcast through cutoff."""
    f = DATA / ("statcast_2026_through_apr25.parquet" if extended else "statcast_2026_through_apr24.parquet")
    if not f.exists():
        f = DATA / "statcast_2026_through_apr24.parquet"
    df = pd.read_parquet(f)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["game_date"] <= pd.Timestamp(CUTOFF)]
    df = df[df["events"].notna() & df["events"].isin(PA_EVENTS)]
    df = df.drop_duplicates(subset=["game_pk", "batter", "at_bat_number"]).reset_index(drop=True)
    df["season"] = 2026
    return df


def annotate_hitter_pa(pa: pd.DataFrame) -> pd.DataFrame:
    pa = pa.copy()
    pa["is_pa"] = 1
    pa["is_bb"] = pa["events"].isin({"walk", "intent_walk"}).astype(int)
    pa["is_k"] = pa["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
    pa["is_hbp"] = (pa["events"] == "hit_by_pitch").astype(int)
    pa["is_sac"] = pa["events"].isin({"sac_fly", "sac_fly_double_play",
                                       "sac_bunt", "sac_bunt_double_play"}).astype(int)
    pa["is_ab"] = (pa["is_pa"] & ~(pa["is_bb"] | pa["is_hbp"] | pa["is_sac"]
                  | (pa["events"] == "catcher_interf"))).astype(int)
    pa["is_1b"] = (pa["events"] == "single").astype(int)
    pa["is_2b"] = (pa["events"] == "double").astype(int)
    pa["is_3b"] = (pa["events"] == "triple").astype(int)
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_hit"] = pa[["is_1b", "is_2b", "is_3b", "is_hr"]].sum(axis=1)
    pa["total_bases"] = pa["is_1b"] + 2 * pa["is_2b"] + 3 * pa["is_3b"] + 4 * pa["is_hr"]
    pa["babip_num"] = (pa["is_hit"] - pa["is_hr"]).clip(lower=0)
    pa["babip_den"] = (pa["is_ab"] - pa["is_k"] - pa["is_hr"]
                       + pa["events"].isin({"sac_fly", "sac_fly_double_play"}).astype(int)).clip(lower=0)
    pa["iso_num"] = (pa["total_bases"] - pa["is_hit"]).clip(lower=0)
    pa["iso_den"] = pa["is_ab"]
    pa["woba_num"] = pa["woba_value"].fillna(0.0).astype(float)
    pa["woba_den"] = pa["woba_denom"].fillna(0).astype(float)
    # Contact-quality (BIP only)
    pa["is_bip"] = (pa["bb_type"].notna()
                    & ~pa["events"].isin({"strikeout", "strikeout_double_play",
                                          "walk", "intent_walk", "hit_by_pitch"})).astype(int)
    ls = pd.to_numeric(pa["launch_speed"], errors="coerce")
    pa["is_hard_hit"] = ((ls >= 95) & (pa["is_bip"] == 1)).fillna(False).astype(int)
    lsa = pd.to_numeric(pa["launch_speed_angle"], errors="coerce")
    pa["is_barrel"] = (lsa == 6).fillna(False).astype(int)  # MLB barrel coding
    pa["xwoba_num"] = pa["estimated_woba_using_speedangle"].fillna(0.0).astype(float)
    # xwOBA denom: for non-BIP events that have known wOBA contribution (BB, K, HBP)
    # we use woba_value as part of the xwOBA calc. Standard MLB definition:
    # xwOBA includes BB, K, HBP at standard weights, plus xwOBA-ip on contact.
    # estimated_woba_using_speedangle is non-null only for BIP. We approximate xwOBA
    # at the PA level by replacing BIP wOBA contributions with xwOBA-on-contact.
    pa["xwoba_per_pa_num"] = np.where(
        pa["is_bip"] == 1,
        pa["xwoba_num"],
        pa["woba_num"],
    )
    pa["xwoba_per_pa_den"] = pa["woba_den"].astype(float)
    return pa


def annotate_pitcher_pa(pa: pd.DataFrame) -> pd.DataFrame:
    pa = pa.copy()
    pa["is_pa"] = 1
    pa["is_bb"] = pa["events"].isin({"walk", "intent_walk"}).astype(int)
    pa["is_k"] = pa["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_hbp"] = (pa["events"] == "hit_by_pitch").astype(int)
    pa["is_baserunner_allowed"] = pa["events"].isin(
        {"single", "double", "triple", "home_run", "walk", "intent_walk", "hit_by_pitch"}
    ).astype(int)
    pa["is_1b"] = (pa["events"] == "single").astype(int)
    pa["is_2b"] = (pa["events"] == "double").astype(int)
    pa["is_3b"] = (pa["events"] == "triple").astype(int)
    pa["is_hit"] = pa[["is_1b", "is_2b", "is_3b", "is_hr"]].sum(axis=1)
    pa["is_ab"] = (pa["is_pa"]
                   & ~pa["events"].isin({"walk", "intent_walk", "hit_by_pitch",
                                         "sac_fly", "sac_fly_double_play",
                                         "sac_bunt", "sac_bunt_double_play",
                                         "catcher_interf"})).astype(int)
    pa["is_out"] = ((pa["is_ab"] == 1) & (pa["is_hit"] == 0)).astype(int)
    return pa


def build_hitter_universe(pa26: pd.DataFrame, *, min_pa: int = 50) -> pd.DataFrame:
    """Aggregate per-batter 2026 stats over PA filter."""
    g = pa26.groupby("batter").agg(
        PA=("is_pa", "sum"),
        BB=("is_bb", "sum"),
        K=("is_k", "sum"),
        HBP=("is_hbp", "sum"),
        AB=("is_ab", "sum"),
        SAC=("is_sac", "sum"),
        H_1B=("is_1b", "sum"),
        H_2B=("is_2b", "sum"),
        H_3B=("is_3b", "sum"),
        HR=("is_hr", "sum"),
        H=("is_hit", "sum"),
        TB=("total_bases", "sum"),
        BABIP_num=("babip_num", "sum"),
        BABIP_den=("babip_den", "sum"),
        ISO_num=("iso_num", "sum"),
        ISO_den=("iso_den", "sum"),
        wOBA_num=("woba_num", "sum"),
        wOBA_den=("woba_den", "sum"),
        BIP=("is_bip", "sum"),
        HardHit=("is_hard_hit", "sum"),
        Barrel=("is_barrel", "sum"),
        xwOBA_num=("xwoba_per_pa_num", "sum"),
        xwOBA_den=("xwoba_per_pa_den", "sum"),
        last_game=("game_date", "max"),
        first_game=("game_date", "min"),
        # EV percentiles only meaningful with enough BIP
        EV_n=("launch_speed", "count"),
    ).reset_index()
    g = g[g["PA"] >= min_pa].copy()
    # Rate computations
    g["BBpct"] = g["BB"] / g["PA"]
    g["Kpct"] = g["K"] / g["PA"]
    g["BABIP"] = g["BABIP_num"] / g["BABIP_den"].replace(0, np.nan)
    g["ISO"] = g["ISO_num"] / g["ISO_den"].replace(0, np.nan)
    g["wOBA"] = g["wOBA_num"] / g["wOBA_den"].replace(0, np.nan)
    g["xwOBA"] = g["xwOBA_num"] / g["xwOBA_den"].replace(0, np.nan)
    g["xwOBA_minus_wOBA"] = g["xwOBA"] - g["wOBA"]
    g["HardHitPct"] = g["HardHit"] / g["BIP"].replace(0, np.nan)
    g["BarrelPct"] = g["Barrel"] / g["BIP"].replace(0, np.nan)
    # EV p90 by player
    ev_p90 = pa26[pa26["is_bip"] == 1].groupby("batter")["launch_speed"].quantile(0.90).rename("EV_p90")
    g = g.merge(ev_p90, on="batter", how="left")
    # Player name via Chadwick register (Statcast's player_name is the PITCHER's name)
    g = _attach_names(g, "batter")
    g = g.sort_values("PA", ascending=False).reset_index(drop=True)
    return g


def _attach_names(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Attach player_name from Chadwick register, then patch with MLB Stats API for misses."""
    out = DATA / "chadwick_register_cache.parquet"
    if out.exists():
        ch = pd.read_parquet(out)
    else:
        from pybaseball import chadwick_register
        ch = chadwick_register()
        keep = ["key_mlbam", "name_first", "name_last", "mlb_played_last"]
        ch = ch[keep].copy()
        ch.to_parquet(out, index=False)
    ch = ch.drop_duplicates("key_mlbam", keep="last").copy()
    ch["player_name"] = (ch["name_last"].fillna("") + ", " + ch["name_first"].fillna("")).str.strip(", ")
    df = df.merge(ch[["key_mlbam", "player_name"]], left_on=id_col, right_on="key_mlbam", how="left")
    df = df.drop(columns=["key_mlbam"])
    # Patch missing names via MLB Stats API
    missing_ids = df.loc[df["player_name"].isna(), id_col].unique().tolist()
    if missing_ids:
        import requests, json as _json
        cache_f = DATA / "mlbam_id_to_name_cache.json"
        cache = {}
        if cache_f.exists():
            try:
                cache = _json.load(open(cache_f))
            except Exception:
                cache = {}
        patched = {}
        for mid in missing_ids:
            mid_s = str(int(mid))
            if mid_s in cache:
                patched[int(mid)] = cache[mid_s]
                continue
            try:
                r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{int(mid)}", timeout=10)
                if r.ok:
                    data = r.json()
                    people = data.get("people", [])
                    if people:
                        full = people[0].get("fullName", "")
                        # Convert "First Last" to "Last, First"
                        parts = full.split(maxsplit=1)
                        if len(parts) == 2:
                            label = f"{parts[1]}, {parts[0]}"
                        else:
                            label = full
                        cache[mid_s] = label
                        patched[int(mid)] = label
            except Exception:
                pass
        cache_f.write_text(_json.dumps(cache, indent=2, sort_keys=True))
        if patched:
            df["player_name"] = df.apply(
                lambda r: patched.get(int(r[id_col]), r["player_name"]) if pd.isna(r["player_name"]) else r["player_name"],
                axis=1,
            )
    return df


def build_reliever_universe(pa26: pd.DataFrame, *,
                            min_bf: int = 25,
                            max_bf_per_app: float = 12.0) -> pd.DataFrame:
    """Aggregate per-pitcher 2026 stats; filter to relievers via BF/appearance."""
    g = pa26.groupby("pitcher").agg(
        BF=("is_pa", "sum"),
        BB=("is_bb", "sum"),
        K=("is_k", "sum"),
        HR_allowed=("is_hr", "sum"),
        HBP=("is_hbp", "sum"),
        baserunners=("is_baserunner_allowed", "sum"),
        outs=("is_out", "sum"),
        AB=("is_ab", "sum"),
        H=("is_hit", "sum"),
        appearances=("game_pk", "nunique"),
        last_game=("game_date", "max"),
        first_game=("game_date", "min"),
    ).reset_index()
    g["bf_per_app"] = g["BF"] / g["appearances"].clip(lower=1)
    # Reliever classification: avg BF/appearance <= max_bf_per_app
    g = g[(g["BF"] >= min_bf) & (g["bf_per_app"] <= max_bf_per_app)].copy()
    # Rate stats
    g["BBpct"] = g["BB"] / g["BF"]
    g["Kpct"] = g["K"] / g["BF"]
    g["IP_est"] = g["outs"] / 3.0
    # Filter < 30 IP (per brief)
    g = g[g["IP_est"] < 30].copy()
    # Pitcher names — for pitcher data, statcast player_name IS the pitcher's name,
    # but to be safe and consistent we still use Chadwick.
    g = _attach_names(g, "pitcher")
    g = g.sort_values("BF", ascending=False).reset_index(drop=True)
    return g


def build_2026_pa_table_with_features(extended: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (annotated_hitter_pa_2026, annotated_pitcher_pa_2026).

    Pitcher-perspective and hitter-perspective annotations differ; we keep both.
    """
    pa = load_2026_pa(extended=extended)
    pah = annotate_hitter_pa(pa)
    pap = annotate_pitcher_pa(pa)
    return pah, pap


def run() -> dict:
    print("[universe] loading 2026 PA")
    pah, pap = build_2026_pa_table_with_features(extended=True)
    print(f"[universe] {len(pah):,} 2026 PA total")
    hitters = build_hitter_universe(pah, min_pa=50)
    relievers = build_reliever_universe(pap, min_bf=25, max_bf_per_app=12.0)
    print(f"[universe] hitters >= 50 PA: {len(hitters)}")
    print(f"[universe] relievers >= 25 BF, <= 12 BF/app, < 30 IP: {len(relievers)}")
    if len(hitters) < 250:
        print(f"  NOTE: hitter universe under brief threshold of 250; lowering not auto-applied; documented")
    if len(relievers) < 70:
        print(f"  NOTE: reliever universe under brief threshold of 70")

    hitters.to_parquet(DATA / "r2_hitter_universe.parquet", index=False)
    relievers.to_parquet(DATA / "r2_reliever_universe.parquet", index=False)
    print(f"[universe] saved")
    return {
        "n_hitters": int(len(hitters)),
        "n_relievers": int(len(relievers)),
        "cutoff": CUTOFF,
        "min_pa_hitter": 50,
        "min_bf_reliever": 25,
        "max_bf_per_app_reliever": 12.0,
        "max_ip_reliever": 30,
    }


if __name__ == "__main__":
    s = run()
    print(s)

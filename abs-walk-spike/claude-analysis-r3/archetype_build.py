"""archetype_build — build per-pitcher 2025 stuff+/command+ lookup.

Strategy:
  1. Prefer FanGraphs leaderboard via API. (Round 3 brief: try first.)
  2. If FanGraphs is unavailable (403/Cloudflare), fall back to a *proxy* built from
     2025 Statcast:
       - stuff_pct: arsenal-weighted whiff-rate percentile
         (per pitch_type within pitcher: whiff_rate; weighted average across the
          pitcher's pitches, then percentile-rank the result)
       - command_pct: percentile of [zone_pct − bb_pct], i.e., high zone% AND
         low walk% → high command+. We normalize each component to a percentile,
         then average.
       - data_source = "fangraphs" or "proxy"
  3. Save to R3_DATA / "pitcher_archetype_2025.parquet" with columns:
       pitcher_id, name, stuff_pct, command_pct, stuff_minus_command, data_source

Sample threshold for inclusion: ≥40 IP in 2025.

Note: we tried `pybaseball.pitching_stats(2025,2025,qual=40)` for FanGraphs but
hit a 403 (Cloudflare bot protection). The proxy is then the only path. We
document that explicitly in the data_source column.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import R3_DATA, ensure_dirs, load_2025_full, rulebook_zone_flag, WALK_EVENTS


ARCHETYPE_PATH = R3_DATA / "pitcher_archetype_2025.parquet"
ARCHETYPE_META_PATH = R3_DATA / "pitcher_archetype_meta.json"


def _try_fangraphs_pitching_stats() -> pd.DataFrame | None:
    """Try fetching FanGraphs 2025 leaderboard (Stuff+, Location+, Pitching+).

    Returns DataFrame on success with cols stuff_pct, command_pct or None on failure.
    """
    try:
        from pybaseball import pitching_stats
        df = pitching_stats(2025, 2025, qual=40, ind=1)
        if df is None or len(df) == 0:
            return None
        # Stuff+, Location+ are FanGraphs columns (Driveline-derived).
        # If present, we map: stuff+ -> stuff_pct (already-scaled, so just rank); location+ -> command_pct
        keep = {"IDfg", "Name", "MLBAMID", "Stuff+", "Location+", "Pitching+", "BB%", "IP"}
        cols = [c for c in df.columns if c in keep]
        if "Stuff+" not in cols or "Location+" not in cols:
            return None
        out = df[cols].copy()
        out["stuff_pct"] = out["Stuff+"].rank(pct=True)
        out["command_pct"] = out["Location+"].rank(pct=True)
        out["data_source"] = "fangraphs"
        # MLBAMID → pitcher_id; Name → name
        if "MLBAMID" not in out.columns:
            return None
        out = out.rename(columns={"MLBAMID": "pitcher_id", "Name": "name"})
        return out[["pitcher_id", "name", "stuff_pct", "command_pct", "data_source"]].dropna()
    except Exception as e:
        print(f"[archetype] FanGraphs path failed: {e}")
        return None


# Whiff-eligible swing descriptions
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt", "foul_tip"}
SWING_DESCS = WHIFF_DESCS | {"foul", "hit_into_play", "foul_bunt", "foul_pitchout"}


def _build_proxy_from_statcast() -> pd.DataFrame:
    print("[archetype] building proxy from 2025 Statcast...")
    df = load_2025_full()
    # Eligibility: ≥40 IP. Use outs-from-events as IP proxy.
    df["is_term"] = df["events"].notna() & (df["events"].astype(str) != "None") & (df["events"].astype(str).str.len() > 0)
    out_evs = {
        "field_out", "strikeout", "grounded_into_double_play", "force_out",
        "sac_fly", "sac_bunt", "double_play", "fielders_choice_out",
        "strikeout_double_play", "sac_fly_double_play",
    }
    df["n_outs"] = 0
    df.loc[df["events"].isin(out_evs), "n_outs"] = 1
    df.loc[df["events"].isin({"double_play", "grounded_into_double_play", "strikeout_double_play"}), "n_outs"] = 2
    ip_table = df.groupby("pitcher")["n_outs"].sum() / 3.0
    eligible = ip_table[ip_table >= 40.0].index.tolist()
    print(f"[archetype] {len(eligible)} pitchers with ≥40 IP in 2025")

    df_e = df.loc[df["pitcher"].isin(eligible)].copy()

    # ----- STUFF+ proxy: arsenal-weighted whiff rate
    # For each pitch_type within pitcher: whiff% = whiff / swings
    # Then arsenal-weight by usage and rank-pct
    df_e["is_whiff"] = df_e["description"].isin(WHIFF_DESCS).astype(int)
    df_e["is_swing"] = df_e["description"].isin(SWING_DESCS).astype(int)
    pt_grp = (
        df_e.groupby(["pitcher", "pitch_type"], observed=True)
        .agg(n=("is_swing", "size"), swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"))
        .reset_index()
    )
    pt_grp = pt_grp.loc[pt_grp["pitch_type"].notna() & (pt_grp["pitch_type"] != "")]
    pt_grp["whiff_rate"] = pt_grp["whiffs"] / pt_grp["swings"].clip(lower=1)
    # Pitcher usage of this pitch as fraction of pitcher's total swings (proxy for arsenal weight)
    total_swings = pt_grp.groupby("pitcher")["swings"].transform("sum")
    pt_grp["arsenal_weight"] = pt_grp["swings"] / total_swings.clip(lower=1)
    weighted_stuff = (pt_grp["whiff_rate"] * pt_grp["arsenal_weight"]).groupby(pt_grp["pitcher"]).sum()
    weighted_stuff = weighted_stuff.rename("stuff_raw")

    # ----- COMMAND+ proxy: rank of (zone% − walk%)
    # Walk rate (per PA)
    pa = df.loc[df["is_term"]].copy()
    pa_pit = pa.groupby("pitcher").size().rename("n_pa")
    walks = pa.loc[pa["events"].isin(WALK_EVENTS)].groupby("pitcher").size().rename("n_walks")
    walks = walks.reindex(eligible, fill_value=0)
    pa_pit = pa_pit.reindex(eligible, fill_value=0).clip(lower=1)
    walk_rate = walks / pa_pit
    walk_rate = walk_rate.rename("walk_rate")

    # Zone rate (rulebook approximate)
    df_e["zone"] = rulebook_zone_flag(df_e["plate_x"], df_e["plate_z"])
    zone_rate = df_e.groupby("pitcher")["zone"].mean().rename("zone_rate")

    # Build raw command and stuff series
    raw = pd.concat([weighted_stuff, walk_rate, zone_rate], axis=1).loc[eligible].copy()
    raw["walk_rate"] = raw["walk_rate"].fillna(raw["walk_rate"].median())
    raw["zone_rate"] = raw["zone_rate"].fillna(raw["zone_rate"].median())
    raw["stuff_raw"] = raw["stuff_raw"].fillna(raw["stuff_raw"].median())

    # Rank to percentiles
    raw["stuff_pct"] = raw["stuff_raw"].rank(pct=True)
    raw["zone_rate_pct"] = raw["zone_rate"].rank(pct=True)
    raw["walk_rate_pct_inv"] = (-raw["walk_rate"]).rank(pct=True)  # lower walks = better command
    raw["command_pct"] = (raw["zone_rate_pct"] + raw["walk_rate_pct_inv"]) / 2

    # Pitcher names
    name_map = df.groupby("pitcher")["player_name"].agg(
        lambda s: s.dropna().iloc[0] if len(s.dropna()) else f"id_{int(s.name)}"
    )
    raw["name"] = raw.index.map(name_map)
    raw["pitcher_id"] = raw.index.astype(int)
    raw["data_source"] = "proxy"
    raw = raw.reset_index(drop=True)
    keep = ["pitcher_id", "name", "stuff_pct", "command_pct", "data_source"]
    out = raw[keep].copy()
    out["stuff_minus_command"] = out["stuff_pct"] - out["command_pct"]
    return out


def build(force: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if ARCHETYPE_PATH.exists() and not force:
        print(f"[archetype] cached at {ARCHETYPE_PATH}")
        return pd.read_parquet(ARCHETYPE_PATH)
    fg = _try_fangraphs_pitching_stats()
    if fg is not None and len(fg) > 50:
        print(f"[archetype] FanGraphs path succeeded: {len(fg)} pitchers")
        fg["stuff_minus_command"] = fg["stuff_pct"] - fg["command_pct"]
        df_out = fg
    else:
        print("[archetype] FanGraphs path unavailable; using Statcast proxy.")
        df_out = _build_proxy_from_statcast()
    df_out.to_parquet(ARCHETYPE_PATH, index=False)
    meta = {
        "n_rows": int(len(df_out)),
        "data_sources": df_out["data_source"].value_counts().to_dict(),
        "ip_threshold": 40,
        "fields": list(df_out.columns),
    }
    ARCHETYPE_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[archetype] saved {len(df_out)} rows to {ARCHETYPE_PATH}")
    return df_out


if __name__ == "__main__":
    build(force=False)

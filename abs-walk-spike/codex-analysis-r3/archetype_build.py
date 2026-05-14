from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_prep_r3 import (
    ARTIFACTS_DIR,
    DATA_DIR,
    SWING_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
    Round3Data,
    save_json,
    terminal_pa_rows,
)


OUT_EVENTS_1 = {
    "strikeout",
    "strikeout_double_play",
    "field_out",
    "force_out",
    "fielders_choice_out",
    "sac_fly",
    "sac_bunt",
    "sac_fly_double_play",
    "sac_bunt_double_play",
}
OUT_EVENTS_2 = {"double_play", "grounded_into_double_play"}
OUT_EVENTS_3 = {"triple_play"}


@dataclass
class ArchetypeResult:
    table: pd.DataFrame
    summary: dict


def _event_outs(events: pd.Series) -> pd.Series:
    out = pd.Series(0, index=events.index, dtype=int)
    out.loc[events.isin(OUT_EVENTS_1)] = 1
    out.loc[events.isin(OUT_EVENTS_2)] = 2
    out.loc[events.isin(OUT_EVENTS_3)] = 3
    return out


def _weighted_arsenal_whiff(df: pd.DataFrame) -> pd.DataFrame:
    pitch_type = (
        df.groupby(["pitcher", "pitch_type"], observed=True)
        .agg(
            pitches=("pitch_type", "size"),
            swings=("is_swing", "sum"),
            whiffs=("is_whiff", "sum"),
        )
        .reset_index()
    )
    pitch_type["whiff_rate"] = pitch_type["whiffs"] / pitch_type["swings"].replace(0, np.nan)
    league_by_type = pitch_type.groupby("pitch_type", observed=True)["whiff_rate"].mean().to_dict()
    pitch_type["whiff_rate"] = pitch_type.apply(
        lambda r: league_by_type.get(r.pitch_type, np.nan) if pd.isna(r.whiff_rate) else r.whiff_rate,
        axis=1,
    )
    pitch_type["weight"] = pitch_type["pitches"] / pitch_type.groupby("pitcher", observed=True)["pitches"].transform("sum")
    out = (
        pitch_type.assign(weighted_whiff=lambda x: x["weight"] * x["whiff_rate"])
        .groupby("pitcher", observed=True)
        .agg(arsenal_whiff_rate=("weighted_whiff", "sum"), arsenal_pitch_types=("pitch_type", "nunique"))
        .reset_index()
    )
    return out


def build_pitcher_archetypes(data: Round3Data) -> ArchetypeResult:
    print("R3-H3: building fallback 2025 stuff/command archetype lookup")
    df = data.full_2025.dropna(subset=["pitcher", "plate_x", "plate_z"]).copy()
    terminal = terminal_pa_rows(df)
    terminal["outs_recorded"] = _event_outs(terminal["events"])
    pa = (
        terminal.groupby(["pitcher", "player_name"], observed=True)
        .agg(
            batters_faced=("pa_id", "size"),
            walks=("walk_event", "sum"),
            outs_recorded=("outs_recorded", "sum"),
        )
        .reset_index()
    )
    pa["estimated_ip"] = pa["outs_recorded"] / 3.0
    pa["walk_rate_2025_full"] = pa["walks"] / pa["batters_faced"]
    zone = (
        df.groupby("pitcher", observed=True)
        .agg(
            pitches_2025_full=("pitch_type", "size"),
            zone_rate_2025_full=("in_fixed_zone", "mean"),
            top_share_2025_full=("is_top_edge", "mean"),
            mean_release_speed_2025=("release_speed", "mean"),
            mean_plate_z_2025=("plate_z", "mean"),
        )
        .reset_index()
    )
    whiff = _weighted_arsenal_whiff(df)
    table = pa.merge(zone, on="pitcher", how="inner").merge(whiff, on="pitcher", how="left")
    table = table[table["estimated_ip"] >= 40.0].copy()
    table["stuff_pct"] = table["arsenal_whiff_rate"].rank(pct=True) * 100
    bb_rank = table["walk_rate_2025_full"].rank(pct=True)
    zone_rank = table["zone_rate_2025_full"].rank(pct=True)
    table["command_pct"] = ((1.0 - bb_rank) * 0.55 + zone_rank * 0.45) * 100
    table["stuff_minus_command"] = table["stuff_pct"] - table["command_pct"]
    table["data_source"] = "proxy"
    table = table[
        [
            "pitcher",
            "player_name",
            "stuff_pct",
            "command_pct",
            "stuff_minus_command",
            "data_source",
            "estimated_ip",
            "batters_faced",
            "walk_rate_2025_full",
            "zone_rate_2025_full",
            "top_share_2025_full",
            "arsenal_whiff_rate",
            "arsenal_pitch_types",
            "pitches_2025_full",
            "mean_release_speed_2025",
            "mean_plate_z_2025",
        ]
    ].rename(columns={"pitcher": "pitcher_id", "player_name": "name"})
    table["pitcher_id"] = table["pitcher_id"].astype(int)
    table.to_parquet(DATA_DIR / "pitcher_archetype_2025.parquet", index=False)
    table.to_csv(ARTIFACTS_DIR / "pitcher_archetype_2025.csv", index=False)
    summary = {
        "data_source": "proxy",
        "pitchers_qualified_40ip": int(len(table)),
        "source_note": "FanGraphs was not required because the local Statcast fallback was available; proxy uses arsenal-weighted whiff percentile for stuff and low-walk plus zone-rate percentile for command.",
        "median_stuff_pct": float(table["stuff_pct"].median()),
        "median_command_pct": float(table["command_pct"].median()),
    }
    save_json(summary, ARTIFACTS_DIR / "archetype_summary.json")
    return ArchetypeResult(table=table, summary=summary)

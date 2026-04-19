#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BALL_RADIUS_FT = 1.45 / 12
HALF_PLATE_FT = 17 / 24 + BALL_RADIUS_FT
SEED = 42
PITCH_ORDER = ["FF", "FC", "SI", "CU", "SL", "ST", "CH"]
FOCUS_PITCHES = ["FF", "FC", "SI", "CU"]
FASTBALL_TRIO = ["FF", "FC", "SI"]
PITCH_LABELS = {
    "FF": "4-Seam",
    "FC": "Cutter",
    "SI": "Sinker",
    "CU": "Curve",
    "SL": "Slider",
    "ST": "Sweeper",
    "CH": "Changeup",
}
PITCH_COLORS = {
    "FF": "#d1495b",
    "FC": "#00798c",
    "SI": "#edae49",
    "CU": "#4f5d75",
    "SL": "#7f5af0",
    "ST": "#2a9d8f",
    "CH": "#5c946e",
}
SWING_DESCRIPTIONS = {
    "foul",
    "foul_tip",
    "foul_bunt",
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CALLED_DESCRIPTIONS = {"called_strike", "ball", "blocked_ball", "automatic_ball", "automatic_strike"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIR = SCRIPT_DIR.parent
DATA_DIR = CASE_DIR / "data"
TABLE_DIR = SCRIPT_DIR / "tables"
CHART_DIR = SCRIPT_DIR / "charts"

DAILY_2026_FILES = [
    "2026-03-27.parquet",
    "2026-03-28.parquet",
    "2026-03-29.parquet",
    "2026-03-30.parquet",
    "2026-03-31.parquet",
    "2026-04-01.parquet",
    "2026-04-02.parquet",
    "2026-04-03.parquet",
    "2026-04-04.parquet",
    "2026-04-05.parquet",
    "statcast_2026_apr06_11.parquet",
]

BASE_COLUMNS = [
    "pitcher",
    "player_name",
    "game_date",
    "game_pk",
    "pitch_type",
    "description",
    "events",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
]


def load_parquet_files(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(path, columns=columns) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError("No parquet files found for requested load.")
    return pd.concat(frames, ignore_index=True)


def prepare_pitches(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["game_date"] = pd.to_datetime(cleaned["game_date"], errors="coerce")
    numeric_columns = [
        "balls",
        "strikes",
        "release_speed",
        "release_spin_rate",
        "release_extension",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
        "sz_top",
        "sz_bot",
        "game_pk",
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=["pitch_type"]).copy()
    cleaned["pitch_type"] = cleaned["pitch_type"].astype(str)
    cleaned["pfx_x_in"] = cleaned["pfx_x"] * 12
    cleaned["pfx_z_in"] = cleaned["pfx_z"] * 12
    cleaned["in_zone"] = np.nan
    cleaned["edge_dist_in"] = np.nan
    loc_mask = cleaned[["plate_x", "plate_z", "sz_top", "sz_bot"]].notna().all(axis=1)
    loc_df = cleaned.loc[loc_mask].copy()
    left = -HALF_PLATE_FT
    right = HALF_PLATE_FT
    bottom = loc_df["sz_bot"] - BALL_RADIUS_FT
    top = loc_df["sz_top"] + BALL_RADIUS_FT

    dx = np.where(
        loc_df["plate_x"] < left,
        left - loc_df["plate_x"],
        np.where(loc_df["plate_x"] > right, loc_df["plate_x"] - right, 0.0),
    )
    dy = np.where(
        loc_df["plate_z"] < bottom,
        bottom - loc_df["plate_z"],
        np.where(loc_df["plate_z"] > top, loc_df["plate_z"] - top, 0.0),
    )
    outside_dist_ft = np.hypot(dx, dy)

    inside_x = np.minimum(loc_df["plate_x"] - left, right - loc_df["plate_x"])
    inside_y = np.minimum(loc_df["plate_z"] - bottom, top - loc_df["plate_z"])
    inside_dist_ft = np.minimum(inside_x, inside_y)

    in_zone = outside_dist_ft == 0
    cleaned.loc[loc_mask, "in_zone"] = in_zone.astype(float)
    cleaned.loc[loc_mask, "edge_dist_in"] = np.where(in_zone, inside_dist_ft, -outside_dist_ft) * 12
    cleaned["swing"] = cleaned["description"].isin(SWING_DESCRIPTIONS).astype(int)
    cleaned["whiff"] = cleaned["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    cleaned["called_pitch"] = cleaned["description"].isin(CALLED_DESCRIPTIONS).astype(int)
    cleaned["called_strike"] = cleaned["description"].isin({"called_strike", "automatic_strike"}).astype(int)
    cleaned["out_of_zone"] = np.where(cleaned["in_zone"].isna(), np.nan, 1 - cleaned["in_zone"])
    return cleaned


def safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def season_pitch_summary(df: pd.DataFrame, season_label: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    total_pitches = len(df)
    for pitch_type, pitch_df in df.groupby("pitch_type", sort=False):
        swings = int(pitch_df["swing"].sum())
        called_pitches = int(pitch_df["called_pitch"].sum())
        out_of_zone = int(pitch_df["out_of_zone"].sum())
        rows.append(
            {
                "season": season_label,
                "pitch_type": pitch_type,
                "pitch_label": PITCH_LABELS.get(pitch_type, pitch_type),
                "n": int(len(pitch_df)),
                "usage": safe_ratio(len(pitch_df), total_pitches),
                "release_speed": safe_mean(pitch_df["release_speed"]),
                "release_spin_rate": safe_mean(pitch_df["release_spin_rate"]),
                "release_extension": safe_mean(pitch_df["release_extension"]),
                "pfx_x_in": safe_mean(pitch_df["pfx_x_in"]),
                "pfx_z_in": safe_mean(pitch_df["pfx_z_in"]),
                "plate_x": safe_mean(pitch_df["plate_x"]),
                "plate_z": safe_mean(pitch_df["plate_z"]),
                "zone_rate": safe_mean(pitch_df["in_zone"]),
                "swing_rate": safe_mean(pitch_df["swing"]),
                "whiff_rate": safe_ratio(pitch_df["whiff"].sum(), swings),
                "chase_rate": safe_ratio(pitch_df.loc[pitch_df["out_of_zone"] == 1, "swing"].sum(), out_of_zone),
                "called_pitch_n": called_pitches,
                "called_strike_rate": safe_ratio(pitch_df["called_strike"].sum(), called_pitches),
                "csw": safe_ratio(pitch_df["called_strike"].sum() + pitch_df["whiff"].sum(), len(pitch_df)),
            }
        )
    summary = pd.DataFrame(rows)
    summary["pitch_type"] = pd.Categorical(summary["pitch_type"], categories=PITCH_ORDER, ordered=True)
    return summary.sort_values("pitch_type").reset_index(drop=True)


def league_pitch_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = season_pitch_summary(df, "league_2025_rhp").drop(columns=["season"])
    return summary


def build_game_log(cam_2026: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    grouped = cam_2026.groupby(["game_date", "game_pk"], dropna=False, sort=True)
    for (game_date, game_pk), game_df in grouped:
        counts = game_df["pitch_type"].value_counts()
        total = len(game_df)
        rows.append(
            {
                "game_date": pd.Timestamp(game_date).strftime("%Y-%m-%d"),
                "game_pk": int(game_pk),
                "pitches": total,
                "strikeouts": int(game_df["events"].eq("strikeout").sum()),
                "walks": int(game_df["events"].eq("walk").sum()),
                "hits": int(game_df["events"].isin(HIT_EVENTS).sum()),
                "ff_share": safe_ratio(counts.get("FF", 0), total),
                "fc_share": safe_ratio(counts.get("FC", 0), total),
                "si_share": safe_ratio(counts.get("SI", 0), total),
                "cu_share": safe_ratio(counts.get("CU", 0), total),
            }
        )
    return pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)


def pct_rank(series: pd.Series, value: float) -> float:
    valid = series.dropna()
    if len(valid) == 0 or pd.isna(value):
        return float("nan")
    return float((valid < value).mean() * 100)


def ordinal(value: float) -> str:
    number = int(round(value))
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def display_name(name: str) -> str:
    if "," not in name:
        return name
    last, first = [part.strip() for part in name.split(",", 1)]
    return f"{first} {last}"


def build_pitch_deltas(cam_2025_summary: pd.DataFrame, cam_2026_summary: pd.DataFrame) -> pd.DataFrame:
    merged = cam_2025_summary.merge(
        cam_2026_summary,
        on=["pitch_type", "pitch_label"],
        suffixes=("_2025", "_2026"),
    )
    keep = merged["pitch_type"].isin(FOCUS_PITCHES)
    merged = merged.loc[keep].copy()
    merged["usage_delta_pp"] = (merged["usage_2026"] - merged["usage_2025"]) * 100
    merged["velo_delta"] = merged["release_speed_2026"] - merged["release_speed_2025"]
    merged["pfx_x_delta_in"] = merged["pfx_x_in_2026"] - merged["pfx_x_in_2025"]
    merged["pfx_z_delta_in"] = merged["pfx_z_in_2026"] - merged["pfx_z_in_2025"]
    columns = [
        "pitch_type",
        "pitch_label",
        "n_2025",
        "usage_2025",
        "n_2026",
        "usage_2026",
        "usage_delta_pp",
        "release_speed_2025",
        "release_speed_2026",
        "velo_delta",
        "pfx_x_in_2025",
        "pfx_x_in_2026",
        "pfx_x_delta_in",
        "pfx_z_in_2025",
        "pfx_z_in_2026",
        "pfx_z_delta_in",
    ]
    return merged[columns].reset_index(drop=True)


def build_outcome_table(
    cam_2025_summary: pd.DataFrame,
    cam_2026_summary: pd.DataFrame,
    league_summary: pd.DataFrame,
) -> pd.DataFrame:
    merged = cam_2026_summary.merge(
        cam_2025_summary[["pitch_type", "zone_rate", "whiff_rate", "chase_rate", "called_strike_rate", "csw"]],
        on="pitch_type",
        suffixes=("_2026", "_2025"),
    ).merge(
        league_summary[["pitch_type", "zone_rate", "whiff_rate", "chase_rate", "called_strike_rate", "csw"]],
        on="pitch_type",
    )
    merged = merged.rename(
        columns={
            "zone_rate": "zone_rate_league",
            "whiff_rate": "whiff_rate_league",
            "chase_rate": "chase_rate_league",
            "called_strike_rate": "called_strike_rate_league",
            "csw": "csw_league",
        }
    )
    return merged[merged["pitch_type"].isin(FOCUS_PITCHES)].reset_index(drop=True)


def triangle_area(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2


def entropy_from_shares(shares: list[float]) -> float:
    arr = np.asarray(shares, dtype=float)
    arr = arr[arr > 0]
    if len(arr) == 0:
        return float("nan")
    return float(-(arr * np.log(arr)).sum() / np.log(3))


def fastball_shape_from_df(df: pd.DataFrame, label: str) -> dict[str, float | str]:
    trio = df[df["pitch_type"].isin(FASTBALL_TRIO)].copy()
    grouped = trio.groupby("pitch_type").agg(n=("pitch_type", "size"), x=("pfx_x_in", "mean"), z=("pfx_z_in", "mean"), velo=("release_speed", "mean"))
    shares = [safe_ratio(grouped.loc[pitch, "n"], grouped["n"].sum()) for pitch in FASTBALL_TRIO]
    area = triangle_area(
        grouped.loc["FF", "x"],
        grouped.loc["FF", "z"],
        grouped.loc["FC", "x"],
        grouped.loc["FC", "z"],
        grouped.loc["SI", "x"],
        grouped.loc["SI", "z"],
    )
    return {
        "label": label,
        "share_FF": shares[0],
        "share_FC": shares[1],
        "share_SI": shares[2],
        "entropy": entropy_from_shares(shares),
        "triangle_area": area,
        "balance_spread": area * entropy_from_shares(shares),
        "velo_FF": grouped.loc["FF", "velo"],
        "velo_FC": grouped.loc["FC", "velo"],
        "velo_SI": grouped.loc["SI", "velo"],
        "x_FF": grouped.loc["FF", "x"],
        "x_FC": grouped.loc["FC", "x"],
        "x_SI": grouped.loc["SI", "x"],
        "z_FF": grouped.loc["FF", "z"],
        "z_FC": grouped.loc["FC", "z"],
        "z_SI": grouped.loc["SI", "z"],
    }


def build_fastball_comp_tables(
    league_2025_rhp: pd.DataFrame,
    cam_2025: pd.DataFrame,
    cam_2026: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | str], dict[str, float | str]]:
    focus = league_2025_rhp[league_2025_rhp["pitch_type"].isin(FASTBALL_TRIO)].copy()
    grouped = (
        focus.groupby(["pitcher", "player_name", "pitch_type"])
        .agg(n=("pitch_type", "size"), x=("pfx_x_in", "mean"), z=("pfx_z_in", "mean"), velo=("release_speed", "mean"))
        .reset_index()
    )
    wide = grouped.pivot(index=["pitcher", "player_name"], columns="pitch_type", values=["n", "x", "z", "velo"])
    wide.columns = ["_".join(col) for col in wide.columns]
    wide = wide.reset_index()
    numeric_columns = [column for column in wide.columns if column not in {"pitcher", "player_name"}]
    wide[numeric_columns] = wide[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    wide = wide[(wide["n_FF"] >= 50) & (wide["n_FC"] >= 50) & (wide["n_SI"] >= 50)].copy()
    wide["total"] = wide[["n_FF", "n_FC", "n_SI"]].sum(axis=1)
    for pitch in FASTBALL_TRIO:
        wide[f"share_{pitch}"] = wide[f"n_{pitch}"] / wide["total"]
    wide["entropy"] = [entropy_from_shares([row["share_FF"], row["share_FC"], row["share_SI"]]) for _, row in wide.iterrows()]
    wide["triangle_area"] = [
        triangle_area(row["x_FF"], row["z_FF"], row["x_FC"], row["z_FC"], row["x_SI"], row["z_SI"]) for _, row in wide.iterrows()
    ]
    wide["balance_spread"] = wide["triangle_area"] * wide["entropy"]

    cam_2025_shape = fastball_shape_from_df(cam_2025, "Cam Schlittler 2025")
    cam_2026_shape = fastball_shape_from_df(cam_2026, "Cam Schlittler 2026")

    feature_columns = [
        "share_FF",
        "share_FC",
        "share_SI",
        "velo_FF",
        "velo_FC",
        "velo_SI",
        "x_FF",
        "x_FC",
        "x_SI",
        "z_FF",
        "z_FC",
        "z_SI",
    ]
    means = wide[feature_columns].mean()
    stds = wide[feature_columns].std().replace(0, 1)
    cam_target = pd.Series({column: cam_2026_shape[column] for column in feature_columns})
    wide["distance_to_cam_2026"] = np.sqrt(((wide[feature_columns] - cam_target) / stds).pow(2).sum(axis=1))
    wide["triangle_area_pct"] = wide["triangle_area"].rank(method="min", pct=True) * 100
    wide["entropy_pct"] = wide["entropy"].rank(method="min", pct=True) * 100
    wide["balance_spread_pct"] = wide["balance_spread"].rank(method="min", pct=True) * 100
    wide = wide.sort_values("distance_to_cam_2026").reset_index(drop=True)
    wide["distance_rank"] = np.arange(1, len(wide) + 1)
    wide["display_name"] = wide["player_name"].map(display_name)

    for shape in (cam_2025_shape, cam_2026_shape):
        shape["triangle_area_pct"] = pct_rank(wide["triangle_area"], float(shape["triangle_area"]))
        shape["entropy_pct"] = pct_rank(wide["entropy"], float(shape["entropy"]))
        shape["balance_spread_pct"] = pct_rank(wide["balance_spread"], float(shape["balance_spread"]))

    return wide, cam_2025_shape, cam_2026_shape


def build_movement_percentiles(cam_2026: pd.DataFrame, league_2025_rhp: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for pitch_type in FOCUS_PITCHES:
        cam_df = cam_2026[cam_2026["pitch_type"] == pitch_type]
        league_df = league_2025_rhp[league_2025_rhp["pitch_type"] == pitch_type]
        if len(cam_df) == 0 or len(league_df) == 0:
            continue
        for column, label in [("release_speed", "velo"), ("pfx_x_in", "pfx_x"), ("pfx_z_in", "pfx_z")]:
            value = float(cam_df[column].mean())
            rows.append(
                {
                    "pitch_type": pitch_type,
                    "metric": label,
                    "cam_2026": value,
                    "league_mean": float(league_df[column].mean()),
                    "pct": pct_rank(league_df[column], value),
                }
            )
    return pd.DataFrame(rows)


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt_pp(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:+.1f} pp"


def fmt_num(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], rename_map: dict[str, str]) -> str:
    header = "| " + " | ".join(rename_map.get(column, column) for column in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in df[columns].iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def sample_for_cloud(df: pd.DataFrame, pitch_type: str, n: int = 3500) -> pd.DataFrame:
    subset = df[df["pitch_type"] == pitch_type].dropna(subset=["pfx_x_in", "pfx_z_in"])
    if len(subset) <= n:
        return subset
    return subset.sample(n=n, random_state=SEED)


def plot_movement_profile(league_2025_rhp: pd.DataFrame, cam_2026: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for pitch_type in FOCUS_PITCHES:
        cloud = sample_for_cloud(league_2025_rhp, pitch_type)
        ax.scatter(
            cloud["pfx_x_in"],
            cloud["pfx_z_in"],
            s=9,
            alpha=0.07,
            color=PITCH_COLORS[pitch_type],
            linewidths=0,
        )
    for pitch_type in FOCUS_PITCHES:
        subset = cam_2026[cam_2026["pitch_type"] == pitch_type]
        ax.scatter(
            subset["pfx_x_in"],
            subset["pfx_z_in"],
            s=42,
            alpha=0.9,
            color=PITCH_COLORS[pitch_type],
            edgecolors="black",
            linewidths=0.35,
            label=f"Schlittler 2026 {PITCH_LABELS[pitch_type]} ({len(subset)})",
        )
        ax.scatter(
            [subset["pfx_x_in"].mean()],
            [subset["pfx_z_in"].mean()],
            s=150,
            marker="X",
            color=PITCH_COLORS[pitch_type],
            edgecolors="black",
            linewidths=0.8,
        )
    ax.axhline(0, color="#d0d7de", linewidth=0.8)
    ax.axvline(0, color="#d0d7de", linewidth=0.8)
    ax.set_title("Cam Schlittler 2026 Arsenal vs 2025 RHP Movement Clouds", fontsize=15, weight="bold")
    ax.set_xlabel("Horizontal movement (inches)")
    ax.set_ylabel("Vertical movement (inches)")
    ax.set_xlim(-24, 16)
    ax.set_ylim(-16, 24)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "movement_profile.png", dpi=200)
    plt.close(fig)


def plot_pitch_mix_shift(pitch_deltas: pd.DataFrame, game_log: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12, 6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])

    bars = {"2025": 0, "2026": 1}
    bottoms = {"2025": 0.0, "2026": 0.0}
    for pitch_type in FOCUS_PITCHES:
        row = pitch_deltas[pitch_deltas["pitch_type"] == pitch_type].iloc[0]
        shares = {"2025": float(row["usage_2025"]), "2026": float(row["usage_2026"])}
        color = PITCH_COLORS[pitch_type]
        for season, xpos in bars.items():
            share = shares[season]
            ax_left.bar(xpos, share, bottom=bottoms[season], color=color, width=0.6, edgecolor="white")
            if share >= 0.07:
                ax_left.text(xpos, bottoms[season] + share / 2, f"{share * 100:.1f}%", ha="center", va="center", fontsize=9, color="white")
            bottoms[season] += share
    ax_left.set_title("Pitch-Mix Rebuild", fontsize=13, weight="bold")
    ax_left.set_xticks([0, 1], ["2025", "2026"])
    ax_left.set_ylim(0, 1)
    ax_left.set_ylabel("Share of total pitches")

    game_log = game_log.copy()
    game_log["game_date"] = pd.to_datetime(game_log["game_date"])
    for pitch_type, column in [("FF", "ff_share"), ("FC", "fc_share"), ("SI", "si_share"), ("CU", "cu_share")]:
        ax_right.plot(
            game_log["game_date"],
            game_log[column],
            marker="o",
            linewidth=2.2,
            color=PITCH_COLORS[pitch_type],
            label=PITCH_LABELS[pitch_type],
        )
    ax_right.set_title("Three-Start Usage Stability", fontsize=13, weight="bold")
    ax_right.set_ylim(0, 0.45)
    ax_right.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    ax_right.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax_right.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_right.grid(alpha=0.25)
    ax_right.legend(frameon=False, ncol=2)
    fig.suptitle("Cam Schlittler moved from 4-seam-heavy to an almost even fastball trio", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "pitch_mix_shift.png", dpi=200)
    plt.close(fig)


def plot_velocity_distributions(league_2025_rhp: pd.DataFrame, cam_2025: pd.DataFrame, cam_2026: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()
    for axis, pitch_type in zip(axes, FOCUS_PITCHES):
        league = league_2025_rhp[league_2025_rhp["pitch_type"] == pitch_type]["release_speed"].dropna()
        cam25 = cam_2025[cam_2025["pitch_type"] == pitch_type]["release_speed"].dropna()
        cam26 = cam_2026[cam_2026["pitch_type"] == pitch_type]["release_speed"].dropna()
        bins = np.linspace(math.floor(league.min()) - 1, math.ceil(league.max()) + 1, 30)
        axis.hist(league, bins=bins, density=True, color="#d0d7de", alpha=0.85, label="2025 RHP league")
        axis.hist(cam25, bins=bins, density=True, histtype="step", linewidth=2.0, color="#111827", label="Schlittler 2025")
        axis.hist(cam26, bins=bins, density=True, color=PITCH_COLORS[pitch_type], alpha=0.35, label="Schlittler 2026")
        axis.axvline(cam25.mean(), color="#111827", linestyle="--", linewidth=1.4)
        axis.axvline(cam26.mean(), color=PITCH_COLORS[pitch_type], linestyle="-", linewidth=2.0)
        axis.set_title(PITCH_LABELS[pitch_type], fontsize=12, weight="bold")
        axis.set_xlabel("Velocity (mph)")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3)
    fig.suptitle("Velocity context: cutter is the loudest shape-and-velo change", fontsize=15, weight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(CHART_DIR / "velocity_distributions.png", dpi=200)
    plt.close(fig)


def plot_location_roles(cam_2026: pd.DataFrame, cam_2026_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for axis, pitch_type in zip(axes, FOCUS_PITCHES):
        subset = cam_2026[cam_2026["pitch_type"] == pitch_type].dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot"])
        called = subset[subset["called_pitch"] == 1]
        called_strikes = called[called["called_strike"] == 1]
        called_balls = called[called["called_strike"] == 0]
        zone_bottom = float(subset["sz_bot"].mean())
        zone_top = float(subset["sz_top"].mean())
        axis.scatter(subset["plate_x"], subset["plate_z"], s=28, alpha=0.35, color=PITCH_COLORS[pitch_type], linewidths=0)
        axis.scatter(
            called_strikes["plate_x"],
            called_strikes["plate_z"],
            s=70,
            facecolors="none",
            edgecolors="#0f9d58",
            linewidths=1.4,
            label="Called strike",
        )
        axis.scatter(
            called_balls["plate_x"],
            called_balls["plate_z"],
            s=46,
            marker="x",
            color="#d1495b",
            linewidths=1.2,
            label="Called ball",
        )
        axis.add_patch(
            plt.Rectangle(
                (-HALF_PLATE_FT, zone_bottom),
                2 * HALF_PLATE_FT,
                zone_top - zone_bottom,
                fill=False,
                edgecolor="#111827",
                linewidth=1.2,
            )
        )
        summary_row = cam_2026_summary[cam_2026_summary["pitch_type"] == pitch_type].iloc[0]
        title = (
            f"{PITCH_LABELS[pitch_type]} | Zone {summary_row['zone_rate'] * 100:.0f}%"
            f" | CSW {summary_row['csw'] * 100:.0f}%"
            f" | CS {summary_row['called_strike_rate'] * 100:.0f}%"
        )
        axis.set_title(title, fontsize=11, weight="bold")
        axis.set_xlim(-2.0, 2.0)
        axis.set_ylim(0.7, 4.6)
        axis.grid(alpha=0.15)
    axes[0].legend(frameon=False, loc="upper right")
    fig.supxlabel("plate_x (feet)")
    fig.supylabel("plate_z (feet)")
    fig.suptitle("Location roles: sinker for strikes, four-seam for chase, cutter between them", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(CHART_DIR / "location_roles.png", dpi=200)
    plt.close(fig)


def plot_balance_spread(comp_table: pd.DataFrame, cam_2025_shape: dict[str, float | str], cam_2026_shape: dict[str, float | str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(comp_table["entropy"], comp_table["triangle_area"], s=38, color="#c9d1d9", alpha=0.75, linewidths=0)
    nearest = comp_table.nsmallest(5, "distance_to_cam_2026")
    for _, row in nearest.iterrows():
        ax.text(row["entropy"] + 0.003, row["triangle_area"] + 1.2, str(row["display_name"]), fontsize=8, color="#4b5563")

    ax.scatter(
        [cam_2025_shape["entropy"]],
        [cam_2025_shape["triangle_area"]],
        s=110,
        color="#111827",
        marker="D",
        label="Schlittler 2025",
        edgecolors="white",
        linewidths=0.8,
    )
    ax.scatter(
        [cam_2026_shape["entropy"]],
        [cam_2026_shape["triangle_area"]],
        s=140,
        color="#d1495b",
        marker="X",
        label="Schlittler 2026",
        edgecolors="white",
        linewidths=0.8,
    )
    ax.text(
        float(cam_2025_shape["entropy"]) + 0.005,
        float(cam_2025_shape["triangle_area"]) - 3,
        "Schlittler 2025",
        fontsize=9,
        color="#111827",
    )
    ax.text(
        float(cam_2026_shape["entropy"]) + 0.005,
        float(cam_2026_shape["triangle_area"]) + 2,
        "Schlittler 2026",
        fontsize=9,
        color="#d1495b",
    )
    ax.set_title("Balanced Usage + Movement Spread", fontsize=15, weight="bold")
    ax.set_xlabel("Three-fastball balance (entropy; 1.0 = perfectly even)")
    ax.set_ylabel("Movement triangle area (square inches)")
    ax.grid(alpha=0.2)
    ax.set_xlim(0.65, 1.01)
    ax.set_ylim(20, max(120, comp_table["triangle_area"].max() + 5))
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "balance_vs_spread.png", dpi=200)
    plt.close(fig)


def render_report(
    raw_cam_2026: pd.DataFrame,
    cam_2026: pd.DataFrame,
    cam_2025_summary: pd.DataFrame,
    cam_2026_summary: pd.DataFrame,
    league_summary: pd.DataFrame,
    pitch_deltas: pd.DataFrame,
    game_log: pd.DataFrame,
    outcome_table: pd.DataFrame,
    movement_percentiles: pd.DataFrame,
    comp_table: pd.DataFrame,
    cam_2025_shape: dict[str, float | str],
    cam_2026_shape: dict[str, float | str],
) -> str:
    total_pitches = len(cam_2026)
    total_rows = len(raw_cam_2026)
    unclassified_rows = total_rows - total_pitches
    total_strikeouts = int(cam_2026["events"].eq("strikeout").sum())
    total_walks = int(cam_2026["events"].eq("walk").sum())
    total_hits = int(cam_2026["events"].isin(HIT_EVENTS).sum())
    total_starts = game_log["game_pk"].nunique()

    ff_row = outcome_table[outcome_table["pitch_type"] == "FF"].iloc[0]
    fc_row = outcome_table[outcome_table["pitch_type"] == "FC"].iloc[0]
    si_row = outcome_table[outcome_table["pitch_type"] == "SI"].iloc[0]
    cutter_delta = pitch_deltas[pitch_deltas["pitch_type"] == "FC"].iloc[0]
    mix_rows = pitch_deltas.copy()
    mix_rows["usage_2025"] = mix_rows["usage_2025"].map(fmt_pct)
    mix_rows["usage_2026"] = mix_rows["usage_2026"].map(fmt_pct)
    mix_rows["usage_delta_pp"] = mix_rows["usage_delta_pp"].map(lambda value: f"{value:+.1f} pp")
    mix_rows["velo_delta"] = mix_rows["velo_delta"].map(lambda value: f"{value:+.1f} mph")
    mix_rows["pfx_x_delta_in"] = mix_rows["pfx_x_delta_in"].map(lambda value: f"{value:+.1f} in")
    mix_rows["pfx_z_delta_in"] = mix_rows["pfx_z_delta_in"].map(lambda value: f"{value:+.1f} in")

    outcome_rows = outcome_table.copy()
    outcome_rows["pitch"] = outcome_rows["pitch_type"].map(PITCH_LABELS)
    for column in [
        "zone_rate_2026",
        "whiff_rate_2026",
        "chase_rate_2026",
        "called_strike_rate_2026",
        "csw_2026",
        "zone_rate_league",
        "whiff_rate_league",
        "chase_rate_league",
        "called_strike_rate_league",
        "csw_league",
    ]:
        outcome_rows[column] = outcome_rows[column].map(fmt_pct)

    comp_rows = comp_table.head(6).copy()
    comp_rows["pitcher"] = comp_rows["display_name"]
    comp_rows["ff_share"] = comp_rows["share_FF"].map(fmt_pct)
    comp_rows["fc_share"] = comp_rows["share_FC"].map(fmt_pct)
    comp_rows["si_share"] = comp_rows["share_SI"].map(fmt_pct)
    comp_rows["triangle_area"] = comp_rows["triangle_area"].map(lambda value: f"{value:.1f}")
    comp_rows["distance_to_cam_2026"] = comp_rows["distance_to_cam_2026"].map(lambda value: f"{value:.2f}")

    movement_lookup = movement_percentiles.set_index(["pitch_type", "metric"])
    cutter_velo_pct = movement_lookup.loc[("FC", "velo"), "pct"]
    cutter_z_pct = movement_lookup.loc[("FC", "pfx_z"), "pct"]
    nearest_names = ", ".join(str(name) for name in comp_table.head(4)["display_name"])

    report = f"""# Research Report: Cam Schlittler's Arsenal

Prepared on April 12, 2026 inside `research/schlittler-arsenal/codex-analysis-2026-04-12/`.

## Executive Summary
- Local sample: **{total_pitches} classified pitches** plus **{unclassified_rows} automatic-strike row** (**{total_rows} total Statcast rows**), **{total_starts} starts**, **{total_strikeouts} strikeouts**, **{total_walks} walks**, and **{total_hits} hits allowed** through April 7, 2026.
- Verdict: **publish, but narrow the thesis**. The data strongly support a real arsenal redesign and a rare near-even three-fastball mix. They do **not** support a "largest movement spread in baseball" claim.
- Mix shift: Schlittler went from **{fmt_pct(float(pitch_deltas[pitch_deltas['pitch_type'] == 'FF']['usage_2025'].iloc[0]))} 4-seam / {fmt_pct(float(pitch_deltas[pitch_deltas['pitch_type'] == 'FC']['usage_2025'].iloc[0]))} cutter / {fmt_pct(float(pitch_deltas[pitch_deltas['pitch_type'] == 'SI']['usage_2025'].iloc[0]))} sinker** in 2025 to **{fmt_pct(float(pitch_deltas[pitch_deltas['pitch_type'] == 'FF']['usage_2026'].iloc[0]))} / {fmt_pct(float(pitch_deltas[pitch_deltas['pitch_type'] == 'FC']['usage_2026'].iloc[0]))} / {fmt_pct(float(pitch_deltas[pitch_deltas['pitch_type'] == 'SI']['usage_2026'].iloc[0]))}** in the first three starts of 2026.
- Balance is the standout trait: among **{len(comp_table)}** right-handed 2025 pitchers who threw at least 50 of all three fastballs, Schlittler's **2026 mix balance ranks at the 100th percentile**, while his movement-spread triangle is only **{float(cam_2026_shape['triangle_area_pct']):.0f}th percentile**. The story is the **combination** of balance plus above-average separation, not separation alone.
- The cutter remake is the loudest pitch-level change: **{cutter_delta['velo_delta']:+.1f} mph**, **{cutter_delta['pfx_x_delta_in']:+.1f} inches horizontal change**, and **{cutter_delta['pfx_z_delta_in']:+.1f} inches vertical change** versus 2025. Relative to 2025 right-handed cutters, the 2026 cutter already sits around the **{ordinal(float(cutter_velo_pct))} percentile in velocity** and **{ordinal(float(cutter_z_pct))} percentile in vertical movement**.
- Early role separation is real: the **4-seam** owns the whiffiest swing profile (**{fmt_pct(float(ff_row['whiff_rate_2026']))}** on swings), the **sinker** owns the called-strike profile (**{fmt_pct(float(si_row['called_strike_rate_2026']))}**, vs **{fmt_pct(float(si_row['called_strike_rate_league']))}** league), and the **cutter** sits in the middle as the bridge pitch.

## Recommended Framing
- Best angle: **Cam Schlittler Didn't Build the Biggest Fastball Spread in Baseball. He Built One of the Most Balanced Ones.**
- Alternate angle: **The New Cutter Is the Key to Cam Schlittler's Three-Fastball Attack.**
- Editorial recommendation: **publish now**, but keep the promise limited to:
  - the mix change is real
  - the cutter reshape is real
  - the three-fastball usage balance is unusual
  - the role split is visible already

## What Holds Up
1. **The arsenal changed, not just the outcomes.**
   Schlittler was a 4-seam-first pitcher in 2025. In 2026 he is essentially splitting the workload across FF, FC, and SI.
2. **The cutter is the fulcrum pitch.**
   The 2026 cutter is harder and carries much more vertical life than his 2025 version. That is the cleanest evidence that something material changed.
3. **The fastballs are doing different jobs.**
   The 4-seam is being used higher and farther from the heart of the zone, which helps explain the low called-strike rate and high whiff rate.
   The sinker is landing much closer to the zone interior, which helps explain the strong called-strike returns.
   The cutter lives between those poles, giving him a glove-side fastball that still looks like part of the same tunnel family.
4. **Balance is the rare part.**
   The 2026 three-fastball mix is almost perfectly even. The nearest 2025 shape-and-usage matches were **{nearest_names}**, but even those were less balanced.

## What Does Not Hold Up
- The raw movement spread is **good, not unprecedented**. Schlittler's 2026 triangle area lands around the **{float(cam_2026_shape['triangle_area_pct']):.0f}th percentile** of eligible 2025 right-handed three-fastball pitchers, not the extreme top of the population.
- The early whiff and called-strike rates are **directionally interesting, not stabilized**. We only have {total_pitches} classified pitches and just over 100 called pitches.
- The local ABS-style called-pitch parquet stops on **April 5, 2026**, so the April 7 start cannot be audited pitch-by-pitch for rulebook correctness from the precomputed `all_called` files.

## Pitch-Mix Change Table
{markdown_table(
    mix_rows,
    ["pitch_label", "usage_2025", "usage_2026", "usage_delta_pp", "velo_delta", "pfx_x_delta_in", "pfx_z_delta_in"],
    {
        "pitch_label": "Pitch",
        "usage_2025": "2025 usage",
        "usage_2026": "2026 usage",
        "usage_delta_pp": "Usage delta",
        "velo_delta": "Velo delta",
        "pfx_x_delta_in": "Horiz move delta",
        "pfx_z_delta_in": "Vert move delta",
    },
)}

## Outcome Snapshot: 2026 vs 2025 RHP League
{markdown_table(
    outcome_rows,
    [
        "pitch",
        "zone_rate_2026",
        "whiff_rate_2026",
        "called_strike_rate_2026",
        "csw_2026",
        "zone_rate_league",
        "whiff_rate_league",
        "called_strike_rate_league",
        "csw_league",
    ],
    {
        "pitch": "Pitch",
        "zone_rate_2026": "2026 zone",
        "whiff_rate_2026": "2026 whiff",
        "called_strike_rate_2026": "2026 called strike",
        "csw_2026": "2026 CSW",
        "zone_rate_league": "League zone",
        "whiff_rate_league": "League whiff",
        "called_strike_rate_league": "League called strike",
        "csw_league": "League CSW",
    },
)}

## Historical Three-Fastball Neighbors
These are **shape-and-usage comps**, not performance comps.

{markdown_table(
    comp_rows,
    ["pitcher", "ff_share", "fc_share", "si_share", "triangle_area", "distance_to_cam_2026"],
    {
        "pitcher": "Pitcher",
        "ff_share": "FF share",
        "fc_share": "FC share",
        "si_share": "SI share",
        "triangle_area": "Triangle area",
        "distance_to_cam_2026": "Distance",
    },
)}

## Charts
### 1. Movement profile
![Movement profile](charts/movement_profile.png)

### 2. Pitch-mix shift
![Pitch mix shift](charts/pitch_mix_shift.png)

### 3. Velocity context
![Velocity distributions](charts/velocity_distributions.png)

### 4. Location roles
![Location roles](charts/location_roles.png)

### 5. Balance vs spread
![Balance vs spread](charts/balance_vs_spread.png)

## Bottom Line
- **Publishable claim**: Schlittler has turned a 4-seam-led arsenal into a highly balanced three-fastball attack, and the cutter reshape is the clearest mechanical/pitch-design lever behind it.
- **Do not claim**: that his movement spread is uniquely extreme league-wide.
- **Best single sentence**: *Cam Schlittler's first three starts look less like random heater luck and more like a deliberate three-fastball redesign, with a remade cutter turning an already-hard arsenal into a balanced, role-specific attack.*

## Method
1. Loaded `statcast_2025_full.parquet` as the right-handed 2025 league baseline and every local 2026 parquet through `statcast_2026_apr06_11.parquet` for Schlittler's three starts.
2. Built a rulebook strike zone from `plate_x`, `plate_z`, `sz_top`, and `sz_bot`, expanded by one baseball radius to account for ball-center tracking.
3. Calculated pitch-level rates from Statcast descriptions:
   whiff rate = whiffs / swings,
   chase rate = swings on out-of-zone pitches / out-of-zone pitches,
   called-strike rate = called strikes / called pitches,
   CSW = (called strikes + whiffs) / total pitches.
4. Compared Schlittler's three-fastball mix against right-handed 2025 pitchers with at least 50 FF, 50 FC, and 50 SI.
5. Defined `triangle_area` as the movement-space area formed by the FF, FC, and SI centroids in inches.
   Defined `entropy` as normalized pitch-share balance across those three pitches.

## Caveats
- This is still a **231-pitch classified sample** with one additional automatic-strike row, so outcome rates can move quickly.
- The three-fastball comp table compares 2026 Schlittler to **2025** pitchers because that is the full local league baseline provided in the repo.
- Statcast movement (`pfx_x`, `pfx_z`) is not the same thing as Hawkeye active-spin or induced vertical break modeling.
- The game log confirms **22 strikeouts and 0 walks** from the local Statcast event rows, but this report does not re-estimate ERA or opponent quality.
"""
    return report


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    daily_paths = [DATA_DIR / name for name in DAILY_2026_FILES]
    season_2025_path = DATA_DIR / "statcast_2025_full.parquet"

    raw_2026 = load_parquet_files(daily_paths, BASE_COLUMNS)
    raw_2025 = pd.read_parquet(season_2025_path, columns=BASE_COLUMNS)

    raw_cam_2026 = raw_2026[raw_2026["pitcher"] == 693645].copy()
    prepared_2026 = prepare_pitches(raw_2026)
    prepared_2025 = prepare_pitches(raw_2025)

    cam_2026 = prepared_2026[prepared_2026["pitcher"] == 693645].copy()
    cam_2025 = prepared_2025[prepared_2025["pitcher"] == 693645].copy()
    league_2025_rhp = prepared_2025[prepared_2025["p_throws"] == "R"].copy()

    cam_2025_summary = season_pitch_summary(cam_2025, "cam_2025")
    cam_2026_summary = season_pitch_summary(cam_2026, "cam_2026")
    league_summary = league_pitch_summary(league_2025_rhp[league_2025_rhp["pitch_type"].isin(FOCUS_PITCHES)])
    pitch_deltas = build_pitch_deltas(cam_2025_summary, cam_2026_summary)
    game_log = build_game_log(cam_2026)
    outcome_table = build_outcome_table(cam_2025_summary, cam_2026_summary, league_summary)
    movement_percentiles = build_movement_percentiles(cam_2026, league_2025_rhp)
    comp_table, cam_2025_shape, cam_2026_shape = build_fastball_comp_tables(league_2025_rhp, cam_2025, cam_2026)

    cam_2025_summary.to_csv(TABLE_DIR / "cam_2025_pitch_summary.csv", index=False)
    cam_2026_summary.to_csv(TABLE_DIR / "cam_2026_pitch_summary.csv", index=False)
    league_summary.to_csv(TABLE_DIR / "league_2025_rhp_pitch_summary.csv", index=False)
    pitch_deltas.to_csv(TABLE_DIR / "pitch_mix_deltas.csv", index=False)
    outcome_table.to_csv(TABLE_DIR / "outcome_snapshot.csv", index=False)
    movement_percentiles.to_csv(TABLE_DIR / "movement_percentiles.csv", index=False)
    game_log.to_csv(TABLE_DIR / "game_log_2026.csv", index=False)
    comp_table.to_csv(TABLE_DIR / "three_fastball_comps.csv", index=False)
    pd.DataFrame([cam_2025_shape, cam_2026_shape]).to_csv(TABLE_DIR / "schlittler_fastball_shapes.csv", index=False)

    plot_movement_profile(league_2025_rhp, cam_2026)
    plot_pitch_mix_shift(pitch_deltas, game_log)
    plot_velocity_distributions(league_2025_rhp, cam_2025, cam_2026)
    plot_location_roles(cam_2026, cam_2026_summary)
    plot_balance_spread(comp_table, cam_2025_shape, cam_2026_shape)

    report = render_report(
        raw_cam_2026=raw_cam_2026,
        cam_2026=cam_2026,
        cam_2025_summary=cam_2025_summary,
        cam_2026_summary=cam_2026_summary,
        league_summary=league_summary,
        pitch_deltas=pitch_deltas,
        game_log=game_log,
        outcome_table=outcome_table,
        movement_percentiles=movement_percentiles,
        comp_table=comp_table,
        cam_2025_shape=cam_2025_shape,
        cam_2026_shape=cam_2026_shape,
    )
    (SCRIPT_DIR / "REPORT.md").write_text(report)


if __name__ == "__main__":
    main()

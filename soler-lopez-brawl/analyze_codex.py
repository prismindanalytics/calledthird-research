#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BALL_RADIUS_FT = 1.45 / 12
HALF_PLATE_FT = 17 / 24 + BALL_RADIUS_FT
SHADOW_BAND_IN = 2.0
EDGE_BIN_BREAKS = np.arange(-8.0, 9.0, 1.0)

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIR = SCRIPT_DIR.parent
DATA_DIR = CASE_DIR / "data"
TABLE_DIR = SCRIPT_DIR / "tables"
CHART_DIR = SCRIPT_DIR / "charts"


def safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.mean())


def fmt_rate(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt_pp(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:+.1f} pp"


def fmt_number(value: float | int) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{int(value):,}"


def ordinal(value: float | int) -> str:
    number = int(value)
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def stem_for_label(label: str) -> str:
    return label.replace(" ", "_")


def markdown_table(df: pd.DataFrame, columns: list[str], renames: dict[str, str]) -> str:
    header = "| " + " | ".join(renames.get(column, column) for column in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def load_pitch_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_columns = ["plate_x", "plate_z", "sz_top", "sz_bot", "game_pk"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot"]).copy()
    df["game_pk"] = df["game_pk"].astype("Int64")
    df["called_strike"] = (df["description"] == "called_strike").astype(int)

    left = -HALF_PLATE_FT
    right = HALF_PLATE_FT
    bottom = df["sz_bot"] - BALL_RADIUS_FT
    top = df["sz_top"] + BALL_RADIUS_FT

    dx = np.where(
        df["plate_x"] < left,
        left - df["plate_x"],
        np.where(df["plate_x"] > right, df["plate_x"] - right, 0.0),
    )
    dy = np.where(
        df["plate_z"] < bottom,
        bottom - df["plate_z"],
        np.where(df["plate_z"] > top, df["plate_z"] - top, 0.0),
    )
    outside_dist_ft = np.hypot(dx, dy)

    inside_x = np.minimum(df["plate_x"] - left, right - df["plate_x"])
    inside_y = np.minimum(df["plate_z"] - bottom, top - df["plate_z"])
    inside_dist_ft = np.minimum(inside_x, inside_y)

    in_zone = outside_dist_ft == 0
    signed_edge_dist_in = np.where(in_zone, inside_dist_ft, -outside_dist_ft) * 12

    df["in_zone"] = in_zone.astype(int)
    df["edge_dist_in"] = signed_edge_dist_in
    df["shadow"] = (np.abs(df["edge_dist_in"]) <= SHADOW_BAND_IN).astype(int)
    df["oob_strike"] = ((df["in_zone"] == 0) & (df["called_strike"] == 1)).astype(int)
    df["zone_ball"] = ((df["in_zone"] == 1) & (df["called_strike"] == 0)).astype(int)
    df["miss"] = ((df["oob_strike"] == 1) | (df["zone_ball"] == 1)).astype(int)
    return df


def sample_metrics(df: pd.DataFrame) -> dict[str, float]:
    shadow_df = df[df["shadow"] == 1]
    out_df = df[df["in_zone"] == 0]
    in_df = df[df["in_zone"] == 1]

    return {
        "pitches": float(len(df)),
        "called_strike_rate": safe_rate(df["called_strike"]),
        "shadow_pitches": float(len(shadow_df)),
        "shadow_strike_rate": safe_rate(shadow_df["called_strike"]),
        "oob_pitches": float(len(out_df)),
        "oob_strike_rate": safe_rate(out_df["called_strike"]),
        "zone_pitches": float(len(in_df)),
        "zone_ball_rate": float("nan") if len(in_df) == 0 else float(1 - in_df["called_strike"].mean()),
        "miss_rate": safe_rate(df["miss"]),
        "in_zone_share": safe_rate(df["in_zone"]),
    }


def per_game_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    grouped = df.groupby(["game_pk", "game_date"], dropna=False, sort=True)
    for (game_pk, game_date), game_df in grouped:
        row: dict[str, float | int | str] = {
            "game_pk": int(game_pk) if not pd.isna(game_pk) else -1,
            "game_date": game_date,
        }
        row.update(sample_metrics(game_df))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_percentile(control: pd.Series, value: float) -> float:
    if pd.isna(value):
        return float("nan")
    return float((control < value).mean() * 100)


def build_case_summary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(DATA_DIR / "incidents.json") as handle:
        incidents = json.load(handle)

    case_rows: list[dict[str, float | str | int]] = []
    control_rows: list[pd.DataFrame] = []
    historical_next_frames: list[pd.DataFrame] = []
    historical_control_frames: list[pd.DataFrame] = []

    for incident in incidents:
        label = incident["label"]
        stem = stem_for_label(label)

        incident_df = load_pitch_file(DATA_DIR / "incident_games" / f"{stem}.csv")
        baseline_df = load_pitch_file(DATA_DIR / "baselines" / f"{stem}_season.csv")

        next_path = DATA_DIR / "next_games" / f"{stem}_next.csv"
        next_df = load_pitch_file(next_path) if next_path.exists() else None

        baseline_mask = ~baseline_df["game_pk"].isin(
            [incident["game_pk"], incident.get("next_game_pk")]
        )
        control_df = baseline_df.loc[baseline_mask].copy()
        control_games = per_game_metrics(control_df)
        control_games["label"] = label
        control_games["umpire_name"] = incident["umpire_name"]
        control_rows.append(control_games)

        incident_metrics = sample_metrics(incident_df)
        row: dict[str, float | str | int] = {
            "label": label,
            "date": incident["date"],
            "umpire_name": incident["umpire_name"],
            "incident_game_pk": incident["game_pk"],
            "baseline_game_count": int(len(control_games)),
            "baseline_pitch_count": int(len(control_df)),
        }

        for metric_name, metric_value in incident_metrics.items():
            row[f"incident_{metric_name}"] = metric_value

        for metric_name in [
            "called_strike_rate",
            "shadow_strike_rate",
            "oob_strike_rate",
            "zone_ball_rate",
            "miss_rate",
        ]:
            control_mean = float(control_games[metric_name].mean())
            row[f"baseline_{metric_name}_mean"] = control_mean
            row[f"incident_{metric_name}_delta_vs_baseline"] = incident_metrics[metric_name] - control_mean
            row[f"incident_{metric_name}_percentile"] = compute_percentile(
                control_games[metric_name], incident_metrics[metric_name]
            )

        if next_df is not None:
            next_metrics = sample_metrics(next_df)
            row["next_game_pk"] = incident["next_game_pk"]
            row["next_game_date"] = incident["next_game_date"]
            row["next_pitch_count"] = int(len(next_df))

            next_df = next_df.copy()
            next_df["label"] = label
            historical_next_frames.append(next_df)

            control_hist_df = control_df.copy()
            control_hist_df["label"] = label
            historical_control_frames.append(control_hist_df)

            for metric_name, metric_value in next_metrics.items():
                row[f"next_{metric_name}"] = metric_value

            for metric_name in [
                "called_strike_rate",
                "shadow_strike_rate",
                "oob_strike_rate",
                "zone_ball_rate",
                "miss_rate",
            ]:
                control_mean = row[f"baseline_{metric_name}_mean"]
                row[f"next_{metric_name}_delta_vs_baseline"] = next_metrics[metric_name] - control_mean
                row[f"next_{metric_name}_percentile"] = compute_percentile(
                    control_games[metric_name], next_metrics[metric_name]
                )

        case_rows.append(row)

    case_summary = pd.DataFrame(case_rows)
    game_controls = pd.concat(control_rows, ignore_index=True)
    historical_next = pd.concat(historical_next_frames, ignore_index=True)
    historical_controls = pd.concat(historical_control_frames, ignore_index=True)
    return case_summary, game_controls, historical_next, historical_controls


def paired_tests(case_summary: pd.DataFrame) -> pd.DataFrame:
    completed = case_summary.dropna(subset=["next_called_strike_rate"]).copy()
    rows = []
    metric_labels = {
        "called_strike_rate": "Called-strike rate",
        "shadow_strike_rate": "Shadow strike rate",
        "oob_strike_rate": "Out-of-zone strike rate",
        "zone_ball_rate": "In-zone ball rate",
        "miss_rate": "Rulebook miss rate",
    }

    for metric_name, label in metric_labels.items():
        deltas = completed[f"next_{metric_name}_delta_vs_baseline"].dropna().to_numpy(dtype=float)
        mean_delta = float(deltas.mean())
        t_stat, t_pvalue = stats.ttest_1samp(deltas, 0.0)
        ci_low, ci_high = stats.t.interval(
            0.95, len(deltas) - 1, loc=mean_delta, scale=stats.sem(deltas)
        )
        sign_negative = int((deltas < 0).sum())
        sign_positive = int((deltas > 0).sum())
        sign_pvalue = stats.binomtest(sign_negative, len(deltas), 0.5, alternative="greater").pvalue

        rows.append(
            {
                "metric": label,
                "metric_key": metric_name,
                "n_cases": int(len(deltas)),
                "mean_delta": mean_delta,
                "mean_delta_pp": mean_delta * 100,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_low_pp": ci_low * 100,
                "ci_high_pp": ci_high * 100,
                "t_pvalue": float(t_pvalue),
                "t_stat": float(t_stat),
                "cases_lower_than_baseline": sign_negative,
                "cases_higher_than_baseline": sign_positive,
                "exact_sign_pvalue_one_sided": float(sign_pvalue),
            }
        )

    return pd.DataFrame(rows)


def edge_profile(df: pd.DataFrame, label: str) -> pd.DataFrame:
    clipped = df[df["edge_dist_in"].between(EDGE_BIN_BREAKS.min(), EDGE_BIN_BREAKS.max())].copy()
    clipped["edge_bin"] = pd.cut(clipped["edge_dist_in"], EDGE_BIN_BREAKS, right=False, include_lowest=True)
    rows = []
    for interval, group in clipped.groupby("edge_bin", observed=False):
        if pd.isna(interval):
            continue
        strikes = int(group["called_strike"].sum())
        total = int(len(group))
        rate = strikes / total if total else float("nan")
        ci_low, ci_high = stats.binomtest(strikes, total).proportion_ci(confidence_level=0.95, method="wilson")
        rows.append(
            {
                "series": label,
                "bin_left": float(interval.left),
                "bin_right": float(interval.right),
                "bin_mid": float((interval.left + interval.right) / 2),
                "n": total,
                "called_strikes": strikes,
                "rate": rate,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
    return pd.DataFrame(rows)


def save_chart_historical_deltas(completed: pd.DataFrame, output_path: Path) -> None:
    chart_df = completed.copy().sort_values("next_oob_strike_rate_delta_vs_baseline")
    chart_df["case_label"] = chart_df["label"].str.replace(" ", "\n", n=1)
    y = np.arange(len(chart_df))

    metrics = [
        ("next_called_strike_rate_delta_vs_baseline", "Called-strike delta", "#35618f"),
        ("next_oob_strike_rate_delta_vs_baseline", "Out-of-zone strike delta", "#c44536"),
        ("next_miss_rate_delta_vs_baseline", "Rulebook miss delta", "#2a7f62"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6), sharey=True)
    for axis, (metric_name, title, color) in zip(axes, metrics):
        values = chart_df[metric_name] * 100
        axis.axvline(0, color="#222222", linewidth=1, alpha=0.5)
        axis.hlines(y, 0, values, color=color, linewidth=2.5, alpha=0.9)
        axis.scatter(values, y, color=color, s=65, zorder=3)
        for y_pos, value in zip(y, values):
            ha = "left" if value >= 0 else "right"
            offset = 0.35 if value >= 0 else -0.35
            axis.text(value + offset, y_pos, f"{value:+.1f}", va="center", ha=ha, fontsize=9)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlabel("Percentage points")
        axis.grid(axis="x", linestyle=":", alpha=0.35)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(chart_df["case_label"], fontsize=9)
    fig.suptitle("Next-game deltas vs each umpire's season control distribution", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_chart_historical_percentiles(completed: pd.DataFrame, output_path: Path) -> None:
    chart_df = completed.copy().sort_values("next_oob_strike_rate_percentile")
    chart_df["case_label"] = chart_df["label"].str.replace(" ", "\n", n=1)
    y = np.arange(len(chart_df))

    metrics = [
        ("next_called_strike_rate_percentile", "Called-strike percentile", "#35618f"),
        ("next_oob_strike_rate_percentile", "Out-of-zone strike percentile", "#c44536"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    for axis, (metric_name, title, color) in zip(axes, metrics):
        values = chart_df[metric_name]
        axis.axvspan(25, 75, color="#f2f2f2", zorder=0)
        axis.axvline(50, color="#222222", linewidth=1, alpha=0.5)
        axis.hlines(y, 0, values, color=color, linewidth=2.5, alpha=0.9)
        axis.scatter(values, y, color=color, s=65, zorder=3)
        for y_pos, value in zip(y, values):
            axis.text(value + 1.5, y_pos, f"{value:.0f}", va="center", ha="left", fontsize=9)
        axis.set_xlim(0, 100)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.set_xlabel("Percentile within season control games")
        axis.grid(axis="x", linestyle=":", alpha=0.35)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(chart_df["case_label"], fontsize=9)
    fig.suptitle("How extreme were the post-incident games?", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_chart_edge_profile(
    historical_controls: pd.DataFrame,
    historical_next: pd.DataFrame,
    soler_incident: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    baseline_profile = edge_profile(historical_controls, "Historical controls")
    next_profile = edge_profile(historical_next, "Historical next games")
    soler_profile = edge_profile(soler_incident, "Soler incident game")
    profile_df = pd.concat([baseline_profile, next_profile, soler_profile], ignore_index=True)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    style_map = {
        "Historical controls": ("#6c757d", "-", 2.5),
        "Historical next games": ("#c44536", "-", 2.5),
        "Soler incident game": ("#2f6db5", "--", 2.2),
    }
    for series_name, series_df in profile_df.groupby("series"):
        color, linestyle, linewidth = style_map[series_name]
        ax.plot(
            series_df["bin_mid"],
            series_df["rate"] * 100,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker="o",
            markersize=4,
            label=series_name,
        )
        ax.fill_between(
            series_df["bin_mid"],
            series_df["ci_low"] * 100,
            series_df["ci_high"] * 100,
            color=color,
            alpha=0.12,
        )

    ax.axvline(0, color="#222222", linewidth=1, alpha=0.5)
    ax.axvspan(-2, 2, color="#f4ede3", alpha=0.35)
    ax.set_title("Called-strike probability by distance from the rulebook edge", fontsize=14, fontweight="bold")
    ax.set_xlabel("Inches from rulebook edge (negative = off the plate, positive = inside zone)")
    ax.set_ylabel("Called-strike probability")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return profile_df


def save_chart_soler_live(case_summary: pd.DataFrame, output_path: Path) -> None:
    soler = case_summary.loc[case_summary["label"] == "Soler-Lopez 2026"].iloc[0]
    chart_rows = [
        {
            "metric": "Called-strike rate",
            "delta_pp": soler["incident_called_strike_rate_delta_vs_baseline"] * 100,
            "percentile": soler["incident_called_strike_rate_percentile"],
        },
        {
            "metric": "Shadow strike rate",
            "delta_pp": soler["incident_shadow_strike_rate_delta_vs_baseline"] * 100,
            "percentile": soler["incident_shadow_strike_rate_percentile"],
        },
        {
            "metric": "Out-of-zone strike rate",
            "delta_pp": soler["incident_oob_strike_rate_delta_vs_baseline"] * 100,
            "percentile": soler["incident_oob_strike_rate_percentile"],
        },
        {
            "metric": "In-zone ball rate",
            "delta_pp": soler["incident_zone_ball_rate_delta_vs_baseline"] * 100,
            "percentile": soler["incident_zone_ball_rate_percentile"],
        },
        {
            "metric": "Rulebook miss rate",
            "delta_pp": soler["incident_miss_rate_delta_vs_baseline"] * 100,
            "percentile": soler["incident_miss_rate_percentile"],
        },
    ]
    chart_df = pd.DataFrame(chart_rows).sort_values("delta_pp")

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    y = np.arange(len(chart_df))
    colors = ["#c44536" if value > 0 else "#2f6db5" for value in chart_df["delta_pp"]]
    ax.axvline(0, color="#222222", linewidth=1, alpha=0.5)
    ax.barh(y, chart_df["delta_pp"], color=colors, alpha=0.9)
    for y_pos, delta_pp, percentile in zip(y, chart_df["delta_pp"], chart_df["percentile"]):
        label = f"{delta_pp:+.1f} pp | {ordinal(percentile)} pct"
        if delta_pp >= 0:
            anchor = delta_pp + 0.35
            ha = "left"
            color = "#222222"
        else:
            anchor = delta_pp + 0.45
            ha = "left"
            color = "white" if abs(delta_pp) >= 4 else "#222222"
        ax.text(anchor, y_pos, label, va="center", ha=ha, fontsize=9, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(chart_df["metric"], fontsize=10)
    ax.set_xlabel("Incident-game delta vs Edwin Moscoso control mean")
    ax.set_title("April 7, 2026 Soler-Lopez game vs Moscoso's local 2026 baseline", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_report(
    case_summary: pd.DataFrame,
    tests_df: pd.DataFrame,
    historical_next: pd.DataFrame,
    historical_controls: pd.DataFrame,
) -> str:
    completed = case_summary.dropna(subset=["next_called_strike_rate"]).copy()
    soler = case_summary.loc[case_summary["label"] == "Soler-Lopez 2026"].iloc[0]

    called_test = tests_df.loc[tests_df["metric_key"] == "called_strike_rate"].iloc[0]
    shadow_test = tests_df.loc[tests_df["metric_key"] == "shadow_strike_rate"].iloc[0]
    oob_test = tests_df.loc[tests_df["metric_key"] == "oob_strike_rate"].iloc[0]
    miss_test = tests_df.loc[tests_df["metric_key"] == "miss_rate"].iloc[0]

    pooled_next = sample_metrics(historical_next)
    pooled_controls = sample_metrics(historical_controls)

    summary_table = completed[
        [
            "label",
            "next_called_strike_rate",
            "next_called_strike_rate_delta_vs_baseline",
            "next_called_strike_rate_percentile",
            "next_oob_strike_rate",
            "next_oob_strike_rate_delta_vs_baseline",
            "next_oob_strike_rate_percentile",
            "next_miss_rate",
            "next_miss_rate_delta_vs_baseline",
        ]
    ].copy()
    summary_table["next_called_strike_rate"] = summary_table["next_called_strike_rate"].map(fmt_rate)
    summary_table["next_called_strike_rate_delta_vs_baseline"] = summary_table[
        "next_called_strike_rate_delta_vs_baseline"
    ].map(fmt_pp)
    summary_table["next_called_strike_rate_percentile"] = summary_table[
        "next_called_strike_rate_percentile"
    ].map(lambda value: f"{value:.0f}")
    summary_table["next_oob_strike_rate"] = summary_table["next_oob_strike_rate"].map(fmt_rate)
    summary_table["next_oob_strike_rate_delta_vs_baseline"] = summary_table[
        "next_oob_strike_rate_delta_vs_baseline"
    ].map(fmt_pp)
    summary_table["next_oob_strike_rate_percentile"] = summary_table[
        "next_oob_strike_rate_percentile"
    ].map(lambda value: f"{value:.0f}")
    summary_table["next_miss_rate"] = summary_table["next_miss_rate"].map(fmt_rate)
    summary_table["next_miss_rate_delta_vs_baseline"] = summary_table[
        "next_miss_rate_delta_vs_baseline"
    ].map(fmt_pp)

    soler_table = pd.DataFrame(
        [
            {
                "metric": "Called-strike rate",
                "incident_game": fmt_rate(soler["incident_called_strike_rate"]),
                "baseline_mean": fmt_rate(soler["baseline_called_strike_rate_mean"]),
                "delta_vs_baseline": fmt_pp(soler["incident_called_strike_rate_delta_vs_baseline"]),
                "percentile": f"{soler['incident_called_strike_rate_percentile']:.0f}",
            },
            {
                "metric": "Shadow strike rate",
                "incident_game": fmt_rate(soler["incident_shadow_strike_rate"]),
                "baseline_mean": fmt_rate(soler["baseline_shadow_strike_rate_mean"]),
                "delta_vs_baseline": fmt_pp(soler["incident_shadow_strike_rate_delta_vs_baseline"]),
                "percentile": f"{soler['incident_shadow_strike_rate_percentile']:.0f}",
            },
            {
                "metric": "Out-of-zone strike rate",
                "incident_game": fmt_rate(soler["incident_oob_strike_rate"]),
                "baseline_mean": fmt_rate(soler["baseline_oob_strike_rate_mean"]),
                "delta_vs_baseline": fmt_pp(soler["incident_oob_strike_rate_delta_vs_baseline"]),
                "percentile": f"{soler['incident_oob_strike_rate_percentile']:.0f}",
            },
            {
                "metric": "In-zone ball rate",
                "incident_game": fmt_rate(soler["incident_zone_ball_rate"]),
                "baseline_mean": fmt_rate(soler["baseline_zone_ball_rate_mean"]),
                "delta_vs_baseline": fmt_pp(soler["incident_zone_ball_rate_delta_vs_baseline"]),
                "percentile": f"{soler['incident_zone_ball_rate_percentile']:.0f}",
            },
            {
                "metric": "Rulebook miss rate",
                "incident_game": fmt_rate(soler["incident_miss_rate"]),
                "baseline_mean": fmt_rate(soler["baseline_miss_rate_mean"]),
                "delta_vs_baseline": fmt_pp(soler["incident_miss_rate_delta_vs_baseline"]),
                "percentile": f"{soler['incident_miss_rate_percentile']:.0f}",
            },
        ]
    )

    return f"""# Research Report: Bench-Clearing Incidents and the Next Umpire Zone

Prepared on April 10, 2026 inside `research/soler-lopez-brawl/codex-analysis-2026-04-10/`.

## Executive Summary
- Historical sample: **{len(completed)} completed incident/next-game pairs**, **{fmt_number(len(historical_next))} next-game called pitches**, and **{fmt_number(len(historical_controls))} same-umpire control pitches** from the repo baselines.
- Broad verdict: **kill the simple \"umpires change the whole zone after a fight\" angle**. The next-game called-strike rate moved only **{called_test['mean_delta_pp']:+.1f} percentage points**, with a 95% CI of **{called_test['ci_low_pp']:+.1f} to {called_test['ci_high_pp']:+.1f} pp** and a paired t-test p-value of **{called_test['t_pvalue']:.3f}**.
- Narrower signal: the **outside edge tightened**. Out-of-zone called strikes fell by **{oob_test['mean_delta_pp']:+.1f} pp** on average (95% CI **{oob_test['ci_low_pp']:+.1f} to {oob_test['ci_high_pp']:+.1f} pp**, paired t-test p-value **{oob_test['t_pvalue']:.3f}**), and **all {int(oob_test['cases_lower_than_baseline'])} of {int(oob_test['n_cases'])}** completed follow-up games were below the umpire's own baseline.
- The shadow-band rate also leaned lower (**{shadow_test['mean_delta_pp']:+.1f} pp**) but the interval still crosses zero, so that piece is suggestive rather than decisive.
- Rulebook miss rate dropped by **{miss_test['mean_delta_pp']:+.1f} pp** (95% CI **{miss_test['ci_low_pp']:+.1f} to {miss_test['ci_high_pp']:+.1f} pp**, paired t-test p-value **{miss_test['t_pvalue']:.3f}**), which means the post-incident games were, if anything, **cleaner and slightly more literal**, not more theatrical.
- Live 2026 status: the local folder includes the April 7, 2026 incident game and Edwin Moscoso's 2026 baseline, but **does not include `data/next_games/Soler-Lopez_2026_next.csv`** as of April 10, 2026. That makes the Soler-Lopez case a watchlist item, not a finished answer.

## Recommended Framing
- Primary take: **Umpires don't suddenly open up the whole zone after a brawl.**
- Secondary take: **What does change is the freebie off the plate.** Historical next games gave hitters fewer out-of-zone called strikes than the same umpires normally do.
- Editorial read: the cleanest publishable angle is **\"Umpires Don't Flinch, But They Do Stop Giving the Black\"** or **\"After a Fight, the Zone Doesn't Get Bigger. It Gets Cleaner.\"**

## Historical Case Table
{markdown_table(
    summary_table,
    [
        "label",
        "next_called_strike_rate",
        "next_called_strike_rate_delta_vs_baseline",
        "next_called_strike_rate_percentile",
        "next_oob_strike_rate",
        "next_oob_strike_rate_delta_vs_baseline",
        "next_oob_strike_rate_percentile",
        "next_miss_rate",
        "next_miss_rate_delta_vs_baseline",
    ],
    {
        "label": "Incident",
        "next_called_strike_rate": "Next called-strike",
        "next_called_strike_rate_delta_vs_baseline": "Delta vs base",
        "next_called_strike_rate_percentile": "CS pctile",
        "next_oob_strike_rate": "Next OOB strike",
        "next_oob_strike_rate_delta_vs_baseline": "OOB delta",
        "next_oob_strike_rate_percentile": "OOB pctile",
        "next_miss_rate": "Next miss",
        "next_miss_rate_delta_vs_baseline": "Miss delta",
    },
)}

## Charts
### 1. Historical deltas
![Historical deltas](charts/historical_next_game_deltas.png)

### 2. Historical percentiles
![Historical percentiles](charts/historical_next_game_percentiles.png)

### 3. Called-strike profile by edge distance
![Edge profile](charts/edge_distance_profile.png)

### 4. Soler-Lopez live watch
![Soler live watch](charts/soler_incident_live_watch.png)

## Soler-Lopez 2026: What We Can Say Right Now
The live April 7, 2026 game already looks unusual relative to Moscoso's local 2026 control sample, but it is still **the incident game itself**, not the follow-up game.

{markdown_table(
    soler_table,
    ["metric", "incident_game", "baseline_mean", "delta_vs_baseline", "percentile"],
    {
        "metric": "Metric",
        "incident_game": "Apr 7 game",
        "baseline_mean": "Control mean",
        "delta_vs_baseline": "Delta",
        "percentile": "Pctile",
    },
)}

What that means:
- Moscoso's April 7 game was already at just the **{ordinal(soler['incident_called_strike_rate_percentile'])} percentile** for overall called-strike rate.
- His out-of-zone strike rate was even lower, at the **{ordinal(soler['incident_oob_strike_rate_percentile'])} percentile**.
- The tradeoff was more missed in-zone strikes: the in-zone ball rate was at the **{ordinal(soler['incident_zone_ball_rate_percentile'])} percentile**.
- The local folder stops before the key test. Once the next Edwin Moscoso plate assignment is pulled into `data/next_games/Soler-Lopez_2026_next.csv`, rerunning this script will slot it into the historical table automatically.

## Method
1. Treated each CSV as a called-pitch sample only (`ball` or `called_strike`).
2. Built a rulebook zone from `plate_x`, `plate_z`, `sz_top`, and `sz_bot`, expanding by one baseball radius so the ball center can legally clip the zone.
3. Defined `edge_dist_in` as signed inches from the rulebook boundary.
   Positive values are inside the zone.
   Negative values are off the plate or out of the vertical zone.
4. Defined the shadow band as pitches within **2 inches** of the rulebook edge.
5. Compared each next game to a same-umpire control distribution built from the provided season baseline CSV, excluding the incident game and the tracked next game.
6. Evaluated both overall called-strike rate and location-specific metrics:
   out-of-zone strike rate,
   shadow strike rate,
   in-zone ball rate,
   and total rulebook miss rate.

## What Is Strong vs Weak
- Strongest finding: **fewer called strikes on pitches off the plate in the next game**.
- Weak finding: any sweeping change in total called-strike rate.
- Important caveat: this is still only **{len(completed)} historical follow-up games**, so the headline should stay narrow and honest.
- Another caveat: the baseline files are repo-provided season windows, not full multi-year umpire histories.
- Final caveat: this uses rulebook geometry from Statcast location fields, not CalledThird's D1 grading table.

## Pooled Context
- Pooled historical next games: called-strike rate **{fmt_rate(pooled_next['called_strike_rate'])}**, out-of-zone strike rate **{fmt_rate(pooled_next['oob_strike_rate'])}**, miss rate **{fmt_rate(pooled_next['miss_rate'])}**.
- Pooled historical controls: called-strike rate **{fmt_rate(pooled_controls['called_strike_rate'])}**, out-of-zone strike rate **{fmt_rate(pooled_controls['oob_strike_rate'])}**, miss rate **{fmt_rate(pooled_controls['miss_rate'])}**.

## Source Context
- April 7, 2026 game recap: [ESPN/AP](https://www.espn.com/mlb/recap/_/gameId/401814853)
- April 8, 2026 suspension update: [ESPN/AP](https://www.espn.com/mlb/story/_/id/48432086/braves-lopez-angels-soler-suspended-7-games-brawl)

Note on the current hook: the April 8, 2026 ESPN/AP update reports that MLB initially handed both players seven-game suspensions, then later reduced Reynaldo Lopez's suspension to five games while Jorge Soler's remained at seven. The original brief in this folder lists Soler at seven and Lopez at five, so the brief matches the later update rather than the initial announcement.
"""


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    case_summary, game_controls, historical_next, historical_controls = build_case_summary()
    tests_df = paired_tests(case_summary)

    completed = case_summary.dropna(subset=["next_called_strike_rate"]).copy()
    soler_incident = load_pitch_file(DATA_DIR / "incident_games" / "Soler-Lopez_2026.csv")

    case_summary.to_csv(TABLE_DIR / "case_summary.csv", index=False)
    completed.to_csv(TABLE_DIR / "historical_completed_cases.csv", index=False)
    game_controls.to_csv(TABLE_DIR / "game_level_controls.csv", index=False)
    tests_df.to_csv(TABLE_DIR / "historical_paired_tests.csv", index=False)

    save_chart_historical_deltas(completed, CHART_DIR / "historical_next_game_deltas.png")
    save_chart_historical_percentiles(completed, CHART_DIR / "historical_next_game_percentiles.png")
    edge_profile_df = save_chart_edge_profile(
        historical_controls,
        historical_next,
        soler_incident,
        CHART_DIR / "edge_distance_profile.png",
    )
    edge_profile_df.to_csv(TABLE_DIR / "edge_profile.csv", index=False)
    save_chart_soler_live(case_summary, CHART_DIR / "soler_incident_live_watch.png")

    report_text = render_report(case_summary, tests_df, historical_next, historical_controls)
    (SCRIPT_DIR / "REPORT.md").write_text(report_text)


if __name__ == "__main__":
    main()

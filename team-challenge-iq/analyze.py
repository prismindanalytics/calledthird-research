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

COUNT_BUCKET_ORDER = ["early", "middle", "late"]
COUNT_ORDER = ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"]

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIR = SCRIPT_DIR.parent
DATA_DIR = CASE_DIR / "data"
CHART_DIR = SCRIPT_DIR / "charts"
TABLE_DIR = SCRIPT_DIR / "tables"


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def ensure_dirs() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def wilson_interval(successes: float, trials: float, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    phat = successes / trials
    denom = 1 + z**2 / trials
    center = (phat + z**2 / (2 * trials)) / denom
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * trials)) / trials) / denom
    return center - margin, center + margin


def pct(value: float) -> float:
    return value * 100


def load_challenges() -> pd.DataFrame:
    with open(DATA_DIR / "all_challenges_detail.json") as handle:
        challenges = pd.DataFrame(json.load(handle))

    challenges = challenges.copy()
    challenges["game_date"] = pd.to_datetime(challenges["game_date"], errors="coerce")
    challenges = challenges.loc[
        (challenges["game_date"] >= pd.Timestamp("2026-03-27"))
        & (challenges["game_date"] <= pd.Timestamp("2026-04-14"))
    ].copy()
    challenges["challenge_value"] = pd.to_numeric(challenges["challenge_value"], errors="coerce").fillna(0.0)
    challenges["count"] = challenges["balls"].astype(int).astype(str) + "-" + challenges["strikes"].astype(int).astype(str)
    challenges["count_bucket"] = [
        "late" if count in {"2-2", "3-0", "3-1", "3-2"} else "early" if count in {"0-0", "0-1", "1-0"} else "middle"
        for count in challenges["count"]
    ]
    challenges["challenging_team"] = np.where(
        challenges["challenger"].eq("batter"),
        challenges["team_batting"],
        challenges["team_fielding"],
    )
    return challenges


def load_games() -> pd.DataFrame:
    with open(DATA_DIR / "all_games_with_abs.json") as handle:
        games = pd.DataFrame(json.load(handle))
    return games


def build_team_summary(challenges: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    game_rows = []
    teams = sorted(set(games["home_team"]).union(games["away_team"]))
    for team in teams:
        game_rows.append(
            {
                "team": team,
                "games": int(((games["home_team"] == team) | (games["away_team"] == team)).sum()),
            }
        )
    team_games = pd.DataFrame(game_rows)

    summary = (
        challenges.groupby("challenging_team")
        .agg(
            challenges=("id", "count"),
            overturns=("overturned", "sum"),
            overturn_rate=("overturned", "mean"),
            total_value=("challenge_value", "sum"),
            mean_value=("challenge_value", "mean"),
            mean_edge=("edge_distance_in", "mean"),
        )
        .reset_index()
        .rename(columns={"challenging_team": "team"})
    )
    summary = summary.merge(team_games, on="team", how="left")
    summary["challenges_per_game"] = summary["challenges"] / summary["games"]

    role = challenges.pivot_table(index="challenging_team", columns="challenger", values="id", aggfunc="count", fill_value=0)
    role = role.div(role.sum(axis=1), axis=0).add_suffix("_share").reset_index().rename(columns={"challenging_team": "team"})

    bucket = challenges.pivot_table(index="challenging_team", columns="count_bucket", values="id", aggfunc="count", fill_value=0)
    bucket = bucket.div(bucket.sum(axis=1), axis=0).add_suffix("_share").reset_index().rename(columns={"challenging_team": "team"})

    summary = summary.merge(role, on="team", how="left").merge(bucket, on="team", how="left")

    ci = summary.apply(lambda row: wilson_interval(float(row["overturns"]), float(row["challenges"])), axis=1, result_type="expand")
    summary["ci_low"] = ci[0]
    summary["ci_high"] = ci[1]

    league_mean_value_by_count = challenges.groupby("count")["challenge_value"].mean()
    overall_mean_value = float(challenges["challenge_value"].mean())
    bonuses = []
    for team, team_df in challenges.groupby("challenging_team"):
        expected_total = float(team_df["count"].map(league_mean_value_by_count).sum())
        actual_total = float(team_df["challenge_value"].sum())
        bonuses.append(
            {
                "team": team,
                "expected_total_value_from_mix": expected_total,
                "selection_bonus": expected_total - len(team_df) * overall_mean_value,
                "execution_bonus": actual_total - expected_total,
            }
        )
    summary = summary.merge(pd.DataFrame(bonuses), on="team", how="left")
    summary.to_csv(TABLE_DIR / "team_summary.csv", index=False)
    return summary


def build_twins_dbacks_tables(challenges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = challenges.loc[challenges["challenging_team"].isin(["MIN", "AZ"])].copy()

    role = (
        subset.groupby(["challenging_team", "challenger"])
        .agg(
            challenges=("id", "count"),
            overturn_rate=("overturned", "mean"),
            total_value=("challenge_value", "sum"),
            mean_value=("challenge_value", "mean"),
            mean_edge=("edge_distance_in", "mean"),
        )
        .reset_index()
        .rename(columns={"challenging_team": "team"})
    )

    bucket = (
        subset.groupby(["challenging_team", "count_bucket"])
        .agg(
            challenges=("id", "count"),
            overturn_rate=("overturned", "mean"),
            total_value=("challenge_value", "sum"),
        )
        .reset_index()
        .rename(columns={"challenging_team": "team"})
    )

    count = (
        subset.groupby(["challenging_team", "count"])
        .agg(
            challenges=("id", "count"),
            overturn_rate=("overturned", "mean"),
            total_value=("challenge_value", "sum"),
        )
        .reset_index()
        .rename(columns={"challenging_team": "team"})
    )

    role.to_csv(TABLE_DIR / "twins_dbacks_role.csv", index=False)
    bucket.to_csv(TABLE_DIR / "twins_dbacks_bucket.csv", index=False)
    count.to_csv(TABLE_DIR / "twins_dbacks_count.csv", index=False)
    return role, bucket, count


def plot_volume_efficiency(team_summary: pd.DataFrame) -> None:
    plot_df = team_summary.copy()
    fig, ax = plt.subplots(figsize=(10.5, 7))
    scatter = ax.scatter(
        plot_df["challenges"],
        pct(plot_df["overturn_rate"]),
        s=plot_df["total_value"] * 160,
        c=plot_df["mean_value"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xlabel("Total challenges")
    ax.set_ylabel("Overturn rate (%)")
    ax.set_title("Minnesota Is the Volume Team; Arizona Is the Precision Team")

    label_teams = set(plot_df.nlargest(6, "total_value")["team"])
    label_teams.update({"MIN", "AZ"})
    for _, row in plot_df.iterrows():
        if row["team"] in label_teams:
            ax.text(row["challenges"] + 0.35, pct(row["overturn_rate"]) + 0.2, row["team"], fontsize=9)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.92)
    cbar.set_label("Run value per challenge")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "volume_efficiency_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_twins_dbacks_role(role_df: pd.DataFrame) -> None:
    plot_df = role_df.loc[role_df["challenger"].isin(["batter", "catcher"])].copy()
    colors = {"MIN": "#4c7a89", "AZ": "#d17c2f"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    x = np.arange(2)
    width = 0.34
    for idx, team in enumerate(["MIN", "AZ"]):
        team_df = plot_df.loc[plot_df["team"] == team].set_index("challenger").reindex(["batter", "catcher"])
        axes[0].bar(x + (idx - 0.5) * width, team_df["challenges"], width=width, color=colors[team], label=team)
        axes[1].bar(x + (idx - 0.5) * width, pct(team_df["overturn_rate"]), width=width, color=colors[team], label=team)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(["Batter", "Catcher"])

    axes[0].set_ylabel("Challenges")
    axes[0].set_title("Arizona Leans More Heavily on Catchers")

    axes[1].set_ylabel("Overturn rate (%)")
    axes[1].set_title("Minnesota's Batter Challenges Lag Badly Behind")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(CHART_DIR / "twins_dbacks_role_split.png", bbox_inches="tight")
    plt.close(fig)


def plot_twins_dbacks_bucket(bucket_df: pd.DataFrame) -> None:
    plot_df = bucket_df.copy()
    colors = {"MIN": "#4c7a89", "AZ": "#d17c2f"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    x = np.arange(len(COUNT_BUCKET_ORDER))
    width = 0.34

    for idx, team in enumerate(["MIN", "AZ"]):
        team_df = plot_df.loc[plot_df["team"] == team].set_index("count_bucket").reindex(COUNT_BUCKET_ORDER)
        share = team_df["challenges"] / team_df["challenges"].sum()
        axes[0].bar(x + (idx - 0.5) * width, pct(share), width=width, color=colors[team], label=team)
        axes[1].bar(x + (idx - 0.5) * width, team_df["total_value"], width=width, color=colors[team], label=team)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(COUNT_BUCKET_ORDER)
        ax.set_xlabel("Count bucket")

    axes[0].set_ylabel("Share of team challenges (%)")
    axes[0].set_title("Twins Spend More of Their Volume in Late Counts")

    axes[1].set_ylabel("Total run value")
    axes[1].set_title("Arizona's Caution Leaves the Late-Count Board Empty")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(CHART_DIR / "twins_dbacks_count_buckets.png", bbox_inches="tight")
    plt.close(fig)


def plot_selection_execution(team_summary: pd.DataFrame) -> None:
    plot_df = team_summary.copy()
    fig, ax = plt.subplots(figsize=(10.5, 7))
    scatter = ax.scatter(
        plot_df["selection_bonus"],
        plot_df["execution_bonus"],
        s=plot_df["challenges"] * 8,
        c=plot_df["total_value"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(0, color="#444444", lw=1, linestyle="--")
    ax.axvline(0, color="#444444", lw=1, linestyle="--")
    ax.set_xlabel("Selection bonus from count mix (runs vs league-average mix)")
    ax.set_ylabel("Execution bonus within chosen counts (runs vs expected)")
    ax.set_title("Twins Buy Better Leverage; Diamondbacks Choose Cleaner Spots")

    label_teams = {"MIN", "AZ"}
    label_teams.update(set(plot_df.nlargest(5, "total_value")["team"]))
    for _, row in plot_df.iterrows():
        if row["team"] in label_teams:
            ax.text(row["selection_bonus"] + 0.015, row["execution_bonus"] + 0.015, row["team"], fontsize=9)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.92)
    cbar.set_label("Total run value captured")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "selection_execution_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_twins_dbacks_count_detail(count_df: pd.DataFrame) -> None:
    plot_df = count_df.copy()
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    x = np.arange(len(COUNT_ORDER))
    width = 0.34
    colors = {"MIN": "#4c7a89", "AZ": "#d17c2f"}

    for idx, team in enumerate(["MIN", "AZ"]):
        team_df = plot_df.loc[plot_df["team"] == team].set_index("count").reindex(COUNT_ORDER).fillna(0.0)
        ax.bar(x + (idx - 0.5) * width, team_df["challenges"], width=width, color=colors[team], label=team)

    ax.set_xticks(x)
    ax.set_xticklabels(COUNT_ORDER)
    ax.set_ylabel("Challenges")
    ax.set_xlabel("Count")
    ax.set_title("Minnesota Attacks 2-2 and 3-2; Arizona Barely Lives There")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "twins_dbacks_count_detail.png", bbox_inches="tight")
    plt.close(fig)


def print_summary(team_summary: pd.DataFrame) -> None:
    subset = team_summary.loc[team_summary["team"].isin(["MIN", "AZ"])].copy().set_index("team")
    print("Twins vs Diamondbacks")
    for team in ["MIN", "AZ"]:
        row = subset.loc[team]
        print(
            f"  {team}: {int(row['challenges'])} challenges over {int(row['games'])} games "
            f"({row['challenges_per_game']:.2f}/game), overturn {pct(row['overturn_rate']):.1f}%, "
            f"value {row['total_value']:.3f}, late share {pct(row['late_share']):.1f}%, catcher share {pct(row['catcher_share']):.1f}%"
        )
    print(
        f"  selection bonus MIN {subset.loc['MIN', 'selection_bonus']:.3f} vs AZ {subset.loc['AZ', 'selection_bonus']:.3f}; "
        f"execution bonus MIN {subset.loc['MIN', 'execution_bonus']:.3f} vs AZ {subset.loc['AZ', 'execution_bonus']:.3f}"
    )


def main() -> None:
    configure_style()
    ensure_dirs()
    challenges = load_challenges()
    games = load_games()
    team_summary = build_team_summary(challenges, games)
    role_df, bucket_df, count_df = build_twins_dbacks_tables(challenges)

    plot_volume_efficiency(team_summary)
    plot_twins_dbacks_role(role_df)
    plot_twins_dbacks_bucket(bucket_df)
    plot_selection_execution(team_summary)
    plot_twins_dbacks_count_detail(count_df)
    print_summary(team_summary)


if __name__ == "__main__":
    main()

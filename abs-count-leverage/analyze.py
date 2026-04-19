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

COUNT_ORDER = ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"]
EDGE_BUCKETS = ["<=0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0+"]
LATE_COUNTS = {"2-2", "3-0", "3-1", "3-2"}

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIR = SCRIPT_DIR.parent
RESEARCH_DIR = CASE_DIR.parent
COUNT_SOURCE_DIR = RESEARCH_DIR / "count-distribution-abs" / "data"
CHALLENGE_SOURCE_DIR = RESEARCH_DIR / "team-challenge-iq" / "data"
TABLE_DIR = SCRIPT_DIR / "tables"
CHART_DIR = SCRIPT_DIR / "charts"


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "DejaVu Sans",
        }
    )


def ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> float:
    return value * 100


def count_sort_key(value: str) -> int:
    return COUNT_ORDER.index(value)


def fmt_pp(value: float) -> str:
    return f"{value:+.2f} pp"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_runs(value: float) -> str:
    return f"{value:.3f}"


def wilson_interval(successes: float, trials: float, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    phat = successes / trials
    denom = 1 + z**2 / trials
    center = (phat + z**2 / (2 * trials)) / denom
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * trials)) / trials) / denom
    return center - margin, center + margin


def markdown_table(df: pd.DataFrame, columns: list[str], renames: dict[str, str]) -> str:
    header = "| " + " | ".join(renames.get(column, column) for column in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def label_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["count"] = out["balls"].astype(int).astype(str) + "-" + out["strikes"].astype(int).astype(str)
    return out


def load_count_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    statcast_2026_files = [
        COUNT_SOURCE_DIR / "2026-03-27.parquet",
        COUNT_SOURCE_DIR / "2026-03-28.parquet",
        COUNT_SOURCE_DIR / "2026-03-29.parquet",
        COUNT_SOURCE_DIR / "2026-03-30.parquet",
        COUNT_SOURCE_DIR / "2026-03-31.parquet",
        COUNT_SOURCE_DIR / "2026-04-01.parquet",
        COUNT_SOURCE_DIR / "2026-04-02.parquet",
        COUNT_SOURCE_DIR / "2026-04-03.parquet",
        COUNT_SOURCE_DIR / "2026-04-04.parquet",
        COUNT_SOURCE_DIR / "2026-04-05.parquet",
        COUNT_SOURCE_DIR / "statcast_2026_apr06_14.parquet",
    ]
    statcast_2025 = pd.read_parquet(COUNT_SOURCE_DIR / "statcast_2025_mar27_apr14.parquet")
    statcast_2026 = pd.concat([pd.read_parquet(path) for path in statcast_2026_files], ignore_index=True)

    statcast_2025 = label_counts(statcast_2025)
    statcast_2026 = label_counts(statcast_2026)
    return statcast_2025, statcast_2026


def load_challenges() -> pd.DataFrame:
    with open(CHALLENGE_SOURCE_DIR / "all_challenges_detail.json") as handle:
        challenges = pd.DataFrame(json.load(handle))

    challenges = challenges.copy()
    challenges["game_date"] = pd.to_datetime(challenges["game_date"], errors="coerce")
    challenges = challenges.loc[
        (challenges["game_date"] >= pd.Timestamp("2026-03-27"))
        & (challenges["game_date"] <= pd.Timestamp("2026-04-14"))
    ].copy()
    challenges = label_counts(challenges)
    challenges["challenge_value"] = pd.to_numeric(challenges["challenge_value"], errors="coerce").fillna(0.0)
    challenges["overturned"] = challenges["overturned"].astype(int)
    challenges["challenging_team"] = np.where(
        challenges["challenger"].eq("batter"),
        challenges["team_batting"],
        challenges["team_fielding"],
    )
    return challenges


def build_count_context(statcast_2025: pd.DataFrame, statcast_2026: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, float | str]] = []
    total_2025 = len(statcast_2025)
    total_2026 = len(statcast_2026)
    counts_2025 = statcast_2025["count"].value_counts()
    counts_2026 = statcast_2026["count"].value_counts()

    for count in COUNT_ORDER:
        n_2025 = int(counts_2025.get(count, 0))
        n_2026 = int(counts_2026.get(count, 0))
        share_2025 = n_2025 / total_2025
        share_2026 = n_2026 / total_2026
        rows.append(
            {
                "count": count,
                "pitches_2025": n_2025,
                "pitches_2026": n_2026,
                "share_2025": share_2025,
                "share_2026": share_2026,
                "delta_pp": pct(share_2026 - share_2025),
            }
        )

    count_context = pd.DataFrame(rows)
    observed = pd.DataFrame(
        {
            "2025": count_context.set_index("count")["pitches_2025"],
            "2026": count_context.set_index("count")["pitches_2026"],
        }
    )
    chi2, p_value, dof, _ = stats.chi2_contingency(observed)

    at_bats_2025 = statcast_2025.groupby(["game_pk", "at_bat_number"]).agg(pitches=("pitch_number", "max"))
    at_bats_2026 = statcast_2026.groupby(["game_pk", "at_bat_number"]).agg(pitches=("pitch_number", "max"))
    summary = {
        "chi2": float(chi2),
        "chi2_p_value": float(p_value),
        "chi2_dof": int(dof),
        "sample_2025": int(total_2025),
        "sample_2026": int(total_2026),
        "mean_pitches_per_pa_2025": float(at_bats_2025["pitches"].mean()),
        "mean_pitches_per_pa_2026": float(at_bats_2026["pitches"].mean()),
        "deep_pa_share_2025": float((at_bats_2025["pitches"] >= 6).mean()),
        "deep_pa_share_2026": float((at_bats_2026["pitches"] >= 6).mean()),
    }
    return count_context, summary


def build_count_leverage(challenges: pd.DataFrame, statcast_2026: pd.DataFrame) -> pd.DataFrame:
    called = statcast_2026.loc[statcast_2026["description"].isin(["ball", "called_strike"])].copy()
    called_counts = (
        called.groupby("count")
        .size()
        .rename("called_pitches")
        .reset_index()
    )

    leverage = (
        challenges.groupby("count")
        .agg(
            challenges=("id", "count"),
            overturns=("overturned", "sum"),
            overturn_rate=("overturned", "mean"),
            total_value=("challenge_value", "sum"),
            avg_value=("challenge_value", "mean"),
            median_edge_in=("edge_distance_in", "median"),
        )
        .reset_index()
    )
    leverage = leverage.merge(called_counts, on="count", how="left")
    leverage["called_pitches"] = leverage["called_pitches"].fillna(0).astype(int)
    leverage["called_share"] = leverage["called_pitches"] / leverage["called_pitches"].sum()
    leverage["challenge_share"] = leverage["challenges"] / leverage["challenges"].sum()
    leverage["value_share"] = leverage["total_value"] / leverage["total_value"].sum()
    leverage["pressure_index"] = leverage["challenge_share"] / leverage["called_share"]
    leverage["value_index"] = leverage["value_share"] / leverage["called_share"]
    leverage["challenges_per_1000_called"] = leverage["challenges"] / leverage["called_pitches"] * 1000
    leverage["value_per_1000_called"] = leverage["total_value"] / leverage["called_pitches"] * 1000

    ci_bounds = leverage.apply(
        lambda row: wilson_interval(float(row["overturns"]), float(row["challenges"])),
        axis=1,
        result_type="expand",
    )
    leverage["overturn_ci_low"] = ci_bounds[0]
    leverage["overturn_ci_high"] = ci_bounds[1]
    leverage = leverage.sort_values("count", key=lambda series: series.map(count_sort_key)).reset_index(drop=True)
    return leverage


def build_role_table(challenges: pd.DataFrame) -> pd.DataFrame:
    role_table = (
        challenges.groupby(["count", "challenger"])
        .agg(
            challenges=("id", "count"),
            overturns=("overturned", "sum"),
            overturn_rate=("overturned", "mean"),
            total_value=("challenge_value", "sum"),
            avg_value=("challenge_value", "mean"),
        )
        .reset_index()
    )
    role_table["count_total"] = role_table.groupby("count")["challenges"].transform("sum")
    role_table["challenge_share_within_count"] = role_table["challenges"] / role_table["count_total"]
    role_table = role_table.sort_values(
        ["count", "challenger"],
        key=lambda series: series.map(count_sort_key) if series.name == "count" else series,
    ).reset_index(drop=True)
    return role_table


def build_edge_table(challenges: pd.DataFrame) -> pd.DataFrame:
    edge_table = challenges.dropna(subset=["edge_distance_in"]).copy()
    edge_table["edge_bucket"] = pd.cut(
        edge_table["edge_distance_in"],
        bins=[-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf],
        labels=EDGE_BUCKETS,
    )
    edge_table = (
        edge_table.groupby(["count", "edge_bucket"], observed=False)
        .agg(
            challenges=("id", "count"),
            overturn_rate=("overturned", "mean"),
            total_value=("challenge_value", "sum"),
            avg_value=("challenge_value", "mean"),
        )
        .reset_index()
    )
    edge_table = edge_table.sort_values(
        ["count", "edge_bucket"],
        key=lambda series: series.map(count_sort_key) if series.name == "count" else series,
    ).reset_index(drop=True)
    return edge_table


def build_team_strategy(challenges: pd.DataFrame, leverage: pd.DataFrame) -> pd.DataFrame:
    league_mean_value_by_count = leverage.set_index("count")["avg_value"]
    overall_mean_value = float(challenges["challenge_value"].mean())

    rows: list[dict[str, float | str]] = []
    for team, team_df in challenges.groupby("challenging_team"):
        challenge_count = len(team_df)
        expected_total = float(team_df["count"].map(league_mean_value_by_count).sum())
        actual_total = float(team_df["challenge_value"].sum())
        rows.append(
            {
                "team": team,
                "challenges": challenge_count,
                "actual_total_value": actual_total,
                "actual_value_per_challenge": actual_total / challenge_count,
                "expected_total_value_from_mix": expected_total,
                "selection_bonus_vs_average_mix": expected_total - challenge_count * overall_mean_value,
                "execution_bonus_vs_mix": actual_total - expected_total,
                "late_count_share": float(team_df["count"].isin(LATE_COUNTS).mean()),
                "overturn_rate": float(team_df["overturned"].mean()),
            }
        )

    team_strategy = pd.DataFrame(rows).sort_values("actual_total_value", ascending=False).reset_index(drop=True)
    return team_strategy


def save_tables(
    count_context: pd.DataFrame,
    leverage: pd.DataFrame,
    role_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    team_strategy: pd.DataFrame,
    context_summary: dict[str, float],
) -> None:
    count_context.to_csv(TABLE_DIR / "count_context.csv", index=False)
    leverage.to_csv(TABLE_DIR / "count_leverage.csv", index=False)
    role_table.to_csv(TABLE_DIR / "role_by_count.csv", index=False)
    edge_table.to_csv(TABLE_DIR / "edge_bucket_value.csv", index=False)
    team_strategy.to_csv(TABLE_DIR / "team_strategy.csv", index=False)
    with open(TABLE_DIR / "context_summary.json", "w") as handle:
        json.dump(context_summary, handle, indent=2)


def plot_count_distribution_delta(count_context: pd.DataFrame, context_summary: dict[str, float]) -> None:
    plot_df = count_context.copy().sort_values("delta_pp")
    colors = np.where(plot_df["delta_pp"] >= 0, "#d17c2f", "#2b6f87")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(plot_df["count"], plot_df["delta_pp"], color=colors, alpha=0.9)
    ax.axvline(0, color="#1f1f1f", lw=1)
    ax.set_xlabel("2026 minus 2025 share of all pitches (percentage points)")
    ax.set_ylabel("Count")
    ax.set_title("Matched-Window Count Distribution Shift Is Real but Small")
    subtitle = (
        f"Chi-square p={context_summary['chi2_p_value']:.2e}; "
        f"mean PA length {context_summary['mean_pitches_per_pa_2025']:.3f} -> {context_summary['mean_pitches_per_pa_2026']:.3f}"
    )
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color="#555555")

    for _, row in plot_df.iterrows():
        x = row["delta_pp"]
        ax.text(
            x + (0.02 if x >= 0 else -0.02),
            row["count"],
            f"{x:+.2f}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(CHART_DIR / "count_distribution_delta.png", bbox_inches="tight")
    plt.close(fig)


def plot_pressure_value(leverage: pd.DataFrame) -> None:
    plot_df = leverage.copy()
    x = np.arange(len(plot_df))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(x - width, pct(plot_df["called_share"]), width=width, color="#c7d6d5", label="Called pitch share")
    ax.bar(x, pct(plot_df["challenge_share"]), width=width, color="#4c7a89", label="Challenge share")
    ax.bar(x + width, pct(plot_df["value_share"]), width=width, color="#d17c2f", label="Run-value share")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["count"])
    ax.set_ylabel("Share of pool (%)")
    ax.set_xlabel("Count")
    ax.set_title("Late Counts Capture Far More ABS Value Than Their Called-Pitch Volume")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, max(pct(plot_df["called_share"]).max(), pct(plot_df["value_share"]).max()) + 8)

    for idx, row in plot_df.iterrows():
        ax.text(idx + width, pct(row["value_share"]) + 0.8, f"{row['value_index']:.1f}x", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(CHART_DIR / "pressure_value_by_count.png", bbox_inches="tight")
    plt.close(fig)


def plot_value_density(leverage: pd.DataFrame) -> None:
    plot_df = leverage.copy().sort_values("value_per_1000_called", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 7))
    ax.barh(plot_df["count"], plot_df["value_per_1000_called"], color="#d17c2f", alpha=0.9)
    ax.set_xlabel("Run value captured per 1,000 called pitches")
    ax.set_ylabel("Count")
    ax.set_title("Full Counts Are the Densest ABS Leverage Point")

    ax2 = ax.twiny()
    overturn = pct(plot_df["overturn_rate"])
    low_err = pct(plot_df["overturn_rate"] - plot_df["overturn_ci_low"])
    high_err = pct(plot_df["overturn_ci_high"] - plot_df["overturn_rate"])
    ax2.errorbar(
        overturn,
        plot_df["count"],
        xerr=[low_err, high_err],
        fmt="o",
        color="#24424d",
        ecolor="#24424d",
        elinewidth=1,
        capsize=3,
    )
    ax2.set_xlabel("Overturn rate (%) with Wilson 95% CI")
    ax2.set_xlim(0, max(85, float(np.nanmax(pct(leverage["overturn_ci_high"]))) + 5))

    for _, row in plot_df.iterrows():
        ax.text(row["value_per_1000_called"] + 0.15, row["count"], f"{row['challenges_per_1000_called']:.1f} ch/1k", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(CHART_DIR / "value_density_by_count.png", bbox_inches="tight")
    plt.close(fig)


def plot_role_mix(role_table: pd.DataFrame) -> None:
    pivot = (
        role_table.pivot(index="count", columns="challenger", values="challenge_share_within_count")
        .reindex(COUNT_ORDER)
        .fillna(0.0)
    )
    total_challenges = role_table.groupby("count")["challenges"].sum().reindex(COUNT_ORDER)

    colors = {"batter": "#d17c2f", "catcher": "#4c7a89", "pitcher": "#7a7a7a"}
    fig, ax = plt.subplots(figsize=(12, 6.5))
    bottom = np.zeros(len(pivot))
    for role in ["batter", "catcher", "pitcher"]:
        values = pivot.get(role, pd.Series(index=pivot.index, data=0.0)).to_numpy()
        ax.bar(pivot.index, pct(values), bottom=pct(bottom), color=colors[role], label=role.title())
        bottom += values

    ax.set_ylim(0, 110)
    ax.set_ylabel("Share of challenges within count (%)")
    ax.set_xlabel("Count")
    ax.set_title("Catchers Own Early Counts, but Batters Gain Share as Leverage Climbs")
    ax.legend(frameon=False, ncol=3, loc="upper left")

    for idx, count in enumerate(COUNT_ORDER):
        ax.text(idx, 102, f"n={int(total_challenges.loc[count])}", ha="center", fontsize=9, color="#555555")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "role_mix_by_count.png", bbox_inches="tight")
    plt.close(fig)


def plot_edge_heatmap(edge_table: pd.DataFrame) -> None:
    plot_df = edge_table.copy()
    matrix = (
        plot_df.pivot(index="count", columns="edge_bucket", values="total_value")
        .reindex(index=COUNT_ORDER, columns=EDGE_BUCKETS)
        .fillna(0.0)
    )
    counts_matrix = (
        plot_df.pivot(index="count", columns="edge_bucket", values="challenges")
        .reindex(index=COUNT_ORDER, columns=EDGE_BUCKETS)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(10, 7.5))
    im = ax.imshow(matrix.to_numpy(), cmap="YlOrBr", aspect="auto")
    ax.set_xticks(np.arange(len(EDGE_BUCKETS)))
    ax.set_xticklabels(EDGE_BUCKETS)
    ax.set_yticks(np.arange(len(COUNT_ORDER)))
    ax.set_yticklabels(COUNT_ORDER)
    ax.set_xlabel("Distance from edge (inches inside or just off the zone)")
    ax.set_ylabel("Count")
    ax.set_title("The Biggest ABS Value Lives in Late Counts on Near-Edge Calls")

    for i, count in enumerate(COUNT_ORDER):
        for j, bucket in enumerate(EDGE_BUCKETS):
            value = matrix.loc[count, bucket]
            challenges = counts_matrix.loc[count, bucket]
            label = f"{value:.1f}\n(n={int(challenges)})" if challenges > 0 else ""
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color="#2b1f0f")

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Total run value captured")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "edge_value_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_team_strategy(team_strategy: pd.DataFrame) -> None:
    plot_df = team_strategy.copy()
    fig, ax = plt.subplots(figsize=(10.5, 7))
    scatter = ax.scatter(
        plot_df["selection_bonus_vs_average_mix"],
        plot_df["execution_bonus_vs_mix"],
        s=plot_df["challenges"] * 7,
        c=plot_df["actual_total_value"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(0, color="#444444", lw=1, linestyle="--")
    ax.axvline(0, color="#444444", lw=1, linestyle="--")
    ax.set_xlabel("Selection bonus from count mix (runs vs league-average mix)")
    ax.set_ylabel("Execution bonus within chosen counts (runs vs expected)")
    ax.set_title("Some Teams Buy Better Count Mixes, but Execution Still Drives Total Value")

    label_teams = set(plot_df.nlargest(8, "actual_total_value")["team"])
    label_teams.update(plot_df.nlargest(4, "selection_bonus_vs_average_mix")["team"])
    label_teams.update(plot_df.nsmallest(4, "execution_bonus_vs_mix")["team"])
    for _, row in plot_df.iterrows():
        if row["team"] in label_teams:
            ax.text(
                row["selection_bonus_vs_average_mix"] + 0.015,
                row["execution_bonus_vs_mix"] + 0.015,
                row["team"],
                fontsize=9,
            )

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.9)
    cbar.set_label("Actual run value captured")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "team_selection_execution.png", bbox_inches="tight")
    plt.close(fig)


def print_summary(count_context: pd.DataFrame, leverage: pd.DataFrame, team_strategy: pd.DataFrame, context_summary: dict[str, float]) -> None:
    full_count = leverage.loc[leverage["count"] == "3-2"].iloc[0]
    two_two = leverage.loc[leverage["count"] == "2-2"].iloc[0]
    top_selection = team_strategy.nlargest(5, "selection_bonus_vs_average_mix")[
        ["team", "selection_bonus_vs_average_mix", "late_count_share"]
    ]
    top_execution = team_strategy.nlargest(5, "execution_bonus_vs_mix")[
        ["team", "execution_bonus_vs_mix", "actual_total_value"]
    ]

    print("Count context")
    print(
        f"  2025 sample={context_summary['sample_2025']:,} 2026 sample={context_summary['sample_2026']:,} "
        f"chi2 p={context_summary['chi2_p_value']:.2e}"
    )
    print(
        "  mean pitches per PA "
        f"{context_summary['mean_pitches_per_pa_2025']:.3f} -> {context_summary['mean_pitches_per_pa_2026']:.3f}"
    )
    print(
        "  6+ pitch PA share "
        f"{fmt_pct(context_summary['deep_pa_share_2025'])} -> {fmt_pct(context_summary['deep_pa_share_2026'])}"
    )
    print("Leverage leaders")
    print(
        f"  3-2: called share={pct(full_count['called_share']):.2f}% "
        f"challenge share={pct(full_count['challenge_share']):.2f}% "
        f"value share={pct(full_count['value_share']):.2f}% "
        f"value density={full_count['value_per_1000_called']:.2f}"
    )
    print(
        f"  2-2: called share={pct(two_two['called_share']):.2f}% "
        f"challenge share={pct(two_two['challenge_share']):.2f}% "
        f"value share={pct(two_two['value_share']):.2f}% "
        f"value density={two_two['value_per_1000_called']:.2f}"
    )
    print("Top selection teams")
    print(top_selection.round(3).to_string(index=False))
    print("Top execution teams")
    print(top_execution.round(3).to_string(index=False))


def main() -> None:
    configure_style()
    ensure_output_dirs()

    statcast_2025, statcast_2026 = load_count_windows()
    challenges = load_challenges()

    count_context, context_summary = build_count_context(statcast_2025, statcast_2026)
    leverage = build_count_leverage(challenges, statcast_2026)
    role_table = build_role_table(challenges)
    edge_table = build_edge_table(challenges)
    team_strategy = build_team_strategy(challenges, leverage)

    save_tables(count_context, leverage, role_table, edge_table, team_strategy, context_summary)
    plot_count_distribution_delta(count_context, context_summary)
    plot_pressure_value(leverage)
    plot_value_density(leverage)
    plot_role_mix(role_table)
    plot_edge_heatmap(edge_table)
    plot_team_strategy(team_strategy)
    print_summary(count_context, leverage, team_strategy, context_summary)


if __name__ == "__main__":
    main()

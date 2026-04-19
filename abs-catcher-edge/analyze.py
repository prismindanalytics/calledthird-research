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
import statsmodels.formula.api as smf
from scipy import stats

COUNT_ORDER = ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"]
COUNT_BUCKET_ORDER = ["early", "middle", "late"]
EDGE_BUCKET_ORDER = ["<=0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0+"]

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_DIR = SCRIPT_DIR.parent
RESEARCH_DIR = CASE_DIR.parent
DATA_DIR = RESEARCH_DIR / "team-challenge-iq" / "data"
CHART_DIR = SCRIPT_DIR / "charts"
TABLE_DIR = SCRIPT_DIR / "tables"


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
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
    challenges["catcher_flag"] = challenges["challenger"].eq("catcher").astype(int)
    return challenges


def build_role_summary(challenges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    bc = challenges.loc[challenges["challenger"].isin(["batter", "catcher"])].copy()
    summary = (
        bc.groupby("challenger")
        .agg(
            challenges=("id", "count"),
            overturns=("overturned", "sum"),
            overturn_rate=("overturned", "mean"),
            total_value=("challenge_value", "sum"),
            mean_value=("challenge_value", "mean"),
            mean_edge=("edge_distance_in", "mean"),
        )
        .reset_index()
    )

    ci = summary.apply(
        lambda row: wilson_interval(float(row["overturns"]), float(row["challenges"])),
        axis=1,
        result_type="expand",
    )
    summary["ci_low"] = ci[0]
    summary["ci_high"] = ci[1]

    table = pd.crosstab(bc["challenger"], bc["overturned"]).reindex(index=["batter", "catcher"], columns=[0, 1], fill_value=0)
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    raw_or = (table.loc["catcher", 1] / table.loc["catcher", 0]) / (table.loc["batter", 1] / table.loc["batter", 0])

    model_df = bc.dropna(subset=["edge_distance_in"]).copy()
    model = smf.logit(
        "overturned ~ catcher_flag + edge_distance_in + C(count_bucket)",
        data=model_df,
    ).fit(disp=False)
    model_table = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    model_table["odds_ratio"] = np.exp(model_table["Coef."])
    model_table["or_ci_low"] = np.exp(model_table["[0.025"])
    model_table["or_ci_high"] = np.exp(model_table["0.975]"])

    stats_summary = {
        "raw_chi2": float(chi2),
        "raw_p_value": float(p_value),
        "raw_odds_ratio_catcher_vs_batter": float(raw_or),
        "model_n": int(len(model_df)),
        "model_catcher_coef": float(model.params["catcher_flag"]),
        "model_catcher_p_value": float(model.pvalues["catcher_flag"]),
        "model_catcher_odds_ratio": float(np.exp(model.params["catcher_flag"])),
        "model_catcher_or_ci_low": float(np.exp(model.conf_int().loc["catcher_flag", 0])),
        "model_catcher_or_ci_high": float(np.exp(model.conf_int().loc["catcher_flag", 1])),
    }

    summary.to_csv(TABLE_DIR / "role_summary.csv", index=False)
    model_table.to_csv(TABLE_DIR / "model_summary.csv", index=False)
    with open(TABLE_DIR / "role_stats_summary.json", "w") as handle:
        json.dump(stats_summary, handle, indent=2)

    return summary, stats_summary


def build_bucket_summary(challenges: pd.DataFrame) -> pd.DataFrame:
    bc = challenges.loc[challenges["challenger"].isin(["batter", "catcher"])].copy()
    summary = (
        bc.groupby(["count_bucket", "challenger"])
        .agg(
            challenges=("id", "count"),
            overturns=("overturned", "sum"),
            overturn_rate=("overturned", "mean"),
            total_value=("challenge_value", "sum"),
            mean_value=("challenge_value", "mean"),
            mean_edge=("edge_distance_in", "mean"),
        )
        .reset_index()
    )
    summary["count_bucket"] = pd.Categorical(summary["count_bucket"], categories=COUNT_BUCKET_ORDER, ordered=True)
    summary = summary.sort_values(["count_bucket", "challenger"]).reset_index(drop=True)
    summary.to_csv(TABLE_DIR / "bucket_role_summary.csv", index=False)
    return summary


def build_edge_summary(challenges: pd.DataFrame) -> pd.DataFrame:
    bc = challenges.loc[challenges["challenger"].isin(["batter", "catcher"]) & challenges["edge_distance_in"].notna()].copy()
    bc["edge_bucket"] = pd.cut(
        bc["edge_distance_in"],
        bins=[-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf],
        labels=EDGE_BUCKET_ORDER,
    )
    summary = (
        bc.groupby(["edge_bucket", "challenger"], observed=False)
        .agg(
            challenges=("id", "count"),
            overturn_rate=("overturned", "mean"),
            total_value=("challenge_value", "sum"),
            mean_value=("challenge_value", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(TABLE_DIR / "edge_role_summary.csv", index=False)
    return summary


def build_count_summary(challenges: pd.DataFrame) -> pd.DataFrame:
    bc = challenges.loc[challenges["challenger"].isin(["batter", "catcher"])].copy()
    summary = (
        bc.groupby(["count", "challenger"])
        .agg(
            challenges=("id", "count"),
            overturn_rate=("overturned", "mean"),
            total_value=("challenge_value", "sum"),
            mean_value=("challenge_value", "mean"),
            mean_edge=("edge_distance_in", "mean"),
        )
        .reset_index()
    )
    wide = summary.pivot(index="count", columns="challenger")
    out = pd.DataFrame({"count": COUNT_ORDER})
    for metric in ["challenges", "overturn_rate", "total_value", "mean_value", "mean_edge"]:
        out[f"batter_{metric}"] = [wide.get((metric, "batter"), pd.Series()).get(count, np.nan) for count in COUNT_ORDER]
        out[f"catcher_{metric}"] = [wide.get((metric, "catcher"), pd.Series()).get(count, np.nan) for count in COUNT_ORDER]
    out["overturn_diff_pp"] = pct(out["catcher_overturn_rate"] - out["batter_overturn_rate"])
    out["mean_value_diff"] = out["catcher_mean_value"] - out["batter_mean_value"]
    out["mean_edge_diff"] = out["catcher_mean_edge"] - out["batter_mean_edge"]
    out.to_csv(TABLE_DIR / "count_role_summary.csv", index=False)
    return out


def build_team_mix(challenges: pd.DataFrame) -> pd.DataFrame:
    team = (
        challenges.groupby("challenging_team")
        .agg(
            challenges=("id", "count"),
            overturn_rate=("overturned", "mean"),
            total_value=("challenge_value", "sum"),
        )
        .reset_index()
    )
    role_counts = challenges.pivot_table(index="challenging_team", columns="challenger", values="id", aggfunc="count", fill_value=0)
    role = role_counts.div(role_counts.sum(axis=1), axis=0).add_suffix("_share").reset_index()
    mix = team.merge(role, on="challenging_team", how="left").fillna(0.0)
    mix.to_csv(TABLE_DIR / "team_role_mix.csv", index=False)
    return mix


def plot_overall_role_summary(summary: pd.DataFrame) -> None:
    plot_df = summary.copy().set_index("challenger").loc[["batter", "catcher"]].reset_index()
    colors = {"batter": "#d17c2f", "catcher": "#4c7a89"}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    axes[0].bar(plot_df["challenger"], pct(plot_df["overturn_rate"]), color=[colors[x] for x in plot_df["challenger"]])
    axes[0].errorbar(
        plot_df["challenger"],
        pct(plot_df["overturn_rate"]),
        yerr=[pct(plot_df["overturn_rate"] - plot_df["ci_low"]), pct(plot_df["ci_high"] - plot_df["overturn_rate"])],
        fmt="none",
        ecolor="#333333",
        capsize=4,
    )
    axes[0].set_title("Overturn Rate")
    axes[0].set_ylabel("Percent")
    for _, row in plot_df.iterrows():
        axes[0].text(row["challenger"], pct(row["overturn_rate"]) + 1.4, f"{pct(row['overturn_rate']):.1f}%", ha="center")

    axes[1].bar(plot_df["challenger"], plot_df["mean_value"], color=[colors[x] for x in plot_df["challenger"]])
    axes[1].set_title("Run Value per Challenge")
    axes[1].set_ylabel("Runs")
    for _, row in plot_df.iterrows():
        axes[1].text(row["challenger"], row["mean_value"] + 0.002, f"{row['mean_value']:.3f}", ha="center")

    axes[2].bar(plot_df["challenger"], plot_df["mean_edge"], color=[colors[x] for x in plot_df["challenger"]])
    axes[2].set_title("Average Edge Distance")
    axes[2].set_ylabel("Inches from edge")
    for _, row in plot_df.iterrows():
        axes[2].text(row["challenger"], row["mean_edge"] + 0.03, f'{row["mean_edge"]:.2f}"', ha="center")

    fig.suptitle("Catchers Beat Batters on Accuracy, Value, and Pitch Difficulty", y=1.03, fontsize=16)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "overall_role_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_bucket_profiles(bucket_summary: pd.DataFrame) -> None:
    plot_df = bucket_summary.copy()
    colors = {"batter": "#d17c2f", "catcher": "#4c7a89"}
    x = np.arange(len(COUNT_BUCKET_ORDER))
    width = 0.33

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx, role in enumerate(["batter", "catcher"]):
        role_df = plot_df.loc[plot_df["challenger"] == role].set_index("count_bucket").reindex(COUNT_BUCKET_ORDER)
        axes[0].bar(x + (idx - 0.5) * width, pct(role_df["overturn_rate"]), width=width, color=colors[role], label=role.title())
        axes[1].bar(x + (idx - 0.5) * width, role_df["mean_value"], width=width, color=colors[role], label=role.title())

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(COUNT_BUCKET_ORDER)
    axes[0].set_ylabel("Overturn rate (%)")
    axes[0].set_title("Catchers Win in Every Count Bucket")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(COUNT_BUCKET_ORDER)
    axes[1].set_ylabel("Run value per challenge")
    axes[1].set_title("Late Counts Magnify the Role Gap")
    axes[1].legend(frameon=False, loc="upper left")

    for ax in axes:
        ax.set_xlabel("Count bucket")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "bucket_role_profiles.png", bbox_inches="tight")
    plt.close(fig)


def plot_edge_profiles(edge_summary: pd.DataFrame) -> None:
    plot_df = edge_summary.copy()
    colors = {"batter": "#d17c2f", "catcher": "#4c7a89"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for role in ["batter", "catcher"]:
        role_df = plot_df.loc[plot_df["challenger"] == role].set_index("edge_bucket").reindex(EDGE_BUCKET_ORDER)
        axes[0].plot(EDGE_BUCKET_ORDER, pct(role_df["overturn_rate"]), marker="o", linewidth=2.2, color=colors[role], label=role.title())
        axes[1].plot(EDGE_BUCKET_ORDER, role_df["mean_value"], marker="o", linewidth=2.2, color=colors[role], label=role.title())

    axes[0].set_title("Catchers Outperform Across Edge Buckets")
    axes[0].set_ylabel("Overturn rate (%)")
    axes[0].set_xlabel("Edge bucket (inches from border)")

    axes[1].set_title("Value Gap Is Largest on the Closest Calls")
    axes[1].set_ylabel("Run value per challenge")
    axes[1].set_xlabel("Edge bucket (inches from border)")
    axes[1].legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(CHART_DIR / "edge_role_profiles.png", bbox_inches="tight")
    plt.close(fig)


def plot_count_advantage(count_summary: pd.DataFrame) -> None:
    plot_df = count_summary.copy()
    colors = np.where(plot_df["overturn_diff_pp"] >= 0, "#4c7a89", "#d17c2f")

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.bar(plot_df["count"], plot_df["overturn_diff_pp"], color=colors)
    ax.axhline(0, color="#222222", lw=1)
    ax.set_title("Catcher Advantage Is Largest in 0-0, 0-1, 0-2, 2-2, and 3-2")
    ax.set_ylabel("Catcher minus batter overturn rate (pp)")
    ax.set_xlabel("Count")

    for _, row in plot_df.iterrows():
        if not np.isnan(row["overturn_diff_pp"]):
            ax.text(
                row["count"],
                row["overturn_diff_pp"] + (0.8 if row["overturn_diff_pp"] >= 0 else -0.8),
                f"{row['overturn_diff_pp']:+.1f}",
                ha="center",
                va="bottom" if row["overturn_diff_pp"] >= 0 else "top",
                fontsize=8.5,
            )

    fig.tight_layout()
    fig.savefig(CHART_DIR / "count_role_advantage.png", bbox_inches="tight")
    plt.close(fig)


def plot_team_mix(team_mix: pd.DataFrame) -> None:
    plot_df = team_mix.loc[team_mix["challenges"] >= 20].copy()

    fig, ax = plt.subplots(figsize=(10.5, 7))
    scatter = ax.scatter(
        plot_df["catcher_share"] * 100,
        plot_df["overturn_rate"] * 100,
        s=plot_df["challenges"] * 8,
        c=plot_df["total_value"],
        cmap="viridis",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xlabel("Catcher share of team challenges (%)")
    ax.set_ylabel("Team overturn rate (%)")
    ax.set_title("Using More Catcher Challenges Alone Does Not Guarantee Team Success")

    label_teams = set(plot_df.nlargest(4, "catcher_share")["challenging_team"])
    label_teams.update(plot_df.nsmallest(4, "catcher_share")["challenging_team"])
    label_teams.update(plot_df.nlargest(5, "total_value")["challenging_team"])
    for _, row in plot_df.iterrows():
        if row["challenging_team"] in label_teams:
            ax.text(row["catcher_share"] * 100 + 0.4, row["overturn_rate"] * 100 + 0.2, row["challenging_team"], fontsize=9)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.92)
    cbar.set_label("Total run value captured")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "team_role_mix.png", bbox_inches="tight")
    plt.close(fig)


def print_summary(role_summary: pd.DataFrame, stats_summary: dict[str, float], bucket_summary: pd.DataFrame) -> None:
    role_df = role_summary.set_index("challenger")
    batter = role_df.loc["batter"]
    catcher = role_df.loc["catcher"]
    late = bucket_summary.set_index(["count_bucket", "challenger"])
    print("Overall role split")
    print(
        f"  batter {int(batter['challenges'])} challenges, {pct(batter['overturn_rate']):.1f}% overturn, "
        f"{batter['mean_value']:.3f} value/challenge, {batter['mean_edge']:.2f} in edge distance"
    )
    print(
        f"  catcher {int(catcher['challenges'])} challenges, {pct(catcher['overturn_rate']):.1f}% overturn, "
        f"{catcher['mean_value']:.3f} value/challenge, {catcher['mean_edge']:.2f} in edge distance"
    )
    print(
        f"  raw OR={stats_summary['raw_odds_ratio_catcher_vs_batter']:.2f}, raw p={stats_summary['raw_p_value']:.2e}, "
        f"controlled OR={stats_summary['model_catcher_odds_ratio']:.2f}, controlled p={stats_summary['model_catcher_p_value']:.3f}"
    )
    print("Late bucket")
    print(
        f"  batter late overturn={pct(late.loc[('late', 'batter'), 'overturn_rate']):.1f}% "
        f"value={late.loc[('late', 'batter'), 'mean_value']:.3f}"
    )
    print(
        f"  catcher late overturn={pct(late.loc[('late', 'catcher'), 'overturn_rate']):.1f}% "
        f"value={late.loc[('late', 'catcher'), 'mean_value']:.3f}"
    )


def main() -> None:
    configure_style()
    ensure_dirs()
    challenges = load_challenges()
    role_summary, stats_summary = build_role_summary(challenges)
    bucket_summary = build_bucket_summary(challenges)
    edge_summary = build_edge_summary(challenges)
    count_summary = build_count_summary(challenges)
    team_mix = build_team_mix(challenges)

    plot_overall_role_summary(role_summary)
    plot_bucket_profiles(bucket_summary)
    plot_edge_profiles(edge_summary)
    plot_count_advantage(count_summary)
    plot_team_mix(team_mix)
    print_summary(role_summary, stats_summary, bucket_summary)


if __name__ == "__main__":
    main()

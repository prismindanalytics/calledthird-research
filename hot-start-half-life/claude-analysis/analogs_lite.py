"""analogs_lite.py — *light-touch* descriptive analog lookup.

Per AGENT_PROMPT.md: NO k-NN; that's Codex. Here we only do simple percentile/rank
lookups: where do each named hot starter's 22-game stats land in the 2022-2025 distribution
of all qualified hitters' first-22-game performance?

Method:
  - For each season 2022-2025, compute *first-22-game* per-player rate stats (BB%, K%,
    BABIP, ISO, wOBA) and full-season ROS (PA after game 22) rate stats.
  - Combine across seasons. For each 2026 named hot starter, compute the percentile of
    their through-22-game stat in the 2022-2025 first-22-game distribution.
  - Also report: median ROS performance for the analog 2022-2025 cohort whose 22-game
    stat fell in the same percentile decile.

This is descriptive only — not a model.

Outputs:
  data/analogs_lite.json
  charts/league_env_2022_2025.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"
CHARTS = CLAUDE / "charts"
CHARTS.mkdir(exist_ok=True)

from stabilization import load_pa_table, annotate_pa, PA_EVENTS, STAT_SPECS

STATS = ["BB%", "K%", "BABIP", "ISO", "wOBA"]


def first_22g_split(pa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each (season, batter), split PAs into first-22-team-games window vs rest.

    Approximation: define "first 22 games" as the first 22 unique game_pks that batter
    appeared in within their season. Returns two dataframes (first22, rest).
    """
    pa = pa.sort_values(["season", "batter", "game_date"])
    pa["game_n"] = pa.groupby(["season", "batter"])["game_pk"].transform(
        lambda s: pd.factorize(s)[0] + 1
    )
    first22 = pa[pa.game_n <= 22].copy()
    rest = pa[pa.game_n > 22].copy()
    return first22, rest


def player_season_rates(pa: pd.DataFrame) -> pd.DataFrame:
    g = pa.groupby(["season", "batter"])
    out = pd.DataFrame({
        "PA": g["is_pa"].sum(),
        "BB": g["is_bb"].sum(),
        "K": g["is_k"].sum(),
        "babip_num": g["babip_num"].sum(),
        "babip_den": g["babip_den"].sum(),
        "iso_num": g["iso_num"].sum(),
        "iso_den": g["iso_den"].sum(),
        "woba_num": g["woba_num"].sum(),
        "woba_den": g["woba_den"].sum(),
    }).reset_index()
    out["BB%"] = out.BB / out.PA.replace(0, np.nan)
    out["K%"] = out.K / out.PA.replace(0, np.nan)
    out["BABIP"] = out.babip_num / out.babip_den.replace(0, np.nan)
    out["ISO"] = out.iso_num / out.iso_den.replace(0, np.nan)
    out["wOBA"] = out.woba_num / out.woba_den.replace(0, np.nan)
    return out


def percentile_rank(value: float, dist: np.ndarray) -> float:
    if not np.isfinite(value) or len(dist) == 0:
        return float("nan")
    return float(np.mean(dist <= value) * 100)


def plot_league_env(pa_2226: pd.DataFrame, out_path: Path) -> None:
    """League BB%, K%, BABIP, ISO, wOBA per season for 2022-2025."""
    rows = []
    for s, sub in pa_2226.groupby("season"):
        rows.append({
            "season": int(s),
            "BB%": float(sub.is_bb.sum() / sub.is_pa.sum()),
            "K%": float(sub.is_k.sum() / sub.is_pa.sum()),
            "BABIP": float(sub.babip_num.sum() / max(sub.babip_den.sum(), 1)),
            "ISO": float(sub.iso_num.sum() / max(sub.iso_den.sum(), 1)),
            "wOBA": float(sub.woba_num.sum() / max(sub.woba_den.sum(), 1)),
        })
    df = pd.DataFrame(rows).sort_values("season")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, stat in zip(axes, STATS):
        ax.plot(df.season, df[stat], marker="o", color="#1f4e79")
        ax.set_title(stat)
        ax.set_xticks(df.season)
        ax.grid(True, alpha=0.3)
    fig.suptitle("League rate-stat environment, 2022-2025 (Statcast PA-level)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run() -> dict:
    print("[analogs] loading 2022-2025 PA table")
    pa_2226 = annotate_pa(load_pa_table((2022, 2023, 2024, 2025)))

    print("[analogs] splitting into first-22g vs rest")
    first22, rest = first_22g_split(pa_2226)

    f22_rates = player_season_rates(first22)
    rest_rates = player_season_rates(rest)
    # Restrict to player-seasons with >= 50 PA in first-22g and >= 100 PA rest of season
    f22_rates = f22_rates[f22_rates.PA >= 50]
    rest_rates = rest_rates[rest_rates.PA >= 100]
    pair = f22_rates.merge(rest_rates, on=["season", "batter"], suffixes=("_22g", "_ros"))
    print(f"[analogs] qualifying player-seasons (50+/100+): {len(pair)}")

    plot_league_env(pa_2226, CHARTS / "league_env_2022_2025.png")

    # Build per-stat distributions
    dist = {s: pair[f"{s}_22g"].dropna().values for s in STATS}

    # Load 2026 data for hot starters
    pa_2026_a = pd.read_parquet(DATA / "statcast_2026_mar27_apr22.parquet",
                                 columns=["game_pk", "batter", "at_bat_number", "events",
                                          "woba_value", "woba_denom", "babip_value", "iso_value"])
    pa_2026_b = pd.read_parquet(DATA / "statcast_2026_apr23_24.parquet",
                                 columns=["game_pk", "batter", "at_bat_number", "events",
                                          "woba_value", "woba_denom", "babip_value", "iso_value"])
    pa_2026 = pd.concat([pa_2026_a, pa_2026_b], ignore_index=True)
    pa_2026 = pa_2026[pa_2026["events"].notna() & pa_2026["events"].isin(PA_EVENTS)]
    pa_2026 = pa_2026.drop_duplicates(["game_pk", "batter", "at_bat_number"]).reset_index(drop=True)
    pa_2026["season"] = 2026
    pa_2026 = annotate_pa(pa_2026)

    named = pd.read_parquet(DATA / "named_hot_starters.parquet")

    # Compute 22-game stats for each hot starter
    out = {"n_analogs_2022_2025": int(len(pair)), "stats": {}}
    for _, row in named.iterrows():
        slug = row.slug
        mlbam = int(row.mlbam) if pd.notna(row.mlbam) else None
        if mlbam is None or row.role != "hitter":
            continue
        sub = pa_2026[pa_2026.batter == mlbam]
        if sub.empty:
            continue
        rates_player = {
            "PA": int(sub.is_pa.sum()),
            "BB%": float(sub.is_bb.sum() / max(sub.is_pa.sum(), 1)),
            "K%": float(sub.is_k.sum() / max(sub.is_pa.sum(), 1)),
            "BABIP": float(sub.babip_num.sum() / max(sub.babip_den.sum(), 1)),
            "ISO": float(sub.iso_num.sum() / max(sub.iso_den.sum(), 1)),
            "wOBA": float(sub.woba_num.sum() / max(sub.woba_den.sum(), 1)),
        }
        # Pct rank vs 2022-2025 first-22g distribution
        ranks = {}
        analog_ros = {}
        for stat in STATS:
            pct = percentile_rank(rates_player[stat], dist[stat])
            ranks[stat] = pct
            # Find analog cohort: same decile of 22g distribution
            if np.isfinite(pct):
                d = pair[f"{stat}_22g"].values
                lo_p = max(0, pct - 5) / 100
                hi_p = min(100, pct + 5) / 100
                lo_v = np.quantile(d, lo_p)
                hi_v = np.quantile(d, hi_p)
                cohort = pair[(pair[f"{stat}_22g"] >= lo_v) & (pair[f"{stat}_22g"] <= hi_v)]
                if len(cohort):
                    analog_ros[stat] = {
                        "cohort_size": int(len(cohort)),
                        "cohort_22g_q50": float(cohort[f"{stat}_22g"].median()),
                        "cohort_ros_q10": float(cohort[f"{stat}_ros"].quantile(0.10)),
                        "cohort_ros_q50": float(cohort[f"{stat}_ros"].median()),
                        "cohort_ros_q90": float(cohort[f"{stat}_ros"].quantile(0.90)),
                        "regression_to_mean": float(cohort[f"{stat}_22g"].median()
                                                    - cohort[f"{stat}_ros"].median()),
                    }
        out["stats"][slug] = {"22g": rates_player, "percentile_rank": ranks, "analog_ros": analog_ros}

    with open(DATA / "analogs_lite.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[analogs] saved -> {DATA / 'analogs_lite.json'}")
    return out


if __name__ == "__main__":
    run()

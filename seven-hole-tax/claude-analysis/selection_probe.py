"""selection_probe.py — Selection-effect diagnostic.

Question: do 7-hole batters challenge or take systematically harder pitches than
3-hole batters? If yes, the 'raw' overturn / called-strike gap is mostly an
artifact of *who challenges what*, not 'umpires call them differently'.

Two views:
  (1) On CHALLENGES: distribution of |edge_distance_in| by lineup spot. If
      7-hole batters challenge pitches farther from the edge (i.e. more obvious
      strikes), they will lose more challenges regardless of umpire bias.
  (2) On TAKEN PITCHES: distribution of (plate_x, plate_z) by lineup spot, and
      called-strike rate as a function of edge_distance_ft by lineup spot.

Tests reported:
  - Empirical KS test of |edge_distance_in| dist for spot=7 vs spot=3 (challenges)
  - Empirical KS test of edge_distance_ft dist for spot=7 vs spot=3 (taken)
  - Mean / median |edge| comparison
  - Per-bin called-strike rate (taken pitches): does the rate-vs-edge curve
    differ between spots?

Outputs:
  charts/selection_effect_distributions.png
  charts/selection_effect_called_rate_by_edge.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import data_prep

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)


def _ks_test(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 5 or len(y) < 5:
        return {"ks_stat": float("nan"), "pvalue": float("nan"), "n_x": int(len(x)), "n_y": int(len(y))}
    res = stats.ks_2samp(x, y)
    return {"ks_stat": float(res.statistic), "pvalue": float(res.pvalue), "n_x": int(len(x)), "n_y": int(len(y))}


def run() -> dict:
    challenges = data_prep.load_challenges()
    challenges = challenges.dropna(subset=["lineup_spot", "edge_distance_in_final"]).copy()
    challenges["lineup_spot"] = challenges["lineup_spot"].astype(int)
    taken = data_prep.load_taken_pitches()
    taken = taken.dropna(subset=["lineup_spot"]).copy()
    taken["lineup_spot"] = taken["lineup_spot"].astype(int)

    # --- Challenge-side selection
    edge_by_spot = {int(s): challenges[challenges["lineup_spot"] == s]["edge_distance_in_final"].values for s in range(1, 10)}
    ks_challenges = {f"spot_{k}_vs_spot_3": _ks_test(edge_by_spot[k], edge_by_spot[3]) for k in range(1, 10) if k != 3}
    chal_summary = (
        challenges.groupby("lineup_spot")["edge_distance_in_final"]
        .agg(n="size", median="median", mean="mean", q25=lambda v: v.quantile(0.25), q75=lambda v: v.quantile(0.75))
        .reset_index()
    )

    # Also: among BATTER-challenged calls (initial_call='Strike'), do 7-hole batters
    # pick obviously-bad challenges (high edge distance)?
    bat = challenges[challenges["challenger"] == "batter"]
    bat_summary = (
        bat.groupby("lineup_spot")["edge_distance_in_final"]
        .agg(n="size", median="median", mean="mean")
        .reset_index()
    )

    # --- Taken-pitch side selection
    taken_in_zone = taken[taken["edge_distance_ft"] <= 0]   # inside zone
    taken_out_of_zone = taken[taken["edge_distance_ft"] > 0]

    # KS on edge_distance_ft (signed) by spot vs spot 3
    ks_taken = {}
    for k in range(1, 10):
        if k == 3:
            continue
        ks_taken[f"spot_{k}_vs_spot_3"] = _ks_test(
            taken[taken["lineup_spot"] == k]["edge_distance_ft"].values,
            taken[taken["lineup_spot"] == 3]["edge_distance_ft"].values,
        )

    # Average |edge| by spot
    taken_summary = (
        taken.assign(absedge=lambda d: d["edge_distance_ft"].abs())
        .groupby("lineup_spot")["absedge"]
        .agg(n="size", median="median", mean="mean")
        .reset_index()
    )

    # Borderline-only summary (the H3 sample): mean called-strike rate AND mean edge
    borderline = taken[taken["edge_distance_ft"].abs() <= 0.3]
    bd_summary = (
        borderline.groupby("lineup_spot")
        .agg(n=("called_strike", "size"),
             mean_edge_ft=("edge_distance_ft", lambda v: v.abs().mean()),
             rate=("called_strike", "mean"))
        .reset_index()
    )

    # --- Plots
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), dpi=160)
    # Left: challenge edge distance by spot (boxplot)
    data = [edge_by_spot[s] for s in range(1, 10)]
    bp = axs[0].boxplot(data, tick_labels=[str(s) for s in range(1, 10)], showfliers=False, patch_artist=True, widths=0.6)
    for i, patch in enumerate(bp["boxes"]):
        spot = i + 1
        patch.set_facecolor("#ef4444" if spot == 7 else "#94a3b8")
        patch.set_edgecolor("black")
    axs[0].set_xlabel("Lineup spot")
    axs[0].set_ylabel("|edge distance| (inches)")
    axs[0].set_title("Selection effect: challenge edge-distance by lineup spot")
    axs[0].spines[["top", "right"]].set_visible(False)
    axs[0].grid(axis="y", linestyle=":", alpha=0.5)

    # Right: edge distance distribution among ALL TAKEN pitches by spot (signed)
    data_t = [taken[taken["lineup_spot"] == s]["edge_distance_ft"].values for s in range(1, 10)]
    bp2 = axs[1].boxplot(data_t, tick_labels=[str(s) for s in range(1, 10)], showfliers=False, patch_artist=True, widths=0.6)
    for i, patch in enumerate(bp2["boxes"]):
        spot = i + 1
        patch.set_facecolor("#ef4444" if spot == 7 else "#94a3b8")
        patch.set_edgecolor("black")
    axs[1].axhline(0, color="#1d4ed8", linestyle="--", linewidth=0.8)
    axs[1].set_xlabel("Lineup spot")
    axs[1].set_ylabel("signed edge_distance (ft); + = outside zone")
    axs[1].set_title("Selection effect: taken-pitch location by lineup spot")
    axs[1].spines[["top", "right"]].set_visible(False)
    axs[1].grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(CHARTS / "selection_effect_distributions.png")
    plt.close(fig)

    # Called-strike rate vs edge_distance, by spot (focus on spots 3 and 7)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    bins = np.linspace(-0.5, 0.5, 21)
    centers = 0.5 * (bins[1:] + bins[:-1])
    for spot, color in [(3, "#1d4ed8"), (7, "#ef4444"), (1, "#22c55e"), (9, "#f59e0b")]:
        sub = taken[taken["lineup_spot"] == spot]
        if len(sub) < 10:
            continue
        digit = np.digitize(sub["edge_distance_ft"].values, bins) - 1
        rates = []
        ns = []
        for b in range(len(bins) - 1):
            mask = digit == b
            n = int(mask.sum())
            ns.append(n)
            rates.append(sub.loc[mask, "called_strike"].mean() if n > 0 else np.nan)
        ax.plot(centers, rates, "o-", label=f"spot {spot}", color=color, linewidth=1.2, markersize=4)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlabel("signed edge_distance_ft (negative = inside zone)")
    ax.set_ylabel("called-strike rate")
    ax.set_title("Selection probe: called-strike rate vs edge distance, by lineup spot")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS / "selection_effect_called_rate_by_edge.png")
    plt.close(fig)

    # Interpretation: does spot 7 have systematically larger |edge| in challenges?
    seven_mean = float(chal_summary.loc[chal_summary["lineup_spot"] == 7, "mean"].iloc[0])
    three_mean = float(chal_summary.loc[chal_summary["lineup_spot"] == 3, "mean"].iloc[0])
    league_mean = float(challenges["edge_distance_in_final"].mean())
    explains = "explains" if (seven_mean - three_mean) >= 0.20 else ("partial" if (seven_mean - three_mean) >= 0.05 else "doesn't explain")
    return {
        "challenge_summary": chal_summary.to_dict(orient="records"),
        "challenge_batter_only_summary": bat_summary.to_dict(orient="records"),
        "taken_absedge_summary": taken_summary.to_dict(orient="records"),
        "borderline_summary": bd_summary.to_dict(orient="records"),
        "ks_challenges_vs_spot3": ks_challenges,
        "ks_taken_vs_spot3": ks_taken,
        "interpretation": {
            "spot7_chal_mean_edge_in": seven_mean,
            "spot3_chal_mean_edge_in": three_mean,
            "league_chal_mean_edge_in": league_mean,
            "spot7_minus_spot3_in": seven_mean - three_mean,
            "verdict": explains,
        },
    }


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str))

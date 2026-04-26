"""r2_charts.py — additional R2 charts.

  - R1 sanity check comparison: prior / observation / posterior for the 5 named starters.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"
CHARTS = CLAUDE / "charts/r2"
CHARTS.mkdir(parents=True, exist_ok=True)


def chart_named_starter_sanity():
    proj = json.load(open(DATA / "r2_named_hot_starter_projections.json"))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    names = []
    obs = []
    priors = []
    post_q10 = []
    post_q50 = []
    post_q90 = []
    r1_v = []
    r2_v = []
    for slug, d in proj.items():
        if "error" in d:
            continue
        if d.get("BF_22g"):
            # reliever — show K%
            names.append(d["name"] + " (RP K%)")
            obs.append(d["obs_K%"])
            priors.append(d["prior_K%"])
            post_q10.append(d["post_K%_q10"])
            post_q50.append(d["post_K%_q50"])
            post_q90.append(d["post_K%_q90"])
            r1_v.append(d.get("r1_verdict", ""))
            r2_v.append(d.get("r2_verdict_K%", ""))
        else:
            names.append(d["name"] + " (wOBA)")
            obs.append(d["obs_wOBA"])
            priors.append(d["prior_wOBA"])
            post_q10.append(d["post_wOBA_q10"])
            post_q50.append(d["post_wOBA_q50"])
            post_q90.append(d["post_wOBA_q90"])
            r1_v.append(d.get("r1_verdict", ""))
            r2_v.append(d.get("r2_verdict", ""))

    y = np.arange(len(names))
    obs = np.array(obs); priors = np.array(priors); post_q10 = np.array(post_q10)
    post_q50 = np.array(post_q50); post_q90 = np.array(post_q90)
    ax.errorbar(post_q50, y, xerr=[post_q50 - post_q10, post_q90 - post_q50],
                fmt="o", color="#1f4e79", ecolor="#1f4e79", capsize=5, lw=1.2,
                label="R2 posterior q10-q50-q90")
    ax.scatter(priors, y, color="#888", marker="s", s=60, label="3-yr prior")
    ax.scatter(obs, y, color="#b8392b", marker="x", s=80, label="2026 22-game obs")
    ax.set_yticks(y)
    labels = [f"{nm}  (R1: {r1}; R2: {r2})" for nm, r1, r2 in zip(names, r1_v, r2_v)]
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Rate (wOBA for hitters; K% for RP)")
    ax.set_title("R1 sanity check — 5 named hot starters under R2 corrected methodology")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(CHARTS / "r1_sanity_check_comparison.png", dpi=130)
    plt.close(fig)


def chart_xwoba_gap():
    """Top-10 over-performers and under-performers by xwOBA - wOBA gap."""
    import pandas as pd
    posts = pd.read_parquet(DATA / "r2_universe_posteriors.parquet")
    posts = posts.dropna(subset=["obs_xwOBA", "obs_wOBA"]).copy()
    posts["xwOBA_gap"] = posts["obs_xwOBA"] - posts["obs_wOBA"]
    over = posts.nsmallest(10, "xwOBA_gap")  # most negative gap = over-performers
    under = posts.nlargest(10, "xwOBA_gap")  # positive gap = under-performers (under-luck)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, df, title in [
        (axes[0], over, "Top-10 OVER-performers (wOBA >> xwOBA, BABIP-luck)"),
        (axes[1], under, "Top-10 UNDER-performers (xwOBA >> wOBA, bad-luck victims)"),
    ]:
        y = np.arange(len(df))
        gaps = df["xwOBA_gap"].values
        colors = ["#b8392b" if v <= 0 else "#1f4e79" for v in gaps]
        ax.barh(y, gaps, color=colors, alpha=0.85)
        ax.set_yticks(y)
        labels = [f"{r.player_name}  (PA {r.PA_22g})" for _, r in df.iterrows()]
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="#444", lw=0.8)
        ax.set_xlabel("xwOBA - wOBA (negative = over-performing)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(CHARTS / "xwoba_gap.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    chart_named_starter_sanity()
    chart_xwoba_gap()
    print("[charts] saved r1_sanity_check_comparison.png, xwoba_gap.png")

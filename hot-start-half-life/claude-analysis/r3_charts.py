"""r3_charts.py — supplementary R3 charts.

Generates:
  - charts/r3/named_starter_r2_vs_r3.png  (verdict & delta diff for 5 named starters)

The blend-RMSE bar and stabilization curves and sleeper bars are produced inside
their own modules.
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
CHARTS = CLAUDE / "charts/r3"
CHARTS.mkdir(parents=True, exist_ok=True)


def chart_named_starter_diff() -> None:
    r2 = json.load(open(DATA / "r2_named_hot_starter_projections.json"))
    r3 = json.load(open(DATA / "r3_named_starter_projections.json"))
    rows = []
    for slug in ["andy_pages", "ben_rice", "munetaka_murakami", "mike_trout"]:
        r2v = r2[slug]
        r3v = r3[slug]
        rows.append({
            "name": r3v["name"],
            "r2_q10": r2v["ROS_wOBA_minus_prior_q10"],
            "r2_q50": r2v["ROS_wOBA_minus_prior_q50"],
            "r2_q90": r2v["ROS_wOBA_minus_prior_q90"],
            "r3_q10": r3v["ROS_wOBA_minus_prior_q10"],
            "r3_q50": r3v["ROS_wOBA_minus_prior_q50"],
            "r3_q90": r3v["ROS_wOBA_minus_prior_q90"],
            "r2_verdict": r2v["r2_verdict"],
            "r3_verdict": r3v["verdict"],
        })

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        # R2 bar
        ax.errorbar(r["r2_q50"], y[i] + 0.18,
                    xerr=[[r["r2_q50"] - r["r2_q10"]], [r["r2_q90"] - r["r2_q50"]]],
                    fmt="o", color="#888", ecolor="#888", capsize=3, lw=1.0,
                    label="R2" if i == 0 else None)
        # R3 bar
        ax.errorbar(r["r3_q50"], y[i] - 0.18,
                    xerr=[[r["r3_q50"] - r["r3_q10"]], [r["r3_q90"] - r["r3_q50"]]],
                    fmt="o", color="#1f4e79", ecolor="#1f4e79", capsize=3, lw=1.2,
                    label="R3" if i == 0 else None)
        # Verdict labels
        ax.text(r["r3_q90"] + 0.005, y[i],
                f"R2: {r['r2_verdict']}  -->  R3: {r['r3_verdict']}",
                fontsize=9, va="center")
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("ROS wOBA - prior wOBA (q10/q50/q90)")
    ax.set_title("Named hot starters — R2 (hand-tuned 50/50 blend) vs R3 (learned blend on 2025 holdout)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS / "named_starter_r2_vs_r3.png", dpi=130)
    plt.close(fig)
    print("[chart] named_starter_r2_vs_r3.png")


if __name__ == "__main__":
    chart_named_starter_diff()

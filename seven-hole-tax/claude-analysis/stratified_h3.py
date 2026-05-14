"""stratified_h3.py — Stratified robustness for H3.

Strata (spot 7 vs spot 3 marginal effect on borderline called-strike rate):
  - batter handedness (L vs R)
  - count quadrant (hitter / pitcher / even)
  - pitch group (fastball / breaking / offspeed)

Lighter sampler than the main fit (2 chains, fewer draws). Same model spec.
Outputs JSON dict + a heat / forest chart at charts/h3_stratified_forest.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bayes_gam_h3 as h3
import data_prep

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"


def run(edge_cutoff_ft: float = 0.3) -> dict:
    df_full = data_prep.load_taken_pitches()
    df_full = h3._prepare_borderline(df_full, edge_cutoff_ft=edge_cutoff_ft)

    out: dict = {}
    for col in ["stand", "count_quadrant", "pitch_group"]:
        levels = df_full[col].dropna().unique().tolist()
        out[col] = {}
        for lvl in levels:
            sub = df_full[df_full[col] == lvl]
            if len(sub) < 1500 or sub["lineup_spot"].nunique() < 9:
                print(f"  Skip {col}={lvl}: n={len(sub)}, spots={sub['lineup_spot'].nunique()}")
                continue
            print(f"  Fitting {col}={lvl}: n={len(sub):,}")
            try:
                model, info = h3._build_model(sub, n_knots=4)
                import pymc as pm
                with model:
                    idata = pm.sample(
                        draws=600, tune=1000, chains=2, cores=2,
                        target_accept=0.95, progressbar=False, random_seed=hash(str(lvl)) % 10_000,
                        init="adapt_diag",
                    )
                forest = h3._marginal_pp_diff(idata, info, n_keep=200)
                spot7 = next((r for r in forest if r["spot"] == 7), None)
                out[col][str(lvl)] = {"n": len(sub), "spot7": spot7, "forest": forest}
            except Exception as e:
                print(f"    FAILED {col}={lvl}: {e}")
                continue

    # Summary forest plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160, sharey=False)
    titles = {"stand": "by batter handedness", "count_quadrant": "by count quadrant", "pitch_group": "by pitch group"}
    for ax, col in zip(axes, ["stand", "count_quadrant", "pitch_group"]):
        groups = list(out[col].keys())
        if not groups:
            ax.text(0.5, 0.5, "no fit", ha="center", va="center")
            ax.set_title(f"H3 spot-7 effect, {titles[col]}")
            continue
        med = [out[col][g]["spot7"]["median_pp"] if out[col][g]["spot7"] else np.nan for g in groups]
        lo = [(out[col][g]["spot7"]["median_pp"] - out[col][g]["spot7"]["ci_low_pp"]) if out[col][g]["spot7"] else 0 for g in groups]
        hi = [(out[col][g]["spot7"]["ci_high_pp"] - out[col][g]["spot7"]["median_pp"]) if out[col][g]["spot7"] else 0 for g in groups]
        labels = [f"{g}\nn={out[col][g]['n']:,}" for g in groups]
        pos = list(range(len(groups)))
        ax.errorbar(med, pos, xerr=[lo, hi], fmt="o", color="#1f2937", ecolor="#ef4444", elinewidth=4, capsize=0, markersize=6)
        ax.axvline(0, color="#1d4ed8", linestyle="--", linewidth=1)
        ax.set_yticks(pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Spot 7 - spot 3 (pp)")
        ax.set_title(f"H3 stratified by {col}")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("H3 stratified robustness: spot-7-vs-spot-3 borderline called-strike effect", fontsize=12)
    fig.tight_layout()
    fig.savefig(CHARTS / "h3_stratified_forest.png")
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str))

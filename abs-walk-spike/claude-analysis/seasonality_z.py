"""H2: Z-score of 2026 walk rate vs 2018-2025 April distribution.

Reproduces the substrate baseline:
  historical mean = 9.02% (incl IBB), SD = 0.171pp
  2026 same window (Mar 27 - Apr 22) walk rate = 9.77%
  Z = +4.41 sigma

We rebuild from primary data (april_walk_history.csv + 2026 parquet) so the
number is independently confirmed in our pipeline rather than blindly copied.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    APRIL_HISTORY_CSV, PROJECT_ROOT, load_2026,
    plate_appearances, walk_rate, walk_flag, restrict_to_primary_window,
)

OUT_DIR = PROJECT_ROOT / "claude-analysis"
CHART_DIR = OUT_DIR / "charts"
ART_DIR = OUT_DIR / "artifacts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)


def compute_2026_walk_rate(df26: pd.DataFrame) -> dict:
    pa = plate_appearances(df26)
    n = len(pa)
    w_incl = int(walk_flag(pa).sum())
    w_excl = int((pa["events"] == "walk").sum())
    return {
        "pa": n,
        "walks_incl_ibb": w_incl,
        "walks_excl_ibb": w_excl,
        "walk_rate_incl_ibb": w_incl / max(n, 1),
        "walk_rate_excl_ibb": w_excl / max(n, 1),
    }


def run() -> dict:
    history = pd.read_csv(APRIL_HISTORY_CSV)
    print(f"[seasonality] history rows: {len(history)}, years: {history['year'].tolist()}")
    incl = history["walk_rate_incl_ibb"].to_numpy(dtype=float)
    excl = history["walk_rate_excl_ibb"].to_numpy(dtype=float)
    yrs = history["year"].to_numpy(dtype=int)

    df26 = load_2026()
    full26 = compute_2026_walk_rate(df26)
    df26_pri = restrict_to_primary_window(df26)
    pri26 = compute_2026_walk_rate(df26_pri)
    print(f"[seasonality] 2026 full Mar27-Apr22 walk_rate (incl IBB): {full26['walk_rate_incl_ibb']:.4f}")
    print(f"[seasonality] 2026 primary Mar27-Apr14 walk_rate (incl IBB): {pri26['walk_rate_incl_ibb']:.4f}")

    mu_incl = float(incl.mean())
    sd_incl = float(incl.std(ddof=1))
    mu_excl = float(excl.mean())
    sd_excl = float(excl.std(ddof=1))
    z_incl = (full26["walk_rate_incl_ibb"] - mu_incl) / sd_incl
    z_excl = (full26["walk_rate_excl_ibb"] - mu_excl) / sd_excl
    above_max_incl_pp = (full26["walk_rate_incl_ibb"] - incl.max()) * 100.0
    above_max_excl_pp = (full26["walk_rate_excl_ibb"] - excl.max()) * 100.0

    # Percentile rank: 2026 vs the historical 7 years -> rank in 8 values
    all_incl = np.concatenate([incl, [full26["walk_rate_incl_ibb"]]])
    rank_2026 = int((all_incl < full26["walk_rate_incl_ibb"]).sum() + 1)
    n_total = len(all_incl)

    # Bootstrap CI for the Z-score using historical sample
    rng = np.random.default_rng(424242)
    n_boot = 5000
    z_boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(incl, size=len(incl), replace=True)
        mu_b = sample.mean(); sd_b = sample.std(ddof=1)
        if sd_b > 0:
            z_boot[i] = (full26["walk_rate_incl_ibb"] - mu_b) / sd_b
        else:
            z_boot[i] = np.nan
    z_lo = float(np.nanquantile(z_boot, 0.025))
    z_hi = float(np.nanquantile(z_boot, 0.975))

    # Plot
    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=140)
    bar_yrs = list(yrs) + [2026]
    bar_vals = list(incl * 100) + [full26["walk_rate_incl_ibb"] * 100]
    bar_colors = ["#1f77b4"] * len(yrs) + ["#d62728"]
    ax.bar(bar_yrs, bar_vals, color=bar_colors, alpha=0.85,
           edgecolor="black", linewidth=0.5)
    # Mean ± SD band
    ax.axhline(mu_incl * 100, color="0.4", linestyle="--", linewidth=1.0,
               label=f"2018-2025 mean = {mu_incl*100:.2f}%")
    ax.fill_between([2017, 2027], (mu_incl - sd_incl) * 100,
                    (mu_incl + sd_incl) * 100, color="0.7", alpha=0.25,
                    label=f"+/- 1 SD ({sd_incl*100:.2f}pp)")
    ax.set_xlim(2017, 2027)
    ax.set_xticks(bar_yrs)
    ax.set_xticklabels([str(y) for y in bar_yrs], rotation=0)
    # Add covid skip note
    if 2020 not in bar_yrs:
        ax.text(2020, mu_incl * 100 + 0.05, "no 2020\n(COVID)", ha="center",
                fontsize=8, color="0.4")
    ax.set_ylim(8.0, 11.0)
    # Z annotation in lower-right region
    ax.annotate(
        f"2026 = {full26['walk_rate_incl_ibb']*100:.2f}%\nZ = +{z_incl:.2f} sigma\n+{above_max_incl_pp:.2f}pp above prior max",
        xy=(2026, full26["walk_rate_incl_ibb"] * 100),
        xytext=(2022.0, 10.4),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10, ha="left",
    )
    ax.set_xlabel("Year"); ax.set_ylabel("Walk rate, incl IBB (%) — Mar 27 to Apr 22")
    ax.set_title("April walk rate, apples-to-apples Mar 27 - Apr 22 window — incl intentional walks")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "april_walk_history.png", bbox_inches="tight")
    plt.close(fig)

    out = {
        "history_years": yrs.tolist(),
        "history_mean_incl_ibb": mu_incl,
        "history_sd_incl_ibb": sd_incl,
        "history_min_incl_ibb": float(incl.min()),
        "history_max_incl_ibb": float(incl.max()),
        "history_mean_excl_ibb": mu_excl,
        "history_sd_excl_ibb": sd_excl,
        "year_2026_full_window": full26,
        "year_2026_primary_window": pri26,
        "z_score_incl_ibb": z_incl,
        "z_score_excl_ibb": z_excl,
        "z_ci95_incl_ibb": [z_lo, z_hi],
        "above_prior_max_incl_pp": float(above_max_incl_pp),
        "above_prior_max_excl_pp": float(above_max_excl_pp),
        "rank_2026_in_2018_2026": [rank_2026, n_total],
    }
    print(f"[seasonality] Z (incl IBB): {z_incl:.3f}, CI [{z_lo:.2f},{z_hi:.2f}]")
    print(f"[seasonality] Z (excl IBB): {z_excl:.3f}")
    print(f"[seasonality] above prior max: {above_max_incl_pp:.2f}pp incl, {above_max_excl_pp:.2f}pp excl")
    return out


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, default=str, indent=2))

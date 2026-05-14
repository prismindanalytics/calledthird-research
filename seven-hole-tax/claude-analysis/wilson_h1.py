"""wilson_h1.py — Raw H1 replication: overturn rate by lineup spot, Wilson CIs.

Two views:
  * ALL challenges (batter + catcher + pitcher).  This is the cleanest comparison
    of how each lineup spot's challenges resolve.
  * BATTER-only challenges.  This is the FanSided/Ringer "30.2%" comparison —
    a 7-hole batter wins THEIR challenge X% of the time.

Multiple-comparisons correction: Holm and Benjamini-Hochberg on each-spot vs
spot-3 z tests.

Outputs:
  charts/h1_overturn_by_spot.png
  charts/h1_overturn_batter_only.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import data_prep

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float, float]:
    if trials == 0:
        return float("nan"), float("nan"), float("nan")
    phat = successes / trials
    denom = 1 + z**2 / trials
    center = (phat + z**2 / (2 * trials)) / denom
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * trials)) / trials) / denom
    return phat, max(0.0, center - margin), min(1.0, center + margin)


def two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> float:
    if min(n1, n2) == 0:
        return float("nan")
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    z = (p1 - p2) / se
    from scipy.stats import norm
    return 2 * (1 - norm.cdf(abs(z)))


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spot in range(1, 10):
        sub = df[df["lineup_spot"] == spot]
        n = len(sub)
        s = int(sub["overturned"].sum())
        rate, lo, hi = wilson_ci(s, n)
        rows.append({"spot": spot, "n": int(n), "overturned": s, "rate": rate, "wilson_low": lo, "wilson_high": hi})
    return pd.DataFrame(rows)


def _pvals_vs_spot3(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    s3 = summary.loc[summary["spot"] == 3].iloc[0]
    pvals = []
    for _, r in summary.iterrows():
        if r["spot"] == 3:
            pvals.append(np.nan)
            continue
        pvals.append(two_proportion_z(int(r["overturned"]), int(r["n"]), int(s3["overturned"]), int(s3["n"])))
    summary = summary.copy()
    summary["pval_vs_spot3"] = pvals
    finite = ~np.isnan(pvals)
    if finite.sum() >= 2:
        _, qvals_holm, *_ = multipletests(np.array(pvals)[finite], method="holm")
        _, qvals_bh, *_ = multipletests(np.array(pvals)[finite], method="fdr_bh")
        full_holm = np.full(len(pvals), np.nan)
        full_bh = np.full(len(pvals), np.nan)
        full_holm[finite] = qvals_holm
        full_bh[finite] = qvals_bh
        summary["qval_holm"] = full_holm
        summary["qval_bh"] = full_bh
    else:
        summary["qval_holm"] = np.nan
        summary["qval_bh"] = np.nan
    return summary


def _plot(summary: pd.DataFrame, title: str, out_path: Path, league_rate: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=160)
    spots = summary["spot"].astype(int)
    rates = summary["rate"]
    los = rates - summary["wilson_low"]
    his = summary["wilson_high"] - rates
    bar_color = ["#94a3b8"] * 9
    seven_idx = (spots == 7).idxmax() if (spots == 7).any() else None
    if seven_idx is not None and (spots == 7).any():
        idx_seven = list(spots).index(7)
        bar_color[idx_seven] = "#ef4444"
    bars = ax.bar(spots.astype(str), rates, color=bar_color, edgecolor="black", linewidth=0.6)
    ax.errorbar(spots.astype(str), rates, yerr=[los, his], fmt="none", color="black", capsize=3, linewidth=0.8)
    if league_rate is not None:
        ax.axhline(league_rate, color="#1d4ed8", linestyle="--", linewidth=1.0, label=f"League rate = {league_rate:.1%}")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_ylim(0, max(0.7, summary["wilson_high"].max() * 1.05))
    ax.set_ylabel("Overturn rate (Wilson 95% CI)")
    ax.set_xlabel("Lineup spot")
    ax.set_title(title)
    for sp, r, n in zip(spots, rates, summary["n"]):
        ax.text(int(sp) - 1, r + 0.012, f"n={int(n)}", ha="center", va="bottom", fontsize=8, color="#475569")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run() -> dict:
    challenges = data_prep.load_challenges()
    challenges = challenges.dropna(subset=["lineup_spot"]).copy()
    challenges["lineup_spot"] = challenges["lineup_spot"].astype(int)

    # ALL-challenges view (every challenge regardless of who challenged)
    all_summary = _summarize(challenges)
    all_summary = _pvals_vs_spot3(challenges, all_summary)
    league_all = challenges["overturned"].mean()
    _plot(all_summary, f"H1: ABS Overturn Rate by Lineup Spot — ALL challenges (n={len(challenges):,})", CHARTS / "h1_overturn_by_spot.png", league_rate=league_all)

    # BATTER-only view (the FanSided/Ringer comparison)
    batter = challenges[challenges["challenger"] == "batter"].copy()
    bat_summary = _summarize(batter)
    bat_summary = _pvals_vs_spot3(batter, bat_summary)
    league_batter = batter["overturned"].mean()
    _plot(bat_summary, f"H1 (batter-challenged): ABS Overturn Rate by Lineup Spot (n={len(batter):,})", CHARTS / "h1_overturn_batter_only.png", league_rate=league_batter)

    # Pinch-hitter robustness
    no_ph = challenges[~challenges["is_pinch_hitter"]].copy()
    no_ph_batter = no_ph[no_ph["challenger"] == "batter"].copy()
    no_ph_batter_summary = _summarize(no_ph_batter)
    no_ph_batter_summary = _pvals_vs_spot3(no_ph_batter, no_ph_batter_summary)
    _plot(no_ph_batter_summary, f"H1 (batter, NO pinch hitters): n={len(no_ph_batter):,}", CHARTS / "h1_overturn_batter_no_ph.png", league_rate=no_ph_batter["overturned"].mean())

    # Determine H1 verdict
    spot7_batter = bat_summary.loc[bat_summary["spot"] == 7].iloc[0]
    spot7_all = all_summary.loc[all_summary["spot"] == 7].iloc[0]
    h1_verdict = {
        "league_batter_rate": float(league_batter),
        "spot7_batter_rate": float(spot7_batter["rate"]),
        "spot7_batter_low": float(spot7_batter["wilson_low"]),
        "spot7_batter_high": float(spot7_batter["wilson_high"]),
        "spot7_batter_n": int(spot7_batter["n"]),
        "deficit_vs_league_batter_pp": float((league_batter - spot7_batter["rate"]) * 100),
        "h1_threshold_pp": 10.0,
        "h1_replicates_strict": float((league_batter - spot7_batter["rate"]) * 100) >= 10.0,
        # All challenges view
        "spot7_all_rate": float(spot7_all["rate"]),
        "league_all_rate": float(league_all),
        "deficit_all_view_pp": float((league_all - spot7_all["rate"]) * 100),
    }
    return {
        "all_summary": all_summary.to_dict(orient="records"),
        "batter_summary": bat_summary.to_dict(orient="records"),
        "no_ph_batter_summary": no_ph_batter_summary.to_dict(orient="records"),
        "h1_verdict": h1_verdict,
    }


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str))

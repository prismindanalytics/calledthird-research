"""h6_catcher_mechanism.py — Bayesian distribution comparison of catcher-initiated challenges.

Question: For challenges where `challenger == "catcher"`, do catchers pick
systematically different pitches in 7-hole at-bats than in 3-hole at-bats?

Outcome: edge_distance_in (in inches; absolute distance from rulebook zone edge).
The story: catchers in 7-hole AB might be challenging on tighter pitches (lower
edge_distance, harder fights) — which would produce higher overturn rates
because those tighter pitches more often were genuine misses.

Bayesian model:
  edge_distance_in ~ Gamma(alpha, beta)
  log(mu_i) = intercept + beta_spot[lineup_spot_i] + beta_count[count_state_i]
              + beta_in_zone * in_zone_i + (1|catcher_i) + (1|umpire_i)
  alpha    ~ HalfNormal(2)         # shape

We use a Gamma likelihood (positive-only, right-skewed) parameterized via the mean.

Posterior of interest: marginal mean(edge_distance_in | spot=7) - mean(edge_distance_in | spot=3).
A negative effect = catchers in 7-hole AB challenge on tighter pitches (closer
to the edge / more borderline).

Also report an overturn-rate comparison for context (overturned ~ batter_lineup_spot).
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=,floatX=float64")

import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

import data_prep_r2

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CHARTS = ROOT / "charts"
DIAG = CHARTS / "diagnostics"
CACHE.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)


def _prepare_catcher_subset() -> pd.DataFrame:
    df = data_prep_r2.load_challenges_full()
    df = df[df["challenger"] == "catcher"].copy()
    df = df.dropna(subset=["edge_distance_in_final", "lineup_spot", "count_state", "in_zone", "catcher_id", "umpire"])
    df["lineup_spot"] = df["lineup_spot"].astype(int)
    df = df[df["lineup_spot"].between(1, 9)]
    df["edge"] = df["edge_distance_in_final"].astype(float)
    df = df[df["edge"] > 0.001]   # gamma needs strictly positive
    df["count_state"] = df["count_state"].astype(str)
    df["catcher_id"] = df["catcher_id"].astype(int)
    df["umpire"] = df["umpire"].astype(str)
    df["in_zone"] = df["in_zone"].astype(int)
    df["overturned"] = df["overturned"].astype(int)
    return df.reset_index(drop=True)


def fit(df: pd.DataFrame, n_draws: int = 1500, n_tune: int = 1500):
    spot_levels = sorted(df["lineup_spot"].unique())
    spot_levels = [s for s in spot_levels if s != 3]
    spot_idx = np.asarray(pd.Categorical(df["lineup_spot"], categories=[3] + spot_levels).codes)

    count_levels = ["0-0"] + sorted(c for c in df["count_state"].unique() if c != "0-0")
    count_idx = np.asarray(pd.Categorical(df["count_state"], categories=count_levels).codes)

    catcher_idx, catcher_cats = pd.factorize(df["catcher_id"])
    umpire_idx, umpire_cats = pd.factorize(df["umpire"])

    coords = {
        "spot_levels": spot_levels,
        "count_levels": count_levels[1:],
        "catcher": list(catcher_cats),
        "umpire": list(umpire_cats),
    }
    y = df["edge"].values.astype(float)
    in_zone = df["in_zone"].values.astype(float)

    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", np.log(np.median(y)), 1.0)
        b_spot = pm.Normal("b_spot", 0, 0.5, dims="spot_levels")
        b_count = pm.Normal("b_count", 0, 0.5, dims="count_levels")
        b_zone = pm.Normal("b_zone", 0, 0.5)

        sd_catcher = pm.HalfNormal("sd_catcher", 0.3)
        sd_umpire = pm.HalfNormal("sd_umpire", 0.3)
        z_c = pm.Normal("z_c", 0, 1.0, dims="catcher")
        z_u = pm.Normal("z_u", 0, 1.0, dims="umpire")
        u_c = pm.Deterministic("u_c", sd_catcher * z_c, dims="catcher")
        u_u = pm.Deterministic("u_u", sd_umpire * z_u, dims="umpire")

        spot_eff = pm.math.concatenate([pt.zeros(1), b_spot])
        count_eff = pm.math.concatenate([pt.zeros(1), b_count])
        log_mu = (
            intercept
            + spot_eff[spot_idx]
            + count_eff[count_idx]
            + b_zone * in_zone
            + u_c[catcher_idx]
            + u_u[umpire_idx]
        )
        mu = pm.math.exp(log_mu)
        alpha = pm.HalfNormal("alpha", 2.0)
        # Gamma likelihood: mean=mu, shape=alpha => beta = alpha/mu
        beta = alpha / mu
        pm.Gamma("y", alpha=alpha, beta=beta, observed=y)

        idata = pm.sample(
            draws=n_draws, tune=n_tune, chains=4, cores=4,
            target_accept=0.95, progressbar=False, random_seed=2026,
            init="adapt_diag",
        )

    info = {
        "spot_levels": spot_levels,
        "count_levels": count_levels[1:],
        "spot_idx": spot_idx,
        "count_idx": count_idx,
        "catcher_idx": catcher_idx,
        "umpire_idx": umpire_idx,
    }
    return idata, info


def _posterior_spot7_vs_3_pp(idata, info, df: pd.DataFrame, n_keep: int = 1000) -> dict:
    """Posterior of (mean edge | spot=7) - (mean edge | spot=3), counterfactually evaluated
    over the empirical mix of count, in_zone, catcher, umpire in the catcher-challenge corpus."""
    posterior = idata.posterior
    rng = np.random.default_rng(19)
    flat = posterior.sizes["chain"] * posterior.sizes["draw"]
    n_keep = min(n_keep, flat)
    sample_ids = rng.choice(flat, n_keep, replace=False)

    intercept = posterior["intercept"].values.reshape(-1)[sample_ids]
    spot_levels = info["spot_levels"]
    b_spot = posterior["b_spot"].values.reshape(-1, len(spot_levels))[sample_ids]
    count_eff_full = posterior["b_count"].values.reshape(-1, len(info["count_levels"]))[sample_ids]
    b_zone = posterior["b_zone"].values.reshape(-1)[sample_ids]
    u_c = posterior["u_c"].values.reshape(-1, posterior.sizes["catcher"])[sample_ids]
    u_u = posterior["u_u"].values.reshape(-1, posterior.sizes["umpire"])[sample_ids]

    count_idx = info["count_idx"]
    catcher_idx = info["catcher_idx"]
    umpire_idx = info["umpire_idx"]
    in_zone = df["in_zone"].values.astype(float)

    count_eff = np.where(count_idx[None, :] == 0, 0.0, count_eff_full[:, np.maximum(count_idx - 1, 0)])
    base = (
        intercept[:, None]
        + count_eff
        + b_zone[:, None] * in_zone[None, :]
        + u_c[:, catcher_idx]
        + u_u[:, umpire_idx]
    )

    spot_to_idx = {3: 0}
    for i, s in enumerate(spot_levels, start=1):
        spot_to_idx[int(s)] = i

    if 7 in spot_to_idx and spot_to_idx[7] > 0:
        sp7 = b_spot[:, spot_to_idx[7] - 1][:, None]
    else:
        sp7 = np.zeros((b_zone.shape[0], 1))
    sp3 = np.zeros_like(sp7)

    mu_7 = np.exp(base + sp7).mean(axis=1)
    mu_3 = np.exp(base + sp3).mean(axis=1)
    diff = mu_7 - mu_3       # in inches

    # Also: bot-of-order vs top-of-order
    bot_spots = [s for s in (7, 8, 9) if s in spot_to_idx and spot_to_idx[s] > 0]
    top_spots = [s for s in (1, 2) if s in spot_to_idx and spot_to_idx[s] > 0]
    b_bot = np.mean([b_spot[:, spot_to_idx[s] - 1] for s in bot_spots], axis=0)[:, None] if bot_spots else np.zeros_like(sp7)
    b_top_partial = np.mean([b_spot[:, spot_to_idx[s] - 1] for s in top_spots], axis=0) if top_spots else np.zeros(b_zone.shape[0])
    b_top = (b_top_partial * len(top_spots) / 3.0)[:, None]
    mu_bot = np.exp(base + b_bot).mean(axis=1)
    mu_top = np.exp(base + b_top).mean(axis=1)
    diff_bot = mu_bot - mu_top

    return {
        "spot7_minus_spot3_in": {
            "median": float(np.median(diff)),
            "ci_low": float(np.percentile(diff, 2.5)),
            "ci_high": float(np.percentile(diff, 97.5)),
            "p_neg": float((diff < 0).mean()),
        },
        "bot_minus_top_in": {
            "median": float(np.median(diff_bot)),
            "ci_low": float(np.percentile(diff_bot, 2.5)),
            "ci_high": float(np.percentile(diff_bot, 97.5)),
            "p_neg": float((diff_bot < 0).mean()),
        },
    }


def overturn_by_spot(df: pd.DataFrame) -> pd.DataFrame:
    """Wilson CI on overturn rate per lineup spot (catcher-initiated only)."""
    from scipy.stats import norm
    z = norm.ppf(0.975)
    out = []
    for s, sub in df.groupby("lineup_spot"):
        n = len(sub)
        k = int(sub["overturned"].sum())
        p = k / n if n else 0.0
        if n == 0:
            lo, hi = 0.0, 0.0
        else:
            denom = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denom
            half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
            lo, hi = center - half, center + half
        out.append({"spot": int(s), "n": n, "overturn_rate": p, "wilson_low": float(lo), "wilson_high": float(hi)})
    return pd.DataFrame(out).sort_values("spot")


def plot(df: pd.DataFrame, post: dict, out_path: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    # Left: edge_distance_in by spot, mean + 95% bootstrap CI
    rng = np.random.default_rng(23)
    levels = sorted(df["lineup_spot"].unique())
    means = []
    los = []
    his = []
    ns = []
    for s in levels:
        e = df.loc[df["lineup_spot"] == s, "edge"].values
        ns.append(len(e))
        m = e.mean() if len(e) else np.nan
        means.append(m)
        if len(e) >= 5:
            boots = np.array([rng.choice(e, size=len(e), replace=True).mean() for _ in range(1000)])
            los.append(np.percentile(boots, 2.5))
            his.append(np.percentile(boots, 97.5))
        else:
            los.append(m); his.append(m)
    pos = np.arange(len(levels))
    axs[0].errorbar(pos, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                    fmt="o", color="#1f2937", ecolor="#94a3b8", elinewidth=2, capsize=3, markersize=6)
    for i, s in enumerate(levels):
        if int(s) == 7:
            axs[0].plot(pos[i], means[i], "o", color="#ef4444", markersize=10)
    axs[0].set_xticks(pos)
    axs[0].set_xticklabels([f"{s}\n(n={n})" for s, n in zip(levels, ns)])
    axs[0].set_xlabel("Batter lineup spot")
    axs[0].set_ylabel("Mean edge_distance (in)")
    axs[0].set_title("Catcher-initiated challenges:\nmean edge_distance by batter lineup spot")
    axs[0].spines[["top", "right"]].set_visible(False)

    # Right: posterior of spot-7 minus spot-3 effect
    s = post["spot7_minus_spot3_in"]
    bot = post["bot_minus_top_in"]
    labels = ["spot 7 - spot 3", "spots 7-9 - spots 1-3"]
    medians = [s["median"], bot["median"]]
    ci_lo = [s["ci_low"], bot["ci_low"]]
    ci_hi = [s["ci_high"], bot["ci_high"]]
    los = np.array(medians) - np.array(ci_lo)
    his = np.array(ci_hi) - np.array(medians)
    pos = np.arange(2)
    axs[1].errorbar(medians, pos, xerr=[los, his], fmt="o", color="#1f2937",
                    ecolor="#94a3b8", elinewidth=3, capsize=0, markersize=8)
    axs[1].axvline(0, color="#1d4ed8", linestyle="--", linewidth=1)
    axs[1].set_yticks(pos)
    axs[1].set_yticklabels(labels)
    axs[1].set_xlabel("Posterior effect on mean edge_distance (in)\nnegative = catchers challenge tighter pitches in this spot")
    axs[1].set_title("Bayesian posterior: edge-distance contrast")
    axs[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run() -> dict:
    df = _prepare_catcher_subset()
    print(f"[H6] catcher-initiated challenges with non-null edge_distance: n={len(df):,}")
    print(df.groupby("lineup_spot").size().rename("n"))

    print("[H6] fitting Bayesian Gamma model on edge_distance ...")
    idata, info = fit(df)
    summary = az.summary(idata, var_names=["intercept", "b_spot", "b_count", "b_zone", "alpha", "sd_catcher", "sd_umpire"], hdi_prob=0.95)
    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    print(f"  R-hat max = {rhat_max:.3f}, ESS bulk min = {ess_min:.0f}")

    post = _posterior_spot7_vs_3_pp(idata, info, df)
    print(f"  posterior spot7-spot3 effect on edge: {post['spot7_minus_spot3_in']}")
    print(f"  posterior bot-top effect on edge: {post['bot_minus_top_in']}")

    overturn = overturn_by_spot(df)
    overturn.to_csv(ROOT / "h6_overturn_by_spot.csv", index=False)

    plot(df, post, CHARTS / "h6_catcher_mechanism.png")

    az.plot_trace(idata, var_names=["intercept", "b_spot", "alpha", "sd_catcher", "sd_umpire"], compact=True)
    plt.tight_layout()
    plt.savefig(DIAG / "h6_trace.png", dpi=110)
    plt.close()
    summary.to_csv(DIAG / "h6_summary.csv")

    p7 = post["spot7_minus_spot3_in"]
    interpretation = (
        "Catchers pick tighter pitches against 7-hole batters than 3-hole batters."
        if p7["ci_high"] < 0
        else "Catchers pick wider/easier pitches against 7-hole batters than 3-hole batters."
        if p7["ci_low"] > 0
        else "No credible difference in catcher pitch selection between 7-hole and 3-hole at-bats."
    )
    return {
        "n_catcher_challenges": int(len(df)),
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "spot7_vs_spot3_edge_in_med": p7["median"],
        "spot7_vs_spot3_edge_in_lo": p7["ci_low"],
        "spot7_vs_spot3_edge_in_hi": p7["ci_high"],
        "p_neg": p7["p_neg"],
        "bot_vs_top_edge_in_med": post["bot_minus_top_in"]["median"],
        "bot_vs_top_edge_in_lo": post["bot_minus_top_in"]["ci_low"],
        "bot_vs_top_edge_in_hi": post["bot_minus_top_in"]["ci_high"],
        "overturn_by_spot_csv": str((ROOT / "h6_overturn_by_spot.csv").resolve()),
        "interpretation": interpretation,
    }


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str))

"""bayes_glm_h2.py — Hierarchical Bayesian logistic GLM for H2.

Model:
  overturned ~ lineup_spot (categorical, baseline=spot 3)
             + edge_distance_in (centered)
             + count_state (categorical)
             + (1 | pitcher) + (1 | catcher) + (1 | umpire)
             + fame_quartile + framing_tier   (fixed-effect controls in case
                                                random effects can't fully soak
                                                up pitcher/catcher quality)

Priors (pre-registered, weakly informative):
  intercept              ~ Normal(0, 1.5)
  spot effects (8)       ~ Normal(0, 1.0)
  edge_distance_in       ~ Normal(0, 1.0)        # log-OR per inch
  count_state effects    ~ Normal(0, 1.0)        # vs '0-0' baseline
  fame_quartile          ~ Normal(0, 0.5)
  framing_tier           ~ Normal(0, 0.5)
  group sd (pitcher/catcher/umpire)  ~ HalfNormal(0.5)

Sampling: 2 chains, 1000 tune + 1000 draws each, target_accept=0.92.

Outputs:
  charts/h2_lineup_effect_forest.png
  charts/diagnostics/h2_trace.png, h2_energy.png, h2_summary.csv
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

import data_prep

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"
DIAG = CHARTS / "diagnostics"
CHARTS.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing critical fields; engineer feature columns."""
    df = df.dropna(subset=["lineup_spot", "edge_distance_in_final", "balls", "strikes"]).copy()
    df["lineup_spot"] = df["lineup_spot"].astype(int)
    df = df[df["lineup_spot"].between(1, 9)]
    df["count_state"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    df["edge_dist_in_c"] = df["edge_distance_in_final"] - df["edge_distance_in_final"].mean()
    df["fame_quartile"] = df["fame_quartile"].fillna(2.5)  # baseline-ish
    df["framing_tier"] = df["framing_tier"].fillna("unknown").astype(str)
    df["pitcher_id"] = df["pitcher_id"].astype(int)
    df["catcher_id"] = df["catcher_id"].astype(int)
    df["umpire"] = df["umpire"].fillna("unknown").astype(str)
    return df


def _build_model(df: pd.DataFrame) -> tuple[pm.Model, dict]:
    spot_vals = sorted(df["lineup_spot"].unique())
    # baseline = spot 3
    spot_levels = [s for s in spot_vals if s != 3]
    spot_idx = np.asarray(pd.Categorical(df["lineup_spot"], categories=[3] + spot_levels).codes)
    count_levels = ["0-0"] + sorted(c for c in df["count_state"].unique() if c != "0-0")
    count_idx = np.asarray(pd.Categorical(df["count_state"], categories=count_levels).codes)
    framing_levels = ["mid"] + sorted(t for t in df["framing_tier"].unique() if t != "mid")
    framing_idx = np.asarray(pd.Categorical(df["framing_tier"], categories=framing_levels).codes)

    pitcher_idx, pitcher_cats = pd.factorize(df["pitcher_id"])
    catcher_idx, catcher_cats = pd.factorize(df["catcher_id"])
    umpire_idx, umpire_cats = pd.factorize(df["umpire"])

    edge = df["edge_dist_in_c"].values.astype(float)
    fame = df["fame_quartile"].values.astype(float) - 2.5  # center

    y = df["overturned"].astype(int).values

    coords = {
        "spot_levels": spot_levels,
        "count_levels": count_levels[1:],         # excluding baseline
        "framing_levels": framing_levels[1:],
        "pitcher": list(pitcher_cats),
        "catcher": list(catcher_cats),
        "umpire": list(umpire_cats),
    }
    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", 0, 1.5)
        b_spot = pm.Normal("b_spot", 0, 1.0, dims="spot_levels")
        b_count = pm.Normal("b_count", 0, 1.0, dims="count_levels")
        b_framing = pm.Normal("b_framing", 0, 0.5, dims="framing_levels")
        b_edge = pm.Normal("b_edge_in", 0, 1.0)
        b_fame = pm.Normal("b_fame", 0, 0.5)
        # Non-centered parameterization: this is essential for thin-data random effects
        sd_pitcher = pm.HalfNormal("sd_pitcher", 0.5)
        sd_catcher = pm.HalfNormal("sd_catcher", 0.5)
        sd_umpire = pm.HalfNormal("sd_umpire", 0.5)
        z_pitcher = pm.Normal("z_pitcher", 0, 1.0, dims="pitcher")
        z_catcher = pm.Normal("z_catcher", 0, 1.0, dims="catcher")
        z_umpire = pm.Normal("z_umpire", 0, 1.0, dims="umpire")
        u_pitcher = pm.Deterministic("u_pitcher", sd_pitcher * z_pitcher, dims="pitcher")
        u_catcher = pm.Deterministic("u_catcher", sd_catcher * z_catcher, dims="catcher")
        u_umpire = pm.Deterministic("u_umpire", sd_umpire * z_umpire, dims="umpire")

        # Build fixed-effect contributions
        spot_eff = pm.math.concatenate([pt.zeros(1), b_spot])  # baseline (spot 3) is 0
        count_eff = pm.math.concatenate([pt.zeros(1), b_count])
        framing_eff = pm.math.concatenate([pt.zeros(1), b_framing])
        eta = (
            intercept
            + spot_eff[spot_idx]
            + count_eff[count_idx]
            + framing_eff[framing_idx]
            + b_edge * edge
            + b_fame * fame
            + u_pitcher[pitcher_idx]
            + u_catcher[catcher_idx]
            + u_umpire[umpire_idx]
        )
        pm.Bernoulli("y", logit_p=eta, observed=y)

    info = {
        "spot_levels": spot_levels,
        "count_levels": count_levels[1:],
        "framing_levels": framing_levels[1:],
        "n_pitcher": len(pitcher_cats),
        "n_catcher": len(catcher_cats),
        "n_umpire": len(umpire_cats),
        "n_obs": len(df),
        "edge_mean_in": float(df["edge_distance_in_final"].mean()),
    }
    return model, info


def run(challenger_filter: str | None = "batter", drop_pinch: bool = False, n_draws: int = 1500, n_tune: int = 2000) -> dict:
    df_full = data_prep.load_challenges()
    df = df_full.copy()
    if challenger_filter is not None:
        df = df[df["challenger"] == challenger_filter]
    if drop_pinch:
        df = df[~df["is_pinch_hitter"]]
    df = _prepare(df)
    print(f"H2 model on n={len(df):,} challenges (challenger={challenger_filter}, drop_pinch={drop_pinch})")

    model, info = _build_model(df)
    with model:
        idata = pm.sample(
            draws=n_draws, tune=n_tune, chains=4, cores=4,
            target_accept=0.95, progressbar=False, random_seed=421,
        )
    summary = az.summary(idata, hdi_prob=0.95)
    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    print(f"  R-hat max = {rhat_max:.3f}, ESS bulk min = {ess_min:.0f}")

    # Posterior of b_spot dimensions in pp scale (approximate via marginal effect at the empirical mean of all other predictors)
    # We compute predicted probability gap directly: pred(spot=k) - pred(spot=3) given empirical covariate distribution.
    posterior = idata.posterior
    spot_levels = info["spot_levels"]

    # Build a counterfactual prediction: hold all other covariates at empirical values; only swap spot.
    # We'll use a Monte Carlo style: sample (chain, draw) of all parameters, compute mean prob over all rows.
    # For speed, take 200 random draws from the posterior.
    rng = np.random.default_rng(7)
    chain_dim = posterior.dims["chain"]
    draw_dim = posterior.dims["draw"]
    flat = chain_dim * draw_dim
    n_keep = min(400, flat)
    sample_ids = rng.choice(flat, n_keep, replace=False)

    intercept = posterior["intercept"].values.reshape(-1)[sample_ids]
    b_spot = posterior["b_spot"].values.reshape(-1, len(spot_levels))[sample_ids]
    b_count = posterior["b_count"].values.reshape(-1, len(info["count_levels"]))[sample_ids]
    b_framing = posterior["b_framing"].values.reshape(-1, len(info["framing_levels"]))[sample_ids]
    b_edge = posterior["b_edge_in"].values.reshape(-1)[sample_ids]
    b_fame = posterior["b_fame"].values.reshape(-1)[sample_ids]
    u_pitcher = posterior["u_pitcher"].values.reshape(-1, info["n_pitcher"])[sample_ids]
    u_catcher = posterior["u_catcher"].values.reshape(-1, info["n_catcher"])[sample_ids]
    u_umpire = posterior["u_umpire"].values.reshape(-1, info["n_umpire"])[sample_ids]

    # Recover per-row indices
    df = df.reset_index(drop=True)
    count_idx = pd.Categorical(df["count_state"], categories=["0-0"] + info["count_levels"]).codes
    framing_idx = pd.Categorical(df["framing_tier"], categories=["mid"] + info["framing_levels"]).codes
    pitcher_idx = pd.Categorical(df["pitcher_id"]).codes
    catcher_idx = pd.Categorical(df["catcher_id"]).codes
    umpire_idx = pd.Categorical(df["umpire"]).codes
    edge = df["edge_dist_in_c"].values.astype(float)
    fame = df["fame_quartile"].values.astype(float) - 2.5

    # Compute expected probability gap (spot=k - spot=3) averaged over rows
    base_logits = (
        intercept[:, None]
        + np.where(count_idx == 0, 0.0, b_count[:, np.maximum(count_idx - 1, 0)])
        + np.where(framing_idx == 0, 0.0, b_framing[:, np.maximum(framing_idx - 1, 0)])
        + b_edge[:, None] * edge[None, :]
        + b_fame[:, None] * fame[None, :]
        + u_pitcher[:, pitcher_idx]
        + u_catcher[:, catcher_idx]
        + u_umpire[:, umpire_idx]
    )
    p_spot3 = 1 / (1 + np.exp(-base_logits)).mean(axis=1)
    spot_gaps_pp = []
    for k_idx, k in enumerate(spot_levels):
        p_k = 1 / (1 + np.exp(-(base_logits + b_spot[:, k_idx, None]))).mean(axis=1)
        spot_gaps_pp.append((p_k - p_spot3) * 100)
    gaps = np.array(spot_gaps_pp)  # (len(spot_levels), n_keep)

    # Forest plot of marginal pp differences
    forest = []
    for i, k in enumerate(spot_levels):
        med = float(np.median(gaps[i]))
        lo = float(np.percentile(gaps[i], 2.5))
        hi = float(np.percentile(gaps[i], 97.5))
        prob_neg = float((gaps[i] < 0).mean())
        forest.append({"spot": int(k), "median_pp": med, "ci_low_pp": lo, "ci_high_pp": hi, "P(effect<0)": prob_neg})
    forest_df = pd.DataFrame(forest).sort_values("spot")
    print("Marginal lineup-spot effects vs spot 3 (pp diff in overturn prob, after controls):")
    print(forest_df.round(2))

    # Save forest plot
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    spot_pos = list(range(len(spot_levels)))
    medians = forest_df["median_pp"].values
    los = medians - forest_df["ci_low_pp"].values
    his = forest_df["ci_high_pp"].values - medians
    colors = ["#94a3b8"] * len(spot_levels)
    if 7 in forest_df["spot"].values:
        idx7 = list(forest_df["spot"]).index(7)
        colors[idx7] = "#ef4444"
    ax.errorbar(medians, spot_pos, xerr=[los, his], fmt="o", color="#1f2937", ecolor=colors, elinewidth=4, capsize=0, markersize=6)
    for i, c in enumerate(colors):
        if c == "#ef4444":
            ax.plot(medians[i], spot_pos[i], "o", markersize=8, color=c)
    ax.axvline(0, color="#1d4ed8", linestyle="--", linewidth=1)
    ax.set_yticks(spot_pos)
    ax.set_yticklabels([f"spot {int(s)}" for s in forest_df["spot"]])
    ax.set_xlabel("Marginal effect on overturn rate vs spot 3 (pp), controlled")
    ax.set_title(f"H2: Lineup-spot marginal effect after controls (n={len(df):,}, batter-only)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS / "h2_lineup_effect_forest.png")
    plt.close(fig)

    # Diagnostics: trace + energy plots, summary CSV
    az.plot_trace(idata, var_names=["intercept", "b_spot", "b_edge_in", "b_fame", "sd_pitcher", "sd_catcher", "sd_umpire"], compact=True)
    plt.tight_layout()
    plt.savefig(DIAG / "h2_trace.png", dpi=120)
    plt.close()
    az.plot_energy(idata)
    plt.tight_layout()
    plt.savefig(DIAG / "h2_energy.png", dpi=120)
    plt.close()
    summary.to_csv(DIAG / "h2_summary.csv")

    # Pull spot 7 specifically
    spot7_row = forest_df.loc[forest_df["spot"] == 7].iloc[0] if 7 in forest_df["spot"].values else None
    out = {
        "n_obs": int(len(df)),
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "forest": forest_df.to_dict(orient="records"),
        "spot7_vs_spot3": {
            "median_pp": float(spot7_row["median_pp"]) if spot7_row is not None else None,
            "ci_low_pp": float(spot7_row["ci_low_pp"]) if spot7_row is not None else None,
            "ci_high_pp": float(spot7_row["ci_high_pp"]) if spot7_row is not None else None,
            "P_negative": float(spot7_row["P(effect<0)"]) if spot7_row is not None else None,
        },
        "model_info": info,
        "challenger_filter": challenger_filter,
        "drop_pinch": drop_pinch,
    }
    return out


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str))

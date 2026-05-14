"""bayes_gam_h3.py — Hierarchical Bayesian model for H3 (borderline called-pitch bias).

Restriction: TAKEN pitches with |edge_distance_ft| <= 0.3 ft.

Model:
  is_called_strike ~ lineup_spot
                   + s(plate_x, plate_z)        # 2D smooth
                   + count_state
                   + framing_tier + fame_quartile
                   + (1|pitcher) + (1|catcher) + (1|umpire)

Strategy: a fully-Bayesian 2D BSpline tensor-product is heavy on this data
volume. Pre-process by fitting a *fixed* 2D BSpline basis (k=8 in each axis,
order=3), use that as design matrix columns, then run a Bayesian logistic
regression with weakly-informative priors on basis coefficients (penalised by
a Normal(0, sigma_smooth) prior on the basis, with sigma_smooth ~ HalfNormal).
This gives a hierarchical-flavour GAM that is well-identified given the
sample size (~28k borderline pitches).

Pre-registered priors:
  intercept                  ~ Normal(0, 1.5)
  lineup-spot effects (8)    ~ Normal(0, 0.5)
  count effects (11)         ~ Normal(0, 0.5)
  framing-tier effects (3)   ~ Normal(0, 0.3)
  fame-quartile slope        ~ Normal(0, 0.3)
  basis coefficients         ~ Normal(0, sigma_basis)
  sigma_basis                ~ HalfNormal(1.0)
  sd_pitcher/catcher/umpire  ~ HalfNormal(0.5)

Sampling: 4 chains, 1500 tune + 1000 draws, target_accept=0.92.

Outputs:
  charts/h3_called_strike_rate_borderline.png
  charts/diagnostics/h3_trace.png, h3_summary.csv
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
from scipy.interpolate import BSpline

import data_prep

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "charts"
DIAG = CHARTS / "diagnostics"
CHARTS.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)


def _bspline_basis(x: np.ndarray, n_knots: int = 8, order: int = 3) -> np.ndarray:
    """Return (n_obs, n_basis) BSpline basis evaluated at x. Knots are quantile-based."""
    quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]
    interior = np.quantile(x, quantiles)
    boundary_lo = np.repeat(x.min() - 1e-6, order + 1)
    boundary_hi = np.repeat(x.max() + 1e-6, order + 1)
    knots = np.concatenate([boundary_lo, interior, boundary_hi])
    n_basis = len(knots) - order - 1
    out = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        out[:, i] = BSpline(knots, c, order)(x)
    return out


def _prepare_borderline(df: pd.DataFrame, edge_cutoff_ft: float = 0.3) -> pd.DataFrame:
    """Restrict to borderline taken pitches and drop NA."""
    df = df.dropna(subset=["plate_x", "plate_z", "balls", "strikes", "lineup_spot", "umpire"]).copy()
    df = df[df["edge_distance_ft"].abs() <= edge_cutoff_ft]
    df["lineup_spot"] = df["lineup_spot"].astype(int)
    df = df[df["lineup_spot"].between(1, 9)]
    df["count_state"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    df["framing_tier"] = df["framing_tier"].fillna("unknown").astype(str)
    df["fame_quartile"] = df["fame_quartile"].fillna(2.5)
    df["umpire"] = df["umpire"].astype(str)
    return df


def _build_model(df: pd.DataFrame, n_knots: int = 6) -> tuple[pm.Model, dict]:
    """Build hierarchical Bayesian logistic model with a fixed 2D smooth.

    To keep convergence tractable on this borderline-zone scale, we use a small
    2D B-spline tensor-product basis (default 4×4=16 columns), centered &
    standardized, with a per-column Normal(0, sigma_basis) prior. sigma_basis is
    HalfNormal-bounded to avoid runaway. This is a simpler GAM than full pygam
    but it captures the spatial trend (which is exactly what we need to control
    for, not interpret).
    """
    spot_vals = sorted(df["lineup_spot"].unique())
    spot_levels = [s for s in spot_vals if s != 3]
    spot_idx = np.asarray(pd.Categorical(df["lineup_spot"], categories=[3] + spot_levels).codes)
    count_levels = ["0-0"] + sorted(c for c in df["count_state"].unique() if c != "0-0")
    count_idx = np.asarray(pd.Categorical(df["count_state"], categories=count_levels).codes)
    framing_levels = ["mid"] + sorted(t for t in df["framing_tier"].unique() if t != "mid")
    framing_idx = np.asarray(pd.Categorical(df["framing_tier"], categories=framing_levels).codes)

    pitcher_idx, pitcher_cats = pd.factorize(df["pitcher"])
    catcher_idx, catcher_cats = pd.factorize(df["fielder_2"])
    umpire_idx, umpire_cats = pd.factorize(df["umpire"])

    # 2D BSpline tensor-product basis on (plate_x, plate_z) — lean: 4 knots each
    n_basis_per_axis = 4
    Bx = _bspline_basis(df["plate_x"].values.astype(float), n_knots=n_basis_per_axis, order=3)
    Bz = _bspline_basis(df["plate_z"].values.astype(float), n_knots=n_basis_per_axis, order=3)
    n_obs = len(df)
    Kx, Kz = Bx.shape[1], Bz.shape[1]
    B = np.einsum("ni,nj->nij", Bx, Bz).reshape(n_obs, Kx * Kz)
    col_var = B.var(axis=0)
    B = B[:, col_var > 1e-10]
    # Center but DO NOT scale (preserves smoothness penalty interpretation)
    B = B - B.mean(axis=0)

    y = df["called_strike"].astype(int).values

    coords = {
        "spot_levels": spot_levels,
        "count_levels": count_levels[1:],
        "framing_levels": framing_levels[1:],
        "basis": list(range(B.shape[1])),
        "pitcher": list(pitcher_cats),
        "catcher": list(catcher_cats),
        "umpire": list(umpire_cats),
    }
    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", 0, 1.5)
        b_spot = pm.Normal("b_spot", 0, 0.3, dims="spot_levels")
        b_count = pm.Normal("b_count", 0, 0.5, dims="count_levels")
        b_framing = pm.Normal("b_framing", 0, 0.3, dims="framing_levels")
        # Fixed-scale prior on basis coefficients to avoid sigma_basis funnel
        b_basis = pm.Normal("b_basis", 0, 2.0, dims="basis")
        sd_pitcher = pm.HalfNormal("sd_pitcher", 0.5)
        sd_catcher = pm.HalfNormal("sd_catcher", 0.5)
        sd_umpire = pm.HalfNormal("sd_umpire", 0.5)
        z_pitcher = pm.Normal("z_pitcher", 0, 1.0, dims="pitcher")
        z_catcher = pm.Normal("z_catcher", 0, 1.0, dims="catcher")
        z_umpire = pm.Normal("z_umpire", 0, 1.0, dims="umpire")
        u_pitcher = pm.Deterministic("u_pitcher", sd_pitcher * z_pitcher, dims="pitcher")
        u_catcher = pm.Deterministic("u_catcher", sd_catcher * z_catcher, dims="catcher")
        u_umpire = pm.Deterministic("u_umpire", sd_umpire * z_umpire, dims="umpire")

        spot_eff = pm.math.concatenate([pt.zeros(1), b_spot])
        count_eff = pm.math.concatenate([pt.zeros(1), b_count])
        framing_eff = pm.math.concatenate([pt.zeros(1), b_framing])
        eta = (
            intercept
            + spot_eff[spot_idx]
            + count_eff[count_idx]
            + framing_eff[framing_idx]
            + pm.math.dot(B, b_basis)
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
        "n_obs": n_obs,
        "n_basis": B.shape[1],
        "B": B,
        "spot_idx": spot_idx,
        "count_idx": count_idx,
        "framing_idx": framing_idx,
        "pitcher_idx": pitcher_idx,
        "catcher_idx": catcher_idx,
        "umpire_idx": umpire_idx,
    }
    return model, info


def _marginal_pp_diff(idata, info, n_keep: int = 250) -> dict:
    """Compute marginal pp difference vs spot 3 for each lineup spot, averaged over the empirical sample."""
    posterior = idata.posterior
    rng = np.random.default_rng(11)
    flat = posterior.sizes["chain"] * posterior.sizes["draw"]
    n_keep = min(n_keep, flat)
    sample_ids = rng.choice(flat, n_keep, replace=False)

    spot_levels = info["spot_levels"]
    intercept = posterior["intercept"].values.reshape(-1)[sample_ids]
    b_spot = posterior["b_spot"].values.reshape(-1, len(spot_levels))[sample_ids]
    b_count = posterior["b_count"].values.reshape(-1, len(info["count_levels"]))[sample_ids]
    b_framing = posterior["b_framing"].values.reshape(-1, len(info["framing_levels"]))[sample_ids]
    b_basis = posterior["b_basis"].values.reshape(-1, info["n_basis"])[sample_ids]
    u_pitcher = posterior["u_pitcher"].values.reshape(-1, info["n_pitcher"])[sample_ids]
    u_catcher = posterior["u_catcher"].values.reshape(-1, info["n_catcher"])[sample_ids]
    u_umpire = posterior["u_umpire"].values.reshape(-1, info["n_umpire"])[sample_ids]

    B = info["B"]                         # (n_obs, n_basis)
    spot_idx = info["spot_idx"]           # 0=baseline (spot 3)
    count_idx = info["count_idx"]
    framing_idx = info["framing_idx"]
    pitcher_idx = info["pitcher_idx"]
    catcher_idx = info["catcher_idx"]
    umpire_idx = info["umpire_idx"]

    # base (spot=3) logits per row
    base = (
        intercept[:, None]                                   # (S, 1)
        + np.where(count_idx == 0, 0.0, b_count[:, np.maximum(count_idx - 1, 0)])
        + np.where(framing_idx == 0, 0.0, b_framing[:, np.maximum(framing_idx - 1, 0)])
        + b_basis @ B.T                                      # (S, n_obs)
        + u_pitcher[:, pitcher_idx]
        + u_catcher[:, catcher_idx]
        + u_umpire[:, umpire_idx]
    )
    p_spot3 = (1.0 / (1.0 + np.exp(-base))).mean(axis=1)
    forest = []
    for i, k in enumerate(spot_levels):
        p_k = (1.0 / (1.0 + np.exp(-(base + b_spot[:, i, None])))).mean(axis=1)
        diffs = (p_k - p_spot3) * 100.0
        forest.append({
            "spot": int(k),
            "median_pp": float(np.median(diffs)),
            "ci_low_pp": float(np.percentile(diffs, 2.5)),
            "ci_high_pp": float(np.percentile(diffs, 97.5)),
            "P(effect>0)": float((diffs > 0).mean()),
            "P(effect<0)": float((diffs < 0).mean()),
        })
    return forest


def run_main(drop_pinch: bool = False, n_draws: int = 1000, n_tune: int = 1500, edge_cutoff_ft: float = 0.3) -> dict:
    df_full = data_prep.load_taken_pitches()
    df = df_full.copy()
    if drop_pinch:
        df = df[~df["is_pinch_hitter"]]
    df = _prepare_borderline(df, edge_cutoff_ft=edge_cutoff_ft)
    print(f"H3 model on n={len(df):,} borderline taken pitches (drop_pinch={drop_pinch}, cutoff={edge_cutoff_ft}ft)")

    model, info = _build_model(df, n_knots=5)
    with model:
        idata = pm.sample(
            draws=n_draws, tune=n_tune, chains=4, cores=4,
            target_accept=0.95, progressbar=False, random_seed=2026,
            init="adapt_diag",
        )
    summary = az.summary(idata, var_names=["intercept", "b_spot", "b_count", "b_framing", "sd_pitcher", "sd_catcher", "sd_umpire"], hdi_prob=0.95)
    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    print(f"  R-hat max = {rhat_max:.3f}, ESS bulk min = {ess_min:.0f}")

    forest = _marginal_pp_diff(idata, info)
    forest_df = pd.DataFrame(forest)
    print("Marginal lineup-spot effects vs spot 3 on borderline called-strike prob (pp):")
    print(forest_df.round(2))

    # Plot: forest + raw rate
    raw = df.groupby("lineup_spot")["called_strike"].agg(["mean", "size"]).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), dpi=160)
    # Left: raw called-strike rate
    spots = raw["lineup_spot"].astype(int)
    rates = raw["mean"]
    axs[0].bar(spots.astype(str), rates, color=["#94a3b8" if s != 7 else "#ef4444" for s in spots], edgecolor="black", linewidth=0.6)
    for s, r, n in zip(spots, rates, raw["size"]):
        axs[0].text(s - 1, r + 0.005, f"n={n:,}", ha="center", va="bottom", fontsize=8, color="#475569")
    axs[0].set_ylim(0.45, 0.65)
    axs[0].set_xlabel("Lineup spot")
    axs[0].set_ylabel("Raw called-strike rate (borderline takes)")
    axs[0].set_title(f"Borderline called-strike rate, raw\n|edge|<={edge_cutoff_ft}ft (n={len(df):,})")
    axs[0].spines[["top", "right"]].set_visible(False)
    axs[0].grid(axis="y", linestyle=":", alpha=0.5)
    # Right: forest
    pos = list(range(len(forest_df)))
    medians = forest_df["median_pp"].values
    los = medians - forest_df["ci_low_pp"].values
    his = forest_df["ci_high_pp"].values - medians
    colors = ["#94a3b8"] * len(forest_df)
    if 7 in forest_df["spot"].values:
        idx7 = list(forest_df["spot"]).index(7)
        colors[idx7] = "#ef4444"
    axs[1].errorbar(medians, pos, xerr=[los, his], fmt="o", color="#1f2937", ecolor=colors, elinewidth=4, capsize=0, markersize=6)
    for i, c in enumerate(colors):
        if c == "#ef4444":
            axs[1].plot(medians[i], pos[i], "o", markersize=8, color=c)
    axs[1].axvline(0, color="#1d4ed8", linestyle="--", linewidth=1)
    axs[1].set_yticks(pos)
    axs[1].set_yticklabels([f"spot {int(s)}" for s in forest_df["spot"]])
    axs[1].set_xlabel("Marginal effect on called-strike rate vs spot 3 (pp)")
    axs[1].set_title(f"H3 controlled effect (R̂={rhat_max:.2f}, ESS={ess_min:.0f})")
    axs[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS / "h3_called_strike_rate_borderline.png")
    plt.close(fig)

    # Diagnostics
    az.plot_trace(idata, var_names=["intercept", "b_spot", "sd_pitcher", "sd_catcher", "sd_umpire"], compact=True)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(DIAG / "h3_trace.png", dpi=120)
    plt.close()
    summary.to_csv(DIAG / "h3_summary.csv")

    spot7 = next((r for r in forest if r["spot"] == 7), None)
    return {
        "n_obs": len(df),
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "edge_cutoff_ft": edge_cutoff_ft,
        "drop_pinch": drop_pinch,
        "forest": forest,
        "spot7_vs_spot3": spot7,
        "raw_rates": raw.to_dict(orient="records"),
    }


def run_stratified(strata_col: str, edge_cutoff_ft: float = 0.3, n_draws: int = 800, n_tune: int = 1200) -> dict:
    """Stratify the H3 fit by a column (handedness, count_quadrant, pitch_group). Lighter sampler."""
    df_full = data_prep.load_taken_pitches()
    out = {}
    for level, sub in df_full.groupby(strata_col):
        sub = _prepare_borderline(sub, edge_cutoff_ft=edge_cutoff_ft)
        if len(sub) < 1000 or sub["lineup_spot"].nunique() < 9:
            print(f"Skipping {strata_col}={level}: n={len(sub)}, spots={sub['lineup_spot'].nunique()}")
            continue
        print(f"  Fitting {strata_col}={level}: n={len(sub):,}")
        model, info = _build_model(sub, n_knots=6)
        with model:
            idata = pm.sample(
                draws=n_draws, tune=n_tune, chains=2, cores=2,
                target_accept=0.92, progressbar=False, random_seed=hash(level) % 10_000,
            )
        forest = _marginal_pp_diff(idata, info, n_keep=200)
        out[str(level)] = {
            "n": len(sub),
            "forest": forest,
            "spot7": next((r for r in forest if r["spot"] == 7), None),
        }
    return out


if __name__ == "__main__":
    out = run_main()
    print(json.dumps(out, indent=2, default=str))

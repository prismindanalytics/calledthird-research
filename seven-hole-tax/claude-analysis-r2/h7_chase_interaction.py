"""h7_chase_interaction.py — lineup_spot x chase_rate_tertile interaction GAM.

Adds a `lineup_spot x chase_rate_tertile` interaction term to Round 1's H3 GAM.
Tests the FanSided "elite-pitch-recognition" mechanism: are *low-chase-rate*
(disciplined) hitters in spot 7 the ones losing borderline calls relative to
3-hole hitters with the same discipline level?

Model:
  is_called_strike ~ lineup_spot + chase_tertile + lineup_spot:chase_tertile
                   + s(plate_x, plate_z)
                   + count_state + framing_tier
                   + (1|pitcher) + (1|catcher) + (1|umpire)

Restricted to borderline (|edge|<=0.3) pitches whose batter has a 2025 chase
rate (>=200 PA in 2025 sample). Tertiles are computed within the qualifying
batter set.

Outputs:
  Per-tertile marginal pp effect (spot 7 vs spot 3)
  Interaction p-value (posterior tail probability of the interaction term being negative)
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=,floatX=float64")

import json
import pickle
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.interpolate import BSpline

import data_prep_r2

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CHARTS = ROOT / "charts"
DIAG = CHARTS / "diagnostics"
CACHE.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)

H7_CACHE = CACHE / "h7_idata.pkl"


def _bspline_basis(x: np.ndarray, n_knots: int = 4, order: int = 3) -> np.ndarray:
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


def _prepare(df: pd.DataFrame, chase: pd.DataFrame) -> pd.DataFrame:
    """Join chase rate and assign tertiles. Drops batters without 2025 history (>=200 PA)."""
    df = df.copy()
    chase_q = chase[chase["qualified_200pa"]].copy()
    # Tertiles within the qualifying batter set
    cuts = chase_q["chase_rate"].quantile([1.0/3.0, 2.0/3.0]).values
    print(f"  chase tertile cutpoints (qualified batters): {cuts[0]:.3f}, {cuts[1]:.3f}")
    chase_q["chase_tertile"] = pd.cut(
        chase_q["chase_rate"], bins=[-np.inf, cuts[0], cuts[1], np.inf],
        labels=["low", "mid", "high"], include_lowest=True,
    ).astype(str)
    df = df.merge(chase_q[["batter", "chase_rate", "chase_tertile"]], on="batter", how="inner")
    # Drop rows whose chase_tertile is missing (shouldn't happen after inner merge)
    df = df[df["chase_tertile"].isin(["low", "mid", "high"])].copy()
    df["lineup_spot"] = df["lineup_spot"].astype(int)
    df = df[df["lineup_spot"].between(1, 9)]
    df["umpire"] = df["umpire"].astype(str)
    df["framing_tier"] = df["framing_tier"].fillna("unknown").astype(str)
    df["count_state"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    df = df.reset_index(drop=True)
    return df


def fit(df: pd.DataFrame, n_draws: int = 1000, n_tune: int = 1500, force: bool = False):
    if H7_CACHE.exists() and not force:
        with open(H7_CACHE, "rb") as f:
            blob = pickle.load(f)
        return blob["idata"], blob["info"], blob["df"]

    spot_levels = sorted(df["lineup_spot"].unique())
    spot_levels = [s for s in spot_levels if s != 3]
    spot_idx = np.asarray(pd.Categorical(df["lineup_spot"], categories=[3] + spot_levels).codes)
    count_levels = ["0-0"] + sorted(c for c in df["count_state"].unique() if c != "0-0")
    count_idx = np.asarray(pd.Categorical(df["count_state"], categories=count_levels).codes)
    framing_levels = ["mid"] + sorted(t for t in df["framing_tier"].unique() if t != "mid")
    framing_idx = np.asarray(pd.Categorical(df["framing_tier"], categories=framing_levels).codes)
    tertile_levels = ["mid", "low", "high"]   # mid is baseline
    tertile_idx = np.asarray(pd.Categorical(df["chase_tertile"], categories=tertile_levels).codes)

    pitcher_idx, pitcher_cats = pd.factorize(df["pitcher"])
    catcher_idx, catcher_cats = pd.factorize(df["fielder_2"])
    umpire_idx, umpire_cats = pd.factorize(df["umpire"])

    # 2D BSpline tensor-product basis on (plate_x, plate_z)
    Bx = _bspline_basis(df["plate_x"].values.astype(float), n_knots=4, order=3)
    Bz = _bspline_basis(df["plate_z"].values.astype(float), n_knots=4, order=3)
    n_obs = len(df)
    Kx, Kz = Bx.shape[1], Bz.shape[1]
    B = np.einsum("ni,nj->nij", Bx, Bz).reshape(n_obs, Kx * Kz)
    col_var = B.var(axis=0)
    B = B[:, col_var > 1e-10]
    B = B - B.mean(axis=0)

    # Interaction encoding: matrix of shape (n_obs, K_spot_levels * 2) where the 2 = low/high tertile
    # We omit "mid" tertile (baseline). The interaction term is b_int[k_spot, k_tertile] applied
    # when (spot==spot_levels[k_spot]) AND (tertile==tertile_levels[k_tertile+1])  (since tertile
    # index 0 is mid).
    # We unroll this as a flat matrix of indicators, one column per (spot_level, non-mid-tertile) pair.
    int_pairs = []
    for k_spot, s in enumerate(spot_levels):
        for tert in ("low", "high"):
            int_pairs.append((k_spot, s, tert))
    n_int = len(int_pairs)
    int_idx = -np.ones(n_obs, dtype=int)   # for each row, which interaction column applies (or -1)
    for col, (k_spot, s, tert) in enumerate(int_pairs):
        mask = (df["lineup_spot"] == s) & (df["chase_tertile"] == tert)
        int_idx[mask.values] = col

    coords = {
        "spot_levels": spot_levels,
        "count_levels": count_levels[1:],
        "framing_levels": framing_levels[1:],
        "tertile_levels": tertile_levels[1:],     # low, high
        "interaction_pairs": [f"spot{p[1]}:{p[2]}" for p in int_pairs],
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
        b_tertile = pm.Normal("b_tertile", 0, 0.3, dims="tertile_levels")
        b_int = pm.Normal("b_int", 0, 0.3, dims="interaction_pairs")
        b_basis = pm.Normal("b_basis", 0, 2.0, dims="basis")

        sd_pitcher = pm.HalfNormal("sd_pitcher", 0.5)
        sd_catcher = pm.HalfNormal("sd_catcher", 0.5)
        sd_umpire = pm.HalfNormal("sd_umpire", 0.5)
        z_p = pm.Normal("z_p", 0, 1.0, dims="pitcher")
        z_c = pm.Normal("z_c", 0, 1.0, dims="catcher")
        z_u = pm.Normal("z_u", 0, 1.0, dims="umpire")
        u_p = pm.Deterministic("u_p", sd_pitcher * z_p, dims="pitcher")
        u_c = pm.Deterministic("u_c", sd_catcher * z_c, dims="catcher")
        u_u = pm.Deterministic("u_u", sd_umpire * z_u, dims="umpire")

        spot_eff = pm.math.concatenate([pt.zeros(1), b_spot])
        count_eff = pm.math.concatenate([pt.zeros(1), b_count])
        framing_eff = pm.math.concatenate([pt.zeros(1), b_framing])
        tertile_eff = pm.math.concatenate([pt.zeros(1), b_tertile])

        # Interaction effect lookup: pad b_int with a zero so int_idx == -1 maps to 0
        int_eff_padded = pm.math.concatenate([pt.zeros(1), b_int])
        # int_idx >= 0 selects b_int[int_idx]; int_idx==-1 selects 0
        int_lookup = int_idx + 1   # shift so -1 -> 0

        eta = (
            intercept
            + spot_eff[spot_idx]
            + count_eff[count_idx]
            + framing_eff[framing_idx]
            + tertile_eff[tertile_idx]
            + int_eff_padded[int_lookup]
            + pm.math.dot(B, b_basis)
            + u_p[pitcher_idx]
            + u_c[catcher_idx]
            + u_u[umpire_idx]
        )
        pm.Bernoulli("y", logit_p=eta, observed=df["called_strike"].astype(int).values)

        idata = pm.sample(
            draws=n_draws, tune=n_tune, chains=4, cores=4,
            target_accept=0.95, progressbar=False, random_seed=2026,
            init="adapt_diag",
        )

    info = {
        "spot_levels": spot_levels,
        "count_levels": count_levels[1:],
        "framing_levels": framing_levels[1:],
        "tertile_levels": tertile_levels[1:],
        "int_pairs": int_pairs,
        "spot_idx": spot_idx,
        "count_idx": count_idx,
        "framing_idx": framing_idx,
        "tertile_idx": tertile_idx,
        "int_idx": int_idx,
        "pitcher_idx": pitcher_idx,
        "catcher_idx": catcher_idx,
        "umpire_idx": umpire_idx,
        "B": B,
        "n_basis": B.shape[1],
        "umpire_cats": list(umpire_cats),
        "pitcher_cats": list(pitcher_cats),
        "catcher_cats": list(catcher_cats),
    }
    with open(H7_CACHE, "wb") as f:
        pickle.dump({"idata": idata, "info": info, "df": df}, f)
    return idata, info, df


def _per_tertile_pp_effect(idata, info, df: pd.DataFrame, n_keep: int = 800) -> dict:
    """Marginal posterior of P(CS | spot=7, tertile=T) - P(CS | spot=3, tertile=T)
    for each tertile T in {low, mid, high}, evaluated over the empirical sample.
    """
    posterior = idata.posterior
    rng = np.random.default_rng(7)
    flat = posterior.sizes["chain"] * posterior.sizes["draw"]
    n_keep = min(n_keep, flat)
    sample_ids = rng.choice(flat, n_keep, replace=False)

    intercept = posterior["intercept"].values.reshape(-1)[sample_ids]
    spot_levels = info["spot_levels"]
    b_spot = posterior["b_spot"].values.reshape(-1, len(spot_levels))[sample_ids]
    b_count = posterior["b_count"].values.reshape(-1, len(info["count_levels"]))[sample_ids]
    b_framing = posterior["b_framing"].values.reshape(-1, len(info["framing_levels"]))[sample_ids]
    b_tertile = posterior["b_tertile"].values.reshape(-1, len(info["tertile_levels"]))[sample_ids]
    b_int = posterior["b_int"].values.reshape(-1, len(info["int_pairs"]))[sample_ids]
    b_basis = posterior["b_basis"].values.reshape(-1, info["n_basis"])[sample_ids]
    u_p = posterior["u_p"].values.reshape(-1, len(info["pitcher_cats"]))[sample_ids]
    u_c = posterior["u_c"].values.reshape(-1, len(info["catcher_cats"]))[sample_ids]
    u_u = posterior["u_u"].values.reshape(-1, len(info["umpire_cats"]))[sample_ids]

    B = info["B"]
    count_idx = info["count_idx"]
    framing_idx = info["framing_idx"]
    pitcher_idx = info["pitcher_idx"]
    catcher_idx = info["catcher_idx"]
    umpire_idx = info["umpire_idx"]

    count_eff = np.where(count_idx[None, :] == 0, 0.0, b_count[:, np.maximum(count_idx - 1, 0)])
    framing_eff = np.where(framing_idx[None, :] == 0, 0.0, b_framing[:, np.maximum(framing_idx - 1, 0)])

    base = (
        intercept[:, None]
        + count_eff
        + framing_eff
        + b_basis @ B.T
        + u_p[:, pitcher_idx]
        + u_c[:, catcher_idx]
        + u_u[:, umpire_idx]
    )

    spot_to_idx = {3: 0}
    for i, s in enumerate(spot_levels, start=1):
        spot_to_idx[int(s)] = i

    # Helper: for given spot=s and tertile=t, the additional logit contribution = b_spot + b_tertile + b_int[s,t]
    int_pair_to_col = {(k_spot, tert): col for col, (k_spot, _, tert) in enumerate(info["int_pairs"])}
    if 7 in spot_to_idx and spot_to_idx[7] > 0:
        sp7 = b_spot[:, spot_to_idx[7] - 1]
        k_spot7 = spot_to_idx[7] - 1
    else:
        return {}

    results = {}
    for tert in ("low", "mid", "high"):
        if tert == "mid":
            te = np.zeros_like(intercept)
            int7 = np.zeros_like(intercept)
            int3 = np.zeros_like(intercept)
        else:
            te = b_tertile[:, info["tertile_levels"].index(tert)]
            col7 = int_pair_to_col.get((k_spot7, tert))
            int7 = b_int[:, col7] if col7 is not None else np.zeros_like(intercept)
            int3 = np.zeros_like(intercept)   # spot 3 is baseline
        # spot 3, tertile t: logit = base + 0 + te + 0
        # spot 7, tertile t: logit = base + sp7 + te + int7
        eta_3 = base + te[:, None] + int3[:, None]
        eta_7 = base + sp7[:, None] + te[:, None] + int7[:, None]
        p_3 = (1.0 / (1.0 + np.exp(-eta_3))).mean(axis=1)
        p_7 = (1.0 / (1.0 + np.exp(-eta_7))).mean(axis=1)
        diff = (p_7 - p_3) * 100.0
        results[f"{tert}_chase_spot7_vs_spot3_pp"] = {
            "median": float(np.median(diff)),
            "ci_low": float(np.percentile(diff, 2.5)),
            "ci_high": float(np.percentile(diff, 97.5)),
            "p_neg": float((diff < 0).mean()),
        }

    # Interaction tests on the b_int term itself for spot 7
    # b_int[k_spot7, "low"] is the difference (low_chase × spot7) − (mid_chase × spot7)  in logit
    int_low = b_int[:, int_pair_to_col[(k_spot7, "low")]]   # shape (S,)
    int_high = b_int[:, int_pair_to_col[(k_spot7, "high")]]
    # difference between low and high interaction effects (test the FanSided narrative)
    int_low_minus_high = int_low - int_high
    # Test posterior tail prob that low chase has more negative effect than mid (i.e. int_low<0)
    results["interaction_b_int_low_spot7"] = {
        "median": float(np.median(int_low)),
        "ci_low": float(np.percentile(int_low, 2.5)),
        "ci_high": float(np.percentile(int_low, 97.5)),
        "p_neg": float((int_low < 0).mean()),
    }
    results["interaction_b_int_high_spot7"] = {
        "median": float(np.median(int_high)),
        "ci_low": float(np.percentile(int_high, 2.5)),
        "ci_high": float(np.percentile(int_high, 97.5)),
        "p_neg": float((int_high < 0).mean()),
    }
    results["int_low_minus_high_spot7"] = {
        "median": float(np.median(int_low_minus_high)),
        "ci_low": float(np.percentile(int_low_minus_high, 2.5)),
        "ci_high": float(np.percentile(int_low_minus_high, 97.5)),
        "p_neg": float((int_low_minus_high < 0).mean()),
    }
    return results


def plot(results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
    labels = ["Low-chase\n(disciplined)", "Mid-chase", "High-chase"]
    keys = ["low_chase_spot7_vs_spot3_pp", "mid_chase_spot7_vs_spot3_pp", "high_chase_spot7_vs_spot3_pp"]
    medians = [results[k]["median"] for k in keys]
    los = [results[k]["ci_low"] for k in keys]
    his = [results[k]["ci_high"] for k in keys]
    pos = np.arange(3)
    err_lo = np.array(medians) - np.array(los)
    err_hi = np.array(his) - np.array(medians)
    colors = ["#ef4444", "#94a3b8", "#10b981"]
    ax.errorbar(pos, medians, yerr=[err_lo, err_hi], fmt="o", color="#1f2937",
                ecolor=colors, elinewidth=4, capsize=0, markersize=10)
    for i, c in enumerate(colors):
        ax.plot(pos[i], medians[i], "o", markersize=12, color=c)
    ax.axhline(0, color="#1d4ed8", linestyle="--", linewidth=1)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spot-7-vs-spot-3 effect on borderline\ncalled-strike rate (pp)")
    ax.set_title("H7: 7-vs-3 effect by 2025 chase-rate tertile\n(low chase = most disciplined hitters)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run() -> dict:
    df0 = data_prep_r2.load_borderline()
    chase = data_prep_r2.load_chase_rate()
    print(f"[H7] borderline n={len(df0):,}, chase batters {chase.qualified_200pa.sum()}")
    df = _prepare(df0, chase)
    print(f"[H7] joined to qualified-2025 batters: n={len(df):,} ({df['batter'].nunique()} unique batters)")
    print(df.groupby(["lineup_spot", "chase_tertile"]).size().unstack(fill_value=0))

    print("[H7] fitting ...")
    idata, info, df = fit(df)
    summary = az.summary(idata, var_names=["intercept", "b_spot", "b_tertile", "b_int", "sd_pitcher", "sd_catcher", "sd_umpire"], hdi_prob=0.95)
    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    print(f"  R-hat max = {rhat_max:.3f}, ESS bulk min = {ess_min:.0f}")

    results = _per_tertile_pp_effect(idata, info, df)
    print("Per-tertile spot-7-vs-spot-3 effects:")
    for k, v in results.items():
        print(f"  {k}: median={v['median']:.2f}, CI=[{v['ci_low']:.2f}, {v['ci_high']:.2f}], P(neg)={v['p_neg']:.3f}")

    plot(results, CHARTS / "h7_chase_tertile_effect.png")

    az.plot_trace(idata, var_names=["intercept", "b_spot", "b_tertile", "b_int"], compact=True)
    plt.tight_layout()
    plt.savefig(DIAG / "h7_trace.png", dpi=110)
    plt.close()
    summary.to_csv(DIAG / "h7_summary.csv")

    out = {
        "n_obs": int(len(df)),
        "n_batters": int(df["batter"].nunique()),
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "results": results,
    }
    return out


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str))

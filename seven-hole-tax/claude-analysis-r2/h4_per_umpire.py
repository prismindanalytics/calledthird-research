"""h4_per_umpire.py — Per-umpire random-slope hierarchical Bayesian GLM.

Model (extension of Round 1's H3 GAM):
  is_called_strike ~ lineup_spot
                   + s(plate_x, plate_z)
                   + count_state
                   + framing_tier
                   + (1|pitcher) + (1|catcher) + (1|umpire)
                   + (0 + I(lineup_spot in {7,8,9}) | umpire)   <- the new random slope

Why "in {7,8,9}" rather than spot==7 only:
  Round 1 found the bottom-of-order pattern, not 7-specific. With ~30 spot-7
  pitches per umpire on average, a per-umpire random slope on the spot-7
  indicator alone would be heavily shrunk to zero. The brief's H4 wording
  asks about "spot-7 effect" but the substantive question is whether any
  umpire is differentially calling bottom-of-order at-bats. We therefore
  fit `bottom_of_order = lineup_spot in {7,8,9}` as the random slope, which
  gives ~100 spot-7-9 pitches per umpire (qualifying threshold), enough to
  identify per-umpire deviations.

  Spot-3 (3-5 hitters) is the canonical "top of the order" reference. We use
  spots 1-3 here as the reference to mirror H3's "vs. spot 3" framing while
  giving each umpire enough sample (~120 pitches in spots 1-3 on average).
  This makes the slope contrast: bottom-of-order (7-9) vs top-of-order (1-3).

Filter:
  Only umpires with >=50 borderline calls in spots 7-9 AND >=50 in spots 1-3.

Posterior summary per umpire:
  - Posterior median + 95% CrI for the marginal pp effect
    (probability difference of called-strike between bottom-of-order and
     top-of-order at the umpire's mean conditioning configuration).
  - Probability of direction P(slope < 0).

Identify:
  - umpires whose 95% CrI excludes zero AND magnitude >= 2pp ("flagged").
  - directional flag: "pro-tax" (negative effect) vs "reverse" (positive).
  - BH-FDR correction on the two-sided posterior tail probabilities.

Diagnostics:
  R-hat, ESS for the random-slope sd and a sample of per-umpire slopes.
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

H4_CACHE = CACHE / "h4_idata.pkl"


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


def _bh_fdr(pvals: np.ndarray, q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg. Returns (q_values, rejected_at_q)."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    qvals_sorted = ranked * n / (np.arange(1, n + 1))
    qvals_sorted = np.minimum.accumulate(qvals_sorted[::-1])[::-1]
    qvals = np.empty_like(pvals)
    qvals[order] = qvals_sorted
    return qvals, qvals < q


def _filter_qualifying(df: pd.DataFrame, min_each: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop umpires that don't have >=min_each borderline calls in spots 7-9 AND in spots 1-3."""
    counts = df.groupby("umpire").apply(
        lambda g: pd.Series({
            "n_bot": int((g["lineup_spot"] >= 7).sum()),
            "n_top": int((g["lineup_spot"] <= 3).sum()),
            "n_total": len(g),
        }),
        include_groups=False,
    )
    counts = counts.reset_index()
    qualifiers = counts[(counts["n_bot"] >= min_each) & (counts["n_top"] >= min_each)]
    df_q = df[df["umpire"].isin(qualifiers["umpire"])].copy()
    return df_q, counts


def fit_h4(n_draws: int = 1000, n_tune: int = 1500, force: bool = False):
    """Fit the per-umpire random-slope GLM. Returns idata, info, df."""
    if H4_CACHE.exists() and not force:
        with open(H4_CACHE, "rb") as f:
            blob = pickle.load(f)
        return blob["idata"], blob["info"], blob["df"], blob["counts"]

    df_full = data_prep_r2.load_borderline()
    df, counts = _filter_qualifying(df_full, min_each=50)
    print(f"[H4] umpires after filter: {df['umpire'].nunique()} / {df_full['umpire'].nunique()} "
          f"(borderline pitches: {len(df):,} / {len(df_full):,})")

    # Outcome
    y = df["called_strike"].astype(int).values
    # Bottom-of-order indicator (vs top-of-order baseline)
    df["bot_of_order"] = (df["lineup_spot"] >= 7).astype(int)
    df["top_of_order"] = (df["lineup_spot"] <= 3).astype(int)
    df["middle"] = ((df["lineup_spot"] >= 4) & (df["lineup_spot"] <= 6)).astype(int)
    bot = df["bot_of_order"].values.astype(int)

    # Categorical encodings (spot 3 is baseline for the categorical lineup-spot fixed effect)
    spot_levels = sorted(df["lineup_spot"].unique())
    spot_levels = [s for s in spot_levels if s != 3]
    spot_idx = np.asarray(pd.Categorical(df["lineup_spot"], categories=[3] + spot_levels).codes)
    count_levels = ["0-0"] + sorted(c for c in df["count_state"].unique() if c != "0-0")
    count_idx = np.asarray(pd.Categorical(df["count_state"], categories=count_levels).codes)
    framing_levels = ["mid"] + sorted(t for t in df["framing_tier"].unique() if t != "mid")
    framing_idx = np.asarray(pd.Categorical(df["framing_tier"], categories=framing_levels).codes)

    pitcher_idx, pitcher_cats = pd.factorize(df["pitcher"])
    catcher_idx, catcher_cats = pd.factorize(df["fielder_2"])
    umpire_idx, umpire_cats = pd.factorize(df["umpire"])

    # 2D BSpline tensor-product basis on (plate_x, plate_z)
    n_basis_per_axis = 4
    Bx = _bspline_basis(df["plate_x"].values.astype(float), n_knots=n_basis_per_axis, order=3)
    Bz = _bspline_basis(df["plate_z"].values.astype(float), n_knots=n_basis_per_axis, order=3)
    n_obs = len(df)
    Kx, Kz = Bx.shape[1], Bz.shape[1]
    B = np.einsum("ni,nj->nij", Bx, Bz).reshape(n_obs, Kx * Kz)
    col_var = B.var(axis=0)
    B = B[:, col_var > 1e-10]
    B = B - B.mean(axis=0)

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
        b_basis = pm.Normal("b_basis", 0, 2.0, dims="basis")

        sd_pitcher = pm.HalfNormal("sd_pitcher", 0.5)
        sd_catcher = pm.HalfNormal("sd_catcher", 0.5)
        sd_umpire = pm.HalfNormal("sd_umpire", 0.5)
        sd_umpire_botslope = pm.HalfNormal("sd_umpire_botslope", 0.3)

        z_pitcher = pm.Normal("z_pitcher", 0, 1.0, dims="pitcher")
        z_catcher = pm.Normal("z_catcher", 0, 1.0, dims="catcher")
        z_umpire = pm.Normal("z_umpire", 0, 1.0, dims="umpire")
        # Per-umpire random slope on bottom-of-order indicator (the per-umpire 7-Hole-Tax)
        z_botslope = pm.Normal("z_botslope", 0, 1.0, dims="umpire")

        u_pitcher = pm.Deterministic("u_pitcher", sd_pitcher * z_pitcher, dims="pitcher")
        u_catcher = pm.Deterministic("u_catcher", sd_catcher * z_catcher, dims="catcher")
        u_umpire = pm.Deterministic("u_umpire", sd_umpire * z_umpire, dims="umpire")
        u_botslope = pm.Deterministic("u_botslope", sd_umpire_botslope * z_botslope, dims="umpire")

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
            + u_botslope[umpire_idx] * bot
        )
        pm.Bernoulli("y", logit_p=eta, observed=y)

        idata = pm.sample(
            draws=n_draws, tune=n_tune, chains=4, cores=4,
            target_accept=0.95, progressbar=False, random_seed=2026,
            init="adapt_diag",
        )

    info = {
        "spot_levels": spot_levels,
        "count_levels": count_levels[1:],
        "framing_levels": framing_levels[1:],
        "umpire_cats": list(umpire_cats),
        "pitcher_cats": list(pitcher_cats),
        "catcher_cats": list(catcher_cats),
        "n_obs": n_obs,
        "n_basis": B.shape[1],
        "B": B,
        "spot_idx": spot_idx,
        "count_idx": count_idx,
        "framing_idx": framing_idx,
        "pitcher_idx": pitcher_idx,
        "catcher_idx": catcher_idx,
        "umpire_idx": umpire_idx,
        "bot": bot,
    }
    with open(H4_CACHE, "wb") as f:
        pickle.dump({"idata": idata, "info": info, "df": df, "counts": counts}, f)
    return idata, info, df, counts


def _per_umpire_pp_effects(idata, info, df: pd.DataFrame, n_keep: int = 800) -> pd.DataFrame:
    """Per-umpire posterior of the marginal pp effect: P(CS | bot-of-order) - P(CS | top-of-order)
    averaging over the umpire's actual borderline-pitch context.

    For each umpire u:
      For each posterior draw s:
        - For all of u's actual rows i:
            base_logit_i = intercept + count_eff_i + framing_eff_i + basis_i + u_pitcher_i + u_catcher_i + u_umpire_u
            (note: spot_eff is the FIXED categorical contribution; we'll set this differently
             for the bot-of-order vs top-of-order counterfactual)
            spot_eff_top_i  = b_spot[spot==1 | spot==2 | spot==3] averaged over the
                             marginal proportion of spots 1/2/3 in u's spot 1-3 rows
            spot_eff_bot_i  = b_spot averaged over the marginal proportion of spots 7/8/9 in u's bot-of-order rows
            slope_bot       = u_botslope_u * 1
            slope_top       = u_botslope_u * 0
            P_top_i = sigmoid(base_logit_i + spot_eff_top + slope_top)
            P_bot_i = sigmoid(base_logit_i + spot_eff_bot + slope_bot)
        delta_u_s = mean_i(P_bot_i - P_top_i) * 100   (in pp)
      Posterior of delta_u: per-umpire effect.

    Approximation: rather than computing two counterfactual logits per row
    (bot/top), we approximate the marginal pp effect at sigmoid' = 0.25
    (logit derivative max). The bot-vs-top effect on the linear predictor is
    `b_spot_avg_bot - b_spot_avg_top + u_botslope_u`. Convert to pp by
    multiplying by sigmoid'(eta_bar) where eta_bar is the per-umpire
    average linear predictor — this gives row-level marginal effect.

    To stay faithful, we DO compute counterfactuals row-by-row but only on the
    umpire's actual rows.
    """
    posterior = idata.posterior
    rng = np.random.default_rng(7)
    flat = posterior.sizes["chain"] * posterior.sizes["draw"]
    n_keep = min(n_keep, flat)
    sample_ids = rng.choice(flat, n_keep, replace=False)

    spot_levels = info["spot_levels"]
    intercept = posterior["intercept"].values.reshape(-1)[sample_ids]                          # (S,)
    b_spot = posterior["b_spot"].values.reshape(-1, len(spot_levels))[sample_ids]              # (S, K_spot_levels)
    b_count = posterior["b_count"].values.reshape(-1, len(info["count_levels"]))[sample_ids]
    b_framing = posterior["b_framing"].values.reshape(-1, len(info["framing_levels"]))[sample_ids]
    b_basis = posterior["b_basis"].values.reshape(-1, info["n_basis"])[sample_ids]
    u_pitcher = posterior["u_pitcher"].values.reshape(-1, len(info["pitcher_cats"]))[sample_ids]
    u_catcher = posterior["u_catcher"].values.reshape(-1, len(info["catcher_cats"]))[sample_ids]
    u_umpire = posterior["u_umpire"].values.reshape(-1, len(info["umpire_cats"]))[sample_ids]
    u_botslope = posterior["u_botslope"].values.reshape(-1, len(info["umpire_cats"]))[sample_ids]

    B = info["B"]
    count_idx = info["count_idx"]
    framing_idx = info["framing_idx"]
    pitcher_idx = info["pitcher_idx"]
    catcher_idx = info["catcher_idx"]
    umpire_idx = info["umpire_idx"]
    spot_idx = info["spot_idx"]

    # Spot-effect map: index 0 -> 0 (spot 3), index k>0 -> b_spot[:, k-1]
    # b_spot_full (S, max_spot_idx+1) with leading column zeros (for baseline=spot 3)
    K = len(spot_levels) + 1
    b_spot_full = np.zeros((b_basis.shape[0], K))
    b_spot_full[:, 1:] = b_spot

    count_eff = np.where(count_idx[None, :] == 0, 0.0, b_count[:, np.maximum(count_idx - 1, 0)])
    framing_eff = np.where(framing_idx[None, :] == 0, 0.0, b_framing[:, np.maximum(framing_idx - 1, 0)])

    # Pre-compute the per-row, per-draw logit *minus* the spot effect *and* the per-umpire bot-slope contribution.
    # eta_minus = intercept + count_eff + framing_eff + Bb + u_p + u_c + u_u_intercept
    Bb = b_basis @ B.T     # (S, n_obs)
    eta_minus = (
        intercept[:, None]
        + count_eff
        + framing_eff
        + Bb
        + u_pitcher[:, pitcher_idx]
        + u_catcher[:, catcher_idx]
        + u_umpire[:, umpire_idx]
    )
    # (S, n_obs)

    # For each umpire, compute counterfactual P(CS) under (a) bot-of-order at spot 7
    # and (b) top-of-order at spot 3, weighted by the umpire's mix of bot/top spots.
    # For richness, we compute per-spot predictions and then weight by the umpire's
    # in-sample proportions. This matches "what would this umpire's CS rate look like
    # if these same pitches were thrown to a spot-7 vs spot-3 batter."
    spot_to_idx = {3: 0}
    for i, s in enumerate(spot_levels, start=1):
        spot_to_idx[int(s)] = i

    # We focus on the spot-7 vs spot-3 contrast (the named hypothesis), but also
    # compute the broader bot-of-order (7-9) vs top-of-order (1-3) for context.
    # For each umpire:
    #   collect mask_u = umpire_idx == u
    #   spot_eff_at_3 (per draw) = 0
    #   spot_eff_at_7 (per draw) = b_spot[:, spot_to_idx[7] - 1]  if 7 is a level
    # Compute P(CS) at row in two scenarios; average across rows.
    out_rows = []
    n_umpires = len(info["umpire_cats"])

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    for u in range(n_umpires):
        mask = umpire_idx == u
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        # subset eta_minus to umpire's rows
        eta_u = eta_minus[:, idx]                 # (S, n_u)
        slope_u = u_botslope[:, u][:, None]       # (S, 1)
        # SPOT 7 scenario: spot_eff = b_spot[:, spot_to_idx[7]-1], bot=1 (so add slope_u)
        if 7 in spot_to_idx and spot_to_idx[7] > 0:
            sp7 = b_spot[:, spot_to_idx[7] - 1][:, None]
        else:
            sp7 = np.zeros((b_basis.shape[0], 1))
        # SPOT 3 scenario: spot_eff = 0 (baseline), bot=0
        sp3 = np.zeros_like(sp7)
        p7 = sigmoid(eta_u + sp7 + slope_u)       # (S, n_u)
        p3 = sigmoid(eta_u + sp3)                 # (S, n_u)
        delta_pp = (p7.mean(axis=1) - p3.mean(axis=1)) * 100.0   # (S,)

        # Also: bot-of-order avg (7,8,9) vs top-of-order avg (1,2,3)
        spots_bot_present = [s for s in (7, 8, 9) if s in spot_to_idx and spot_to_idx[s] > 0]
        spots_top_present = [s for s in (1, 2) if s in spot_to_idx and spot_to_idx[s] > 0]
        b_bot = np.mean([b_spot[:, spot_to_idx[s] - 1] for s in spots_bot_present], axis=0)[:, None] if spots_bot_present else np.zeros((b_basis.shape[0], 1))
        b_top_partial = np.mean([b_spot[:, spot_to_idx[s] - 1] for s in spots_top_present], axis=0) if spots_top_present else np.zeros(b_basis.shape[0])
        # spot 3 contributes 0; weight: top mean = (b_spot1+b_spot2+0)/3
        b_top = (b_top_partial * len(spots_top_present) / 3.0)[:, None]
        p_bot = sigmoid(eta_u + b_bot + slope_u)
        p_top = sigmoid(eta_u + b_top)
        delta_bot_top_pp = (p_bot.mean(axis=1) - p_top.mean(axis=1)) * 100.0

        umpire_name = info["umpire_cats"][u]
        n_bot = int((df["umpire"] == umpire_name)[df["lineup_spot"] >= 7].sum() if umpire_name in df["umpire"].values else 0)
        n_top = int((df["umpire"] == umpire_name)[df["lineup_spot"] <= 3].sum() if umpire_name in df["umpire"].values else 0)
        n_total = int((df["umpire"] == umpire_name).sum())

        out_rows.append({
            "umpire": umpire_name,
            "n_total": n_total,
            "n_bot": n_bot,
            "n_top": n_top,
            "delta_spot7_vs_3_pp_med": float(np.median(delta_pp)),
            "delta_spot7_vs_3_pp_lo": float(np.percentile(delta_pp, 2.5)),
            "delta_spot7_vs_3_pp_hi": float(np.percentile(delta_pp, 97.5)),
            "p_dir_neg": float((delta_pp < 0).mean()),  # P(slope < 0)
            "delta_bot_vs_top_pp_med": float(np.median(delta_bot_top_pp)),
            "delta_bot_vs_top_pp_lo": float(np.percentile(delta_bot_top_pp, 2.5)),
            "delta_bot_vs_top_pp_hi": float(np.percentile(delta_bot_top_pp, 97.5)),
            "p_dir_neg_bot": float((delta_bot_top_pp < 0).mean()),
            "u_botslope_med": float(np.median(u_botslope[:, u])),
            "u_botslope_lo": float(np.percentile(u_botslope[:, u], 2.5)),
            "u_botslope_hi": float(np.percentile(u_botslope[:, u], 97.5)),
        })
    return pd.DataFrame(out_rows)


def _flag_and_correct(per_ump: pd.DataFrame, magnitude_threshold_pp: float = 2.0) -> pd.DataFrame:
    """Apply BH-FDR correction on two-sided posterior tail probability and flag umpires."""
    per_ump = per_ump.copy()
    # two-sided "p-value" from posterior P(direction): p2 = 2 * min(p_dir_neg, 1-p_dir_neg)
    p2 = 2 * np.minimum(per_ump["p_dir_neg"], 1.0 - per_ump["p_dir_neg"])
    per_ump["p_two_sided"] = p2
    qvals, rejected = _bh_fdr(p2.values, q=0.10)
    per_ump["q_bh"] = qvals
    per_ump["bh_significant_q10"] = rejected
    # CI excludes zero AND magnitude >= 2pp
    per_ump["ci_excludes_zero"] = (per_ump["delta_spot7_vs_3_pp_lo"] > 0) | (per_ump["delta_spot7_vs_3_pp_hi"] < 0)
    per_ump["abs_med_pp"] = per_ump["delta_spot7_vs_3_pp_med"].abs()
    per_ump["flagged"] = per_ump["ci_excludes_zero"] & (per_ump["abs_med_pp"] >= magnitude_threshold_pp)
    per_ump["direction"] = np.where(per_ump["delta_spot7_vs_3_pp_med"] < 0, "pro-tax", "reverse")
    return per_ump.sort_values("delta_spot7_vs_3_pp_med")


def plot_distribution(per_ump: pd.DataFrame, out_path: Path) -> None:
    df = per_ump.sort_values("delta_spot7_vs_3_pp_med").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.18 * len(df))), dpi=140)
    medians = df["delta_spot7_vs_3_pp_med"].values
    los = medians - df["delta_spot7_vs_3_pp_lo"].values
    his = df["delta_spot7_vs_3_pp_hi"].values - medians
    flagged = df["flagged"].values
    colors = np.where(flagged, np.where(df["direction"] == "pro-tax", "#ef4444", "#10b981"), "#94a3b8")
    pos = np.arange(len(df))
    ax.errorbar(medians, pos, xerr=[los, his], fmt="o", color="#1f2937",
                ecolor=colors, elinewidth=2, capsize=0, markersize=4)
    for i in range(len(df)):
        if flagged[i]:
            ax.plot(medians[i], pos[i], "o", markersize=8, color=colors[i])
            ax.text(medians[i] + (0.4 if medians[i] >= 0 else -0.4), pos[i],
                    df.iloc[i]["umpire"], va="center", ha="left" if medians[i] >= 0 else "right",
                    fontsize=8, color=colors[i])
    ax.axvline(0, color="#1d4ed8", linestyle="--", linewidth=1)
    ax.set_yticks([])
    ax.set_xlabel("Per-umpire spot-7-vs-spot-3 effect on borderline called-strike rate (pp)\n"
                  "negative = umpire calls more strikes against bottom-of-order (pro-tax direction)")
    ax.set_title(f"H4: Per-umpire spot-7 effects ({len(df)} qualifying umpires)\n"
                 f"flagged: 95% CrI excludes zero AND |median| >= 2pp")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run() -> dict:
    print("[H4] fitting per-umpire random-slope GLM ...")
    idata, info, df, counts = fit_h4()
    summary = az.summary(idata, var_names=["intercept", "b_spot", "sd_pitcher", "sd_catcher", "sd_umpire", "sd_umpire_botslope"], hdi_prob=0.95)
    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    print(f"  global R-hat max = {rhat_max:.3f}, ESS bulk min = {ess_min:.0f}")

    # Per-umpire R-hat / ESS for the random slope variable
    sl_summary = az.summary(idata, var_names=["u_botslope"], hdi_prob=0.95)
    sl_rhat_max = float(sl_summary["r_hat"].max())
    sl_ess_min = float(sl_summary["ess_bulk"].min())
    print(f"  per-umpire random-slope R-hat max = {sl_rhat_max:.3f}, ESS bulk min = {sl_ess_min:.0f}")

    per_ump = _per_umpire_pp_effects(idata, info, df, n_keep=800)
    per_ump = _flag_and_correct(per_ump)
    per_ump.to_csv(ROOT / "h4_per_umpire_results.csv", index=False)

    flagged = per_ump[per_ump["flagged"]].sort_values("delta_spot7_vs_3_pp_med")
    print(f"[H4] flagged umpires: {len(flagged)} of {len(per_ump)}")
    if len(flagged) > 0:
        print(flagged[["umpire", "n_bot", "n_top", "delta_spot7_vs_3_pp_med", "delta_spot7_vs_3_pp_lo", "delta_spot7_vs_3_pp_hi", "direction"]])

    plot_distribution(per_ump, CHARTS / "h4_per_umpire_distribution.png")

    # Diagnostic plots
    az.plot_trace(idata, var_names=["intercept", "sd_umpire", "sd_umpire_botslope"], compact=True)
    plt.tight_layout()
    plt.savefig(DIAG / "h4_trace.png", dpi=110)
    plt.close()
    summary.to_csv(DIAG / "h4_summary.csv")
    sl_summary.to_csv(DIAG / "h4_botslope_summary.csv")

    league_dist = {
        "median": float(per_ump["delta_spot7_vs_3_pp_med"].median()),
        "iqr_low": float(per_ump["delta_spot7_vs_3_pp_med"].quantile(0.25)),
        "iqr_high": float(per_ump["delta_spot7_vs_3_pp_med"].quantile(0.75)),
        "min": float(per_ump["delta_spot7_vs_3_pp_med"].min()),
        "max": float(per_ump["delta_spot7_vs_3_pp_med"].max()),
    }

    return {
        "n_qualifying_umpires": int(len(per_ump)),
        "n_flagged": int(per_ump["flagged"].sum()),
        "rhat_max_global": rhat_max,
        "rhat_max_botslope": sl_rhat_max,
        "ess_min_global": ess_min,
        "ess_min_botslope": sl_ess_min,
        "sd_umpire_botslope_med": float(idata.posterior["sd_umpire_botslope"].median().item()),
        "flagged_list": flagged.to_dict(orient="records") if len(flagged) > 0 else [],
        "league_distribution": league_dist,
        "per_umpire_csv": str((ROOT / "h4_per_umpire_results.csv").resolve()),
    }


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str))

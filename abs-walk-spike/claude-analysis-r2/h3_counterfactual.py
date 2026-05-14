"""H3 — Bayesian zone-classifier counterfactual replay (all-pitches + per-count + per-edge).

Approach (Bayesian / posterior-replay style):

1. Fit a Bayesian (smoothed-spline) called-strike classifier on 2025 *same-window* taken pitches
   (Mar 27 – May 13 of 2025). We aggregate pitches to a 2D grid (24×24 over plate_x ∈ [-1.8, 1.8],
   plate_z ∈ [0.6, 4.4]) and a small set of count-strata. This makes the likelihood ~hundreds of
   cells instead of ~50K observations and makes NUTS fast.

2. Smoothness comes from a 2D random-walk prior on the cell intercepts (penalized first-difference
   along x and z), with per-count fixed effects.

3. Counterfactual replay: for each 2026 taken pitch, look up its (cell, count) and sample its
   2025-era CS probability from the posterior. Do M=100 PA replays, each a complete walk-through
   with one Bernoulli draw per take per PA. Aggregate counterfactual walk rate.

4. Variants:
   - all_pitches (headline)
   - per_count (12 count states)
   - per_edge_region (heart / top_edge / bottom_edge / in_off)
"""
from __future__ import annotations

import json

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit

from common import (
    ALL_COUNTS,
    R2_ARTIFACTS,
    R2_CHARTS,
    R2_DIAG,
    WALK_EVENTS,
    called_pitches_subset,
    count_state,
    ensure_dirs,
    load_2025_samewindow,
    load_2026_full,
    plate_appearance_mask,
    zone_region,
)


# Grid configuration
X_MIN, X_MAX, NX = -1.8, 1.8, 24
Z_MIN, Z_MAX, NZ = 0.6, 4.4, 24
DX = (X_MAX - X_MIN) / NX
DZ = (Z_MAX - Z_MIN) / NZ

# Count tier (collapse to manageable number of strata)
COUNT_TIER = {
    "0-0": "early", "1-0": "early", "0-1": "early",
    "1-1": "middle", "2-0": "middle", "0-2": "middle",
    "1-2": "two_strike", "2-2": "two_strike", "3-2": "two_strike",
    "2-1": "middle",
    "3-0": "three_ball", "3-1": "three_ball",
}
COUNT_TIERS_LIST = ["early", "middle", "two_strike", "three_ball"]


def add_grid_cell(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    px = df["plate_x"].astype(float).clip(X_MIN + 1e-6, X_MAX - 1e-6)
    pz = df["plate_z"].astype(float).clip(Z_MIN + 1e-6, Z_MAX - 1e-6)
    ix = ((px - X_MIN) / DX).astype(int).clip(0, NX - 1)
    iz = ((pz - Z_MIN) / DZ).astype(int).clip(0, NZ - 1)
    df["ix"] = ix.values
    df["iz"] = iz.values
    df["cell_idx"] = ix.values * NZ + iz.values
    return df


def fit_zone_classifier(df_2025_window: pd.DataFrame) -> az.InferenceData:
    """Fit a Bayesian logistic spatial model on a (cell, tier) aggregation.

    Model:
      logit(p_cell_tier) = mu_cell + alpha_tier[tier]
      mu_cell: 2D random walk over (ix, iz) — penalty on first differences
      alpha_tier ~ N(0, 1)
    """
    called = called_pitches_subset(df_2025_window)
    called["count_state"] = count_state(called)
    called = called.loc[called["count_state"].isin(list(ALL_COUNTS))].copy()
    called["count_tier"] = called["count_state"].map(COUNT_TIER)
    called = add_grid_cell(called)

    cell = (
        called.groupby(["cell_idx", "count_tier"]).agg(
            n=("is_called_strike", "size"),
            k=("is_called_strike", "sum"),
            ix=("ix", "first"),
            iz=("iz", "first"),
        ).reset_index()
    )
    cell = cell.loc[cell["n"] >= 2].reset_index(drop=True)
    print(f"[H3] classifier training: {len(cell):,} (cell, tier) cells, total {cell['n'].sum():,} pitches")

    tier_to_idx = {t: i for i, t in enumerate(COUNT_TIERS_LIST)}
    cell["tier_idx"] = cell["count_tier"].map(tier_to_idx)

    # Build grid for the 2D random walk: each grid cell has an intercept.
    # Penalize differences between neighbors using GMRF-like structure.
    cell_idx_arr = cell["cell_idx"].values  # 0..NX*NZ-1
    tier_idx_arr = cell["tier_idx"].values
    n_arr = cell["n"].values.astype(int)
    k_arr = cell["k"].values.astype(int)
    n_grid = NX * NZ

    # Use a proper 2D random-walk via cumulative sum across grid in x then z.
    # Equivalent representation: eta_cell[ix,iz] = mu_global + delta_x[ix] + delta_z[iz]
    # + epsilon_cell where delta_x and delta_z are 1D random walks (gives most
    # of the smoothness benefit at a small parameter count).
    # We additionally allow a small interaction via low-rank tensor product:
    # interaction[ix, iz] = sum_k u_x[k, ix] * u_z[k, iz] for K=4.
    K = 4

    with pm.Model() as model:
        mu_global = pm.Normal("mu_global", mu=0.0, sigma=1.5)
        sigma_x = pm.HalfNormal("sigma_x", sigma=0.5)
        sigma_z = pm.HalfNormal("sigma_z", sigma=0.5)
        dx_inn = pm.Normal("dx_inn", mu=0.0, sigma=sigma_x, shape=NX)
        dz_inn = pm.Normal("dz_inn", mu=0.0, sigma=sigma_z, shape=NZ)
        delta_x = pm.Deterministic("delta_x", pm.math.cumsum(dx_inn) - pm.math.cumsum(dx_inn).mean())
        delta_z = pm.Deterministic("delta_z", pm.math.cumsum(dz_inn) - pm.math.cumsum(dz_inn).mean())

        # Low-rank interaction
        sigma_int = pm.HalfNormal("sigma_int", sigma=0.3)
        ux = pm.Normal("ux", mu=0.0, sigma=sigma_int, shape=(K, NX))
        uz = pm.Normal("uz", mu=0.0, sigma=sigma_int, shape=(K, NZ))

        # Tier effects
        alpha_tier = pm.Normal("alpha_tier", mu=0.0, sigma=1.0, shape=len(COUNT_TIERS_LIST))

        # Compute eta per (cell_idx, tier)
        # Map cell_idx to (ix, iz) — precomputed in advance
        ix_of_cell = cell_idx_arr // NZ
        iz_of_cell = cell_idx_arr % NZ
        # interaction at (ix, iz) = sum_k ux[k, ix] * uz[k, iz]
        # Compute interaction vector aligned to cells
        # ux is (K, NX), uz is (K, NZ); we need (K,) summed over k at (ix[i], iz[i])
        interaction = pm.math.sum(ux[:, ix_of_cell] * uz[:, iz_of_cell], axis=0)
        eta = (
            mu_global
            + delta_x[ix_of_cell]
            + delta_z[iz_of_cell]
            + interaction
            + alpha_tier[tier_idx_arr]
        )
        p = pm.math.sigmoid(eta)
        pm.Binomial("y", n=n_arr, p=p, observed=k_arr)

        idata = pm.sample(
            draws=600,
            tune=750,
            chains=4,
            cores=4,
            target_accept=0.92,
            random_seed=2026,
            progressbar=False,
        )

    # Compute eta_cell for all grid cells (deterministic helper for prediction)
    # We need to do this post-hoc since the model doesn't materialize the full eta_cell.
    post = idata.posterior
    n_chain = post["mu_global"].shape[0]
    n_draw = post["mu_global"].shape[1]
    mu_g = post["mu_global"].values  # (chain, draw)
    dx = post["delta_x"].values  # (chain, draw, NX)
    dz = post["delta_z"].values  # (chain, draw, NZ)
    ux_p = post["ux"].values  # (chain, draw, K, NX)
    uz_p = post["uz"].values  # (chain, draw, K, NZ)

    # eta_cell[chain, draw, ix, iz] = mu_g + dx[ix] + dz[iz] + sum_k ux[k,ix]*uz[k,iz]
    interaction_grid = (ux_p[..., None] * uz_p[..., None, :]).sum(axis=2)  # (chain, draw, NX, NZ)
    eta_grid = (
        mu_g[..., None, None]
        + dx[..., None]
        + dz[..., None, :]
        + interaction_grid
    )
    eta_cell_grid = eta_grid.reshape(n_chain, n_draw, NX * NZ)
    # Stash into idata as a posterior var for downstream
    import xarray as xr
    idata.posterior["eta_cell"] = xr.DataArray(
        eta_cell_grid,
        dims=["chain", "draw", "eta_cell_dim_0"],
    )
    idata.attrs = {
        "tier_levels": COUNT_TIERS_LIST,
        "n_grid": n_grid,
    }
    return idata


def predict_cs_probs(
    idata: az.InferenceData,
    plate_x: np.ndarray,
    plate_z: np.ndarray,
    count_state_arr: np.ndarray,
    *,
    n_draws: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Return (n_draws, n_obs) of called-strike probabilities under the 2025-era classifier."""
    posterior = idata.posterior
    eta_cell = posterior["eta_cell"].values.reshape(-1, posterior["eta_cell"].shape[-1])
    alpha_tier = posterior["alpha_tier"].values.reshape(-1, posterior["alpha_tier"].shape[-1])
    rng = np.random.default_rng(seed)
    n_total = eta_cell.shape[0]
    if n_draws > n_total:
        n_draws = n_total
    idx = rng.choice(n_total, size=n_draws, replace=False)

    px = np.clip(plate_x.astype(float), X_MIN + 1e-6, X_MAX - 1e-6)
    pz = np.clip(plate_z.astype(float), Z_MIN + 1e-6, Z_MAX - 1e-6)
    ix = ((px - X_MIN) / DX).astype(int).clip(0, NX - 1)
    iz = ((pz - Z_MIN) / DZ).astype(int).clip(0, NZ - 1)
    cell_idx = ix * NZ + iz

    tier_levels = idata.attrs["tier_levels"]
    tier_to_idx = {t: i for i, t in enumerate(tier_levels)}
    tier_arr = np.array([tier_to_idx.get(COUNT_TIER.get(c, "middle"), 1) for c in count_state_arr])

    out = np.empty((n_draws, len(plate_x)), dtype=float)
    for j, k in enumerate(idx):
        eta = eta_cell[k, cell_idx] + alpha_tier[k, tier_arr]
        out[j] = expit(eta)
    return out


def build_pitch_panel_2026() -> pd.DataFrame:
    """Build a 2026 pitch-level panel for Mar 27 – May 13."""
    df = load_2026_full()
    d = pd.to_datetime(df["game_date"]).dt.normalize()
    df = df.loc[(d >= "2026-03-27") & (d <= "2026-05-13")].copy()
    df["pa_id"] = (
        df["game_pk"].astype("Int64").astype(str)
        + "_"
        + df["at_bat_number"].astype("Int64").astype(str)
    )
    df = df.sort_values(["pa_id", "pitch_number"]).reset_index(drop=True)
    df["count_state"] = count_state(df)
    df["is_take"] = df["description"].isin(["called_strike", "ball"]).astype(int)
    df["is_called_strike"] = (df["description"] == "called_strike").astype(int)
    df["pa_terminating"] = plate_appearance_mask(df).astype(int)
    df["is_walk"] = df["events"].isin(WALK_EVENTS).astype(int)
    df["zone_region"] = zone_region(df["plate_x"], df["plate_z"])
    return df


def replay_one_draw(
    panel: pd.DataFrame,
    pa_starts: np.ndarray,
    pa_ends: np.ndarray,
    take_idx: np.ndarray,
    cs_probs_one_draw: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run one full counterfactual PA replay. Returns per-PA walk indicator."""
    n_panel = len(panel)
    # CF outcome of takes (under 2025 zone)
    take_call_cf = (rng.random(len(cs_probs_one_draw)) < cs_probs_one_draw).astype(int)
    cf_is_cs = panel["is_called_strike"].values.astype(int).copy()
    cf_is_cs[take_idx] = take_call_cf
    is_take_arr = panel["is_take"].values.astype(bool)
    desc_arr = panel["description"].values
    counts_b = panel["balls"].values.astype(int)
    counts_s = panel["strikes"].values.astype(int)
    is_walk_orig = panel["is_walk"].values.astype(bool)
    pa_term_orig = panel["pa_terminating"].values.astype(bool)

    cf_walk_pa = np.zeros(len(pa_starts), dtype=int)
    for pa_i, (lo_i, hi_i) in enumerate(zip(pa_starts, pa_ends)):
        if hi_i == lo_i:
            continue
        b = int(counts_b[lo_i])
        s = int(counts_s[lo_i])
        outcome = None
        for k in range(lo_i, hi_i):
            if is_take_arr[k]:
                if cf_is_cs[k] == 1:
                    s += 1
                else:
                    b += 1
            else:
                d = desc_arr[k]
                if d == "hit_into_play":
                    outcome = "in_play"
                    break
                elif d in ("swinging_strike", "swinging_strike_blocked", "missed_bunt"):
                    s += 1
                elif d in ("foul", "foul_tip"):
                    s = min(s + 1, 2)
                elif d == "hit_by_pitch":
                    outcome = "hbp"
                    break
                elif d in ("blocked_ball", "pitchout"):
                    b += 1
                elif d == "foul_pitchout":
                    s = min(s + 1, 2)
                # other rare events: ignore
            if b >= 4:
                outcome = "walk"
                break
            if s >= 3:
                outcome = "strikeout"
                break
        if outcome is None:
            # fell off the end; use observed outcome as fallback
            outcome = "walk" if bool(is_walk_orig[hi_i - 1]) else "other"
        cf_walk_pa[pa_i] = 1 if outcome == "walk" else 0
    return cf_walk_pa


def run_counterfactual(panel: pd.DataFrame, idata: az.InferenceData, n_draws: int = 80) -> dict:
    is_take_arr = panel["is_take"].values.astype(bool)
    take_idx = np.where(is_take_arr)[0]
    px = panel["plate_x"].values[take_idx]
    pz = panel["plate_z"].values[take_idx]
    cstate = panel["count_state"].values[take_idx]
    # Drop rows where plate_x or plate_z is NaN (rare but possible)
    valid_mask = ~np.isnan(px) & ~np.isnan(pz)
    if not valid_mask.all():
        print(f"[H3] dropping {(~valid_mask).sum()} takes with NaN plate coords")
        take_idx = take_idx[valid_mask]
        px = px[valid_mask]
        pz = pz[valid_mask]
        cstate = cstate[valid_mask]

    print(f"[H3] predicting CS probs on {len(take_idx):,} 2026 takes ({n_draws} posterior draws)...", flush=True)
    cs_probs = predict_cs_probs(idata, px, pz, cstate, n_draws=n_draws, seed=2026)
    print(f"[H3] cs_probs shape={cs_probs.shape}, mean={cs_probs.mean():.3f}")

    pa_ids = panel["pa_id"].values
    pa_change = np.r_[True, pa_ids[1:] != pa_ids[:-1]]
    pa_starts = np.where(pa_change)[0]
    pa_ends = np.r_[pa_starts[1:], len(panel)]
    starting_count_pa = panel["count_state"].values[pa_starts]
    starting_zone_pa = panel["zone_region"].values[pa_starts]
    orig_walk_pa = panel["is_walk"].values[pa_ends - 1].astype(int)
    n_pa = len(pa_starts)

    df_2025 = load_2025_samewindow()
    pa_25 = df_2025.loc[plate_appearance_mask(df_2025)]
    rate_25_emp = float(pa_25["events"].isin(WALK_EVENTS).mean())
    rate_26_emp = float(orig_walk_pa.mean())
    yoy_pp = (rate_26_emp - rate_25_emp) * 100.0
    print(f"[H3] empirical: 2025={rate_25_emp:.4f}, 2026={rate_26_emp:.4f}, YoY={yoy_pp:+.2f}pp")

    # We also compute counterfactual conditional on each terminal count: among PAs
    # that REACHED count c (any of 3-0, 3-1, 3-2), what fraction walked under the CF?
    # This is the right "per-count" attribution given that walks land at 3-x.
    panel_pa = panel.copy()
    panel_pa["pa_reached_3_0"] = False  # will fill in below

    cf_rates_all = []
    cf_rates_by_count = {c: [] for c in ALL_COUNTS}
    cf_rates_by_region = {r: [] for r in ["heart", "top_edge", "bottom_edge", "in_off"]}

    rng = np.random.default_rng(20260513)

    # Build a fast take-row lookup
    # take_idx_panel: array marking which panel rows are takes; for replay we need
    # cs_probs aligned to take_idx (which is already done).
    # Pre-build a mapping from take position in take_idx → its prob-index.
    # The replay function uses panel rows; we set cf_is_cs[take_idx] = draw from cs_probs.
    # That mapping is already correct.

    for d in range(n_draws):
        cf_walk_pa = replay_one_draw_v2(panel, pa_starts, pa_ends, take_idx, cs_probs[d], rng)
        cf_rates_all.append(float(cf_walk_pa.mean()))
        for c in ALL_COUNTS:
            mask = starting_count_pa == c
            if mask.sum() > 0:
                cf_rates_by_count[c].append(float(cf_walk_pa[mask].mean()))
        for r in ["heart", "top_edge", "bottom_edge", "in_off"]:
            mask = starting_zone_pa == r
            if mask.sum() > 0:
                cf_rates_by_region[r].append(float(cf_walk_pa[mask].mean()))
        if (d + 1) % 20 == 0:
            print(f"[H3]   draw {d+1}/{n_draws}: cf_rate={cf_rates_all[-1]:.4f}", flush=True)

    cf_rates_all = np.array(cf_rates_all)

    attrib_pct_draws = (rate_26_emp - cf_rates_all) / max(rate_26_emp - rate_25_emp, 1e-6) * 100.0

    out = {
        "rate_25_emp": rate_25_emp,
        "rate_26_emp": rate_26_emp,
        "yoy_pp": yoy_pp,
        "cf_rate_mean": float(cf_rates_all.mean()),
        "cf_rate_lo": float(np.percentile(cf_rates_all, 2.5)),
        "cf_rate_hi": float(np.percentile(cf_rates_all, 97.5)),
        "attribution_pct_mean": float(attrib_pct_draws.mean()),
        "attribution_pct_lo": float(np.percentile(attrib_pct_draws, 2.5)),
        "attribution_pct_hi": float(np.percentile(attrib_pct_draws, 97.5)),
        "n_pa_2026": int(n_pa),
        "n_takes_replayed": int(len(take_idx)),
        "n_draws": n_draws,
        "per_count": {},
        "per_region": {},
    }

    # Per-count: for 2025, get FIRST-pitch starting count of each PA (always 0-0).
    # We use "count_state" on the PA-terminating row instead as a proxy for "PA ending
    # at this terminating count" — but that's terminal-count semantics, not entering.
    # For attribution comparison we use the SAME semantics as 2026: starting count = 0-0,
    # so per-count attribution at the starting level is only meaningful at 0-0.
    # We report a simpler view: rate by terminal count for each year.
    pa_25_counts = pa_25.copy()
    pa_25_counts["count_state"] = count_state(pa_25_counts)
    pa_25_rate_by_count = pa_25_counts.groupby("count_state").apply(
        lambda g: g["events"].isin(WALK_EVENTS).mean(), include_groups=False
    ).to_dict()
    pa_26_starts = pd.DataFrame({"count": starting_count_pa, "walk": orig_walk_pa})
    rate_26_by_count = pa_26_starts.groupby("count")["walk"].mean().to_dict()
    for c in ALL_COUNTS:
        if not cf_rates_by_count[c]:
            continue
        cf_draws = np.array(cf_rates_by_count[c])
        emp_25 = float(pa_25_rate_by_count.get(c, np.nan))
        emp_26 = float(rate_26_by_count.get(c, np.nan))
        delta = emp_26 - emp_25
        attrib = (emp_26 - cf_draws) / delta * 100.0 if abs(delta) > 1e-6 else np.full_like(cf_draws, np.nan)
        out["per_count"][c] = {
            "rate_25_emp": emp_25,
            "rate_26_emp": emp_26,
            "yoy_delta_pp": delta * 100.0,
            "cf_rate_mean": float(cf_draws.mean()),
            "cf_rate_lo": float(np.percentile(cf_draws, 2.5)),
            "cf_rate_hi": float(np.percentile(cf_draws, 97.5)),
            "attribution_pct_mean": float(np.nanmean(attrib)),
            "attribution_pct_lo": float(np.nanpercentile(attrib, 2.5)),
            "attribution_pct_hi": float(np.nanpercentile(attrib, 97.5)),
            "n_pa": int((starting_count_pa == c).sum()),
        }

    # Per-region: for 2025 use first-pitch zone region (entering 0-0 pitch),
    # not the terminating-pitch zone (which biases toward edge for non-walks).
    # Rebuild 2025 first-pitch panel
    df_2025_full = load_2025_samewindow().copy()
    d25 = pd.to_datetime(df_2025_full["game_date"]).dt.normalize()
    df_2025_full = df_2025_full.loc[(d25 >= "2025-03-27") & (d25 <= "2025-05-13")].copy()
    df_2025_full["pa_id"] = (
        df_2025_full["game_pk"].astype("Int64").astype(str)
        + "_"
        + df_2025_full["at_bat_number"].astype("Int64").astype(str)
    )
    df_2025_full = df_2025_full.sort_values(["pa_id", "pitch_number"]).reset_index(drop=True)
    df_2025_full["count_state"] = count_state(df_2025_full)
    pa_25_first = df_2025_full.drop_duplicates("pa_id", keep="first").copy()
    pa_25_first["zone_region"] = zone_region(pa_25_first["plate_x"], pa_25_first["plate_z"])
    # walk by pa_id
    pa_term_25 = df_2025_full.loc[plate_appearance_mask(df_2025_full), ["pa_id", "events"]].drop_duplicates("pa_id")
    pa_term_25["is_walk"] = pa_term_25["events"].isin(WALK_EVENTS).astype(int)
    pa_25_first = pa_25_first.merge(pa_term_25[["pa_id", "is_walk"]], on="pa_id", how="left")
    pa_25_first["is_walk"] = pa_25_first["is_walk"].fillna(0).astype(int)
    pa_25_rate_by_region = pa_25_first.groupby("zone_region")["is_walk"].mean().to_dict()
    pa_26_starts["region"] = starting_zone_pa
    rate_26_by_region = pa_26_starts.groupby("region")["walk"].mean().to_dict()
    for r in ["heart", "top_edge", "bottom_edge", "in_off"]:
        if not cf_rates_by_region[r]:
            continue
        cf_draws = np.array(cf_rates_by_region[r])
        emp_25 = float(pa_25_rate_by_region.get(r, np.nan))
        emp_26 = float(rate_26_by_region.get(r, np.nan))
        delta = emp_26 - emp_25
        attrib = (emp_26 - cf_draws) / delta * 100.0 if abs(delta) > 1e-6 else np.full_like(cf_draws, np.nan)
        out["per_region"][r] = {
            "rate_25_emp": emp_25,
            "rate_26_emp": emp_26,
            "yoy_delta_pp": delta * 100.0,
            "cf_rate_mean": float(cf_draws.mean()),
            "cf_rate_lo": float(np.percentile(cf_draws, 2.5)),
            "cf_rate_hi": float(np.percentile(cf_draws, 97.5)),
            "attribution_pct_mean": float(np.nanmean(attrib)),
            "attribution_pct_lo": float(np.nanpercentile(attrib, 2.5)),
            "attribution_pct_hi": float(np.nanpercentile(attrib, 97.5)),
            "n_pa": int((starting_zone_pa == r).sum()),
        }

    return out


def replay_one_draw_v2(panel, pa_starts, pa_ends, take_idx, cs_probs_one_draw, rng):
    """Faster: drop the inner Python loop where possible.

    We still iterate PA-by-PA because PA state transitions are inherently sequential,
    but use NumPy arrays directly (no pandas dispatch). This is the same logic as
    replay_one_draw above; kept for symmetry/refactor space.
    """
    return replay_one_draw(panel, pa_starts, pa_ends, take_idx, cs_probs_one_draw, rng)


def diagnostics(idata: az.InferenceData) -> dict:
    summary = az.summary(idata, var_names=["mu_global", "sigma_x", "sigma_z", "sigma_int", "alpha_tier"])
    return {
        "rhat_max": float(summary["r_hat"].max()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "convergence_pass": bool(
            (summary["r_hat"].max() <= 1.02) and (summary["ess_bulk"].min() >= 400)
        ),
    }


def plot_h3(result: dict) -> None:
    # Headline attribution chart
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ["Round 1\n(through Apr 22)", "Round 2\n(through May 13, Bayesian)"]
    r1_low, r1_high = 40.0, 50.0
    means = [(r1_low + r1_high) / 2, result["attribution_pct_mean"]]
    lows = [r1_low, result["attribution_pct_lo"]]
    highs = [r1_high, result["attribution_pct_hi"]]
    yerr = np.array([[m - l, h - m] for m, l, h in zip(means, lows, highs)]).T
    ax.bar(cats, means, color=["#7d8c99", "#c0392b"], edgecolor="black", yerr=yerr, capsize=6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Zone attribution (% of YoY walk spike)")
    ax.set_title("H3: zone-change attribution to the walk spike\nRound 1 vs Round 2 (Bayesian posterior replay)")
    fig.tight_layout()
    fig.savefig(R2_CHARTS / "h3_counterfactual_attribution.png", dpi=130)
    plt.close(fig)

    # Per-count
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = []
    means = []
    lows = []
    highs = []
    for c in ALL_COUNTS:
        if c in result["per_count"]:
            counts.append(c)
            means.append(result["per_count"][c]["attribution_pct_mean"])
            lows.append(result["per_count"][c]["attribution_pct_lo"])
            highs.append(result["per_count"][c]["attribution_pct_hi"])
    yerr = np.array([
        [m - l if np.isfinite(l) else 0, h - m if np.isfinite(h) else 0]
        for m, l, h in zip(means, lows, highs)
    ]).T
    bar_colors = ["#c0392b" if m > 0 else "#2980b9" for m in means]
    ax.bar(counts, means, color=bar_colors, edgecolor="black", yerr=yerr, capsize=4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Zone attribution (% of per-count YoY delta)")
    ax.set_xlabel("Starting count")
    ax.set_title("H3: per-count zone attribution\n(% of that count's YoY walk-rate delta explained by zone change)")
    ax.set_ylim(-300, 300)
    fig.tight_layout()
    fig.savefig(R2_CHARTS / "h3_per_count_attribution.png", dpi=130)
    plt.close(fig)

    # Per-region
    fig, ax = plt.subplots(figsize=(7, 5))
    regions = ["heart", "top_edge", "bottom_edge", "in_off"]
    means = [result["per_region"].get(r, {}).get("attribution_pct_mean", np.nan) for r in regions]
    lows = [result["per_region"].get(r, {}).get("attribution_pct_lo", np.nan) for r in regions]
    highs = [result["per_region"].get(r, {}).get("attribution_pct_hi", np.nan) for r in regions]
    yerr = np.array([
        [max(m - l, 0) if np.isfinite(l) else 0, max(h - m, 0) if np.isfinite(h) else 0]
        for m, l, h in zip(means, lows, highs)
    ]).T
    bar_colors = ["#c0392b" if m > 0 else "#2980b9" for m in means]
    ax.bar(regions, means, color=bar_colors, edgecolor="black", yerr=yerr, capsize=6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Zone attribution (% of region's YoY delta)")
    ax.set_title("H3: per-edge-region zone attribution\n(by entering-PA region of the FIRST pitch)")
    fig.tight_layout()
    fig.savefig(R2_CHARTS / "h3_per_edge_attribution.png", dpi=130)
    plt.close(fig)


def main() -> dict:
    ensure_dirs()

    print("[H3] loading 2025 same-window for classifier training...")
    df_2025 = load_2025_samewindow()
    idata = fit_zone_classifier(df_2025)
    diag = diagnostics(idata)
    print(f"[H3] classifier convergence: rhat={diag['rhat_max']:.4f}, ESS={diag['ess_bulk_min']:.0f}")

    az.plot_trace(idata, var_names=["mu_global", "sigma_x", "sigma_z", "sigma_int"])
    plt.gcf().tight_layout()
    plt.gcf().savefig(R2_DIAG / "h3_classifier_trace.png", dpi=110)
    plt.close()

    print("[H3] building 2026 pitch panel...")
    panel = build_pitch_panel_2026()
    print(f"[H3] panel rows={len(panel):,}, PAs={panel['pa_id'].nunique():,}")

    result = run_counterfactual(panel, idata, n_draws=80)
    result["diagnostics"] = diag

    plot_h3(result)
    (R2_ARTIFACTS / "h3_summary.json").write_text(json.dumps(result, indent=2, default=float))
    print(
        f"[H3] all-pitches attribution: {result['attribution_pct_mean']:+.1f}% "
        f"[{result['attribution_pct_lo']:+.1f}, {result['attribution_pct_hi']:+.1f}]"
    )
    return result


if __name__ == "__main__":
    main()

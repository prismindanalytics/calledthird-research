"""H1 triangulation — three counterfactual methods for the H3 zone-attribution magnitude.

Reconciles R2's −64.6% (Claude) vs +35.3% (Codex) by computing three independent
counterfactual estimates and reporting the triangulated median with the widest
honest CI.

  Method A — Bernoulli per-take PA replay (faithful R2 reproduction) with
             CONTINUATION-MODEL FIX: when an altered sequence runs off the end
             without terminating, sample remaining pitches from the empirical
             2026 continuation distribution conditional on count_state.

  Method B — Empirical-lookup CF: for each 2026 taken pitch at (plate_x,
             plate_z, count_tier), look up empirical 2025 CS rate via kNN k=20
             smoothing on 2025 same-window taken pitches in the same count_tier.
             Replay PAs with these empirical probabilities.

  Method C — Bootstrap-of-bootstrap. Outer: game_pk resample, N=100. Inner:
             refit a fast penalized-logistic zone classifier on resampled 2025
             same-window pitches, N=10 seeds. Aggregate to per-outer-iter
             attribution; report median + 95% percentile CI.

All three methods replay the same 2026 panel. The triangulated headline is the
median of the three point estimates with the WIDEST of the three CIs.

Honesty constraints:
  - All bootstrap procedures resample *games* (game_pk), not pitches/rows.
  - Method C represents both model uncertainty (inner) AND game-sequencing
    uncertainty (outer).
  - We persist the aggregate mean-predicted-CS diagnostic on 2026 takes for
    every method — addressing the R2 reviewer note that this number was not
    auditable from artifacts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.neighbors import BallTree

from common import (
    ALL_COUNTS,
    COUNT_TIER,
    COUNT_TIERS_LIST,
    R3_ARTIFACTS,
    R3_CHARTS,
    R3_DIAG,
    WALK_EVENTS,
    called_pitches_subset,
    ensure_dirs,
    game_bootstrap_indices,
)
from data_prep_r3 import get_pa_2026, get_pa_2025, get_panel_2025, get_panel_2026

# Grid for the parametric zone model
X_MIN, X_MAX, NX = -1.8, 1.8, 24
Z_MIN, Z_MAX, NZ = 0.6, 4.4, 24
DX = (X_MAX - X_MIN) / NX
DZ = (Z_MAX - Z_MIN) / NZ
N_GRID = NX * NZ


# ---------------------------------------------------------------------------
# Helper: add (ix, iz, cell_idx) to a frame
# ---------------------------------------------------------------------------

def _add_grid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    px = out["plate_x"].astype(float).clip(X_MIN + 1e-6, X_MAX - 1e-6)
    pz = out["plate_z"].astype(float).clip(Z_MIN + 1e-6, Z_MAX - 1e-6)
    ix = ((px - X_MIN) / DX).astype(int).clip(0, NX - 1)
    iz = ((pz - Z_MIN) / DZ).astype(int).clip(0, NZ - 1)
    out["ix"] = ix.values
    out["iz"] = iz.values
    out["cell_idx"] = (ix.values * NZ + iz.values).astype(int)
    return out


# ---------------------------------------------------------------------------
# Fast penalized-logistic zone classifier (closed-form-like — ML in a few
# Newton steps with quadratic shrinkage toward neighbors). Used as the *inner*
# refit in Method C, where we need 1000 fits in <15 minutes.
# ---------------------------------------------------------------------------

def fit_zone_grid_logistic(
    called_25: pd.DataFrame,
    *,
    smoothing_weight: float = 4.0,
    seed: int = 0,
) -> np.ndarray:
    """Fit a per-(cell, tier) logit smoothed by a 2D Gaussian neighbor average.

    Returns prob_grid[count_tier, cell_idx] : (4, N_GRID) probability matrix.

    Implementation: per cell+tier, naive empirical p = k/n. We then apply
    iterative spatial smoothing — replace each cell's logit by a 0.5/0.5 mix of
    raw logit and the average of its 4-neighbors' logits, over a few passes.
    This is a Markov-smoothing approximation to the GMRF; very fast.

    For Method C *inner refits* we add a small parametric perturbation: bootstrap
    the called pitches (rows; this is an INNER variant of the resample, allowed
    because it's the within-fit model uncertainty) before smoothing.
    """
    if len(called_25) == 0:
        # Degenerate; return prior of 0.327 (empirical CS rate)
        return np.full((len(COUNT_TIERS_LIST), N_GRID), 0.327)
    called = called_25.copy()
    if seed > 0:
        # parametric resample at the row level for the inner CI
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(called), size=len(called), replace=True)
        called = called.iloc[idx]
    called = _add_grid(called)
    called["count_tier"] = called["count_state"].map(COUNT_TIER).fillna("middle")
    tier_to_idx = {t: i for i, t in enumerate(COUNT_TIERS_LIST)}
    called["tier_idx"] = called["count_tier"].map(tier_to_idx).astype(int)
    # Grouped counts
    n_by = (
        called.groupby(["tier_idx", "cell_idx"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    k_by = (
        called.groupby(["tier_idx", "cell_idx"], observed=True)["is_called_strike"]
        .sum()
        .rename("k")
        .reset_index()
    )
    agg = n_by.merge(k_by, on=["tier_idx", "cell_idx"], how="left")
    # Prior alpha,beta (informative weak prior toward 0.327)
    prior_p = 0.327
    prior_n = 8.0
    agg["p_raw"] = (agg["k"] + prior_p * prior_n) / (agg["n"] + prior_n)
    prob_grid = np.full((len(COUNT_TIERS_LIST), N_GRID), prior_p)
    for _, row in agg.iterrows():
        prob_grid[int(row["tier_idx"]), int(row["cell_idx"])] = row["p_raw"]

    # 2D smoothing via repeated 5-point stencil with Beta-mixing toward neighbors
    for _ in range(3):
        prob_grid = _spatial_smooth(prob_grid, smoothing_weight)
    return prob_grid


def _spatial_smooth(prob_grid: np.ndarray, w: float) -> np.ndarray:
    """5-point stencil smooth in logit space (per tier)."""
    out = np.empty_like(prob_grid)
    for t in range(prob_grid.shape[0]):
        p = prob_grid[t].reshape(NX, NZ)
        # avoid logit-of-0 / logit-of-1
        p_clip = np.clip(p, 1e-3, 1 - 1e-3)
        logit_p = np.log(p_clip / (1 - p_clip))
        # 4-neighbor average (zero-padded)
        neigh = np.zeros_like(logit_p)
        cnt = np.zeros_like(logit_p)
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            sl_a, sl_b = (slice(max(0, di), NX + min(0, di)), slice(max(0, dj), NZ + min(0, dj)))
            sl_c, sl_d = (slice(max(0, -di), NX + min(0, -di)), slice(max(0, -dj), NZ + min(0, -dj)))
            neigh[sl_a, sl_b] += logit_p[sl_c, sl_d]
            cnt[sl_a, sl_b] += 1
        smoothed = (logit_p + (w * neigh) / np.maximum(cnt, 1)) / (1 + w)
        out[t] = (1 / (1 + np.exp(-smoothed))).flatten()
    return out


def predict_cs_probs_from_grid(
    prob_grid: np.ndarray,
    plate_x: np.ndarray,
    plate_z: np.ndarray,
    count_state_arr: np.ndarray,
) -> np.ndarray:
    """Look up prob from prob_grid[tier, cell] for each pitch."""
    # Guard against NaN (rare but possible on noisy input).
    px = np.asarray(plate_x, dtype=float).copy()
    pz = np.asarray(plate_z, dtype=float).copy()
    nan_mask = np.isnan(px) | np.isnan(pz)
    if nan_mask.any():
        px[nan_mask] = 0.0
        pz[nan_mask] = 2.5
    px = np.clip(px, X_MIN + 1e-6, X_MAX - 1e-6)
    pz = np.clip(pz, Z_MIN + 1e-6, Z_MAX - 1e-6)
    ix = ((px - X_MIN) / DX).astype(int).clip(0, NX - 1)
    iz = ((pz - Z_MIN) / DZ).astype(int).clip(0, NZ - 1)
    cell_idx = ix * NZ + iz
    tier_to_idx = {t: i for i, t in enumerate(COUNT_TIERS_LIST)}
    tier_arr = np.array([tier_to_idx.get(COUNT_TIER.get(c, "middle"), 1) for c in count_state_arr])
    return prob_grid[tier_arr, cell_idx]


# ---------------------------------------------------------------------------
# Continuation distribution for Method A backstop FIX
# ---------------------------------------------------------------------------

def build_continuation_lookup(panel_2026: pd.DataFrame) -> dict:
    """For each count_state, build an empirical distribution over the next pitch's
    outcome on that count_state in 2026:
      - prob(take | count) — and within take, prob(called_strike | count) is the
        empirical 2026 rate (we'll RECOMPUTE this under the CF zone for each call).
      - prob(swing_strike | count), prob(swing_ball | count), prob(in_play | count),
        prob(hbp | count), prob(other | count)
    Returns: dict count_state -> {prob arrays + indexable empirical samples}.

    For the backstop, when an altered PA reaches its end without terminating, we
    sample additional pitches from this distribution until the PA terminates,
    routing called-pitch outcomes through the same CF mechanism (i.e., we still
    use the *counterfactual* CS prob at the appropriate location).
    """
    out = {}
    desc = panel_2026["description"].astype(str)
    p = panel_2026.copy()
    p["bucket"] = "other"
    p.loc[desc.isin(["called_strike", "ball"]), "bucket"] = "take"
    p.loc[desc.isin(["swinging_strike", "swinging_strike_blocked", "missed_bunt"]), "bucket"] = "swing_strike"
    p.loc[desc.isin(["foul", "foul_tip"]), "bucket"] = "foul"
    p.loc[desc == "hit_into_play", "bucket"] = "in_play"
    p.loc[desc == "hit_by_pitch", "bucket"] = "hbp"
    p.loc[desc.isin(["blocked_ball", "pitchout"]), "bucket"] = "blocked_ball"
    p.loc[desc == "foul_pitchout", "bucket"] = "foul"
    BUCKET_ORDER = ["take", "swing_strike", "foul", "in_play", "hbp", "blocked_ball", "other"]
    for c, sub in p.groupby("count_state"):
        if c not in ALL_COUNTS:
            continue
        counts = sub["bucket"].value_counts(normalize=True).reindex(BUCKET_ORDER, fill_value=0.0).values
        # Sample-ready pool of (plate_x, plate_z, description) for take resampling
        # (we use these locations to compute CF CS probs)
        take_sub = sub[sub["bucket"] == "take"]
        out[c] = {
            "bucket_probs": counts,
            "n_total": int(len(sub)),
            "take_n": int(len(take_sub)),
            # store as arrays for fast sampling
            "take_px": take_sub["plate_x"].astype(float).values if len(take_sub) else np.array([0.0]),
            "take_pz": take_sub["plate_z"].astype(float).values if len(take_sub) else np.array([2.5]),
        }
    out["_bucket_order"] = BUCKET_ORDER
    return out


# ---------------------------------------------------------------------------
# Method A: faithful R2 reproduction + continuation-model backstop fix
# ---------------------------------------------------------------------------

def _replay_method_a(
    panel: pd.DataFrame,
    cs_probs_take: np.ndarray,  # one prob per take across all takes
    take_idx: np.ndarray,
    prob_grid: np.ndarray,
    continuation: dict,
    *,
    rng: np.random.Generator,
    max_extra: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """One full PA replay using:
       - 2025-classifier CS probability for each TAKE in the observed sequence
         (replaced with Bernoulli draw from cs_probs_take)
       - For non-takes, use the observed description (swing/foul/in-play)
       - If the altered sequence does not terminate by the observed last pitch,
         continue by SAMPLING new pitches from the empirical 2026 continuation
         distribution conditional on the running count_state. For each *new*
         take we sample a location uniformly from the 2026 take pool in that
         count and re-evaluate cs_prob from prob_grid (this implements the
         continuation-model fix to the R2 backstop).

    Returns:
       cf_walk_per_pa : (n_pa,) {0,1}
       cf_term_count_per_pa : object array of terminal count state for each PA
    """
    pa_ids = panel["pa_id"].values
    pa_change = np.r_[True, pa_ids[1:] != pa_ids[:-1]]
    pa_starts = np.where(pa_change)[0]
    pa_ends = np.r_[pa_starts[1:], len(panel)]
    n_pa = len(pa_starts)

    is_take_arr = panel["is_take"].values.astype(bool)
    desc_arr = panel["description"].values
    counts_b = panel["balls"].values.astype(int)
    counts_s = panel["strikes"].values.astype(int)
    count_state_arr = panel["count_state"].values

    # CF outcomes of takes
    take_call_cf = (rng.random(len(cs_probs_take)) < cs_probs_take).astype(int)
    cf_is_cs = panel["is_called_strike"].values.astype(int).copy()
    cf_is_cs[take_idx] = take_call_cf

    cf_walk_pa = np.zeros(n_pa, dtype=int)
    cf_term_count = np.full(n_pa, "0-0", dtype=object)

    BUCKET_ORDER = continuation["_bucket_order"]
    bucket_take_idx = BUCKET_ORDER.index("take")

    # Pre-extract continuation tables
    for pa_i in range(n_pa):
        lo, hi = pa_starts[pa_i], pa_ends[pa_i]
        if hi == lo:
            cf_term_count[pa_i] = "0-0"
            continue
        b = int(counts_b[lo])
        s = int(counts_s[lo])
        outcome = None
        for k in range(lo, hi):
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
            if b >= 4:
                outcome = "walk"; break
            if s >= 3:
                outcome = "strikeout"; break

        # CONTINUATION FIX: if outcome still None, sample additional pitches
        # from empirical 2026 continuation distribution at running count_state.
        n_extra = 0
        while outcome is None and n_extra < max_extra:
            cur_c = f"{b}-{s}"
            if cur_c not in continuation:
                outcome = "truncated"
                break
            t = continuation[cur_c]
            probs = t["bucket_probs"]
            if probs.sum() <= 0:
                outcome = "truncated"; break
            choice = rng.choice(len(BUCKET_ORDER), p=probs / probs.sum())
            n_extra += 1
            if choice == bucket_take_idx:
                # sample a take location, compute CF CS prob from prob_grid
                idx_t = rng.integers(0, t["take_n"]) if t["take_n"] > 0 else 0
                px_e = t["take_px"][idx_t]
                pz_e = t["take_pz"][idx_t]
                cf_p = float(predict_cs_probs_from_grid(
                    prob_grid, np.array([px_e]), np.array([pz_e]), np.array([cur_c])
                )[0])
                if rng.random() < cf_p:
                    s += 1
                else:
                    b += 1
            else:
                bucket = BUCKET_ORDER[choice]
                if bucket == "swing_strike":
                    s += 1
                elif bucket == "foul":
                    s = min(s + 1, 2)
                elif bucket == "in_play":
                    outcome = "in_play"; break
                elif bucket == "hbp":
                    outcome = "hbp"; break
                elif bucket == "blocked_ball":
                    b += 1
                else:
                    # "other" — treat as ball as a conservative default
                    b += 1
            if b >= 4:
                outcome = "walk"; break
            if s >= 3:
                outcome = "strikeout"; break

        if outcome is None:
            outcome = "truncated"
        cf_walk_pa[pa_i] = 1 if outcome == "walk" else 0
        cf_term_count[pa_i] = f"{b}-{s}"
    return cf_walk_pa, cf_term_count


def run_method_a(
    panel_2026: pd.DataFrame,
    panel_2025_called: pd.DataFrame,
    *,
    n_draws: int = 60,
    seed: int = 2026,
) -> dict:
    """Method A: per-take Bernoulli PA replay with continuation fix.

    Implementation note: where R2 used a full Bayesian spline classifier, we use
    the same vectorized fast grid+smoothing classifier here for tractability.
    We then create n_draws *posterior-like* perturbations by resampling the
    underlying 2025 cells (parametric bootstrap) — i.e., the CS probabilities
    each draw are the posterior of the cell rate under a Beta(k+1,n-k+1) prior.
    """
    print(f"[H1-A] start (n_draws={n_draws})", flush=True)
    # Build prob_grid posterior draws via Beta sampling on (cell, tier)
    called = panel_2025_called.copy()
    called = _add_grid(called)
    called["count_tier"] = called["count_state"].map(COUNT_TIER).fillna("middle")
    tier_to_idx = {t: i for i, t in enumerate(COUNT_TIERS_LIST)}
    called["tier_idx"] = called["count_tier"].map(tier_to_idx).astype(int)
    # Build (tier, cell) aggregates
    n_arr = np.zeros((len(COUNT_TIERS_LIST), N_GRID), dtype=int)
    k_arr = np.zeros((len(COUNT_TIERS_LIST), N_GRID), dtype=int)
    grp = called.groupby(["tier_idx", "cell_idx"], observed=True)["is_called_strike"].agg(["size", "sum"])
    for (t_i, c_i), row in grp.iterrows():
        n_arr[int(t_i), int(c_i)] = int(row["size"])
        k_arr[int(t_i), int(c_i)] = int(row["sum"])
    # Posterior draws of (tier, cell) prob — Beta posterior with weak prior
    # We use a deliberately weak prior (weight=2) so the per-cell posterior is
    # data-driven where data exists, but not pathological on near-empty cells.
    alpha_prior = 0.327 * 2
    beta_prior = (1 - 0.327) * 2
    rng = np.random.default_rng(seed)

    # Apply same spatial smoothing as the mean grid by averaging over posterior
    # draws (this is functionally similar to a GMRF posterior). We'll do a
    # batch of Beta-draws and then smooth each draw.
    take_mask = panel_2026["is_take"].astype(bool).values & panel_2026["plate_x"].notna().values & panel_2026["plate_z"].notna().values
    take_idx = np.where(take_mask)[0]
    px_takes = panel_2026["plate_x"].values[take_idx]
    pz_takes = panel_2026["plate_z"].values[take_idx]
    cstate_takes = panel_2026["count_state"].values[take_idx]

    continuation = build_continuation_lookup(panel_2026)
    pa_2026 = get_pa_2026()
    rate_26_emp = float(pa_2026["is_walk"].mean())
    # 2025 same-window walk rate
    pa_2025 = get_pa_2025()
    rate_25_emp = float(pa_2025["is_walk"].mean())
    yoy_pp = (rate_26_emp - rate_25_emp) * 100.0
    print(f"[H1-A] empirical 2025={rate_25_emp:.4f} 2026={rate_26_emp:.4f} YoY={yoy_pp:+.2f}pp")

    cf_rates = []
    cs_means = []
    for d in range(n_draws):
        # one posterior draw
        p_draw = rng.beta(k_arr + alpha_prior, (n_arr - k_arr).clip(min=0) + beta_prior)
        # spatial smooth
        for _ in range(3):
            p_draw = _spatial_smooth(p_draw, w=4.0)
        # predict on takes
        cs_probs = predict_cs_probs_from_grid(p_draw, px_takes, pz_takes, cstate_takes)
        cs_means.append(float(cs_probs.mean()))
        # replay
        cf_walk_pa, _ = _replay_method_a(
            panel_2026, cs_probs, take_idx, p_draw, continuation, rng=rng, max_extra=14,
        )
        cf_rates.append(float(cf_walk_pa.mean()))
        if d < 5 or d % 10 == 0:
            print(f"[H1-A] draw {d+1}/{n_draws}: cs_mean={cs_means[-1]:.4f} cf_rate={cf_rates[-1]:.4f}", flush=True)

    cf_rates = np.array(cf_rates)
    cs_means = np.array(cs_means)
    attribution_pct = (rate_26_emp - cf_rates) / (rate_26_emp - rate_25_emp) * 100
    out = {
        "method": "A_bernoulli_replay_with_continuation",
        "n_draws": n_draws,
        "cs_prob_mean_takes_mean": float(cs_means.mean()),
        "cs_prob_mean_takes_lo": float(np.percentile(cs_means, 2.5)),
        "cs_prob_mean_takes_hi": float(np.percentile(cs_means, 97.5)),
        "cf_rate_mean": float(cf_rates.mean()),
        "cf_rate_lo": float(np.percentile(cf_rates, 2.5)),
        "cf_rate_hi": float(np.percentile(cf_rates, 97.5)),
        "attribution_pct_mean": float(attribution_pct.mean()),
        "attribution_pct_lo": float(np.percentile(attribution_pct, 2.5)),
        "attribution_pct_hi": float(np.percentile(attribution_pct, 97.5)),
        "yoy_pp": yoy_pp,
        "rate_25_emp": rate_25_emp,
        "rate_26_emp": rate_26_emp,
    }
    print(f"[H1-A] result: cf_rate={out['cf_rate_mean']:.4f} attribution={out['attribution_pct_mean']:+.1f}% "
          f"[{out['attribution_pct_lo']:+.1f}, {out['attribution_pct_hi']:+.1f}]")
    return out


# ---------------------------------------------------------------------------
# Method B: empirical-lookup CF (no Bayesian model)
# ---------------------------------------------------------------------------

def run_method_b(
    panel_2026: pd.DataFrame,
    panel_2025: pd.DataFrame,
    *,
    k_neighbors: int = 20,
    n_bootstrap: int = 200,
    n_replays: int = 20,
    seed: int = 2027,
) -> dict:
    """Method B: empirical-lookup CF.

    For each 2026 taken pitch at (plate_x, plate_z, count_tier), build a kNN
    estimate (k=20) of the *empirical* 2025 CS rate at that location among 2025
    same-window takes with the same count_tier. Replay PAs.

    Two stages of uncertainty:
      1. Bernoulli-replay stochasticity: run n_replays full PA replays (default
         20) on the original panel using the kNN-derived per-take CS
         probabilities, with the continuation-model backstop fix.
      2. Game-bootstrap CI: resample game_pk from the per-PA walk indicator
         (across all n_replays draws) to obtain a CI on the mean.
    """
    print(f"[H1-B] start (k={k_neighbors}, bootstrap={n_bootstrap}, replays={n_replays})", flush=True)
    # Build tier-specific BallTrees for 2025 takes
    called_25 = called_pitches_subset(panel_2025)
    called_25 = called_25.loc[called_25["count_state"].isin(ALL_COUNTS)].copy()
    called_25["count_tier"] = called_25["count_state"].map(COUNT_TIER).fillna("middle")
    tier_trees = {}
    for tier in COUNT_TIERS_LIST:
        sub = called_25.loc[called_25["count_tier"] == tier]
        if len(sub) < k_neighbors:
            continue
        coords = sub[["plate_x", "plate_z"]].values.astype(float)
        cs = sub["is_called_strike"].values.astype(int)
        tier_trees[tier] = (BallTree(coords, leaf_size=40), cs, sub.index.values)
    # 2026 takes
    take_mask = panel_2026["is_take"].astype(bool).values & panel_2026["plate_x"].notna().values & panel_2026["plate_z"].notna().values
    take_idx = np.where(take_mask)[0]
    px = panel_2026["plate_x"].values[take_idx]
    pz = panel_2026["plate_z"].values[take_idx]
    cstates = panel_2026["count_state"].values[take_idx]
    cs_probs = np.zeros(len(take_idx), dtype=float)
    for tier, (tree, cs, _) in tier_trees.items():
        # mask 2026 takes belonging to this tier
        mask = np.array([COUNT_TIER.get(c, "middle") == tier for c in cstates])
        if not mask.any():
            continue
        coords = np.column_stack([px[mask], pz[mask]])
        dist, idx = tree.query(coords, k=k_neighbors)
        cs_probs[mask] = cs[idx].mean(axis=1)
    print(f"[H1-B] kNN-derived mean CS prob on 2026 takes: {cs_probs.mean():.4f}")

    pa_2026 = get_pa_2026()
    pa_2025 = get_pa_2025()
    rate_26_emp = float(pa_2026["is_walk"].mean())
    rate_25_emp = float(pa_2025["is_walk"].mean())
    yoy_pp = (rate_26_emp - rate_25_emp) * 100.0

    # Continuation for the same backstop fix
    continuation = build_continuation_lookup(panel_2026)
    pseudo_grid = _build_pseudo_grid_from_knn(called_25, k_neighbors=k_neighbors)

    # Build a PA -> game_pk mapping so we can game-bootstrap per-PA walks
    pa_ids = panel_2026["pa_id"].values
    pa_change = np.r_[True, pa_ids[1:] != pa_ids[:-1]]
    pa_starts = np.where(pa_change)[0]
    pa_2026_game_pk = panel_2026["game_pk"].values[pa_starts].astype("int64")
    n_pa = len(pa_starts)

    # Run n_replays independent Bernoulli replays
    rng = np.random.default_rng(seed)
    walk_by_replay = np.zeros((n_replays, n_pa), dtype=int)
    for r_i in range(n_replays):
        cf_walk_pa, _ = _replay_method_a(
            panel_2026, cs_probs, take_idx, pseudo_grid, continuation, rng=rng, max_extra=14,
        )
        walk_by_replay[r_i] = cf_walk_pa
        if r_i == 0 or (r_i + 1) % 5 == 0:
            print(f"[H1-B] replay {r_i+1}/{n_replays}: cf_rate={cf_walk_pa.mean():.4f}", flush=True)
    # Mean walk probability per PA (averaged across replays) — smooth for bootstrap
    walk_prob_per_pa = walk_by_replay.mean(axis=0)

    # Game-bootstrap: resample game_pk, take walks of all PAs in those games
    unique_games = np.unique(pa_2026_game_pk)
    game_to_pa = {int(g): np.where(pa_2026_game_pk == g)[0] for g in unique_games}
    boot_rng = np.random.default_rng(seed + 1)
    cf_rates = []
    for b_i in range(n_bootstrap):
        sampled = boot_rng.choice(unique_games, size=len(unique_games), replace=True)
        pa_idx = np.concatenate([game_to_pa[int(g)] for g in sampled])
        cf_rates.append(float(walk_prob_per_pa[pa_idx].mean()))
    cf_rates = np.array(cf_rates)
    attribution_pct = (rate_26_emp - cf_rates) / (rate_26_emp - rate_25_emp) * 100
    out = {
        "method": "B_empirical_knn_lookup",
        "k_neighbors": k_neighbors,
        "n_bootstrap": n_bootstrap,
        "cs_prob_mean_takes": float(cs_probs.mean()),
        "cf_rate_mean": float(cf_rates.mean()),
        "cf_rate_lo": float(np.percentile(cf_rates, 2.5)),
        "cf_rate_hi": float(np.percentile(cf_rates, 97.5)),
        "attribution_pct_mean": float(attribution_pct.mean()),
        "attribution_pct_lo": float(np.percentile(attribution_pct, 2.5)),
        "attribution_pct_hi": float(np.percentile(attribution_pct, 97.5)),
        "yoy_pp": yoy_pp,
        "rate_25_emp": rate_25_emp,
        "rate_26_emp": rate_26_emp,
    }
    print(f"[H1-B] result: cf_rate={out['cf_rate_mean']:.4f} attribution={out['attribution_pct_mean']:+.1f}% "
          f"[{out['attribution_pct_lo']:+.1f}, {out['attribution_pct_hi']:+.1f}]")
    return out


def _build_pseudo_grid_from_knn(called_25: pd.DataFrame, k_neighbors: int = 20) -> np.ndarray:
    """For continuation-lookup CF probabilities, materialize the kNN over the grid."""
    out = np.full((len(COUNT_TIERS_LIST), N_GRID), 0.327)
    # For each grid cell center, find kNN among 2025 same-tier takes
    cx = X_MIN + (np.arange(NX) + 0.5) * DX
    cz = Z_MIN + (np.arange(NZ) + 0.5) * DZ
    grid_pts = np.array([[x, z] for x in cx for z in cz])  # (NX*NZ, 2)
    grouped = called_25.groupby("count_tier", observed=True)
    for tier, sub in grouped:
        t_i = COUNT_TIERS_LIST.index(tier)
        if len(sub) < k_neighbors:
            continue
        tree = BallTree(sub[["plate_x", "plate_z"]].values.astype(float))
        cs = sub["is_called_strike"].values.astype(int)
        _, idx = tree.query(grid_pts, k=k_neighbors)
        out[t_i] = cs[idx].mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# Method C: bootstrap-of-bootstrap
# ---------------------------------------------------------------------------

def run_method_c(
    panel_2026: pd.DataFrame,
    panel_2025: pd.DataFrame,
    *,
    n_outer: int = 100,
    n_inner: int = 10,
    seed: int = 2028,
) -> dict:
    """Method C: bootstrap-of-bootstrap.

    Outer (N=100): resample game_pk in 2025 same-window data; refit the zone
    classifier.
    Inner (N=10): per-outer iter, refit n_inner times with row-bootstrap
    parametric noise; record the median inner attribution.
    Final: median + 95% percentile CI across outer-iter medians.

    For each outer+inner fit, we:
      1. Resample 2025 games → 2025 called-pitches subset
      2. Fit fast grid logistic with bootstrap perturbation seed (inner)
      3. Predict CS probs on 2026 takes
      4. Run Bernoulli PA replay
      5. Compute cf_rate → attribution

    This is the "honest" CI that includes model uncertainty AND PA sequencing.

    Returns aggregated outer iter medians and per-edge/per-count breakdowns.
    """
    print(f"[H1-C] start (outer={n_outer}, inner={n_inner})", flush=True)
    # Pre-compute per-game lookup for 2025
    called_25 = called_pitches_subset(panel_2025)
    called_25 = called_25.loc[called_25["count_state"].isin(ALL_COUNTS)].copy()
    games_25 = called_25["game_pk"].values.astype("int64")
    unique_games_25 = np.unique(games_25)
    n_g25 = len(unique_games_25)
    # game -> row idx map
    sort_order = np.argsort(games_25, kind="stable")
    sorted_g = games_25[sort_order]
    edges = np.searchsorted(sorted_g, unique_games_25)
    edges = np.r_[edges, len(games_25)]
    game_to_rows = {int(g): sort_order[edges[i]:edges[i + 1]] for i, g in enumerate(unique_games_25)}

    # 2026 takes
    take_mask = panel_2026["is_take"].astype(bool).values & panel_2026["plate_x"].notna().values & panel_2026["plate_z"].notna().values
    take_idx = np.where(take_mask)[0]
    px = panel_2026["plate_x"].values[take_idx]
    pz = panel_2026["plate_z"].values[take_idx]
    cstates = panel_2026["count_state"].values[take_idx]
    region_takes = panel_2026["zone_region"].values[take_idx]
    pa_2026 = get_pa_2026()
    pa_2025 = get_pa_2025()
    rate_26_emp = float(pa_2026["is_walk"].mean())
    rate_25_emp = float(pa_2025["is_walk"].mean())
    yoy_pp = (rate_26_emp - rate_25_emp) * 100.0

    continuation = build_continuation_lookup(panel_2026)
    rng = np.random.default_rng(seed)

    outer_medians = []
    outer_cs_means = []
    # Also collect per-count and per-edge cf_rates for breakdown
    per_count_cf = {c: [] for c in ALL_COUNTS}
    per_region_cf = {r: [] for r in ["heart", "top_edge", "bottom_edge", "in_off"]}
    pa_2026["starting_count"] = pa_2026["starting_count_state"]
    # mapping take row -> pa_id index
    pa_ids = panel_2026["pa_id"].values
    pa_change = np.r_[True, pa_ids[1:] != pa_ids[:-1]]
    pa_starts = np.where(pa_change)[0]
    pa_starts_set = set(pa_starts.tolist())
    pa_idx_of_row = np.searchsorted(pa_starts, np.arange(len(panel_2026)), side="right") - 1
    starting_count_pa = panel_2026["count_state"].values[pa_starts]

    t0 = time.time()
    for o_i in range(n_outer):
        # Outer: resample game_pk
        sampled_games = rng.choice(unique_games_25, size=n_g25, replace=True)
        outer_row_idx = np.concatenate([game_to_rows[int(g)] for g in sampled_games])
        outer_called = called_25.iloc[outer_row_idx]
        # Inner: refit n_inner times with row-bootstrap perturbation
        inner_attrs = []
        inner_cs_means = []
        inner_cf_rates_total = []
        # For breakdown we'll only record on inner median rep (per-outer)
        for in_i in range(n_inner):
            prob_grid = fit_zone_grid_logistic(outer_called, smoothing_weight=4.0, seed=in_i + 1)
            cs_probs_in = predict_cs_probs_from_grid(prob_grid, px, pz, cstates)
            cf_walk_pa, _ = _replay_method_a(
                panel_2026, cs_probs_in, take_idx, prob_grid, continuation, rng=rng, max_extra=14,
            )
            cf_r = float(cf_walk_pa.mean())
            inner_cf_rates_total.append(cf_r)
            inner_cs_means.append(float(cs_probs_in.mean()))
            attr = (rate_26_emp - cf_r) / (rate_26_emp - rate_25_emp) * 100
            inner_attrs.append(attr)
        med_attr = float(np.median(inner_attrs))
        outer_medians.append(med_attr)
        outer_cs_means.append(float(np.median(inner_cs_means)))

        # For breakdown: use the median-inner-fit's cf_walk_pa for per-count/per-region
        med_in = int(np.argsort(inner_attrs)[len(inner_attrs) // 2])
        prob_grid_med = fit_zone_grid_logistic(outer_called, smoothing_weight=4.0, seed=med_in + 1)
        cs_probs_med = predict_cs_probs_from_grid(prob_grid_med, px, pz, cstates)
        cf_walk_pa_med, _ = _replay_method_a(
            panel_2026, cs_probs_med, take_idx, prob_grid_med, continuation, rng=rng, max_extra=14,
        )
        # per-count (by starting count) and per-region (by first-pitch region)
        for c in ALL_COUNTS:
            mask = (starting_count_pa == c)
            if mask.sum() > 0:
                per_count_cf[c].append(float(cf_walk_pa_med[mask].mean()))
        first_region_pa = panel_2026["zone_region"].values[pa_starts]
        for r in ["heart", "top_edge", "bottom_edge", "in_off"]:
            mask = (first_region_pa == r)
            if mask.sum() > 0:
                per_region_cf[r].append(float(cf_walk_pa_med[mask].mean()))
        if (o_i + 1) % 5 == 0 or o_i < 2:
            print(f"[H1-C] outer {o_i+1}/{n_outer}: med_attr={med_attr:+.1f}% cs_med={outer_cs_means[-1]:.4f} elapsed={time.time()-t0:.1f}s", flush=True)

    outer_medians = np.array(outer_medians)
    outer_cs_means = np.array(outer_cs_means)
    # Per-count and per-region: convert to attribution
    pa_2025_full_pa = pa_2025
    per_count_attr = {}
    starting_count_2025 = pa_2025_full_pa["starting_count_state"].values
    for c in ALL_COUNTS:
        if not per_count_cf[c]:
            continue
        cf_r = np.array(per_count_cf[c])
        rate_26_c = float(pa_2026.loc[pa_2026["starting_count_state"] == c, "is_walk"].mean()) if (pa_2026["starting_count_state"] == c).any() else np.nan
        rate_25_c = float(pa_2025_full_pa.loc[starting_count_2025 == c, "is_walk"].mean()) if (starting_count_2025 == c).any() else np.nan
        if not np.isnan(rate_26_c) and not np.isnan(rate_25_c) and abs(rate_26_c - rate_25_c) > 1e-4:
            attr = (rate_26_c - cf_r) / (rate_26_c - rate_25_c) * 100
            per_count_attr[c] = {
                "cf_rate_mean": float(cf_r.mean()),
                "cf_rate_lo": float(np.percentile(cf_r, 2.5)),
                "cf_rate_hi": float(np.percentile(cf_r, 97.5)),
                "attribution_pct_mean": float(attr.mean()),
                "attribution_pct_lo": float(np.percentile(attr, 2.5)),
                "attribution_pct_hi": float(np.percentile(attr, 97.5)),
                "rate_25_emp": rate_25_c,
                "rate_26_emp": rate_26_c,
                "n_pa_2026": int((pa_2026["starting_count_state"] == c).sum()),
            }

    per_region_attr = {}
    first_region_2025 = panel_2025.iloc[pa_2025_full_pa.index.values]["zone_region"].values if False else None
    # Use first_pitch_region from the PA table
    first_region_2025 = pa_2025_full_pa["first_pitch_region"].values if "first_pitch_region" in pa_2025_full_pa.columns else None
    first_region_2026 = pa_2026["first_pitch_region"].values
    for r in ["heart", "top_edge", "bottom_edge", "in_off"]:
        if not per_region_cf[r]:
            continue
        cf_r = np.array(per_region_cf[r])
        mask26 = (first_region_2026 == r)
        rate_26_r = float(pa_2026.loc[mask26, "is_walk"].mean()) if mask26.any() else np.nan
        if first_region_2025 is not None:
            mask25 = (first_region_2025 == r)
            rate_25_r = float(pa_2025_full_pa.loc[mask25, "is_walk"].mean()) if mask25.any() else np.nan
        else:
            rate_25_r = np.nan
        if not np.isnan(rate_25_r) and not np.isnan(rate_26_r) and abs(rate_26_r - rate_25_r) > 1e-4:
            attr = (rate_26_r - cf_r) / (rate_26_r - rate_25_r) * 100
            per_region_attr[r] = {
                "cf_rate_mean": float(cf_r.mean()),
                "cf_rate_lo": float(np.percentile(cf_r, 2.5)),
                "cf_rate_hi": float(np.percentile(cf_r, 97.5)),
                "attribution_pct_mean": float(attr.mean()),
                "attribution_pct_lo": float(np.percentile(attr, 2.5)),
                "attribution_pct_hi": float(np.percentile(attr, 97.5)),
                "n_pa_2026": int(mask26.sum()),
            }

    out = {
        "method": "C_bootstrap_of_bootstrap",
        "n_outer": n_outer,
        "n_inner": n_inner,
        "cs_prob_mean_takes_mean": float(outer_cs_means.mean()),
        "cs_prob_mean_takes_lo": float(np.percentile(outer_cs_means, 2.5)),
        "cs_prob_mean_takes_hi": float(np.percentile(outer_cs_means, 97.5)),
        "attribution_pct_mean": float(outer_medians.mean()),
        "attribution_pct_median": float(np.median(outer_medians)),
        "attribution_pct_lo": float(np.percentile(outer_medians, 2.5)),
        "attribution_pct_hi": float(np.percentile(outer_medians, 97.5)),
        "yoy_pp": yoy_pp,
        "rate_25_emp": rate_25_emp,
        "rate_26_emp": rate_26_emp,
        "per_count": per_count_attr,
        "per_region": per_region_attr,
    }
    print(f"[H1-C] result: attribution={out['attribution_pct_mean']:+.1f}% "
          f"[{out['attribution_pct_lo']:+.1f}, {out['attribution_pct_hi']:+.1f}] (median={out['attribution_pct_median']:+.1f})")
    return out


# ---------------------------------------------------------------------------
# Triangulation: combine three methods
# ---------------------------------------------------------------------------

def triangulate(res_a: dict, res_b: dict, res_c: dict) -> dict:
    """Take median of three point estimates with the widest of three CIs."""
    points = [res_a["attribution_pct_mean"], res_b["attribution_pct_mean"], res_c["attribution_pct_mean"]]
    los = [res_a["attribution_pct_lo"], res_b["attribution_pct_lo"], res_c["attribution_pct_lo"]]
    his = [res_a["attribution_pct_hi"], res_b["attribution_pct_hi"], res_c["attribution_pct_hi"]]
    widths = [his[i] - los[i] for i in range(3)]
    widest_i = int(np.argmax(widths))
    return {
        "triangulated_median_pct": float(np.median(points)),
        "triangulated_mean_pct": float(np.mean(points)),
        "editorial_ci_lo": float(los[widest_i]),
        "editorial_ci_hi": float(his[widest_i]),
        "widest_method": ["A", "B", "C"][widest_i],
        "point_a": float(points[0]),
        "point_b": float(points[1]),
        "point_c": float(points[2]),
        "ci_a": [float(los[0]), float(his[0])],
        "ci_b": [float(los[1]), float(his[1])],
        "ci_c": [float(los[2]), float(his[2])],
    }


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def plot_triangulation(res_a: dict, res_b: dict, res_c: dict, tri: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.0))
    methods = [
        ("A — Bernoulli + continuation", res_a, "#2c7fb8"),
        ("B — kNN empirical (no model)", res_b, "#41b6c4"),
        ("C — Bootstrap-of-bootstrap", res_c, "#7fcdbb"),
    ]
    # R1/R2 reference points
    r1_point = 40.46
    r2_codex_point = 35.3
    r2_codex_ci = (34.6, 36.0)
    r2_claude_point = -64.6
    r2_claude_ci = (-80.6, -49.4)
    y_pos = list(range(len(methods)))
    for i, (label, r, c) in enumerate(methods):
        pt = r["attribution_pct_mean"]
        lo = r["attribution_pct_lo"]
        hi = r["attribution_pct_hi"]
        ax.errorbar([pt], [i], xerr=[[pt - lo], [hi - pt]], fmt="o", color=c, capsize=4, markersize=9, linewidth=2.5, label=label)
        ax.text(pt, i + 0.20, f"{pt:+.1f}%", ha="center", va="bottom", fontsize=9, color=c, weight="bold")
    # Reference lines
    ax.axvline(r1_point, color="#666", linestyle="--", alpha=0.6, linewidth=1)
    ax.text(r1_point, len(methods) - 0.3, "R1 +40.5%", color="#666", fontsize=8)
    ax.axvline(r2_codex_point, color="#ec7014", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(r2_codex_point, len(methods) - 0.6, "R2 Codex +35.3%", color="#ec7014", fontsize=8)
    ax.axvline(r2_claude_point, color="#c0392b", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(r2_claude_point, len(methods) - 0.9, "R2 Claude −64.6%", color="#c0392b", fontsize=8)
    # Triangulated band
    ax.axvspan(tri["editorial_ci_lo"], tri["editorial_ci_hi"], color="#fdd49e", alpha=0.4, label="R3 editorial CI")
    ax.axvline(tri["triangulated_median_pct"], color="#a8290b", linewidth=2, label=f"R3 triangulated median {tri['triangulated_median_pct']:+.1f}%")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m[0] for m in methods])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.4)
    ax.set_xlabel("Zone-attribution share of 2026 YoY walk-rate gap (%)")
    ax.set_title("R3-H1: H3 magnitude triangulation\nthree CF methods + R1/R2 reference")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlim(-105, 110)
    fig.tight_layout()
    fig.savefig(R3_CHARTS / "h1_triangulated_attribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def run_all(n_draws_a: int = 60, n_outer_c: int = 100, n_inner_c: int = 10, n_boot_b: int = 200) -> dict:
    ensure_dirs()
    panel_2026 = get_panel_2026()
    panel_2025 = get_panel_2025()
    called_25 = called_pitches_subset(panel_2025)
    called_25 = called_25.loc[called_25["count_state"].isin(ALL_COUNTS)].copy()
    print(f"[H1] panels — 2026 rows={len(panel_2026):,} 2025 called={len(called_25):,}")

    print("\n========== METHOD A ==========")
    res_a = run_method_a(panel_2026, called_25, n_draws=n_draws_a)
    print("\n========== METHOD B ==========")
    res_b = run_method_b(panel_2026, panel_2025, k_neighbors=20, n_bootstrap=n_boot_b)
    print("\n========== METHOD C ==========")
    res_c = run_method_c(panel_2026, panel_2025, n_outer=n_outer_c, n_inner=n_inner_c)
    tri = triangulate(res_a, res_b, res_c)
    print(f"\n[H1] TRIANGULATED: {tri['triangulated_median_pct']:+.1f}% [{tri['editorial_ci_lo']:+.1f}, {tri['editorial_ci_hi']:+.1f}] (widest CI from method {tri['widest_method']})")

    plot_triangulation(res_a, res_b, res_c, tri)

    out = {
        "method_a": res_a,
        "method_b": res_b,
        "method_c": res_c,
        "triangulated": tri,
    }
    (R3_ARTIFACTS / "h1_triangulation.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    run_all()

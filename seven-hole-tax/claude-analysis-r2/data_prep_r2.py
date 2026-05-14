"""data_prep_r2.py — Round 2 data prep.

Reuses Round 1 cached corpus (`taken_pitches.parquet`, `challenges_full.parquet`)
and augments with:

1. Round 1 H3 GAM posterior-predictive expected called-strike rate per pitch
   (so H5 can compute per-batter expected vs actual).
2. Borderline subset (|edge_distance_ft| <= 0.3) with batter chase-rate join.
3. Per-batter name lookup (from challenges JSON) for prettier reporting.

Public API:
  load_borderline()          -> pd.DataFrame   borderline taken pitches with all controls
  load_challenges_full()     -> pd.DataFrame   per-challenge with controls (Round 1)
  load_chase_rate()          -> pd.DataFrame   2025 chase per batter
  load_batter_names()        -> dict[int,str]  batter_id -> name from challenges
  fit_round1_h3_gam()        -> arviz.InferenceData, info
                               (Round 1 GAM, no batter-specific features)
                               cached via pickle to avoid re-fitting
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Reuse Round 1 cached files
ROUND1 = ROOT.parent / "claude-analysis"
ROUND1_CACHE = ROUND1 / "cache"
ROUND1_TAKEN = ROUND1_CACHE / "taken_pitches.parquet"
ROUND1_CHALLENGES = ROUND1_CACHE / "challenges_full.parquet"

ROUND1_CHALLENGES_JSON_NEW = ROUND1 / "data/all_challenges_apr15_may04.json"
ROUND1_CHALLENGES_JSON_OLD = Path("/Users/haohu/Documents/GitHub/calledthird/research/team-challenge-iq/data/all_challenges_detail.json")

# Round 2 outputs
CHASE_OUT = DATA / "batter_chase_rate_2025.parquet"
H3_GAM_CACHE = CACHE / "h3_gam_round1.pkl"


# ----- Loaders --------------------------------------------------------------

def load_borderline() -> pd.DataFrame:
    """Load Round-1 cached borderline taken pitches (|edge_distance_ft| <= 0.3)."""
    if not ROUND1_TAKEN.exists():
        raise FileNotFoundError(
            f"Round 1 taken_pitches cache missing at {ROUND1_TAKEN}. "
            f"Run `cd ../claude-analysis && python data_prep.py` first."
        )
    df = pd.read_parquet(ROUND1_TAKEN)
    df = df[df["edge_distance_ft"].abs() <= 0.3].copy()
    df = df.dropna(subset=["plate_x", "plate_z", "balls", "strikes", "lineup_spot", "umpire"])
    df["lineup_spot"] = df["lineup_spot"].astype(int)
    df = df[df["lineup_spot"].between(1, 9)]
    df["count_state"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    df["framing_tier"] = df["framing_tier"].fillna("unknown").astype(str)
    df["fame_quartile"] = df["fame_quartile"].fillna(2.5)
    df["umpire"] = df["umpire"].astype(str)
    df = df.reset_index(drop=True)
    return df


def load_challenges_full() -> pd.DataFrame:
    """Load Round-1 cached challenges_full (per-challenge with controls)."""
    if not ROUND1_CHALLENGES.exists():
        raise FileNotFoundError(
            f"Round 1 challenges_full cache missing at {ROUND1_CHALLENGES}."
        )
    df = pd.read_parquet(ROUND1_CHALLENGES)
    return df


def load_chase_rate() -> pd.DataFrame:
    if not CHASE_OUT.exists():
        from chase_rate_build import build
        return build()
    return pd.read_parquet(CHASE_OUT)


def load_batter_names() -> dict[int, str]:
    names: dict[int, str] = {}
    for path in (ROUND1_CHALLENGES_JSON_OLD, ROUND1_CHALLENGES_JSON_NEW):
        if path.exists():
            with open(path) as f:
                rows = json.load(f)
            for r in rows:
                bid = r.get("batter_id")
                bn = r.get("batter_name")
                if bid is not None and bn:
                    names.setdefault(int(bid), str(bn))
    return names


# ----- Round-1 H3 GAM (re-fit and cache) ------------------------------------
# We re-fit a slim version of the Round-1 H3 GAM here so H5 can compute
# posterior-predictive expected called-strike rate per pitch using a model
# that does NOT include batter-specific features. The Round-1 H3 GAM lives in
# claude-analysis/bayes_gam_h3.py — we mirror the spec but cache the idata
# locally so analyze.py can load it once.

def fit_round1_h3_gam(force: bool = False, n_draws: int = 1000, n_tune: int = 1500):
    """Fit (or load) the Round 1 H3 GAM. Returns (idata, info, df).

    Uses spot-3 as baseline. Borderline only. Random effects on pitcher,
    catcher, umpire. NO batter random effect (so per-hitter residuals
    measure pure deviation from the league model).
    """
    if H3_GAM_CACHE.exists() and not force:
        with open(H3_GAM_CACHE, "rb") as f:
            blob = pickle.load(f)
        return blob["idata"], blob["info"], blob["df"]

    # Lazy-import Round 1's GAM builder so this module doesn't require pymc at top level
    sys.path.insert(0, str(ROUND1))
    import bayes_gam_h3 as r1_gam  # type: ignore
    import pymc as pm
    df = load_borderline()
    print(f"[r2 H3 GAM] Fitting on n={len(df):,} borderline taken pitches")
    model, info = r1_gam._build_model(df, n_knots=5)
    with model:
        idata = pm.sample(
            draws=n_draws, tune=n_tune, chains=4, cores=4,
            target_accept=0.95, progressbar=False, random_seed=2026,
            init="adapt_diag",
        )
    # cache
    with open(H3_GAM_CACHE, "wb") as f:
        pickle.dump({"idata": idata, "info": info, "df": df}, f)
    return idata, info, df


def h3_pp_expected_logits(idata, info, df_eval=None, n_keep: int = 600, rng=None):
    """Compute posterior-predictive expected called-strike probability per pitch.

    Returns:
      probs_mean    (n_obs,)            posterior-mean P(called_strike) per pitch
      probs_samples (n_keep, n_obs)     posterior draws of P(called_strike)
    """
    posterior = idata.posterior
    if rng is None:
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

    B = info["B"]
    spot_idx = info["spot_idx"]
    count_idx = info["count_idx"]
    framing_idx = info["framing_idx"]
    pitcher_idx = info["pitcher_idx"]
    catcher_idx = info["catcher_idx"]
    umpire_idx = info["umpire_idx"]

    # spot effect: 0 when spot_idx==0 (baseline=spot 3), else b_spot[:, spot_idx-1]
    spot_eff = np.where(spot_idx[None, :] == 0, 0.0, b_spot[:, np.maximum(spot_idx - 1, 0)])
    count_eff = np.where(count_idx[None, :] == 0, 0.0, b_count[:, np.maximum(count_idx - 1, 0)])
    framing_eff = np.where(framing_idx[None, :] == 0, 0.0, b_framing[:, np.maximum(framing_idx - 1, 0)])

    eta = (
        intercept[:, None]
        + spot_eff
        + count_eff
        + framing_eff
        + b_basis @ B.T
        + u_pitcher[:, pitcher_idx]
        + u_catcher[:, catcher_idx]
        + u_umpire[:, umpire_idx]
    )
    probs = 1.0 / (1.0 + np.exp(-eta))
    return probs.mean(axis=0), probs


if __name__ == "__main__":
    df = load_borderline()
    print(f"borderline: {len(df):,}")
    print(df["lineup_spot"].value_counts().sort_index())
    chase = load_chase_rate()
    print(f"chase rate batters: {len(chase):,}")
    names = load_batter_names()
    print(f"batter names: {len(names):,}")

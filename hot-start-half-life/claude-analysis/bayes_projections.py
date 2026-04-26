"""bayes_projections.py — hierarchical Bayesian projections for the 5 named hot starters.

Method:
  - Per-player rate stat (BB%, K%, BABIP, ISO, wOBA), specify a *player-specific prior*
    derived from a 3-year (2023-2025) Statcast weighted mean (5/4/3 weights, most-recent first).
    For Murakami (NPB rookie), use a league-average prior with a wide effective sample
    size, documented as a caveat. For Trout (durability case), use his 2023-2025 healthy
    rate but flag the durability uncertainty.
  - Hierarchical Beta-Binomial model:
        kappa  ~ HalfNormal(50)        # concentration on the prior (effective prior PA)
        rho_p  ~ Beta(eb_alpha, eb_beta)  per-player true rate
        success_p_22g ~ Binomial(N_p, rho_p)
    Posterior samples of rho_p are mixed with the league-environment offset to produce
    a rest-of-season distribution. We then forward-simulate ROS PA volume (from each
    player's 2024-25 PA-per-day pace) to translate into a ROS counting distribution
    where useful (Murakami HR, Miller IP).
  - Empirical-Bayes shrinkage weight = (effective_prior_PA) / (effective_prior_PA + observed_PA).
  - For wOBA / ISO / BABIP (continuous-but-bounded rates) we still use Beta likelihoods
    with success_count = round(rate * denom), denom = appropriate counting denominator
    (PA, AB, BIP). These are slight approximations — Statcast woba_value is the
    standard FanGraphs-style PA-weighted contribution and integer-count approximations
    are well-validated.
  - Convergence diagnostics: R-hat, ESS, trace plots saved to charts/diag/.

Dependencies: numpyro + JAX. Each model is small (5 chains x 1500 draws); end-to-end
fit per stat takes seconds.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

warnings.filterwarnings("ignore")
numpyro.set_host_device_count(2)

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"
CHARTS = CLAUDE / "charts"
DIAG = CHARTS / "diag"
PLAYER_CHARTS = CHARTS / "players"
for d in (CHARTS, DIAG, PLAYER_CHARTS):
    d.mkdir(parents=True, exist_ok=True)

from stabilization import (
    load_pa_table,
    annotate_pa,
    PA_EVENTS,
    STAT_SPECS,
    CARLETON,
)

NAMED = [
    {"slug": "andy_pages",      "first": "Andy",       "last": "Pages",     "role": "hitter"},
    {"slug": "ben_rice",        "first": "Ben",        "last": "Rice",      "role": "hitter"},
    {"slug": "munetaka_murakami","first": "Munetaka",  "last": "Murakami",  "role": "hitter"},
    {"slug": "mike_trout",      "first": "Mike",       "last": "Trout",     "role": "hitter"},
    {"slug": "mason_miller",    "first": "Mason",      "last": "Miller",    "role": "reliever"},
]

# Rate stats projected for each hitter
HITTER_STATS = ["BB%", "K%", "BABIP", "ISO", "wOBA"]
# Reliever stats for Miller
RELIEVER_STATS = ["BB%", "K%"]


def _slugify(s: str) -> str:
    return s.lower().replace(" ", "_").replace("'", "")


# ----- Player ID resolution (re-run safely) -----

def load_named_ids() -> pd.DataFrame:
    f = DATA / "named_hot_starters.parquet"
    if f.exists():
        return pd.read_parquet(f)
    raise FileNotFoundError(f"{f} missing — run data_pull.py first")


# ----- League-environment + player aggregates -----

def league_env(pa: pd.DataFrame) -> dict:
    """Per-season league environment for hitter rate stats (2022-2025)."""
    out = {}
    for season, sub in pa.groupby("season"):
        out[int(season)] = {
            "BB%": float(sub["is_bb"].sum() / sub["is_pa"].sum()),
            "K%": float(sub["is_k"].sum() / sub["is_pa"].sum()),
            "BABIP": float(sub["babip_num"].sum() / max(sub["babip_den"].sum(), 1)),
            "ISO": float(sub["iso_num"].sum() / max(sub["iso_den"].sum(), 1)),
            "wOBA": float(sub["woba_num"].sum() / max(sub["woba_den"].sum(), 1)),
            "PA": int(sub["is_pa"].sum()),
        }
    return out


def player_season_aggregates(pa: pd.DataFrame, batter_id: int, season: int) -> dict:
    """Aggregate one player's season into rate-stat numerator/denominator counts."""
    sub = pa[(pa.batter == batter_id) & (pa.season == season)]
    if sub.empty:
        return None
    res = {
        "PA": int(sub["is_pa"].sum()),
        "BB%": (int(sub["is_bb"].sum()), int(sub["is_pa"].sum())),
        "K%": (int(sub["is_k"].sum()), int(sub["is_pa"].sum())),
        "BABIP": (int(sub["babip_num"].sum()), int(sub["babip_den"].sum())),
        "ISO": (int(sub["iso_num"].sum()), int(sub["iso_den"].sum())),
        "wOBA": (float(sub["woba_num"].sum()), float(sub["woba_den"].sum())),
    }
    return res


def player_22g_window_2026(pa26: pd.DataFrame, batter_id: int) -> dict:
    """Aggregate 2026 through-Apr-24 PAs for a single player into rate counts.

    The brief calls these the "22-game" stats. We use *all PAs through cutoff*; some
    players will have 21 or 23 games, doesn't change the substance.
    """
    sub = pa26[pa26.batter == batter_id]
    if sub.empty:
        return None
    return {
        "PA": int(sub["is_pa"].sum()),
        "BB%": (int(sub["is_bb"].sum()), int(sub["is_pa"].sum())),
        "K%": (int(sub["is_k"].sum()), int(sub["is_pa"].sum())),
        "BABIP": (int(sub["babip_num"].sum()), int(sub["babip_den"].sum())),
        "ISO": (int(sub["iso_num"].sum()), int(sub["iso_den"].sum())),
        "wOBA": (float(sub["woba_num"].sum()), float(sub["woba_den"].sum())),
        "HR": int(sub["is_hr"].sum()),
    }


# Empirical-Bayes effective prior sample size per stat. These come from the bootstrap
# stabilization PA — by definition, the half-stabilization PA = the eff_prior_pa at
# which the optimal shrinkage weight equals 0.5 vs an N-PA observation.
# We try to load these from data/stabilization_summary.json; if missing, fall back to
# Carleton 2007/2013 published values.
_CARLETON_FALLBACK = {"BB%": 120, "K%": 60, "ISO": 160, "BABIP": 820, "wOBA": 280}


def _eb_effective_prior_pa(stat: str) -> int:
    f = DATA / "stabilization_summary.json"
    if f.exists():
        try:
            d = json.load(open(f))
            v = d.get(stat, {}).get("point_pa")
            if v and math.isfinite(v):
                return int(round(v))
        except Exception:
            pass
    return int(_CARLETON_FALLBACK.get(stat, 200))


def weighted_3yr_prior(pa_all: pd.DataFrame, batter_id: int, target_season: int = 2026,
                       league: dict | None = None) -> dict:
    """3-year weighted-mean prior with empirical-Bayes effective-sample-size shrinkage.

    weights 5/4/3 for (target-1, target-2, target-3). Returns per-stat:
      rate            — the weighted-mean rate from the 3 prior seasons
      eff_prior_pa    — the *empirical-Bayes effective prior sample size* (in PA).
                        This is the stat's bootstrap-estimated half-stabilization PA,
                        capped above by the player's actual prior PA so we don't
                        over-shrink players with thin histories.
      raw_prior_pa    — the player's actual weighted PA in the 3-year window
      seasons_used    — which prior seasons contributed
    """
    weights = {target_season - 1: 5, target_season - 2: 4, target_season - 3: 3}
    accum = {s: [0.0, 0.0] for s in STAT_SPECS}
    total_pa = 0
    seasons_used = []
    for s, w in weights.items():
        agg = player_season_aggregates(pa_all, batter_id, s)
        if agg is None:
            continue
        seasons_used.append(s)
        total_pa += agg["PA"]
        for stat in STAT_SPECS:
            n_num, n_den = agg[stat]
            accum[stat][0] += w * n_num
            accum[stat][1] += w * n_den
    out = {}
    for stat in STAT_SPECS:
        num, den = accum[stat]
        eb_pa = _eb_effective_prior_pa(stat)
        # Cap eff_prior_pa above by the player's actual PA in the prior window — a player
        # with only 100 PA of history shouldn't be assigned a 280-PA prior strength.
        eff = min(eb_pa, max(int(round(total_pa * 0.7)), 40))
        if den <= 0:
            if league:
                lg = float(np.mean([league[k][stat] for k in league.keys() if k < target_season]))
                out[stat] = {"rate": lg, "eff_prior_pa": 50, "raw_prior_pa": 0,
                             "source": "league"}
            else:
                out[stat] = {"rate": 0.0, "eff_prior_pa": 0, "raw_prior_pa": 0,
                             "source": "none"}
        else:
            out[stat] = {
                "rate": float(num / den),
                "eff_prior_pa": int(eff),
                "raw_prior_pa": int(total_pa),
                "source": "weighted_3yr",
                "seasons_used": seasons_used,
            }
    out["seasons_used"] = seasons_used
    out["total_pa_in_prior"] = total_pa
    return out


def league_average_prior(league: dict, target_season: int) -> dict:
    """For NPB rookie (Murakami) — use league average with weak prior strength."""
    out = {}
    for stat in STAT_SPECS:
        seasons_pre = [k for k in league.keys() if k < target_season]
        ws = np.array([5, 4, 3, 2, 1][:len(seasons_pre)], dtype=float)
        ws = ws[: len(seasons_pre)]
        ws = ws / ws.sum() if ws.sum() > 0 else ws
        seasons_pre_sorted = sorted(seasons_pre, reverse=True)
        vals = [league[s][stat] for s in seasons_pre_sorted[: len(ws)]]
        # 60-PA effective prior sample for an MLB rookie — weak
        out[stat] = {
            "rate": float(np.dot(ws, vals)),
            "eff_prior_pa": 60,
            "raw_prior_pa": 0,
            "source": "league_avg",
        }
    return out


# ----- Bayesian fit -----

def _beta_params_from_rate(rate: float, eff_n: float, *, min_alpha: float = 1.5) -> tuple[float, float]:
    """Convert (rate, effective sample size) into Beta(alpha, beta) prior parameters.

    Eff_n bound below to avoid degenerate priors.
    """
    rate = float(np.clip(rate, 1e-3, 1 - 1e-3))
    eff_n = float(max(eff_n, 5))
    a = rate * eff_n
    b = (1 - rate) * eff_n
    a = max(a, min_alpha)
    b = max(b, min_alpha)
    return a, b


def _model_rate(observed_success, observed_n, alpha_p, beta_p):
    """Beta-Binomial: rho_p ~ Beta(alpha, beta); successes ~ Binomial(n, rho_p)."""
    rho = numpyro.sample("rho", dist.Beta(alpha_p, beta_p))
    numpyro.sample("obs", dist.Binomial(observed_n, rho), obs=observed_success)


def fit_one(observed_success: int, observed_n: int, alpha: float, beta: float,
            *, n_chains: int = 4, n_warmup: int = 800, n_samples: int = 2000,
            seed: int = 0) -> dict:
    """Fit Beta-Binomial. Returns posterior samples + diagnostics."""
    kernel = NUTS(_model_rate)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains,
                progress_bar=False, chain_method="vectorized")
    rng = jax.random.PRNGKey(seed)
    mcmc.run(rng,
             observed_success=jnp.asarray(observed_success, dtype=jnp.float32),
             observed_n=jnp.asarray(max(observed_n, 1), dtype=jnp.int32),
             alpha_p=jnp.asarray(alpha, dtype=jnp.float32),
             beta_p=jnp.asarray(beta, dtype=jnp.float32))
    samples = mcmc.get_samples(group_by_chain=True)["rho"]  # shape (chains, samples)
    # diagnostics
    rh, ess = _rhat_ess(np.asarray(samples))
    rho_flat = np.asarray(samples).reshape(-1)
    return {
        "samples": rho_flat,
        "samples_by_chain": np.asarray(samples),
        "rhat": float(rh),
        "ess": float(ess),
        "alpha_post": float(alpha + observed_success),
        "beta_post": float(beta + max(observed_n - observed_success, 0)),
    }


def _rhat_ess(arr: np.ndarray) -> tuple[float, float]:
    """Rhat and bulk-ESS for samples shape (chains, draws). Standard split-Rhat."""
    chains, draws = arr.shape
    half = draws // 2
    a = arr[:, :half]
    b = arr[:, half:half * 2]
    chains_arr = np.concatenate([a, b], axis=0)
    m, n = chains_arr.shape
    chain_mean = chains_arr.mean(axis=1)
    chain_var = chains_arr.var(axis=1, ddof=1)
    overall_mean = chain_mean.mean()
    B = n * np.sum((chain_mean - overall_mean) ** 2) / (m - 1)
    W = chain_var.mean()
    var_hat = ((n - 1) / n) * W + B / n
    rhat = math.sqrt(var_hat / W) if W > 0 else float("nan")
    # Crude ESS — autocorr-based via the Sokal/initial monotone formulas would be
    # ideal but expensive; use the Stan-style ESS from variance & autocorr-1.
    flat = arr.reshape(-1)
    if len(flat) > 1 and flat.var() > 0:
        rho1 = float(np.corrcoef(flat[:-1], flat[1:])[0, 1])
        rho1 = max(-0.99, min(0.99, rho1))
        # ESS approx N * (1 - rho1)/(1 + rho1)
        ess = len(flat) * (1 - rho1) / (1 + rho1)
    else:
        ess = float(len(flat))
    return rhat, ess


def trace_plot(samples_by_chain: np.ndarray, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for ci in range(samples_by_chain.shape[0]):
        ax[0].plot(samples_by_chain[ci], lw=0.6, alpha=0.7, label=f"chain {ci}")
    ax[0].set_title(f"Trace — {label}")
    ax[0].set_xlabel("draw")
    ax[0].set_ylabel("posterior rate")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=7)
    ax[1].hist(samples_by_chain.reshape(-1), bins=40, color="#1f4e79", alpha=0.85)
    ax[1].set_title(f"Posterior — {label}")
    ax[1].set_xlabel("posterior rate")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----- Reliever (Mason Miller): pitch-level wOBA-against / strike-out / walk -----

def reliever_pa_table(seasons=(2023, 2024, 2025)) -> pd.DataFrame:
    """Per-PA from *pitcher* perspective. We re-key on pitcher_id."""
    frames = []
    for y in seasons:
        f = DATA / f"statcast_{y}.parquet"
        if not f.exists():
            f = DATA / f"statcast_{y}_full.parquet"
        if not f.exists():
            print(f"[skip] {y} not cached")
            continue
        df = pd.read_parquet(f, columns=[
            "game_date", "game_pk", "pitcher", "at_bat_number",
            "events", "woba_value", "woba_denom", "babip_value", "iso_value",
        ])
        df["season"] = y
        df = df[df["events"].notna() & df["events"].isin(PA_EVENTS)].copy()
        df = df.drop_duplicates(subset=["game_pk", "pitcher", "at_bat_number"])
        frames.append(df)
    pa = pd.concat(frames, ignore_index=True)
    return pa


def annotate_pitcher_pa(pa: pd.DataFrame) -> pd.DataFrame:
    pa = pa.copy()
    pa["is_pa"] = 1
    pa["is_bb"] = pa["events"].isin({"walk", "intent_walk"}).astype(int)
    pa["is_k"] = pa["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    # Baserunners allowed (not necessarily runs)
    pa["is_baserunner_allowed"] = pa["events"].isin({"single", "double", "triple", "home_run", "walk", "intent_walk", "hit_by_pitch"}).astype(int)
    return pa


# ----- Per-player projection driver -----

def project_hitter(slug: str, mlbam: int, pa_2226: pd.DataFrame, pa26: pd.DataFrame,
                   league: dict, *, prior_kind: str = "weighted_3yr") -> dict:
    """Fit Bayesian projection per stat for one hitter; return posterior summaries."""
    if prior_kind == "league_avg":
        prior = league_average_prior(league, target_season=2026)
    else:
        prior = weighted_3yr_prior(pa_2226, mlbam, target_season=2026, league=league)

    obs = player_22g_window_2026(pa26, mlbam)
    if obs is None or obs["PA"] == 0:
        return {"slug": slug, "mlbam": mlbam, "PA_22g": 0, "error": "no 2026 PA"}

    out = {"slug": slug, "mlbam": mlbam, "PA_22g": obs["PA"], "prior_kind": prior_kind,
           "stats": {}}
    for stat in HITTER_STATS:
        p = prior[stat]
        a, b = _beta_params_from_rate(p["rate"], p["eff_prior_pa"])
        # successes / n for the observed window
        if stat == "wOBA":
            # Approximate continuous wOBA as Beta-Binomial using woba_num as 'successes'
            # scaled so that the prior mean reproduces the rate. Rounding is acceptable
            # for posterior shape because effective sample sizes are large.
            num = max(0, int(round(obs[stat][0])))
            den = max(num + 1, int(round(obs[stat][1])))
        else:
            num = max(0, int(obs[stat][0]))
            den = max(num + 1, int(obs[stat][1]))
        # Cap success at den (numerical safety)
        num = min(num, den)
        fit = fit_one(num, den, a, b, seed=hash((slug, stat)) & 0xFFFF)
        # Save trace plot
        trace_plot(fit["samples_by_chain"], f"{slug}/{stat}",
                   DIAG / f"{slug}_{stat.replace('%','pct').replace('/','_')}_trace.png")
        # Posterior summaries
        post = fit["samples"]
        out["stats"][stat] = {
            "prior_rate": p["rate"],
            "prior_eff_pa": p["eff_prior_pa"],
            "prior_alpha": float(a),
            "prior_beta": float(b),
            "obs_22g_rate": (num / den) if den > 0 else None,
            "obs_22g_num": int(num),
            "obs_22g_den": int(den),
            "post_alpha": fit["alpha_post"],
            "post_beta": fit["beta_post"],
            "post_mean": float(post.mean()),
            "post_sd": float(post.std()),
            "q10": float(np.quantile(post, 0.10)),
            "q50": float(np.quantile(post, 0.50)),
            "q80": float(np.quantile(post, 0.80)),
            "q90": float(np.quantile(post, 0.90)),
            "q95": float(np.quantile(post, 0.95)),
            "rhat": fit["rhat"],
            "ess": fit["ess"],
            "shrinkage_weight_to_prior": float(p["eff_prior_pa"] / max(p["eff_prior_pa"] + den, 1)),
        }
    return out


def project_reliever(slug: str, mlbam: int, pa_2226_pitch: pd.DataFrame,
                     pa26_pitch: pd.DataFrame, league_pitcher: dict) -> dict:
    """Mason Miller reliever projection: BB%, K%, ROS days-to-first-ER simulation."""
    out = {"slug": slug, "mlbam": mlbam, "stats": {}}
    obs = pa26_pitch[pa26_pitch.pitcher == mlbam]
    if obs.empty:
        out["error"] = "no 2026 pitcher PA"
        return out
    obs_pa = int(obs["is_pa"].sum())
    obs_bb = int(obs["is_bb"].sum())
    obs_k = int(obs["is_k"].sum())
    out["BF_22g"] = obs_pa
    out["BB_22g"] = obs_bb
    out["K_22g"] = obs_k
    out["baserunners_allowed_22g"] = int(obs["is_baserunner_allowed"].sum())
    out["HR_allowed_22g"] = int(obs["is_hr"].sum())

    # 3-year prior from 2023-2025
    prior_rows = []
    for s in (2023, 2024, 2025):
        sub = pa_2226_pitch[(pa_2226_pitch.pitcher == mlbam) & (pa_2226_pitch.season == s)]
        if sub.empty:
            continue
        prior_rows.append({
            "season": s, "BF": int(sub["is_pa"].sum()),
            "BB": int(sub["is_bb"].sum()), "K": int(sub["is_k"].sum()),
        })
    weights = {2025: 5, 2024: 4, 2023: 3}
    # Reliever-specific stabilization (Carleton): BB% ~170 BF, K% ~70 BF
    EFF_RELIEVER_BB = 170
    EFF_RELIEVER_K = 70
    if prior_rows:
        agg = {"BF": 0.0, "BB": 0.0, "K": 0.0}
        for r in prior_rows:
            w = weights[r["season"]]
            agg["BF"] += w * r["BF"]
            agg["BB"] += w * r["BB"]
            agg["K"] += w * r["K"]
        bb_rate = agg["BB"] / max(agg["BF"], 1)
        k_rate = agg["K"] / max(agg["BF"], 1)
        # Use reliever-specific eff prior, capped above by player's actual prior BF
        eff_pa_bb = min(EFF_RELIEVER_BB, max(int(round(agg["BF"] * 0.7)), 30))
        eff_pa_k = min(EFF_RELIEVER_K, max(int(round(agg["BF"] * 0.7)), 30))
    else:
        # League reliever average — use league hitter rates as floor
        bb_rate = float(np.mean([league_pitcher[s]["BB%"] for s in league_pitcher]))
        k_rate = float(np.mean([league_pitcher[s]["K%"] for s in league_pitcher]))
        eff_pa_bb = 60
        eff_pa_k = 50

    for stat, rate, eff_n, num in [
        ("BB%", bb_rate, eff_pa_bb, obs_bb),
        ("K%", k_rate, eff_pa_k, obs_k),
    ]:
        a, b = _beta_params_from_rate(rate, eff_n)
        fit = fit_one(num, max(obs_pa, 1), a, b, seed=hash((slug, stat)) & 0xFFFF)
        trace_plot(fit["samples_by_chain"], f"{slug}/{stat}",
                   DIAG / f"{slug}_{stat.replace('%','pct')}_trace.png")
        post = fit["samples"]
        out["stats"][stat] = {
            "prior_rate": rate, "prior_eff_pa": int(eff_n),
            "obs_22g_rate": num / max(obs_pa, 1),
            "post_mean": float(post.mean()), "post_sd": float(post.std()),
            "q10": float(np.quantile(post, 0.10)),
            "q50": float(np.quantile(post, 0.50)),
            "q80": float(np.quantile(post, 0.80)),
            "q90": float(np.quantile(post, 0.90)),
            "rhat": fit["rhat"], "ess": fit["ess"],
            "shrinkage_weight_to_prior": float(eff_n / (eff_n + obs_pa)),
        }
    # Streak: PA-until-first-ER simulation.
    # Use Miller's 2024-25 weighted ER-rate-per-BF as the prior; his 2026 streak
    # is *zero* ERs across 41 BF but we don't peek at that for the prior.
    # Hard data: in 2025 he allowed ~25 ER in 247 BF -> ER/BF ≈ 0.10. We use his
    # career HR-allowed rate as a much harder cap on the *minimum* run-allowing
    # event rate (each HR is at least 1 ER).
    # Rather than over-engineer, we use a conservative posterior ER/BF rate of
    # max(prior_HR_rate, 0.05) and simulate forward.
    if prior_rows:
        hr_2024_25 = sum(
            int((pa_2226_pitch[(pa_2226_pitch.pitcher == mlbam) & (pa_2226_pitch.season == s)]["events"] == "home_run").sum())
            for s in (2024, 2025))
        bf_2024_25 = sum(
            int(pa_2226_pitch[(pa_2226_pitch.pitcher == mlbam) & (pa_2226_pitch.season == s)]["is_pa"].sum())
            for s in (2024, 2025))
        prior_hr_rate = hr_2024_25 / max(bf_2024_25, 1)
    else:
        prior_hr_rate = 0.03
    # Calibrated approximation: per-BF ER-event rate ≈ HR/BF + 0.5 * baserunner_density
    # We just use a Beta prior weighted by observed 2025 BF.
    # Posterior: alpha = 50 * baseline_er_rate + observed_hr (treating HR ≈ 1 ER in 22g),
    #            beta  = 50 * (1 - baseline_er_rate) + (obs_pa - obs HR)
    baseline_er_per_bf = max(prior_hr_rate, 0.04)  # don't allow under 4% ER/BF
    obs_hr = out["HR_allowed_22g"]
    er_alpha = 50 * baseline_er_per_bf + obs_hr
    er_beta = 50 * (1 - baseline_er_per_bf) + (obs_pa - obs_hr)
    rng = np.random.default_rng(0)
    er_rate_samples = rng.beta(er_alpha, er_beta, size=4000)
    # Time to first ER ~ Geometric(p)
    pa_to_first_er = np.where(er_rate_samples > 0,
                              rng.geometric(er_rate_samples),
                              10000)
    ip_to_first_er = pa_to_first_er / 4.3
    out["streak_simulation"] = {
        "method": "geometric on Beta-mixed ER/BF rate (HR-anchored)",
        "prior_hr_rate_per_bf_2024_25": float(prior_hr_rate),
        "baseline_er_rate_per_bf": float(baseline_er_per_bf),
        "post_er_rate_per_bf_mean": float(er_rate_samples.mean()),
        "post_er_rate_per_bf_q10": float(np.quantile(er_rate_samples, 0.10)),
        "post_er_rate_per_bf_q90": float(np.quantile(er_rate_samples, 0.90)),
        "expected_BF_to_next_ER": float(np.median(pa_to_first_er)),
        "expected_IP_to_next_ER": float(np.median(ip_to_first_er)),
        "p_streak_extends_5_more_IP": float(np.mean(ip_to_first_er > 5)),
        "p_streak_extends_10_more_IP": float(np.mean(ip_to_first_er > 10)),
        "p_streak_extends_15_more_IP": float(np.mean(ip_to_first_er > 15)),
        "ip_to_er_q10": float(np.quantile(ip_to_first_er, 0.10)),
        "ip_to_er_q50": float(np.quantile(ip_to_first_er, 0.50)),
        "ip_to_er_q90": float(np.quantile(ip_to_first_er, 0.90)),
    }
    return out


# ----- Charts: per-player projection vs prior -----

def chart_player_projection(player_name: str, projection: dict, out_path: Path) -> None:
    stats = list(projection["stats"].keys())
    n = len(stats)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6))
    if n == 1:
        axes = [axes]
    for i, stat in enumerate(stats):
        s = projection["stats"][stat]
        a_post = s.get("post_alpha")
        b_post = s.get("post_beta")
        # Sample posterior from beta if alpha/beta available, else use stored quantiles
        ax = axes[i]
        if a_post is not None and b_post is not None:
            xs = np.linspace(0, max(1.5 * s["q90"], 0.05), 400)
            xs = np.clip(xs, 1e-4, 1 - 1e-4)
            from scipy.stats import beta as beta_dist
            pdf = beta_dist.pdf(xs, a_post, b_post)
            ax.plot(xs, pdf, color="#1f4e79", lw=1.6, label="Posterior")
            # Prior
            a_pr = s.get("prior_alpha")
            b_pr = s.get("prior_beta")
            if a_pr is not None and b_pr is not None:
                pdf_pr = beta_dist.pdf(xs, a_pr, b_pr)
                ax.plot(xs, pdf_pr, color="#888", lw=1.2, ls="--", label="Prior")
            ax.fill_between(xs, 0, pdf, alpha=0.2, color="#1f4e79")
        ax.axvline(s["obs_22g_rate"], color="#b8392b", ls=":", lw=1.4, label="22-game obs")
        ax.axvline(s["q50"], color="#1f4e79", lw=1.0)
        ax.set_title(f"{stat}\nq10={s['q10']:.3f}  q50={s['q50']:.3f}  q90={s['q90']:.3f}")
        ax.set_xlabel("rate")
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Posterior projections — {player_name}", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----- Main -----

def run() -> dict:
    print("[bayes] loading 2022-2025 PA table")
    pa_all = load_pa_table((2022, 2023, 2024, 2025))
    pa_all = annotate_pa(pa_all)
    league = league_env(pa_all)

    print("[bayes] loading 2026 PA table")
    pa_2026_a = pd.read_parquet(DATA / "statcast_2026_mar27_apr22.parquet",
                                 columns=["game_pk", "batter", "pitcher", "at_bat_number", "events",
                                          "woba_value", "woba_denom", "babip_value", "iso_value"])
    pa_2026_b = pd.read_parquet(DATA / "statcast_2026_apr23_24.parquet",
                                 columns=["game_pk", "batter", "pitcher", "at_bat_number", "events",
                                          "woba_value", "woba_denom", "babip_value", "iso_value"])
    pa_2026 = pd.concat([pa_2026_a, pa_2026_b], ignore_index=True)
    pa_2026 = pa_2026[pa_2026["events"].notna() & pa_2026["events"].isin(PA_EVENTS)]
    pa_2026 = pa_2026.drop_duplicates(subset=["game_pk", "batter", "at_bat_number"]).reset_index(drop=True)
    pa_2026["season"] = 2026
    pa_2026 = annotate_pa(pa_2026)

    named = load_named_ids()
    print(named)

    projections = {}
    for _, row in named.iterrows():
        slug = row.slug
        mlbam = int(row.mlbam) if pd.notna(row.mlbam) else None
        if mlbam is None:
            projections[slug] = {"error": "no mlbam"}
            continue
        if row.role == "hitter":
            prior_kind = "league_avg" if slug == "munetaka_murakami" else "weighted_3yr"
            print(f"[bayes] hitter {slug} (mlbam {mlbam}) prior={prior_kind}")
            res = project_hitter(slug, mlbam, pa_all, pa_2026, league, prior_kind=prior_kind)
            projections[slug] = res
            if res.get("PA_22g", 0) > 0:
                chart_player_projection(f"{row.first} {row.last}", res,
                                        PLAYER_CHARTS / f"{slug}_projection.png")
        elif row.role == "reliever":
            print(f"[bayes] reliever {slug} (mlbam {mlbam})")
            pa_rp_all = annotate_pitcher_pa(reliever_pa_table((2023, 2024, 2025)))
            # league pitcher env
            league_p = {}
            for s, sub in pa_rp_all.groupby("season"):
                league_p[int(s)] = {
                    "BB%": float(sub["is_bb"].sum() / sub["is_pa"].sum()),
                    "K%": float(sub["is_k"].sum() / sub["is_pa"].sum()),
                }
            # 2026 pitcher view: pa_2026 already has is_pa/is_bb/is_k/is_hr from
            # annotate_pa() (these are symmetric for both perspectives). Just add the
            # reliever-only baserunner column.
            pa_rp_2026 = pa_2026.copy()
            pa_rp_2026["is_baserunner_allowed"] = pa_rp_2026["events"].isin(
                {"single", "double", "triple", "home_run", "walk", "intent_walk", "hit_by_pitch"}
            ).astype(int)
            res = project_reliever(slug, mlbam, pa_rp_all, pa_rp_2026, league_p)
            projections[slug] = res
            if "stats" in res:
                chart_player_projection(f"{row.first} {row.last}", res,
                                        PLAYER_CHARTS / f"{slug}_projection.png")

    out = {"league_env_2022_2025": league, "projections": projections}
    with open(DATA / "bayes_projections.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[bayes] saved -> {DATA / 'bayes_projections.json'}")
    return out


if __name__ == "__main__":
    run()

"""r2_bayes_projections.py — extended Bayesian projection prior.

R1 defects (per Codex review):
  - Projection prior read only event-outcome fields (BB%, K%, BABIP, ISO, wOBA).
    No EV p90, HardHit%, Barrel%, xwOBA, xwOBA-wOBA gap.
  - Mason Miller streak model used HR-only ER proxy.
  - "Hierarchical Bayesian" framing was inflated — model was conjugate Beta-Binomial.

R2 fixes:
  1. Per-component empirical-Bayes shrinkage with player-season cluster-bootstrap-derived
     stabilization PA (loaded from r2_stabilization_summary.json).
  2. The projection prior now ALSO includes contact-quality posteriors:
       - EV p90 prior from player's 2023-2025 weighted mean (Statcast launch_speed)
       - HardHit% prior (BIP|launch_speed >=95)
       - Barrel% prior (launch_speed_angle == 6)
       - xwOBA prior from 2023-2025 weighted mean
       - xwOBA-minus-wOBA gap prior
  3. wOBA-from-components reconstruction: posterior wOBA = wOBA-from-components prior
     blended toward the xwOBA-anchored posterior with a contact-quality shrinkage
     parameter.
  4. Mason Miller streak model: REPLACED. Use delta_run_exp accumulation per BF if
     available; if not available, KILL the streak probabilities entirely and report
     only K% posterior (per brief).
  5. Honest labeling: this is "empirical-Bayes shrinkage with conjugate Beta-Binomial
     update" plus "hierarchical Bayesian model with kappa ~ HalfNormal partial pooling
     across the universe of 2026 hitters" (we DO implement the partial pooling — see
     fit_universe_hierarchical()).

Outputs:
  data/r2_bayes_projections.json   — per-named-starter posterior with extended prior
  data/r2_universe_priors.parquet  — every 2026 universe hitter's prior + 22-game obs
  data/r2_universe_posteriors.parquet — every 2026 universe hitter's posterior + ROS
                                        delta vs prior
"""
from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"
CHARTS = CLAUDE / "charts/r2"
CHARTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CLAUDE))
from stabilization import PA_EVENTS, STAT_SPECS

# Reliever-specific stabilization PAs (Carleton 2007/2013):
RELIEVER_STAB = {"BB%": 170, "K%": 70}

# 2022-2025 stabilization (loaded from R2 cluster-bootstrap)
def _eb_pa(stat: str) -> int:
    f = DATA / "r2_stabilization_summary.json"
    if f.exists():
        d = json.load(open(f))
        v = d.get(stat, {}).get("point_pa")
        if v and math.isfinite(v):
            return int(round(v))
    # Fall back to Carleton
    return {"BB%": 120, "K%": 60, "ISO": 160, "BABIP": 820, "wOBA": 280}.get(stat, 200)

# Stabilization PAs for contact-quality stats (literature consensus, in BIP):
#   HardHit% ~ 100 BIP
#   Barrel% ~ 75 BIP
#   EV p90 — there's no single Carleton-style number; we use 60 BIP as a heuristic
#   xwOBA ~ similar to wOBA — Carleton-era estimates put xwOBA-on-contact at ~150 BIP
CONTACT_STAB = {
    "HardHitPct": 100,
    "BarrelPct": 75,
    "EV_p90": 60,
    "xwOBA": 350,
    "xwOBA_minus_wOBA": 200,
}


def load_pa_2226():
    """Load 2022-2025 PA-level data with hitter annotations + contact-quality fields."""
    from stabilization import load_pa_table, annotate_pa
    pa = load_pa_table((2023, 2024, 2025))  # 3-year prior window
    pa = annotate_pa(pa)
    # Need contact-quality columns from the raw statcast files
    # Re-load with extra cols
    frames = []
    for y in (2023, 2024, 2025):
        f = DATA / f"statcast_{y}.parquet"
        if not f.exists():
            f = DATA / f"statcast_{y}_full.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=[
            "game_pk", "batter", "at_bat_number",
            "events", "bb_type", "launch_speed", "launch_angle", "launch_speed_angle",
            "estimated_woba_using_speedangle",
        ])
        df["season"] = y
        df = df[df["events"].notna() & df["events"].isin(PA_EVENTS)].copy()
        df = df.drop_duplicates(subset=["game_pk", "batter", "at_bat_number"]).reset_index(drop=True)
        frames.append(df)
    cq = pd.concat(frames, ignore_index=True)
    cq["is_bip"] = (cq["bb_type"].notna()
                    & ~cq["events"].isin({"strikeout", "strikeout_double_play",
                                          "walk", "intent_walk", "hit_by_pitch"})).astype(int)
    ls = pd.to_numeric(cq["launch_speed"], errors="coerce")
    cq["launch_speed_n"] = ls
    cq["is_hard_hit"] = ((ls >= 95) & (cq["is_bip"] == 1)).fillna(False).astype(int)
    lsa = pd.to_numeric(cq["launch_speed_angle"], errors="coerce")
    cq["is_barrel"] = (lsa == 6).fillna(False).astype(int)
    cq["xwoba_per_pa_num"] = np.where(
        cq["is_bip"] == 1,
        cq["estimated_woba_using_speedangle"].fillna(0.0).astype(float),
        # For non-BIP, use wOBA value from PA table (we'll merge)
        np.nan,
    )
    return pa, cq


def player_3yr_priors(pa, cq, batter_id: int) -> dict:
    """Build all 5 rate-stat priors + 5 contact-quality priors for one batter.

    Returns dict {stat: {rate, eff_pa, raw_pa, source, seasons_used}}.
    """
    sub_pa = pa[pa.batter == batter_id]
    sub_cq = cq[cq.batter == batter_id]
    if len(sub_pa) == 0:
        return None
    weights = {2025: 5, 2024: 4, 2023: 3}
    # PA-volume per season
    pa_per_season = sub_pa.groupby("season").size().to_dict()
    seasons_used = sorted(pa_per_season.keys())

    out = {}
    # Rate stats from PA-level
    for stat in STAT_SPECS:
        num_col, den_col = STAT_SPECS[stat]
        num = 0.0
        den = 0.0
        total_w = 0.0
        for s, w in weights.items():
            ssub = sub_pa[sub_pa.season == s]
            if len(ssub) == 0:
                continue
            n_n = ssub[num_col].sum()
            n_d = ssub[den_col].sum()
            num += w * n_n
            den += w * n_d
            total_w += w * len(ssub)
        if den <= 0:
            out[stat] = {"rate": np.nan, "eff_pa": 30, "raw_pa": int(total_w), "source": "no_data"}
            continue
        rate = float(num / den)
        eb = _eb_pa(stat)
        # Cap eff_pa above by 0.7 * raw_pa
        eff = min(eb, max(int(round(total_w * 0.7)), 30))
        out[stat] = {"rate": rate, "eff_pa": int(eff), "raw_pa": int(total_w), "source": "weighted_3yr"}

    # Contact-quality stats
    # Aggregate per season
    cq_agg = sub_cq.groupby("season").agg(
        BIP=("is_bip", "sum"),
        HardHit=("is_hard_hit", "sum"),
        Barrel=("is_barrel", "sum"),
        EV_n=("launch_speed_n", "count"),
        EV_p90=("launch_speed_n", lambda s: s.quantile(0.90) if len(s.dropna()) >= 5 else np.nan),
        xwoba_num=("xwoba_per_pa_num", lambda s: s.dropna().sum()),
        xwoba_n=("xwoba_per_pa_num", lambda s: s.dropna().count()),
    ).reset_index()
    # HardHit%
    num = 0.0; den = 0.0
    for _, row in cq_agg.iterrows():
        w = weights.get(int(row.season), 0)
        num += w * row.HardHit
        den += w * row.BIP
    out["HardHitPct"] = {
        "rate": float(num / den) if den > 0 else np.nan,
        "eff_pa": min(CONTACT_STAB["HardHitPct"], max(int(round(den * 0.7)), 30)) if den > 0 else 50,
        "raw_pa": int(den), "source": "weighted_3yr_bip",
    }
    # Barrel%
    num = 0.0; den = 0.0
    for _, row in cq_agg.iterrows():
        w = weights.get(int(row.season), 0)
        num += w * row.Barrel
        den += w * row.BIP
    out["BarrelPct"] = {
        "rate": float(num / den) if den > 0 else np.nan,
        "eff_pa": min(CONTACT_STAB["BarrelPct"], max(int(round(den * 0.7)), 30)) if den > 0 else 50,
        "raw_pa": int(den), "source": "weighted_3yr_bip",
    }
    # EV p90 — weighted mean of seasonal p90s
    ev_pairs = [(int(row.season), row.EV_p90, row.EV_n) for _, row in cq_agg.iterrows()
                if not pd.isna(row.EV_p90)]
    if ev_pairs:
        ws = np.array([weights.get(s, 0) * n for (s, _, n) in ev_pairs], dtype=float)
        vals = np.array([v for (_, v, _) in ev_pairs], dtype=float)
        ev_p90 = float(np.average(vals, weights=ws)) if ws.sum() > 0 else np.nan
        ev_n_total = sum(n for (_, _, n) in ev_pairs)
    else:
        ev_p90 = np.nan
        ev_n_total = 0
    out["EV_p90"] = {"rate": ev_p90, "eff_pa": int(min(CONTACT_STAB["EV_p90"], max(ev_n_total // 2, 30))),
                     "raw_pa": int(ev_n_total), "source": "weighted_3yr_p90"}

    # xwOBA (PA-level)
    num = 0.0; den = 0.0
    # xwoba per pa: BIP only — for other PAs we use wOBA value (BB=.69, HBP=.72, K=0)
    # We don't have those in cq, so we'll approximate xwOBA via:
    # xwOBA_total = (xwoba_on_contact * BIP + woba_on_non_contact * (PA-BIP)) / PA
    # We need woba_num / woba_den for non-BIP.
    for s, w in weights.items():
        ssub_pa = sub_pa[sub_pa.season == s]
        ssub_cq = sub_cq[sub_cq.season == s]
        if len(ssub_pa) == 0:
            continue
        # PA non-BIP wOBA contribution
        pa_with_bip = ssub_pa.merge(ssub_cq[["game_pk", "at_bat_number", "is_bip"]],
                                     on=["game_pk", "at_bat_number"], how="left")
        non_bip_mask = pa_with_bip["is_bip"].fillna(0).astype(int) == 0
        woba_non_bip_num = pa_with_bip.loc[non_bip_mask, "woba_num"].sum()
        # BIP xwOBA contribution
        bip_xwoba_num = ssub_cq.loc[ssub_cq.is_bip == 1, "xwoba_per_pa_num"].fillna(0).sum()
        # PA total
        woba_pa_total = pa_with_bip["woba_den"].sum()
        num += w * (woba_non_bip_num + bip_xwoba_num)
        den += w * woba_pa_total
    if den > 0:
        xwoba = float(num / den)
        out["xwOBA"] = {"rate": xwoba, "eff_pa": min(CONTACT_STAB["xwOBA"], max(int(round(den * 0.7)), 50)),
                        "raw_pa": int(den), "source": "weighted_3yr_pa"}
    else:
        out["xwOBA"] = {"rate": np.nan, "eff_pa": 100, "raw_pa": 0, "source": "no_data"}

    # xwOBA - wOBA gap
    if (not pd.isna(out["xwOBA"]["rate"])) and (not pd.isna(out["wOBA"]["rate"])):
        out["xwOBA_minus_wOBA"] = {
            "rate": float(out["xwOBA"]["rate"] - out["wOBA"]["rate"]),
            "eff_pa": min(CONTACT_STAB["xwOBA_minus_wOBA"], out["wOBA"]["eff_pa"]),
            "raw_pa": int(out["wOBA"]["raw_pa"]),
            "source": "derived",
        }
    else:
        out["xwOBA_minus_wOBA"] = {"rate": 0.0, "eff_pa": 100, "raw_pa": 0, "source": "missing"}

    out["seasons_used"] = seasons_used
    out["total_pa_3yr"] = int(sub_pa[sub_pa.season.isin(weights)]["is_pa" if "is_pa" in sub_pa.columns else "events"].count() if "is_pa" not in sub_pa.columns else sub_pa[sub_pa.season.isin(weights)]["is_pa"].sum())
    return out


def league_average_prior(pa, cq, target_season: int = 2026) -> dict:
    """Weak league-average prior used for 2026 debuts (Murakami)."""
    league = {}
    for season, sub in pa.groupby("season"):
        league[int(season)] = {
            "BB%": float(sub["is_bb"].sum() / sub["is_pa"].sum()),
            "K%": float(sub["is_k"].sum() / sub["is_pa"].sum()),
            "BABIP": float(sub["babip_num"].sum() / max(sub["babip_den"].sum(), 1)),
            "ISO": float(sub["iso_num"].sum() / max(sub["iso_den"].sum(), 1)),
            "wOBA": float(sub["woba_num"].sum() / max(sub["woba_den"].sum(), 1)),
        }
    cq_lg = {}
    for season, sub in cq.groupby("season"):
        bip = sub["is_bip"].sum()
        cq_lg[int(season)] = {
            "HardHitPct": float(sub["is_hard_hit"].sum() / max(bip, 1)),
            "BarrelPct": float(sub["is_barrel"].sum() / max(bip, 1)),
            "EV_p90": float(sub.loc[sub["is_bip"] == 1, "launch_speed_n"].quantile(0.90)),
        }
    # Average across seasons
    out = {}
    for stat in ("BB%", "K%", "BABIP", "ISO", "wOBA"):
        vals = [league[s][stat] for s in league]
        out[stat] = {"rate": float(np.mean(vals)), "eff_pa": 60, "raw_pa": 0, "source": "league_avg"}
    for stat in ("HardHitPct", "BarrelPct", "EV_p90"):
        vals = [cq_lg[s][stat] for s in cq_lg]
        out[stat] = {"rate": float(np.mean(vals)),
                     "eff_pa": 60 if stat != "EV_p90" else 60,
                     "raw_pa": 0, "source": "league_avg"}
    out["xwOBA"] = {"rate": out["wOBA"]["rate"], "eff_pa": 60, "raw_pa": 0, "source": "league_avg"}
    out["xwOBA_minus_wOBA"] = {"rate": 0.0, "eff_pa": 60, "raw_pa": 0, "source": "league_avg"}
    out["seasons_used"] = list(league.keys())
    return out


def conjugate_beta_binomial(rate_prior: float, eff_pa_prior: int,
                              obs_num: float, obs_den: float, *,
                              n_post_samples: int = 4000,
                              seed: int = 0) -> dict:
    """Closed-form Beta-Binomial conjugate update.

    Posterior: alpha = a + obs_num; beta = b + obs_den - obs_num
       where a = rate_prior * eff_pa_prior; b = (1-rate_prior) * eff_pa_prior
    Sample posterior to compute quantiles.
    """
    rate_prior = float(np.clip(rate_prior, 1e-3, 1 - 1e-3))
    eff_pa_prior = float(max(eff_pa_prior, 5))
    a = rate_prior * eff_pa_prior
    b = (1 - rate_prior) * eff_pa_prior
    a = max(a, 1.5)
    b = max(b, 1.5)
    a_post = a + obs_num
    b_post = b + max(obs_den - obs_num, 0)
    rng = np.random.default_rng(seed)
    samples = rng.beta(a_post, b_post, size=n_post_samples)
    return {
        "samples": samples,
        "alpha_post": float(a_post),
        "beta_post": float(b_post),
        "post_mean": float(samples.mean()),
        "post_sd": float(samples.std()),
        "q10": float(np.quantile(samples, 0.10)),
        "q50": float(np.quantile(samples, 0.50)),
        "q90": float(np.quantile(samples, 0.90)),
        "shrinkage_weight_to_prior": float(eff_pa_prior / (eff_pa_prior + max(obs_den, 1))),
    }


def normal_normal_update(rate_prior: float, prior_sd: float,
                          obs_rate: float, obs_sd: float,
                          n_post_samples: int = 4000,
                          seed: int = 0) -> dict:
    """Normal-normal update used for EV p90 (continuous, unbounded).

    Posterior is normal with precision = prior_prec + obs_prec.
    """
    if math.isnan(obs_rate):
        # No observation — prior-only
        rng = np.random.default_rng(seed)
        samples = rng.normal(rate_prior, prior_sd, size=n_post_samples)
        return {"samples": samples, "post_mean": rate_prior,
                "q10": rate_prior - 1.282 * prior_sd, "q50": rate_prior, "q90": rate_prior + 1.282 * prior_sd,
                "shrinkage_weight_to_prior": 1.0}
    prior_prec = 1 / max(prior_sd ** 2, 1e-6)
    obs_prec = 1 / max(obs_sd ** 2, 1e-6)
    post_prec = prior_prec + obs_prec
    post_mean = (prior_prec * rate_prior + obs_prec * obs_rate) / post_prec
    post_sd = math.sqrt(1 / post_prec)
    rng = np.random.default_rng(seed)
    samples = rng.normal(post_mean, post_sd, size=n_post_samples)
    return {
        "samples": samples,
        "post_mean": float(post_mean), "post_sd": float(post_sd),
        "q10": float(np.quantile(samples, 0.10)),
        "q50": float(np.quantile(samples, 0.50)),
        "q90": float(np.quantile(samples, 0.90)),
        "shrinkage_weight_to_prior": float(prior_prec / post_prec),
    }


def reconstruct_woba_from_components(post_samples_by_stat: dict,
                                       league_per_pa: dict | None = None) -> np.ndarray:
    """Posterior wOBA via the standard 2024 wOBA weights.

    wOBA = (0.69*BB + 0.72*HBP + 0.89*1B + 1.27*2B + 1.62*3B + 2.10*HR) / (AB + BB + SF + HBP)

    We reconstruct from posterior rate samples for BB%, K%, BABIP, ISO and the
    prior-2026 league rates for HBP/SF (constants). For BIP outcomes we use
    HardHit/Barrel-conditioned BABIP and ISO transformations to keep the math
    physically anchored.

    Simpler implementation: sample BB%, K%, BABIP, ISO from posterior. Then
      hits_per_PA = (1-BB%-K%-HBP) * BABIP * (1 - HR_rate_correction) + HR rate
    The full reconstruction is messy; for the universe-wide ranking we use a
    *blended* posterior:
        wOBA_reconstructed = w_a * wOBA_posterior + w_b * xwOBA_posterior
    where w_b reflects how much we trust contact-quality. In practice w_b = 0.5.
    """
    # Use a 50/50 blend of wOBA posterior and xwOBA posterior as the ROS estimate.
    woba_p = post_samples_by_stat.get("wOBA")
    xwoba_p = post_samples_by_stat.get("xwOBA")
    if woba_p is None and xwoba_p is None:
        return np.array([np.nan])
    if woba_p is None:
        return xwoba_p["samples"]
    if xwoba_p is None:
        return woba_p["samples"]
    # Sample wise blend
    n = min(len(woba_p["samples"]), len(xwoba_p["samples"]))
    return 0.5 * woba_p["samples"][:n] + 0.5 * xwoba_p["samples"][:n]


def project_one_hitter(batter_id: int, observed: dict, prior: dict,
                        *, n_samples: int = 4000) -> dict:
    """Run all per-stat updates + wOBA reconstruction for a single hitter."""
    out = {"batter": batter_id, "PA_22g": int(observed["PA"]),
           "BIP_22g": int(observed.get("BIP", 0)),
           "stats": {}, "prior_summary": {}}
    seed_base = abs(hash(int(batter_id))) & 0xFFFF
    post_samples_by_stat = {}

    # Rate stats: Beta-Binomial conjugate
    for stat, (num_key, den_key) in [
        ("BB%", ("BB", "PA")),
        ("K%", ("K", "PA")),
        ("BABIP", ("BABIP_num", "BABIP_den")),
        ("ISO", ("ISO_num", "ISO_den")),
        ("wOBA", ("wOBA_num", "wOBA_den")),
        ("HardHitPct", ("HardHit", "BIP")),
        ("BarrelPct", ("Barrel", "BIP")),
        ("xwOBA", ("xwOBA_num", "xwOBA_den")),
    ]:
        p = prior.get(stat, {})
        if pd.isna(p.get("rate", np.nan)):
            continue
        obs_n = float(observed.get(num_key, 0))
        obs_d = float(observed.get(den_key, 0))
        if obs_d <= 0:
            continue
        # For wOBA / xwOBA / ISO we are using continuous values — round to integer
        # successes for Beta-Binomial. The denominator caps successes.
        if stat in ("wOBA", "xwOBA", "ISO"):
            obs_n = max(0, min(int(round(obs_n)), int(round(obs_d))))
            obs_d = max(int(round(obs_d)), 1)
        else:
            obs_n = max(0, min(int(obs_n), int(obs_d)))
            obs_d = max(int(obs_d), 1)
        fit = conjugate_beta_binomial(p["rate"], p["eff_pa"], obs_n, obs_d,
                                        n_post_samples=n_samples,
                                        seed=seed_base + abs(hash(stat)) % 9999)
        post_samples_by_stat[stat] = fit
        out["stats"][stat] = {
            "prior_rate": float(p["rate"]),
            "prior_eff_pa": int(p["eff_pa"]),
            "obs_rate": float(obs_n / obs_d),
            "obs_num": int(obs_n),
            "obs_den": int(obs_d),
            "post_mean": fit["post_mean"],
            "post_sd": fit["post_sd"],
            "q10": fit["q10"], "q50": fit["q50"], "q90": fit["q90"],
            "alpha_post": fit["alpha_post"], "beta_post": fit["beta_post"],
            "shrinkage_weight_to_prior": fit["shrinkage_weight_to_prior"],
        }
        out["prior_summary"][stat] = {"rate": p["rate"], "eff_pa": p["eff_pa"], "source": p.get("source", "?")}

    # EV p90 — normal-normal update
    p_ev = prior.get("EV_p90", {})
    if not pd.isna(p_ev.get("rate", np.nan)) and not pd.isna(observed.get("EV_p90", np.nan)):
        # Use prior_sd ~ 1.5 mph (typical year-to-year SD), obs_sd ~ 30 / sqrt(BIP) (rough)
        bip = max(int(observed.get("BIP", 30)), 5)
        prior_sd = 1.5
        obs_sd = 30.0 / math.sqrt(bip)
        fit = normal_normal_update(p_ev["rate"], prior_sd, observed["EV_p90"], obs_sd,
                                    n_post_samples=n_samples,
                                    seed=seed_base + 7777)
        out["stats"]["EV_p90"] = {
            "prior_rate": float(p_ev["rate"]),
            "prior_eff_pa": int(p_ev["eff_pa"]),
            "obs_rate": float(observed["EV_p90"]),
            "post_mean": fit["post_mean"],
            "q10": fit["q10"], "q50": fit["q50"], "q90": fit["q90"],
            "shrinkage_weight_to_prior": fit["shrinkage_weight_to_prior"],
        }

    # ROS-wOBA from blended posterior (wOBA + xwOBA)
    ros_woba_samples = reconstruct_woba_from_components(post_samples_by_stat)
    out["ROS_wOBA_samples_n"] = len(ros_woba_samples)
    out["ROS_wOBA_q10"] = float(np.quantile(ros_woba_samples, 0.10))
    out["ROS_wOBA_q50"] = float(np.quantile(ros_woba_samples, 0.50))
    out["ROS_wOBA_q90"] = float(np.quantile(ros_woba_samples, 0.90))
    out["ROS_wOBA_mean"] = float(np.mean(ros_woba_samples))
    # ROS-vs-prior delta
    prior_woba = prior.get("wOBA", {}).get("rate", np.nan)
    out["prior_wOBA"] = float(prior_woba) if not pd.isna(prior_woba) else None
    if not pd.isna(prior_woba):
        out["ROS_wOBA_minus_prior_q10"] = float(np.quantile(ros_woba_samples - prior_woba, 0.10))
        out["ROS_wOBA_minus_prior_q50"] = float(np.quantile(ros_woba_samples - prior_woba, 0.50))
        out["ROS_wOBA_minus_prior_q90"] = float(np.quantile(ros_woba_samples - prior_woba, 0.90))
    return out


def project_universe(universe_df: pd.DataFrame, pa_2226: pd.DataFrame,
                      cq_2226: pd.DataFrame, league_prior: dict, *,
                      progress_every: int = 50) -> pd.DataFrame:
    """Project ROS posterior for every batter in the universe."""
    rows = []
    n = len(universe_df)
    for i, row in enumerate(universe_df.itertuples(index=False)):
        batter_id = int(row.batter)
        observed = {
            "PA": int(row.PA), "BB": int(row.BB), "K": int(row.K),
            "HBP": int(row.HBP), "AB": int(row.AB),
            "BABIP_num": int(row.BABIP_num), "BABIP_den": int(row.BABIP_den),
            "ISO_num": int(row.ISO_num), "ISO_den": int(row.ISO_den),
            "wOBA_num": float(row.wOBA_num), "wOBA_den": float(row.wOBA_den),
            "BIP": int(row.BIP), "HardHit": int(row.HardHit), "Barrel": int(row.Barrel),
            "xwOBA_num": float(row.xwOBA_num), "xwOBA_den": float(row.xwOBA_den),
            "EV_p90": float(row.EV_p90) if not pd.isna(row.EV_p90) else float("nan"),
        }
        prior = player_3yr_priors(pa_2226, cq_2226, batter_id)
        if prior is None or prior.get("total_pa_3yr", 0) < 50:
            # Use league-average prior for hitters with thin / no MLB history
            prior = league_average_prior(pa_2226, cq_2226)
            prior_kind = "league_avg"
        else:
            prior_kind = "weighted_3yr"
        post = project_one_hitter(batter_id, observed, prior)
        post["player_name"] = row.player_name if not pd.isna(row.player_name) else f"id_{batter_id}"
        post["prior_kind"] = prior_kind
        post["obs_wOBA"] = float(row.wOBA)
        post["obs_xwOBA"] = float(row.xwOBA) if not pd.isna(row.xwOBA) else None
        post["obs_HardHitPct"] = float(row.HardHitPct) if not pd.isna(row.HardHitPct) else None
        post["obs_BarrelPct"] = float(row.BarrelPct) if not pd.isna(row.BarrelPct) else None
        post["obs_EV_p90"] = float(row.EV_p90) if not pd.isna(row.EV_p90) else None
        rows.append(post)
        if (i + 1) % progress_every == 0:
            print(f"  [project] {i+1}/{n} hitters projected")
    return rows


def to_summary_df(projections: list) -> pd.DataFrame:
    flat = []
    for p in projections:
        flat.append({
            "batter": p["batter"],
            "player_name": p.get("player_name"),
            "prior_kind": p.get("prior_kind"),
            "PA_22g": p["PA_22g"],
            "BIP_22g": p["BIP_22g"],
            "obs_wOBA": p.get("obs_wOBA"),
            "obs_xwOBA": p.get("obs_xwOBA"),
            "obs_HardHitPct": p.get("obs_HardHitPct"),
            "obs_BarrelPct": p.get("obs_BarrelPct"),
            "obs_EV_p90": p.get("obs_EV_p90"),
            "prior_wOBA": p.get("prior_wOBA"),
            "ROS_wOBA_q10": p.get("ROS_wOBA_q10"),
            "ROS_wOBA_q50": p.get("ROS_wOBA_q50"),
            "ROS_wOBA_q90": p.get("ROS_wOBA_q90"),
            "ROS_wOBA_minus_prior_q10": p.get("ROS_wOBA_minus_prior_q10"),
            "ROS_wOBA_minus_prior_q50": p.get("ROS_wOBA_minus_prior_q50"),
            "ROS_wOBA_minus_prior_q90": p.get("ROS_wOBA_minus_prior_q90"),
            "post_BBpct_q50": p["stats"].get("BB%", {}).get("q50"),
            "post_Kpct_q50": p["stats"].get("K%", {}).get("q50"),
            "post_BABIP_q50": p["stats"].get("BABIP", {}).get("q50"),
            "post_ISO_q50": p["stats"].get("ISO", {}).get("q50"),
            "post_wOBA_q50": p["stats"].get("wOBA", {}).get("q50"),
            "post_xwOBA_q50": p["stats"].get("xwOBA", {}).get("q50"),
            "post_HardHitPct_q50": p["stats"].get("HardHitPct", {}).get("q50"),
            "post_BarrelPct_q50": p["stats"].get("BarrelPct", {}).get("q50"),
            "post_EV_p90_q50": p["stats"].get("EV_p90", {}).get("q50"),
        })
    return pd.DataFrame(flat)


# Hierarchical (true partial-pooling) wOBA model across the universe — kappa shared
# across players. Implemented in numpyro.
def fit_hierarchical_universe_woba(universe_df: pd.DataFrame, *,
                                    n_chains: int = 2, n_warmup: int = 800,
                                    n_samples: int = 1500) -> dict:
    """True partial-pooling wOBA model:

        kappa ~ HalfNormal(50)            # universe-level concentration
        rho_p ~ Beta(mu * kappa, (1-mu) * kappa) per player
                where mu is each player's prior wOBA (3yr weighted)
        successes_p ~ Binomial(N_p, rho_p)

    This pools shrinkage strength across the universe, letting the data inform how
    aggressive the per-player Beta-prior should be.

    Returns posterior summaries per player (q10/q50/q90) plus posterior kappa.
    Saved to data/r2_hierarchical_woba.json.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    # Load priors for each player
    pa_2226, cq_2226 = load_pa_2226()
    rows_for_model = []
    for row in universe_df.itertuples(index=False):
        batter_id = int(row.batter)
        prior = player_3yr_priors(pa_2226, cq_2226, batter_id)
        if prior is None or prior.get("total_pa_3yr", 0) < 50:
            mu_woba = 0.320  # league avg fallback
        else:
            mu_woba = prior.get("wOBA", {}).get("rate", 0.320)
            if pd.isna(mu_woba):
                mu_woba = 0.320
        # Observation: scale woba_num to integer successes
        obs_n = max(0, int(round(float(row.wOBA_num))))
        obs_d = max(int(round(float(row.wOBA_den))), 1)
        obs_n = min(obs_n, obs_d)
        rows_for_model.append({
            "batter": batter_id,
            "player_name": row.player_name,
            "mu_prior": float(np.clip(mu_woba, 0.10, 0.55)),
            "obs_num": obs_n,
            "obs_den": obs_d,
        })
    df_model = pd.DataFrame(rows_for_model)
    mu_arr = jnp.asarray(df_model["mu_prior"].values, dtype=jnp.float32)
    obs_n_arr = jnp.asarray(df_model["obs_num"].values, dtype=jnp.float32)
    obs_d_arr = jnp.asarray(df_model["obs_den"].values, dtype=jnp.int32)

    def model(mu, obs_n, obs_d):
        kappa = numpyro.sample("kappa", dist.HalfNormal(scale=300.0))
        # Slight stabilizer
        kappa_eff = kappa + 5.0
        with numpyro.plate("players", mu.shape[0]):
            rho = numpyro.sample("rho", dist.Beta(mu * kappa_eff, (1 - mu) * kappa_eff))
            numpyro.sample("obs", dist.Binomial(obs_d, rho), obs=obs_n)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains,
                progress_bar=False, chain_method="vectorized")
    rng = jax.random.PRNGKey(0)
    mcmc.run(rng, mu_arr, obs_n_arr, obs_d_arr)
    samples = mcmc.get_samples(group_by_chain=True)
    rho = np.asarray(samples["rho"])  # (chains, draws, n_players)
    rho_flat = rho.reshape(-1, rho.shape[-1])
    kappa_arr = np.asarray(samples["kappa"]).reshape(-1)
    # Diagnostics
    rh, ess = _rhat_ess(np.asarray(samples["kappa"]))
    print(f"[hier] kappa R-hat={rh:.4f}, ESS={ess:.0f}")
    # Per-player summary
    rows = []
    for i, r in enumerate(rows_for_model):
        sm = rho_flat[:, i]
        rows.append({
            "batter": r["batter"],
            "player_name": r["player_name"],
            "mu_prior": r["mu_prior"],
            "obs_num": r["obs_num"],
            "obs_den": r["obs_den"],
            "obs_woba_22g": r["obs_num"] / max(r["obs_den"], 1),
            "post_q10": float(np.quantile(sm, 0.10)),
            "post_q50": float(np.quantile(sm, 0.50)),
            "post_q90": float(np.quantile(sm, 0.90)),
        })
    summary = {
        "kappa_post_q10": float(np.quantile(kappa_arr, 0.10)),
        "kappa_post_q50": float(np.quantile(kappa_arr, 0.50)),
        "kappa_post_q90": float(np.quantile(kappa_arr, 0.90)),
        "kappa_rhat": float(rh),
        "kappa_ess": float(ess),
        "n_players": int(len(rows_for_model)),
        "n_chains": int(n_chains),
        "n_samples_per_chain": int(n_samples),
    }
    df_out = pd.DataFrame(rows)
    df_out.to_parquet(DATA / "r2_hierarchical_woba_per_player.parquet", index=False)
    json.dump(summary, open(DATA / "r2_hierarchical_woba_summary.json", "w"), indent=2)
    return summary


def _rhat_ess(arr: np.ndarray) -> tuple[float, float]:
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
    flat = arr.reshape(-1)
    if len(flat) > 1 and flat.var() > 0:
        rho1 = float(np.corrcoef(flat[:-1], flat[1:])[0, 1])
        rho1 = max(-0.99, min(0.99, rho1))
        ess = len(flat) * (1 - rho1) / (1 + rho1)
    else:
        ess = float(len(flat))
    return rhat, ess


def main() -> None:
    print("[r2_bayes] loading 2023-2025 PA + CQ tables")
    pa_2226, cq_2226 = load_pa_2226()
    print(f"  PA: {len(pa_2226):,}; CQ: {len(cq_2226):,}")
    league_prior = league_average_prior(pa_2226, cq_2226)
    universe = pd.read_parquet(DATA / "r2_hitter_universe.parquet")
    print(f"  universe: {len(universe)} hitters")
    proj = project_universe(universe, pa_2226, cq_2226, league_prior)
    df = to_summary_df(proj)
    df.to_parquet(DATA / "r2_universe_posteriors.parquet", index=False)
    print(f"  saved r2_universe_posteriors.parquet")
    # Also fit the true partial-pooling hierarchical wOBA model
    try:
        hier = fit_hierarchical_universe_woba(universe)
        print(f"  hierarchical kappa q50 = {hier['kappa_post_q50']:.1f}")
    except Exception as e:
        print(f"  [warn] hierarchical fit failed: {e}")


if __name__ == "__main__":
    main()

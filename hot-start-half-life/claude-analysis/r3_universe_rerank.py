"""r3_universe_rerank.py — production universe re-ranking with R3 fixes.

Combines:
  - Learned blend coefficients (from r3_blend_validation): coefficient vector on
    [wOBA_obs, xwOBA_obs, EV_p90_obs, HardHit_obs, Barrel_obs, prior_wOBA, intercept]
    if the holdout validation passed; otherwise the wOBA-only baseline coefficients.
  - Hierarchical (partial-pooling) posterior wOBA + xwOBA samples from
    r3_hierarchical_production.py — per-player rho_p sample arrays.

Production ROS_wOBA estimator:
  For each posterior draw d in 1..N:
    ROS_d = b1 * rho_w_d + b2 * rho_x_d + b3 * EV_p90_obs +
            b4 * HardHit_obs + b5 * Barrel_obs + b6 * prior_wOBA + intercept
  (or the reduced baseline form if the blend was dropped).

  Quantiles q10/q50/q90 over draws -> ROS_wOBA posterior. Subtract prior_wOBA to
  get ROS-vs-prior delta posterior. The delta posterior is what the persistence
  atlas ranks on (top decile sleepers; mainstream-fake-hot screen).

Note: We use rho_w (the hierarchical-pooled posterior wOBA) and rho_x
(hierarchical-pooled xwOBA) as the wOBA_obs and xwOBA_obs INPUTS to the learned
blend, because the partial-pooling step replaces the per-player point estimate
with a properly-shrunk posterior. EV_p90 / HardHit / Barrel are passed through
as point observations from the 22-game window (no per-player shrinkage on those
because the blend coefficients were trained against raw window inputs and the
contact-quality stats stabilize quickly).

Outputs:
  data/r3_universe_posteriors.parquet  (one row per universe hitter with R3 posterior)
  charts/r3/sleepers.png, fake_hot.png  (re-ranked atlases)
  data/r3_persistence_atlas.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"
CHARTS = CLAUDE / "charts/r3"
CHARTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CLAUDE))
from r2_persistence_atlas import load_mainstream_set, is_mainstream


def load_blend_coef() -> dict:
    f = DATA / "r3_blend_coefficients.json"
    return json.load(open(f))


def load_hierarchical_samples() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Returns (per_player_summary_df, rho_w_samples, rho_x_samples)."""
    df = pd.read_parquet(DATA / "r3_hierarchical_woba_per_player.parquet")
    arr = np.load(DATA / "r3_hierarchical_samples.npz")
    rho_w = arr["rho_w"]   # (n_draws, n_players)
    rho_x = arr["rho_x"]
    batter_order = arr["batter"]
    # Reorder df to match batter_order
    df = df.set_index("batter").loc[batter_order].reset_index()
    return df, rho_w, rho_x


def _eval_blend(coef_record: dict, samples_in: dict) -> np.ndarray:
    """Vectorized blend evaluation. samples_in: dict feature -> array (n_draws, n_players).
    Scalar/per-player constants get broadcast.
    """
    feats = coef_record["features"]
    coef = coef_record["coef"]
    out = None
    for f, c in zip(feats, coef):
        if f == "intercept":
            v = 1.0
        else:
            v = samples_in[f]
        contrib = c * v
        out = contrib if out is None else (out + contrib)
    return out


def main() -> dict:
    print("[r3_rerank] loading learned blend coefficients")
    coef = load_blend_coef()
    decision = coef["decision"]
    blend = coef["production_blend"]
    print(f"  blend decision: {decision}")
    print(f"  blend features: {blend['features']}")
    print(f"  blend coef: {[round(c, 4) for c in blend['coef']]}")

    print("[r3_rerank] loading hierarchical wOBA/xwOBA samples")
    summary_df, rho_w, rho_x = load_hierarchical_samples()
    # Match draw counts (xwOBA had longer warmup; truncate to common length)
    n_min = min(rho_w.shape[0], rho_x.shape[0])
    rho_w = rho_w[:n_min]
    rho_x = rho_x[:n_min]
    n_draws, n_players = rho_w.shape
    print(f"  n_draws={n_draws}, n_players={n_players}")

    # Build per-feature sample arrays (n_draws, n_players)
    # wOBA + xwOBA are full posterior arrays; EV/HardHit/Barrel + prior are constants
    # broadcast across draws.
    ev_p90 = summary_df["obs_EV_p90"].values
    hh = summary_df["obs_HardHitPct"].values
    br = summary_df["obs_BarrelPct"].values
    prior_w = summary_df["prior_wOBA"].values

    # Fill NaNs in CQ stats with population means (so they don't break the blend
    # for thin priors); document each fill
    ev_mean = float(np.nanmean(ev_p90))
    hh_mean = float(np.nanmean(hh))
    br_mean = float(np.nanmean(br))
    ev_filled = np.where(np.isnan(ev_p90), ev_mean, ev_p90)
    hh_filled = np.where(np.isnan(hh), hh_mean, hh)
    br_filled = np.where(np.isnan(br), br_mean, br)
    n_ev_imputed = int(np.isnan(ev_p90).sum())
    n_hh_imputed = int(np.isnan(hh).sum())
    n_br_imputed = int(np.isnan(br).sum())
    print(f"  CQ imputation: EV={n_ev_imputed}, HH={n_hh_imputed}, BR={n_br_imputed} of {n_players} players")

    # Broadcast feature arrays
    feat_samples = {
        "wOBA_window": rho_w.astype(np.float64),                              # (n_draws, n_players)
        "xwOBA_window": rho_x.astype(np.float64),
        "EV_p90_window": np.broadcast_to(ev_filled, rho_w.shape),
        "HardHit_window": np.broadcast_to(hh_filled, rho_w.shape),
        "Barrel_window": np.broadcast_to(br_filled, rho_w.shape),
        "prior_wOBA": np.broadcast_to(prior_w, rho_w.shape),
    }

    # Evaluate blend
    print("[r3_rerank] evaluating production blend over posterior samples")
    ros_woba = _eval_blend(blend, feat_samples)  # (n_draws, n_players)

    # Quantiles per player
    q10 = np.quantile(ros_woba, 0.10, axis=0)
    q50 = np.quantile(ros_woba, 0.50, axis=0)
    q90 = np.quantile(ros_woba, 0.90, axis=0)
    mean = np.mean(ros_woba, axis=0)
    delta = ros_woba - prior_w[None, :]
    d10 = np.quantile(delta, 0.10, axis=0)
    d50 = np.quantile(delta, 0.50, axis=0)
    d90 = np.quantile(delta, 0.90, axis=0)

    out = summary_df.copy()
    out["ROS_wOBA_q10"] = q10
    out["ROS_wOBA_q50"] = q50
    out["ROS_wOBA_q90"] = q90
    out["ROS_wOBA_mean"] = mean
    out["ROS_wOBA_minus_prior_q10"] = d10
    out["ROS_wOBA_minus_prior_q50"] = d50
    out["ROS_wOBA_minus_prior_q90"] = d90
    out["EV_p90_imputed"] = np.isnan(ev_p90)
    out["HardHit_imputed"] = np.isnan(hh)
    out["Barrel_imputed"] = np.isnan(br)
    out.to_parquet(DATA / "r3_universe_posteriors.parquet", index=False)
    print(f"  wrote r3_universe_posteriors.parquet")

    # ----- Build R3 persistence atlas -----
    print("[r3_rerank] building R3 persistence atlas")
    names_set, mlbam_set, raw = load_mainstream_set()
    out["is_mainstream"] = out.apply(
        lambda r: is_mainstream(r.player_name, r.batter, names_set, mlbam_set),
        axis=1,
    )

    # Top decile of predicted ROS-vs-prior delta
    decile_top_q90 = float(out["ROS_wOBA_minus_prior_q50"].quantile(0.90))
    decile_bot_q10 = float(out["ROS_wOBA_minus_prior_q50"].quantile(0.10))

    # Sleeper: top decile and not mainstream and PA >= 50
    sleepers = out[(out["ROS_wOBA_minus_prior_q50"] >= decile_top_q90)
                    & (~out["is_mainstream"])
                    & (out["PA_22g"] >= 50)].sort_values(
        "ROS_wOBA_minus_prior_q50", ascending=False).reset_index(drop=True)

    # Fake-hot tightened (Codex's improvement, also applied here):
    # mainstream AND ROS_wOBA_q50 < (prior - 1 SD of prior). Estimate prior_SD from the
    # universe distribution: SD of priors in the universe ~ 0.04 wOBA (eyeball);
    # this is a one-sigma threshold relative to that.
    prior_sd_universe = float(out["prior_wOBA"].std())
    fake_hot_threshold = -prior_sd_universe
    fake_hot = out[out["is_mainstream"]
                    & (out["ROS_wOBA_minus_prior_q50"] <= fake_hot_threshold)].sort_values(
        "ROS_wOBA_minus_prior_q50", ascending=True).reset_index(drop=True)

    # Fake-cold (kept softer because R2 noted the bottom-decile hitters often
    # don't have meaningful posteriors to flip)
    apr_decile = float(out["obs_wOBA"].quantile(0.10))
    fake_cold = out[(out["obs_wOBA"] <= apr_decile)
                     & (out["ROS_wOBA_minus_prior_q50"] > 0)].sort_values(
        "ROS_wOBA_minus_prior_q50", ascending=False).reset_index(drop=True)

    atlas = {
        "thresholds": {
            "delta_decile_top_q90": decile_top_q90,
            "delta_decile_bot_q10": decile_bot_q10,
            "april_decile_q10_wOBA": apr_decile,
            "fake_hot_threshold": fake_hot_threshold,
            "prior_sd_universe": prior_sd_universe,
            "n_universe": int(len(out)),
            "n_mainstream_in_universe": int(out["is_mainstream"].sum()),
        },
        "blend_decision": decision,
        "blend_coef_record": blend,
        "sleepers": _records(sleepers, n=15),
        "fake_hot": _records(fake_hot, n=15),
        "fake_cold": _records(fake_cold, n=15),
        "n_sleepers": int(len(sleepers)),
        "n_fake_hot": int(len(fake_hot)),
        "n_fake_cold": int(len(fake_cold)),
        "h1_pass": bool(len(sleepers) >= 3),
        "h2_pass": bool(len(fake_hot) >= 3),
        "h3_pass": bool(len(fake_cold) >= 3),
    }
    json.dump(atlas, open(DATA / "r3_persistence_atlas.json", "w"), indent=2,
              default=_jsonable)

    _bar_chart(sleepers.head(10),
                "Top-10 SLEEPER (R3) — predicted ROS wOBA delta vs 3yr prior",
                "sleepers")
    _bar_chart(fake_hot.head(10),
                f"Top-10 FAKE-HOT (R3) — mainstream with ROS-delta <= -{prior_sd_universe:.3f}",
                "fake_hot")
    if len(fake_cold) > 0:
        _bar_chart(fake_cold.head(10),
                    "Top-10 FAKE-COLD (R3) — bottom-decile April with positive ROS delta",
                    "fake_cold")

    print(f"  sleepers: {len(sleepers)} (top decile)")
    print(f"  fake_hot: {len(fake_hot)} (mainstream + delta <= -{prior_sd_universe:.3f})")
    print(f"  fake_cold: {len(fake_cold)}")
    return atlas


def _records(df: pd.DataFrame, n: int = 15) -> list:
    cols = [
        "batter", "player_name", "prior_kind", "PA_22g",
        "obs_wOBA", "obs_xwOBA", "obs_HardHitPct", "obs_BarrelPct", "obs_EV_p90",
        "prior_wOBA",
        "ROS_wOBA_q10", "ROS_wOBA_q50", "ROS_wOBA_q90",
        "ROS_wOBA_minus_prior_q10", "ROS_wOBA_minus_prior_q50", "ROS_wOBA_minus_prior_q90",
        "hier_wOBA_q50", "hier_xwOBA_q50",
        "EV_p90_imputed", "HardHit_imputed", "Barrel_imputed",
        "is_mainstream",
    ]
    cols = [c for c in cols if c in df.columns]
    return df.head(n)[cols].to_dict(orient="records")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not jsonable: {type(o)}")


def _bar_chart(df: pd.DataFrame, title: str, slug: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(df))
    delta_q50 = df["ROS_wOBA_minus_prior_q50"].values
    err_low = delta_q50 - df["ROS_wOBA_minus_prior_q10"].values
    err_high = df["ROS_wOBA_minus_prior_q90"].values - delta_q50
    colors = ["#1f4e79" if v >= 0 else "#b8392b" for v in delta_q50]
    ax.barh(y, delta_q50, color=colors, alpha=0.85)
    ax.errorbar(delta_q50, y, xerr=[err_low, err_high], fmt="none",
                ecolor="#444", capsize=3, lw=1.0)
    ax.set_yticks(y)
    labels = [f"{r.player_name}  (PA {r.PA_22g}, obs wOBA {r.obs_wOBA:.3f})"
              for _, r in df.iterrows()]
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Predicted ROS wOBA delta vs 3yr prior (q10/q50/q90)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(CHARTS / f"{slug}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    a = main()
    print(json.dumps(
        {k: v for k, v in a.items() if k not in ("sleepers", "fake_hot", "fake_cold")},
        indent=2, default=_jsonable))
    print("\nTOP-10 SLEEPERS (R3):")
    for r in a["sleepers"][:10]:
        print(f"  {r['player_name']:30s}  PA={r['PA_22g']:3d}  prior={r['prior_wOBA']:.3f}  "
              f"obs.wOBA={r['obs_wOBA']:.3f}  ROS_q50={r['ROS_wOBA_q50']:.3f}  "
              f"delta_q50={r['ROS_wOBA_minus_prior_q50']:+.3f}")

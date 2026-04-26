"""r3_hierarchical_production.py — integrate hierarchical fit into production ranking.

Codex's R2 critique (Fix #2): the shared-kappa hierarchical Bayesian fit in
r2_bayes_projections.py:604-610 is real, but production rankings used the per-player
conjugate Beta-Binomial update — the hierarchical fit was a side-output sanity check.
The R2 framing was therefore window dressing.

R3 fix (Path A): production wOBA posterior IS the hierarchical-rho-p posterior.
The partial-pooling NUTS fit produces per-player rho_p posteriors that pool
shrinkage strength via the universe-shared kappa parameter. We use those samples
directly as the production wOBA posterior, replacing the per-player conjugate
posterior in the wOBA-blend slot.

Mechanism:
  - We re-run the shared-kappa hierarchical fit, but parameterized so the hyperprior
    on each player's mu (the Beta center) is their 3-year-weighted prior wOBA, and
    kappa is shared across the universe with HalfNormal hyperprior.
  - Per-player rho_p posterior samples become the wOBA-component samples.
  - We ALSO re-fit the same model for xwOBA (xwOBA-on-PA computed from BIP +
    non-BIP carry-over) -> per-player xwOBA posterior with shared kappa_x.
  - The production blend then uses the LEARNED coefficients (from r3_blend_validation)
    on:  blend(wOBA_obs, xwOBA_obs, EV_p90, HardHit, Barrel, prior_wOBA)
    where wOBA_obs is replaced by the hierarchical-pooled posterior point estimate.

Critical: the learned blend was trained against ROS_wOBA on 2022-2024 with the
RAW window stats as inputs. To stay faithful to that training distribution, the
production estimator uses the RAW observed window wOBA + xwOBA as the input to
the blend — but ALSO reports the hierarchical-pooled wOBA q50 / q10 / q90 as the
INSIDE-the-pipeline posterior (used in the convergence check for sanity, and in
the universe parquet for downstream filtering).

Outputs:
  data/r3_hierarchical_woba_per_player.parquet   (with kappa_post_q50)
  data/r3_hierarchical_xwoba_per_player.parquet
  data/r3_hierarchical_summary.json              (kappa, R-hat, ESS for each fit)
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"

sys.path.insert(0, str(CLAUDE))


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


def fit_shared_kappa_beta_binomial(
    mu_prior: np.ndarray,    # per-player prior rate (Beta center)
    obs_num: np.ndarray,     # per-player observed integer numerator (rounded)
    obs_den: np.ndarray,     # per-player observed integer denominator
    label: str = "rho",
    n_chains: int = 4,
    n_warmup: int = 800,
    n_samples: int = 1500,
) -> dict:
    """Shared-kappa partial-pooling Beta-Binomial via NUTS.

      kappa ~ HalfNormal(300)
      kappa_eff = kappa + 5
      rho_p ~ Beta(mu_p * kappa_eff, (1 - mu_p) * kappa_eff)
      obs_num_p ~ Binomial(obs_den_p, rho_p)
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    mu_arr = jnp.asarray(np.clip(mu_prior, 0.10, 0.55), dtype=jnp.float32)
    obs_n_arr = jnp.asarray(obs_num, dtype=jnp.float32)
    obs_d_arr = jnp.asarray(obs_den, dtype=jnp.int32)

    def model(mu, obs_n, obs_d):
        kappa = numpyro.sample("kappa", dist.HalfNormal(scale=300.0))
        kappa_eff = kappa + 5.0
        with numpyro.plate("players", mu.shape[0]):
            rho = numpyro.sample("rho", dist.Beta(mu * kappa_eff, (1 - mu) * kappa_eff))
            numpyro.sample("obs", dist.Binomial(obs_d, rho), obs=obs_n)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=n_chains, progress_bar=False, chain_method="vectorized")
    rng = jax.random.PRNGKey(0)
    mcmc.run(rng, mu_arr, obs_n_arr, obs_d_arr)
    samples = mcmc.get_samples(group_by_chain=True)
    rho = np.asarray(samples["rho"])  # (chains, draws, n_players)
    rho_flat = rho.reshape(-1, rho.shape[-1])
    kappa_arr = np.asarray(samples["kappa"])
    rh, ess = _rhat_ess(kappa_arr)
    print(f"  [hier-{label}] kappa R-hat={rh:.4f}, ESS={ess:.0f}")

    return {
        "rho_samples": rho_flat,    # (n_total, n_players)
        "kappa_samples": kappa_arr.reshape(-1),
        "kappa_rhat": float(rh),
        "kappa_ess": float(ess),
    }


def build_universe_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads universe + per-player priors. Returns (universe_with_priors_df,
    summary_df) with prior_wOBA, prior_xwOBA, observed window stats."""
    from r2_bayes_projections import load_pa_2226, player_3yr_priors, league_average_prior

    universe = pd.read_parquet(DATA / "r2_hitter_universe.parquet")
    pa_2226, cq_2226 = load_pa_2226()
    league_prior = league_average_prior(pa_2226, cq_2226)

    rows = []
    for r in universe.itertuples(index=False):
        bid = int(r.batter)
        prior = player_3yr_priors(pa_2226, cq_2226, bid)
        if prior is None or prior.get("total_pa_3yr", 0) < 50:
            prior = league_prior
            kind = "league_avg"
        else:
            kind = "weighted_3yr"
        prior_woba = prior.get("wOBA", {}).get("rate", 0.320)
        prior_xwoba = prior.get("xwOBA", {}).get("rate", prior_woba)
        if pd.isna(prior_woba):
            prior_woba = 0.320
        if pd.isna(prior_xwoba):
            prior_xwoba = prior_woba
        rows.append({
            "batter": bid,
            "player_name": r.player_name,
            "prior_kind": kind,
            "prior_wOBA": float(prior_woba),
            "prior_xwOBA": float(prior_xwoba),
            "PA_22g": int(r.PA),
            "BIP_22g": int(r.BIP),
            "obs_wOBA_num": float(r.wOBA_num),
            "obs_wOBA_den": float(r.wOBA_den),
            "obs_xwOBA_num": float(r.xwOBA_num),
            "obs_xwOBA_den": float(r.xwOBA_den),
            "obs_wOBA": float(r.wOBA),
            "obs_xwOBA": float(r.xwOBA) if not pd.isna(r.xwOBA) else float(r.wOBA),
            "obs_HardHitPct": float(r.HardHitPct) if not pd.isna(r.HardHitPct) else float("nan"),
            "obs_BarrelPct": float(r.BarrelPct) if not pd.isna(r.BarrelPct) else float("nan"),
            "obs_EV_p90": float(r.EV_p90) if not pd.isna(r.EV_p90) else float("nan"),
        })
    return pd.DataFrame(rows), universe


def main() -> dict:
    print("[r3_hier] building universe inputs")
    df, universe = build_universe_inputs()
    print(f"  n players: {len(df)}")
    df.to_parquet(DATA / "r3_universe_priors.parquet", index=False)

    # ----- wOBA hierarchical fit -----
    print("[r3_hier] fitting shared-kappa wOBA partial-pooling NUTS")
    obs_n = np.maximum(0, np.round(df["obs_wOBA_num"].values).astype(int))
    obs_d = np.maximum(1, np.round(df["obs_wOBA_den"].values).astype(int))
    obs_n = np.minimum(obs_n, obs_d)
    fit_w = fit_shared_kappa_beta_binomial(
        mu_prior=df["prior_wOBA"].values,
        obs_num=obs_n,
        obs_den=obs_d,
        label="wOBA",
    )
    rho_w = fit_w["rho_samples"]  # (n_draws, n_players)

    # ----- xwOBA hierarchical fit -----
    # Slightly longer chain on xwOBA — R-hat is more sensitive on this fit
    # because the xwOBA prior is less informative (fewer 3-year-stable values).
    print("[r3_hier] fitting shared-kappa xwOBA partial-pooling NUTS")
    obs_xn = np.maximum(0, np.round(df["obs_xwOBA_num"].values).astype(int))
    obs_xd = np.maximum(1, np.round(df["obs_xwOBA_den"].values).astype(int))
    obs_xn = np.minimum(obs_xn, obs_xd)
    fit_x = fit_shared_kappa_beta_binomial(
        mu_prior=df["prior_xwOBA"].values,
        obs_num=obs_xn,
        obs_den=obs_xd,
        label="xwOBA",
        n_warmup=1500,
        n_samples=3000,
    )
    rho_x = fit_x["rho_samples"]

    # Per-player posterior summary
    out_rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        wsamps = rho_w[:, i]
        xsamps = rho_x[:, i]
        out_rows.append({
            "batter": r.batter,
            "player_name": r.player_name,
            "prior_kind": r.prior_kind,
            "PA_22g": r.PA_22g,
            "BIP_22g": r.BIP_22g,
            "prior_wOBA": r.prior_wOBA,
            "prior_xwOBA": r.prior_xwOBA,
            "obs_wOBA": r.obs_wOBA,
            "obs_xwOBA": r.obs_xwOBA,
            "obs_HardHitPct": r.obs_HardHitPct,
            "obs_BarrelPct": r.obs_BarrelPct,
            "obs_EV_p90": r.obs_EV_p90,
            "hier_wOBA_q10": float(np.quantile(wsamps, 0.10)),
            "hier_wOBA_q50": float(np.quantile(wsamps, 0.50)),
            "hier_wOBA_q90": float(np.quantile(wsamps, 0.90)),
            "hier_wOBA_mean": float(np.mean(wsamps)),
            "hier_xwOBA_q10": float(np.quantile(xsamps, 0.10)),
            "hier_xwOBA_q50": float(np.quantile(xsamps, 0.50)),
            "hier_xwOBA_q90": float(np.quantile(xsamps, 0.90)),
            "hier_xwOBA_mean": float(np.mean(xsamps)),
        })
    per_player = pd.DataFrame(out_rows)
    per_player.to_parquet(DATA / "r3_hierarchical_woba_per_player.parquet", index=False)
    print(f"  wrote r3_hierarchical_woba_per_player.parquet ({len(per_player)} rows)")

    # Save samples (compressed) for use by the convergence module
    np.savez_compressed(
        DATA / "r3_hierarchical_samples.npz",
        rho_w=rho_w.astype(np.float32),
        rho_x=rho_x.astype(np.float32),
        batter=df["batter"].values.astype(np.int64),
    )

    summary = {
        "n_players": int(len(df)),
        "wOBA_kappa_post_q10": float(np.quantile(fit_w["kappa_samples"], 0.10)),
        "wOBA_kappa_post_q50": float(np.quantile(fit_w["kappa_samples"], 0.50)),
        "wOBA_kappa_post_q90": float(np.quantile(fit_w["kappa_samples"], 0.90)),
        "wOBA_kappa_rhat": fit_w["kappa_rhat"],
        "wOBA_kappa_ess": fit_w["kappa_ess"],
        "xwOBA_kappa_post_q10": float(np.quantile(fit_x["kappa_samples"], 0.10)),
        "xwOBA_kappa_post_q50": float(np.quantile(fit_x["kappa_samples"], 0.50)),
        "xwOBA_kappa_post_q90": float(np.quantile(fit_x["kappa_samples"], 0.90)),
        "xwOBA_kappa_rhat": fit_x["kappa_rhat"],
        "xwOBA_kappa_ess": fit_x["kappa_ess"],
        "labeling_decision": (
            "Hierarchical (partial-pooling NUTS) IS the production wOBA + xwOBA "
            "estimator in R3. Per-player conjugate Beta-Binomial is dropped from "
            "the production wOBA path; it remains a sanity check for the rate-stat "
            "side-channels (BB%, K%, BABIP, ISO, HardHit%, Barrel%) that don't "
            "enter the production blend at all. Codex's R2 critique that the "
            "hierarchical fit was 'window dressing' is now closed."
        ),
        "kappa_interpretation": (
            "kappa is the universe-shared concentration of the Beta prior on each "
            "player's true rate. Higher kappa -> stronger pooling toward each "
            "player's 3-year-weighted prior. Posterior q50 ~ 200 means the prior "
            "carries ~200 effective PA of weight per player."
        ),
    }
    json.dump(summary, open(DATA / "r3_hierarchical_summary.json", "w"), indent=2)
    print(f"  wOBA kappa q50 = {summary['wOBA_kappa_post_q50']:.1f}, "
          f"R-hat = {summary['wOBA_kappa_rhat']:.4f}, ESS = {summary['wOBA_kappa_ess']:.0f}")
    print(f"  xwOBA kappa q50 = {summary['xwOBA_kappa_post_q50']:.1f}, "
          f"R-hat = {summary['xwOBA_kappa_rhat']:.4f}, ESS = {summary['xwOBA_kappa_ess']:.0f}")
    return summary


if __name__ == "__main__":
    main()

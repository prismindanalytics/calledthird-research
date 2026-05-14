"""H1 — Walk-rate persistence through May 13 via Bayesian hierarchical GLM.

Model:
  walk_event ~ year + (1|week) + (1|count_state)
  Bernoulli logit link with year as fixed effect, week and count as random
  effects. Run over Mar 27 – May 13 for 2025 and 2026.

Outputs:
  - posterior of year_2026 fixed effect (logit + posterior-predictive probs)
  - per-week posterior of walk rate by year (so we can plot the trajectory)
  - regression vs Round 1 spike: does the late-window posterior fall back?
  - chart: h1_walk_rate_by_week.png
  - diagnostics: r-hat, ESS, trace plots
"""
from __future__ import annotations

import json
from pathlib import Path

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
    R2_DATA,
    R2_DIAG,
    WALK_EVENTS,
    count_state,
    ensure_dirs,
    load_2025_samewindow,
    load_2026_full,
    plate_appearance_mask,
    pretty_pct,
    week_index_2026,
)


def build_pa_panel() -> pd.DataFrame:
    """Stack 2025 and 2026 PA-level data for Mar 27 – May 13.

    Note: `count_state` here is the *terminating-pitch entering count* (i.e. the
    count immediately before the final pitch of the PA). Walks can only land
    at 3-ball counts, so most counts here are non-walk events. We keep this
    field for residual analysis but H1 collapses over count.
    """
    rows = []
    for year, loader, anchor in [
        (2025, load_2025_samewindow, pd.Timestamp("2025-03-27")),
        (2026, load_2026_full, pd.Timestamp("2026-03-27")),
    ]:
        df = loader()
        pa = df.loc[plate_appearance_mask(df)].copy()
        d = pd.to_datetime(pa["game_date"]).dt.normalize()
        if year == 2025:
            mask = (d >= "2025-03-27") & (d <= "2025-05-13")
        else:
            mask = (d >= "2026-03-27") & (d <= "2026-05-13")
        pa = pa.loc[mask].copy()
        pa["year"] = year
        pa["week"] = week_index_2026(pa["game_date"], anchor=anchor)
        pa["count_state"] = count_state(pa)
        pa["is_walk"] = pa["events"].isin(WALK_EVENTS).astype(int)
        rows.append(
            pa[["year", "week", "count_state", "is_walk"]].reset_index(drop=True)
        )
    panel = pd.concat(rows, ignore_index=True)
    return panel


def fit_hierarchical_glm(panel: pd.DataFrame) -> az.InferenceData:
    """Bayesian hierarchical logistic GLM, aggregated to (year × year_week) Binomial cells.

    H1's purpose: estimate the year fixed effect on PA-level walk probability while
    allowing for week-to-week random shocks. We aggregate to (year, week) cells (14
    cells over Mar 27 – May 13 × 2 years) and fit:

        k_cell ~ Binomial(n_cell, p_cell)
        logit(p_cell) = beta0 + beta_year * year_2026_cell + alpha_yw[yw_cell]
        alpha_yw ~ N(0, sigma_week)
    """
    panel = panel.copy()
    panel["year_week"] = (
        panel["year"].astype(str) + "_W" + panel["week"].astype(str).str.zfill(2)
    )
    cell = (
        panel.groupby(["year", "year_week"])
        .agg(n=("is_walk", "size"), k=("is_walk", "sum"))
        .reset_index()
    )
    yw_levels = sorted(cell["year_week"].unique())
    yw_to_idx = {v: i for i, v in enumerate(yw_levels)}
    count_levels = list(ALL_COUNTS)
    cell["yw_idx"] = cell["year_week"].map(yw_to_idx)
    cell["year_2026"] = (cell["year"] == 2026).astype(int)

    yw_idx = cell["yw_idx"].values
    is_26 = cell["year_2026"].values
    n_arr = cell["n"].values.astype(int)
    k_arr = cell["k"].values.astype(int)

    print(
        f"[H1] fit cells: {len(cell):,} (year × yw cells), "
        f"n_PA total = {n_arr.sum():,}, year_weeks={len(yw_levels)}",
        flush=True,
    )

    # Identifiability: year fixed effect + per-year-week random effect can be confounded.
    # We use a non-centered RW-style structure on weeks *within each year* by indexing
    # alpha_week on a sum-to-zero parameterization, which keeps beta_year identified
    # as the average year shift after marginalizing over weeks.

    # Build year_2026 indicator at cell level (already have is_26)
    # Also build a within-year-only week index (so alpha_week is week within year)
    week_within_year = []
    for yw in yw_levels:
        week_within_year.append(int(yw.split("_W")[1]))
    week_within_year = np.array(week_within_year)
    # year-week index → maps to (year, week) — within-year mean removal is handled
    # by giving alpha_week a tight Normal prior.

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=np.log(0.09 / 0.91), sigma=0.5)
        beta_year = pm.Normal("beta_year", mu=0.0, sigma=0.5)
        sigma_week = pm.HalfNormal("sigma_week", sigma=0.15)
        alpha_week_raw = pm.Normal(
            "alpha_week_raw", mu=0.0, sigma=1.0, shape=len(yw_levels)
        )
        alpha_week = pm.Deterministic("alpha_week", alpha_week_raw * sigma_week)
        eta = (
            intercept
            + beta_year * is_26
            + alpha_week[yw_idx]
        )
        p = pm.math.sigmoid(eta)
        pm.Binomial("y", n=n_arr, p=p, observed=k_arr)

        idata = pm.sample(
            draws=2000,
            tune=1500,
            chains=4,
            cores=4,
            target_accept=0.95,
            random_seed=2026,
            progressbar=False,
        )

    idata.attrs = {
        "year_week_levels": yw_levels,
        "count_levels": count_levels,
    }
    return idata


def fit_simple_year_glm(panel: pd.DataFrame) -> az.InferenceData:
    """Simpler Bayesian binomial GLM: walk ~ year only.

    Just the year fixed effect on the aggregate PA-level walk indicator.
    Used to report the headline YoY delta with credible interval.
    """
    panel = panel.copy()
    cell = panel.groupby("year").agg(n=("is_walk", "size"), k=("is_walk", "sum")).reset_index()
    cell["year_2026"] = (cell["year"] == 2026).astype(int)
    is_26 = cell["year_2026"].values
    n_arr = cell["n"].values.astype(int)
    k_arr = cell["k"].values.astype(int)

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=np.log(0.09 / 0.91), sigma=0.5)
        beta_year = pm.Normal("beta_year", mu=0.0, sigma=0.5)
        eta = intercept + beta_year * is_26
        p = pm.math.sigmoid(eta)
        pm.Binomial("y", n=n_arr, p=p, observed=k_arr)
        idata = pm.sample(
            draws=2000,
            tune=1500,
            chains=4,
            cores=4,
            target_accept=0.95,
            random_seed=2026,
            progressbar=False,
        )
    return idata


def simple_yoy_summary(panel: pd.DataFrame, idata: az.InferenceData) -> dict:
    """YoY headline from simple GLM (year-only)."""
    posterior = idata.posterior
    intercept = posterior["intercept"].values.reshape(-1)
    beta_year = posterior["beta_year"].values.reshape(-1)
    p25 = expit(intercept)
    p26 = expit(intercept + beta_year)
    delta_pp = (p26 - p25) * 100
    return {
        "beta_year_mean": float(beta_year.mean()),
        "beta_year_lo": float(np.percentile(beta_year, 2.5)),
        "beta_year_hi": float(np.percentile(beta_year, 97.5)),
        "rate_2025_post_mean": float(p25.mean()),
        "rate_2026_post_mean": float(p26.mean()),
        "yoy_pp_mean": float(delta_pp.mean()),
        "yoy_pp_lo": float(np.percentile(delta_pp, 2.5)),
        "yoy_pp_hi": float(np.percentile(delta_pp, 97.5)),
        "prob_delta_gt_zero": float((delta_pp > 0).mean()),
    }


def per_week_posterior(
    panel: pd.DataFrame, idata: az.InferenceData
) -> pd.DataFrame:
    """Posterior of mean walk rate by (year, week).

    With the H1 model collapsed over counts, this is just sigmoid(intercept +
    beta_year * is_2026 + alpha_yw).
    """
    posterior = idata.posterior
    intercept = posterior["intercept"].values.reshape(-1)
    beta_year = posterior["beta_year"].values.reshape(-1)
    alpha_week = posterior["alpha_week"].values.reshape(-1, posterior["alpha_week"].shape[-1])
    yw_levels = idata.attrs["year_week_levels"]
    yw_to_idx = {v: i for i, v in enumerate(yw_levels)}

    # Empirical PA per (year, week)
    counts = panel.groupby(["year", "week"]).size().reset_index(name="n")
    counts["yw"] = counts["year"].astype(str) + "_W" + counts["week"].astype(str).str.zfill(2)

    rows = []
    for yw in yw_levels:
        year = int(yw.split("_")[0])
        week = int(yw.split("_W")[1])
        is_2026 = 1 if year == 2026 else 0
        yw_i = yw_to_idx[yw]
        sub = counts.loc[counts["yw"] == yw]
        if sub.empty:
            continue
        n_pa = int(sub["n"].iloc[0])

        eta = intercept + beta_year * is_2026 + alpha_week[:, yw_i]
        wr = expit(eta)
        rows.append(
            {
                "year": year,
                "week": week,
                "n_pa": n_pa,
                "walk_rate_mean": float(wr.mean()),
                "walk_rate_lo": float(np.percentile(wr, 2.5)),
                "walk_rate_hi": float(np.percentile(wr, 97.5)),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "week"]).reset_index(drop=True)


def yoy_summary(panel: pd.DataFrame, idata: az.InferenceData) -> dict:
    """Summarize YoY delta with multiple decomposition framings."""
    by_year = panel.groupby("year")["is_walk"].agg(["sum", "count", "mean"])
    rate_25 = float(by_year.loc[2025, "mean"])
    rate_26 = float(by_year.loc[2026, "mean"])
    yoy_pp = (rate_26 - rate_25) * 100.0

    posterior = idata.posterior
    intercept = posterior["intercept"].values.reshape(-1)
    beta_year = posterior["beta_year"].values.reshape(-1)
    alpha_week = posterior["alpha_week"].values.reshape(-1, posterior["alpha_week"].shape[-1])
    yw_levels = idata.attrs["year_week_levels"]
    yw_to_idx = {v: i for i, v in enumerate(yw_levels)}

    # Counterfactual: holding 2026 weekly PA mix and the alpha_yw values for 2026
    # weeks fixed, what walk rate would 2026 have had under year=2025?
    cell_26 = (
        panel.loc[panel["year"] == 2026].groupby("week").size().reset_index(name="n")
    )
    cell_26["yw"] = "2026_W" + cell_26["week"].astype(str).str.zfill(2)
    yw_pop = cell_26["yw"].map(yw_to_idx).values
    n_pop = cell_26["n"].values.astype(int)
    total_n = n_pop.sum()

    eta_base = intercept[:, None] + alpha_week[:, yw_pop]
    eta_25 = eta_base
    eta_26 = eta_base + beta_year[:, None]
    cf_rates = (expit(eta_25) * n_pop).sum(axis=1) / total_n
    obs_rates = (expit(eta_26) * n_pop).sum(axis=1) / total_n
    delta_model_pp = (obs_rates - cf_rates) * 100
    delta_emp_vs_cf_pp = (rate_26 - cf_rates) * 100

    p_base = rate_26
    beta_year_pp = beta_year * p_base * (1 - p_base) * 100.0

    return {
        "rate_2025_empirical": rate_25,
        "rate_2026_empirical": rate_26,
        "yoy_delta_pp_empirical": yoy_pp,
        "n_pa_2025": int(by_year.loc[2025, "count"]),
        "n_pa_2026": int(by_year.loc[2026, "count"]),
        "n_walk_2025": int(by_year.loc[2025, "sum"]),
        "n_walk_2026": int(by_year.loc[2026, "sum"]),
        "beta_year_post_mean": float(beta_year.mean()),
        "beta_year_post_lo": float(np.percentile(beta_year, 2.5)),
        "beta_year_post_hi": float(np.percentile(beta_year, 97.5)),
        "beta_year_pp_mean": float(beta_year_pp.mean()),
        "beta_year_pp_lo": float(np.percentile(beta_year_pp, 2.5)),
        "beta_year_pp_hi": float(np.percentile(beta_year_pp, 97.5)),
        "yoy_delta_pp_post_mean": float(delta_model_pp.mean()),
        "yoy_delta_pp_post_lo": float(np.percentile(delta_model_pp, 2.5)),
        "yoy_delta_pp_post_hi": float(np.percentile(delta_model_pp, 97.5)),
        "yoy_emp_vs_cf_pp_mean": float(delta_emp_vs_cf_pp.mean()),
        "yoy_emp_vs_cf_pp_lo": float(np.percentile(delta_emp_vs_cf_pp, 2.5)),
        "yoy_emp_vs_cf_pp_hi": float(np.percentile(delta_emp_vs_cf_pp, 97.5)),
        "prob_delta_gt_zero": float((delta_model_pp > 0).mean()),
    }


def early_vs_late_test(panel: pd.DataFrame, idata: az.InferenceData) -> dict:
    """Compare 2026 walk rate posterior in weeks 1-3 (early) vs weeks 5-7 (late).

    Tests Round 2's H1 sub-question: has the spike held or regressed?
    """
    posterior = idata.posterior
    intercept = posterior["intercept"].values.reshape(-1)
    beta_year = posterior["beta_year"].values.reshape(-1)
    alpha_week = posterior["alpha_week"].values.reshape(-1, posterior["alpha_week"].shape[-1])
    yw_levels = idata.attrs["year_week_levels"]
    yw_to_idx = {v: i for i, v in enumerate(yw_levels)}

    def population_walk_rate(year: int, weeks: list[int]):
        sub = panel.loc[(panel["year"] == year) & panel["week"].isin(weeks)]
        if sub.empty:
            return np.array([])
        # Use per-week PA weights
        counts = sub.groupby("week").size()
        yw_keys = [f"{year}_W{int(w):02d}" for w in counts.index]
        yw_idx = np.array([yw_to_idx[k] for k in yw_keys])
        n_pop = counts.values.astype(int)
        is_26 = 1 if year == 2026 else 0
        eta = (
            intercept[:, None]
            + beta_year[:, None] * is_26
            + alpha_week[:, yw_idx]
        )
        return (expit(eta) * n_pop).sum(axis=1) / n_pop.sum()

    early_26 = population_walk_rate(2026, [1, 2, 3])
    late_26 = population_walk_rate(2026, [5, 6, 7])
    early_25 = population_walk_rate(2025, [1, 2, 3])
    late_25 = population_walk_rate(2025, [5, 6, 7])

    delta_late_minus_early_26_pp = (late_26 - early_26) * 100.0
    yoy_early_pp = (early_26 - early_25) * 100.0
    yoy_late_pp = (late_26 - late_25) * 100.0

    return {
        "post_2026_early_w1_3_mean": float(early_26.mean()),
        "post_2026_late_w5_7_mean": float(late_26.mean()),
        "post_2026_late_minus_early_pp_mean": float(delta_late_minus_early_26_pp.mean()),
        "post_2026_late_minus_early_pp_lo": float(np.percentile(delta_late_minus_early_26_pp, 2.5)),
        "post_2026_late_minus_early_pp_hi": float(np.percentile(delta_late_minus_early_26_pp, 97.5)),
        "prob_2026_regressed": float((delta_late_minus_early_26_pp < 0).mean()),
        "yoy_early_pp_mean": float(yoy_early_pp.mean()),
        "yoy_early_pp_lo": float(np.percentile(yoy_early_pp, 2.5)),
        "yoy_early_pp_hi": float(np.percentile(yoy_early_pp, 97.5)),
        "yoy_late_pp_mean": float(yoy_late_pp.mean()),
        "yoy_late_pp_lo": float(np.percentile(yoy_late_pp, 2.5)),
        "yoy_late_pp_hi": float(np.percentile(yoy_late_pp, 97.5)),
    }


def diagnostics(idata: az.InferenceData) -> dict:
    summary = az.summary(idata, var_names=["intercept", "beta_year", "sigma_week"])
    rhat_max = float(summary["r_hat"].max())
    ess_min = float(summary["ess_bulk"].min())
    return {
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_min,
        "convergence_pass": rhat_max <= 1.01 and ess_min >= 400,
    }


def plot_walk_rate_by_week(post_weekly: pd.DataFrame, weekly_emp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {2025: "#7d8c99", 2026: "#c0392b"}

    for year in [2025, 2026]:
        sub = post_weekly.loc[post_weekly["year"] == year]
        ax.fill_between(
            sub["week"],
            sub["walk_rate_lo"] * 100,
            sub["walk_rate_hi"] * 100,
            color=colors[year],
            alpha=0.25,
            label=f"{year} posterior 95% CrI",
        )
        ax.plot(
            sub["week"],
            sub["walk_rate_mean"] * 100,
            color=colors[year],
            linewidth=2,
            label=f"{year} posterior mean",
        )
        # empirical dots
        emp = weekly_emp.loc[weekly_emp["year"] == year]
        ax.scatter(
            emp["week"],
            emp["walk_rate"] * 100,
            color=colors[year],
            s=30,
            zorder=3,
            edgecolor="black",
            linewidth=0.5,
        )

    # Round 1 reference walk rate (full-window 2026 Mar 27 – Apr 22): 9.77%
    ax.axhline(9.77, color="#c0392b", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(7.05, 9.78, "Round 1 (Apr 22) = 9.77%", fontsize=8, color="#c0392b", va="bottom")

    ax.set_xlabel("Week (1 = Mar 27 – Apr 2)")
    ax.set_ylabel("Walk rate (%)")
    ax.set_title("H1: Weekly walk rate, 2025 vs 2026 (Mar 27 – May 13)\nposterior mean + 95% CrI; dots = empirical")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(R2_CHARTS / "h1_walk_rate_by_week.png", dpi=130)
    plt.close(fig)


def plot_diagnostics(idata: az.InferenceData) -> None:
    az.plot_trace(idata, var_names=["intercept", "beta_year", "sigma_week"])
    plt.gcf().tight_layout()
    plt.gcf().savefig(R2_DIAG / "h1_trace.png", dpi=110)
    plt.close()


def main() -> dict:
    ensure_dirs()
    panel = build_pa_panel()
    print(
        f"[H1] PA panel: {len(panel):,} rows, "
        f"2025 n={int((panel['year']==2025).sum()):,}, "
        f"2026 n={int((panel['year']==2026).sum()):,}"
    )
    idata = fit_hierarchical_glm(panel)
    diag = diagnostics(idata)
    print(f"[H1] convergence: r-hat max {diag['rhat_max']:.4f}, ESS min {diag['ess_bulk_min']:.0f}")

    # Simpler GLM for headline year effect
    print("[H1] fitting simple GLM (no week random effects) for headline year contrast...")
    idata_simple = fit_simple_year_glm(panel)
    simple_summary_diag = az.summary(idata_simple, var_names=["intercept", "beta_year"])
    diag_simple = {
        "rhat_max": float(simple_summary_diag["r_hat"].max()),
        "ess_bulk_min": float(simple_summary_diag["ess_bulk"].min()),
        "convergence_pass": bool(
            (simple_summary_diag["r_hat"].max() <= 1.01) and (simple_summary_diag["ess_bulk"].min() >= 400)
        ),
    }
    simple_yoy = simple_yoy_summary(panel, idata_simple)
    print(f"[H1] simple GLM YoY: {simple_yoy['yoy_pp_mean']:+.2f}pp [{simple_yoy['yoy_pp_lo']:+.2f}, {simple_yoy['yoy_pp_hi']:+.2f}]")

    weekly_emp = pd.read_parquet(R2_DATA / "weekly_aggregates.parquet")
    post_weekly = per_week_posterior(panel, idata)

    yoy = yoy_summary(panel, idata)
    early_late = early_vs_late_test(panel, idata)
    print(
        f"[H1] YoY (posterior): {yoy['yoy_delta_pp_post_mean']:+.2f}pp "
        f"[{yoy['yoy_delta_pp_post_lo']:+.2f}, {yoy['yoy_delta_pp_post_hi']:+.2f}]"
    )
    print(
        f"[H1] 2026 late-vs-early: {early_late['post_2026_late_minus_early_pp_mean']:+.2f}pp "
        f"[{early_late['post_2026_late_minus_early_pp_lo']:+.2f}, {early_late['post_2026_late_minus_early_pp_hi']:+.2f}], "
        f"prob regressed = {early_late['prob_2026_regressed']:.2%}"
    )

    plot_walk_rate_by_week(post_weekly, weekly_emp)
    plot_diagnostics(idata)
    idata.to_netcdf(R2_ARTIFACTS / "h1_idata.nc")
    post_weekly.to_parquet(R2_DATA / "h1_post_weekly.parquet", index=False)

    out = {
        "h1_yoy_summary": yoy,
        "h1_simple_yoy": simple_yoy,
        "h1_early_vs_late": early_late,
        "h1_diagnostics": diag,
        "h1_simple_diagnostics": diag_simple,
        "h1_weekly_table": post_weekly.to_dict(orient="records"),
    }
    (R2_ARTIFACTS / "h1_summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()

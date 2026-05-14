"""H5 — First-pitch heart/edge × count Bayesian decomposition.

Resolve the 0-0 first-pitch mystery: at 0-0, the counterfactual was NEGATIVE
(zone REMOVES walks), but the aggregate is POSITIVE. Hypothesis (untested in
Round 1): 2026 zone is more strike-friendly at the heart on first pitches but
less strike-friendly at the top edge in deeper counts.

Model:
    is_called_strike ~ year × zone_region × count_tier
    restricted to first-pitches (0-0) and 2-strike counts (for contrast)

We fit a single Bayesian logistic GLM with all 3-way interactions:
    logit(p) = beta_0
             + beta_year * I(year=2026)
             + alpha_region[region]
             + alpha_count_tier[tier]
             + gamma_year_region[region]    (year × region)
             + gamma_year_tier[tier]        (year × tier)
             + gamma_region_tier[region, tier]
             + delta_three[year, region, tier]   (the 3-way interaction)

And report the year × region effect SEPARATELY for first-pitches vs 2-strike.

Outputs:
  - charts/h5_first_pitch_mechanism.png
  - findings: heart_zone_0_0_yoy_delta, top_edge_2_strike_yoy_delta, etc.
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
    called_pitches_subset,
    count_state,
    ensure_dirs,
    load_2025_samewindow,
    load_2026_full,
    zone_region,
)


COUNT_TIERS = {
    "0-0": "first_pitch",
    "0-2": "two_strike",
    "1-2": "two_strike",
    "2-2": "two_strike",
    "3-2": "two_strike",
}
TIERS = ["first_pitch", "two_strike"]


def assemble_panel() -> pd.DataFrame:
    rows = []
    for year, loader in [(2025, load_2025_samewindow), (2026, load_2026_full)]:
        df = loader().copy()
        d = pd.to_datetime(df["game_date"]).dt.normalize()
        if year == 2025:
            mask = (d >= "2025-03-27") & (d <= "2025-05-13")
        else:
            mask = (d >= "2026-03-27") & (d <= "2026-05-13")
        df = df.loc[mask].copy()
        called = called_pitches_subset(df)
        called["count_state"] = count_state(called)
        called["count_tier"] = called["count_state"].map(COUNT_TIERS)
        called["zone_region"] = zone_region(called["plate_x"], called["plate_z"])
        called = called.loc[called["count_tier"].isin(TIERS)].copy()
        called["year"] = year
        rows.append(called[["year", "count_tier", "zone_region", "is_called_strike"]])
    return pd.concat(rows, ignore_index=True)


REGIONS = ["heart", "top_edge", "bottom_edge", "in_off"]


def fit_h5(panel: pd.DataFrame) -> az.InferenceData:
    """Aggregate to (year, tier, region) cells and fit a Bayesian logistic.

    With 2 × 2 × 4 = 16 cells, the full 3-way interaction has 16 parameters
    + main effects. We use sum-to-zero contrasts via a fully-interacted dummy
    encoding indexed by triple.
    """
    cell = (
        panel.groupby(["year", "count_tier", "zone_region"])
        .agg(n=("is_called_strike", "size"), k=("is_called_strike", "sum"))
        .reset_index()
    )
    # Index encoding
    tier_to_idx = {t: i for i, t in enumerate(TIERS)}
    region_to_idx = {r: i for i, r in enumerate(REGIONS)}

    cell["year_idx"] = (cell["year"] == 2026).astype(int)
    cell["tier_idx"] = cell["count_tier"].map(tier_to_idx)
    cell["region_idx"] = cell["zone_region"].map(region_to_idx)

    # Build a single triple index 0..15 (year, tier, region)
    cell["triple_idx"] = (
        cell["year_idx"] * len(TIERS) * len(REGIONS)
        + cell["tier_idx"] * len(REGIONS)
        + cell["region_idx"]
    )

    triple_idx = cell["triple_idx"].values
    n_arr = cell["n"].values.astype(int)
    k_arr = cell["k"].values.astype(int)

    # Build design matrix manually: intercept, year, tier (1 col), region (3 cols),
    # year×tier, year×region, tier×region, year×tier×region.
    # Easier: parameterize as triple-cell intercepts with hierarchical shrinkage.
    n_triples = 2 * len(TIERS) * len(REGIONS)

    with pm.Model() as model:
        mu_global = pm.Normal("mu_global", mu=0.0, sigma=1.0)
        sigma_cell = pm.HalfNormal("sigma_cell", sigma=1.5)
        alpha_cell_raw = pm.Normal("alpha_cell_raw", mu=0.0, sigma=1.0, shape=n_triples)
        alpha_cell = pm.Deterministic("alpha_cell", mu_global + alpha_cell_raw * sigma_cell)
        p = pm.math.sigmoid(alpha_cell[triple_idx])
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
        "tiers": TIERS,
        "regions": REGIONS,
    }
    return idata, cell


def post_yoy_by_tier_region(idata: az.InferenceData) -> pd.DataFrame:
    posterior = idata.posterior
    alpha_cell = posterior["alpha_cell"].values.reshape(-1, 2 * len(TIERS) * len(REGIONS))
    rows = []
    for ti, tier in enumerate(TIERS):
        for ri, region in enumerate(REGIONS):
            i25 = 0 * len(TIERS) * len(REGIONS) + ti * len(REGIONS) + ri
            i26 = 1 * len(TIERS) * len(REGIONS) + ti * len(REGIONS) + ri
            p25 = expit(alpha_cell[:, i25])
            p26 = expit(alpha_cell[:, i26])
            delta = (p26 - p25) * 100.0
            rows.append({
                "count_tier": tier,
                "zone_region": region,
                "p25_mean_pct": float(p25.mean()) * 100,
                "p25_lo_pct": float(np.percentile(p25, 2.5)) * 100,
                "p25_hi_pct": float(np.percentile(p25, 97.5)) * 100,
                "p26_mean_pct": float(p26.mean()) * 100,
                "p26_lo_pct": float(np.percentile(p26, 2.5)) * 100,
                "p26_hi_pct": float(np.percentile(p26, 97.5)) * 100,
                "yoy_delta_pp_mean": float(delta.mean()),
                "yoy_delta_pp_lo": float(np.percentile(delta, 2.5)),
                "yoy_delta_pp_hi": float(np.percentile(delta, 97.5)),
                "prob_delta_gt_zero": float((delta > 0).mean()),
            })
    return pd.DataFrame(rows)


def post_interaction_test(idata: az.InferenceData) -> dict:
    """Test: does the year × top_edge effect differ between 0-0 and 2-strike?

    Diff-in-diff:
        D = (top_edge YoY at first_pitch) - (top_edge YoY at two_strike)
        where each YoY is on the CS-rate (pp) scale.
    """
    posterior = idata.posterior
    alpha = posterior["alpha_cell"].values.reshape(-1, 2 * len(TIERS) * len(REGIONS))

    def cell(year_idx: int, tier_idx: int, region_idx: int) -> np.ndarray:
        i = year_idx * len(TIERS) * len(REGIONS) + tier_idx * len(REGIONS) + region_idx
        return expit(alpha[:, i])

    tier_first = TIERS.index("first_pitch")
    tier_two = TIERS.index("two_strike")
    region_top = REGIONS.index("top_edge")
    region_heart = REGIONS.index("heart")

    top_yoy_first = (cell(1, tier_first, region_top) - cell(0, tier_first, region_top)) * 100
    top_yoy_two = (cell(1, tier_two, region_top) - cell(0, tier_two, region_top)) * 100
    heart_yoy_first = (cell(1, tier_first, region_heart) - cell(0, tier_first, region_heart)) * 100
    heart_yoy_two = (cell(1, tier_two, region_heart) - cell(0, tier_two, region_heart)) * 100

    # Diff-in-diff on top edge: first - two
    did_top = top_yoy_first - top_yoy_two
    # Diff-in-diff on heart: first - two
    did_heart = heart_yoy_first - heart_yoy_two

    return {
        "top_edge_yoy_first_pitch_pp_mean": float(top_yoy_first.mean()),
        "top_edge_yoy_first_pitch_pp_lo": float(np.percentile(top_yoy_first, 2.5)),
        "top_edge_yoy_first_pitch_pp_hi": float(np.percentile(top_yoy_first, 97.5)),
        "top_edge_yoy_two_strike_pp_mean": float(top_yoy_two.mean()),
        "top_edge_yoy_two_strike_pp_lo": float(np.percentile(top_yoy_two, 2.5)),
        "top_edge_yoy_two_strike_pp_hi": float(np.percentile(top_yoy_two, 97.5)),
        "heart_yoy_first_pitch_pp_mean": float(heart_yoy_first.mean()),
        "heart_yoy_first_pitch_pp_lo": float(np.percentile(heart_yoy_first, 2.5)),
        "heart_yoy_first_pitch_pp_hi": float(np.percentile(heart_yoy_first, 97.5)),
        "heart_yoy_two_strike_pp_mean": float(heart_yoy_two.mean()),
        "heart_yoy_two_strike_pp_lo": float(np.percentile(heart_yoy_two, 2.5)),
        "heart_yoy_two_strike_pp_hi": float(np.percentile(heart_yoy_two, 97.5)),
        "diff_in_diff_top_pp_mean": float(did_top.mean()),
        "diff_in_diff_top_pp_lo": float(np.percentile(did_top, 2.5)),
        "diff_in_diff_top_pp_hi": float(np.percentile(did_top, 97.5)),
        "diff_in_diff_heart_pp_mean": float(did_heart.mean()),
        "diff_in_diff_heart_pp_lo": float(np.percentile(did_heart, 2.5)),
        "diff_in_diff_heart_pp_hi": float(np.percentile(did_heart, 97.5)),
        "interaction_credible_top": bool(np.percentile(did_top, 2.5) * np.percentile(did_top, 97.5) > 0),
        "interaction_credible_heart": bool(np.percentile(did_heart, 2.5) * np.percentile(did_heart, 97.5) > 0),
    }


def plot_h5(post_table: pd.DataFrame, interaction: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    df = post_table.copy()

    # Left: YoY CS-rate delta per region × tier
    ax = axes[0]
    width = 0.4
    x = np.arange(len(REGIONS))
    for i, tier in enumerate(TIERS):
        sub = df.loc[df["count_tier"] == tier].set_index("zone_region").reindex(REGIONS).reset_index()
        offset = (i - 0.5) * width
        color = "#3498db" if tier == "first_pitch" else "#e67e22"
        yerr = np.array([[m - l, h - m] for m, l, h in zip(
            sub["yoy_delta_pp_mean"], sub["yoy_delta_pp_lo"], sub["yoy_delta_pp_hi"]
        )]).T
        ax.bar(x + offset, sub["yoy_delta_pp_mean"], width=width,
               color=color, edgecolor="black", linewidth=0.6,
               yerr=yerr, capsize=3, label=tier.replace("_", " "))
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS, rotation=0)
    ax.set_ylabel("YoY Δ called-strike rate (pp)")
    ax.set_title("H5: Year-on-year CS-rate change by region × count tier")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Right: diff-in-diff first - two_strike for heart and top edge
    ax = axes[1]
    targets = ["top_edge", "heart"]
    means = [interaction[f"diff_in_diff_top_pp_mean"], interaction[f"diff_in_diff_heart_pp_mean"]]
    lows = [interaction[f"diff_in_diff_top_pp_lo"], interaction[f"diff_in_diff_heart_pp_lo"]]
    highs = [interaction[f"diff_in_diff_top_pp_hi"], interaction[f"diff_in_diff_heart_pp_hi"]]
    yerr = np.array([[m - l, h - m] for m, l, h in zip(means, lows, highs)]).T
    bar_colors = ["#c0392b" if m < 0 else "#27ae60" for m in means]
    ax.bar(targets, means, color=bar_colors, edgecolor="black", yerr=yerr, capsize=6)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("ΔYoY first-pitch − ΔYoY two-strike (pp)")
    ax.set_title("Diff-in-diff: does the year × region\ninteraction differ between 0-0 and 2-strike?")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("H5: First-pitch heart/edge × count interaction (called-strike rate)", y=1.02)
    fig.tight_layout()
    fig.savefig(R2_CHARTS / "h5_first_pitch_mechanism.png", dpi=130)
    plt.close(fig)


def main() -> dict:
    ensure_dirs()
    panel = assemble_panel()
    print(f"[H5] panel rows: {len(panel):,}")
    print(panel.groupby(["year", "count_tier", "zone_region"]).size().to_string())

    idata, cell = fit_h5(panel)
    summary = az.summary(idata, var_names=["mu_global", "sigma_cell"])
    diag = {
        "rhat_max": float(summary["r_hat"].max()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "convergence_pass": bool(
            (summary["r_hat"].max() <= 1.01) and (summary["ess_bulk"].min() >= 400)
        ),
    }
    print(f"[H5] convergence: rhat={diag['rhat_max']:.3f}, ess={diag['ess_bulk_min']:.0f}")

    az.plot_trace(idata, var_names=["mu_global", "sigma_cell"])
    plt.gcf().tight_layout()
    plt.gcf().savefig(R2_DIAG / "h5_trace.png", dpi=110)
    plt.close()

    post_table = post_yoy_by_tier_region(idata)
    interaction = post_interaction_test(idata)

    print("[H5] year × region per-tier deltas:")
    print(post_table.to_string(index=False))

    print(
        f"[H5] DiD top_edge (first - two): {interaction['diff_in_diff_top_pp_mean']:+.2f}pp "
        f"[{interaction['diff_in_diff_top_pp_lo']:+.2f}, {interaction['diff_in_diff_top_pp_hi']:+.2f}], "
        f"credible: {interaction['interaction_credible_top']}"
    )
    print(
        f"[H5] DiD heart (first - two): {interaction['diff_in_diff_heart_pp_mean']:+.2f}pp "
        f"[{interaction['diff_in_diff_heart_pp_lo']:+.2f}, {interaction['diff_in_diff_heart_pp_hi']:+.2f}], "
        f"credible: {interaction['interaction_credible_heart']}"
    )

    plot_h5(post_table, interaction)
    out = {
        "h5_post_table": post_table.to_dict(orient="records"),
        "h5_interaction": interaction,
        "h5_diagnostics": diag,
    }
    (R2_ARTIFACTS / "h5_summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()

"""H2 — Per-count Bayesian binomial decomposition with credible intervals.

For each of the 12 count states, fit a Bayesian binomial:

    walk_count ~ Binomial(n_PA, p)
    logit(p) = intercept + beta_year * 2026

That gives a per-count posterior of the YoY walk-rate delta.

Then aggregate via posterior simulation:
    contribution_c = (p_26_c - p_25_c) * (PA_share_2026 of c)
    sum contributions across counts = aggregate YoY delta

This decomposition resolves whether the spike is from per-count walk-rate
increases vs PA flow into walk-prone counts.

Outputs:
  - per-count delta posterior table
  - posterior contribution-to-aggregate
  - PA-flow analysis: PA share by count 2025 vs 2026 (descriptive)
  - h2_per_count_contribution.png
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
    count_state,
    ensure_dirs,
    load_2025_samewindow,
    load_2026_full,
    plate_appearance_mask,
)


def build_count_table() -> pd.DataFrame:
    """Per (year, count) PA / walk totals over Mar 27 – May 13.

    We provide TWO views:
      reach_n_pa: PAs passing through this count
      reach_n_walk: walks among PAs that passed through this count (any terminal count)
      term_n_walk: walks that *ended at* this count (the PA's final pitch was a ball
                   at this count, going to 4 balls) — only nonzero for 3-0/3-1/3-2

    The "reach" rates match Round 1's walk_by_count.json semantics. The "term"
    walks-from-c are what go into the proper traffic + conditional decomposition.
    """
    rows = []
    for year, loader in [(2025, load_2025_samewindow), (2026, load_2026_full)]:
        df = loader().copy()
        d = pd.to_datetime(df["game_date"]).dt.normalize()
        if year == 2025:
            mask = (d >= "2025-03-27") & (d <= "2025-05-13")
        else:
            mask = (d >= "2026-03-27") & (d <= "2026-05-13")
        df = df.loc[mask].copy()
        df["pa_id"] = (
            df["game_pk"].astype("Int64").astype(str)
            + "_"
            + df["at_bat_number"].astype("Int64").astype(str)
        )
        df["count_state"] = count_state(df)
        df = df.loc[df["count_state"].isin(list(ALL_COUNTS))].copy()

        # PA-terminating row mask
        pa_end_mask = plate_appearance_mask(df)
        pa_end = df.loc[pa_end_mask, ["pa_id", "events", "count_state"]].drop_duplicates("pa_id")
        pa_end["is_walk"] = pa_end["events"].isin(WALK_EVENTS).astype(int)
        walk_by_pa = pa_end.set_index("pa_id")["is_walk"]
        # For terminal-count semantics: count_state on the terminating row = entering count for last pitch
        # A walk lands when entering count is 3-x AND last pitch is a ball.
        term_walks_by_count = (
            pa_end.loc[pa_end["is_walk"] == 1]
            .groupby("count_state").size()
            .reindex(list(ALL_COUNTS), fill_value=0)
        )

        # PAs reaching each count (unique pa-count pairs)
        reach = (
            df[["pa_id", "count_state"]].drop_duplicates()
        )
        reach["is_walk"] = reach["pa_id"].map(walk_by_pa).fillna(0).astype(int)
        agg = reach.groupby("count_state").agg(reach_n_pa=("pa_id", "count"), reach_n_walk=("is_walk", "sum"))
        for c in ALL_COUNTS:
            rn = int(agg.loc[c, "reach_n_pa"]) if c in agg.index else 0
            rk = int(agg.loc[c, "reach_n_walk"]) if c in agg.index else 0
            tk = int(term_walks_by_count.loc[c]) if c in term_walks_by_count.index else 0
            rows.append({
                "year": year, "count": c,
                "n_pa": rn,                # reach semantics
                "n_walk": rk,              # reach semantics walks (any terminal)
                "term_n_walk": tk,         # walks landing AT this count
            })

    out = pd.DataFrame(rows)
    out["walk_rate"] = out["n_walk"] / out["n_pa"]
    return out


def fit_per_count(
    table: pd.DataFrame, *, count: str, walks_col: str = "n_walk", pa_col: str = "n_pa"
) -> dict:
    sub = table.loc[table["count"] == count]
    n25 = int(sub.loc[sub["year"] == 2025, pa_col].iloc[0])
    n26 = int(sub.loc[sub["year"] == 2026, pa_col].iloc[0])
    k25 = int(sub.loc[sub["year"] == 2025, walks_col].iloc[0])
    k26 = int(sub.loc[sub["year"] == 2026, walks_col].iloc[0])

    rate25 = k25 / max(n25, 1)
    rate26 = k26 / max(n26, 1)

    with pm.Model() as model:
        beta0 = pm.Normal("beta0", mu=np.log(max(rate25, 1e-3) / max(1 - rate25, 1e-3)), sigma=1.5)
        beta_year = pm.Normal("beta_year", mu=0.0, sigma=1.5)
        p25 = pm.Deterministic("p25", pm.math.sigmoid(beta0))
        p26 = pm.Deterministic("p26", pm.math.sigmoid(beta0 + beta_year))
        pm.Binomial("y25", n=n25, p=p25, observed=k25)
        pm.Binomial("y26", n=n26, p=p26, observed=k26)
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
            target_accept=0.9,
            random_seed=2026,
            progressbar=False,
        )
    p25_draws = idata.posterior["p25"].values.reshape(-1)
    p26_draws = idata.posterior["p26"].values.reshape(-1)
    delta_draws = (p26_draws - p25_draws) * 100.0  # pp
    summary = az.summary(idata, var_names=["beta_year"])
    rhat = float(summary["r_hat"].iloc[0])
    ess = float(summary["ess_bulk"].iloc[0])
    return {
        "count": count,
        "n_pa_2025": n25,
        "n_pa_2026": n26,
        "rate_2025_emp": rate25,
        "rate_2026_emp": rate26,
        "p25_draws": p25_draws,
        "p26_draws": p26_draws,
        "delta_pp_draws": delta_draws,
        "delta_pp_mean": float(delta_draws.mean()),
        "delta_pp_lo": float(np.percentile(delta_draws, 2.5)),
        "delta_pp_hi": float(np.percentile(delta_draws, 97.5)),
        "prob_delta_gt_zero": float((delta_draws > 0).mean()),
        "rhat": rhat,
        "ess_bulk": ess,
    }


def aggregate_contributions(
    per_count_fits: dict,
    term_fits: dict,
    table: pd.DataFrame,
    *,
    total_pa_25: int,
    total_pa_26: int,
) -> dict:
    """Decomposition of the YoY walk-rate spike.

    Two views:
      (a) Descriptive: per-count YoY in conditional walk-rate (P(walk | reach c)),
          which matches Round 1's `walk_by_count.json`.
      (b) Proper decomposition over terminal-walking counts (3-0, 3-1, 3-2):
          walk_rate = sum_c P(reach c) * P_terminal(walk lands at c | reach c)
          Δwalk_rate = sum_c [ΔP(reach c) * P25_term(c)] + sum_c [P26(reach c) * ΔP_term(c)]
          Empirically should sum to aggregate YoY walk-rate delta.
    """
    pa_by_count = {
        2025: table.loc[table["year"] == 2025].set_index("count")["n_pa"],
        2026: table.loc[table["year"] == 2026].set_index("count")["n_pa"],
    }
    total = {2025: total_pa_25, 2026: total_pa_26}
    reach_rate = {
        y: pa_by_count[y] / total[y] for y in (2025, 2026)
    }

    rows = []
    for c in ALL_COUNTS:
        fit = per_count_fits[c]
        s25 = float(reach_rate[2025].loc[c]) if c in reach_rate[2025].index else 0.0
        s26 = float(reach_rate[2026].loc[c]) if c in reach_rate[2026].index else 0.0
        rows.append(
            {
                "count": c,
                "delta_pp_mean": fit["delta_pp_mean"],
                "delta_pp_lo": fit["delta_pp_lo"],
                "delta_pp_hi": fit["delta_pp_hi"],
                "reach_2025_pct": s25 * 100,
                "reach_2026_pct": s26 * 100,
                "reach_delta_pp": (s26 - s25) * 100,
                "rate_2025_emp_pct": fit["rate_2025_emp"] * 100,
                "rate_2026_emp_pct": fit["rate_2026_emp"] * 100,
                "n_pa_2025": fit["n_pa_2025"],
                "n_pa_2026": fit["n_pa_2026"],
                "rhat": fit["rhat"],
                "ess_bulk": fit["ess_bulk"],
            }
        )

    # Proper decomposition over terminal counts using terminal-walk semantics
    terminal_counts = ["3-0", "3-1", "3-2"]
    traffic_draws_total = None
    cond_draws_total = None
    for c in terminal_counts:
        tfit = term_fits[c]  # uses terminal-walk semantics
        s25 = float(reach_rate[2025].loc[c]) if c in reach_rate[2025].index else 0.0
        s26 = float(reach_rate[2026].loc[c]) if c in reach_rate[2026].index else 0.0
        p25_draws = tfit["p25_draws"]
        p26_draws = tfit["p26_draws"]
        traffic = (s26 - s25) * p25_draws * 100.0  # pp
        conditional = s26 * (p26_draws - p25_draws) * 100.0
        if traffic_draws_total is None:
            traffic_draws_total = traffic.copy()
            cond_draws_total = conditional.copy()
        else:
            traffic_draws_total = traffic_draws_total + traffic
            cond_draws_total = cond_draws_total + conditional

    total_draws = traffic_draws_total + cond_draws_total
    decomp = {
        "traffic_pp_mean": float(traffic_draws_total.mean()),
        "traffic_pp_lo": float(np.percentile(traffic_draws_total, 2.5)),
        "traffic_pp_hi": float(np.percentile(traffic_draws_total, 97.5)),
        "conditional_pp_mean": float(cond_draws_total.mean()),
        "conditional_pp_lo": float(np.percentile(cond_draws_total, 2.5)),
        "conditional_pp_hi": float(np.percentile(cond_draws_total, 97.5)),
        "total_pp_mean": float(total_draws.mean()),
        "total_pp_lo": float(np.percentile(total_draws, 2.5)),
        "total_pp_hi": float(np.percentile(total_draws, 97.5)),
        "per_terminal_count": {
            c: {
                "reach_2025_pct": float(reach_rate[2025].loc[c]) * 100,
                "reach_2026_pct": float(reach_rate[2026].loc[c]) * 100,
                "term_rate_2025_pct": term_fits[c]["rate_2025_emp"] * 100,
                "term_rate_2026_pct": term_fits[c]["rate_2026_emp"] * 100,
                "term_delta_pp_mean": term_fits[c]["delta_pp_mean"],
                "term_delta_pp_lo": term_fits[c]["delta_pp_lo"],
                "term_delta_pp_hi": term_fits[c]["delta_pp_hi"],
            }
            for c in terminal_counts
        },
    }
    table_out = pd.DataFrame(rows)
    return {"table": table_out, "decomp": decomp}


def plot_per_count_contribution(table_out: pd.DataFrame, decomp: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    df = table_out.copy()
    df["count"] = pd.Categorical(df["count"], categories=list(ALL_COUNTS), ordered=True)
    df = df.sort_values("count")

    ax = axes[0]
    colors = ["#c0392b" if d > 0 else "#2980b9" for d in df["delta_pp_mean"]]
    ax.barh(
        range(len(df)),
        df["delta_pp_mean"],
        color=colors,
        xerr=[
            df["delta_pp_mean"] - df["delta_pp_lo"],
            df["delta_pp_hi"] - df["delta_pp_mean"],
        ],
        capsize=2,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["count"])
    ax.set_xlabel("YoY ΔP(walk | reach count) (pp, with 95% CrI)")
    ax.set_title("Per-count conditional walk-rate change (reach semantics)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    decomp_means = [decomp["traffic_pp_mean"], decomp["conditional_pp_mean"], decomp["total_pp_mean"]]
    decomp_lo = [decomp["traffic_pp_lo"], decomp["conditional_pp_lo"], decomp["total_pp_lo"]]
    decomp_hi = [decomp["traffic_pp_hi"], decomp["conditional_pp_hi"], decomp["total_pp_hi"]]
    labels = ["traffic\n(more PAs at 3-0/3-1/3-2)", "conditional\n(higher walk %\nat 3-0/3-1/3-2)", "total\n(traffic + conditional)"]
    cols = ["#3498db", "#e67e22", "#2c3e50"]
    yerr = np.array([[m - l, h - m] for m, l, h in zip(decomp_means, decomp_lo, decomp_hi)]).T
    ax.bar(range(3), decomp_means, color=cols, edgecolor="black", yerr=yerr, capsize=5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Contribution to YoY walk-rate spike (pp)")
    ax.set_title(
        f"Decomposition over terminal counts (3-0/3-1/3-2)\n"
        f"traffic = {decomp['traffic_pp_mean']:+.2f}pp [{decomp['traffic_pp_lo']:+.2f}, {decomp['traffic_pp_hi']:+.2f}]\n"
        f"conditional = {decomp['conditional_pp_mean']:+.2f}pp [{decomp['conditional_pp_lo']:+.2f}, {decomp['conditional_pp_hi']:+.2f}]"
    )
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("H2: Per-count decomposition of the YoY walk spike (Mar 27 – May 13)", y=1.01)
    fig.tight_layout()
    fig.savefig(R2_CHARTS / "h2_per_count_contribution.png", dpi=130)
    plt.close(fig)


def main() -> dict:
    ensure_dirs()
    table = build_count_table()
    print("[H2] count table built")
    print(table.to_string(index=False))
    # Total PAs by year (= n_pa at 0-0, since every PA reaches 0-0)
    total_pa_25 = int(table.loc[(table["year"] == 2025) & (table["count"] == "0-0"), "n_pa"].iloc[0])
    total_pa_26 = int(table.loc[(table["year"] == 2026) & (table["count"] == "0-0"), "n_pa"].iloc[0])
    print(f"[H2] total PAs 2025={total_pa_25:,}, 2026={total_pa_26:,}")
    per_count_fits = {}
    diag_rows = []
    for c in ALL_COUNTS:
        print(f"[H2] fitting reach-semantics count {c} ...", flush=True)
        fit = fit_per_count(table, count=c, walks_col="n_walk", pa_col="n_pa")
        per_count_fits[c] = fit
        diag_rows.append({"count": c, "view": "reach", "rhat": fit["rhat"], "ess_bulk": fit["ess_bulk"]})
        print(
            f"[H2]   reach delta {fit['delta_pp_mean']:+.2f}pp "
            f"[{fit['delta_pp_lo']:+.2f}, {fit['delta_pp_hi']:+.2f}]  "
            f"rhat={fit['rhat']:.3f}  ess={fit['ess_bulk']:.0f}"
        )

    term_fits = {}
    for c in ["3-0", "3-1", "3-2"]:
        print(f"[H2] fitting terminal-walk count {c} ...", flush=True)
        fit = fit_per_count(table, count=c, walks_col="term_n_walk", pa_col="n_pa")
        term_fits[c] = fit
        diag_rows.append({"count": c, "view": "terminal", "rhat": fit["rhat"], "ess_bulk": fit["ess_bulk"]})
        print(
            f"[H2]   terminal delta {fit['delta_pp_mean']:+.2f}pp "
            f"[{fit['delta_pp_lo']:+.2f}, {fit['delta_pp_hi']:+.2f}]  "
            f"rhat={fit['rhat']:.3f}  ess={fit['ess_bulk']:.0f}"
        )

    agg = aggregate_contributions(per_count_fits, term_fits, table, total_pa_25=total_pa_25, total_pa_26=total_pa_26)
    plot_per_count_contribution(agg["table"], agg["decomp"])

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_parquet(R2_DIAG.parent / "h2_diagnostics.parquet", index=False)
    print(f"[H2] worst rhat: {diag_df['rhat'].max():.3f}, worst ess: {diag_df['ess_bulk'].min():.0f}")

    convergence_pass = (diag_df["rhat"].max() <= 1.01) and (diag_df["ess_bulk"].min() >= 400)

    table_out = agg["table"].copy()
    decomp = agg["decomp"]

    out = {
        "h2_per_count": table_out.to_dict(orient="records"),
        "h2_decomp": decomp,
        "h2_diagnostics": {
            "rhat_max": float(diag_df["rhat"].max()),
            "ess_bulk_min": float(diag_df["ess_bulk"].min()),
            "convergence_pass": bool(convergence_pass),
        },
    }
    (R2_ARTIFACTS / "h2_summary.json").write_text(json.dumps(out, indent=2))
    table_out.to_parquet(R2_ARTIFACTS / "h2_table.parquet", index=False)
    print(
        f"[H2] decomp: traffic={decomp['traffic_pp_mean']:+.2f}pp [{decomp['traffic_pp_lo']:+.2f}, {decomp['traffic_pp_hi']:+.2f}], "
        f"conditional={decomp['conditional_pp_mean']:+.2f}pp [{decomp['conditional_pp_lo']:+.2f}, {decomp['conditional_pp_hi']:+.2f}], "
        f"total={decomp['total_pp_mean']:+.2f}pp [{decomp['total_pp_lo']:+.2f}, {decomp['total_pp_hi']:+.2f}]"
    )
    return out


if __name__ == "__main__":
    main()

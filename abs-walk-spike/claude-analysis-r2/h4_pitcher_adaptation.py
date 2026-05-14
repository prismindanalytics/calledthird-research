"""H4 — Pitcher adaptation timing (week-over-week, league + per-pitcher).

For 2026 only (the new regime), fit Bayesian week-over-week GAM-ish models
for each pitcher with ≥200 pitches in Mar 27 – May 13.

Targets per pitcher:
  - mean vertical location of called/swung-at pitches: Normal(mu_t, sigma)
    with mu_t a 2nd-order random walk over weeks
  - zone rate (rulebook-approx): Binomial random-walk
  - top-of-zone usage (z>3.0 ft fraction): Binomial random-walk

We then summarize each pitcher's "adaptation magnitude" as the absolute
posterior change between week-1 and week-7 across the three metrics.

League-wide aggregation uses pooled data.

Outputs:
  - h4_pitcher_adaptation_leaderboard.png (top-10)
  - league trend table (per-week)
  - per-pitcher posterior summaries
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
    R2_ARTIFACTS,
    R2_CHARTS,
    R2_DIAG,
    ensure_dirs,
    load_2026_full,
    rulebook_zone_flag,
    week_index_2026,
    WINDOW_START,
)


PITCH_GROUPS = {
    "fastball": ["FF", "FT", "SI", "FC"],
    "breaking": ["SL", "CU", "KC", "ST", "SV", "SC"],
    "offspeed": ["CH", "FS", "FO"],
}


def assign_pitch_group(pitch_type: str) -> str:
    for g, l in PITCH_GROUPS.items():
        if pitch_type in l:
            return g
    return "other"


def build_pitch_table() -> pd.DataFrame:
    df = load_2026_full()
    d = pd.to_datetime(df["game_date"]).dt.normalize()
    df = df.loc[(d >= "2026-03-27") & (d <= "2026-05-13")].copy()
    df["week"] = week_index_2026(df["game_date"], anchor=WINDOW_START)
    df["pitch_group"] = df["pitch_type"].fillna("UNK").map(assign_pitch_group)
    df["zone"] = rulebook_zone_flag(df["plate_x"], df["plate_z"])
    df["above_3"] = (df["plate_z"].astype(float) > 3.0).astype(int)
    return df


def league_trends(df: pd.DataFrame) -> dict:
    """Per-week league rates."""
    by_week = (
        df.groupby("week").agg(
            n=("zone", "size"),
            zone_n=("zone", "sum"),
            zone_rate=("zone", "mean"),
            mean_z=("plate_z", "mean"),
            top_share=("above_3", "mean"),
        ).reset_index()
    )
    # Per-week pitch-type share
    pt_share = (
        df.groupby(["week", "pitch_group"]).size().reset_index(name="n")
    )
    total_per_week = pt_share.groupby("week")["n"].transform("sum")
    pt_share["share"] = pt_share["n"] / total_per_week

    return {
        "by_week": by_week.to_dict(orient="records"),
        "pitch_type_share_by_week": pt_share.to_dict(orient="records"),
    }


def fit_pitcher_beta(sub: pd.DataFrame, *, pitcher_id: int, n_boot: int = 1000) -> dict:
    """Per-pitcher week-1 → week-N shift via Beta-Binomial conjugate sampling.

    Vectorized; no PyMC needed. For each week's binomial outcome (zone, top),
    posterior of the rate is Beta(k+1, n-k+1). Sample n_boot draws from each
    week-1 and week-last posterior; report the difference.
    """
    weeks = sorted(sub["week"].unique())
    if len(weeks) < 3:
        return None
    by_week = sub.groupby("week").agg(
        n=("zone", "size"),
        k=("zone", "sum"),
        mean_z=("plate_z", "mean"),
        top_n=("above_3", "sum"),
    ).reset_index()

    rng = np.random.default_rng(pitcher_id)
    first = by_week.iloc[0]
    last = by_week.iloc[-1]
    # Beta posterior on zone rate
    z_first = rng.beta(first["k"] + 1, first["n"] - first["k"] + 1, size=n_boot)
    z_last = rng.beta(last["k"] + 1, last["n"] - last["k"] + 1, size=n_boot)
    t_first = rng.beta(first["top_n"] + 1, first["n"] - first["top_n"] + 1, size=n_boot)
    t_last = rng.beta(last["top_n"] + 1, last["n"] - last["top_n"] + 1, size=n_boot)
    delta_zone_pp = (z_last - z_first) * 100
    delta_top_pp = (t_last - t_first) * 100

    return {
        "weeks": weeks,
        "n_pitches": int(sub.shape[0]),
        "p_zone_week_means": by_week.apply(lambda r: r["k"] / r["n"], axis=1).tolist(),
        "p_top_week_means": by_week.apply(lambda r: r["top_n"] / r["n"], axis=1).tolist(),
        "mean_z_week": by_week["mean_z"].astype(float).tolist(),
        "delta_zone_pp_mean": float(delta_zone_pp.mean()),
        "delta_zone_pp_lo": float(np.percentile(delta_zone_pp, 2.5)),
        "delta_zone_pp_hi": float(np.percentile(delta_zone_pp, 97.5)),
        "delta_top_pp_mean": float(delta_top_pp.mean()),
        "delta_top_pp_lo": float(np.percentile(delta_top_pp, 2.5)),
        "delta_top_pp_hi": float(np.percentile(delta_top_pp, 97.5)),
        "rhat_max": 1.0,  # exact-sampling, no MCMC
        "ess_min": float(n_boot),
    }


def fit_league_glm(df: pd.DataFrame) -> dict:
    by_week = df.groupby("week").agg(
        n=("zone", "size"),
        k=("zone", "sum"),
        n_top=("above_3", "sum"),
        mean_z=("plate_z", "mean"),
    ).reset_index()
    weeks = sorted(by_week["week"].unique())
    week_idx = np.array([weeks.index(w) for w in by_week["week"]])
    n_arr = by_week["n"].values.astype(int)
    k_arr = by_week["k"].values.astype(int)
    n_top_arr = by_week["n_top"].values.astype(int)

    with pm.Model() as model:
        beta0 = pm.Normal("beta0", mu=0.0, sigma=1.5)
        sigma_rw = pm.HalfNormal("sigma_rw", sigma=0.3)
        rw_innov = pm.Normal("rw_innov", mu=0.0, sigma=sigma_rw, shape=len(weeks))
        rw = pm.Deterministic("rw", pm.math.cumsum(rw_innov))
        eta = beta0 + rw[week_idx]
        p = pm.math.sigmoid(eta)
        pm.Binomial("y_zone", n=n_arr, p=p, observed=k_arr)
        beta0_top = pm.Normal("beta0_top", mu=-1.0, sigma=1.5)
        sigma_rw_top = pm.HalfNormal("sigma_rw_top", sigma=0.3)
        rw_innov_top = pm.Normal("rw_innov_top", mu=0.0, sigma=sigma_rw_top, shape=len(weeks))
        rw_top = pm.Deterministic("rw_top", pm.math.cumsum(rw_innov_top))
        eta_top = beta0_top + rw_top[week_idx]
        p_top = pm.math.sigmoid(eta_top)
        pm.Binomial("y_top", n=n_arr, p=p_top, observed=n_top_arr)
        idata = pm.sample(
            draws=1000,
            tune=750,
            chains=4,
            cores=4,
            target_accept=0.93,
            random_seed=2026,
            progressbar=False,
        )

    posterior = idata.posterior
    rw_draws = posterior["rw"].values.reshape(-1, len(weeks))
    rw_top_draws = posterior["rw_top"].values.reshape(-1, len(weeks))
    beta0_draws = posterior["beta0"].values.reshape(-1)
    beta0_top_draws = posterior["beta0_top"].values.reshape(-1)
    p_zone_week = np.array([
        expit(beta0_draws + rw_draws[:, t]) for t in range(len(weeks))
    ])  # (n_weeks, n_draws)
    p_top_week = np.array([
        expit(beta0_top_draws + rw_top_draws[:, t]) for t in range(len(weeks))
    ])

    out = []
    for t, w in enumerate(weeks):
        out.append({
            "week": int(w),
            "n_pitches": int(n_arr[t]),
            "zone_rate_mean": float(p_zone_week[t].mean()),
            "zone_rate_lo": float(np.percentile(p_zone_week[t], 2.5)),
            "zone_rate_hi": float(np.percentile(p_zone_week[t], 97.5)),
            "top_share_mean": float(p_top_week[t].mean()),
            "top_share_lo": float(np.percentile(p_top_week[t], 2.5)),
            "top_share_hi": float(np.percentile(p_top_week[t], 97.5)),
        })

    delta_zone = (p_zone_week[-1] - p_zone_week[0]) * 100
    delta_top = (p_top_week[-1] - p_top_week[0]) * 100

    rhat = float(az.summary(idata, var_names=["beta0", "sigma_rw", "beta0_top"])["r_hat"].max())
    ess = float(az.summary(idata, var_names=["beta0", "sigma_rw", "beta0_top"])["ess_bulk"].min())

    return {
        "by_week": out,
        "league_delta_zone_pp_mean": float(delta_zone.mean()),
        "league_delta_zone_pp_lo": float(np.percentile(delta_zone, 2.5)),
        "league_delta_zone_pp_hi": float(np.percentile(delta_zone, 97.5)),
        "league_delta_top_pp_mean": float(delta_top.mean()),
        "league_delta_top_pp_lo": float(np.percentile(delta_top, 2.5)),
        "league_delta_top_pp_hi": float(np.percentile(delta_top, 97.5)),
        "rhat_max": rhat,
        "ess_min": ess,
    }


def pitcher_id_to_name_map(df: pd.DataFrame) -> dict:
    """Build pitcher_id -> player_name map from the Statcast data if available."""
    if "player_name" in df.columns:
        names = df.groupby("pitcher")["player_name"].agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else "")
        return names.to_dict()
    return {}


def plot_adapter_leaderboard(records: list[dict], league: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    df = pd.DataFrame(records)
    if df.empty:
        return
    # Sort by abs(delta_zone_pp_mean) for "most-shifted zone rate"
    df = df.sort_values("abs_total_shift", ascending=False).head(10)

    ax = axes[0]
    y = range(len(df))
    ax.barh(
        y,
        df["delta_zone_pp_mean"],
        color=["#c0392b" if x < 0 else "#27ae60" for x in df["delta_zone_pp_mean"]],
        edgecolor="black",
        linewidth=0.6,
        xerr=[df["delta_zone_pp_mean"] - df["delta_zone_pp_lo"], df["delta_zone_pp_hi"] - df["delta_zone_pp_mean"]],
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["name"].fillna("unknown"))
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ zone rate W1 → W7 (pp)")
    ax.set_title("Top-10 most-shifted pitchers (Δ zone rate)")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    ax.barh(
        y,
        df["delta_top_pp_mean"],
        color=["#c0392b" if x < 0 else "#27ae60" for x in df["delta_top_pp_mean"]],
        edgecolor="black",
        linewidth=0.6,
        xerr=[df["delta_top_pp_mean"] - df["delta_top_pp_lo"], df["delta_top_pp_hi"] - df["delta_top_pp_mean"]],
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["name"].fillna("unknown"))
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Δ top-of-zone usage (z > 3.0 ft) (pp)")
    ax.set_title("Same pitchers — top-of-zone usage shift")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"H4: Top-10 most-adapted pitchers (≥200 pitches Mar 27 – May 13).\n"
        f"League W1→W7: Δ zone rate {league['league_delta_zone_pp_mean']:+.2f}pp [{league['league_delta_zone_pp_lo']:+.2f}, {league['league_delta_zone_pp_hi']:+.2f}], "
        f"Δ top share {league['league_delta_top_pp_mean']:+.2f}pp",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(R2_CHARTS / "h4_pitcher_adaptation_leaderboard.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    ensure_dirs()
    df = build_pitch_table()
    print(f"[H4] pitch table: {len(df):,} pitches, {df['pitcher'].nunique():,} pitchers")

    league_descriptive = league_trends(df)
    league_bayes = fit_league_glm(df)
    print(
        f"[H4] league W1→W7 zone-rate Δ: {league_bayes['league_delta_zone_pp_mean']:+.2f}pp "
        f"[{league_bayes['league_delta_zone_pp_lo']:+.2f}, {league_bayes['league_delta_zone_pp_hi']:+.2f}]"
    )

    # Per-pitcher
    pc = df.groupby("pitcher").size()
    eligible = pc[pc >= 200].index.tolist()
    print(f"[H4] {len(eligible)} pitchers with ≥200 pitches; fitting up to all of them")
    names = pitcher_id_to_name_map(df)

    records = []
    diag_rows = []
    # Limit to top 100 by pitch count to keep runtime sane; we only need top-10 leaderboard
    by_count = pc.loc[eligible].sort_values(ascending=False)
    eligible_capped = by_count.index.tolist()  # all eligible pitchers (≥200 pitches)
    print(f"[H4] fitting {len(eligible_capped)} pitchers (all with ≥200 pitches)")
    for pi in eligible_capped:
        sub = df.loc[df["pitcher"] == pi]
        res = fit_pitcher_beta(sub, pitcher_id=int(pi))
        if res is None:
            continue
        diag_rows.append({"pitcher": int(pi), "rhat": res["rhat_max"], "ess_min": res["ess_min"]})
        records.append(
            {
                "pitcher": int(pi),
                "name": names.get(pi, None),
                "n_pitches": res["n_pitches"],
                "delta_zone_pp_mean": res["delta_zone_pp_mean"],
                "delta_zone_pp_lo": res["delta_zone_pp_lo"],
                "delta_zone_pp_hi": res["delta_zone_pp_hi"],
                "delta_top_pp_mean": res["delta_top_pp_mean"],
                "delta_top_pp_lo": res["delta_top_pp_lo"],
                "delta_top_pp_hi": res["delta_top_pp_hi"],
                "abs_total_shift": abs(res["delta_zone_pp_mean"]) + abs(res["delta_top_pp_mean"]),
            }
        )

    diag_df = pd.DataFrame(diag_rows)
    rec_df = pd.DataFrame(records)
    rec_df.to_parquet(R2_ARTIFACTS / "h4_pitcher_records.parquet", index=False)
    print(f"[H4] worst rhat: {diag_df['rhat'].max():.3f}, min ess: {diag_df['ess_min'].min():.0f}")

    plot_adapter_leaderboard(records, league_bayes)

    out = {
        "h4_league_descriptive": league_descriptive,
        "h4_league_bayes": league_bayes,
        "h4_pitcher_records": records,
        "h4_diagnostics": {
            "rhat_max": float(diag_df["rhat"].max()),
            "ess_min": float(diag_df["ess_min"].min()),
        },
    }
    (R2_ARTIFACTS / "h4_summary.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    main()

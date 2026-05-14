"""h5_per_hitter.py — Per-hitter posterior-predictive residuals.

For each batter with >=30 borderline take decisions in spots 7-9, compute:
  expected called-strike rate from the Round 1 H3 GAM (no batter features)
  actual called-strike rate
  residual = actual - expected
  posterior-predictive simulation: draw expected probabilities from the H3
    GAM posterior, simulate Bernoulli outcomes, compute the simulated rate,
    repeat. The 95% CI on (actual - simulated) is the residual CI.

Identify hitters whose 95% CI on the residual excludes zero AND magnitude >= 3pp.
BH-FDR correction across qualifying hitters.

Cross-reference with 2025 chase rate.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTENSOR_FLAGS", "blas__ldflags=,floatX=float64")

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_prep_r2

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
CHARTS = ROOT / "charts"
DIAG = CHARTS / "diagnostics"
CACHE.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)
DIAG.mkdir(parents=True, exist_ok=True)


def _bh_fdr(pvals: np.ndarray, q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    qvals_sorted = ranked * n / (np.arange(1, n + 1))
    qvals_sorted = np.minimum.accumulate(qvals_sorted[::-1])[::-1]
    qvals = np.empty_like(pvals)
    qvals[order] = qvals_sorted
    return qvals, qvals < q


def fit_or_load_h3():
    """Get cached Round 1 H3 GAM idata, info, df. Re-fits if cache is missing."""
    idata, info, df = data_prep_r2.fit_round1_h3_gam()
    return idata, info, df


def per_hitter_residuals(
    idata,
    info,
    df: pd.DataFrame,
    spots_filter=(7, 8, 9),
    min_n: int = 30,
    n_keep: int = 1000,
    drop_pinch: bool = False,
) -> pd.DataFrame:
    """Compute per-hitter posterior-predictive residuals on borderline pitches.

    Workflow:
      1. Filter df to rows where lineup_spot in spots_filter (and optionally non-pinch).
      2. Per qualifying batter (>=min_n rows), compute:
         - Posterior-predictive expected probabilities: pp_probs (n_keep, n_obs_b)
         - Actual called-strike rate: actual_rate
         - For each posterior draw, simulate y_sim ~ Bern(pp_probs) and average:
             rate_sim_draw = mean(y_sim_draw)
         - residual_draw = actual_rate - rate_sim_draw
         - posterior median + 95% CI on residual
      3. BH-FDR p-value from posterior-tail probability:
         p_two_sided = 2 * min(P(residual<0), P(residual>0))
    """
    df = df.copy()
    if drop_pinch and "is_pinch_hitter" in df.columns:
        df = df[~df["is_pinch_hitter"].astype(bool)]
    df = df[df["lineup_spot"].isin(spots_filter)].copy()
    df = df.reset_index(drop=True)

    posterior = idata.posterior
    rng = np.random.default_rng(13)
    flat = posterior.sizes["chain"] * posterior.sizes["draw"]
    n_keep = min(n_keep, flat)
    sample_ids = rng.choice(flat, n_keep, replace=False)

    # Build the linear-predictor for ALL borderline pitches (not just spots 7-9)
    # because the H3 model was fit on all borderline pitches. We recompute on
    # only spots-7-9 rows here to align with the H5 question.
    # But info["spot_idx"], info["B"], etc. are aligned with the FULL borderline df
    # used to fit the H3 GAM. We need to extract the subset in df and align with that.
    df_full = info_df_used_to_fit = data_prep_r2.load_borderline()
    # Find the row indices of df rows in df_full
    # Use a unique key (game_pk, at_bat_number, pitch_number)
    df_full = df_full.reset_index().rename(columns={"index": "_row"})
    key_cols = ["game_pk", "at_bat_number", "pitch_number"]
    df_keyed = df.merge(df_full[key_cols + ["_row"]], on=key_cols, how="left")
    if df_keyed["_row"].isna().any():
        # this shouldn't happen because df is the same df_full but filtered
        n_missing = int(df_keyed["_row"].isna().sum())
        print(f"  warning: {n_missing} rows could not be matched back to H3 fit (likely duplicates).")
        df_keyed = df_keyed.dropna(subset=["_row"])
    rows = df_keyed["_row"].astype(int).values

    # Compute eta for these rows under all draws
    spot_levels = info["spot_levels"]
    intercept = posterior["intercept"].values.reshape(-1)[sample_ids]
    b_spot = posterior["b_spot"].values.reshape(-1, len(spot_levels))[sample_ids]
    b_count = posterior["b_count"].values.reshape(-1, len(info["count_levels"]))[sample_ids]
    b_framing = posterior["b_framing"].values.reshape(-1, len(info["framing_levels"]))[sample_ids]
    b_basis = posterior["b_basis"].values.reshape(-1, info["n_basis"])[sample_ids]
    u_pitcher_full = posterior["u_pitcher"].values.reshape(-1, info["n_pitcher"])[sample_ids]
    u_catcher_full = posterior["u_catcher"].values.reshape(-1, info["n_catcher"])[sample_ids]
    u_umpire_full = posterior["u_umpire"].values.reshape(-1, info["n_umpire"])[sample_ids]

    B_all = info["B"][rows, :]
    spot_idx_all = info["spot_idx"][rows]
    count_idx_all = info["count_idx"][rows]
    framing_idx_all = info["framing_idx"][rows]
    pitcher_idx_all = info["pitcher_idx"][rows]
    catcher_idx_all = info["catcher_idx"][rows]
    umpire_idx_all = info["umpire_idx"][rows]

    spot_eff_all = np.where(spot_idx_all[None, :] == 0, 0.0, b_spot[:, np.maximum(spot_idx_all - 1, 0)])
    count_eff_all = np.where(count_idx_all[None, :] == 0, 0.0, b_count[:, np.maximum(count_idx_all - 1, 0)])
    framing_eff_all = np.where(framing_idx_all[None, :] == 0, 0.0, b_framing[:, np.maximum(framing_idx_all - 1, 0)])

    eta_all = (
        intercept[:, None]
        + spot_eff_all
        + count_eff_all
        + framing_eff_all
        + b_basis @ B_all.T
        + u_pitcher_full[:, pitcher_idx_all]
        + u_catcher_full[:, catcher_idx_all]
        + u_umpire_full[:, umpire_idx_all]
    )
    pp_probs = 1.0 / (1.0 + np.exp(-eta_all))    # (S, n_obs)

    # For each batter, compute residual
    df_keyed["actual"] = df_keyed["called_strike"].astype(int).values
    df_keyed = df_keyed.reset_index(drop=True)
    out_rows = []
    rng2 = np.random.default_rng(17)
    for batter, idx in df_keyed.groupby("batter").groups.items():
        idx = np.array(idx)
        n = len(idx)
        if n < min_n:
            continue
        actual_rate = df_keyed.loc[idx, "actual"].mean()
        probs_b = pp_probs[:, idx]                           # (S, n)
        # posterior-mean expected
        expected_mean = probs_b.mean(axis=0).mean()
        # simulate Bernoulli outcomes for each draw and compute rate
        sim_y = (rng2.random(probs_b.shape) < probs_b).astype(int)   # (S, n)
        sim_rate = sim_y.mean(axis=1)                                # (S,)
        residual_draws = (actual_rate - sim_rate) * 100.0            # (S,) pp
        med = float(np.median(residual_draws))
        lo = float(np.percentile(residual_draws, 2.5))
        hi = float(np.percentile(residual_draws, 97.5))
        p_dir_pos = float((residual_draws > 0).mean())
        p_two_sided = 2 * min(p_dir_pos, 1 - p_dir_pos)
        out_rows.append({
            "batter": int(batter),
            "n_borderline_spot79": int(n),
            "actual_pct": float(actual_rate * 100.0),
            "expected_pct": float(expected_mean * 100.0),
            "residual_pp_med": med,
            "residual_pp_lo": lo,
            "residual_pp_hi": hi,
            "p_dir_pos": p_dir_pos,
            "p_two_sided": p_two_sided,
        })
    return pd.DataFrame(out_rows)


def run(drop_pinch: bool = False, magnitude_threshold_pp: float = 3.0) -> dict:
    print("[H5] loading or fitting Round 1 H3 GAM ...")
    idata, info, df_h3 = fit_or_load_h3()
    df_borderline = data_prep_r2.load_borderline()
    print(f"[H5] computing per-hitter residuals (spot 7-9, n_borderline={len(df_borderline):,}) ...")
    per_h = per_hitter_residuals(idata, info, df_borderline, spots_filter=(7, 8, 9), min_n=30, drop_pinch=drop_pinch)

    print(f"[H5] qualifying batters (>=30 borderline take decisions in spots 7-9): {len(per_h)}")
    if len(per_h) == 0:
        return {"n_qualifying_hitters": 0, "n_flagged": 0, "flagged_list": []}

    qvals, _ = _bh_fdr(per_h["p_two_sided"].values, q=0.10)
    per_h["q_bh"] = qvals
    per_h["bh_significant_q10"] = per_h["q_bh"] < 0.10
    per_h["ci_excludes_zero"] = (per_h["residual_pp_lo"] > 0) | (per_h["residual_pp_hi"] < 0)
    per_h["abs_residual_pp"] = per_h["residual_pp_med"].abs()
    per_h["flagged"] = per_h["ci_excludes_zero"] & (per_h["abs_residual_pp"] >= magnitude_threshold_pp)

    # Cross-reference with chase rate and batter names
    chase = data_prep_r2.load_chase_rate()
    chase_map = chase.set_index("batter")[["chase_rate", "n_total_pa", "qualified_200pa"]]
    per_h = per_h.merge(chase_map.reset_index(), on="batter", how="left")
    names = data_prep_r2.load_batter_names()
    per_h["batter_name"] = per_h["batter"].map(names).fillna(per_h["batter"].astype(str))

    per_h = per_h.sort_values("residual_pp_med", ascending=False)
    out_csv = ROOT / f"h5_per_hitter_results{'_drop_pinch' if drop_pinch else ''}.csv"
    per_h.to_csv(out_csv, index=False)

    flagged = per_h[per_h["flagged"]].sort_values("residual_pp_med", ascending=False)
    print(f"[H5] flagged hitters: {len(flagged)}")
    if len(flagged) > 0:
        print(flagged[["batter_name", "n_borderline_spot79", "actual_pct", "expected_pct",
                       "residual_pp_med", "residual_pp_lo", "residual_pp_hi", "chase_rate"]])

    plot_residuals(per_h, CHARTS / f"h5_per_hitter_residuals{'_drop_pinch' if drop_pinch else ''}.png", magnitude_threshold_pp)

    return {
        "n_qualifying_hitters": int(len(per_h)),
        "n_flagged": int(per_h["flagged"].sum()),
        "drop_pinch": drop_pinch,
        "magnitude_threshold_pp": magnitude_threshold_pp,
        "flagged_list": flagged.to_dict(orient="records") if len(flagged) > 0 else [],
        "per_hitter_csv": str(out_csv.resolve()),
        "league_residual_distribution": {
            "median": float(per_h["residual_pp_med"].median()),
            "iqr_low": float(per_h["residual_pp_med"].quantile(0.25)),
            "iqr_high": float(per_h["residual_pp_med"].quantile(0.75)),
            "min": float(per_h["residual_pp_med"].min()),
            "max": float(per_h["residual_pp_med"].max()),
        },
    }


def plot_residuals(per_h: pd.DataFrame, out_path: Path, magnitude_threshold_pp: float = 3.0) -> None:
    df = per_h.sort_values("residual_pp_med").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.18 * len(df))), dpi=140)
    medians = df["residual_pp_med"].values
    los = medians - df["residual_pp_lo"].values
    his = df["residual_pp_hi"].values - medians
    flagged = df["flagged"].values
    colors = np.where(flagged, np.where(df["residual_pp_med"] > 0, "#ef4444", "#10b981"), "#94a3b8")
    pos = np.arange(len(df))
    ax.errorbar(medians, pos, xerr=[los, his], fmt="o", color="#1f2937",
                ecolor=colors, elinewidth=2, capsize=0, markersize=4)
    for i in range(len(df)):
        if flagged[i]:
            ax.plot(medians[i], pos[i], "o", markersize=8, color=colors[i])
            ax.text(medians[i] + (0.4 if medians[i] >= 0 else -0.4), pos[i],
                    df.iloc[i].get("batter_name", str(df.iloc[i]["batter"])),
                    va="center", ha="left" if medians[i] >= 0 else "right",
                    fontsize=8, color=colors[i])
    ax.axvline(0, color="#1d4ed8", linestyle="--", linewidth=1)
    ax.set_yticks([])
    ax.set_xlabel("Per-hitter residual: actual - expected called-strike rate (pp), borderline pitches in spots 7-9\n"
                  "positive = hitter receives MORE called strikes than the league model predicts")
    ax.set_title(f"H5: Per-hitter residuals on borderline pitches in spots 7-9 ({len(df)} qualifying hitters)\n"
                 f"flagged: 95% CI excludes zero AND |residual| >= {magnitude_threshold_pp}pp")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    out = run(drop_pinch=False)
    print(json.dumps(out, indent=2, default=str))

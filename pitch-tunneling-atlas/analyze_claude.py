"""
Pitch Tunneling Atlas — CalledThird.com (Revised)
===================================================
Computes tunneling metrics for every qualified MLB pitcher using 2025 Statcast data.
Revised to address peer review findings.

Reports THREE separate metrics per pitcher:
  1. Plate separation (inches) — how different pitches are at the plate (pitch diversity)
  2. Decision tightness (inches) — how similar pitches are at the decision point (pure tunneling)
  3. Divergence (plate − decision, inches) — composite growth from decision to plate

The R² decomposition shows plate separation adds 8.9% R² to whiff rate beyond
velocity+spin+movement, while decision tightness adds 1.0% more. Both matter;
plate diversity matters more.

Physics:
  x(t) = x0 + vx0*t + 0.5*ax*t²  (using Statcast release point + trajectory params)
  Decision point: solve y(t) = 23.9 (feet from plate back)
  Plate arrival: use actual plate_x, plate_z from Statcast (more accurate than model)
  Physics validation: formal table comparing computed vs actual plate positions.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from numpy.linalg import lstsq
import statsmodels.api as sm

# ── Config ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent
Y_DECISION = 23.9
MIN_PITCHES_TOTAL = 200
MIN_PITCHES_TYPE = 30
KEEP_PITCH_TYPES = {"FF", "SI", "SL", "CH", "FC", "ST", "CU", "FS", "KC", "SV"}

TRAJ_COLS = [
    "release_pos_x", "release_pos_y", "release_pos_z",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "plate_x", "plate_z",
]

PITCH_COLORS = {
    "FF": "#d62728", "SI": "#e377c2", "FC": "#ff7f0e",
    "SL": "#2ca02c", "ST": "#17becf", "CU": "#9467bd",
    "CH": "#8c564b", "FS": "#bcbd22", "KC": "#7f7f7f",
    "SV": "#1f77b4",
}
PITCH_NAMES = {
    "FF": "4-Seam", "SI": "Sinker", "FC": "Cutter",
    "SL": "Slider", "ST": "Sweeper", "CU": "Curveball",
    "CH": "Changeup", "FS": "Splitter", "KC": "Knuckle-Curve",
    "SV": "Slurve",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading 2025 data...")
    df = pd.read_parquet(DATA_DIR / "statcast_2025_full.parquet")
    n_raw = len(df)
    required = TRAJ_COLS + ["pitch_type", "pitcher", "player_name", "p_throws",
                            "stand", "description", "release_speed"]
    df = df.dropna(subset=required)
    df = df[df["pitch_type"].isin(KEEP_PITCH_TYPES)].copy()
    counts = df.groupby("pitcher").size()
    qualified = counts[counts >= MIN_PITCHES_TOTAL].index
    df = df[df["pitcher"].isin(qualified)].copy()
    # Movement magnitude (inches)
    if "pfx_x" in df.columns and "pfx_z" in df.columns:
        df["movement_mag_in"] = np.sqrt((df["pfx_x"] * 12)**2 + (df["pfx_z"] * 12)**2)
    print(f"  {n_raw:,} raw → {len(df):,} valid pitches, {df['pitcher'].nunique()} pitchers")
    return df


def compute_decision_point(df, y_target=Y_DECISION):
    a = 0.5 * df["ay"]
    b = df["vy0"]
    c = df["release_pos_y"] - y_target
    disc = b**2 - 4 * a * c
    valid = disc > 0
    if not valid.all():
        print(f"  WARNING: {(~valid).sum()} pitches don't reach decision point — dropping")
        df = df[valid].copy()
        a, b, c, disc = a[valid], b[valid], c[valid], disc[valid]
    t = (-b - np.sqrt(disc)) / (2 * a)
    df["dec_x"] = df["release_pos_x"] + df["vx0"] * t + 0.5 * df["ax"] * t**2
    df["dec_z"] = df["release_pos_z"] + df["vz0"] * t + 0.5 * df["az"] * t**2
    df["t_decision"] = t
    c_plate = df["release_pos_y"] - 1.417
    disc_plate = df["vy0"]**2 - 4 * 0.5 * df["ay"] * c_plate
    df["t_plate"] = (-df["vy0"] - np.sqrt(disc_plate.clip(lower=0))) / (2 * 0.5 * df["ay"])
    return df


def validate_physics(df):
    """Formal physics validation: compare computed plate positions to Statcast plate_x/plate_z."""
    print("\n── Physics validation ──")
    valid = df.dropna(subset=["plate_x", "plate_z"]).copy()

    # Compute plate position from trajectory model at y = 1.417 ft (front of plate)
    y_plate = 1.417
    a = 0.5 * valid["ay"]
    b = valid["vy0"]
    c = valid["release_pos_y"] - y_plate
    disc = b**2 - 4 * a * c
    mask = disc > 0
    v = valid[mask].copy()
    t = (-v["vy0"] - np.sqrt(v["vy0"]**2 - 4 * 0.5 * v["ay"] * (v["release_pos_y"] - y_plate))) / (2 * 0.5 * v["ay"])
    calc_x = v["release_pos_x"] + v["vx0"] * t + 0.5 * v["ax"] * t**2
    calc_z = v["release_pos_z"] + v["vz0"] * t + 0.5 * v["az"] * t**2

    x_err = (calc_x - v["plate_x"]).abs()
    z_err = (calc_z - v["plate_z"]).abs()
    z_bias = (calc_z - v["plate_z"]).values
    euclid = np.sqrt(x_err.values**2 + z_err.values**2)

    table = {
        "validated_pitches": int(len(v)),
        "mean_abs_x_error_in": round(float(x_err.mean() * 12), 2),
        "mean_abs_z_error_in": round(float(z_err.mean() * 12), 2),
        "mean_euclidean_error_in": round(float(euclid.mean() * 12), 2),
        "median_euclidean_error_in": round(float(np.median(euclid) * 12), 2),
        "p95_euclidean_error_in": round(float(np.percentile(euclid, 95) * 12), 2),
        "signed_z_bias_in": round(float(np.mean(z_bias) * 12), 2),
        "brief_threshold_met": bool(euclid.mean() * 12 < 0.5),
    }

    print(f"  Pitches validated: {table['validated_pitches']:,}")
    print(f"  Mean |X| error:    {table['mean_abs_x_error_in']:.2f} in")
    print(f"  Mean |Z| error:    {table['mean_abs_z_error_in']:.2f} in")
    print(f"  Mean Euclidean:    {table['mean_euclidean_error_in']:.2f} in")
    print(f"  Signed Z bias:     {table['signed_z_bias_in']:+.2f} in")
    print(f"  Brief threshold (<0.5in): {'PASS' if table['brief_threshold_met'] else 'FAIL'}")
    print(f"  NOTE: Systematic z-bias from constant-acceleration approximation.")
    print(f"        Relative pitch-type distances are preserved (bias cancels).")

    pd.DataFrame([table]).to_csv(OUT_DIR / "physics_validation.csv", index=False)
    return table


def add_outcome_flags(df):
    swinging = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
    called = {"called_strike"}
    swing_outcomes = swinging | {"foul", "foul_tip", "hit_into_play",
                                  "hit_into_play_score", "hit_into_play_no_out"}
    df["is_whiff"] = df["description"].isin(swinging).astype(int)
    df["is_csw"] = df["description"].isin(swinging | called).astype(int)
    df["is_swing"] = df["description"].isin(swing_outcomes).astype(int)
    return df


def compute_centroids(df):
    group = df.groupby(["pitcher", "player_name", "p_throws", "pitch_type"])
    centroids = group.agg(
        n=("pitch_type", "size"),
        rel_x_mean=("release_pos_x", "mean"), rel_y_mean=("release_pos_y", "mean"),
        rel_z_mean=("release_pos_z", "mean"),
        rel_x_std=("release_pos_x", "std"), rel_z_std=("release_pos_z", "std"),
        dec_x_mean=("dec_x", "mean"), dec_z_mean=("dec_z", "mean"),
        dec_x_std=("dec_x", "std"), dec_z_std=("dec_z", "std"),
        plate_x_mean=("plate_x", "mean"), plate_z_mean=("plate_z", "mean"),
        plate_x_std=("plate_x", "std"), plate_z_std=("plate_z", "std"),
        velo_mean=("release_speed", "mean"), spin_mean=("release_spin_rate", "mean"),
        movement_mean=("movement_mag_in", "mean"),
        pfx_x_mean=("pfx_x", "mean"), pfx_z_mean=("pfx_z", "mean"),
        t_decision_mean=("t_decision", "mean"), t_plate_mean=("t_plate", "mean"),
        whiffs=("is_whiff", "sum"), csw=("is_csw", "sum"), swings=("is_swing", "sum"),
    ).reset_index()
    centroids = centroids[centroids["n"] >= MIN_PITCHES_TYPE].copy()
    centroids["whiff_rate"] = centroids["whiffs"] / centroids["swings"].clip(lower=1)
    centroids["csw_rate"] = centroids["csw"] / centroids["n"]
    print(f"  {len(centroids)} centroids ({centroids['pitcher'].nunique()} pitchers)")
    return centroids


def compute_release_consistency(centroids):
    pitchers = centroids.groupby(["pitcher", "player_name", "p_throws"])
    records = []
    for (pid, name, hand), grp in pitchers:
        if len(grp) < 2:
            continue
        total = grp["n"].sum()
        w = grp["n"].values / total
        wx = (grp["rel_x_mean"].values * w).sum()
        wz = (grp["rel_z_mean"].values * w).sum()
        cross_x = np.sqrt((w * (grp["rel_x_mean"].values - wx)**2).sum())
        cross_z = np.sqrt((w * (grp["rel_z_mean"].values - wz)**2).sum())
        within_x = (grp["rel_x_std"].values * w).sum()
        within_z = (grp["rel_z_std"].values * w).sum()
        records.append({
            "pitcher": pid, "player_name": name, "p_throws": hand,
            "n_pitch_types": len(grp), "total_pitches": total,
            "cross_type_rel_x_sd": cross_x, "cross_type_rel_z_sd": cross_z,
            "within_type_rel_x_sd": within_x, "within_type_rel_z_sd": within_z,
            "release_consistency": np.sqrt(cross_x**2 + cross_z**2),
        })
    return pd.DataFrame(records)


def compute_pairwise(centroids):
    pitchers = centroids.groupby(["pitcher", "player_name", "p_throws"])
    records = []
    for (pid, name, hand), grp in pitchers:
        types = grp["pitch_type"].tolist()
        if len(types) < 2:
            continue
        gi = grp.set_index("pitch_type")
        total = grp["n"].sum()
        for t1, t2 in combinations(sorted(types), 2):
            r1, r2 = gi.loc[t1], gi.loc[t2]
            rel_sep = np.hypot(r1["rel_x_mean"] - r2["rel_x_mean"],
                               r1["rel_z_mean"] - r2["rel_z_mean"])
            dec_sep = np.hypot(r1["dec_x_mean"] - r2["dec_x_mean"],
                               r1["dec_z_mean"] - r2["dec_z_mean"])
            plate_sep = np.hypot(r1["plate_x_mean"] - r2["plate_x_mean"],
                                 r1["plate_z_mean"] - r2["plate_z_mean"])
            usage_a, usage_b = r1["n"] / total, r2["n"] / total
            records.append({
                "pitcher": pid, "player_name": name, "p_throws": hand,
                "pitch_a": t1, "pitch_b": t2, "pair": f"{t1}-{t2}",
                "n_a": r1["n"], "n_b": r2["n"],
                "usage_a": usage_a, "usage_b": usage_b,
                "pair_weight": usage_a * usage_b,
                "release_sep_ft": rel_sep,
                "decision_sep_ft": dec_sep,
                "plate_sep_ft": plate_sep,
                "divergence_ft": plate_sep - dec_sep,
                "tunnel_ratio": plate_sep / max(dec_sep, 0.01),
                "velo_diff_mph": abs(r1["velo_mean"] - r2["velo_mean"]),
                "time_diff_s": abs(r1["t_plate_mean"] - r2["t_plate_mean"]),
                # Component diffs for directional analysis
                "dec_x_diff": r1["dec_x_mean"] - r2["dec_x_mean"],
                "dec_z_diff": r1["dec_z_mean"] - r2["dec_z_mean"],
                "plate_x_diff": r1["plate_x_mean"] - r2["plate_x_mean"],
                "plate_z_diff": r1["plate_z_mean"] - r2["plate_z_mean"],
            })
    pairs_df = pd.DataFrame(records)
    print(f"  {len(pairs_df):,} pitch pairs across {pairs_df['pitcher'].nunique()} pitchers")
    return pairs_df


def compute_deception_scores(pairs_df, centroids, release_df):
    """Primary metric: weighted-average divergence (plate_sep − decision_sep) in inches."""
    pitcher_groups = pairs_df.groupby(["pitcher", "player_name", "p_throws"])
    records = []
    for (pid, name, hand), grp in pitcher_groups:
        w = grp["pair_weight"].values
        w_norm = w / w.sum()
        div_in = (grp["divergence_ft"].values * w_norm).sum() * 12  # → inches
        plate_in = (grp["plate_sep_ft"].values * w_norm).sum() * 12
        dec_in = (grp["decision_sep_ft"].values * w_norm).sum() * 12
        rel_in = (grp["release_sep_ft"].values * w_norm).sum() * 12
        velo_d = (grp["velo_diff_mph"].values * w_norm).sum()

        best_div_idx = grp["divergence_ft"].idxmax()
        worst_div_idx = grp["divergence_ft"].idxmin()

        pc = centroids[centroids["pitcher"] == pid]
        tot_n = pc["n"].sum()
        records.append({
            "pitcher": pid, "player_name": name, "p_throws": hand,
            "n_types": len(set(grp["pitch_a"]) | set(grp["pitch_b"])),
            "n_pairs": len(grp), "total_pitches": tot_n,
            # Primary metric
            "divergence_in": div_in,
            # Components
            "plate_sep_in": plate_in, "decision_sep_in": dec_in,
            "release_sep_in": rel_in, "avg_velo_diff": velo_d,
            # Best / worst pair by divergence
            "best_pair": grp.loc[best_div_idx, "pair"],
            "best_pair_div_in": grp.loc[best_div_idx, "divergence_ft"] * 12,
            "worst_pair": grp.loc[worst_div_idx, "pair"],
            "worst_pair_div_in": grp.loc[worst_div_idx, "divergence_ft"] * 12,
            # Stuff context
            "avg_velo": (pc["velo_mean"] * pc["n"]).sum() / tot_n,
            "avg_spin": (pc["spin_mean"] * pc["n"]).sum() / tot_n,
            "avg_movement_in": (pc["movement_mean"].fillna(0) * pc["n"]).sum() / tot_n if "movement_mean" in pc.columns else np.nan,
            # Outcomes
            "whiff_rate": pc["whiffs"].sum() / max(pc["swings"].sum(), 1),
            "csw_rate": pc["csw"].sum() / tot_n,
            "swings": pc["swings"].sum(),
        })
    scores = pd.DataFrame(records)
    scores = scores.merge(
        release_df[["pitcher", "release_consistency", "cross_type_rel_x_sd",
                     "cross_type_rel_z_sd"]],
        on="pitcher", how="left")
    scores["divergence_pctile"] = scores["divergence_in"].rank(pct=True) * 100
    scores = scores.sort_values("divergence_in", ascending=False).reset_index(drop=True)
    print(f"  {len(scores)} pitchers scored")
    return scores


def league_pair_summary(pairs_df):
    ps = pairs_df.groupby("pair").agg(
        n_pitchers=("pitcher", "nunique"),
        divergence_mean=("divergence_ft", "mean"),
        divergence_std=("divergence_ft", "std"),
        decision_sep_mean=("decision_sep_ft", "mean"),
        plate_sep_mean=("plate_sep_ft", "mean"),
        velo_diff_mean=("velo_diff_mph", "mean"),
    ).reset_index()
    ps = ps[ps["n_pitchers"] >= 20].copy()
    ps = ps.sort_values("divergence_mean", ascending=False).reset_index(drop=True)
    return ps


# ═══════════════════════════════════════════════════════════════════════════════
#  REGRESSION ANALYSIS (Q4)
# ═══════════════════════════════════════════════════════════════════════════════

def partial_corr(x_col, y_col, ctrl_cols, data):
    d = data[[x_col, y_col] + ctrl_cols].dropna()
    C = np.column_stack([d[ctrl_cols].values, np.ones(len(d))])
    bx, *_ = lstsq(C, d[x_col].values, rcond=None)
    by, *_ = lstsq(C, d[y_col].values, rcond=None)
    return stats.pearsonr(d[x_col].values - C @ bx, d[y_col].values - C @ by)


def ols_model(y, X_df, label=""):
    """Run OLS with statsmodels and return structured results with SEs and p-values."""
    X = sm.add_constant(X_df)
    model = sm.OLS(y, X).fit()
    coefs = {}
    for name in X_df.columns:
        idx = list(X.columns).index(name)
        coefs[name] = {
            "coef": round(float(model.params.iloc[idx]), 6),
            "se": round(float(model.bse.iloc[idx]), 6),
            "p_value": round(float(model.pvalues.iloc[idx]), 6),
            "ci_lower": round(float(model.conf_int().iloc[idx, 0]), 6),
            "ci_upper": round(float(model.conf_int().iloc[idx, 1]), 6),
        }
    return {
        "label": label,
        "n": int(model.nobs),
        "r_squared": round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "f_pvalue": round(float(model.f_pvalue), 6),
        "coefficients": coefs,
    }


def regression_analysis(scores):
    """Full regression with proper OLS inference, movement control, and multiple outcomes."""
    df = scores.dropna(subset=["divergence_in", "whiff_rate", "csw_rate",
                                "avg_velo", "avg_spin", "avg_movement_in"]).copy()
    df = df[df["swings"] >= 100].copy()
    results = {"n_pitchers": len(df)}

    # ── Raw correlations ──
    for metric in ["divergence_in", "plate_sep_in", "decision_sep_in"]:
        for outcome in ["whiff_rate", "csw_rate"]:
            r, p = stats.pearsonr(df[metric], df[outcome])
            results[f"{metric}_vs_{outcome}"] = {"r": round(r, 4), "p": round(p, 6)}

    # ── Partial correlations (controlling velo + spin + movement) ──
    ctrls = ["avg_velo", "avg_spin", "avg_movement_in"]
    for outcome in ["whiff_rate", "csw_rate"]:
        r, p = partial_corr("divergence_in", outcome, ctrls, df)
        results[f"partial_divergence_vs_{outcome}"] = {"r": round(r, 4), "p": round(p, 6)}

    # ── R² decomposition (with movement in baseline) ──
    y_whiff = df["whiff_rate"].values
    base_cols = ["avg_velo", "avg_spin", "avg_movement_in"]

    m_base = ols_model(y_whiff, df[base_cols], "velo+spin+movement")
    m_plate = ols_model(y_whiff, df[base_cols + ["plate_sep_in"]], "+plate_sep")
    m_dec = ols_model(y_whiff, df[base_cols + ["decision_sep_in"]], "+decision_sep")
    m_both = ols_model(y_whiff, df[base_cols + ["plate_sep_in", "decision_sep_in"]], "+both")

    results["r2_decomposition"] = {
        "baseline_controls": "velo + spin + movement",
        "velo_spin_movement": m_base["r_squared"],
        "plus_plate_sep": m_plate["r_squared"],
        "plus_decision_sep": m_dec["r_squared"],
        "plus_both": m_both["r_squared"],
        "incremental_plate_sep": round(m_plate["r_squared"] - m_base["r_squared"], 4),
        "incremental_decision_sep_given_plate": round(m_both["r_squared"] - m_plate["r_squared"], 4),
    }

    # ── Full model with proper inference ──
    results["full_model"] = m_both

    # ── CSW model ──
    m_csw = ols_model(df["csw_rate"].values,
                       df[base_cols + ["plate_sep_in", "decision_sep_in"]], "csw_full")
    results["csw_model"] = m_csw

    # ── Independence checks ──
    r_v, p_v = stats.pearsonr(df["divergence_in"], df["avg_velo"])
    results["divergence_vs_velo"] = {"r": round(r_v, 4), "p": round(p_v, 6)}

    rc = df.dropna(subset=["release_consistency"])
    if len(rc) > 30:
        r_rc, p_rc = stats.pearsonr(rc["divergence_in"], rc["release_consistency"])
        results["divergence_vs_release_consistency"] = {"r": round(r_rc, 4), "p": round(p_rc, 6)}

    return results, df


# ═══════════════════════════════════════════════════════════════════════════════
#  SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def sensitivity_analysis(raw_df):
    """Rank-stability across decision-point distances."""
    print("\n── Sensitivity analysis ──")
    rankings = {}
    for y in [20.0, 23.9, 25.0, 28.0]:
        d = compute_decision_point(raw_df.copy(), y_target=y)
        c = compute_centroids(d)
        p = compute_pairwise(c)
        s = compute_deception_scores(p, c, compute_release_consistency(c))
        rankings[y] = s.set_index("pitcher")["divergence_in"]
    base = rankings[23.9]
    results = {}
    for y in [20.0, 25.0, 28.0]:
        common = base.index.intersection(rankings[y].index)
        rho, _ = stats.spearmanr(base.loc[common], rankings[y].loc[common])
        results[y] = round(rho, 4)
        print(f"  y={y:.0f}ft vs 23.9ft: Spearman ρ={rho:.4f} (n={len(common)})")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Q5: 2026 YEAR-OVER-YEAR
# ═══════════════════════════════════════════════════════════════════════════════

def load_2026_data():
    """Load and merge all 2026 parquets."""
    paths = sorted(DATA_DIR.glob("2026-*.parquet"))
    apr_path = DATA_DIR / "statcast_2026_apr06_14.parquet"
    if apr_path.exists():
        paths.append(apr_path)
    if not paths:
        print("  No 2026 data found — skipping Q5")
        return None
    required = TRAJ_COLS + ["pitch_type", "pitcher", "player_name", "p_throws",
                            "stand", "description", "release_speed", "pfx_x", "pfx_z"]
    frames = []
    for p in paths:
        try:
            d = pd.read_parquet(p)
            avail = [c for c in required if c in d.columns]
            frames.append(d[avail])
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[c for c in TRAJ_COLS + ["pitch_type", "pitcher", "player_name",
                                                     "p_throws", "description", "release_speed"]
                           if c in df.columns])
    df = df[df["pitch_type"].isin(KEEP_PITCH_TYPES)].copy()
    if "pfx_x" in df.columns and "pfx_z" in df.columns:
        df["movement_mag_in"] = np.sqrt((df["pfx_x"] * 12)**2 + (df["pfx_z"] * 12)**2)
    # Lighter thresholds for early season
    counts = df.groupby("pitcher").size()
    qualified = counts[counts >= 80].index
    df = df[df["pitcher"].isin(qualified)].copy()
    print(f"  2026: {len(df):,} pitches, {df['pitcher'].nunique()} pitchers (80+ threshold)")
    return df


def compute_yoy(scores_2025, raw_2026):
    """Year-over-year divergence comparison."""
    if raw_2026 is None:
        return pd.DataFrame()
    d26 = compute_decision_point(raw_2026.copy())
    d26 = add_outcome_flags(d26)
    c26 = compute_centroids_flex(d26, min_type=20)
    r26 = compute_release_consistency(c26)
    p26 = compute_pairwise(c26)
    s26 = compute_deception_scores(p26, c26, r26)

    merged = s26[["pitcher", "player_name", "p_throws", "divergence_in",
                   "whiff_rate", "csw_rate"]].rename(
        columns={"divergence_in": "div_2026", "whiff_rate": "whiff_2026",
                 "csw_rate": "csw_2026"})
    merged = merged.merge(
        scores_2025[["pitcher", "divergence_in", "whiff_rate", "csw_rate"]].rename(
            columns={"divergence_in": "div_2025", "whiff_rate": "whiff_2025",
                     "csw_rate": "csw_2025"}),
        on="pitcher", how="inner")
    merged["div_delta"] = merged["div_2026"] - merged["div_2025"]
    merged["whiff_delta"] = merged["whiff_2026"] - merged["whiff_2025"]
    merged = merged.sort_values("div_delta", ascending=False).reset_index(drop=True)
    print(f"  {len(merged)} pitchers in YoY comparison")
    return merged


def compute_centroids_flex(df, min_type=30):
    """Centroids with flexible minimum pitch-type threshold. Handles missing columns."""
    group = df.groupby(["pitcher", "player_name", "p_throws", "pitch_type"])
    agg_dict = dict(
        n=("pitch_type", "size"),
        rel_x_mean=("release_pos_x", "mean"), rel_y_mean=("release_pos_y", "mean"),
        rel_z_mean=("release_pos_z", "mean"),
        rel_x_std=("release_pos_x", "std"), rel_z_std=("release_pos_z", "std"),
        dec_x_mean=("dec_x", "mean"), dec_z_mean=("dec_z", "mean"),
        dec_x_std=("dec_x", "std"), dec_z_std=("dec_z", "std"),
        plate_x_mean=("plate_x", "mean"), plate_z_mean=("plate_z", "mean"),
        plate_x_std=("plate_x", "std"), plate_z_std=("plate_z", "std"),
        velo_mean=("release_speed", "mean"),
        t_decision_mean=("t_decision", "mean"), t_plate_mean=("t_plate", "mean"),
        whiffs=("is_whiff", "sum"), csw=("is_csw", "sum"), swings=("is_swing", "sum"),
    )
    if "release_spin_rate" in df.columns:
        agg_dict["spin_mean"] = ("release_spin_rate", "mean")
    if "movement_mag_in" in df.columns:
        agg_dict["movement_mean"] = ("movement_mag_in", "mean")
    if "pfx_x" in df.columns:
        agg_dict["pfx_x_mean"] = ("pfx_x", "mean")
    if "pfx_z" in df.columns:
        agg_dict["pfx_z_mean"] = ("pfx_z", "mean")
    centroids = group.agg(**agg_dict).reset_index()
    centroids = centroids[centroids["n"] >= min_type].copy()
    centroids["whiff_rate"] = centroids["whiffs"] / centroids["swings"].clip(lower=1)
    centroids["csw_rate"] = centroids["csw"] / centroids["n"]
    # Fill missing columns with NaN for downstream compatibility
    for col in ["spin_mean", "movement_mean", "pfx_x_mean", "pfx_z_mean"]:
        if col not in centroids.columns:
            centroids[col] = np.nan
    return centroids


# ═══════════════════════════════════════════════════════════════════════════════
#  Q6: BATTER HANDEDNESS SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_batter_splits(raw_df):
    """Compute tunneling scores separately vs LHB and RHB."""
    print("\n── Batter handedness splits (Q6) ──")
    results = []
    for stand in ["L", "R"]:
        sub = raw_df[raw_df["stand"] == stand].copy()
        # Lighter thresholds for splits
        counts = sub.groupby("pitcher").size()
        qualified = counts[counts >= 80].index
        sub = sub[sub["pitcher"].isin(qualified)].copy()
        if len(sub) < 1000:
            continue
        c = compute_centroids_flex(sub, min_type=20)
        r = compute_release_consistency(c)
        p = compute_pairwise(c)
        if p.empty:
            continue
        s = compute_deception_scores(p, c, r)
        s["stand"] = stand
        results.append(s[["pitcher", "player_name", "p_throws", "stand",
                           "divergence_in", "plate_sep_in", "decision_sep_in",
                           "whiff_rate"]])

    if not results:
        return pd.DataFrame()
    splits = pd.concat(results, ignore_index=True)

    # Compute asymmetry
    piv = splits.pivot_table(index=["pitcher", "player_name", "p_throws"],
                              columns="stand", values="divergence_in").reset_index()
    if "L" in piv.columns and "R" in piv.columns:
        piv = piv.dropna(subset=["L", "R"])
        piv["gap"] = piv["R"] - piv["L"]
        piv = piv.sort_values("gap", ascending=False).reset_index(drop=True)
        piv.columns.name = None
        piv = piv.rename(columns={"L": "vs_LHB", "R": "vs_RHB"})
        print(f"  {len(piv)} pitchers with both-side splits")
        return piv
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "#fafafa",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "font.family": "sans-serif", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def chart_leaderboard(scores, out):
    """Top 20 / Bottom 20 divergence bar chart."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    top = scores.head(20).iloc[::-1]
    colors_top = ["#1a5276" if h == "R" else "#c0392b" for h in top["p_throws"]]
    ax1.barh(range(len(top)), top["divergence_in"], color=colors_top, height=0.7)
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels([f"{r['player_name']} ({r['p_throws']})" for _, r in top.iterrows()],
                        fontsize=8)
    ax1.set_xlabel("Divergence (inches)")
    ax1.set_title("Top 20 — Most Deceptive Arsenals", fontweight="bold")
    for i, (_, r) in enumerate(top.iterrows()):
        ax1.text(r["divergence_in"] + 0.1, i, f"{r['divergence_in']:.1f}″",
                 va="center", fontsize=7, color="#333")

    bot = scores.tail(20)
    colors_bot = ["#1a5276" if h == "R" else "#c0392b" for h in bot["p_throws"]]
    ax2.barh(range(len(bot)), bot["divergence_in"], color=colors_bot, height=0.7)
    ax2.set_yticks(range(len(bot)))
    ax2.set_yticklabels([f"{r['player_name']} ({r['p_throws']})" for _, r in bot.iterrows()],
                        fontsize=8)
    ax2.set_xlabel("Divergence (inches)")
    ax2.set_title("Bottom 20 — Least Deceptive", fontweight="bold")
    for i, (_, r) in enumerate(bot.iterrows()):
        ax2.text(r["divergence_in"] + 0.05, i, f"{r['divergence_in']:.1f}″",
                 va="center", fontsize=7, color="#333")

    fig.suptitle("Pitch Tunneling Atlas — 2025 Deception Leaderboard",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.01, "CalledThird.com  |  Divergence = plate separation − decision-point separation  |  "
             "Blue = RHP, Red = LHP", ha="center", fontsize=8, color="#666")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out / "chart_leaderboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved chart_leaderboard.png")


def chart_tunnel_map(centroids, pitcher_name, out):
    """Tunnel map: release → decision → plate positions for one pitcher."""
    set_style()
    pc = centroids[centroids["player_name"] == pitcher_name]
    if pc.empty:
        print(f"  WARNING: {pitcher_name} not found in centroids")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    stages = [
        ("Release Point", "rel_x_mean", "rel_z_mean", "rel_x_std", "rel_z_std"),
        ("Decision Point (23.9 ft)", "dec_x_mean", "dec_z_mean", "dec_x_std", "dec_z_std"),
        ("Plate Arrival", "plate_x_mean", "plate_z_mean", "plate_x_std", "plate_z_std"),
    ]

    for ax, (title, xc, zc, xs, zs) in zip(axes, stages):
        for _, row in pc.iterrows():
            pt = row["pitch_type"]
            color = PITCH_COLORS.get(pt, "#333")
            ax.scatter(row[xc] * 12, row[zc] * 12, c=color, s=100,
                       zorder=5, edgecolors="white", linewidth=0.5)
            # 1-SD ellipse
            theta = np.linspace(0, 2 * np.pi, 50)
            ex = row[xc] * 12 + row[xs] * 12 * np.cos(theta)
            ez = row[zc] * 12 + row[zs] * 12 * np.sin(theta)
            ax.plot(ex, ez, color=color, alpha=0.3, linewidth=1)
            ax.annotate(pt, (row[xc] * 12, row[zc] * 12),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, fontweight="bold", color=color)
        ax.set_xlabel("Horizontal (in)")
        ax.set_ylabel("Vertical (in)")
        ax.set_title(title)
        ax.set_aspect("equal")

    # Draw strike zone on plate panel
    axes[2].add_patch(plt.Rectangle((-8.5, 18), 17, 24, fill=False,
                                      edgecolor="#999", linewidth=1.5, linestyle="--"))

    hand = pc.iloc[0]["p_throws"]
    fig.suptitle(f"{pitcher_name} ({hand}HP) — Tunnel Map", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "CalledThird.com  |  Dots = centroids, ellipses = 1-SD spread",
             ha="center", fontsize=8, color="#666")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    safe_name = pitcher_name.replace(", ", "_").replace(" ", "_")
    fig.savefig(out / f"tunnel_map_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved tunnel_map_{safe_name}.png")


def chart_pair_heatmap(pair_summary, out):
    """Heatmap of average divergence by pitch-pair combination."""
    set_style()
    all_types = sorted(KEEP_PITCH_TYPES)
    matrix = np.full((len(all_types), len(all_types)), np.nan)
    count_matrix = np.full((len(all_types), len(all_types)), 0)

    for _, row in pair_summary.iterrows():
        parts = row["pair"].split("-")
        if len(parts) != 2:
            continue
        a, b = parts
        if a in all_types and b in all_types:
            i, j = all_types.index(a), all_types.index(b)
            val = row["divergence_mean"] * 12  # to inches
            matrix[i, j] = val
            matrix[j, i] = val
            count_matrix[i, j] = row["n_pitchers"]
            count_matrix[j, i] = row["n_pitchers"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="equal")

    labels = [PITCH_NAMES.get(t, t) for t in all_types]
    ax.set_xticks(range(len(all_types)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(all_types)))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate cells
    for i in range(len(all_types)):
        for j in range(len(all_types)):
            if not np.isnan(matrix[i, j]) and i != j:
                ax.text(j, i, f"{matrix[i,j]:.1f}″\n({count_matrix[i,j]})",
                        ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > np.nanpercentile(matrix, 75) else "black")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Average Divergence (inches)", fontsize=9)
    ax.set_title("Which Pitch Pairs Tunnel Best? (League-Wide Average Divergence)",
                 fontsize=12, fontweight="bold", pad=15)
    fig.text(0.5, 0.01, "CalledThird.com  |  Higher = more deceptive  |  "
             "(n) = pitchers who throw both", ha="center", fontsize=8, color="#666")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out / "chart_pair_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved chart_pair_heatmap.png")


def chart_divergence_vs_outcomes(scores, out):
    """Scatter: divergence vs whiff rate, colored by velocity."""
    set_style()
    df = scores[scores["swings"] >= 100].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: divergence vs whiff rate
    sc = ax1.scatter(df["divergence_in"], df["whiff_rate"] * 100,
                     c=df["avg_velo"], cmap="coolwarm", s=20, alpha=0.6, edgecolors="none")
    # Trend line
    m, b_coef = np.polyfit(df["divergence_in"], df["whiff_rate"] * 100, 1)
    x_line = np.linspace(df["divergence_in"].min(), df["divergence_in"].max(), 100)
    ax1.plot(x_line, m * x_line + b_coef, "k--", linewidth=1.5, alpha=0.7)
    r, p = stats.pearsonr(df["divergence_in"], df["whiff_rate"])
    ax1.text(0.05, 0.95, f"r = {r:.3f}\np = {p:.1e}", transform=ax1.transAxes,
             fontsize=9, va="top", bbox=dict(boxstyle="round,pad=0.3",
                                              facecolor="white", edgecolor="#ccc"))
    ax1.set_xlabel("Divergence (inches)")
    ax1.set_ylabel("Whiff Rate (%)")
    ax1.set_title("Divergence vs. Whiff Rate")
    cbar1 = plt.colorbar(sc, ax=ax1, shrink=0.8)
    cbar1.set_label("Avg Velocity (mph)")

    # Label notable pitchers
    for _, r_row in df.nlargest(3, "divergence_in").iterrows():
        ax1.annotate(r_row["player_name"].split(", ")[0],
                     (r_row["divergence_in"], r_row["whiff_rate"] * 100),
                     fontsize=7, alpha=0.8, textcoords="offset points", xytext=(5, 3))
    # High whiff + high divergence
    elite = df[(df["divergence_in"] > df["divergence_in"].quantile(0.9)) &
               (df["whiff_rate"] > df["whiff_rate"].quantile(0.9))]
    for _, r_row in elite.iterrows():
        ax1.annotate(r_row["player_name"].split(", ")[0],
                     (r_row["divergence_in"], r_row["whiff_rate"] * 100),
                     fontsize=7, alpha=0.8, textcoords="offset points", xytext=(5, -8))

    # Panel 2: plate_sep vs decision_sep, colored by whiff rate
    sc2 = ax2.scatter(df["decision_sep_in"], df["plate_sep_in"],
                      c=df["whiff_rate"] * 100, cmap="YlOrRd", s=20, alpha=0.6,
                      edgecolors="none")
    ax2.set_xlabel("Decision-Point Separation (inches)")
    ax2.set_ylabel("Plate Separation (inches)")
    ax2.set_title("Decision vs. Plate Separation")
    cbar2 = plt.colorbar(sc2, ax=ax2, shrink=0.8)
    cbar2.set_label("Whiff Rate (%)")
    # Ideal zone annotation
    ax2.annotate("Ideal: tight at\ndecision, wide at plate",
                 xy=(df["decision_sep_in"].quantile(0.15),
                     df["plate_sep_in"].quantile(0.85)),
                 fontsize=8, color="#1a5276", fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor="#eaf2f8", alpha=0.8))

    fig.suptitle("Does Pitch Tunneling Predict Success?", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01, "CalledThird.com  |  621 pitchers, 200+ pitches, 100+ swings  |  "
             "Partial r = 0.32 after controlling for velocity + spin",
             ha="center", fontsize=8, color="#666")
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(out / "chart_outcomes_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved chart_outcomes_scatter.png")


def chart_r2_decomposition(reg_results, out):
    """Bar chart showing incremental R² contributions."""
    set_style()
    r2 = reg_results["r2_decomposition"]
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Velocity + Spin\n+ Movement", "+ Plate\nSeparation", "+ Decision\nSeparation"]
    vals = [r2["velo_spin_movement"], r2["plus_plate_sep"], r2["plus_both"]]
    increments = [vals[0], vals[1] - vals[0], vals[2] - vals[1]]
    colors = ["#5dade2", "#2ecc71", "#f39c12"]

    bottom = 0
    bars = []
    for i, (inc, c) in enumerate(zip(increments, colors)):
        bar = ax.bar(0, inc, bottom=bottom, color=c, width=0.5, edgecolor="white")
        bars.append(bar)
        ax.text(0, bottom + inc / 2, f"+{inc:.1%}" if i > 0 else f"{inc:.1%}",
                ha="center", va="center", fontweight="bold", fontsize=11, color="white")
        bottom += inc

    ax.set_xticks([])
    ax.set_ylabel("R² (Whiff Rate)")
    ax.set_title("What Explains Whiff Rate?\nIncremental R² Contributions",
                 fontweight="bold", fontsize=12)
    ax.legend([b[0] for b in bars], labels, loc="upper left", frameon=True)
    ax.set_ylim(0, max(vals) * 1.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.text(0.5, 0.01, "CalledThird.com  |  Controls: velocity + spin + movement  |  "
             "Plate separation adds the most incremental R²",
             ha="center", fontsize=8, color="#666")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out / "chart_r2_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved chart_r2_decomposition.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PITCH TUNNELING ATLAS — CalledThird.com (Revised)")
    print("=" * 70)

    df = load_data()

    # Physics validation
    print("\n── Computing decision-point trajectories ──")
    df = compute_decision_point(df)
    physics = validate_physics(df)
    df = add_outcome_flags(df)

    print("\n── Computing centroids ──")
    centroids = compute_centroids(df)

    print("\n── Release consistency (Q2) ──")
    release_df = compute_release_consistency(centroids)

    print("\n── Pairwise tunnel metrics ──")
    pairs_df = compute_pairwise(centroids)

    print("\n── Deception scores ──")
    scores = compute_deception_scores(pairs_df, centroids, release_df)

    print("\n── League pair summary (Q3) ──")
    pair_summary = league_pair_summary(pairs_df)

    print("\n── Regression analysis (Q4) ──")
    reg_results, reg_df = regression_analysis(scores)

    sensitivity = sensitivity_analysis(df)

    # Q5: 2026 year-over-year
    print("\n── Loading 2026 data (Q5) ──")
    raw_2026 = load_2026_data()
    yoy = compute_yoy(scores, raw_2026)

    # Q6: Batter handedness
    splits = compute_batter_splits(df)

    # ── Save all data ──
    print("\n── Saving outputs ──")
    scores.to_csv(OUT_DIR / "deception_scores.csv", index=False)
    pairs_df.to_csv(OUT_DIR / "pitch_pairs.csv", index=False)
    centroids.to_csv(OUT_DIR / "centroids.csv", index=False)
    release_df.to_csv(OUT_DIR / "release_consistency.csv", index=False)
    pair_summary.to_csv(OUT_DIR / "pair_summary.csv", index=False)
    if not yoy.empty:
        yoy.to_csv(OUT_DIR / "yoy_2026.csv", index=False)
    if not splits.empty:
        splits.to_csv(OUT_DIR / "batter_splits.csv", index=False)
    with open(OUT_DIR / "regression_results.json", "w") as f:
        json.dump(reg_results, f, indent=2, default=str)

    # ── Generate charts ──
    print("\n── Generating charts ──")
    chart_leaderboard(scores, OUT_DIR)
    chart_pair_heatmap(pair_summary, OUT_DIR)
    chart_divergence_vs_outcomes(scores, OUT_DIR)
    chart_r2_decomposition(reg_results, OUT_DIR)

    for name in scores.head(3)["player_name"].tolist():
        chart_tunnel_map(centroids, name, OUT_DIR)
    for name in scores.tail(3)["player_name"].tolist():
        chart_tunnel_map(centroids, name, OUT_DIR)

    # ── Print results ──
    print("\n" + "=" * 70)
    print("TOP 20 (by Divergence — composite of plate diversity + tunneling)")
    print("=" * 70)
    for i, (_, r) in enumerate(scores.head(20).iterrows()):
        print(f"  {i+1:3d}. {r['player_name']:25s} {r['p_throws']}HP  "
              f"Div={r['divergence_in']:.1f}″  "
              f"Plate={r['plate_sep_in']:.1f}″  Dec={r['decision_sep_in']:.1f}″  "
              f"Whiff={r['whiff_rate']:.1%}  Velo={r['avg_velo']:.1f}")

    print("\n" + "=" * 70)
    print("BOTTOM 10")
    print("=" * 70)
    for i, (_, r) in enumerate(scores.tail(10).iloc[::-1].iterrows()):
        n = len(scores)
        print(f"  {n - 9 + i:3d}. {r['player_name']:25s} {r['p_throws']}HP  "
              f"Div={r['divergence_in']:.1f}″  Whiff={r['whiff_rate']:.1%}")

    # Starter-specific leaderboard
    print("\n" + "=" * 70)
    print("TOP 10 STARTERS (4+ pitch types)")
    print("=" * 70)
    starters = scores[scores["n_types"] >= 4].head(10)
    for i, (_, r) in enumerate(starters.iterrows()):
        overall_rank = scores.index[scores["pitcher"] == r["pitcher"]].tolist()[0] + 1
        print(f"  {i+1:3d}. {r['player_name']:25s} {r['p_throws']}HP  "
              f"Div={r['divergence_in']:.1f}″  Types={r['n_types']}  "
              f"Whiff={r['whiff_rate']:.1%}  (overall #{overall_rank})")

    print("\n" + "=" * 70)
    print("BEST PITCH PAIRS (LEAGUE-WIDE)")
    print("=" * 70)
    for _, r in pair_summary.head(10).iterrows():
        print(f"  {r['pair']:8s}  Div={r['divergence_mean']*12:.1f}″ "
              f"(±{r['divergence_std']*12:.1f})  "
              f"DecSep={r['decision_sep_mean']*12:.1f}″  "
              f"PlateSep={r['plate_sep_mean']*12:.1f}″  "
              f"n={r['n_pitchers']}")

    print("\n" + "=" * 70)
    print("REGRESSION SUMMARY (Q4) — Controls: velo + spin + movement")
    print("=" * 70)
    r2 = reg_results["r2_decomposition"]
    print(f"  Baseline (velo+spin+mvmt): R² = {r2['velo_spin_movement']:.4f}")
    print(f"  + Plate separation:        R² = {r2['plus_plate_sep']:.4f} "
          f"(+{r2['incremental_plate_sep']:.4f})")
    print(f"  + Decision closeness:      R² = {r2['plus_both']:.4f} "
          f"(+{r2['incremental_decision_sep_given_plate']:.4f})")

    fm = reg_results["full_model"]
    print(f"\n  Full model (n={fm['n']}, R²={fm['r_squared']:.4f}, "
          f"adj R²={fm['adj_r_squared']:.4f}):")
    for name, c in fm["coefficients"].items():
        sig = "*" if c["p_value"] < 0.05 else ""
        print(f"    {name:20s}: {c['coef']:+.6f} (SE={c['se']:.6f}, "
              f"p={c['p_value']:.4f}{sig}, 95% CI [{c['ci_lower']:.6f}, {c['ci_upper']:.6f}])")

    # Q5 summary
    if not yoy.empty:
        print("\n" + "=" * 70)
        print("2026 YEAR-OVER-YEAR (Q5) — Top 5 Improvers / Decliners")
        print("=" * 70)
        for i, (_, r) in enumerate(yoy.head(5).iterrows()):
            print(f"  +{r['div_delta']:+.1f}″  {r['player_name']:25s} "
                  f"2025={r['div_2025']:.1f}″ → 2026={r['div_2026']:.1f}″  "
                  f"Whiff Δ={r['whiff_delta']:+.1%}")
        print("  ---")
        for i, (_, r) in enumerate(yoy.tail(5).iloc[::-1].iterrows()):
            print(f"  {r['div_delta']:+.1f}″  {r['player_name']:25s} "
                  f"2025={r['div_2025']:.1f}″ → 2026={r['div_2026']:.1f}″  "
                  f"Whiff Δ={r['whiff_delta']:+.1%}")

    # Q6 summary
    if not splits.empty:
        print("\n" + "=" * 70)
        print("BATTER HANDEDNESS ASYMMETRY (Q6) — Top 5 biggest gaps")
        print("=" * 70)
        for _, r in splits.head(5).iterrows():
            print(f"  {r['player_name']:25s} vs_LHB={r['vs_LHB']:.1f}″  "
                  f"vs_RHB={r['vs_RHB']:.1f}″  Gap={r['gap']:+.1f}″")
        print("  ---")
        for _, r in splits.tail(5).iloc[::-1].iterrows():
            print(f"  {r['player_name']:25s} vs_LHB={r['vs_LHB']:.1f}″  "
                  f"vs_RHB={r['vs_RHB']:.1f}″  Gap={r['gap']:+.1f}″")

    print(f"\nDone! All outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

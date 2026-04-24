"""ADJUDICATION: rerun zone classifier + counterfactual in absolute plate_z coords.

The dispute: Codex's normalized-coord analysis (plate_z_norm = plate_z / batter_height_proxy)
showed +48pp middle-zone expansion and -56% counterfactual attribution. Claude's review
showed the batter_height_proxy = (sz_top - sz_bot) / 0.265 is broken across seasons because
Statcast changed sz_top/sz_bot storage (per-pitch posture in 2025, deterministic ABS values
in 2026). This script reruns the same analyses in pure absolute coords (plate_x, plate_z)
with no sz_* features, to resolve the dispute.

Outputs:
  research/abs-walk-spike/reviews/adjudication_results_orchestrator.json
  research/abs-walk-spike/reviews/charts/adjudication_zone_delta_absolute.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[3]
DATA_2026 = ROOT / "research" / "abs-walk-spike" / "data" / "statcast_2026_mar27_apr22.parquet"
DATA_2025 = ROOT / "research" / "count-distribution-abs" / "data" / "statcast_2025_mar27_apr14.parquet"
OUT_DIR = Path(__file__).resolve().parents[1] / "reviews"
CHARTS_DIR = OUT_DIR / "charts"
OUT_JSON = OUT_DIR / "adjudication_results_orchestrator.json"

SEED = 20260423
N_BOOT = 100  # bootstrap pairs


def load(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["game_date"].dt.year == year].copy()
    df = df.dropna(subset=["plate_x", "plate_z"])
    if year == 2026:
        # Apples-to-apples: restrict to Mar 27 - Apr 14 to match 2025 same-window parquet
        df = df[df["game_date"] <= pd.Timestamp("2026-04-14")]
    return df


def called_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["description"].isin(["called_strike", "ball"])].copy()


def fit_zone_classifier(called: pd.DataFrame) -> tuple[LogisticRegression, PolynomialFeatures]:
    """Polynomial-2 logistic regression of called_strike on (plate_x, plate_z) absolute."""
    X = called[["plate_x", "plate_z"]].to_numpy()
    y = (called["description"] == "called_strike").astype(int).to_numpy()
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    clf = LogisticRegression(C=0.2, max_iter=500, random_state=SEED)
    clf.fit(X_poly, y)
    return clf, poly


def grid_predict(clf, poly, x_grid, z_grid):
    XX, ZZ = np.meshgrid(x_grid, z_grid)
    XZ = np.column_stack([XX.ravel(), ZZ.ravel()])
    P = clf.predict_proba(poly.transform(XZ))[:, 1]
    return P.reshape(XX.shape), XX, ZZ


def find_largest_significant_region(delta_grid, ci_low_grid, ci_high_grid, x_grid, z_grid, sign="positive", min_pp=3.0):
    """Find largest contiguous block of cells with |delta| >= min_pp AND CI excludes zero.
    Returns dict with x_range, z_range, mean_delta_pp, n_cells.
    """
    if sign == "positive":
        mask = (delta_grid * 100 >= min_pp) & (ci_low_grid > 0)
    else:
        mask = (delta_grid * 100 <= -min_pp) & (ci_high_grid < 0)
    if not mask.any():
        return None
    # Simple bounding-box of the largest connected component
    from scipy.ndimage import label as nd_label

    labeled, n = nd_label(mask)
    if n == 0:
        return None
    sizes = [(i, int((labeled == i).sum())) for i in range(1, n + 1)]
    sizes.sort(key=lambda x: -x[1])
    largest_id, largest_n = sizes[0]
    cells = labeled == largest_id
    rows, cols = np.where(cells)
    z_lo, z_hi = z_grid[rows.min()], z_grid[rows.max()]
    x_lo, x_hi = x_grid[cols.min()], x_grid[cols.max()]
    mean_delta = float(delta_grid[cells].mean() * 100)
    max_delta = float(np.abs(delta_grid[cells]).max() * 100)
    return {
        "x_range": [float(x_lo), float(x_hi)],
        "z_range": [float(z_lo), float(z_hi)],
        "mean_delta_pp": mean_delta,
        "max_abs_delta_pp": max_delta,
        "n_cells": int(largest_n),
    }


def bootstrap_zone_delta(called_25: pd.DataFrame, called_26: pd.DataFrame, x_grid, z_grid, n_boot=N_BOOT):
    """Bootstrap the per-cell P(strike|loc, year=26) - P(strike|loc, year=25) surface."""
    rng = np.random.default_rng(SEED)
    n25 = len(called_25)
    n26 = len(called_26)
    deltas = []
    for b in range(n_boot):
        i25 = rng.integers(0, n25, n25)
        i26 = rng.integers(0, n26, n26)
        clf25, poly25 = fit_zone_classifier(called_25.iloc[i25])
        clf26, poly26 = fit_zone_classifier(called_26.iloc[i26])
        P25, _, _ = grid_predict(clf25, poly25, x_grid, z_grid)
        P26, _, _ = grid_predict(clf26, poly26, x_grid, z_grid)
        deltas.append(P26 - P25)
    deltas = np.stack(deltas)  # (B, nz, nx)
    mean_delta = deltas.mean(axis=0)
    ci_low = np.quantile(deltas, 0.025, axis=0)
    ci_high = np.quantile(deltas, 0.975, axis=0)
    return mean_delta, ci_low, ci_high


def replay_pa(pa_pitches: pd.DataFrame, p_strike_2025: np.ndarray, rng: np.random.Generator) -> str:
    """Replay one PA pitch-by-pitch under a 2025-zone counterfactual.
    For called pitches, resample called_strike vs ball with p_strike_2025 probability.
    For non-called pitches (swing/foul/in-play/HBP/etc), keep actual outcome.
    Terminate at 4-balls = walk, 3-strikes = strikeout, or PA-ending event.
    Returns the new PA outcome.
    """
    balls = 0
    strikes = 0
    for i, row in enumerate(pa_pitches.itertuples()):
        desc = row.description
        events = row.events if pd.notna(row.events) else None

        if desc in ("called_strike", "ball"):
            # Counterfactual: resample under 2025 zone
            is_strike = rng.random() < p_strike_2025[i]
            if is_strike:
                strikes += 1
                if strikes >= 3:
                    return "strikeout"
            else:
                balls += 1
                if balls >= 4:
                    return "walk"
        else:
            # Non-called: actual outcome, but PA may terminate
            if events is not None:
                if events == "walk":
                    return "walk"
                if events == "intent_walk":
                    return "intent_walk"
                if events in ("strikeout", "strikeout_double_play"):
                    return "strikeout"
                # Any other terminating event (in-play out, hit, HBP, etc): non-walk
                return events
            # Foul, foul_tip, etc: increment strike if applicable (not at 2 strikes)
            if desc in ("foul", "foul_tip"):
                if strikes < 2:
                    strikes += 1
            elif desc in ("swinging_strike", "swinging_strike_blocked"):
                strikes += 1
                if strikes >= 3:
                    return "strikeout"
            elif desc in ("blocked_ball",):
                balls += 1
                if balls >= 4:
                    return "walk"
            elif desc in ("hit_by_pitch",):
                return "hit_by_pitch"
            # automatic_ball, automatic_strike: skip (rare ABS artifacts)
    # Fell off end without termination: use actual final event if present
    final_events = pa_pitches.iloc[-1]["events"]
    if pd.notna(final_events):
        return str(final_events)
    return "unknown"


def counterfactual_walk_rate(
    called_25: pd.DataFrame,
    df_26: pd.DataFrame,
    p_strike_grid_2025: callable,
    *,
    first_pitch_only: bool = False,
) -> dict:
    """Apply 2025 zone classifier to 2026 PAs and replay. Returns walk rate counterfactual."""
    rng = np.random.default_rng(SEED + (1 if first_pitch_only else 0))
    if first_pitch_only:
        # Restrict to PAs and use only the first pitch's adjudication for 2025-zone counterfactual
        pa_groups = df_26.groupby(["game_pk", "at_bat_number"])
    else:
        pa_groups = df_26.groupby(["game_pk", "at_bat_number"])

    results = []
    for (game_pk, ab), pa_pitches in pa_groups:
        pa_pitches = pa_pitches.sort_values("pitch_number").reset_index(drop=True)
        if first_pitch_only:
            # only flip the first called pitch; keep actual outcomes for everything else
            first_called_idx = None
            for i, row in pa_pitches.iterrows():
                if row["description"] in ("called_strike", "ball"):
                    first_called_idx = i
                    break
            if first_called_idx is None:
                # No called pitches in PA; use actual outcome
                actual_event = pa_pitches.iloc[-1]["events"]
                if pd.notna(actual_event):
                    results.append(str(actual_event))
                continue
            # Compute p_strike_2025 only for the first called pitch
            row = pa_pitches.iloc[first_called_idx]
            p25 = p_strike_grid_2025(row["plate_x"], row["plate_z"])
            # Replace first-called outcome with counterfactual; replay rest with actual
            modified = pa_pitches.copy()
            new_desc = "called_strike" if rng.random() < p25 else "ball"
            modified.at[first_called_idx, "description"] = new_desc
            # Re-derive PA outcome via replay
            outcome = replay_pa(modified, np.zeros(len(modified)), rng)
            # The replay function treats called pitches normally; but for first_pitch_only,
            # we want all OTHER called pitches to use actual outcomes too. So: precompute p_strike
            # = 1 if actual called_strike, 0 if actual ball, EXCEPT for first which uses p25.
            p_strikes = np.array(
                [
                    (
                        p25
                        if i == first_called_idx
                        else (1.0 if r["description"] == "called_strike" else 0.0)
                        if r["description"] in ("called_strike", "ball")
                        else 0.0
                    )
                    for i, r in modified.iterrows()
                ]
            )
            outcome = replay_pa(pa_pitches, p_strikes, rng)
        else:
            # Compute p_strike_2025 for every called pitch
            p_strikes = np.array(
                [
                    (
                        p_strike_grid_2025(r.plate_x, r.plate_z)
                        if r.description in ("called_strike", "ball")
                        else 0.0
                    )
                    for r in pa_pitches.itertuples()
                ]
            )
            outcome = replay_pa(pa_pitches, p_strikes, rng)
        results.append(outcome)

    walks = sum(1 for r in results if r in ("walk", "intent_walk"))
    n_pa = len(results)
    walk_rate = walks / n_pa
    return {
        "n_pa": n_pa,
        "walks": walks,
        "walk_rate": walk_rate,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== ADJUDICATION: Absolute coordinates rerun ===\n")

    # Load
    df_25 = load(DATA_2025, 2025)
    df_26 = load(DATA_2026, 2026)
    print(f"2025 rows: {len(df_25):,}")
    print(f"2026 rows (Mar 27-Apr 14): {len(df_26):,}")

    # Called pitches only (exclude ABS artifacts)
    called_25 = called_subset(df_25)
    called_26 = called_subset(df_26)
    print(f"2025 called pitches: {len(called_25):,}")
    print(f"2026 called pitches: {len(called_26):,}")

    # Grid for delta surface (absolute coords)
    x_grid = np.linspace(-1.5, 1.5, 60)  # 0.05 ft bins
    z_grid = np.linspace(0.5, 4.5, 80)  # 0.05 ft bins

    # Bootstrap zone delta
    print("\nBootstrapping zone delta surface (100 reps)...")
    mean_delta, ci_low, ci_high = bootstrap_zone_delta(called_25, called_26, x_grid, z_grid, n_boot=N_BOOT)
    print("  done.")

    # Find largest regions
    pos_region = find_largest_significant_region(mean_delta, ci_low, ci_high, x_grid, z_grid, sign="positive")
    neg_region = find_largest_significant_region(mean_delta, ci_low, ci_high, x_grid, z_grid, sign="negative")
    print(f"\nLargest CI-significant POSITIVE region: {pos_region}")
    print(f"Largest CI-significant NEGATIVE region: {neg_region}")

    # Headline single-cell maxima (analog to Codex's reporting)
    max_pos_idx = np.unravel_index(np.argmax(mean_delta), mean_delta.shape)
    max_neg_idx = np.unravel_index(np.argmin(mean_delta), mean_delta.shape)
    headline_max_positive = {
        "x": float(x_grid[max_pos_idx[1]]),
        "z": float(z_grid[max_pos_idx[0]]),
        "delta_pp": float(mean_delta[max_pos_idx] * 100),
    }
    headline_max_negative = {
        "x": float(x_grid[max_neg_idx[1]]),
        "z": float(z_grid[max_neg_idx[0]]),
        "delta_pp": float(mean_delta[max_neg_idx] * 100),
    }
    print(f"Single-cell max positive: {headline_max_positive}")
    print(f"Single-cell max negative: {headline_max_negative}")

    # Save heatmap
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(
        mean_delta * 100,
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), z_grid.min(), z_grid.max()],
        cmap="RdBu_r",
        vmin=-30,
        vmax=30,
        aspect="auto",
    )
    # Hash CI-non-significant cells
    nonsig = (ci_low <= 0) & (ci_high >= 0)
    XX, ZZ = np.meshgrid(x_grid, z_grid)
    ax.contourf(
        XX,
        ZZ,
        nonsig.astype(int),
        levels=[0.5, 1.5],
        hatches=["//"],
        colors="none",
        alpha=0,
    )
    # Rule-book / ABS zone overlay
    plate_half = 17.0 / 2 / 12
    ax.axvline(-plate_half, color="black", lw=0.8, alpha=0.5)
    ax.axvline(plate_half, color="black", lw=0.8, alpha=0.5)
    ax.axhline(1.62, color="black", lw=0.8, alpha=0.5, label="ABS bot 1.62 ft")
    ax.axhline(3.21, color="black", lw=0.8, alpha=0.5, label="ABS top 3.21 ft")
    ax.set_xlabel("plate_x (ft, batter view)")
    ax.set_ylabel("plate_z (ft, absolute)")
    ax.set_title("Called-strike rate Δ (2026 − 2025), absolute coords\n+ CI-non-significant hatched")
    cbar = plt.colorbar(im, ax=ax, label="Δ (pp)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "adjudication_zone_delta_absolute.png", dpi=120)
    plt.close()

    # Counterfactual: train ONE 2025 zone classifier (no bootstrap), apply to 2026
    print("\n=== Counterfactual ===")
    clf25, poly25 = fit_zone_classifier(called_25)

    def p_strike_2025(plate_x, plate_z):
        v = poly25.transform(np.array([[plate_x, plate_z]]))
        return float(clf25.predict_proba(v)[0, 1])

    # Need pitch_number ordering for replay
    if "pitch_number" not in df_26.columns:
        df_26 = df_26.assign(pitch_number=df_26.groupby(["game_pk", "at_bat_number"]).cumcount() + 1)
    if "at_bat_number" not in df_26.columns:
        print("ERROR: at_bat_number missing", flush=True)
        return

    print("Running ALL-PITCHES counterfactual...")
    cf_all = counterfactual_walk_rate(called_25, df_26, p_strike_2025, first_pitch_only=False)

    print("Running FIRST-PITCH-ONLY counterfactual...")
    cf_first = counterfactual_walk_rate(called_25, df_26, p_strike_2025, first_pitch_only=True)

    # Actual rates for comparison
    pa_26 = df_26[df_26["events"].notna() & (df_26["events"] != "")]
    actual_2026_walk_rate = (pa_26["events"].isin(["walk", "intent_walk"])).sum() / len(pa_26)
    pa_25 = df_25[df_25["events"].notna() & (df_25["events"] != "")]
    actual_2025_walk_rate = (pa_25["events"].isin(["walk", "intent_walk"])).sum() / len(pa_25)
    yoy_delta = actual_2026_walk_rate - actual_2025_walk_rate

    # Attribution: (actual - counterfactual) / yoy_delta * 100
    # If counterfactual_rate < actual_2026_rate: zone change INCREASES walks (positive attribution)
    # If counterfactual_rate > actual_2026_rate: zone change DECREASES walks (negative attribution)
    attribution_all = (actual_2026_walk_rate - cf_all["walk_rate"]) / yoy_delta * 100 if yoy_delta != 0 else None
    attribution_first = (actual_2026_walk_rate - cf_first["walk_rate"]) / yoy_delta * 100 if yoy_delta != 0 else None

    print(f"\nActual 2025 walk rate (incl IBB): {actual_2025_walk_rate*100:.3f}%")
    print(f"Actual 2026 walk rate (incl IBB): {actual_2026_walk_rate*100:.3f}%")
    print(f"YoY delta: {yoy_delta*100:+.3f}pp")
    print(f"\nCounterfactual (ALL called pitches under 2025 zone):")
    print(f"  walk rate: {cf_all['walk_rate']*100:.3f}% (n_pa={cf_all['n_pa']:,})")
    print(f"  attribution: {attribution_all:+.1f}%")
    print(f"\nCounterfactual (FIRST called pitch only under 2025 zone):")
    print(f"  walk rate: {cf_first['walk_rate']*100:.3f}% (n_pa={cf_first['n_pa']:,})")
    print(f"  attribution: {attribution_first:+.1f}%")

    # Year classifier in absolute coords (no sz_*) — quick AUC sanity check
    print("\n=== Year-classifier in absolute coords (no sz_*, no batter_height_proxy) ===")
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.model_selection import cross_val_score, GroupKFold

    yc_features = pd.concat([
        called_25[["plate_x", "plate_z"]].assign(year=2025),
        called_26[["plate_x", "plate_z"]].assign(year=2026),
    ], ignore_index=True)
    yc_features = pd.concat([
        yc_features,
        pd.concat([called_25[["game_pk"]], called_26[["game_pk"]]], ignore_index=True),
    ], axis=1)
    X_yc = yc_features[["plate_x", "plate_z"]].to_numpy()
    y_yc = (yc_features["year"] == 2026).astype(int).to_numpy()
    groups = yc_features["game_pk"].to_numpy()

    poly_yc = PolynomialFeatures(degree=2, include_bias=False)
    X_yc_poly = poly_yc.fit_transform(X_yc)
    yc_model = LR(C=0.2, max_iter=500, random_state=SEED)
    gkf = GroupKFold(n_splits=5)
    aucs = cross_val_score(yc_model, X_yc_poly, y_yc, cv=gkf.split(X_yc_poly, y_yc, groups), scoring="roc_auc")
    print(f"  Year-classifier AUC (poly-2, location-only): {aucs.mean():.4f} ± {aucs.std():.4f}")

    # Save findings
    out = {
        "method": "absolute plate_z coords, polynomial-2 logistic, no sz_top/sz_bot/batter_height features",
        "n_25_called": int(len(called_25)),
        "n_26_called": int(len(called_26)),
        "n_25_pa": int(len(pa_25)),
        "n_26_pa": int(len(pa_26)),
        "actual_2025_walk_rate_incl_ibb": float(actual_2025_walk_rate),
        "actual_2026_walk_rate_incl_ibb": float(actual_2026_walk_rate),
        "yoy_walk_rate_delta_pp": float(yoy_delta * 100),
        "largest_significant_positive_region_absolute": pos_region,
        "largest_significant_negative_region_absolute": neg_region,
        "single_cell_max_positive_absolute": headline_max_positive,
        "single_cell_max_negative_absolute": headline_max_negative,
        "counterfactual_all_called_pitches_walk_rate": float(cf_all["walk_rate"]),
        "counterfactual_all_called_pitches_attribution_pct": float(attribution_all) if attribution_all is not None else None,
        "counterfactual_first_pitch_only_walk_rate": float(cf_first["walk_rate"]),
        "counterfactual_first_pitch_only_attribution_pct": float(attribution_first) if attribution_first is not None else None,
        "year_classifier_location_only_auc_mean": float(aucs.mean()),
        "year_classifier_location_only_auc_std": float(aucs.std()),
        "n_bootstrap_zone_delta": N_BOOT,
        "seed": SEED,
    }
    with OUT_JSON.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {OUT_JSON}")


if __name__ == "__main__":
    main()

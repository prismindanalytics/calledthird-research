"""Reconciliation script: do Codex's normalized-coords zone findings line up
with Claude's absolute-coords findings?

In particular:
  - Codex headline: dominant zone change is +48.38pp middle-full-width
    expansion at z_norm in [0.39, 0.66], and the counterfactual says
    applying the 2025 zone to 2026 RAISES walks (-56% attribution).
  - Claude headline: top-edge shrinkage at z~3.2 ft, bottom-edge expansion
    at z~1.5-1.7 ft. Walks rise via upstream traffic.

What I want to check:
  1. Convert z_norm bands back to absolute z and see where Codex's "expansion"
     band maps in feet for a median hitter.
  2. Compute the called-strike rate (per pitch in cell) directly in
     normalized coords on the SAME bin grid as my absolute-coords analysis,
     to see if the top-edge shrinkage from my map is hidden by Codex's
     polynomial-logistic smoother.
  3. Check the 2D shape of the bin-level delta in z_norm space.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_2026 = ROOT / "data" / "statcast_2026_mar27_apr22.parquet"
DATA_2025_PRIMARY = ROOT.parent / "count-distribution-abs" / "data" / "statcast_2025_mar27_apr14.parquet"

# Codex constants
ABS_ZONE_BOTTOM = 0.27
ABS_ZONE_TOP = 0.535
ABS_VERTICAL_SHARE = ABS_ZONE_TOP - ABS_ZONE_BOTTOM


def load(path: Path, season: int, end_date: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date"] <= pd.Timestamp(end_date)].copy()
    df["season"] = season
    df["zone_height"] = df["sz_top"] - df["sz_bot"]
    df["batter_height_proxy"] = df["zone_height"] / ABS_VERTICAL_SHARE
    df["plate_z_norm"] = df["plate_z"] / df["batter_height_proxy"]
    df = df[df["description"].isin(["called_strike", "ball"])].copy()
    df = df.dropna(subset=["plate_x", "plate_z", "plate_z_norm", "sz_top", "sz_bot"])
    df["is_called_strike"] = (df["description"] == "called_strike").astype(int)
    return df


def main() -> None:
    df_2025 = load(DATA_2025_PRIMARY, 2025, "2025-04-14")
    df_2026 = load(DATA_2026, 2026, "2026-04-14")

    print("=" * 70)
    print("Q1. Codex z_norm regions translated to absolute feet (median hitter)")
    print("=" * 70)
    median_h_2025 = df_2025["batter_height_proxy"].median()
    median_h_2026 = df_2026["batter_height_proxy"].median()
    print(f"median batter_height_proxy 2025 = {median_h_2025:.3f} ft")
    print(f"median batter_height_proxy 2026 = {median_h_2026:.3f} ft")
    pooled_h = (median_h_2025 + median_h_2026) / 2
    print(f"pooled = {pooled_h:.3f} ft")

    # Codex "expansion" region: z_norm in [0.39, 0.66]
    z_lo_exp_ft = 0.39 * pooled_h
    z_hi_exp_ft = 0.66 * pooled_h
    print(f"\nCodex expansion region (z_norm 0.39-0.66) -> absolute z = "
          f"[{z_lo_exp_ft:.2f}, {z_hi_exp_ft:.2f}] ft for median hitter")

    # Codex "shrink" region: z_norm in [0.137, 0.397]
    z_lo_shr_ft = 0.137 * pooled_h
    z_hi_shr_ft = 0.397 * pooled_h
    print(f"Codex shrink region (z_norm 0.137-0.397) -> absolute z = "
          f"[{z_lo_shr_ft:.2f}, {z_hi_shr_ft:.2f}] ft for median hitter")

    # Claude top shrink region: ~3.1-3.55 ft (above zone)
    z_top_norm = 3.2 / pooled_h
    z_bot_norm = 1.6 / pooled_h
    print(f"\nClaude top shrink (z=3.2 ft) -> z_norm = {z_top_norm:.3f}")
    print(f"Claude bottom expansion (z=1.6 ft) -> z_norm = {z_bot_norm:.3f}")

    print("\n" + "=" * 70)
    print("Q2. Bin-level called-strike-rate map in z_norm space (2026 - 2025)")
    print("=" * 70)
    # Use Codex's grid bounds: x in [-1.6, 1.6], z_norm in [0.12, 0.68]
    # but with FINE bins so we can see edge structure.
    nx, nz = 17, 28  # ~0.19 x 0.02 cells
    x_edges = np.linspace(-1.6, 1.6, nx + 1)
    z_edges = np.linspace(0.12, 0.68, nz + 1)

    def csr_grid(df):
        cs = df[df["is_called_strike"] == 1]
        n_cs, _, _ = np.histogram2d(cs["plate_x"], cs["plate_z_norm"],
                                     bins=[x_edges, z_edges])
        n_all, _, _ = np.histogram2d(df["plate_x"], df["plate_z_norm"],
                                      bins=[x_edges, z_edges])
        return n_cs, n_all

    cs_25, n_25 = csr_grid(df_2025)
    cs_26, n_26 = csr_grid(df_2026)

    csr_25 = np.where(n_25 >= 25, cs_25 / n_25, np.nan)
    csr_26 = np.where(n_26 >= 25, cs_26 / n_26, np.nan)
    delta = csr_26 - csr_25

    print(f"\nz_norm bin (low-edge) -> mean delta across x (only |delta|>0.05 cells)")
    print(f"{'z_lo':>8} {'mean_d_pp':>10} {'frac_neg':>10} {'frac_pos':>10} {'n_cells':>8}")
    for j in range(nz):
        col = delta[:, j]
        mask = np.isfinite(col) & (np.abs(col) >= 0.05)
        if mask.sum() < 2:
            continue
        col_vals = col[mask]
        print(f"{z_edges[j]:8.3f} {np.nanmean(col)*100:10.2f}"
              f" {(col_vals<0).mean():10.2f} {(col_vals>0).mean():10.2f}"
              f" {int(np.isfinite(col).sum()):8d}")

    print("\nTop-edge band (z_norm in [0.50, 0.60]) cells:")
    for j in range(nz):
        zlo, zhi = z_edges[j], z_edges[j+1]
        if not (0.50 <= zlo <= 0.60):
            continue
        for i in range(nx):
            xlo, xhi = x_edges[i], x_edges[i+1]
            if abs((xlo+xhi)/2) > 1.0:
                continue  # only inner-plate
            d = delta[i, j]
            if np.isnan(d):
                continue
            print(f"  x in [{xlo:5.2f},{xhi:5.2f}] z_norm in [{zlo:.3f},{zhi:.3f}]"
                  f"  delta_pp = {d*100:+6.2f}  n25={int(n_25[i,j])} n26={int(n_26[i,j])}")

    print("\nBottom-edge band (z_norm in [0.20, 0.30]) cells:")
    for j in range(nz):
        zlo, zhi = z_edges[j], z_edges[j+1]
        if not (0.20 <= zlo <= 0.30):
            continue
        for i in range(nx):
            xlo, xhi = x_edges[i], x_edges[i+1]
            if abs((xlo+xhi)/2) > 1.0:
                continue
            d = delta[i, j]
            if np.isnan(d):
                continue
            print(f"  x in [{xlo:5.2f},{xhi:5.2f}] z_norm in [{zlo:.3f},{zhi:.3f}]"
                  f"  delta_pp = {d*100:+6.2f}  n25={int(n_25[i,j])} n26={int(n_26[i,j])}")

    # Q3 - reconcile with Codex's "+48 pp middle expansion" claim. His region is
    # z_norm in [0.39, 0.66] (a HUGE band that covers the entire upper ~half of
    # the called zone). What does the bin-level mean look like there?
    print("\n" + "=" * 70)
    print("Q3. Mean bin-level delta in Codex's claimed expansion region")
    print("    (z_norm 0.39-0.66, x in [-1.05, 1.12], inner-plate)")
    print("=" * 70)
    cells = []
    for j in range(nz):
        zlo, zhi = z_edges[j], z_edges[j+1]
        if not (0.39 <= zlo <= 0.66):
            continue
        for i in range(nx):
            xlo, xhi = x_edges[i], x_edges[i+1]
            if not (-1.05 <= (xlo+xhi)/2 <= 1.12):
                continue
            d = delta[i, j]
            if np.isnan(d):
                continue
            cells.append((d, n_25[i, j], n_26[i, j]))
    if cells:
        ds = np.array([c[0] for c in cells])
        n25s = np.array([c[1] for c in cells])
        n26s = np.array([c[2] for c in cells])
        wmean = np.average(ds, weights=(n25s+n26s)/2)
        print(f"  {len(cells)} cells; unweighted mean delta = {ds.mean()*100:+.2f} pp")
        print(f"  weighted mean delta = {wmean*100:+.2f} pp")
        print(f"  max delta = {ds.max()*100:+.2f} pp; min = {ds.min()*100:+.2f} pp")
        print(f"  n cells with delta > +5pp: {(ds>0.05).sum()}")
        print(f"  n cells with delta < -5pp: {(ds<-0.05).sum()}")
        # Show all the cells
        print("\n  All cells in Codex region:")
        for j in range(nz):
            zlo, zhi = z_edges[j], z_edges[j+1]
            if not (0.39 <= zlo <= 0.66):
                continue
            for i in range(nx):
                xlo, xhi = x_edges[i], x_edges[i+1]
                if not (-1.05 <= (xlo+xhi)/2 <= 1.12):
                    continue
                d = delta[i, j]
                if np.isnan(d):
                    continue
                print(f"    x[{xlo:5.2f},{xhi:5.2f}] z_norm[{zlo:.3f},{zhi:.3f}]"
                      f" d={d*100:+6.2f}pp  CSR_25={csr_25[i,j]*100:5.1f} CSR_26={csr_26[i,j]*100:5.1f}"
                      f" n25={int(n_25[i,j])} n26={int(n_26[i,j])}")

    print("\n" + "=" * 70)
    print("Q4. The +48 pp peak — does it correspond to a HIGH-population cell")
    print("    or to a low-pitch corner that the polynomial extrapolates wildly?")
    print("=" * 70)
    # Replicate Codex's polynomial logistic
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    def codex_zone_model(seed):
        return Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=0.2, penalty="l2", solver="lbfgs",
                                          max_iter=4000, random_state=seed)),
        ])
    m25 = codex_zone_model(20260423).fit(
        df_2025[["plate_x", "plate_z_norm"]].values,
        df_2025["is_called_strike"].values,
    )
    m26 = codex_zone_model(20260424).fit(
        df_2026[["plate_x", "plate_z_norm"]].values,
        df_2026["is_called_strike"].values,
    )
    # Predict on Codex's grid
    xg = np.linspace(-1.6, 1.6, 100)
    zg = np.linspace(0.12, 0.68, 100)
    xx, zz = np.meshgrid(xg, zg)
    grid = np.column_stack([xx.ravel(), zz.ravel()])
    p25 = m25.predict_proba(grid)[:, 1].reshape(100, 100)
    p26 = m26.predict_proba(grid)[:, 1].reshape(100, 100)
    d = p26 - p25
    # Find peak cell
    iz, ix = np.unravel_index(np.argmax(d), d.shape)
    print(f"Peak +delta cell: x={xg[ix]:.3f}, z_norm={zg[iz]:.3f}, "
          f"p25={p25[iz,ix]*100:.2f}%, p26={p26[iz,ix]*100:.2f}%, "
          f"delta={d[iz,ix]*100:+.2f}pp")
    # How many actual pitches near this cell?
    near = ((np.abs(df_2025["plate_x"] - xg[ix]) < 0.1) &
            (np.abs(df_2025["plate_z_norm"] - zg[iz]) < 0.02)).sum()
    near_26 = ((np.abs(df_2026["plate_x"] - xg[ix]) < 0.1) &
               (np.abs(df_2026["plate_z_norm"] - zg[iz]) < 0.02)).sum()
    print(f"  Empirical pitches within +/- (0.1, 0.02) of peak: 2025={near}, 2026={near_26}")
    iz_neg, ix_neg = np.unravel_index(np.argmin(d), d.shape)
    print(f"\nPeak -delta cell: x={xg[ix_neg]:.3f}, z_norm={zg[iz_neg]:.3f}, "
          f"p25={p25[iz_neg,ix_neg]*100:.2f}%, p26={p26[iz_neg,ix_neg]*100:.2f}%, "
          f"delta={d[iz_neg,ix_neg]*100:+.2f}pp")

    # Now COUNTERFACTUAL sanity:
    # if I apply the 2025 model to 2026 called pitches, what called-strike
    # rate do I get vs the actual 2026 rate?
    feats_26 = df_2026[["plate_x", "plate_z_norm"]].values
    p25_on_26 = m25.predict_proba(feats_26)[:, 1]
    p26_on_26 = m26.predict_proba(feats_26)[:, 1]
    actual_csr_26 = df_2026["is_called_strike"].mean()
    print(f"\nSanity on 2026 called pitches:")
    print(f"  actual CSR on called pitches: {actual_csr_26*100:.2f}%")
    print(f"  model-2025 mean prob on 2026 pitches: {p25_on_26.mean()*100:.2f}%")
    print(f"  model-2026 mean prob on 2026 pitches: {p26_on_26.mean()*100:.2f}%")
    print(f"  -> if 2025 zone applied to 2026 pitches CSR is HIGHER, then \n"
          f"     2025 zone called MORE strikes => fewer balls => fewer walks.")
    print(f"  -> If counterfactual says 'walks RISE', that conflicts with \n"
          f"     this naive intuition and means the simulator picks up tail behaviour")
    print(f"     (e.g. 2-strike conversions) that flip the sign.")


if __name__ == "__main__":
    main()

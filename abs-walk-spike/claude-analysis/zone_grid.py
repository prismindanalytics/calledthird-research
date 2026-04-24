"""H1: 2D grid binning of called-strike rate over the plate, with bootstrap CIs.

Primary geometry: absolute (plate_x, plate_z) in feet, bin width ~0.1 ft.
Supplementary: height-normalized (plate_x, plate_z_norm) bin width 0.05 ft x 0.05 normalized.

Restriction:
- 2025 Mar 27 - Apr 14 vs 2026 Mar 27 - Apr 14 (apples-to-apples primary).
- Called pitches only (description in {'called_strike','ball'} and not auto_*).

Output:
- 2D arrays of cs_rate per season per cell (with cell counts)
- Delta = 2026 - 2025
- Bootstrap 95% CI per cell (Wilson-equivalent via 1000 BS resamples)
- Significance mask (cells whose 95% CI excludes 0)
- Largest contiguous shrinkage region in the rule-book zone box.

Saves:
- charts/heatmap_zone_delta.png (absolute coords)
- charts/heatmap_zone_delta_norm.png (height-normalized)
- charts/heatmap_2025_callstrike.png and heatmap_2026_callstrike.png
- A small artifacts dict pickled to artifacts/zone_grid.npz
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Rectangle

from common import (
    PROJECT_ROOT,
    called_pitches_subset,
    load_2025_extended_to_apr22,
    load_2025_samewin,
    load_2026,
    restrict_to_primary_window,
    safe_plate_z_norm,
)

OUT_DIR = PROJECT_ROOT / "claude-analysis"
CHART_DIR = OUT_DIR / "charts"
ART_DIR = OUT_DIR / "artifacts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

# Grid configuration
X_LO, X_HI, X_BIN = -1.7, 1.7, 0.10  # 34 bins of 0.1 ft, plate is ~17 inches = 1.42 ft
Z_LO, Z_HI, Z_BIN = 1.0, 4.5, 0.10
ZONE_X = (-0.83, 0.83)               # rule-book half-plate width = 17/2 in -> 0.708ft, expanded for ball radius
PLATE_X = (-0.71, 0.71)              # 17 in / 12
RULEBOOK_Z = (1.5, 3.5)              # rough; will overlay with median sz_bot/sz_top too

NORM_BIN_X = 0.10
NORM_BIN_Z = 0.05


def grid_indices(values: np.ndarray, lo: float, hi: float, bin_w: float) -> np.ndarray:
    idx = np.floor((values - lo) / bin_w).astype(int)
    n = int(np.ceil((hi - lo) / bin_w))
    idx[(idx < 0) | (idx >= n)] = -1
    return idx


def aggregate_grid(
    df: pd.DataFrame,
    x_col: str,
    z_col: str,
    x_lo: float, x_hi: float, x_bin: float,
    z_lo: float, z_hi: float, z_bin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (n_pitches, n_strikes) arrays of shape (n_z, n_x)."""
    nx = int(np.ceil((x_hi - x_lo) / x_bin))
    nz = int(np.ceil((z_hi - z_lo) / z_bin))
    xi = grid_indices(df[x_col].to_numpy(dtype=float), x_lo, x_hi, x_bin)
    zi = grid_indices(df[z_col].to_numpy(dtype=float), z_lo, z_hi, z_bin)
    is_strike = df["is_called_strike"].to_numpy(dtype=int)
    n = np.zeros((nz, nx), dtype=np.int64)
    s = np.zeros((nz, nx), dtype=np.int64)
    valid = (xi >= 0) & (zi >= 0)
    np.add.at(n, (zi[valid], xi[valid]), 1)
    np.add.at(s, (zi[valid], xi[valid]), is_strike[valid])
    return n, s


def bootstrap_delta_ci(
    n25: np.ndarray, s25: np.ndarray, n26: np.ndarray, s26: np.ndarray,
    min_pitches: int = 30, n_boot: int = 1000, seed: int = 19841984,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each cell, bootstrap 95% CI for (p26 - p25). Cells with either side <min mark NaN.

    Returns (delta, ci_low, ci_high) arrays.
    """
    rng = np.random.default_rng(seed)
    nz, nx = n25.shape
    delta = np.full((nz, nx), np.nan)
    lo = np.full((nz, nx), np.nan)
    hi = np.full((nz, nx), np.nan)
    p25 = np.where(n25 > 0, s25 / np.maximum(n25, 1), np.nan)
    p26 = np.where(n26 > 0, s26 / np.maximum(n26, 1), np.nan)
    for zi in range(nz):
        for xi in range(nx):
            a = int(n25[zi, xi]); b = int(n26[zi, xi])
            if a < min_pitches or b < min_pitches:
                continue
            ps25 = s25[zi, xi] / a
            ps26 = s26[zi, xi] / b
            d_obs = ps26 - ps25
            # Beta-binomial-style nonparametric BS by drawing binomial(n,p)
            bs25 = rng.binomial(a, ps25, size=n_boot) / a
            bs26 = rng.binomial(b, ps26, size=n_boot) / b
            bs = bs26 - bs25
            delta[zi, xi] = d_obs
            lo[zi, xi] = np.quantile(bs, 0.025)
            hi[zi, xi] = np.quantile(bs, 0.975)
    return delta, lo, hi


def lowess_smooth_2d(delta: np.ndarray, sigma_bins: float = 1.2) -> np.ndarray:
    """Cheap 2D Gaussian smooth as a thin-plate-spline stand-in.

    Mask NaNs by zero-fill weighted; preserves NaN cells in output.
    """
    from scipy.ndimage import gaussian_filter

    mask = np.isfinite(delta).astype(float)
    d = np.where(mask > 0, delta, 0.0)
    num = gaussian_filter(d * mask, sigma=sigma_bins, mode="nearest")
    den = gaussian_filter(mask, sigma=sigma_bins, mode="nearest")
    out = np.where(den > 0.05, num / np.maximum(den, 1e-9), np.nan)
    out[mask == 0] = np.nan
    return out


def plot_heatmap(
    delta: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    x_lo: float, x_hi: float, x_bin: float,
    z_lo: float, z_hi: float, z_bin: float,
    title: str, savepath: Path,
    overlay_zone: bool = True,
    overlay_norm: bool = False,
    median_sz_top: float | None = None,
    median_sz_bot: float | None = None,
):
    fig, ax = plt.subplots(figsize=(7.4, 8.0), dpi=140)
    vmax = float(np.nanmax(np.abs(delta))) if np.isfinite(np.nanmax(np.abs(delta))) else 0.10
    vmax = max(vmax, 0.05)
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    extent = [x_lo, x_hi, z_lo, z_hi]
    im = ax.imshow(
        delta, origin="lower", extent=extent, aspect="auto",
        cmap="RdBu_r", norm=norm, interpolation="nearest",
    )
    # Hatch insignificant cells (CI crosses zero)
    sig_mask = np.isfinite(delta) & ((lo > 0) | (hi < 0))
    insig = np.isfinite(delta) & ~sig_mask
    nz, nx = delta.shape
    for zi in range(nz):
        for xi in range(nx):
            if insig[zi, xi]:
                rect = Rectangle(
                    (x_lo + xi * x_bin, z_lo + zi * z_bin), x_bin, z_bin,
                    fill=False, hatch="////", edgecolor="0.3", linewidth=0.0,
                )
                ax.add_patch(rect)
    if overlay_zone and not overlay_norm:
        # Plate width and rule-book vertical zone
        ax.add_patch(Rectangle((PLATE_X[0], RULEBOOK_Z[0]), PLATE_X[1] - PLATE_X[0],
                                RULEBOOK_Z[1] - RULEBOOK_Z[0],
                                fill=False, edgecolor="black", linewidth=1.5))
        if median_sz_top is not None and median_sz_bot is not None:
            ax.axhline(median_sz_top, color="0.2", linestyle=":", linewidth=1.0,
                       label=f"median sz_top={median_sz_top:.2f}")
            ax.axhline(median_sz_bot, color="0.2", linestyle=":", linewidth=1.0,
                       label=f"median sz_bot={median_sz_bot:.2f}")
            ax.legend(loc="upper right", fontsize=8)
    if overlay_norm:
        ax.add_patch(Rectangle((PLATE_X[0], 0.0), PLATE_X[1] - PLATE_X[0], 1.0,
                                fill=False, edgecolor="black", linewidth=1.5))
        # ABS vertical band 27% to 53.5% of standing height -> in human-zone-normalized space
        # the ABS bottom and top differ from human; we mark the human zone (0..1) as the box
        # and also draw 0.27 and 0.535 reference lines inside it for context
        ax.axhline(0.27, color="darkgreen", linestyle="--", linewidth=1.0,
                   label="ABS bottom (27% standing height)")
        ax.axhline(0.535, color="darkgreen", linestyle="--", linewidth=1.0,
                   label="ABS top (53.5% standing height)")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("plate_x (ft, catcher view)")
    ax.set_ylabel("plate_z (ft)" if not overlay_norm else "plate_z_norm")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(z_lo, z_hi)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("called-strike rate delta (2026 - 2025), pp/100")
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)


def plot_single_rate(
    rate: np.ndarray,
    x_lo: float, x_hi: float, z_lo: float, z_hi: float,
    title: str, savepath: Path, vmax: float = 1.0,
    median_sz_top: float | None = None, median_sz_bot: float | None = None,
):
    fig, ax = plt.subplots(figsize=(7.4, 8.0), dpi=140)
    extent = [x_lo, x_hi, z_lo, z_hi]
    im = ax.imshow(rate, origin="lower", extent=extent, aspect="auto",
                   cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.add_patch(Rectangle((PLATE_X[0], RULEBOOK_Z[0]), PLATE_X[1] - PLATE_X[0],
                            RULEBOOK_Z[1] - RULEBOOK_Z[0],
                            fill=False, edgecolor="white", linewidth=1.5))
    if median_sz_top is not None and median_sz_bot is not None:
        ax.axhline(median_sz_top, color="white", linestyle=":", linewidth=1.0)
        ax.axhline(median_sz_bot, color="white", linestyle=":", linewidth=1.0)
    ax.set_xlabel("plate_x (ft)")
    ax.set_ylabel("plate_z (ft)")
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("called-strike rate")
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)


def find_largest_significant_region(
    delta: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    sign: int,
    x_lo: float, x_hi: float, x_bin: float,
    z_lo: float, z_hi: float, z_bin: float,
    min_delta_pp: float = 0.03,
    pitches: np.ndarray | None = None,
) -> Dict:
    """Find largest contiguous (4-neighbor) region whose CI excludes 0 in given sign."""
    from scipy.ndimage import label as cc_label

    if sign > 0:
        keep = np.isfinite(delta) & (lo > 0) & (delta >= min_delta_pp)
    else:
        keep = np.isfinite(delta) & (hi < 0) & (delta <= -min_delta_pp)
    if not keep.any():
        return {"sign": sign, "found": False}
    labeled, ncomp = cc_label(keep)
    best = None
    for ci in range(1, ncomp + 1):
        m = labeled == ci
        size = int(m.sum())
        zix, xix = np.where(m)
        x_range = (x_lo + xix.min() * x_bin, x_lo + (xix.max() + 1) * x_bin)
        z_range = (z_lo + zix.min() * z_bin, z_lo + (zix.max() + 1) * z_bin)
        # Use weighted mean of delta if pitches given, else uniform
        if pitches is not None:
            w = pitches[m]
            d_mean = float(np.average(delta[m], weights=np.maximum(w, 1)))
            n_pitches = int(w.sum())
        else:
            d_mean = float(delta[m].mean())
            n_pitches = None
        ci_lo_mean = float(np.average(lo[m]))
        ci_hi_mean = float(np.average(hi[m]))
        info = {
            "size_cells": size,
            "area_sqft": size * x_bin * z_bin,
            "x_range": x_range,
            "z_range": z_range,
            "delta_mean_pp": d_mean,
            "ci_low_mean": ci_lo_mean,
            "ci_high_mean": ci_hi_mean,
            "n_pitches_in_region": n_pitches,
        }
        if best is None or size > best["size_cells"]:
            best = info
    return {"sign": sign, "found": True, **best}


def run() -> Dict:
    print("[zone_grid] loading 2025 + 2026")
    df25 = load_2025_samewin()
    df26 = load_2026()
    df26_pri = restrict_to_primary_window(df26)

    cp25 = called_pitches_subset(df25)
    cp26 = called_pitches_subset(df26_pri)
    print(f"[zone_grid] 2025 called pitches (Mar27-Apr14): {len(cp25):,}")
    print(f"[zone_grid] 2026 called pitches (Mar27-Apr14): {len(cp26):,}")

    # Median sz_top and sz_bot (for visual reference)
    med_top = float(pd.concat([cp25["sz_top"], cp26["sz_top"]], ignore_index=True).median())
    med_bot = float(pd.concat([cp25["sz_bot"], cp26["sz_bot"]], ignore_index=True).median())

    # Absolute coordinate grid
    n25, s25 = aggregate_grid(cp25, "plate_x", "plate_z", X_LO, X_HI, X_BIN, Z_LO, Z_HI, Z_BIN)
    n26, s26 = aggregate_grid(cp26, "plate_x", "plate_z", X_LO, X_HI, X_BIN, Z_LO, Z_HI, Z_BIN)
    print(f"[zone_grid] grid 2025 cells with >=30 pitches: {(n25 >= 30).sum()}/{n25.size}")
    print(f"[zone_grid] grid 2026 cells with >=30 pitches: {(n26 >= 30).sum()}/{n26.size}")

    delta, lo, hi = bootstrap_delta_ci(n25, s25, n26, s26)
    p25 = np.where(n25 > 0, s25 / np.maximum(n25, 1), np.nan)
    p26 = np.where(n26 > 0, s26 / np.maximum(n26, 1), np.nan)

    # Smoothed map for the visual companion
    smooth = lowess_smooth_2d(delta, sigma_bins=1.2)

    # Save raw delta plot (primary)
    plot_heatmap(delta, lo, hi, X_LO, X_HI, X_BIN, Z_LO, Z_HI, Z_BIN,
                 title="Called-strike rate delta — 2026 minus 2025 (Mar 27-Apr 14)\nhatch = 95% bootstrap CI crosses 0",
                 savepath=CHART_DIR / "heatmap_zone_delta.png",
                 overlay_zone=True, median_sz_top=med_top, median_sz_bot=med_bot)
    plot_heatmap(smooth, lo, hi, X_LO, X_HI, X_BIN, Z_LO, Z_HI, Z_BIN,
                 title="Smoothed called-strike rate delta (Gaussian, sigma=1.2 bins)",
                 savepath=CHART_DIR / "heatmap_zone_delta_smoothed.png",
                 overlay_zone=True, median_sz_top=med_top, median_sz_bot=med_bot)
    plot_single_rate(p25, X_LO, X_HI, Z_LO, Z_HI,
                     title="2025 called-strike rate (Mar 27 - Apr 14)",
                     savepath=CHART_DIR / "heatmap_2025_callstrike.png",
                     median_sz_top=med_top, median_sz_bot=med_bot)
    plot_single_rate(p26, X_LO, X_HI, Z_LO, Z_HI,
                     title="2026 called-strike rate (Mar 27 - Apr 14)",
                     savepath=CHART_DIR / "heatmap_2026_callstrike.png",
                     median_sz_top=med_top, median_sz_bot=med_bot)

    # Largest significant shrinkage region (negative delta)
    pitches_min = np.minimum(n25, n26)
    largest_neg = find_largest_significant_region(
        delta, lo, hi, sign=-1,
        x_lo=X_LO, x_hi=X_HI, x_bin=X_BIN, z_lo=Z_LO, z_hi=Z_HI, z_bin=Z_BIN,
        min_delta_pp=0.03, pitches=pitches_min,
    )
    largest_pos = find_largest_significant_region(
        delta, lo, hi, sign=+1,
        x_lo=X_LO, x_hi=X_HI, x_bin=X_BIN, z_lo=Z_LO, z_hi=Z_HI, z_bin=Z_BIN,
        min_delta_pp=0.03, pitches=pitches_min,
    )

    # Total significant area (any direction, |delta|>=3pp & CI excl 0)
    sig_neg = np.isfinite(delta) & (hi < 0) & (delta <= -0.03)
    sig_pos = np.isfinite(delta) & (lo > 0) & (delta >= 0.03)
    area_neg = float(sig_neg.sum() * X_BIN * Z_BIN)
    area_pos = float(sig_pos.sum() * X_BIN * Z_BIN)

    # Edge-of-zone summary: split rule-book box edges into top/bottom/left/right "rings"
    # by taking cells whose center is within 0.2 ft of each edge of the rule-book box
    def cell_centers(lo_, bin_, n_):
        return lo_ + (np.arange(n_) + 0.5) * bin_

    nz, nx = delta.shape
    xs = cell_centers(X_LO, X_BIN, nx)
    zs = cell_centers(Z_LO, Z_BIN, nz)
    xx, zz = np.meshgrid(xs, zs)

    # Edges of rule-book zone, with a 0.2 ft band on each side
    band = 0.20
    inside = (np.abs(xx) <= PLATE_X[1]) & (zz >= RULEBOOK_Z[0]) & (zz <= RULEBOOK_Z[1])
    top_band = (zz >= RULEBOOK_Z[1] - band) & (zz <= RULEBOOK_Z[1] + band) & (np.abs(xx) <= PLATE_X[1] + band)
    bot_band = (zz >= RULEBOOK_Z[0] - band) & (zz <= RULEBOOK_Z[0] + band) & (np.abs(xx) <= PLATE_X[1] + band)
    out_top = (zz > RULEBOOK_Z[1])
    out_bot = (zz < RULEBOOK_Z[0])

    def region_summary(mask, label):
        m = mask & np.isfinite(delta)
        if not m.any():
            return {"label": label, "n_cells": 0}
        w = pitches_min[m]
        d_w = float(np.average(delta[m], weights=np.maximum(w, 1)))
        return {
            "label": label,
            "n_cells": int(m.sum()),
            "n_pitches_min_year": int(w.sum()),
            "wmean_delta_pp_x100": d_w * 100,
            "frac_cells_sig_neg": float((sig_neg & m).sum() / max(m.sum(), 1)),
            "frac_cells_sig_pos": float((sig_pos & m).sum() / max(m.sum(), 1)),
        }

    rings = [
        region_summary(inside, "rule-book interior"),
        region_summary(top_band, "top edge band (+/- 0.2ft of 3.5ft)"),
        region_summary(bot_band, "bottom edge band (+/- 0.2ft of 1.5ft)"),
        region_summary(out_top, "above rule-book box (z>3.5ft)"),
        region_summary(out_bot, "below rule-book box (z<1.5ft)"),
    ]

    # ---- Supplementary: extended 2025 window ----
    df25_ext = load_2025_extended_to_apr22()
    cp25_ext = called_pitches_subset(df25_ext)
    cp26_full = called_pitches_subset(df26)
    n25e, s25e = aggregate_grid(cp25_ext, "plate_x", "plate_z", X_LO, X_HI, X_BIN, Z_LO, Z_HI, Z_BIN)
    n26f, s26f = aggregate_grid(cp26_full, "plate_x", "plate_z", X_LO, X_HI, X_BIN, Z_LO, Z_HI, Z_BIN)
    delta_ext, lo_ext, hi_ext = bootstrap_delta_ci(n25e, s25e, n26f, s26f)
    plot_heatmap(delta_ext, lo_ext, hi_ext, X_LO, X_HI, X_BIN, Z_LO, Z_HI, Z_BIN,
                 title="(Supplementary) 2026 minus 2025, both Mar 27-Apr 22",
                 savepath=CHART_DIR / "heatmap_zone_delta_ext.png",
                 overlay_zone=True, median_sz_top=med_top, median_sz_bot=med_bot)

    # ---- Supplementary: height-normalized grid ----
    cp25n = cp25.copy(); cp26n = cp26.copy()
    cp25n["plate_z_norm"] = safe_plate_z_norm(cp25)
    cp26n["plate_z_norm"] = safe_plate_z_norm(cp26)
    cp25n = cp25n.dropna(subset=["plate_z_norm"])
    cp26n = cp26n.dropna(subset=["plate_z_norm"])
    NX_LO, NX_HI, NX_BIN = -1.7, 1.7, 0.10
    NZ_LO, NZ_HI, NZ_BIN = -0.4, 1.4, 0.05
    n25n, s25n = aggregate_grid(cp25n, "plate_x", "plate_z_norm",
                                NX_LO, NX_HI, NX_BIN, NZ_LO, NZ_HI, NZ_BIN)
    n26n, s26n = aggregate_grid(cp26n, "plate_x", "plate_z_norm",
                                NX_LO, NX_HI, NX_BIN, NZ_LO, NZ_HI, NZ_BIN)
    delta_n, lo_n, hi_n = bootstrap_delta_ci(n25n, s25n, n26n, s26n)
    plot_heatmap(delta_n, lo_n, hi_n, NX_LO, NX_HI, NX_BIN, NZ_LO, NZ_HI, NZ_BIN,
                 title="(Supplementary) Called-strike rate delta, height-normalized z\n2026 minus 2025 (Mar 27-Apr 14)",
                 savepath=CHART_DIR / "heatmap_zone_delta_norm.png",
                 overlay_zone=True, overlay_norm=True)

    # ---- Marginal 1D profiles (rate vs plate_z within plate width) ----
    # Used in REPORT and in the GAM step too
    plate_mask25 = cp25["plate_x"].between(PLATE_X[0], PLATE_X[1])
    plate_mask26 = cp26["plate_x"].between(PLATE_X[0], PLATE_X[1])
    cp25p = cp25.loc[plate_mask25].copy()
    cp26p = cp26.loc[plate_mask26].copy()
    z_edges = np.arange(0.5, 5.0 + 0.10, 0.10)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    def rate_by_bin(d):
        bb = pd.cut(d["plate_z"], z_edges, labels=False, include_lowest=True)
        g = d.groupby(bb)["is_called_strike"].agg(["mean", "size"])
        g = g.reindex(range(len(z_centers)))
        return g["mean"].to_numpy(), g["size"].to_numpy()

    r25, c25_ = rate_by_bin(cp25p)
    r26, c26_ = rate_by_bin(cp26p)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=140)
    ax.plot(z_centers, r25 * 100, color="#1f77b4", label="2025 (Mar 27-Apr 14)")
    ax.plot(z_centers, r26 * 100, color="#d62728", label="2026 (Mar 27-Apr 14)")
    ax.fill_between(z_centers, r25 * 100, r26 * 100,
                    where=(r26 < r25), color="#d62728", alpha=0.15, label="2026 < 2025")
    ax.fill_between(z_centers, r25 * 100, r26 * 100,
                    where=(r26 > r25), color="#1f77b4", alpha=0.15, label="2026 > 2025")
    ax.axvline(med_bot, color="0.4", linestyle=":", linewidth=1.0, label=f"med sz_bot={med_bot:.2f}")
    ax.axvline(med_top, color="0.4", linestyle=":", linewidth=1.0, label=f"med sz_top={med_top:.2f}")
    ax.set_xlabel("plate_z (ft, ball center)"); ax.set_ylabel("called-strike rate (%)")
    ax.set_title("Vertical called-strike profile, in-plate-width pitches")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "vertical_profile_in_plate.png", bbox_inches="tight")
    plt.close(fig)

    # Persist artifacts
    np.savez(
        ART_DIR / "zone_grid.npz",
        n25=n25, s25=s25, n26=n26, s26=s26,
        delta=delta, ci_lo=lo, ci_hi=hi,
        x_lo=X_LO, x_hi=X_HI, x_bin=X_BIN,
        z_lo=Z_LO, z_hi=Z_HI, z_bin=Z_BIN,
        med_top=med_top, med_bot=med_bot,
    )
    np.savez(
        ART_DIR / "zone_grid_norm.npz",
        n25=n25n, s25=s25n, n26=n26n, s26=s26n,
        delta=delta_n, ci_lo=lo_n, ci_hi=hi_n,
        x_lo=NX_LO, x_hi=NX_HI, x_bin=NX_BIN,
        z_lo=NZ_LO, z_hi=NZ_HI, z_bin=NZ_BIN,
    )

    return {
        "median_sz_top": med_top,
        "median_sz_bot": med_bot,
        "largest_neg_region_abs": largest_neg,
        "largest_pos_region_abs": largest_pos,
        "area_sigsig_neg_sqft": area_neg,
        "area_sigsig_pos_sqft": area_pos,
        "rings": rings,
    }


if __name__ == "__main__":
    out = run()
    import json
    print(json.dumps(out, default=str, indent=2))

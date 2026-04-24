"""Logistic GAM testing year x zone-shape interaction.

Model:
  is_called_strike ~ s(plate_x) + s(plate_z) + season_2026
                   + season_2026 : (s_low_z + s_top_z) (effective interaction
                   via tensor of season indicator with z spline basis)

We use pygam.LogisticGAM with a tensor `te(plate_x, plate_z)` plus a season
indicator AND a separate season-specific tensor whose coefficients we test for
joint zero. Then we report:
  - GAM deviance pooled vs additive only vs full interaction
  - Likelihood ratio chi-square test of season:zone interaction
  - Partial-dependence plot of called-strike probability vs plate_z by season,
    holding plate_x at center
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pygam import LogisticGAM, s, te, f
from scipy import stats

from common import (
    PROJECT_ROOT, called_pitches_subset,
    load_2025_samewin, load_2026, restrict_to_primary_window,
)

OUT_DIR = PROJECT_ROOT / "claude-analysis"
CHART_DIR = OUT_DIR / "charts"
ART_DIR = OUT_DIR / "artifacts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)


def build_pooled() -> pd.DataFrame:
    df25 = called_pitches_subset(load_2025_samewin())
    df26 = called_pitches_subset(restrict_to_primary_window(load_2026()))
    df25 = df25[["plate_x", "plate_z", "is_called_strike"]].assign(season=2025).dropna()
    df26 = df26[["plate_x", "plate_z", "is_called_strike"]].assign(season=2026).dropna()
    df = pd.concat([df25, df26], ignore_index=True)
    df["s2026"] = (df["season"] == 2026).astype(int)
    # Trim outliers to a sane plate envelope
    df = df.loc[df["plate_x"].between(-1.6, 1.6) & df["plate_z"].between(0.8, 4.6)]
    return df.reset_index(drop=True)


def fit_gam_additive(X, y):
    # te(plate_x, plate_z) + season indicator (no interaction)
    gam = LogisticGAM(te(0, 1, n_splines=8) + f(2)).fit(X, y)
    return gam


def fit_gam_with_interaction(X, y):
    # Season-specific tensor: stack (plate_x, plate_z) twice and use season indicator
    # to gate one. Easiest in pygam: include te(0,1) + s(3) where col 3 is plate_z*season
    # to allow z-shape to differ by season. Add x*season too.
    # We'll construct features: plate_x, plate_z, season, plate_z*season, plate_x*season.
    # Build a richer design by treating season as a categorical and adding s(plate_z) for season=2026 minus 2025.
    # Concrete: model is te(0,1) + f(2) + s(3) + s(4) where col 3 = plate_z if 2026 else 0,
    #                                              col 4 = plate_x if 2026 else 0.
    gam = LogisticGAM(te(0, 1, n_splines=8) + f(2) + s(3, n_splines=12) + s(4, n_splines=12)).fit(X, y)
    return gam


def lrt_p(d_full: float, d_red: float, df_diff: float) -> float:
    chi2 = max(d_red - d_full, 0.0)
    if df_diff <= 0:
        return np.nan
    return float(1 - stats.chi2.cdf(chi2, df_diff))


def run() -> dict:
    df = build_pooled()
    print(f"[gam] rows pooled (2025+2026): {len(df):,}")

    base = df[["plate_x", "plate_z", "s2026"]].to_numpy(dtype=float)
    y = df["is_called_strike"].to_numpy(dtype=int)

    # Additive
    gam_add = fit_gam_additive(base, y)
    dev_add = gam_add.statistics_["deviance"]
    edof_add = gam_add.statistics_["edof"]
    print(f"[gam] additive deviance={dev_add:.2f}, edf={edof_add:.2f}")

    # Interaction model
    Xi = np.column_stack([
        df["plate_x"].to_numpy(),
        df["plate_z"].to_numpy(),
        df["s2026"].to_numpy(),
        df["plate_z"].to_numpy() * df["s2026"].to_numpy(),
        df["plate_x"].to_numpy() * df["s2026"].to_numpy(),
    ]).astype(float)
    gam_int = fit_gam_with_interaction(Xi, y)
    dev_int = gam_int.statistics_["deviance"]
    edof_int = gam_int.statistics_["edof"]
    p_int = lrt_p(dev_int, dev_add, edof_int - edof_add)
    print(f"[gam] interaction deviance={dev_int:.2f}, edf={edof_int:.2f}, LRT p={p_int:.4g}")

    # Per-term p-values from interaction model
    summary_terms = []
    try:
        # pygam summary returns a dataframe of terms
        for i, term in enumerate(gam_int.terms):
            if term.isintercept:
                continue
            summary_terms.append({
                "i": i,
                "term": str(term),
                "lambda": list(term.lam) if hasattr(term, "lam") else None,
                "n_coefs": term.n_coefs,
            })
    except Exception:
        pass

    # Wald statistic for interaction-only terms (s(3) + s(4))
    # pygam exposes statistics_['p_values'] per term
    pvals = gam_int.statistics_.get("p_values", None)
    print(f"[gam] term p-values: {pvals}")

    # Partial-dependence: predicted called-strike prob vs plate_z by season,
    # holding plate_x = 0
    z_grid = np.linspace(1.0, 4.5, 80)
    Xpred25 = np.column_stack([
        np.zeros_like(z_grid), z_grid,
        np.zeros_like(z_grid),                  # season=2025
        np.zeros_like(z_grid),                  # plate_z * season
        np.zeros_like(z_grid),                  # plate_x * season
    ])
    Xpred26 = np.column_stack([
        np.zeros_like(z_grid), z_grid,
        np.ones_like(z_grid),
        z_grid,
        np.zeros_like(z_grid),
    ])
    p25 = gam_int.predict_proba(Xpred25)
    p26 = gam_int.predict_proba(Xpred26)

    # Confidence intervals via predict_proba CI (simulation in pygam uses bootstrap of coefs)
    try:
        lo25, hi25 = gam_int.confidence_intervals(Xpred25, width=0.95).T[[0, 1]]
        lo26, hi26 = gam_int.confidence_intervals(Xpred26, width=0.95).T[[0, 1]]
    except Exception:
        lo25 = hi25 = lo26 = hi26 = None

    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=140)
    ax.plot(z_grid, p25 * 100, color="#1f77b4", lw=2, label="2025 (plate_x=0)")
    ax.plot(z_grid, p26 * 100, color="#d62728", lw=2, label="2026 (plate_x=0)")
    if lo25 is not None:
        ax.fill_between(z_grid, lo25 * 100, hi25 * 100, color="#1f77b4", alpha=0.15)
        ax.fill_between(z_grid, lo26 * 100, hi26 * 100, color="#d62728", alpha=0.15)
    ax.axvspan(1.5, 3.5, color="0.92", zorder=-1, label="rule-book vertical zone")
    ax.set_xlabel("plate_z (ft)"); ax.set_ylabel("P(called strike) %, plate_x=0")
    ax.set_title("Logistic GAM partial dependence on plate_z, by season")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "gam_partial_dependence.png", bbox_inches="tight")
    plt.close(fig)

    # Difference curve
    fig, ax = plt.subplots(figsize=(9.0, 4.5), dpi=140)
    diff = (p26 - p25) * 100
    ax.plot(z_grid, diff, color="black", lw=2)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvspan(1.5, 3.5, color="0.92", zorder=-1, label="rule-book vertical zone")
    ax.set_xlabel("plate_z (ft)"); ax.set_ylabel("P(called strike) 2026 minus 2025 (pp)")
    ax.set_title("GAM-predicted called-strike rate change vs plate_z (plate_x=0)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "gam_partial_dependence_diff.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "n_rows": int(len(df)),
        "additive_deviance": float(dev_add),
        "additive_edf": float(edof_add),
        "interaction_deviance": float(dev_int),
        "interaction_edf": float(edof_int),
        "lrt_p_value_interaction_vs_additive": float(p_int) if np.isfinite(p_int) else None,
        "term_p_values_interaction_model": pvals.tolist() if hasattr(pvals, "tolist") else pvals,
        "summary_terms": summary_terms,
    }


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str))

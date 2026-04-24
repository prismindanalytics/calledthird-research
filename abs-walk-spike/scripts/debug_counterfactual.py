"""DEBUG: aggregate-level sign check for the counterfactual sign dispute.

Question: in absolute coords, does applying the 2025 zone classifier to 2026 called pitches
produce MORE called strikes (→ fewer walks → POSITIVE attribution) or FEWER called strikes
(→ more walks → NEGATIVE attribution)?

This bypasses the PA-replay layer entirely — just aggregate p_strike_2025 over 2026 called
pitches and compare to actual 2026 called-strike rate. If average p_strike_2025 > actual
strike rate, then 2025 model is more strike-friendly on 2026 pitches → counterfactual has
FEWER walks → POSITIVE attribution (Codex's direction). If average p_strike_2025 < actual,
NEGATIVE attribution (orchestrator's direction).

Whichever way the aggregate goes, that's the ground truth for the sign of the average effect.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
DATA_2026 = ROOT / "research" / "abs-walk-spike" / "data" / "statcast_2026_mar27_apr22.parquet"
DATA_2025 = ROOT / "research" / "count-distribution-abs" / "data" / "statcast_2025_mar27_apr14.parquet"

WINDOW_END = pd.Timestamp("2026-04-14")  # apples-to-apples with 2025 file


def load_called(path, year, end=None):
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date"].dt.year == year]
    if end is not None:
        df = df[df["game_date"] <= end]
    df = df.dropna(subset=["plate_x", "plate_z"])
    called = df[df["description"].isin(["called_strike", "ball"])].copy()
    called["is_strike"] = (called["description"] == "called_strike").astype(int)
    return called


def fit_model(called, seed=42):
    """Match Codex's exact spec: poly-2, StandardScaler, LogReg C=0.2."""
    X = called[["plate_x", "plate_z"]].to_numpy()
    y = called["is_strike"].to_numpy()
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=0.2, max_iter=4000, random_state=seed)),
    ])
    pipe.fit(X, y)
    return pipe


def main():
    called_25 = load_called(DATA_2025, 2025)
    called_26 = load_called(DATA_2026, 2026, end=WINDOW_END)
    print(f"2025 called pitches: {len(called_25):,}")
    print(f"2026 called pitches (Mar 27-Apr 14): {len(called_26):,}")

    # Fit each season's zone classifier
    model_25 = fit_model(called_25, seed=42)
    model_26 = fit_model(called_26, seed=43)

    # Predict on 2026 called pitches
    X_26 = called_26[["plate_x", "plate_z"]].to_numpy()
    p25_on_26 = model_25.predict_proba(X_26)[:, 1]
    p26_on_26 = model_26.predict_proba(X_26)[:, 1]

    # Actual 2026 called-strike rate
    actual_cs_rate = called_26["is_strike"].mean()

    print(f"\n=== AGGREGATE STRIKE-RATE COMPARISON (on 2026 called pitches) ===")
    print(f"Actual 2026 called-strike rate:        {actual_cs_rate*100:.3f}%")
    print(f"Mean p_strike under 2026 model (sanity): {p26_on_26.mean()*100:.3f}%")
    print(f"Mean p_strike under 2025 model:          {p25_on_26.mean()*100:.3f}%")
    print(f"Delta (2025_model - actual):             {(p25_on_26.mean() - actual_cs_rate)*100:+.3f}pp")
    print()
    if p25_on_26.mean() > actual_cs_rate:
        print("→ 2025 zone is MORE strike-friendly on 2026 pitches.")
        print("  Counterfactual: more strikes → fewer walks → POSITIVE attribution.")
        print("  (This matches Codex's +40% direction.)")
    else:
        print("→ 2025 zone is LESS strike-friendly on 2026 pitches.")
        print("  Counterfactual: fewer strikes → more walks → NEGATIVE attribution.")
        print("  (This matches orchestrator's -46% direction.)")

    print()
    print("=== BY ZONE REGION ===")
    # Stratify by top half vs bottom half of zone
    called_26 = called_26.assign(p25=p25_on_26, p26=p26_on_26)
    bins = [
        ("dirt (z<1.5)", called_26["plate_z"] < 1.5),
        ("bottom (1.5<=z<2.0)", (called_26["plate_z"] >= 1.5) & (called_26["plate_z"] < 2.0)),
        ("middle (2.0<=z<2.7)", (called_26["plate_z"] >= 2.0) & (called_26["plate_z"] < 2.7)),
        ("upper (2.7<=z<3.2)", (called_26["plate_z"] >= 2.7) & (called_26["plate_z"] < 3.2)),
        ("top (3.2<=z<3.5)", (called_26["plate_z"] >= 3.2) & (called_26["plate_z"] < 3.5)),
        ("above (z>=3.5)", called_26["plate_z"] >= 3.5),
    ]
    print(f"{'Region':<22} {'N':>7} {'Actual %':>10} {'p25 %':>10} {'p26 %':>10} {'Δ p25-actual':>14}")
    for label, mask in bins:
        sub = called_26[mask]
        if len(sub) == 0:
            continue
        actual = sub["is_strike"].mean() * 100
        p25 = sub["p25"].mean() * 100
        p26 = sub["p26"].mean() * 100
        print(f"{label:<22} {len(sub):>7} {actual:>9.2f}% {p25:>9.2f}% {p26:>9.2f}% {p25-actual:>+13.2f}pp")

    print()
    print("=== INSIDE-ZONE MASS-WEIGHTED ANALYSIS ===")
    # Zone box: x in [-0.71, 0.71] (17"/2/12), z in [1.62, 3.21] (ABS rule)
    in_zone = (
        (called_26["plate_x"].abs() <= 0.71)
        & (called_26["plate_z"] >= 1.62)
        & (called_26["plate_z"] <= 3.21)
    )
    print(f"In ABS zone (plate, x∈±0.71, z∈[1.62, 3.21]): {in_zone.sum():,}")
    print(f"  actual called-strike rate: {called_26.loc[in_zone, 'is_strike'].mean()*100:.2f}%")
    print(f"  p25 mean: {called_26.loc[in_zone, 'p25'].mean()*100:.2f}%")
    print(f"  p26 mean: {called_26.loc[in_zone, 'p26'].mean()*100:.2f}%")

    # Outside zone (where most balls happen)
    out_zone = ~in_zone
    print(f"\nOutside ABS zone: {out_zone.sum():,}")
    print(f"  actual called-strike rate: {called_26.loc[out_zone, 'is_strike'].mean()*100:.2f}%")
    print(f"  p25 mean: {called_26.loc[out_zone, 'p25'].mean()*100:.2f}%")
    print(f"  p26 mean: {called_26.loc[out_zone, 'p26'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build pitcher fame quartile and catcher framing tier from 2026 in-season Statcast.

FanGraphs/Savant 2025 endpoints are 403-blocked from this network, so we use the
2026 Statcast corpus already on disk (Mar 27 - May 3) as the basis. This is the
fallback path the research brief explicitly allows ("otherwise current-season-to-date
values"). Note: derived in-season metrics correlate strongly with prior-season K-BB%
and framing runs, but they have a small endogeneity risk because they share data
with the H2/H3 outcome — we therefore *exclude challenge pitches* and *exclude
borderline take-pitches that feed H3* when computing these tiers, so the quartile
assignments are independent of the outcome variables we model later.

Output:
  data/pitcher_fame_quartile.parquet   pitcher_id, k_bb_pct, fame_quartile (1=top..4=bottom)
  data/catcher_framing_tier.parquet    catcher_id, framing_score, framing_tier (top/mid/bottom)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

SC_OLD = Path("/Users/haohu/Documents/GitHub/calledthird/research/abs-walk-spike/data/statcast_2026_mar27_apr22.parquet")
SC_NEW = DATA / "statcast_2026_apr23_may04.parquet"


def _load_full_statcast(cols: list[str]) -> pd.DataFrame:
    frames = []
    for p in (SC_OLD, SC_NEW):
        if p.exists():
            frames.append(pd.read_parquet(p, columns=cols))
    return pd.concat(frames, ignore_index=True)


def build_pitcher_fame() -> int:
    df = _load_full_statcast(["pitcher", "events", "description", "type", "balls", "strikes"])
    # Identify K and BB events. Statcast 'events' is non-null only on terminal pitches.
    df = df.dropna(subset=["pitcher"])
    df["pitcher"] = df["pitcher"].astype(int)
    # Plate appearances = number of unique terminal pitches per pitcher
    term = df["events"].notna() & (df["events"] != "")
    pa = df.loc[term].groupby("pitcher").size().rename("pa")
    is_k = df["events"].isin([
        "strikeout", "strikeout_double_play", "strikeout_triple_play"
    ])
    is_bb = df["events"].isin(["walk", "intent_walk"])
    k = df.loc[is_k].groupby("pitcher").size().rename("k")
    bb = df.loc[is_bb].groupby("pitcher").size().rename("bb")
    out = pd.concat([pa, k, bb], axis=1).fillna(0).astype(int)
    out = out[out["pa"] >= 30]   # qualified-ish floor for stable rate
    out["k_bb_pct"] = (out["k"] - out["bb"]) / out["pa"] * 100.0
    out = out.reset_index().rename(columns={"pitcher": "pitcher_id"})
    out["fame_quartile"] = 5 - pd.qcut(out["k_bb_pct"], 4, labels=False, duplicates="drop").astype(int) - 1
    print(f"Pitcher fame: {len(out)} pitchers (PA>=30)")
    print(out["fame_quartile"].value_counts().sort_index())
    print(f"  K-BB% quartile cuts: {out.groupby('fame_quartile')['k_bb_pct'].agg(['min','max','mean']).round(1)}")
    out_path = DATA / "pitcher_fame_quartile.parquet"
    out[["pitcher_id", "pa", "k", "bb", "k_bb_pct", "fame_quartile"]].to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}")
    return 0


def build_catcher_framing() -> int:
    """In-season catcher framing proxy: called-strike rate on borderline takes,
    EXCLUDING the borderline subset we model in H3 (within ±0.3 ft of edge).
    We use a wider 'borderline' definition (between 0.3 and 0.6 ft from the edge)
    to reduce the leakage risk into our H3 outcome.
    """
    df = _load_full_statcast([
        "fielder_2", "description", "plate_x", "plate_z", "sz_top", "sz_bot",
        "stand", "balls", "strikes",
    ])
    df = df[df["description"].isin(["called_strike", "ball"])].copy()
    df = df.dropna(subset=["fielder_2", "plate_x", "plate_z", "sz_top", "sz_bot"])
    df["fielder_2"] = df["fielder_2"].astype(int)
    # rulebook horizontal half-width = 0.83 ft; vertical zone (sz_bot, sz_top)
    df["edge_x_dist"] = (df["plate_x"].abs() - 0.83).abs()
    above = df["plate_z"] - df["sz_top"]
    below = df["sz_bot"] - df["plate_z"]
    df["edge_z_dist"] = np.minimum(above.abs(), below.abs())
    df["edge_dist_ft"] = np.minimum(df["edge_x_dist"], df["edge_z_dist"])
    # framing band: within 0.6 ft of the edge but NOT within 0.3 ft
    band = (df["edge_dist_ft"] >= 0.30) & (df["edge_dist_ft"] <= 0.60)
    df = df[band].copy()
    df["called_strike"] = (df["description"] == "called_strike").astype(int)
    g = df.groupby("fielder_2").agg(n=("called_strike", "size"), framing_score=("called_strike", "mean"))
    g = g[g["n"] >= 100].reset_index()
    g = g.rename(columns={"fielder_2": "catcher_id"})
    g["framing_tier"] = pd.qcut(g["framing_score"], 3, labels=["bottom", "mid", "top"], duplicates="drop").astype(str)
    out_path = DATA / "catcher_framing_tier.parquet"
    g.to_parquet(out_path, index=False)
    print(f"Catcher framing: {len(g)} catchers")
    print(g["framing_tier"].value_counts())
    print(f"  framing_score per tier: {g.groupby('framing_tier')['framing_score'].agg(['min','max','mean']).round(3)}")
    print(f"Saved -> {out_path}")
    return 0


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    rc1 = build_pitcher_fame()
    rc2 = build_catcher_framing()
    return rc1 or rc2


if __name__ == "__main__":
    sys.exit(main())

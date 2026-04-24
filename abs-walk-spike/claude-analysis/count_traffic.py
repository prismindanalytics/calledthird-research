"""Where the walk-rate spike comes from: PA-flow analysis through the count tree.

Key idea: walks rose from 9.11% to 9.92% (+0.82pp). Conditional walk rate at any
single advanced count (3-2, 3-1, 3-0) didn't change much. The spike instead
shows up as MORE PAs reaching deep counts. Decomposing the walk rate change:

  walkrate = sum_c P(reach c) * P(walk | reach c, never had walk earlier)
           = sum_c P(reach c) * P(walk | exit c with 4 balls)
           ~ for each count P(get a ball) -> next count, etc.

We split the +0.82pp into:
  (A) traffic effect: how much rises if we hold per-count walk rate at 2025 levels
      but use 2026 traffic (PA reach rate) into each count.
  (B) conditional effect: hold traffic at 2025 but apply 2026 per-count walk rates.

Then we look at the EARLY-COUNT called-strike rate (the cause of more traffic):
the first-pitch strike rate, the 0-1 -> 0-2 progression, etc.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    PROJECT_ROOT, ALL_COUNTS,
    load_2025_samewin, load_2026, restrict_to_primary_window,
    plate_appearances, walk_flag, called_pitches_subset,
)

OUT_DIR = PROJECT_ROOT / "claude-analysis"
CHART_DIR = OUT_DIR / "charts"
ART_DIR = OUT_DIR / "artifacts"


def attach_pa_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_pa_id"] = (df["game_pk"].astype("Int64").astype(str) + "_"
                    + df["at_bat_number"].astype("Int64").astype(str))
    return df


def per_count_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = attach_pa_id(df)
    df["count_state"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    pa = df.dropna(subset=["events"])
    pa = pa.loc[pa["events"].astype(str).str.len() > 0]
    pa_walk = pa.assign(_walk=walk_flag(pa))[["_pa_id", "_walk"]].drop_duplicates("_pa_id")
    n_pa_total = pa["_pa_id"].nunique()
    reached = (
        df.drop_duplicates(subset=["_pa_id", "count_state"])[["_pa_id", "count_state"]]
        .merge(pa_walk, on="_pa_id", how="left")
    )
    reached["_walk"] = reached["_walk"].fillna(0).astype(int)
    g = reached.groupby("count_state").agg(pas_reached=("_pa_id", "count"),
                                           walks=("_walk", "sum"))
    g["frac_pa_reaching"] = g["pas_reached"] / n_pa_total
    g["walk_rate_given_reach"] = g["walks"] / g["pas_reached"]
    return g.reindex(list(ALL_COUNTS)), n_pa_total


def first_pitch_outcomes(df: pd.DataFrame) -> dict:
    """Of all PAs, what % first pitch in zone, % first pitch called strike, % swing."""
    df = attach_pa_id(df)
    fp = df.loc[(df["balls"] == 0) & (df["strikes"] == 0)].copy()
    fp = fp.dropna(subset=["description"])
    n = fp["_pa_id"].nunique()
    # Use first row per PA at 0-0 - usually only one
    fp = fp.drop_duplicates(subset=["_pa_id"])
    desc = fp["description"]
    n_total = len(fp)
    out = {
        "n_first_pitches": n_total,
        "called_strike_pct": float((desc == "called_strike").sum() / n_total * 100),
        "ball_pct": float((desc == "ball").sum() / n_total * 100),
        "swung_at_pct": float(desc.isin([
            "swinging_strike", "swinging_strike_blocked", "foul",
            "foul_tip", "foul_bunt", "hit_into_play", "missed_bunt",
        ]).sum() / n_total * 100),
        "automatic_ball_pct": float((desc == "automatic_ball").sum() / n_total * 100),
        "automatic_strike_pct": float((desc == "automatic_strike").sum() / n_total * 100),
    }
    return out


def transition_probs(df: pd.DataFrame, from_count: str) -> dict:
    """For PAs that reached `from_count`, what fraction get a ball / strike on the next pitch?"""
    df = attach_pa_id(df)
    df["count_state"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    df_sorted = df.sort_values(["_pa_id", "pitch_number"])
    cur = df_sorted.loc[df_sorted["count_state"] == from_count].copy()
    if len(cur) == 0:
        return {}
    n = len(cur)
    desc = cur["description"]
    return {
        "from_count": from_count,
        "n_pitches": int(n),
        "ball_pct": float((desc == "ball").sum() / n * 100),
        "called_strike_pct": float((desc == "called_strike").sum() / n * 100),
        "swung_at_pct": float(desc.isin([
            "swinging_strike", "swinging_strike_blocked", "foul",
            "foul_tip", "foul_bunt", "hit_into_play", "missed_bunt",
        ]).sum() / n * 100),
        "auto_strike_pct": float((desc == "automatic_strike").sum() / n * 100),
        "auto_ball_pct": float((desc == "automatic_ball").sum() / n * 100),
        "blocked_ball_pct": float((desc == "blocked_ball").sum() / n * 100),
    }


def decompose_walk_delta(g25: pd.DataFrame, g26: pd.DataFrame) -> dict:
    """Counterfactual decomposition: if 2026 had 2025 conditional rates, what's walk rate?
    And vice versa."""
    # Walk rate at 0-0 = pooled walk rate
    p25_pool = g25.loc["0-0", "walk_rate_given_reach"]
    p26_pool = g26.loc["0-0", "walk_rate_given_reach"]
    delta_total = p26_pool - p25_pool

    # Decomposition by "how much of delta is at 0-0 itself (the only stratum with all PAs)"
    # is uninteresting because every PA passes through 0-0. We instead decompose the
    # 2026 per-count walk rate into its components and ask which counts are responsible
    # for the largest walk-conditional flow.
    # Simple alt: counter-factual sum_c [(reach_26 - reach_25) * walkrate_25]  vs
    #             sum_c [reach_25 * (walkrate_26 - walkrate_25)]
    # using 12 disjoint counts is double counting. Use only "leaf" counts:
    # at any count one of: ball -> next count (deeper), strike -> deeper, in play -> PA ends.
    # A cleaner decomposition: compute walk rate as expected number of walks per PA
    # = sum over deep counts c (c=3-X) of P(PA reaches c with 0 prior walks) * P(walks from c)
    # But every walk passes through 3-X exactly once; so actually:
    # walk rate = sum_c P(PA terminates as walk after exiting count c with another ball)
    # which collapses to total walks/total PAs.
    # We'll instead just report:
    #   - traffic delta into 3-balls counts: delta P(reach 3-0/3-1/3-2)
    #   - conditional delta per 3-X: delta walkrate | reach 3-X
    # And do a simple multiplicative attribution.
    rows = []
    for c in ["3-0", "3-1", "3-2"]:
        r25 = g25.loc[c, "frac_pa_reaching"]
        r26 = g26.loc[c, "frac_pa_reaching"]
        w25 = g25.loc[c, "walk_rate_given_reach"]
        w26 = g26.loc[c, "walk_rate_given_reach"]
        # Walks per PA contributed (rough): r * exit prob to walk; total walks per PA
        # don't equal sum of these because counts overlap. Use: PAs that reach 3-X and then walk = r*w (count-anchored).
        # This double-counts (a 3-2 walk is also a 3-1 reached, also 3-0 maybe).
        # For attribution, we use FIRST hit count tree: a PA's first time at 3 balls is what we anchor.
        rows.append({
            "count": c,
            "reach_25": float(r25), "reach_26": float(r26),
            "walk_rate_25": float(w25), "walk_rate_26": float(w26),
            "traffic_delta_pp": float((r26 - r25) * 100),
            "conditional_delta_pp": float((w26 - w25) * 100),
        })
    # First-time-3-balls reach rate (anchored): use 3-0 reach rate as proxy for "PA's first time at 3 balls"
    # since every 3-1 and 3-2 PA passed through 3-0.
    first_3b_25 = g25.loc["3-0", "frac_pa_reaching"]
    first_3b_26 = g26.loc["3-0", "frac_pa_reaching"]
    return {
        "delta_total_pooled_walkrate_pp": float(delta_total * 100),
        "frac_pa_first_reach_3balls_25": float(first_3b_25 * 100),
        "frac_pa_first_reach_3balls_26": float(first_3b_26 * 100),
        "frac_3balls_traffic_delta_pp": float((first_3b_26 - first_3b_25) * 100),
        "rows": rows,
    }


def first_pitch_called_strike_rate(df: pd.DataFrame) -> float:
    """Fraction of called pitches at 0-0 that are called strikes."""
    cp = called_pitches_subset(df)
    fp = cp.loc[(cp["balls"] == 0) & (cp["strikes"] == 0)]
    if len(fp) == 0:
        return float("nan")
    return float(fp["is_called_strike"].mean())


def run() -> dict:
    df25 = load_2025_samewin()
    df26 = restrict_to_primary_window(load_2026())
    g25, n25 = per_count_summary(df25)
    g26, n26 = per_count_summary(df26)
    print(f"[count_traffic] 2025 PAs: {n25:,}  2026 PAs (Mar27-Apr14): {n26:,}")
    print("[count_traffic] reach fractions:")
    print(g25[["frac_pa_reaching", "walk_rate_given_reach"]])
    print(g26[["frac_pa_reaching", "walk_rate_given_reach"]])

    # Plot reach-fraction comparison
    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    counts = list(g25.index)
    x = np.arange(len(counts)); w = 0.4
    r25 = g25["frac_pa_reaching"].to_numpy() * 100
    r26 = g26["frac_pa_reaching"].to_numpy() * 100
    ax.bar(x - w/2, r25, width=w, color="#1f77b4", label="2025")
    ax.bar(x + w/2, r26, width=w, color="#d62728", label="2026")
    for i, c in enumerate(counts):
        d = (r26[i] - r25[i])
        col = "darkgreen" if d > 0 else "darkred"
        sym = "+" if d >= 0 else ""
        ax.text(i, max(r25[i], r26[i]) + 1.5, f"{sym}{d:.1f}pp", ha="center", fontsize=8, color=col)
    ax.set_xticks(x); ax.set_xticklabels(counts)
    ax.set_xlabel("Count state"); ax.set_ylabel("Fraction of PAs that reached this count (%)")
    ax.set_title("Count-state reach traffic, 2025 vs 2026 (Mar 27 - Apr 14)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "count_reach_traffic.png", bbox_inches="tight")
    plt.close(fig)

    decomp = decompose_walk_delta(g25, g26)

    # Transition probabilities at key counts to see where pitchers fell behind more
    transitions = {}
    for c in ["0-0", "1-0", "2-0", "3-0", "0-1", "1-1", "2-1", "3-1", "0-2", "1-2", "2-2", "3-2"]:
        transitions[c] = {
            "2025": transition_probs(df25, c),
            "2026": transition_probs(df26, c),
        }

    fpcs25 = first_pitch_called_strike_rate(df25)
    fpcs26 = first_pitch_called_strike_rate(df26)
    print(f"[count_traffic] first-pitch called-strike rate 2025: {fpcs25:.4f}, 2026: {fpcs26:.4f}, delta {fpcs26-fpcs25:+.4f}")

    return {
        "first_pitch_called_strike_rate_2025": fpcs25,
        "first_pitch_called_strike_rate_2026": fpcs26,
        "first_pitch_called_strike_delta_pp": (fpcs26 - fpcs25) * 100,
        "decomp": decomp,
        "transitions": transitions,
    }


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2, default=str))

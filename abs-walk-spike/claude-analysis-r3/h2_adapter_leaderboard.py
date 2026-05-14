"""H2 — Named pitcher adaptation leaderboard with bootstrap stability filter.

Per-pitcher × per-week Bayesian:
  - Beta-Binomial for zone rate (Δ first-week vs last-week posterior of rate)
  - Beta-Binomial for top-share (z>3.0 ft fraction)
  - Dirichlet posterior for pitch-mix; week-over-week Jensen-Shannon divergence

Eligibility:
  - ≥200 pitches in 2026 Mar 27 – May 12 window
  - ≥3 weeks of data

Magnitude threshold (pre-registered):
  - |Δ zone rate| ≥ 15pp OR
  - |Δ top-share| ≥ 15pp OR
  - pitch-mix JSD ≥ 0.05

Bootstrap stability (pre-registered):
  - Resample game_pk N≥200; refit per pitcher; pitcher appears in top-15 by
    total shift magnitude in ≥80% of iterations.

If 0 names clear, that's a publishable finding.
"""
from __future__ import annotations

import json
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as scipy_beta

from common import (
    PITCH_GROUP_NAMES,
    R3_ARTIFACTS,
    R3_CHARTS,
    ensure_dirs,
    rulebook_zone_flag,
)
from data_prep_r3 import get_panel_2026

PITCH_THRESHOLD = 200
WEEKS_THRESHOLD = 3
DELTA_ZONE_THRESHOLD_PP = 15.0
DELTA_TOP_THRESHOLD_PP = 15.0
JSD_THRESHOLD = 0.05
STABILITY_THRESHOLD = 0.80  # appears in top-15 in ≥80% of bootstrap iters
TOP_N = 15
N_BOOTSTRAP = 200


def jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence between two probability vectors (log base 2 → range [0,1])."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * (np.log2(a[mask]) - np.log2(b[mask].clip(1e-12)))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def fit_pitcher_metrics(sub_pitcher: pd.DataFrame, pitcher_id: int, n_post: int = 1000) -> dict | None:
    """Build per-pitcher week-1 → week-last posterior deltas via Beta-Binomial
    conjugate sampling (zone rate, top-share) and Dirichlet posterior for
    pitch-mix JSD between first and last week.

    Returns None if <WEEKS_THRESHOLD weeks of data.
    """
    weeks = sorted(sub_pitcher["week"].unique())
    if len(weeks) < WEEKS_THRESHOLD:
        return None
    by_week = sub_pitcher.groupby("week").agg(
        n=("rulebook_zone", "size"),
        zone_n=("rulebook_zone", "sum"),
        top_n=("above_3", "sum"),
    ).reset_index()
    by_week = by_week.sort_values("week").reset_index(drop=True)
    # Drop weeks with very few pitches (<30) for stability
    by_week = by_week.loc[by_week["n"] >= 30].reset_index(drop=True)
    if len(by_week) < 2:
        return None

    rng = np.random.default_rng(pitcher_id)
    first = by_week.iloc[0]
    last = by_week.iloc[-1]
    z_first = rng.beta(first["zone_n"] + 1, first["n"] - first["zone_n"] + 1, size=n_post)
    z_last = rng.beta(last["zone_n"] + 1, last["n"] - last["zone_n"] + 1, size=n_post)
    t_first = rng.beta(first["top_n"] + 1, first["n"] - first["top_n"] + 1, size=n_post)
    t_last = rng.beta(last["top_n"] + 1, last["n"] - last["top_n"] + 1, size=n_post)
    d_zone = (z_last - z_first) * 100
    d_top = (t_last - t_first) * 100

    # Dirichlet posterior for pitch-mix; JSD between first/last week posterior means
    mix_first = (
        sub_pitcher.loc[sub_pitcher["week"] == first["week"], "pitch_group"]
        .value_counts()
        .reindex(PITCH_GROUP_NAMES, fill_value=0)
        .values.astype(float)
    )
    mix_last = (
        sub_pitcher.loc[sub_pitcher["week"] == last["week"], "pitch_group"]
        .value_counts()
        .reindex(PITCH_GROUP_NAMES, fill_value=0)
        .values.astype(float)
    )
    # Posterior mean is (counts + 1) / (sum + K)
    K = len(PITCH_GROUP_NAMES)
    p_first = (mix_first + 1) / (mix_first.sum() + K)
    p_last = (mix_last + 1) / (mix_last.sum() + K)
    jsd = jensen_shannon(p_first, p_last)

    return {
        "pitcher": int(pitcher_id),
        "n_pitches": int(sub_pitcher.shape[0]),
        "n_weeks": int(len(by_week)),
        "zone_first_mean": float(z_first.mean()),
        "zone_last_mean": float(z_last.mean()),
        "delta_zone_pp_mean": float(d_zone.mean()),
        "delta_zone_pp_lo": float(np.percentile(d_zone, 2.5)),
        "delta_zone_pp_hi": float(np.percentile(d_zone, 97.5)),
        "top_first_mean": float(t_first.mean()),
        "top_last_mean": float(t_last.mean()),
        "delta_top_pp_mean": float(d_top.mean()),
        "delta_top_pp_lo": float(np.percentile(d_top, 2.5)),
        "delta_top_pp_hi": float(np.percentile(d_top, 97.5)),
        "pitch_mix_jsd": float(jsd),
        "shift_magnitude": float(abs(d_zone.mean()) + abs(d_top.mean()) + 100 * jsd),
        "n_first_week": int(first["n"]),
        "n_last_week": int(last["n"]),
    }


def filter_eligible(panel: pd.DataFrame) -> list[int]:
    """Pitchers with ≥PITCH_THRESHOLD pitches and ≥3 weeks of data."""
    pc = panel.groupby("pitcher").size()
    return pc[pc >= PITCH_THRESHOLD].index.tolist()


def compute_leaderboard_iter(panel: pd.DataFrame, eligible: Iterable[int]) -> pd.DataFrame:
    """Return top-N pitchers by shift_magnitude for a single panel sample."""
    rows = []
    for pi in eligible:
        sub = panel.loc[panel["pitcher"] == pi]
        if len(sub) < PITCH_THRESHOLD:
            continue
        res = fit_pitcher_metrics(sub, int(pi), n_post=400)
        if res is None:
            continue
        rows.append(res)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("shift_magnitude", ascending=False)
    return df


def bootstrap_stability(
    panel: pd.DataFrame,
    eligible: list[int],
    n_iter: int = N_BOOTSTRAP,
    seed: int = 2026,
) -> dict[int, float]:
    """For each pitcher, fraction of bootstrap iterations they appear in top-N
    (by shift_magnitude). Game-level resample (game_pk).
    """
    rng = np.random.default_rng(seed)
    games = panel["game_pk"].astype("int64").values
    unique_games = np.unique(games)
    n_g = len(unique_games)
    # game -> row idx
    sort_order = np.argsort(games, kind="stable")
    sorted_g = games[sort_order]
    edges = np.searchsorted(sorted_g, unique_games)
    edges = np.r_[edges, len(games)]
    game_to_rows = {int(g): sort_order[edges[i]:edges[i + 1]] for i, g in enumerate(unique_games)}

    top_counts: dict[int, int] = {}
    t0 = time.time()
    print(f"[H2] bootstrap stability: {n_iter} iters, {len(eligible)} eligible pitchers")
    for it in range(n_iter):
        sampled = rng.choice(unique_games, size=n_g, replace=True)
        row_idx = np.concatenate([game_to_rows[int(g)] for g in sampled])
        sub_panel = panel.iloc[row_idx]
        # filter by eligibility within this iter as well
        local_pc = sub_panel.groupby("pitcher").size()
        local_eligible = local_pc[local_pc >= PITCH_THRESHOLD].index.intersection(eligible).tolist()
        lb = compute_leaderboard_iter(sub_panel, local_eligible)
        if len(lb) > 0:
            for pi in lb["pitcher"].head(TOP_N).values:
                top_counts[int(pi)] = top_counts.get(int(pi), 0) + 1
        if (it + 1) % 25 == 0 or it < 2:
            print(f"[H2] boot {it+1}/{n_iter}: elapsed={time.time()-t0:.1f}s", flush=True)
    return {pi: top_counts.get(pi, 0) / n_iter for pi in eligible}


def plot_adapter_leaderboard(named_leaders: pd.DataFrame, full_leader: pd.DataFrame, stability: dict[int, float]) -> None:
    if named_leaders.empty:
        # Render an empty plot stating the publishable null
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(
            0.5, 0.5,
            "No pitchers cleared both bootstrap stability (≥80%)\nAND magnitude threshold "
            "(|Δ zone rate|≥15pp OR |Δ top-share|≥15pp OR JSD≥0.05).\n\n"
            "Publishable finding: adaptation is heterogeneous; no individual\npitcher's shift is stable + large enough to name.",
            ha="center", va="center", fontsize=11, transform=ax.transAxes,
        )
        ax.set_title("R3-H2: Named adapter leaderboard — 0 names cleared filters")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(R3_CHARTS / "h2_adapter_leaderboard.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.4 * max(8, len(named_leaders)) + 1.5))
    metrics = [
        ("delta_zone_pp_mean", "Δ zone rate W1→Wlast (pp)", "delta_zone_pp_lo", "delta_zone_pp_hi"),
        ("delta_top_pp_mean", "Δ top-share W1→Wlast (pp)", "delta_top_pp_lo", "delta_top_pp_hi"),
        ("pitch_mix_jsd", "Pitch-mix JSD W1↔Wlast", None, None),
    ]
    named_leaders = named_leaders.sort_values("shift_magnitude", ascending=False).head(15).reset_index(drop=True)
    y = range(len(named_leaders))
    for ax, (col, lab, lo, hi) in zip(axes, metrics):
        vals = named_leaders[col].values
        colors = ["#27ae60" if v >= 0 else "#c0392b" for v in vals] if col != "pitch_mix_jsd" else ["#7f7fff"] * len(vals)
        if lo:
            ax.barh(
                y, vals,
                color=colors, edgecolor="black", linewidth=0.6,
                xerr=[vals - named_leaders[lo].values, named_leaders[hi].values - vals],
                capsize=3,
            )
        else:
            ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_yticks(y)
        labels = []
        for _, r in named_leaders.iterrows():
            stab = stability.get(int(r["pitcher"]), 0.0)
            labels.append(f"{r['name']}  (stab {stab*100:.0f}%, n={r['n_pitches']})")
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.4)
        ax.set_xlabel(lab)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("R3-H2: Named adapter leaderboard\n(≥200 pitches, ≥3 weeks; "
                 "magnitude threshold + bootstrap stability ≥80%)", y=1.02)
    fig.tight_layout()
    fig.savefig(R3_CHARTS / "h2_adapter_leaderboard.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(n_bootstrap: int = N_BOOTSTRAP) -> dict:
    ensure_dirs()
    panel = get_panel_2026()
    panel = panel.loc[panel["plate_x"].notna() & panel["plate_z"].notna()].copy()
    panel["rulebook_zone"] = rulebook_zone_flag(panel["plate_x"], panel["plate_z"])
    panel["above_3"] = (panel["plate_z"].astype(float) > 3.0).astype(int)
    eligible = filter_eligible(panel)
    print(f"[H2] {len(eligible)} pitchers ≥{PITCH_THRESHOLD} pitches in window")

    # Compute the canonical (full-panel) leaderboard FIRST
    full_lb = compute_leaderboard_iter(panel, eligible)
    # Apply magnitude threshold
    if not full_lb.empty:
        magnitude_pass = (
            (full_lb["delta_zone_pp_mean"].abs() >= DELTA_ZONE_THRESHOLD_PP)
            | (full_lb["delta_top_pp_mean"].abs() >= DELTA_TOP_THRESHOLD_PP)
            | (full_lb["pitch_mix_jsd"] >= JSD_THRESHOLD)
        )
        full_lb["magnitude_pass"] = magnitude_pass
    # Bootstrap stability
    stability = bootstrap_stability(panel, eligible, n_iter=n_bootstrap)
    full_lb["stability"] = full_lb["pitcher"].map(stability)
    # Names
    names = panel.groupby("pitcher")["player_name"].agg(
        lambda s: s.dropna().iloc[0] if len(s.dropna()) else f"id_{int(s.name)}"
    )
    full_lb["name"] = full_lb["pitcher"].map(names)

    # Final filter
    named = full_lb.loc[
        full_lb["magnitude_pass"] & (full_lb["stability"] >= STABILITY_THRESHOLD)
    ].copy()
    named = named.sort_values("shift_magnitude", ascending=False)
    print(f"[H2] {len(named)} pitchers clear BOTH magnitude AND stability filters.")
    plot_adapter_leaderboard(named, full_lb, stability)

    full_lb.to_parquet(R3_ARTIFACTS / "h2_full_pitcher_leaderboard.parquet", index=False)
    out = {
        "n_eligible": int(len(eligible)),
        "n_passed_magnitude": int(full_lb["magnitude_pass"].sum()) if "magnitude_pass" in full_lb.columns else 0,
        "n_named": int(len(named)),
        "thresholds": {
            "pitch_threshold": PITCH_THRESHOLD,
            "weeks_threshold": WEEKS_THRESHOLD,
            "delta_zone_pp": DELTA_ZONE_THRESHOLD_PP,
            "delta_top_pp": DELTA_TOP_THRESHOLD_PP,
            "jsd": JSD_THRESHOLD,
            "stability": STABILITY_THRESHOLD,
            "top_n": TOP_N,
            "n_bootstrap": N_BOOTSTRAP,
        },
        "named_leaders": named.head(15).to_dict(orient="records"),
        # Also include passes-magnitude-only ones for cross-method intersection downstream
        "magnitude_passers": (full_lb.loc[full_lb.get("magnitude_pass", False) == True]
                              .sort_values("shift_magnitude", ascending=False)
                              .head(25).to_dict(orient="records")
                              if "magnitude_pass" in full_lb.columns else []),
    }
    (R3_ARTIFACTS / "h2_adapter_leaderboard.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    main()

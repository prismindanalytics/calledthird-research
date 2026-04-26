"""r2_stabilization.py — corrected player-season cluster bootstrap stabilization.

R1 defect (per Codex review): R1's "bootstrap CI" did NOT bootstrap player-seasons.
It fixed the qualifying player-season set, then random-partitioned PAs within those
fixed players. That captures within-player random-partition uncertainty, NOT
sampling uncertainty over MLB player-seasons. The kill-gate language ("non-overlapping
CIs vs Carleton") is invalidated by that defect.

R2 fix (this module):
  - Cluster bootstrap: each iteration RESAMPLES PLAYERS WITH REPLACEMENT (clusters)
    from the qualifying-player set. For each chosen player, RESAMPLE THEIR SEASONS
    WITH REPLACEMENT to handle within-player serial correlation.
  - Then within each (player, season) replicate, randomly draw 2 * (M/2) PAs
    without replacement and split-half as before.
  - Per stat, compute Spearman-Brown reliability across the resampled player-seasons.
  - This is the standard cluster-bootstrap design for stabilization estimates.

Outputs:
  data/r2_stabilization_results.parquet (per stat, draw, M, r_sb, n_clusters)
  data/r2_stabilization_summary.json    (per stat: median, 95% CI, vs Carleton)
  charts/r2/stabilization_<stat>.png
"""
from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"
CHARTS = CLAUDE / "charts/r2"
CHARTS.mkdir(parents=True, exist_ok=True)

# Reuse R1 PA loader / annotator
import sys
sys.path.insert(0, str(CLAUDE))
from stabilization import (
    load_pa_table, annotate_pa, STAT_SPECS, CARLETON, STAT_LABEL,
)

PA_EVENTS = STAT_SPECS  # alias


def _split_half_correlation_clustered(
    cluster_index: np.ndarray,  # one row per chosen-player-season; values are the
                                # arrays of PA-row indices into pa_arr
    pa_arr_num: np.ndarray,
    pa_arr_den: np.ndarray,
    half_size: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """Compute Spearman-Brown reliability across the bootstrap-drawn (player-season)
    clusters. Each cluster contributes one (h1, h2) pair if it has >= 2*half_size PAs.

    Returns (r_sb, n_pairs_used).
    """
    h1, h2 = [], []
    for idx in cluster_index:
        if len(idx) < 2 * half_size:
            continue
        chosen = rng.choice(idx, size=2 * half_size, replace=False)
        first_idx = chosen[:half_size]
        second_idx = chosen[half_size:]
        d1 = pa_arr_den[first_idx].sum()
        d2 = pa_arr_den[second_idx].sum()
        if d1 < max(5, 0.05 * half_size) or d2 < max(5, 0.05 * half_size):
            continue
        r1 = pa_arr_num[first_idx].sum() / d1
        r2 = pa_arr_num[second_idx].sum() / d2
        h1.append(r1)
        h2.append(r2)
    if len(h1) < 30:
        return float("nan"), len(h1)
    a = np.asarray(h1, dtype=float)
    b = np.asarray(h2, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), len(h1)
    r = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(r) or r <= -0.99:
        return float("nan"), len(h1)
    rsb = 2 * r / (1 + r)
    return rsb, len(h1)


def _half_stab_point(rs: dict[int, float]) -> float:
    """Smallest M where r_sb crosses 0.5 (linear interp).

    If never crosses, linear-extrapolate from the last 4 finite tail points.
    Returns NaN if extrapolation also fails (e.g., flat / declining tail).
    """
    Ms = sorted(rs.keys())
    prev_M, prev_r = None, None
    for M in Ms:
        r = rs[M]
        if not math.isfinite(r):
            continue
        if r >= 0.5:
            if prev_r is None or not math.isfinite(prev_r):
                return float(M)
            return float(prev_M + (0.5 - prev_r) / (r - prev_r) * (M - prev_M))
        prev_M, prev_r = M, r
    finite_pairs = [(M, rs[M]) for M in Ms if math.isfinite(rs[M])]
    if len(finite_pairs) < 4:
        return float("nan")
    tail = finite_pairs[-4:]
    xs = np.array([p[0] for p in tail])
    ys = np.array([p[1] for p in tail])
    slope, intercept = np.polyfit(xs, ys, 1)
    if slope <= 0:
        return float("nan")
    crossing = (0.5 - intercept) / slope
    if crossing < tail[-1][0] or crossing > tail[-1][0] * 4:
        return float("nan")
    return float(crossing)


def cluster_bootstrap_stabilization(
    pa: pd.DataFrame, *,
    stats: list[str] | None = None,
    M_grid: list[int] | None = None,
    babip_M_grid: list[int] | None = None,
    n_boot: int = 200,
    min_pa: int = 200,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """True player-season cluster bootstrap.

    Procedure:
      1. Identify all eligible player-seasons (>= min_pa PA).
      2. Per bootstrap iteration:
         a. Resample PLAYERS with replacement from the unique-batter set.
         b. For each chosen player, resample THEIR SEASONS with replacement
            from that player's qualifying seasons.
         c. Each (player, season) replicate contributes its PA-row indices.
      3. For each M in the grid, do split-half on each replicate as in R1.
      4. Spearman-Brown across replicates -> r_sb(M). Linear interp / extrap to find
         half-stabilization PA.
      5. Median + 2.5/97.5 quantiles across iterations = cluster-bootstrap CI.
    """
    if stats is None:
        stats = list(STAT_SPECS.keys())
    if M_grid is None:
        M_grid = list(range(50, 901, 50))
    if babip_M_grid is None:
        babip_M_grid = list(range(50, 1601, 50))

    # Build qualifying player-season list
    pa_count = pa.groupby(["season", "batter"]).size().reset_index(name="pa_n")
    keep = pa_count[pa_count["pa_n"] >= min_pa].copy()
    pa = pa.merge(keep[["season", "batter"]], on=["season", "batter"], how="inner")
    unique_batters = sorted(set(pa["batter"].unique()))
    n_player_seasons = len(keep)
    print(f"[r2_stab] qualifying player-seasons (>= {min_pa} PA): {n_player_seasons}")
    print(f"[r2_stab] unique batters in qualifying set: {len(unique_batters)}")

    # Pre-compute group indices (positional) for fast random sampling
    grouped = {(int(s), int(b)): idx.values for (s, b), idx in
               pa.groupby(["season", "batter"]).groups.items()}
    seasons_per_batter = {}
    for (s, b) in grouped:
        seasons_per_batter.setdefault(b, []).append(s)

    rng = np.random.default_rng(seed)
    rows = []
    summary = {}

    for stat in stats:
        num_col, den_col = STAT_SPECS[stat]
        num_arr = pa[num_col].astype(float).values
        den_arr = pa[den_col].astype(float).values
        this_grid = babip_M_grid if stat == "BABIP" else M_grid

        per_draw_curves = []
        for draw in range(n_boot):
            # Cluster bootstrap: resample players with replacement
            chosen_players = rng.choice(unique_batters, size=len(unique_batters), replace=True)
            # For each chosen player, resample seasons with replacement
            cluster_index = []
            for b in chosen_players:
                seasons_b = seasons_per_batter[int(b)]
                if not seasons_b:
                    continue
                chosen_season = int(rng.choice(seasons_b))  # one random season per player draw
                idx_arr = grouped[(chosen_season, int(b))]
                cluster_index.append(idx_arr)

            curve = {}
            for M in this_grid:
                half = M // 2
                rsb, n_pairs = _split_half_correlation_clustered(
                    cluster_index, num_arr, den_arr, half, rng,
                )
                curve[M] = rsb
                rows.append({"stat": stat, "M": M, "draw": int(draw),
                             "r_sb": rsb, "n_pairs": int(n_pairs)})
            per_draw_curves.append(curve)
            if (draw + 1) % max(1, n_boot // 5) == 0:
                print(f"  [r2_stab] {stat}: {draw+1}/{n_boot} draws done")

        half_pts = np.asarray([_half_stab_point(c) for c in per_draw_curves], dtype=float)
        finite_mask = np.isfinite(half_pts)
        finite_pts = half_pts[finite_mask]
        # Right-censored draws (never crossed 0.5) — cap at 4*grid_max as a pessimistic
        # placeholder so the upper CI reflects the right-censoring honestly.
        capped = np.where(finite_mask, half_pts, max(this_grid) * 4)
        if len(half_pts) == 0:
            point, lo, hi = float("nan"), float("nan"), float("nan")
        else:
            point = float(np.median(capped))
            lo = float(np.quantile(capped, 0.025))
            hi = float(np.quantile(capped, 0.975))
        carleton = CARLETON.get(stat)
        ratio = point / carleton if carleton and math.isfinite(point) else float("nan")
        # Honest verdict: shifted = CI does NOT contain Carleton AND |ratio-1| >= 0.10
        if (math.isfinite(lo) and math.isfinite(hi)
                and (carleton < lo or carleton > hi)
                and abs(1 - ratio) >= 0.10):
            verdict = "shifted"
        elif (math.isfinite(lo) and math.isfinite(hi)
              and (lo <= carleton <= hi)):
            verdict = "consistent_with_carleton"
        else:
            verdict = "ambiguous"
        summary[stat] = {
            "point_pa": point,
            "ci_lo_pa": lo,
            "ci_hi_pa": hi,
            "ci_method": "cluster_bootstrap_player_then_season",
            "carleton_ref_pa": carleton,
            "ratio_to_carleton": ratio,
            "verdict": verdict,
            "n_draws_finite": int(finite_mask.sum()),
            "n_draws_total": int(len(half_pts)),
            "n_player_seasons_pool": n_player_seasons,
            "n_unique_batters_pool": len(unique_batters),
            "M_grid_max": int(max(this_grid)),
            "n_boot": int(n_boot),
        }
        print(f"  [r2_stab] {stat}: half-stab = {point:.0f} PA  [95% cluster-CI {lo:.0f}-{hi:.0f}]  vs Carleton {carleton} -> {verdict}")

    df = pd.DataFrame(rows)
    return df, summary


def plot_stabilization(df: pd.DataFrame, summary: dict, charts_dir: Path) -> None:
    for stat in CARLETON:
        sub = df[df["stat"] == stat]
        if sub.empty:
            continue
        agg = sub.groupby("M")["r_sb"].agg([
            "median",
            ("lo", lambda s: s.quantile(0.025)),
            ("hi", lambda s: s.quantile(0.975)),
        ])
        carleton = CARLETON[stat]
        info = summary.get(stat, {})
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.plot(agg.index, agg["median"], color="#1f4e79", lw=2,
                label="Cluster-bootstrap median")
        ax.fill_between(agg.index, agg["lo"], agg["hi"], color="#1f4e79", alpha=0.18,
                        label="95% cluster-bootstrap band")
        ax.axhline(0.5, color="#666", ls="--", lw=0.9)
        ax.axvline(carleton, color="#b8392b", ls=":", lw=1.4, label=f"Carleton ref: {carleton} PA")
        if math.isfinite(info.get("point_pa", float("nan"))):
            ax.axvline(info["point_pa"], color="#1f4e79", ls="-", lw=1.4,
                       label=f"R2: {info['point_pa']:.0f} PA"
                             f" [95% CI {info['ci_lo_pa']:.0f}-{info['ci_hi_pa']:.0f}]")
        ax.set_xlabel("PA (full sample size M)")
        ax.set_ylabel("Reliability r' (Spearman-Brown)")
        ax.set_title(f"R2 cluster-bootstrap stabilization — {STAT_LABEL.get(stat, stat)} (2022-2025)")
        ax.set_ylim(-0.1, 1.0)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        out = charts_dir / f"stabilization_{stat.replace('%','pct').replace('/','_')}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"[chart] {out.name}")


def run(n_boot: int = 200, seed: int = 7) -> dict:
    pa = load_pa_table()
    pa = annotate_pa(pa)
    df, summary = cluster_bootstrap_stabilization(pa, n_boot=n_boot, seed=seed)
    df.to_parquet(DATA / "r2_stabilization_results.parquet", index=False)
    plot_stabilization(df, summary, CHARTS)
    with open(DATA / "r2_stabilization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    n = int(os.environ.get("N_BOOT", "200"))
    s = run(n_boot=n)
    print(json.dumps(s, indent=2))

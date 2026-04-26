"""r3_stabilization.py — direct player-season bootstrap (Codex's R2 critique fix).

Codex's R2 critique: r2_stabilization.py resamples players, then takes ONE season
per sampled player per draw. That equal-weights players and throws away multi-season
exposure inside each draw. It's a changed estimand, not a true player-season bootstrap.

R3 fix:
  - Resample player-seasons DIRECTLY with replacement at the player-season level.
    Sample N player-seasons from the population of qualifying player-seasons
    (>= 200 PA in 2022-2025). Do NOT cap at one season per player; some player-
    seasons can appear multiple times in a draw, and a single batter may have
    multiple of their seasons present in the same draw.
  - Within each sampled player-season, do split-half on the PAs as before
    (random-half partition, then Spearman-Brown).
  - For each M in the grid, compute split-half correlation across all N draws of
    player-seasons. Per stat, gather draw-level r_sb -> distribution -> CI.

This is the textbook player-season cluster bootstrap when the estimand is
"reliability of player-season half-samples."

Outputs:
  data/r3_stabilization_results.parquet  (per stat, draw, M)
  data/r3_stabilization_summary.json     (per stat: median, 95% CI, vs Carleton)
  charts/r3/stabilization_<stat>.png
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"
CHARTS = CLAUDE / "charts/r3"
CHARTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CLAUDE))
from stabilization import (
    load_pa_table, annotate_pa, STAT_SPECS, CARLETON, STAT_LABEL,
)


def _split_half_correlation_player_season_draws(
    drawn_clusters: list,  # list of arrays of pa-row indices, one per drawn player-season
    pa_arr_num: np.ndarray,
    pa_arr_den: np.ndarray,
    half_size: int,
    rng: np.random.Generator,
) -> tuple[float, int]:
    """Spearman-Brown across drawn player-seasons; one (h1,h2) pair per draw if PA>=2*half."""
    h1, h2 = [], []
    for idx in drawn_clusters:
        if len(idx) < 2 * half_size:
            continue
        chosen = rng.choice(idx, size=2 * half_size, replace=False)
        first = chosen[:half_size]
        second = chosen[half_size:]
        d1 = pa_arr_den[first].sum()
        d2 = pa_arr_den[second].sum()
        if d1 < max(5, 0.05 * half_size) or d2 < max(5, 0.05 * half_size):
            continue
        r1 = pa_arr_num[first].sum() / d1
        r2 = pa_arr_num[second].sum() / d2
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
    """Smallest M where r_sb >= 0.5 (linear interp). Linear-extrapolate the tail
    if never crossed; return NaN if extrapolation also fails."""
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
    finite = [(M, rs[M]) for M in Ms if math.isfinite(rs[M])]
    if len(finite) < 4:
        return float("nan")
    tail = finite[-4:]
    xs = np.array([p[0] for p in tail])
    ys = np.array([p[1] for p in tail])
    slope, intercept = np.polyfit(xs, ys, 1)
    if slope <= 0:
        return float("nan")
    crossing = (0.5 - intercept) / slope
    if crossing < tail[-1][0] or crossing > tail[-1][0] * 4:
        return float("nan")
    return float(crossing)


def player_season_bootstrap(pa: pd.DataFrame, *,
                              stats: list[str] | None = None,
                              M_grid: list[int] | None = None,
                              babip_M_grid: list[int] | None = None,
                              n_boot: int = 200,
                              min_pa: int = 200,
                              seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """True player-season bootstrap.

    Procedure:
      1. Identify all eligible player-seasons (>= min_pa PA).
      2. Per draw: sample N player-seasons with replacement from the qualifying
         player-season population. (N = len(population), so each draw has the
         same number of player-seasons as the population.)
      3. Within each sampled player-season, do split-half PA partition for each M.
      4. Spearman-Brown across the drawn player-seasons -> r_sb(M).
      5. Linear interp -> half-stab point per draw. Median + 2.5/97.5 quantiles.
    """
    if stats is None:
        stats = list(STAT_SPECS.keys())
    if M_grid is None:
        M_grid = list(range(50, 901, 50))
    if babip_M_grid is None:
        babip_M_grid = list(range(50, 1601, 50))

    pa_count = pa.groupby(["season", "batter"]).size().reset_index(name="pa_n")
    keep = pa_count[pa_count["pa_n"] >= min_pa].copy()
    pa = pa.merge(keep[["season", "batter"]], on=["season", "batter"], how="inner")
    n_player_seasons = len(keep)
    print(f"[r3_stab] qualifying player-seasons (>= {min_pa} PA): {n_player_seasons}")

    # Pre-compute group indices (positional) per player-season
    grouped = {}
    for (s, b), idx in pa.groupby(["season", "batter"]).groups.items():
        grouped[(int(s), int(b))] = idx.values
    ps_keys = list(grouped.keys())  # length = n_player_seasons
    ps_keys_arr = np.arange(len(ps_keys))

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
            # Resample player-seasons with replacement
            chosen_idx = rng.choice(ps_keys_arr, size=len(ps_keys_arr), replace=True)
            drawn_clusters = [grouped[ps_keys[i]] for i in chosen_idx]

            curve = {}
            for M in this_grid:
                half = M // 2
                rsb, n_pairs = _split_half_correlation_player_season_draws(
                    drawn_clusters, num_arr, den_arr, half, rng,
                )
                curve[M] = rsb
                rows.append({"stat": stat, "M": M, "draw": int(draw),
                             "r_sb": rsb, "n_pairs": int(n_pairs)})
            per_draw_curves.append(curve)
            if (draw + 1) % max(1, n_boot // 5) == 0:
                print(f"  [r3_stab] {stat}: {draw+1}/{n_boot} draws done")

        half_pts = np.asarray([_half_stab_point(c) for c in per_draw_curves], dtype=float)
        finite_mask = np.isfinite(half_pts)
        capped = np.where(finite_mask, half_pts, max(this_grid) * 4)
        if len(half_pts) == 0:
            point, lo, hi = float("nan"), float("nan"), float("nan")
        else:
            point = float(np.median(capped))
            lo = float(np.quantile(capped, 0.025))
            hi = float(np.quantile(capped, 0.975))
        carleton = CARLETON.get(stat)
        ratio = point / carleton if carleton and math.isfinite(point) else float("nan")
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
            "point_pa": point, "ci_lo_pa": lo, "ci_hi_pa": hi,
            "ci_method": "player_season_bootstrap_direct",
            "carleton_ref_pa": carleton,
            "ratio_to_carleton": ratio,
            "verdict": verdict,
            "n_draws_finite": int(finite_mask.sum()),
            "n_draws_total": int(len(half_pts)),
            "n_player_seasons_pool": n_player_seasons,
            "M_grid_max": int(max(this_grid)),
            "n_boot": int(n_boot),
        }
        print(f"  [r3_stab] {stat}: half-stab = {point:.0f} PA  "
              f"[95% CI {lo:.0f}-{hi:.0f}]  vs Carleton {carleton}  -> {verdict}")

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
                label="Player-season bootstrap median")
        ax.fill_between(agg.index, agg["lo"], agg["hi"], color="#1f4e79", alpha=0.18,
                        label="95% bootstrap band")
        ax.axhline(0.5, color="#666", ls="--", lw=0.9)
        ax.axvline(carleton, color="#b8392b", ls=":", lw=1.4,
                   label=f"Carleton ref: {carleton} PA")
        if math.isfinite(info.get("point_pa", float("nan"))):
            ax.axvline(info["point_pa"], color="#1f4e79", ls="-", lw=1.4,
                       label=f"R3: {info['point_pa']:.0f} PA "
                             f"[95% CI {info['ci_lo_pa']:.0f}-{info['ci_hi_pa']:.0f}]")
        ax.set_xlabel("PA (full sample size M)")
        ax.set_ylabel("Reliability r' (Spearman-Brown)")
        ax.set_title(f"R3 player-season bootstrap stabilization — "
                     f"{STAT_LABEL.get(stat, stat)} (2022-2025)")
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
    df, summary = player_season_bootstrap(pa, n_boot=n_boot, seed=seed)
    df.to_parquet(DATA / "r3_stabilization_results.parquet", index=False)
    plot_stabilization(df, summary, CHARTS)
    json.dump(summary, open(DATA / "r3_stabilization_summary.json", "w"), indent=2)
    return summary


if __name__ == "__main__":
    n = int(os.environ.get("N_BOOT", "150"))
    s = run(n_boot=n)
    print(json.dumps(s, indent=2))

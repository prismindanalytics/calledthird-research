"""stabilization.py — bootstrap re-estimation of stabilization rates from 2022-2025 Statcast.

Method (split-half reliability, the standard route used by Carleton 2007/2013):
  - Take all hitter-seasons with >= 200 PA in 2022-2025.
  - For each target *full sample size* M in [50, 100, ..., 1200] PAs:
      * For each hitter-season with at least M PAs:
          - Random shuffle their PAs, split into two halves of M/2 each.
          - Compute stat (BB%, K%, BABIP, ISO, wOBA) on each half.
      * Pearson r between the two halves estimates reliability of an M/2-PA sample.
      * Spearman-Brown prophecy r_M = 2r/(1+r) projects to reliability of full M PAs.
      * Half-stabilization point: smallest M where r_M >= 0.5 (linearly interpolated).
  - Bootstrap 200 iterations of the random splits.
  - Report median + 95% bootstrap CI of half-stabilization point per stat.
  - Compare against Carleton 2007/2013 published values (referenced in baseball-research
    literature as the standard reference for the pre-deadened-ball era):
        BB%   ~ 120 PA
        K%    ~  60 PA
        ISO   ~ 160 PA
        BABIP ~ 820 PA
        wOBA  ~ 280 PA

Outputs:
  data/stabilization_results.parquet     (per stat: M, draw, r_sb)
  charts/stabilization_<stat>.png
  data/stabilization_summary.json
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
CHARTS = CLAUDE / "charts"
CHARTS.mkdir(exist_ok=True)

CARLETON = {"BB%": 120, "K%": 60, "ISO": 160, "BABIP": 820, "wOBA": 280}

STAT_LABEL = {
    "BB%": "Walk rate",
    "K%": "Strikeout rate",
    "ISO": "Isolated power",
    "BABIP": "BABIP",
    "wOBA": "wOBA",
}

PA_EVENTS = {
    "single", "double", "triple", "home_run", "walk", "intent_walk", "hit_by_pitch",
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "grounded_into_double_play",
    "fielders_choice", "fielders_choice_out", "field_error",
    "double_play", "triple_play", "sac_fly", "sac_fly_double_play",
    "sac_bunt", "sac_bunt_double_play",
    "catcher_interf",
}


def load_pa_table(seasons=(2022, 2023, 2024, 2025)) -> pd.DataFrame:
    """Build a per-PA table from cached season Statcast parquets."""
    frames = []
    for y in seasons:
        f = DATA / f"statcast_{y}.parquet"
        if not f.exists():
            f = DATA / f"statcast_{y}_full.parquet"
        if not f.exists():
            print(f"[skip] {y} not cached")
            continue
        df = pd.read_parquet(f, columns=[
            "game_date", "game_pk", "batter", "at_bat_number",
            "events", "woba_value", "woba_denom", "babip_value", "iso_value",
        ])
        df["season"] = y
        df = df[df["events"].notna() & df["events"].isin(PA_EVENTS)].copy()
        df = df.drop_duplicates(subset=["game_pk", "batter", "at_bat_number"]).reset_index(drop=True)
        frames.append(df)
    pa = pd.concat(frames, ignore_index=True)
    print(f"[load] {len(pa):,} PA across seasons {sorted(set(pa.season))}")
    return pa


def annotate_pa(pa: pd.DataFrame) -> pd.DataFrame:
    pa = pa.copy()
    pa["is_pa"] = 1
    pa["is_bb"] = pa["events"].isin({"walk", "intent_walk"}).astype(int)
    pa["is_k"] = pa["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
    pa["is_hbp"] = (pa["events"] == "hit_by_pitch").astype(int)
    pa["is_sac"] = pa["events"].isin({"sac_fly", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play"}).astype(int)
    pa["is_ab"] = (pa["is_pa"] & ~(pa["is_bb"] | pa["is_hbp"] | pa["is_sac"] | (pa["events"] == "catcher_interf"))).astype(int)
    pa["is_1b"] = (pa["events"] == "single").astype(int)
    pa["is_2b"] = (pa["events"] == "double").astype(int)
    pa["is_3b"] = (pa["events"] == "triple").astype(int)
    pa["is_hr"] = (pa["events"] == "home_run").astype(int)
    pa["is_hit"] = pa[["is_1b", "is_2b", "is_3b", "is_hr"]].sum(axis=1)
    pa["total_bases"] = pa["is_1b"] + 2 * pa["is_2b"] + 3 * pa["is_3b"] + 4 * pa["is_hr"]
    pa["babip_num"] = (pa["is_hit"] - pa["is_hr"]).clip(lower=0)
    pa["babip_den"] = (pa["is_ab"] - pa["is_k"] - pa["is_hr"] +
                       pa["events"].isin({"sac_fly", "sac_fly_double_play"}).astype(int)).clip(lower=0)
    pa["iso_num"] = (pa["total_bases"] - pa["is_hit"]).clip(lower=0)
    pa["iso_den"] = pa["is_ab"]
    pa["woba_num"] = pa["woba_value"].fillna(0.0).astype(float)
    pa["woba_den"] = pa["woba_denom"].fillna(0).astype(float)
    return pa


STAT_SPECS = {
    "BB%":  ("is_bb",   "is_pa"),
    "K%":   ("is_k",    "is_pa"),
    "ISO":  ("iso_num", "iso_den"),
    "BABIP":("babip_num","babip_den"),
    "wOBA": ("woba_num", "woba_den"),
}


def _split_half_correlation(grouped_indices: dict, pa_arr_num: np.ndarray, pa_arr_den: np.ndarray,
                             half_size: int, rng: np.random.Generator) -> float:
    """For each group with >= 2*half_size, randomly split into two halves of half_size,
    compute rate on each, then return Pearson r across groups."""
    h1, h2 = [], []
    for idx in grouped_indices.values():
        if len(idx) < 2 * half_size:
            continue
        chosen = rng.choice(idx, size=2 * half_size, replace=False)
        first_idx = chosen[:half_size]
        second_idx = chosen[half_size:]
        d1 = pa_arr_den[first_idx].sum()
        d2 = pa_arr_den[second_idx].sum()
        # require some signal in denominator
        if d1 < max(5, 0.05 * half_size) or d2 < max(5, 0.05 * half_size):
            continue
        r1 = pa_arr_num[first_idx].sum() / d1
        r2 = pa_arr_num[second_idx].sum() / d2
        h1.append(r1)
        h2.append(r2)
    if len(h1) < 30:
        return float("nan")
    a = np.asarray(h1, dtype=float)
    b = np.asarray(h2, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    r = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(r) or r <= -0.99:
        return float("nan")
    rsb = 2 * r / (1 + r)  # Spearman-Brown -> full M = 2*half_size
    return rsb


def _half_stab_point(rs: dict[int, float]) -> float:
    """Return smallest M where r_sb crosses 0.5 (linearly interpolated).

    If the curve never reaches 0.5 within the grid, *extrapolate linearly* from the
    last 4 finite points to the projected crossing — this avoids the cap-at-grid-max
    artifact that inflates the upper CI."""
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
    # Never crossed 0.5 — extrapolate linearly from tail
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


def bootstrap_stabilization(pa: pd.DataFrame, *,
                            stats: list[str] = None,
                            M_grid: list[int] = None,
                            babip_M_grid: list[int] = None,
                            n_boot: int = 200,
                            min_pa: int = 200,
                            seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """Bootstrap stabilization estimation.

    M_grid: full sample sizes (PAs). Halves are M/2 each.
    babip_M_grid: separate (extended) grid for BABIP — Carleton's reference is 820 PA
                  so we need to push the grid wider to capture the crossing reliably.
    """
    if stats is None:
        stats = list(STAT_SPECS.keys())
    if M_grid is None:
        M_grid = list(range(50, 901, 50))  # standard
    if babip_M_grid is None:
        # BABIP: extend further
        babip_M_grid = list(range(50, 1601, 50))

    pa_count = pa.groupby(["season", "batter"]).size().reset_index(name="pa_n")
    keep = pa_count[pa_count["pa_n"] >= min_pa]
    pa = pa.merge(keep[["season", "batter"]], on=["season", "batter"], how="inner")
    n_player_seasons = len(keep)
    print(f"[stab] qualifying player-seasons (>= {min_pa} PA): {n_player_seasons}")

    # Pre-compute group indices (positional) for fast random sampling
    grouped = {(int(s), int(b)): idx.values for (s, b), idx in pa.groupby(["season", "batter"]).groups.items()}

    rng = np.random.default_rng(seed)
    rows = []
    summary = {}

    for stat in stats:
        num_col, den_col = STAT_SPECS[stat]
        num_arr = pa[num_col].astype(float).values
        den_arr = pa[den_col].astype(float).values

        # Per-stat M grid; BABIP gets extended.
        this_grid = babip_M_grid if stat == "BABIP" else M_grid

        per_draw_curves = []
        for draw in range(n_boot):
            curve = {}
            for M in this_grid:
                half = M // 2
                rsb = _split_half_correlation(grouped, num_arr, den_arr, half, rng)
                curve[M] = rsb
                rows.append({"stat": stat, "M": M, "draw": int(draw), "r_sb": rsb})
            per_draw_curves.append(curve)
            if (draw + 1) % max(1, n_boot // 5) == 0:
                print(f"  [stab] {stat}: {draw+1}/{n_boot} draws done")

        half_pts = np.asarray([_half_stab_point(c) for c in per_draw_curves], dtype=float)
        # Cap any remaining infinite/NaN draws at 4x the grid max — this signals the
        # curve plateaued below 0.5 and the true crossing is far beyond what we can see.
        capped = np.where(np.isfinite(half_pts), half_pts, max(this_grid) * 4)
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
        else:
            verdict = "consistent_with_carleton"
        summary[stat] = {
            "point_pa": point,
            "ci_lo_pa": lo,
            "ci_hi_pa": hi,
            "carleton_ref_pa": carleton,
            "ratio_to_carleton": ratio,
            "verdict": verdict,
            "n_draws_finite": int(np.sum(np.isfinite(half_pts))),
            "n_draws_total": int(len(half_pts)),
            "n_player_seasons": n_player_seasons,
            "M_grid_max": int(max(this_grid)),
        }
        print(f"  [stab] {stat}: half-stab = {point:.0f} PA  [95% CI {lo:.0f}-{hi:.0f}]  vs Carleton {carleton} -> {verdict}")

    df = pd.DataFrame(rows)
    return df, summary


def plot_stabilization(df: pd.DataFrame, summary: dict, charts_dir: Path) -> None:
    for stat, _ in CARLETON.items():
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
                label="Bootstrap median (Spearman-Brown)")
        ax.fill_between(agg.index, agg["lo"], agg["hi"], color="#1f4e79", alpha=0.18,
                        label="95% bootstrap band")
        ax.axhline(0.5, color="#666", ls="--", lw=0.9)
        ax.axvline(carleton, color="#b8392b", ls=":", lw=1.4, label=f"Carleton ref: {carleton} PA")
        if math.isfinite(info.get("point_pa", float("nan"))):
            ax.axvline(info["point_pa"], color="#1f4e79", ls="-", lw=1.4,
                       label=f"This study: {info['point_pa']:.0f} PA"
                             f" [95% CI {info['ci_lo_pa']:.0f}-{info['ci_hi_pa']:.0f}]")
        ax.set_xlabel("PA (full sample size M)")
        ax.set_ylabel("Reliability r' (Spearman-Brown)")
        ax.set_title(f"Stabilization curve — {STAT_LABEL.get(stat, stat)} (2022-2025)")
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
    df, summary = bootstrap_stabilization(pa, n_boot=n_boot, seed=seed)
    df.to_parquet(DATA / "stabilization_results.parquet", index=False)
    plot_stabilization(df, summary, CHARTS)
    with open(DATA / "stabilization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    n = int(os.environ.get("N_BOOT", "200"))
    s = run(n_boot=n)
    print(json.dumps(s, indent=2))

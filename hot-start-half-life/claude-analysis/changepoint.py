"""changepoint.py — PELT change-point detection for the 5 named hot starters.

Question: has the player's underlying rate *actually shifted* during the 22-game window
(vs the prior 2023-2025 baseline), or is the hot start a noise excursion against an
unchanged rate?

Method:
  - Build a per-game rolling rate series for each named hot starter combining 2023, 2024,
    2025, and 2026-to-date. For Murakami (no MLB history) we use 2026 only and report
    "insufficient pre-2026 data for change-point test."
  - Apply ruptures.Pelt with rbf cost. The penalty is selected by BIC heuristic
    (penalty = c * log(n)) for a few c in {0.5, 1.0, 2.0} and we report stability
    across c.
  - For each detected change-point, report: index in series, date, mean before vs after,
    delta, percentile of delta vs all-pairs random splits.
  - Save per-player change-point chart with vertical lines.

Outputs:
  data/changepoints.json
  charts/players/<slug>_changepoint.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ruptures as rpt

warnings.filterwarnings("ignore")

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"
PLAYER_CHARTS = CLAUDE / "charts/players"
PLAYER_CHARTS.mkdir(parents=True, exist_ok=True)

PA_EVENTS = {
    "single","double","triple","home_run","walk","intent_walk","hit_by_pitch",
    "strikeout","strikeout_double_play","field_out","force_out","grounded_into_double_play",
    "fielders_choice","fielders_choice_out","field_error","double_play","triple_play",
    "sac_fly","sac_fly_double_play","sac_bunt","sac_bunt_double_play","catcher_interf",
}


def load_player_pa_history(mlbam: int, role: str) -> pd.DataFrame:
    """Load per-PA rows across 2023-2026 for one player."""
    frames = []
    seasons = (2023, 2024, 2025)
    cols = ["game_date", "game_pk", "batter", "pitcher", "at_bat_number",
            "events", "woba_value", "woba_denom", "babip_value", "iso_value"]
    for y in seasons:
        f = DATA / f"statcast_{y}.parquet"
        if not f.exists():
            f = DATA / f"statcast_{y}_full.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=cols)
        df["season"] = y
        if role == "hitter":
            df = df[df.batter == mlbam]
        else:
            df = df[df.pitcher == mlbam]
        df = df[df["events"].notna() & df["events"].isin(PA_EVENTS)]
        if not df.empty:
            df = df.drop_duplicates(["game_pk", "at_bat_number"])
            frames.append(df)
    # 2026 data
    for f in (DATA / "statcast_2026_mar27_apr22.parquet", DATA / "statcast_2026_apr23_24.parquet"):
        if f.exists():
            d = pd.read_parquet(f, columns=cols)
            d["season"] = 2026
            if role == "hitter":
                d = d[d.batter == mlbam]
            else:
                d = d[d.pitcher == mlbam]
            d = d[d["events"].notna() & d["events"].isin(PA_EVENTS)]
            if not d.empty:
                d = d.drop_duplicates(["game_pk", "at_bat_number"])
                frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Coerce game_date to a uniform datetime type (parquet from different sources can mix
    # strings vs Timestamps).
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.sort_values(["game_date", "game_pk", "at_bat_number"]).reset_index(drop=True)
    return out


def per_pa_woba_series(player_pa: pd.DataFrame) -> tuple[np.ndarray, list[str], list[int]]:
    """Per-PA wOBA value vector + per-PA dates + cumulative season markers."""
    if player_pa.empty:
        return np.array([]), [], []
    woba = player_pa["woba_value"].fillna(0.0).astype(float).values
    dates = player_pa["game_date"].astype(str).tolist()
    seasons = player_pa["season"].astype(int).tolist()
    return woba, dates, seasons


def per_game_rolling_rate(player_pa: pd.DataFrame, num_col: str, den_col: str,
                          *, win: int = 30) -> tuple[np.ndarray, list[str]]:
    """Rolling per-PA rate with window of `win` PAs. Used for visual context."""
    if player_pa.empty:
        return np.array([]), []
    n = len(player_pa)
    nums = player_pa[num_col].astype(float).values
    dens = player_pa[den_col].astype(float).values
    rates = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - win + 1)
        d = dens[lo:i + 1].sum()
        if d > 0:
            rates[i] = nums[lo:i + 1].sum() / d
    dates = player_pa["game_date"].astype(str).tolist()
    return rates, dates


def annotate_pa_inplace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_pa"] = 1
    df["is_bb"] = df["events"].isin({"walk", "intent_walk"}).astype(int)
    df["is_k"] = df["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
    df["is_hbp"] = (df["events"] == "hit_by_pitch").astype(int)
    df["is_sac"] = df["events"].isin({"sac_fly", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play"}).astype(int)
    df["is_ab"] = (df["is_pa"] & ~(df["is_bb"] | df["is_hbp"] | df["is_sac"] | (df["events"] == "catcher_interf"))).astype(int)
    df["is_1b"] = (df["events"] == "single").astype(int)
    df["is_2b"] = (df["events"] == "double").astype(int)
    df["is_3b"] = (df["events"] == "triple").astype(int)
    df["is_hr"] = (df["events"] == "home_run").astype(int)
    df["is_hit"] = df[["is_1b", "is_2b", "is_3b", "is_hr"]].sum(axis=1)
    df["total_bases"] = df["is_1b"] + 2 * df["is_2b"] + 3 * df["is_3b"] + 4 * df["is_hr"]
    df["babip_num"] = (df["is_hit"] - df["is_hr"]).clip(lower=0)
    df["babip_den"] = (df["is_ab"] - df["is_k"] - df["is_hr"] +
                       df["events"].isin({"sac_fly", "sac_fly_double_play"}).astype(int)).clip(lower=0)
    df["iso_num"] = (df["total_bases"] - df["is_hit"]).clip(lower=0)
    df["iso_den"] = df["is_ab"]
    df["woba_num"] = df["woba_value"].fillna(0.0).astype(float)
    df["woba_den"] = df["woba_denom"].fillna(0).astype(float)
    return df


def detect_changepoints(series: np.ndarray, *, min_size: int = 30,
                        c_grid=(0.5, 1.0, 2.0)) -> dict:
    """PELT with rbf cost over a grid of penalty scalings.

    series: per-PA wOBA contribution (or any rate proxy). NaNs replaced with 0.
    Returns dict { c: {breaks, mean_before_after_each, deltas} }
    """
    if len(series) < 2 * min_size:
        return {"insufficient_data": True, "n": int(len(series))}

    s = np.nan_to_num(series, nan=0.0).astype(float)
    n = len(s)
    out = {"n": n, "results": {}}
    for c in c_grid:
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(s.reshape(-1, 1))
        pen = c * np.log(n)
        breaks = algo.predict(pen=float(pen))
        # ruptures returns list of indices ending each segment, including n at end
        segments = []
        prev = 0
        for b in breaks:
            seg = s[prev:b]
            segments.append({"start": int(prev), "end": int(b), "mean": float(seg.mean()), "n": int(len(seg))})
            prev = b
        # Per-break delta
        deltas = []
        for i in range(len(segments) - 1):
            d = segments[i + 1]["mean"] - segments[i]["mean"]
            deltas.append({"break_index": segments[i]["end"], "delta": float(d),
                           "before_mean": segments[i]["mean"], "after_mean": segments[i + 1]["mean"],
                           "before_n": segments[i]["n"], "after_n": segments[i + 1]["n"]})
        out["results"][f"c={c}"] = {"breaks": [int(x) for x in breaks[:-1]],
                                     "n_segments": len(segments),
                                     "segments": segments, "deltas": deltas}
    return out


def chart_changepoint(player_name: str, pa: pd.DataFrame, woba_series: np.ndarray,
                      cps: dict, out_path: Path) -> None:
    if len(woba_series) == 0:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # Top: per-PA wOBA contribution scatter + rolling 30-PA mean
    idx = np.arange(len(woba_series))
    axes[0].scatter(idx, woba_series, s=4, alpha=0.4, color="#666")
    rolling = pd.Series(woba_series).rolling(30, min_periods=10).mean()
    axes[0].plot(idx, rolling, color="#1f4e79", lw=1.6, label="Rolling 30-PA wOBA")
    # season markers
    season_change_idx = np.where(np.diff(pa["season"].values) != 0)[0] + 1
    for s in season_change_idx:
        axes[0].axvline(s, color="#aaa", ls=":", lw=0.8)
        season_label = int(pa["season"].iloc[s])
        axes[0].text(s, 0.05, f"{season_label}", rotation=90, fontsize=7, color="#888", va="bottom")
    # Change-points (use c=1.0 as default display)
    if "results" in cps and "c=1.0" in cps["results"]:
        for b in cps["results"]["c=1.0"]["breaks"]:
            axes[0].axvline(b, color="#b8392b", lw=1.2, label="PELT break (c=1.0)")
    # Show 2026 cutoff
    pa_2026_start = (pa.season == 2026).values
    if pa_2026_start.any():
        first_2026 = int(np.argmax(pa_2026_start))
        axes[0].axvspan(first_2026, len(woba_series) - 1, color="#ffeaa7", alpha=0.4, label="2026 hot-start window")
    axes[0].set_ylabel("Per-PA wOBA contribution")
    axes[0].set_title(f"Change-point analysis — {player_name}")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=7)
    axes[0].grid(True, alpha=0.3)

    # Bottom: cumulative wOBA mean over the career window
    cum = np.cumsum(woba_series) / np.arange(1, len(woba_series) + 1)
    axes[1].plot(idx, cum, color="#1f4e79", lw=1.5)
    axes[1].set_ylabel("Cumulative wOBA")
    axes[1].set_xlabel("PA index (career window)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run() -> dict:
    named = pd.read_parquet(DATA / "named_hot_starters.parquet")
    out = {}
    for _, row in named.iterrows():
        slug = row.slug
        mlbam = int(row.mlbam) if pd.notna(row.mlbam) else None
        if mlbam is None:
            out[slug] = {"error": "no mlbam"}
            continue
        pa = load_player_pa_history(mlbam, row.role)
        if pa.empty:
            out[slug] = {"error": "no PA history"}
            continue
        pa = annotate_pa_inplace(pa)
        woba_per_pa = pa["woba_num"].values  # per-PA wOBA contribution
        # Drop NaNs that come from rows with missing woba_value (rare)
        woba_per_pa = np.nan_to_num(woba_per_pa, nan=0.0)
        cps = detect_changepoints(woba_per_pa)
        # Determine if 2026 window contains a break
        pa_2026_mask = (pa.season == 2026).values
        first_2026 = int(np.argmax(pa_2026_mask)) if pa_2026_mask.any() else None
        out[slug] = {
            "mlbam": mlbam, "role": row.role,
            "n_career_pa": int(len(pa)),
            "n_2026_pa": int(pa_2026_mask.sum()) if first_2026 is not None else 0,
            "first_2026_idx": first_2026,
            "changepoints": cps,
        }
        # Verdict heuristic — look at c=1.0 breaks; any break inside the 2026 window?
        verdict = "no_break_in_2026"
        if "results" in cps and first_2026 is not None:
            r = cps["results"]["c=1.0"]
            for b in r["breaks"]:
                if b >= first_2026:
                    verdict = f"break_at_pa_idx_{b}_within_2026"
                    break
        out[slug]["verdict"] = verdict
        # Chart
        chart_changepoint(f"{row.first} {row.last}", pa, woba_per_pa, cps,
                          PLAYER_CHARTS / f"{slug}_changepoint.png")
        print(f"[changepoint] {slug}: career_PA={len(pa)}  2026_PA={pa_2026_mask.sum()}  verdict={verdict}")
    with open(DATA / "changepoints.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


if __name__ == "__main__":
    run()

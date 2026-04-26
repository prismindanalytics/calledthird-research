"""r2_persistence_atlas.py — Persistence Atlas (Framing A): universe-wide ranking +
sleeper / fake-hot / fake-cold lists.

Consumes:
  - data/r2_universe_posteriors.parquet (per-hitter posterior + ROS-vs-prior delta)
  - data/r2_hitter_universe.parquet (universe metadata)
  - data/mainstream_top20.json (mainstream coverage reference list)

Produces:
  - data/r2_persistence_atlas.json with sleepers / fake_hot / fake_cold lists
  - charts/r2/sleepers.png, fake_hot.png, fake_cold.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"
CHARTS = CLAUDE / "charts/r2"
CHARTS.mkdir(parents=True, exist_ok=True)


def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return (s.lower()
              .replace(",", " ")
              .replace(".", "")
              .replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
              .replace("ñ", "n").replace("ü", "u")
              .strip())


def load_mainstream_set() -> tuple[set, set, dict]:
    """Returns (mainstream_names_set, mainstream_mlbam_set, raw_dict).

    Supports two schemas: (a) Claude's R1-style with multiple sources, and
    (b) Codex's R2-style with a flat 'players' list (preferred).
    """
    d = json.load(open(DATA / "mainstream_top20.json"))
    names = set()
    mlbam = set()
    if "players" in d:
        for r in d["players"]:
            if r.get("mlbam"):
                mlbam.add(int(r["mlbam"]))
            if r.get("name"):
                names.add(_norm_name(r["name"]))
    # legacy schema fallbacks
    for r in d.get("fangraphs_top25_woba_50pa", []):
        if r.get("mlbam"):
            mlbam.add(int(r["mlbam"]))
        if r.get("name"):
            names.add(_norm_name(r["name"]))
    for r in d.get("mlb_dot_com_named_starters", []):
        if r.get("mlbam"):
            mlbam.add(int(r["mlbam"]))
        if r.get("name"):
            names.add(_norm_name(r["name"]))
    for r in d.get("espn_named_starters", []):
        if r.get("mlbam"):
            mlbam.add(int(r["mlbam"]))
        if r.get("name"):
            names.add(_norm_name(r["name"]))
    for n in d.get("mainstream_top20_set_for_classification", []):
        names.add(_norm_name(n))
    return names, mlbam, d


def is_mainstream(player_name: str, batter_id: int,
                   names_set: set, mlbam_set: set) -> bool:
    if int(batter_id) in mlbam_set:
        return True
    last_first = _norm_name(player_name)
    if last_first in names_set:
        return True
    # Try First Last form
    parts = [p for p in last_first.replace(",", "").split() if p]
    if len(parts) >= 2:
        # Convert "rice ben" -> "ben rice"
        first_last = " ".join(parts[1:] + parts[:1])
        if first_last in names_set:
            return True
        # Also try direct comma-stripped
        if " ".join(parts) in names_set:
            return True
    return False


def build_atlas() -> dict:
    posts = pd.read_parquet(DATA / "r2_universe_posteriors.parquet")
    universe = pd.read_parquet(DATA / "r2_hitter_universe.parquet")
    posts = posts.merge(universe[["batter", "PA"]].rename(columns={"PA": "PA_22g_check"}),
                        on="batter", how="left")

    names_set, mlbam_set, raw = load_mainstream_set()
    posts["is_mainstream"] = posts.apply(
        lambda r: is_mainstream(r.player_name, r.batter, names_set, mlbam_set),
        axis=1,
    )

    # Required columns
    posts = posts.dropna(subset=["ROS_wOBA_minus_prior_q50"]).copy()

    # Top decile of predicted ROS-vs-prior delta
    decile_top_q90 = posts["ROS_wOBA_minus_prior_q50"].quantile(0.90)
    decile_bot_q10 = posts["ROS_wOBA_minus_prior_q50"].quantile(0.10)

    # H1 sleepers: top decile of ROS delta AND not in mainstream
    sleepers = posts[(posts["ROS_wOBA_minus_prior_q50"] >= decile_top_q90)
                     & (~posts["is_mainstream"])].sort_values(
        "ROS_wOBA_minus_prior_q50", ascending=False)

    # H2 fake hots: in mainstream top-20 AND ROS delta < 0
    fake_hot = posts[posts["is_mainstream"] & (posts["ROS_wOBA_minus_prior_q50"] < 0)].sort_values(
        "ROS_wOBA_minus_prior_q50", ascending=True)

    # H3 fake colds: bottom decile of April performance (obs_wOBA) AND ROS delta > 0
    apr_decile = posts["obs_wOBA"].quantile(0.10)
    fake_cold = posts[(posts["obs_wOBA"] <= apr_decile)
                      & (posts["ROS_wOBA_minus_prior_q50"] > 0)].sort_values(
        "ROS_wOBA_minus_prior_q50", ascending=False)

    out = {
        "thresholds": {
            "delta_decile_top_q90": float(decile_top_q90),
            "delta_decile_bot_q10": float(decile_bot_q10),
            "april_decile_q10_wOBA": float(apr_decile),
            "n_universe": int(len(posts)),
            "n_mainstream_in_universe": int(posts["is_mainstream"].sum()),
        },
        "sleepers": _df_to_records(sleepers.head(15)),
        "fake_hot": _df_to_records(fake_hot.head(15)),
        "fake_cold": _df_to_records(fake_cold.head(15)),
        "h1_pass": bool(len(sleepers) >= 3),
        "h2_pass": bool(len(fake_hot) >= 3),
        "h3_pass": bool(len(fake_cold) >= 3),
        "n_sleepers": int(len(sleepers)),
        "n_fake_hot": int(len(fake_hot)),
        "n_fake_cold": int(len(fake_cold)),
        "mainstream_source_summary": (
            [raw.get("source_name", "?") + " — " + raw.get("source_url", "?")]
            if "players" in raw
            else [s["label"] for s in raw.get("sources", [])]
        ),
        "mainstream_top20_n": int(len(mlbam_set) + len(names_set)),
    }
    json.dump(out, open(DATA / "r2_persistence_atlas.json", "w"), indent=2)

    # Charts
    _bar_chart(sleepers.head(10), "Top-10 SLEEPER signals — predicted ROS wOBA delta", "sleepers")
    _bar_chart(fake_hot.head(10), "Top-10 FAKE HOT — mainstream top-20 with predicted ROS delta < 0", "fake_hot")
    _bar_chart(fake_cold.head(10), "Top-10 FAKE COLD — bottom decile April with predicted ROS delta > 0", "fake_cold")
    return out


def _df_to_records(df: pd.DataFrame) -> list:
    cols = ["batter", "player_name", "prior_kind", "PA_22g",
            "obs_wOBA", "obs_xwOBA", "obs_HardHitPct", "obs_BarrelPct", "obs_EV_p90",
            "prior_wOBA",
            "ROS_wOBA_q10", "ROS_wOBA_q50", "ROS_wOBA_q90",
            "ROS_wOBA_minus_prior_q10", "ROS_wOBA_minus_prior_q50", "ROS_wOBA_minus_prior_q90",
            "post_BBpct_q50", "post_Kpct_q50", "post_BABIP_q50",
            "post_ISO_q50", "post_xwOBA_q50",
            "post_HardHitPct_q50", "post_BarrelPct_q50", "post_EV_p90_q50",
            "is_mainstream"]
    return df[[c for c in cols if c in df.columns]].to_dict(orient="records")


def _bar_chart(df: pd.DataFrame, title: str, slug: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(df))
    delta_q50 = df["ROS_wOBA_minus_prior_q50"].values
    err_low = delta_q50 - df["ROS_wOBA_minus_prior_q10"].values
    err_high = df["ROS_wOBA_minus_prior_q90"].values - delta_q50
    colors = ["#1f4e79" if v >= 0 else "#b8392b" for v in delta_q50]
    ax.barh(y, delta_q50, color=colors, alpha=0.85)
    ax.errorbar(delta_q50, y, xerr=[err_low, err_high], fmt="none",
                ecolor="#444", capsize=3, lw=1.0)
    ax.set_yticks(y)
    labels = [f"{r.player_name}  (PA {r.PA_22g}, obs wOBA {r.obs_wOBA:.3f})"
              for _, r in df.iterrows()]
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Predicted ROS wOBA delta vs 3yr prior (q10/q50/q90)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(CHARTS / f"{slug}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    s = build_atlas()
    print(json.dumps({k: v for k, v in s.items() if k not in ("sleepers", "fake_hot", "fake_cold")},
                     indent=2))
    print(f"\nTOP-5 sleepers:")
    for r in s["sleepers"][:5]:
        print(f"  {r['player_name']:30s}  PA={r['PA_22g']:3d}  obs.wOBA={r['obs_wOBA']:.3f}  ROS-vs-prior q50={r['ROS_wOBA_minus_prior_q50']:+.3f}")
    print(f"\nTOP-5 fake-hot:")
    for r in s["fake_hot"][:5]:
        print(f"  {r['player_name']:30s}  PA={r['PA_22g']:3d}  obs.wOBA={r['obs_wOBA']:.3f}  ROS-vs-prior q50={r['ROS_wOBA_minus_prior_q50']:+.3f}")
    print(f"\nTOP-5 fake-cold:")
    for r in s["fake_cold"][:5]:
        print(f"  {r['player_name']:30s}  PA={r['PA_22g']:3d}  obs.wOBA={r['obs_wOBA']:.3f}  ROS-vs-prior q50={r['ROS_wOBA_minus_prior_q50']:+.3f}")

"""r2_reliever_board.py — Framing C: reliever K% True-Talent Board.

Per brief:
  - Build per-reliever K% posterior using reliever-specific stabilization rates
    (Carleton: K% ~ 70 BF for relievers).
  - Anchor prior to 2022-2025 weighted mean K% (or league average for debuts).
  - Rank by posterior K% q50 vs prior K%.

Outputs:
  - Top-5 SLEEPER reliever K% risers: posterior q50 K% materially above prior, NOT
    a known closer (use top-30 saves leaders 2025).
  - Top-5 FAKE-DOMINANT reliever K%: high April K% but posterior shrinks heavily
    back to prior.

  data/r2_reliever_board.json
  charts/r2/sleeper_relievers.png
  charts/r2/fake_dominant_relievers.png
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
CHARTS = CLAUDE / "charts/r2"
CHARTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CLAUDE))
from stabilization import PA_EVENTS

RELIEVER_STAB = {"BB%": 170, "K%": 70}


def load_pitcher_pa_priors(seasons=(2023, 2024, 2025)) -> pd.DataFrame:
    """Load per-pitcher 2023-2025 weighted-mean K% / BB% priors."""
    frames = []
    for y in seasons:
        f = DATA / f"statcast_{y}.parquet"
        if not f.exists():
            f = DATA / f"statcast_{y}_full.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=["game_pk", "pitcher", "at_bat_number", "events"])
        df["season"] = y
        df = df[df["events"].notna() & df["events"].isin(PA_EVENTS)]
        df = df.drop_duplicates(subset=["game_pk", "pitcher", "at_bat_number"])
        df["is_pa"] = 1
        df["is_bb"] = df["events"].isin({"walk", "intent_walk"}).astype(int)
        df["is_k"] = df["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
        frames.append(df)
    pa = pd.concat(frames, ignore_index=True)
    weights = {2025: 5, 2024: 4, 2023: 3}
    rows = []
    for pid, sub in pa.groupby("pitcher"):
        bf_w = 0.0; bb_w = 0.0; k_w = 0.0; total_bf = 0
        seasons_used = []
        for s, w in weights.items():
            ssub = sub[sub.season == s]
            if len(ssub) == 0:
                continue
            seasons_used.append(s)
            bf_w += w * len(ssub)
            bb_w += w * ssub["is_bb"].sum()
            k_w += w * ssub["is_k"].sum()
            total_bf += len(ssub)
        if bf_w <= 0:
            continue
        rows.append({
            "pitcher": int(pid),
            "prior_BB%": bb_w / bf_w,
            "prior_K%": k_w / bf_w,
            "prior_BF_total": int(total_bf),
            "seasons_used": seasons_used,
        })
    return pd.DataFrame(rows)


def conjugate_beta_binomial(rate_prior: float, eff_pa_prior: int,
                              obs_num: float, obs_den: float, *,
                              n_post_samples: int = 4000,
                              seed: int = 0) -> dict:
    rate_prior = float(np.clip(rate_prior, 1e-3, 1 - 1e-3))
    eff_pa_prior = float(max(eff_pa_prior, 5))
    a = max(rate_prior * eff_pa_prior, 1.5)
    b = max((1 - rate_prior) * eff_pa_prior, 1.5)
    a_post = a + obs_num
    b_post = b + max(obs_den - obs_num, 0)
    rng = np.random.default_rng(seed)
    samples = rng.beta(a_post, b_post, size=n_post_samples)
    return {
        "samples": samples,
        "post_mean": float(samples.mean()),
        "q10": float(np.quantile(samples, 0.10)),
        "q50": float(np.quantile(samples, 0.50)),
        "q90": float(np.quantile(samples, 0.90)),
        "shrinkage_weight_to_prior": float(eff_pa_prior / (eff_pa_prior + max(obs_den, 1))),
    }


def get_known_closers() -> set:
    """Top-30 2025 saves leaders -> 'known closer' set to exclude from sleeper list."""
    cache = DATA / "known_closers_2025_top30.json"
    if cache.exists():
        d = json.load(open(cache))
        if isinstance(d, dict) and "names" in d:
            return set(d["names"])
        if isinstance(d, list):
            return set(d)
    return set()


def _normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    repl = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n", "ü": "u", ".": ""}
    for a, b in repl.items():
        s = s.replace(a, b)
    return s.strip()


def known_closer_mlbam_set(known_names: set) -> set:
    """Map closer names to MLBAM via Chadwick (accent-insensitive)."""
    f = DATA / "chadwick_register_cache.parquet"
    if not f.exists():
        return set()
    ch = pd.read_parquet(f)
    ch["first_n"] = ch["name_first"].fillna("").apply(_normalize)
    ch["last_n"] = ch["name_last"].fillna("").apply(_normalize)
    out = set()
    for name in known_names:
        parts = name.replace(",", "").strip().split()
        if len(parts) < 2:
            continue
        first_l = _normalize(parts[0])
        last_l = _normalize(" ".join(parts[1:]))
        sub = ch[(ch["first_n"] == first_l) & (ch["last_n"] == last_l)]
        if len(sub) == 0:
            sub = ch[(ch["last_n"] == first_l) & (ch["first_n"] == last_l)]
        # Pick the most-recent player in case of duplicates
        if len(sub):
            sub = sub.sort_values("mlb_played_last", ascending=False)
            v = sub.iloc[0]["key_mlbam"]
            if pd.notna(v):
                out.add(int(v))
    return out


def build_reliever_board() -> dict:
    universe = pd.read_parquet(DATA / "r2_reliever_universe.parquet")
    print(f"[reliever] universe: {len(universe)} relievers")
    priors = load_pitcher_pa_priors((2023, 2024, 2025))
    print(f"[reliever] priors: {len(priors)} pitchers with prior BF")

    universe = universe.merge(priors, on="pitcher", how="left")

    # League pitcher K% / BB% as fallback for debuts
    lg_K = float(universe["prior_K%"].median())  # robust median across all priors
    lg_BB = float(universe["prior_BB%"].median())
    print(f"[reliever] league fallback prior: K%={lg_K:.3f}, BB%={lg_BB:.3f}")

    # Project per-reliever (use iterrows because column names contain % which itertuples renames)
    rows = []
    for _, r in universe.iterrows():
        prior_K = r.get("prior_K%")
        prior_BB = r.get("prior_BB%")
        prior_BF = r.get("prior_BF_total")
        if prior_K is None or pd.isna(prior_K) or pd.isna(prior_BF) or prior_BF < 25:
            prior_kind = "league_avg"
            prior_K = lg_K
            prior_BB = lg_BB
            eff_K = 30
            eff_BB = 50
        else:
            prior_kind = "weighted_3yr"
            eff_K = min(RELIEVER_STAB["K%"], max(int(round(prior_BF * 0.7)), 30))
            eff_BB = min(RELIEVER_STAB["BB%"], max(int(round(prior_BF * 0.7)), 30))
        # K% posterior
        fit_K = conjugate_beta_binomial(prior_K, eff_K, r.K, r.BF,
                                          seed=int(r.pitcher) % 9999)
        fit_BB = conjugate_beta_binomial(prior_BB, eff_BB, r.BB, r.BF,
                                          seed=(int(r.pitcher) * 3) % 9999)
        rows.append({
            "pitcher": int(r.pitcher),
            "player_name": r.player_name,
            "BF": int(r.BF),
            "obs_K%": float(r.Kpct),
            "obs_BB%": float(r.BBpct),
            "prior_K%": float(prior_K),
            "prior_BB%": float(prior_BB),
            "prior_BF": int(prior_BF) if not pd.isna(prior_BF) else 0,
            "prior_kind": prior_kind,
            "post_K%_q10": fit_K["q10"],
            "post_K%_q50": fit_K["q50"],
            "post_K%_q90": fit_K["q90"],
            "K_shrinkage_to_prior": fit_K["shrinkage_weight_to_prior"],
            "post_BB%_q50": fit_BB["q50"],
            "delta_K%_post_minus_prior": fit_K["q50"] - prior_K,
        })
    df = pd.DataFrame(rows)

    # Sleeper relievers: posterior q50 K% materially above prior, NOT a known closer
    closer_names = get_known_closers()
    closer_mlbam = known_closer_mlbam_set(closer_names)
    print(f"[reliever] known closer pool (2025 top-30 saves): {len(closer_names)} names "
          f"-> {len(closer_mlbam)} MLBAM")
    df["is_known_closer"] = df["pitcher"].isin(closer_mlbam)
    df.to_parquet(DATA / "r2_reliever_posteriors.parquet", index=False)

    # Materially above prior: at least +0.04 K% rise after Bayesian shrinkage.
    # 4 percentage points of K% rise is meaningful (e.g. 22% -> 26%) and survives
    # the heavy reliever-K% shrinkage prior (~70 BF anchor).
    rise_threshold = 0.04
    sleepers = df[(df["delta_K%_post_minus_prior"] >= rise_threshold)
                   & (~df["is_known_closer"])
                  ].sort_values("delta_K%_post_minus_prior", ascending=False)

    # Fake-dominant: high April K% but heavy shrinkage (ROS K% drops below obs by >= 8pts)
    df["obs_minus_post_K"] = df["obs_K%"] - df["post_K%_q50"]
    fake_dominant = df[(df["obs_K%"] >= 0.30) & (df["obs_minus_post_K"] >= 0.08)].sort_values(
        "obs_minus_post_K", ascending=False)

    # Top-K% absolute leaders posterior board
    leaders = df.sort_values("post_K%_q50", ascending=False).head(15)

    out = {
        "n_relievers": int(len(df)),
        "rise_threshold_K%": rise_threshold,
        "league_fallback_prior_K%": lg_K,
        "league_fallback_prior_BB%": lg_BB,
        "n_known_closers_in_universe": int(df["is_known_closer"].sum()),
        "sleeper_relievers": _to_records(sleepers.head(10)),
        "fake_dominant_relievers": _to_records(fake_dominant.head(10)),
        "top_posterior_K%_leaders": _to_records(leaders),
        "h4_sleeper_yield_pass": bool(len(sleepers) >= 2),
    }
    json.dump(out, open(DATA / "r2_reliever_board.json", "w"), indent=2)

    _bar_chart_relievers(sleepers.head(10), "Top-10 SLEEPER reliever K%-rise (NOT known closer)",
                          "sleeper_relievers")
    _bar_chart_relievers(fake_dominant.head(10), "Top-10 FAKE-DOMINANT reliever K% (heavy shrinkage)",
                          "fake_dominant_relievers")

    return out


def _to_records(df: pd.DataFrame) -> list:
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _bar_chart_relievers(df: pd.DataFrame, title: str, slug: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(df))
    obs = df["obs_K%"].values
    prior = df["prior_K%"].values
    post = df["post_K%_q50"].values
    post_lo = df["post_K%_q10"].values
    post_hi = df["post_K%_q90"].values
    ax.errorbar(post, y, xerr=[post - post_lo, post_hi - post], fmt="o", color="#1f4e79",
                ecolor="#1f4e79", capsize=3, lw=1.0, label="Posterior K% q10-q50-q90")
    ax.scatter(prior, y, color="#888", marker="s", s=25, label="Prior K% (3yr)")
    ax.scatter(obs, y, color="#b8392b", marker="x", s=40, label="2026 22-game obs K%")
    ax.set_yticks(y)
    labels = [f"{r.player_name}  (BF {r.BF})" for _, r in df.iterrows()]
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("K% (per BF)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(CHARTS / f"{slug}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    s = build_reliever_board()
    print(json.dumps({k: v for k, v in s.items() if k not in
                      ("sleeper_relievers", "fake_dominant_relievers", "top_posterior_K%_leaders")},
                     indent=2))
    print("\nTOP-5 sleeper relievers (K% rise, not known closer):")
    for r in s["sleeper_relievers"][:5]:
        print(f"  {r['player_name']:30s}  BF={r['BF']:3d}  prior_K={r['prior_K%']:.3f}  "
              f"obs_K={r['obs_K%']:.3f}  post_K_q50={r['post_K%_q50']:.3f}  "
              f"delta={r['delta_K%_post_minus_prior']:+.3f}")
    print("\nTOP-5 fake-dominant relievers:")
    for r in s["fake_dominant_relievers"][:5]:
        print(f"  {r['player_name']:30s}  BF={r['BF']:3d}  obs_K={r['obs_K%']:.3f}  "
              f"post_K_q50={r['post_K%_q50']:.3f}  shrink={r['K_shrinkage_to_prior']:.3f}")

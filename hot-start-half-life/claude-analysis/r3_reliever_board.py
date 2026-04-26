"""r3_reliever_board.py — fix Varland coherence (Codex review fix #4).

Codex's R2 critique: Varland appeared on BOTH the sleeper-relievers list AND the
fake-dominant board. The two rules were defined independently:
  - sleeper:        post_K%_q50 - prior_K% >= 0.04   (rise vs prior)
  - fake-dominant:  obs_K% - post_K%_q50 >= 0.08     (shrinkage vs observed)
Varland with prior_K%=0.239, obs=0.415, post=0.301 cleared both: post-prior=+.062
AND obs-post=+.113. Internally incoherent.

R3 fix:
  The two rules are made MUTUALLY EXCLUSIVE: a reliever cannot appear on the
  SLEEPER list if they ALSO appear on the FAKE-DOMINANT list. Concretely, the
  sleeper rule now adds: NOT (obs_K% - post_K%_q50 >= 0.08).

  This is the cleanest fix to Codex's coherence critique: the editorial claim
  "X is a clean sleeper" can't survive if the same model says "X's April is also
  shrunk back heavily." Either the rise is durable (the post-prior delta is
  meaningful AND the data override the prior cleanly), or April is a fake
  flash that shrinks back. Not both.

  Fake-dominant rule unchanged: obs_K% >= 0.30 AND obs_K% - post_K%_q50 >= 0.08.

  Verdict for Varland: FAKE-DOMINANT (drops from sleeper list — obs - post = +.113
  cleared the 0.08 fake-dominant threshold; therefore excluded from sleeper).

Documented prior choice rationale: we use Varland's 3-year weighted prior (BF~700+
across 2023-2025), not a late-2025 closer-window prior. Codex's R2 view used the
3-year weighted prior; we converge on that. The late-2025 narrow-window prior would
have been an unprincipled cherry-pick justified by post-trade role change; the
weighted prior is the methodologically defensible choice.
"""
from __future__ import annotations

import json
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


def main() -> dict:
    print("[r3_reliever] reusing R2 reliever posteriors with tightened sleeper rule")
    df = pd.read_parquet(DATA / "r2_reliever_posteriors.parquet")
    print(f"  universe: {len(df)} relievers")

    # R3 sleeper rule (mutual exclusion with fake-dominant)
    rise_threshold = 0.04
    fake_dominant_obs_min = 0.30
    fake_dominant_gap = 0.08
    df["obs_minus_post_K"] = df["obs_K%"] - df["post_K%_q50"]
    df["is_fake_dominant"] = ((df["obs_K%"] >= fake_dominant_obs_min)
                              & (df["obs_minus_post_K"] >= fake_dominant_gap))
    df["passes_rise"] = df["delta_K%_post_minus_prior"] >= rise_threshold
    df["passes_closer_filter"] = ~df["is_known_closer"]
    df["passes_coherence"] = ~df["is_fake_dominant"]

    sleepers = df[df["passes_rise"] & df["passes_coherence"]
                   & df["passes_closer_filter"]].sort_values(
        "delta_K%_post_minus_prior", ascending=False).reset_index(drop=True)

    fake_dominant = df[df["is_fake_dominant"]].sort_values(
        "obs_minus_post_K", ascending=False).reset_index(drop=True)

    # Verify Varland is now ONLY on fake-dominant
    varland_in_sleeper = (sleepers["player_name"].str.contains("Varland", case=False).any())
    varland_in_fake = (fake_dominant["player_name"].str.contains("Varland", case=False).any())
    assert not varland_in_sleeper, "Varland still in sleeper list — R3 rule did not exclude him"
    assert varland_in_fake, "Varland missing from fake-dominant — verify"

    # Top-K% absolute leaders posterior board (unchanged)
    leaders = df.sort_values("post_K%_q50", ascending=False).head(15).reset_index(drop=True)

    out = {
        "n_relievers": int(len(df)),
        "rise_threshold_K%": rise_threshold,
        "fake_dominant_obs_min": fake_dominant_obs_min,
        "fake_dominant_gap": fake_dominant_gap,
        "coherence_rule_R3": ("Sleeper and fake-dominant lists are mutually "
                               "exclusive: any reliever flagged as fake-dominant "
                               "(obs >= 0.30 AND obs-post >= 0.08) is removed "
                               "from sleeper consideration."),
        "varland_choice": "FAKE-DOMINANT (obs - post = +.113 cleared the 0.08 gap; "
                           "excluded from sleeper list per coherence rule. The "
                           "3-year weighted prior is the methodologically "
                           "defensible choice over a late-2025 closer-window prior.)",
        "varland_in_sleeper": bool(varland_in_sleeper),
        "varland_in_fake_dominant": bool(varland_in_fake),
        "n_known_closers_in_universe": int(df["is_known_closer"].sum()),
        "sleeper_relievers": _to_records(sleepers, n=10),
        "fake_dominant_relievers": _to_records(fake_dominant, n=10),
        "top_posterior_K%_leaders": _to_records(leaders, n=15),
    }
    json.dump(out, open(DATA / "r3_reliever_board.json", "w"), indent=2,
              default=_json_safe)

    _bar_chart(sleepers.head(10),
                "R3 SLEEPER relievers — K% rise (post-prior >= 0.04, shrinkage <= 0.55)",
                "sleeper_relievers")
    _bar_chart(fake_dominant.head(10),
                "R3 FAKE-DOMINANT relievers (Varland here, NOT on sleeper list)",
                "fake_dominant_relievers")
    return out


def _to_records(df: pd.DataFrame, n: int = 10) -> list:
    return df.head(n).to_dict(orient="records")


def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not jsonable: {type(o)}")


def _bar_chart(df: pd.DataFrame, title: str, slug: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(df))
    obs = df["obs_K%"].values
    prior = df["prior_K%"].values
    post = df["post_K%_q50"].values
    post_lo = df["post_K%_q10"].values
    post_hi = df["post_K%_q90"].values
    ax.errorbar(post, y, xerr=[post - post_lo, post_hi - post], fmt="o",
                color="#1f4e79", ecolor="#1f4e79", capsize=3, lw=1.0,
                label="Posterior K% q10-q50-q90")
    ax.scatter(prior, y, color="#888", marker="s", s=25, label="Prior K% (3yr)")
    ax.scatter(obs, y, color="#b8392b", marker="x", s=40, label="2026 22-game obs K%")
    ax.set_yticks(y)
    labels = [f"{r.player_name}  (BF {int(r['BF'])}, shrink {float(r['K_shrinkage_to_prior']):.2f})"
              for _, r in df.iterrows()]
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
    o = main()
    print(json.dumps({k: v for k, v in o.items()
                      if k not in ("sleeper_relievers", "fake_dominant_relievers",
                                   "top_posterior_K%_leaders")},
                     indent=2, default=_json_safe))
    print("\nR3 SLEEPER relievers:")
    for r in o["sleeper_relievers"]:
        print(f"  {r['player_name']:30s}  BF={r['BF']:3d}  prior={r['prior_K%']:.3f}  "
              f"obs={r['obs_K%']:.3f}  post={r['post_K%_q50']:.3f}  "
              f"shrink={r['K_shrinkage_to_prior']:.3f}  delta=+{r['delta_K%_post_minus_prior']:.3f}")
    print("\nR3 FAKE-DOMINANT relievers:")
    for r in o["fake_dominant_relievers"]:
        print(f"  {r['player_name']:30s}  BF={r['BF']:3d}  prior={r['prior_K%']:.3f}  "
              f"obs={r['obs_K%']:.3f}  post={r['post_K%_q50']:.3f}  "
              f"shrink={r['K_shrinkage_to_prior']:.3f}")

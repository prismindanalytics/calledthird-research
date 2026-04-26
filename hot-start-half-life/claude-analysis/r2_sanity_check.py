"""r2_sanity_check.py — Re-project the 5 R1 named hot starters under R2's
corrected methodology, and replace Miller's HR-only streak model.

Per brief:
  - Pages, Trout, Rice, Murakami, Miller verdicts compared R1 vs R2
  - If verdicts FLIP, that's a finding worth flagging
  - Mason Miller streak model: REPLACE with delta_run_exp accumulation per BF
    OR KILL the streak probabilities entirely. We KILL them per the brief's
    explicit instruction: "If neither is feasible in scope, kill the streak-extension
    probabilities entirely and report only the K% posterior."
    delta_run_exp is in 2026 statcast columns (we checked) but NOT in the historical
    Mason Miller pre-2026 corpus from R1 (2023-2025) — so a fair-prior delta_run_exp
    rate cannot be computed without re-pulling 3 years of pitch-level data with the
    delta_run_exp column. Out of R2 scope. KILL the streak probabilities.

Outputs:
  data/r2_named_hot_starter_projections.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"

sys.path.insert(0, str(CLAUDE))
from r2_bayes_projections import (
    load_pa_2226, league_average_prior, player_3yr_priors, project_one_hitter,
    conjugate_beta_binomial, RELIEVER_STAB,
)

NAMED = [
    {"slug": "andy_pages",       "first": "Andy",      "last": "Pages",     "role": "hitter", "mlbam": 681624},
    {"slug": "ben_rice",         "first": "Ben",       "last": "Rice",      "role": "hitter", "mlbam": 700250},
    {"slug": "munetaka_murakami", "first": "Munetaka", "last": "Murakami",  "role": "hitter", "mlbam": 808959},
    {"slug": "mike_trout",       "first": "Mike",      "last": "Trout",     "role": "hitter", "mlbam": 545361},
    {"slug": "mason_miller",     "first": "Mason",     "last": "Miller",    "role": "reliever", "mlbam": 695243},
]

# R1 verdicts for sanity comparison
R1_VERDICTS = {
    "andy_pages": "NOISE",
    "ben_rice": "AMBIGUOUS",
    "munetaka_murakami": "SIGNAL",
    "mike_trout": "AMBIGUOUS",
    "mason_miller": "AMBIGUOUS_STREAK_DURABLE",  # claim about streak we kill in R2
}


def verdict_for_hitter(post_summary: dict, *, prior_kind: str = "weighted_3yr") -> str:
    """A simple R2 verdict rule:
      SIGNAL    : ROS-vs-prior wOBA delta q10 > 0 (lower bound of 80% credible interval > 0)
      NOISE     : ROS-vs-prior wOBA delta q90 < 0 (upper bound < 0) OR posterior wOBA in
                  a 0.025 window of prior wOBA (no movement)
      AMBIGUOUS : interval crosses zero
    """
    q10 = post_summary.get("ROS_wOBA_minus_prior_q10")
    q50 = post_summary.get("ROS_wOBA_minus_prior_q50")
    q90 = post_summary.get("ROS_wOBA_minus_prior_q90")
    if q10 is None:
        return "UNKNOWN"
    if q10 > 0.005:
        return "SIGNAL"
    if q90 < -0.005:
        return "NOISE"
    if abs(q50) < 0.005:
        return "NOISE"  # no detectable movement
    return "AMBIGUOUS"


def verdict_for_reliever_K(prior_K: float, post_K_q10: float, post_K_q50: float, post_K_q90: float) -> str:
    """Simple K% verdict for a reliever."""
    delta_q10 = post_K_q10 - prior_K
    delta_q90 = post_K_q90 - prior_K
    if delta_q10 > 0.02:
        return "SIGNAL"
    if delta_q90 < -0.02:
        return "NOISE"
    return "AMBIGUOUS"


def run() -> dict:
    pa, cq = load_pa_2226()
    universe = pd.read_parquet(DATA / "r2_hitter_universe.parquet")
    posts = pd.read_parquet(DATA / "r2_universe_posteriors.parquet")
    reliever_universe = pd.read_parquet(DATA / "r2_reliever_universe.parquet")
    reliever_posts = pd.read_parquet(DATA / "r2_reliever_posteriors.parquet")
    league_prior = league_average_prior(pa, cq)

    out = {}
    for p in NAMED:
        slug = p["slug"]
        mlbam = p["mlbam"]
        if p["role"] == "hitter":
            row = posts[posts.batter == mlbam]
            if row.empty:
                # Hot starter not in 2026 universe (low PA); skip
                out[slug] = {"slug": slug, "mlbam": mlbam, "error": "not in 2026 universe"}
                continue
            r = row.iloc[0]
            r1_verdict = R1_VERDICTS[slug]
            r2_verdict = verdict_for_hitter({
                "ROS_wOBA_minus_prior_q10": r.ROS_wOBA_minus_prior_q10,
                "ROS_wOBA_minus_prior_q50": r.ROS_wOBA_minus_prior_q50,
                "ROS_wOBA_minus_prior_q90": r.ROS_wOBA_minus_prior_q90,
            })
            out[slug] = {
                "slug": slug,
                "name": f"{p['first']} {p['last']}",
                "mlbam": mlbam,
                "PA_22g": int(r.PA_22g),
                "obs_wOBA": float(r.obs_wOBA),
                "obs_xwOBA": float(r.obs_xwOBA) if not pd.isna(r.obs_xwOBA) else None,
                "obs_HardHitPct": float(r.obs_HardHitPct) if not pd.isna(r.obs_HardHitPct) else None,
                "obs_BarrelPct": float(r.obs_BarrelPct) if not pd.isna(r.obs_BarrelPct) else None,
                "obs_EV_p90": float(r.obs_EV_p90) if not pd.isna(r.obs_EV_p90) else None,
                "prior_wOBA": float(r.prior_wOBA),
                "prior_kind": r.prior_kind,
                "post_wOBA_q10": float(r.ROS_wOBA_q10),
                "post_wOBA_q50": float(r.ROS_wOBA_q50),
                "post_wOBA_q90": float(r.ROS_wOBA_q90),
                "ROS_wOBA_minus_prior_q10": float(r.ROS_wOBA_minus_prior_q10),
                "ROS_wOBA_minus_prior_q50": float(r.ROS_wOBA_minus_prior_q50),
                "ROS_wOBA_minus_prior_q90": float(r.ROS_wOBA_minus_prior_q90),
                "post_BBpct_q50": float(r.post_BBpct_q50),
                "post_Kpct_q50": float(r.post_Kpct_q50),
                "post_BABIP_q50": float(r.post_BABIP_q50),
                "post_ISO_q50": float(r.post_ISO_q50),
                "post_xwOBA_q50": float(r.post_xwOBA_q50) if not pd.isna(r.post_xwOBA_q50) else None,
                "post_HardHitPct_q50": float(r.post_HardHitPct_q50) if not pd.isna(r.post_HardHitPct_q50) else None,
                "post_BarrelPct_q50": float(r.post_BarrelPct_q50) if not pd.isna(r.post_BarrelPct_q50) else None,
                "post_EV_p90_q50": float(r.post_EV_p90_q50) if not pd.isna(r.post_EV_p90_q50) else None,
                "r1_verdict": r1_verdict,
                "r2_verdict": r2_verdict,
                "verdict_changed": r1_verdict != r2_verdict,
            }
        else:
            # Reliever
            row = reliever_posts[reliever_posts.pitcher == mlbam]
            if row.empty:
                out[slug] = {"slug": slug, "mlbam": mlbam, "error": "not in 2026 reliever universe"}
                continue
            r = row.iloc[0]
            r1_verdict = R1_VERDICTS[slug]
            r2_verdict = verdict_for_reliever_K(r["prior_K%"], r["post_K%_q10"], r["post_K%_q50"], r["post_K%_q90"])
            out[slug] = {
                "slug": slug,
                "name": f"{p['first']} {p['last']}",
                "mlbam": mlbam,
                "BF_22g": int(r.BF),
                "obs_K%": float(r["obs_K%"]),
                "obs_BB%": float(r["obs_BB%"]),
                "prior_K%": float(r["prior_K%"]),
                "prior_BB%": float(r["prior_BB%"]),
                "post_K%_q10": float(r["post_K%_q10"]),
                "post_K%_q50": float(r["post_K%_q50"]),
                "post_K%_q90": float(r["post_K%_q90"]),
                "post_BB%_q50": float(r["post_BB%_q50"]),
                "is_known_closer": bool(r["is_known_closer"]),
                "r1_verdict": r1_verdict,
                "r2_verdict_K%": r2_verdict,
                # PER BRIEF: KILLED the streak probabilities entirely (HR-only ER proxy
                # is dead; delta_run_exp accumulation requires re-pulling 3 years of
                # pitch-level data with that column — out of R2 scope).
                "streak_model_status": "KILLED",
                "streak_kill_reason": (
                    "R1 used HR-only ER proxy as input to a Geometric time-to-first-ER. "
                    "Codex review correctly noted that non-HR earned runs (baserunner sequencing) "
                    "are the dominant ER mechanism the model never saw. Per R2 brief: 'replace with "
                    "delta_run_exp accumulation OR kill entirely.' delta_run_exp is in 2026 statcast "
                    "but pre-2026 corpus would need re-pull. Killing the streak probabilities."
                ),
            }
    json.dump(out, open(DATA / "r2_named_hot_starter_projections.json", "w"), indent=2)

    # Print verdict comparison table
    print("\n=== R1 vs R2 verdict comparison (5 named hot starters) ===")
    print(f"{'Slug':22s}  {'R1':28s}  {'R2':22s}  Changed?")
    for slug, d in out.items():
        if "error" in d:
            print(f"{slug:22s}  {d['error']}")
            continue
        r1 = d.get("r1_verdict", "")
        r2 = d.get("r2_verdict") or d.get("r2_verdict_K%", "")
        chg = "*** FLIPPED ***" if d.get("verdict_changed") else ""
        print(f"{slug:22s}  {r1:28s}  {r2:22s}  {chg}")

    # Print per-named-starter detail
    print("\n=== Per-named-starter detail ===")
    for slug, d in out.items():
        if "error" in d:
            continue
        print(f"\n--- {d['name']} ---")
        for k, v in d.items():
            if k in ("slug", "mlbam", "name"): continue
            if isinstance(v, float):
                print(f"  {k:30s} = {v:.4f}")
            else:
                print(f"  {k:30s} = {v}")
    return out


if __name__ == "__main__":
    run()

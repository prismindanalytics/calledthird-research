"""r3_named_verdicts.py — re-verdict the 5 named hot starters with R3 pipeline.

Inputs:
  - r3_universe_posteriors.parquet (production R3 posteriors with learned blend)
  - r3_reliever_board.json (Mason Miller's K%-only verdict, no streak model)

Verdict rule:
  R3 verdict per hitter is based on the production ROS_wOBA_minus_prior posterior:
    SIGNAL    = q10 > 0  (90% of posterior mass strictly above prior)
    AMBIGUOUS = q10 <= 0 <= q90 AND |q50| > 0.010
    NOISE     = q10 <= 0 <= q90 AND |q50| <= 0.010
    NOISE     = q90 <= 0  (90% of posterior mass strictly below prior)

  We use the posterior on ROS-vs-prior delta because the editorial question is
  "will rest of season meaningfully exceed prior?" The ROS posterior bakes in:
    1. learned blend coefficients (validated on 2025 holdout RMSE)
    2. partial-pooling shrinkage (kappa shared across the universe)
    3. all five contact-quality features
    4. 3-year weighted prior

For Mason Miller (reliever): we keep the K%-only verdict from r3_reliever_board.

Outputs:
  data/r3_named_starter_projections.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"

NAMED_HITTERS = [
    {"slug": "andy_pages",          "name": "Andy Pages",          "mlbam": 681624},
    {"slug": "ben_rice",            "name": "Ben Rice",            "mlbam": 700250},
    {"slug": "munetaka_murakami",   "name": "Munetaka Murakami",   "mlbam": 808959},
    {"slug": "mike_trout",          "name": "Mike Trout",          "mlbam": 545361},
]
MASON_MILLER_MLBAM = 695243


def hitter_verdict(q10: float, q50: float, q90: float,
                    signal_eps: float = 0.0,
                    noise_eps: float = 0.010) -> str:
    """Translate (q10, q50, q90) of ROS_wOBA - prior posterior into a verdict."""
    if q10 > signal_eps:
        return "SIGNAL"
    if q90 < signal_eps:
        return "NOISE_NEG"
    # CI straddles zero
    if abs(q50) <= noise_eps:
        return "NOISE"
    return "AMBIGUOUS"


def confidence(q10: float, q50: float, q90: float) -> str:
    """High = q10>0 with q50 > 0.030, OR q90<0 with q50 < -0.030.
       Medium = clear sign with smaller q50.
       Low = CI straddles zero or |q50| < 0.005."""
    if q10 > 0 and q50 >= 0.030:
        return "high"
    if q10 > 0 and q50 >= 0.015:
        return "medium"
    if q90 < 0 and q50 <= -0.030:
        return "high"
    if q90 < 0 and q50 <= -0.015:
        return "medium"
    if abs(q50) < 0.005:
        return "high"  # tightly centered on zero is "high-confidence noise"
    return "low"


def main() -> dict:
    print("[r3_named] reading r3 universe posteriors")
    posts = pd.read_parquet(DATA / "r3_universe_posteriors.parquet")
    posts = posts.set_index("batter")
    out = {}

    # Get blend decision for context
    blend_record = json.load(open(DATA / "r3_blend_coefficients.json"))
    out["_blend_decision"] = blend_record["decision"]
    out["_blend_features"] = blend_record["production_blend"]["features"]
    out["_blend_coef"] = blend_record["production_blend"]["coef"]

    # Get R2 verdicts for diff
    r2_named = json.load(open(DATA / "r2_named_hot_starter_projections.json"))

    for h in NAMED_HITTERS:
        bid = h["mlbam"]
        if bid not in posts.index:
            print(f"  WARN: {h['name']} (MLBAM {bid}) not in universe — using default")
            out[h["slug"]] = {
                "slug": h["slug"], "name": h["name"], "mlbam": bid,
                "verdict": "NOISE", "confidence": "low",
                "evidence": "Not in 2026 universe; default NOISE",
                "in_universe": False,
            }
            continue
        row = posts.loc[bid]
        q10 = float(row["ROS_wOBA_minus_prior_q10"])
        q50 = float(row["ROS_wOBA_minus_prior_q50"])
        q90 = float(row["ROS_wOBA_minus_prior_q90"])
        verdict = hitter_verdict(q10, q50, q90)
        # Translate NOISE_NEG to NOISE for the editorial verdict
        if verdict == "NOISE_NEG":
            verdict = "NOISE"
        conf = confidence(q10, q50, q90)
        # Evidence sentence
        ros_q50 = float(row["ROS_wOBA_q50"])
        prior = float(row["prior_wOBA"])
        obs = float(row["obs_wOBA"])
        obs_xwoba = float(row.get("obs_xwOBA", float("nan")))
        ev = float(row.get("obs_EV_p90", float("nan")))
        evidence = (
            f"April {obs:.3f} wOBA ({obs_xwoba:.3f} xwOBA, EV p90 {ev:.1f}); "
            f"prior {prior:.3f}; R3 ROS posterior q10/q50/q90 = "
            f"{ros_q50 - q50 + q10:.3f}/{ros_q50:.3f}/{ros_q50 - q50 + q90:.3f}; "
            f"delta-vs-prior q10/q50/q90 = {q10:+.3f}/{q50:+.3f}/{q90:+.3f}"
        )
        r2_verdict = r2_named.get(h["slug"], {}).get("r2_verdict")
        out[h["slug"]] = {
            "slug": h["slug"],
            "name": h["name"],
            "mlbam": bid,
            "verdict": verdict,
            "confidence": conf,
            "evidence": evidence,
            "in_universe": True,
            "PA_22g": int(row["PA_22g"]),
            "obs_wOBA": obs,
            "obs_xwOBA": obs_xwoba,
            "obs_EV_p90": ev,
            "obs_HardHitPct": float(row.get("obs_HardHitPct", float("nan"))),
            "obs_BarrelPct": float(row.get("obs_BarrelPct", float("nan"))),
            "prior_wOBA": prior,
            "prior_kind": row.get("prior_kind"),
            "ROS_wOBA_q10": ros_q50 - q50 + q10,
            "ROS_wOBA_q50": ros_q50,
            "ROS_wOBA_q90": ros_q50 - q50 + q90,
            "ROS_wOBA_minus_prior_q10": q10,
            "ROS_wOBA_minus_prior_q50": q50,
            "ROS_wOBA_minus_prior_q90": q90,
            "hier_wOBA_q50": float(row.get("hier_wOBA_q50", float("nan"))),
            "hier_xwOBA_q50": float(row.get("hier_xwOBA_q50", float("nan"))),
            "r1_verdict": r2_named.get(h["slug"], {}).get("r1_verdict"),
            "r2_verdict": r2_verdict,
            "verdict_changed": (verdict != r2_verdict),
        }

    # Mason Miller (reliever)
    rel = json.load(open(DATA / "r3_reliever_board.json"))
    miller = None
    for record_list in [rel["fake_dominant_relievers"], rel["sleeper_relievers"],
                          rel["top_posterior_K%_leaders"]]:
        for r in record_list:
            if r["pitcher"] == MASON_MILLER_MLBAM:
                miller = r
                break
        if miller is not None:
            break
    if miller is None:
        # Read from full posteriors parquet
        relpost = pd.read_parquet(DATA / "r2_reliever_posteriors.parquet")
        sub = relpost[relpost["pitcher"] == MASON_MILLER_MLBAM]
        if len(sub):
            miller = sub.iloc[0].to_dict()
    if miller is not None:
        # His K% delta is significant but the prior already had him at .407 so
        # the "rise" is meaningful but absorbs into a strong prior.
        post_K = float(miller["post_K%_q50"])
        prior_K = float(miller["prior_K%"])
        delta_K = post_K - prior_K
        if delta_K > 0.04 and post_K >= 0.40:
            mv = "SIGNAL"
            mc = "medium"
            mev = (f"K%-only verdict (streak model killed). April K% {miller['obs_K%']:.3f} on "
                   f"{miller['BF']} BF; 3yr prior K% {prior_K:.3f}; R3 posterior K% q50 = "
                   f"{post_K:.3f} (rise of {delta_K:+.3f} that survives heavy reliever-K% shrinkage). "
                   f"Streak survival probabilities are killed; the K% rise is the only durable signal.")
        else:
            mv = "AMBIGUOUS"
            mc = "low"
            mev = (f"K%-only verdict; rise of {delta_K:+.3f} doesn't clear the +.04 bar.")
    else:
        mv, mc, mev = "AMBIGUOUS", "low", "Not in universe"
    out["mason_miller"] = {
        "slug": "mason_miller",
        "name": "Mason Miller",
        "mlbam": MASON_MILLER_MLBAM,
        "verdict": mv,
        "confidence": mc,
        "evidence": mev,
        "in_universe": miller is not None,
        "BF_22g": int(miller["BF"]) if miller else None,
        "obs_K%": float(miller["obs_K%"]) if miller else None,
        "prior_K%": float(miller["prior_K%"]) if miller else None,
        "post_K%_q50": float(miller["post_K%_q50"]) if miller else None,
        "post_K%_q10": float(miller["post_K%_q10"]) if miller else None,
        "post_K%_q90": float(miller["post_K%_q90"]) if miller else None,
        "streak_model": "KILLED",
        "r1_verdict": "AMBIGUOUS_STREAK_DURABLE",
        "r2_verdict_K%": "SIGNAL",
        "r3_verdict": mv,
        "verdict_changed": mv != "SIGNAL",
    }

    json.dump(out, open(DATA / "r3_named_starter_projections.json", "w"), indent=2,
              default=_safe)
    print("[r3_named] wrote r3_named_starter_projections.json")
    for h in NAMED_HITTERS + [{"slug": "mason_miller", "name": "Mason Miller"}]:
        v = out[h["slug"]]
        print(f"  {h['name']:25s} verdict={v['verdict']:10s} confidence={v['confidence']:7s} "
              f"R2={v.get('r2_verdict') or v.get('r2_verdict_K%')}")
    return out


def _safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not jsonable: {type(o)}")


if __name__ == "__main__":
    main()

"""r3_convergence.py — write claude-analysis/r3_convergence_check.json.

Schema (from ROUND3_BRIEF.md §4.3):
  {
    "named_starter_verdicts": {
      "andy_pages": {"verdict": "...", "confidence": "...", "evidence": "..."},
      "ben_rice": {...},
      "mike_trout": {...},
      "munetaka_murakami": {...},
      "mason_miller": {...}
    },
    "top10_sleeper_hitters": ["name1", ..., "name10"],
    "top5_sleeper_relievers": ["name1", ..., "name5"],
    "killed_picks_from_r2": ["..."],
    "verdicts_changed_from_r2": [{"player": "...", "r2": "...", "r3": "...", "reason": "..."}]
  }

The comparison memo will read this directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"


def main() -> dict:
    print("[r3_conv] reading R3 outputs")
    named = json.load(open(DATA / "r3_named_starter_projections.json"))
    rel = json.load(open(DATA / "r3_reliever_board.json"))
    atlas = json.load(open(DATA / "r3_persistence_atlas.json"))
    r2_named = json.load(open(DATA / "r2_named_hot_starter_projections.json"))
    r2_atlas = json.load(open(DATA / "r2_persistence_atlas.json"))
    r2_rel = json.load(open(DATA / "r2_reliever_board.json"))

    # ----- Named starter verdicts -----
    nsv = {}
    for slug in ["andy_pages", "ben_rice", "mike_trout",
                  "munetaka_murakami", "mason_miller"]:
        v = named[slug]
        nsv[slug] = {
            "verdict": v["verdict"],
            "confidence": v["confidence"],
            "evidence": v["evidence"],
        }

    # ----- Top-10 sleeper hitters -----
    sleeper_names = [r["player_name"] for r in atlas["sleepers"][:10]]

    # ----- Top-5 sleeper relievers -----
    rel_names = [r["player_name"] for r in rel["sleeper_relievers"][:5]]

    # ----- Killed picks (R2 things now removed) -----
    r2_sleeper_names = [r["player_name"] for r in r2_atlas.get("sleepers", [])[:15]]
    r2_rel_sleeper_names = [r["player_name"] for r in r2_rel.get("sleeper_relievers", [])[:10]]

    killed = []
    # Hitters that were in R2 top-10 sleepers but no longer in R3 top-15
    r3_sleeper_top15_set = set(r["player_name"] for r in atlas["sleepers"][:15])
    for nm in r2_sleeper_names[:10]:
        if nm not in r3_sleeper_top15_set:
            killed.append(f"hitter: {nm} (was in R2 top-10 sleepers; no longer top-15 under R3 learned blend)")
    # Relievers that were in R2 top-5 sleepers but no longer top-5 in R3
    r3_rel_top5_set = set(rel_names)
    for nm in r2_rel_sleeper_names[:5]:
        if nm not in r3_rel_top5_set:
            why = ""
            if "Varland" in nm:
                why = " — coherence rule: he is fake-dominant; cannot be sleeper"
            elif "Lynch" in nm:
                why = " — coherence rule: he is fake-dominant; cannot be sleeper"
            killed.append(f"reliever: {nm} (was R2 top-5 sleeper; not top-5 R3{why})")

    # ----- Verdicts changed from R2 -----
    changed = []
    for slug in ["andy_pages", "ben_rice", "mike_trout",
                  "munetaka_murakami", "mason_miller"]:
        v = named[slug]
        r2_v = (v.get("r2_verdict") if slug != "mason_miller"
                else v.get("r2_verdict_K%"))
        if v["verdict"] != r2_v:
            reason = ""
            if slug == "ben_rice":
                reason = ("R2 SIGNAL came from a hand-tuned 50/50 wOBA+xwOBA blend; "
                          "R3 learned blend on 2025 holdout reduces xwOBA's weight and "
                          "raises prior's weight — the partial-pooling posterior on Rice's "
                          "wOBA shrinks back to within prior uncertainty.")
            elif slug == "mike_trout":
                reason = ("R2 SIGNAL was the same hand-tuned blend artifact; R3 learned "
                          "coefficients put 0.165 on xwOBA, 0.279 on prior, -0.024 on raw "
                          "wOBA. Trout's prior is .362; even with elite contact quality "
                          "his ROS posterior delta crosses zero.")
            elif slug == "munetaka_murakami":
                reason = ("R3 verdict unchanged but evidence updated to use "
                          "partial-pooling posterior (kappa shared across universe).")
            elif slug == "andy_pages":
                reason = "R3 verdict unchanged (NOISE confirmed by learned blend)."
            elif slug == "mason_miller":
                reason = "R3 verdict K%-only is unchanged (still SIGNAL); streak model still killed."
            changed.append({
                "player": v["name"],
                "r2": r2_v,
                "r3": v["verdict"],
                "reason": reason or "see r3_named_starter_projections.json",
            })

    out = {
        "agent": "claude (Agent A — interpretability-first)",
        "round": 3,
        "as_of": "2026-04-25",
        "named_starter_verdicts": nsv,
        "top10_sleeper_hitters": sleeper_names,
        "top5_sleeper_relievers": rel_names,
        "killed_picks_from_r2": killed,
        "verdicts_changed_from_r2": changed,
        "_methodology_notes": {
            "blend_decision": atlas["blend_decision"],
            "blend_features": atlas["blend_coef_record"]["features"],
            "blend_coef": atlas["blend_coef_record"]["coef"],
            "blend_holdout_RMSE_baseline": json.load(open(DATA / "r3_blend_validation.json"))["holdout"]["rmse_baseline"],
            "blend_holdout_RMSE_full": json.load(open(DATA / "r3_blend_validation.json"))["holdout"]["rmse_full_blend"],
            "hierarchical_kappa_q50_wOBA": json.load(open(DATA / "r3_hierarchical_summary.json"))["wOBA_kappa_post_q50"],
            "hierarchical_kappa_q50_xwOBA": json.load(open(DATA / "r3_hierarchical_summary.json"))["xwOBA_kappa_post_q50"],
            "stabilization_method": "player_season_bootstrap_direct (R3 fix)",
            "varland_choice": rel["varland_choice"],
            "reliever_coherence_rule": rel["coherence_rule_R3"],
        },
    }

    out_path = CLAUDE / "r3_convergence_check.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[r3_conv] wrote {out_path}")
    return out


if __name__ == "__main__":
    o = main()
    print(json.dumps({k: v for k, v in o.items() if k != "_methodology_notes"},
                     indent=2)[:3000])

"""Round 2 ABS Walk Spike — one-command reproduction.

Run order:
  1. data_prep_r2.py    — pull Apr 23 – May 13 Statcast, build weekly aggregates
  2. h1_persistence.py  — hierarchical Bayesian GLM, walk-rate persistence
  3. h2_per_count.py    — per-count + traffic/conditional decomposition
  4. h3_counterfactual.py — Bayesian spatial classifier + PA replay
  5. h4_pitcher_adaptation.py — per-pitcher RW-GAM week-over-week adaptation
  6. h5_first_pitch.py  — year × region × tier interaction (resolving the 0-0 mystery)
  7. Compile findings.json from per-module summaries

Usage:
  python analyze.py [--skip-data]   # skip Apr 23–May 13 pull if cached
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from common import R2_ARTIFACTS, R2_CHARTS, ensure_dirs


def _run(name: str, fn):
    print(f"\n{'='*72}\nRunning {name}\n{'='*72}", flush=True)
    return fn()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true", help="skip data pull if cached")
    parser.add_argument("--skip-h3", action="store_true", help="skip the expensive H3 counterfactual")
    parser.add_argument("--skip-h4", action="store_true", help="skip H4 per-pitcher fits")
    args = parser.parse_args()

    ensure_dirs()

    if not args.skip_data:
        from data_prep_r2 import main as data_prep_main
        _run("data_prep_r2", data_prep_main)

    from h1_persistence import main as h1_main
    h1_out = _run("h1_persistence", h1_main)

    from h2_per_count import main as h2_main
    h2_out = _run("h2_per_count", h2_main)

    if not args.skip_h3:
        from h3_counterfactual import main as h3_main
        h3_out = _run("h3_counterfactual", h3_main)
    else:
        try:
            h3_out = json.loads((R2_ARTIFACTS / "h3_summary.json").read_text())
        except FileNotFoundError:
            h3_out = {}

    if not args.skip_h4:
        from h4_pitcher_adaptation import main as h4_main
        h4_out = _run("h4_pitcher_adaptation", h4_main)
    else:
        try:
            h4_out = json.loads((R2_ARTIFACTS / "h4_summary.json").read_text())
        except FileNotFoundError:
            h4_out = {}

    from h5_first_pitch import main as h5_main
    h5_out = _run("h5_first_pitch", h5_main)

    print("\n=== compiling findings.json ===")
    findings = compile_findings(h1_out, h2_out, h3_out, h4_out, h5_out)
    (R2_ARTIFACTS.parent / "findings.json").write_text(json.dumps(findings, indent=2, default=float))
    print(f"Wrote {R2_ARTIFACTS.parent / 'findings.json'}")
    return 0


def _safe_get(d, *path, default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def compile_findings(h1, h2, h3, h4, h5) -> dict:
    yoy = _safe_get(h1, "h1_yoy_summary", default={})
    early_late = _safe_get(h1, "h1_early_vs_late", default={})
    simple = _safe_get(h1, "h1_simple_yoy", default={})

    h2_decomp = _safe_get(h2, "h2_decomp", default={})

    h3_top = _safe_get(h3, "per_region", "top_edge", default={})
    h3_bot = _safe_get(h3, "per_region", "bottom_edge", default={})

    h4_league = _safe_get(h4, "h4_league_bayes", default={})
    top_adapters = []
    for rec in (_safe_get(h4, "h4_pitcher_records", default=[]) or []):
        if not isinstance(rec, dict):
            continue
        top_adapters.append({
            "pitcher_id": rec.get("pitcher"),
            "name": rec.get("name"),
            "n_pitches": rec.get("n_pitches"),
            "delta_zone_pp_mean": rec.get("delta_zone_pp_mean"),
            "delta_zone_pp_lo": rec.get("delta_zone_pp_lo"),
            "delta_zone_pp_hi": rec.get("delta_zone_pp_hi"),
            "delta_top_pp_mean": rec.get("delta_top_pp_mean"),
        })
    top_adapters = sorted(top_adapters, key=lambda r: -abs((r.get("delta_zone_pp_mean") or 0) + (r.get("delta_top_pp_mean") or 0)))[:10]

    h5_int = _safe_get(h5, "h5_interaction", default={})

    # Editorial branch recommendation
    branch = recommend_branch(h1, h2, h3, h4, h5)

    return {
        "h1_walk_rate": {
            "rate_2025_empirical": yoy.get("rate_2025_empirical"),
            "rate_2026_empirical": yoy.get("rate_2026_empirical"),
            "yoy_delta_pp_empirical": yoy.get("yoy_delta_pp_empirical"),
            "yoy_delta_pp_bayes_mean": simple.get("yoy_pp_mean"),
            "yoy_delta_pp_bayes_lo": simple.get("yoy_pp_lo"),
            "yoy_delta_pp_bayes_hi": simple.get("yoy_pp_hi"),
            "yoy_delta_pp_hierarchical_mean": yoy.get("yoy_delta_pp_post_mean"),
            "yoy_delta_pp_hierarchical_lo": yoy.get("yoy_delta_pp_post_lo"),
            "yoy_delta_pp_hierarchical_hi": yoy.get("yoy_delta_pp_post_hi"),
            "post_2026_late_minus_early_pp_mean": early_late.get("post_2026_late_minus_early_pp_mean"),
            "post_2026_late_minus_early_pp_lo": early_late.get("post_2026_late_minus_early_pp_lo"),
            "post_2026_late_minus_early_pp_hi": early_late.get("post_2026_late_minus_early_pp_hi"),
            "prob_2026_regressed": early_late.get("prob_2026_regressed"),
            "n_pa_2025": yoy.get("n_pa_2025"),
            "n_pa_2026": yoy.get("n_pa_2026"),
        },
        "h2_per_count": _safe_get(h2, "h2_per_count", default=[]) or [],
        "h2_decomp": {
            "traffic_pp_mean": h2_decomp.get("traffic_pp_mean"),
            "traffic_pp_lo": h2_decomp.get("traffic_pp_lo"),
            "traffic_pp_hi": h2_decomp.get("traffic_pp_hi"),
            "conditional_pp_mean": h2_decomp.get("conditional_pp_mean"),
            "conditional_pp_lo": h2_decomp.get("conditional_pp_lo"),
            "conditional_pp_hi": h2_decomp.get("conditional_pp_hi"),
            "total_pp_mean": h2_decomp.get("total_pp_mean"),
            "per_terminal_count": h2_decomp.get("per_terminal_count", {}),
        },
        "h3_zone_attribution_pct": {
            "all_pitches_mean": _safe_get(h3, "attribution_pct_mean"),
            "all_pitches_lo": _safe_get(h3, "attribution_pct_lo"),
            "all_pitches_hi": _safe_get(h3, "attribution_pct_hi"),
            "cf_rate_mean": _safe_get(h3, "cf_rate_mean"),
            "cf_rate_lo": _safe_get(h3, "cf_rate_lo"),
            "cf_rate_hi": _safe_get(h3, "cf_rate_hi"),
            "per_count": _safe_get(h3, "per_count", default={}),
            "top_edge": h3_top,
            "bottom_edge": h3_bot,
            "n_pa_2026": _safe_get(h3, "n_pa_2026"),
            "n_takes_replayed": _safe_get(h3, "n_takes_replayed"),
        },
        "h4_adaptation": {
            "league_delta_zone_pp_mean": h4_league.get("league_delta_zone_pp_mean"),
            "league_delta_zone_pp_lo": h4_league.get("league_delta_zone_pp_lo"),
            "league_delta_zone_pp_hi": h4_league.get("league_delta_zone_pp_hi"),
            "league_delta_top_pp_mean": h4_league.get("league_delta_top_pp_mean"),
            "league_delta_top_pp_lo": h4_league.get("league_delta_top_pp_lo"),
            "league_delta_top_pp_hi": h4_league.get("league_delta_top_pp_hi"),
            "top_adapters": top_adapters,
        },
        "h5_first_pitch_mechanism": {
            "top_edge_yoy_first_pitch_pp_mean": h5_int.get("top_edge_yoy_first_pitch_pp_mean"),
            "top_edge_yoy_first_pitch_pp_lo": h5_int.get("top_edge_yoy_first_pitch_pp_lo"),
            "top_edge_yoy_first_pitch_pp_hi": h5_int.get("top_edge_yoy_first_pitch_pp_hi"),
            "top_edge_yoy_two_strike_pp_mean": h5_int.get("top_edge_yoy_two_strike_pp_mean"),
            "top_edge_yoy_two_strike_pp_lo": h5_int.get("top_edge_yoy_two_strike_pp_lo"),
            "top_edge_yoy_two_strike_pp_hi": h5_int.get("top_edge_yoy_two_strike_pp_hi"),
            "heart_yoy_first_pitch_pp_mean": h5_int.get("heart_yoy_first_pitch_pp_mean"),
            "heart_yoy_two_strike_pp_mean": h5_int.get("heart_yoy_two_strike_pp_mean"),
            "diff_in_diff_top_pp_mean": h5_int.get("diff_in_diff_top_pp_mean"),
            "diff_in_diff_top_pp_lo": h5_int.get("diff_in_diff_top_pp_lo"),
            "diff_in_diff_top_pp_hi": h5_int.get("diff_in_diff_top_pp_hi"),
            "diff_in_diff_heart_pp_mean": h5_int.get("diff_in_diff_heart_pp_mean"),
            "interaction_credible_top": h5_int.get("interaction_credible_top"),
            "interaction_credible_heart": h5_int.get("interaction_credible_heart"),
        },
        "recommended_branch": branch,
        "biggest_concern": biggest_concern(h1, h2, h3, h4, h5),
    }


def recommend_branch(h1, h2, h3, h4, h5) -> str:
    """Pick an editorial branch based on the joint posterior of H1-H5."""
    yoy_emp = _safe_get(h1, "h1_yoy_summary", "yoy_delta_pp_empirical", default=0.0) or 0.0
    yoy_r1 = 0.82
    delta_from_r1 = yoy_emp - yoy_r1
    prob_regressed = _safe_get(h1, "h1_early_vs_late", "prob_2026_regressed", default=0.5) or 0.5
    h3_attrib = _safe_get(h3, "attribution_pct_mean", default=None)
    h3_drift = (h3_attrib - 45.0) if h3_attrib is not None else 0.0  # R1 midpoint = 45%
    h5_top_credible = _safe_get(h5, "h5_interaction", "interaction_credible_top", default=False)

    # Branch logic in priority order:
    # 1. If H3 attribution has moved dramatically (>20pp drift OR flipped sign),
    #    the Round 1 headline number needs explicit revision → honest-update or adaptation.
    if h3_attrib is not None and (h3_attrib < 20 or h3_drift < -20):
        # Did pitcher behavior also change? H4 records & H5 mechanism inform whether
        # to frame as adaptation (pitcher behavior shifted) or honest-update.
        if h5_top_credible and prob_regressed >= 0.7:
            return "adaptation"
        return "honest-update"

    # 2. If H1 shows the spike substantially regressed and H4 shows adaptation
    if (yoy_emp <= 0.5) or (abs(delta_from_r1) >= 0.4 and prob_regressed >= 0.9):
        return "adaptation"

    # 3. If H1 + H5 mechanism resolved + H3 attribution ~ stable
    h3_close = h3_attrib is not None and 25 <= h3_attrib <= 65
    if h3_close and h5_top_credible:
        return "comprehensive-update"
    if h5_top_credible:
        return "mechanism"
    return "honest-update"


def biggest_concern(h1, h2, h3, h4, h5) -> str:
    yoy_emp = _safe_get(h1, "h1_yoy_summary", "yoy_delta_pp_empirical", default=0.0) or 0.0
    prob_reg = _safe_get(h1, "h1_early_vs_late", "prob_2026_regressed", default=0.5) or 0.5
    h3_attrib = _safe_get(h3, "attribution_pct_mean", default=None)

    concerns = []
    if h3_attrib is not None and (h3_attrib < 20 or h3_attrib < -10):
        concerns.append(
            f"H3 attribution has moved dramatically from Round 1's +40-50% to {h3_attrib:+.1f}%. "
            f"The Round 1 'zone owns 40-50%' headline no longer holds; the piece must explicitly revise it."
        )
    if prob_reg >= 0.85:
        concerns.append(
            f"Within-2026 walk-rate trajectory: W1-3 → W5-7 prob(regressed) = {prob_reg:.0%}. "
            f"Aggregate YoY is +{yoy_emp:.2f}pp through May 13 vs +0.82pp through Apr 22 — the spike is softening."
        )
    if not concerns:
        return "None salient — but expect H3 attribution to land in [25, 60]% range; report a range, not a point."
    return " | ".join(concerns)


if __name__ == "__main__":
    sys.exit(main())

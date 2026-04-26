"""analyze.py — one-command driver for the Hot-Start Half-Life Round 1 analysis.

Runs:
  1. data_pull.py        (idempotent — skips already-cached items)
  2. stabilization.py    (split-half bootstrap, 200 iterations)
  3. bayes_projections.py(hierarchical Bayesian projections)
  4. changepoint.py      (PELT for 5 named starters)
  5. analogs_lite.py     (descriptive percentile lookup)
  6. assemble findings.json
  7. assemble noise-floor (top-5 22-game leaders 2022-2025 sustainment table)

Run:
  cd ./claude-analysis
  ./.venv/bin/python analyze.py
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path("<home>/Documents/GitHub/calledthird")
ROOT = REPO / "research/hot-start-half-life"
DATA = ROOT / "data"
CLAUDE = ROOT / "claude-analysis"
CHARTS = CLAUDE / "charts"
LOGS = CLAUDE / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


def _section(t: str) -> None:
    print()
    print("=" * 70)
    print(t)
    print("=" * 70)


def step_data_pull() -> None:
    _section("STEP 1: data_pull.py")
    import data_pull
    data_pull.main()


def step_stabilization(n_boot: int) -> dict:
    _section(f"STEP 2: stabilization.py  (n_boot={n_boot})")
    import stabilization
    t0 = time.time()
    summary = stabilization.run(n_boot=n_boot)
    print(f"[stab] elapsed {time.time()-t0:.0f}s")
    return summary


def step_bayes() -> dict:
    _section("STEP 3: bayes_projections.py")
    import bayes_projections
    t0 = time.time()
    out = bayes_projections.run()
    print(f"[bayes] elapsed {time.time()-t0:.0f}s")
    return out


def step_changepoint() -> dict:
    _section("STEP 4: changepoint.py")
    import changepoint
    t0 = time.time()
    out = changepoint.run()
    print(f"[changepoint] elapsed {time.time()-t0:.0f}s")
    return out


def step_analogs() -> dict:
    _section("STEP 5: analogs_lite.py")
    import analogs_lite
    t0 = time.time()
    out = analogs_lite.run()
    print(f"[analogs] elapsed {time.time()-t0:.0f}s")
    return out


# --- Noise-floor: 2022-2025 top-5 22-game leaders, ROS sustainment ---

def step_noise_floor() -> dict:
    _section("STEP 6: noise-floor table (top-5 22g leaders 2022-2025 sustainment)")
    from stabilization import load_pa_table, annotate_pa
    from analogs_lite import first_22g_split, player_season_rates

    pa = annotate_pa(load_pa_table((2022, 2023, 2024, 2025)))
    f22, rest = first_22g_split(pa)
    f22r = player_season_rates(f22)
    rstr = player_season_rates(rest)

    f22r = f22r[f22r.PA >= 50]   # qualify
    rstr = rstr[rstr.PA >= 100]
    pair = f22r.merge(rstr, on=["season", "batter"], suffixes=("_22g", "_ros"))

    rows = []
    for season, sub in pair.groupby("season"):
        sub = sub.sort_values("wOBA_22g", ascending=False).head(5)
        for _, r in sub.iterrows():
            sustained = r["wOBA_ros"] >= 0.85 * r["wOBA_22g"]
            rows.append({
                "season": int(season),
                "batter": int(r["batter"]),
                "PA_22g": int(r["PA_22g"]),
                "wOBA_22g": float(r["wOBA_22g"]),
                "wOBA_ros": float(r["wOBA_ros"]),
                "delta": float(r["wOBA_ros"] - r["wOBA_22g"]),
                "sustained_85pct": bool(sustained),
            })
    df = pd.DataFrame(rows)
    sustained_frac = float(df["sustained_85pct"].mean())
    out = {
        "criterion": "wOBA_ros >= 0.85 * wOBA_22g",
        "n_player_seasons": int(len(df)),
        "fraction_sustained": sustained_frac,
        "median_delta": float(df["delta"].median()),
        "rows": df.to_dict(orient="records"),
    }
    with open(DATA / "noise_floor.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[noise-floor] {len(df)} player-seasons, sustained 85% of 22g wOBA: "
          f"{sustained_frac:.0%}; median delta = {out['median_delta']:.3f}")
    return out


# --- Assemble final findings.json ---

def assemble_findings() -> dict:
    _section("STEP 7: assemble findings.json + summary")
    stab = json.load(open(DATA / "stabilization_summary.json"))
    bayes = json.load(open(DATA / "bayes_projections.json"))
    cps = json.load(open(DATA / "changepoints.json"))
    analogs = json.load(open(DATA / "analogs_lite.json"))
    noise = json.load(open(DATA / "noise_floor.json"))

    out = {
        "run_metadata": {
            "agent": "Claude (interpretability-first)",
            "round": 1,
            "data_through": "2026-04-24",
            "method": ["hierarchical-bayes-numpyro", "split-half-bootstrap-stabilization",
                       "PELT-changepoint", "percentile-analog-lookup"],
        },
        "stabilization": {
            stat: {
                "point": v["point_pa"],
                "ci_lo": v["ci_lo_pa"],
                "ci_hi": v["ci_hi_pa"],
                "carleton_ref": v["carleton_ref_pa"],
                "ratio_to_carleton": v["ratio_to_carleton"],
                "verdict": v["verdict"],
            }
            for stat, v in stab.items()
        },
        "league_env_2022_2025": bayes.get("league_env_2022_2025", {}),
        "projections": {},
        "changepoints": {slug: {"verdict": v.get("verdict"),
                                "n_career_pa": v.get("n_career_pa"),
                                "n_2026_pa": v.get("n_2026_pa")}
                          for slug, v in cps.items()},
        "analogs": analogs,
        "noise_floor": {
            "criterion": noise["criterion"],
            "n_player_seasons": noise["n_player_seasons"],
            "fraction_sustained": noise["fraction_sustained"],
            "median_delta": noise["median_delta"],
        },
    }

    # Projections in the requested findings.json format
    for slug, p in bayes.get("projections", {}).items():
        if "stats" not in p:
            out["projections"][slug] = {"error": p.get("error", "no projection")}
            continue
        ps = {}
        for stat, s in p["stats"].items():
            ps[stat] = {
                "point": s["q50"],
                "q10": s["q10"],
                "q50": s["q50"],
                "q80": s["q80"],
                "q90": s["q90"],
                "post_mean": s["post_mean"],
                "post_sd": s["post_sd"],
                "shrinkage_to_prior": s["shrinkage_weight_to_prior"],
                "obs_22g_rate": s["obs_22g_rate"],
                "prior_rate": s["prior_rate"],
                "rhat": s["rhat"],
                "ess": s["ess"],
            }
        out["projections"][slug] = {
            "PA_22g": p.get("PA_22g") or p.get("BF_22g"),
            "prior_kind": p.get("prior_kind", "weighted_3yr"),
            "stats": ps,
        }
        if "streak_simulation" in p:
            out["projections"][slug]["streak_simulation"] = p["streak_simulation"]
        if "BF_22g" in p:
            out["projections"][slug]["BF_22g"] = p["BF_22g"]

    # Per-starter signal/noise verdict per stat
    verdicts = {}
    for slug, p in out["projections"].items():
        if "stats" not in p:
            verdicts[slug] = {"overall": "unknown"}
            continue
        per_stat = {}
        for stat, s in p["stats"].items():
            obs = s["obs_22g_rate"]
            prior = s["prior_rate"]
            q10 = s["q10"]
            q90 = s["q90"]
            # A "signal" stat: posterior 80% interval excludes prior_rate
            if prior < q10 or prior > q90:
                v = "signal"
            elif min(abs(obs - prior), abs(s["q50"] - prior)) < 0.5 * (q90 - q10):
                v = "noise"  # interval substantially overlaps prior
            else:
                v = "ambiguous"
            per_stat[stat] = {"obs_22g": obs, "prior": prior, "q10": q10, "q90": q90,
                              "verdict_per_stat": v}
        # Overall — count signals vs noise vs ambiguous
        sig_count = sum(1 for v in per_stat.values() if v["verdict_per_stat"] == "signal")
        noise_count = sum(1 for v in per_stat.values() if v["verdict_per_stat"] == "noise")
        if sig_count >= 2 and noise_count <= 1:
            overall = "signal"
        elif noise_count >= 3 and sig_count == 0:
            overall = "noise"
        else:
            overall = "ambiguous"
        verdicts[slug] = {"overall": overall, "per_stat": per_stat}

    out["per_starter_verdicts"] = verdicts

    findings_path = CLAUDE / "findings.json"
    with open(findings_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[findings] saved -> {findings_path}")
    return out


def main(n_boot: int = 200, skip: tuple[str, ...] = ()) -> None:
    print(f"=== Hot-Start Half-Life — Claude analysis pipeline ===")
    print(f"Working dir: {CLAUDE}")
    print(f"Data dir:    {DATA}")
    print(f"Skipping:    {skip if skip else '(none)'}")

    if "data" not in skip:
        step_data_pull()
    if "stab" not in skip:
        step_stabilization(n_boot=n_boot)
    if "bayes" not in skip:
        step_bayes()
    if "changepoint" not in skip:
        step_changepoint()
    if "analogs" not in skip:
        step_analogs()
    if "noise" not in skip:
        step_noise_floor()
    assemble_findings()

    print()
    print("=" * 70)
    print("Pipeline complete.")
    print(f"  - findings.json @ {CLAUDE / 'findings.json'}")
    print(f"  - charts/       @ {CHARTS}")
    print(f"  - REPORT.md     -> next step (write manually)")
    print("=" * 70)


if __name__ == "__main__":
    n = int(os.environ.get("N_BOOT", "200"))
    skip = tuple(s.strip() for s in os.environ.get("SKIP", "").split(",") if s.strip())
    main(n_boot=n, skip=skip)

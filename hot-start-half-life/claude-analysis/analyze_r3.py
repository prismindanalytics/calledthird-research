"""analyze_r3.py — one-command R3 entry point for Claude (Agent A).

Pipeline order (each step skips if its output already exists; --force re-runs all):

  1. r3_blend_validation.py      -> learned blend on 2022-2024 -> 2025 holdout RMSE
  2. r3_stabilization.py         -> direct player-season bootstrap (Codex critique fix)
  3. r3_hierarchical_production  -> shared-kappa NUTS for wOBA AND xwOBA per player
  4. r3_universe_rerank.py       -> production posteriors + persistence atlas
  5. r3_reliever_board.py        -> Varland coherence fix + R3 sleeper/fake-dominant
  6. r3_named_verdicts.py        -> Pages, Rice, Trout, Murakami, Miller verdicts
  7. r3_convergence.py           -> claude-analysis/r3_convergence_check.json
  8. compile findings_r3.json    -> one bundle of all R3 outputs

Usage:
  python analyze_r3.py            # incremental
  python analyze_r3.py --force    # re-run everything
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"

sys.path.insert(0, str(CLAUDE))


def _exists(name: str) -> bool:
    return (DATA / name).exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # Step 1: blend validation
    print("\n=== Step 1: r3_blend_validation (learned blend on 2022-2024 -> 2025 holdout) ===")
    if args.force or not _exists("r3_blend_coefficients.json"):
        from r3_blend_validation import main as blend_main
        blend_main()
    else:
        print("[skip] r3_blend_coefficients.json exists")

    # Step 2: stabilization
    print("\n=== Step 2: r3_stabilization (player-season bootstrap direct) ===")
    if args.force or not _exists("r3_stabilization_summary.json"):
        from r3_stabilization import run as stab_run
        n = int(os.environ.get("N_BOOT", "150"))
        stab_run(n_boot=n)
    else:
        print("[skip] r3_stabilization_summary.json exists")

    # Step 3: hierarchical production
    print("\n=== Step 3: r3_hierarchical_production (shared-kappa NUTS) ===")
    if args.force or not _exists("r3_hierarchical_woba_per_player.parquet"):
        from r3_hierarchical_production import main as hier_main
        hier_main()
    else:
        print("[skip] r3_hierarchical_woba_per_player.parquet exists")

    # Step 4: universe re-rank
    print("\n=== Step 4: r3_universe_rerank (learned blend over hierarchical posterior) ===")
    if args.force or not _exists("r3_universe_posteriors.parquet"):
        from r3_universe_rerank import main as rerank_main
        rerank_main()
    else:
        print("[skip] r3_universe_posteriors.parquet exists")

    # Step 5: reliever board
    print("\n=== Step 5: r3_reliever_board (Varland coherence) ===")
    if args.force or not _exists("r3_reliever_board.json"):
        from r3_reliever_board import main as rel_main
        rel_main()
    else:
        print("[skip] r3_reliever_board.json exists")

    # Step 6: named verdicts
    print("\n=== Step 6: r3_named_verdicts (Pages, Rice, Trout, Murakami, Miller) ===")
    if args.force or not _exists("r3_named_starter_projections.json"):
        from r3_named_verdicts import main as named_main
        named_main()
    else:
        print("[skip] r3_named_starter_projections.json exists")

    # Step 7: convergence check
    print("\n=== Step 7: r3_convergence (write r3_convergence_check.json) ===")
    from r3_convergence import main as conv_main
    conv_main()

    # Step 8: compile findings_r3.json
    print("\n=== Step 8: compile findings_r3.json ===")
    compile_findings()


def compile_findings() -> None:
    out = {
        "_round": 3,
        "_as_of": "2026-04-25",
        "blocking_fixes_status": {
            "fix_1_contact_quality_blend": {
                "status": "done",
                "path_taken": "A: validated learned blend",
                "decision": json.load(open(DATA / "r3_blend_coefficients.json"))["decision"],
                "rationale": json.load(open(DATA / "r3_blend_coefficients.json"))["decision_rationale"],
                "validation": json.load(open(DATA / "r3_blend_validation.json"))["holdout"],
            },
            "fix_2_hierarchical_labeling": {
                "status": "done",
                "path_taken": "A: integrated hierarchical fit into production",
                "module": "r3_hierarchical_production.py",
                "summary": json.load(open(DATA / "r3_hierarchical_summary.json")),
            },
            "fix_3_stabilization_bootstrap_estimand": {
                "status": "done",
                "module": "r3_stabilization.py",
                "method": ("Direct player-season bootstrap: sample N player-seasons "
                           "from 1,433 qualifying with replacement (no cap at one "
                           "season per player)."),
                "summary": json.load(open(DATA / "r3_stabilization_summary.json")),
            },
            "fix_4_varland_coherence": {
                "status": "done",
                "module": "r3_reliever_board.py",
                "rule": ("Sleeper and fake-dominant lists are MUTUALLY EXCLUSIVE: "
                         "a reliever flagged fake-dominant cannot also be on sleeper "
                         "list. Varland excluded from sleeper; remains on fake-dominant."),
                "varland_choice": "FAKE-DOMINANT",
            },
        },
        "convergence_check": json.load(open(CLAUDE / "r3_convergence_check.json")),
        "named_starter_verdicts_full": json.load(open(DATA / "r3_named_starter_projections.json")),
        "persistence_atlas": json.load(open(DATA / "r3_persistence_atlas.json")),
        "reliever_board": json.load(open(DATA / "r3_reliever_board.json")),
        "stabilization_summary": json.load(open(DATA / "r3_stabilization_summary.json")),
        "hierarchical_summary": json.load(open(DATA / "r3_hierarchical_summary.json")),
        "blend_validation": json.load(open(DATA / "r3_blend_validation.json")),
    }
    with open(CLAUDE / "findings_r3.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[findings] saved to {CLAUDE / 'findings_r3.json'}")


if __name__ == "__main__":
    main()

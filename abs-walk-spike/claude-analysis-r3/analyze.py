"""analyze.py — Round 3 entry point for the Claude (Agent A) ABS walk-spike analysis.

Runs:
  1. data_prep_r3 (cached)
  2. archetype_build (cached)
  3. R3-H1 triangulation (Methods A, B, C)
  4. R3-H2 named adapter leaderboard (Bayesian + bootstrap stability)
  5. R3-H3 archetype × walk-rate-change Bayesian interaction + leaderboards
  6. Aggregates findings.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import R3_ARTIFACTS, R3_DATA, R3_DIR, ensure_dirs, load_2025_full
import data_prep_r3
import archetype_build
import h1_triangulation
import h2_adapter_leaderboard
import h3_archetype_interaction


FINDINGS_PATH = R3_DIR / "findings.json"


def main(
    *,
    n_draws_a: int = 60,
    n_outer_c: int = 100,
    n_inner_c: int = 10,
    n_boot_b: int = 200,
    n_boot_h2: int = 200,
    n_boot_h3: int = 200,
) -> dict:
    ensure_dirs()
    print("=" * 80); print("R3 ANALYSIS — Claude (Agent A)"); print("=" * 80)

    t0 = time.time()
    print("\n[STEP 1] data_prep_r3 ...")
    data_meta = data_prep_r3.build_all(force=False)
    print(f"  data_prep_r3 elapsed={time.time()-t0:.1f}s")

    print("\n[STEP 2] archetype_build ...")
    arch = archetype_build.build(force=False)
    print(f"  archetype_build elapsed={time.time()-t0:.1f}s")

    print("\n[STEP 3] R3-H1 triangulation ...")
    t1 = time.time()
    h1_out = h1_triangulation.run_all(
        n_draws_a=n_draws_a, n_outer_c=n_outer_c, n_inner_c=n_inner_c, n_boot_b=n_boot_b,
    )
    print(f"  R3-H1 elapsed={time.time()-t1:.1f}s")

    print("\n[STEP 4] R3-H2 adapter leaderboard ...")
    t1 = time.time()
    h2_out = h2_adapter_leaderboard.main(n_bootstrap=n_boot_h2)
    print(f"  R3-H2 elapsed={time.time()-t1:.1f}s")

    print("\n[STEP 5] R3-H3 archetype interaction ...")
    t1 = time.time()
    panel_2025_full = load_2025_full()
    panel_2026 = data_prep_r3.get_panel_2026()
    h3_out = h3_archetype_interaction.main(panel_2025_full, panel_2026, n_bootstrap=n_boot_h3)
    print(f"  R3-H3 elapsed={time.time()-t1:.1f}s")

    print("\n[STEP 6] writing findings.json ...")
    findings = {
        "round": 3,
        "agent": "claude",
        "data_meta": data_meta,
        "archetype_meta": json.loads((R3_DATA / "pitcher_archetype_meta.json").read_text()),
        "h1_triangulation": {
            "method_a": h1_out["method_a"],
            "method_b": h1_out["method_b"],
            "method_c": h1_out["method_c"],
            "triangulated": h1_out["triangulated"],
        },
        "h2_adapter_leaderboard": h2_out,
        "h3_archetype_interaction": h3_out,
        "elapsed_total_seconds": float(time.time() - t0),
    }
    FINDINGS_PATH.write_text(json.dumps(findings, indent=2, default=float))
    print(f"  findings.json → {FINDINGS_PATH}")

    print(f"\n[DONE] total elapsed={time.time()-t0:.1f}s")
    return findings


if __name__ == "__main__":
    import sys
    # Allow quick smoke test with smaller N via env or argv
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        main(n_draws_a=10, n_outer_c=10, n_inner_c=2, n_boot_b=20, n_boot_h2=20, n_boot_h3=20)
    else:
        main()

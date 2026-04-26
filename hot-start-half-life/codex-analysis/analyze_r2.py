from __future__ import annotations

import importlib
import sys
from pathlib import Path


MODULES = [
    "data_pull",
    "features",
    "r2_universe",
    "r2_qrf_coverage",
    "r2_persistence_atlas",
    "r2_xwoba_gap",
    "r2_reliever_board",
    "r2_analog_retrieval",
    "r2_reporting",
]


def main() -> None:
    base = Path(__file__).resolve().parent
    sys.path.insert(0, str(base))
    for name in MODULES:
        print(f"[analyze_r2] running {name}", flush=True)
        module = importlib.import_module(name)
        module.main()
    print("[analyze_r2] complete", flush=True)


if __name__ == "__main__":
    main()


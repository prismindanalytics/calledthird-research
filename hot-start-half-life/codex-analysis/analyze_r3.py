from __future__ import annotations

import importlib
import sys
from pathlib import Path


MODULES = [
    "r3_calibration",
    "r3_persistence_atlas",
    "r3_reliever_board",
    "r3_named_verdicts",
    "r3_convergence",
    "r3_reporting",
]


def main() -> None:
    base = Path(__file__).resolve().parent
    sys.path.insert(0, str(base))
    for name in MODULES:
        print(f"[analyze_r3] running {name}", flush=True)
        module = importlib.import_module(name)
        module.main()
    print("[analyze_r3] complete", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib
import sys
from pathlib import Path


MODULES = [
    "data_pull",
    "features",
    "lgbm_projections",
    "qrf_intervals",
    "shap_analysis",
    "counterfactual",
    "analog_retrieval",
    "reporting",
]


def main() -> None:
    base = Path(__file__).resolve().parent
    sys.path.insert(0, str(base))
    for name in MODULES:
        print(f"[analyze] running {name}", flush=True)
        module = importlib.import_module(name)
        module.main()
    print("[analyze] complete", flush=True)


if __name__ == "__main__":
    main()

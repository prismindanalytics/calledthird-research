from __future__ import annotations

import json

import pandas as pd

from common import BASE_DIR, atomic_write_json, read_json
from r3_utils import R3_MODELS_DIR, R3_TABLES_DIR


def changed_reason(player: str, r2: str, r3: str) -> str:
    if player == "Mason Miller":
        return "Reliever board was rerun with raw QRF framing and the R3 prior floor for sleeper lists."
    if r2 != r3:
        return "R3 rerun used raw QRF framing plus corrected ranking and fake-hot rules."
    return ""


def main() -> dict:
    named_payload = read_json(R3_MODELS_DIR / "r3_named_verdicts.json", {})
    named_records = named_payload.get("records", [])
    named_verdicts = named_payload.get("named_starter_verdicts", {})

    sleepers = pd.read_csv(R3_TABLES_DIR / "r3_sleepers.csv")
    relievers = pd.read_csv(R3_TABLES_DIR / "r3_reliever_sleepers.csv")

    changed = []
    for rec in named_records:
        r2 = rec.get("r2_verdict")
        r3 = rec.get("r3_verdict")
        if r2 and r3 and r2 != r3:
            changed.append(
                {
                    "player": rec["player"],
                    "r2": r2,
                    "r3": r3,
                    "reason": changed_reason(rec["player"], r2, r3),
                }
            )

    killed = ["Tristan Peters (zero-prior hitter sleeper removed by preseason_prior_woba > 0 filter)"]
    if "Cole Wilcox" not in relievers["player"].tolist():
        killed.append("Cole Wilcox (low-prior reliever dropped by R3 prior K% floor)")
    if "Louis Varland" not in relievers["player"].tolist():
        killed.append("Louis Varland (excluded from R3 sleeper-reliever list; remains fake-dominant)")

    payload = {
        "named_starter_verdicts": named_verdicts,
        "top10_sleeper_hitters": sleepers["player"].head(10).tolist(),
        "top5_sleeper_relievers": relievers["player"].head(5).tolist(),
        "killed_picks_from_r2": killed,
        "verdicts_changed_from_r2": changed,
    }
    atomic_write_json(BASE_DIR / "r3_convergence_check.json", payload)
    atomic_write_json(R3_MODELS_DIR / "r3_convergence_check.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import atomic_write_json, set_plot_style
from r3_reliever_board import reliever_verdict
from r3_utils import (
    NAMED_HITTER_KEYS,
    NAMED_PITCHER_KEYS,
    R3_CHARTS_DIR,
    R3_MODELS_DIR,
    R3_TABLES_DIR,
    records_for_json,
)


HITTER_ORDER = ["andy_pages", "ben_rice", "mike_trout", "munetaka_murakami"]
PITCHER_ORDER = ["mason_miller"]


def hitter_verdict(row: pd.Series) -> str:
    q10 = row.get("delta_q10", np.nan)
    q90 = row.get("delta_q90", np.nan)
    mean = row.get("pred_delta_mean", np.nan)
    if np.isfinite(q10) and q10 > 0.010:
        return "SIGNAL"
    if np.isfinite(q10) and np.isfinite(q90) and q10 <= 0 <= q90 and abs(mean) < 0.020:
        return "NOISE"
    return "AMBIGUOUS"


def hitter_confidence(verdict: str, row: pd.Series) -> str:
    mean = float(row.get("pred_delta_mean", 0.0))
    q10 = float(row.get("delta_q10", np.nan))
    q90 = float(row.get("delta_q90", np.nan))
    interval_crosses_zero = np.isfinite(q10) and np.isfinite(q90) and q10 <= 0 <= q90
    if verdict == "SIGNAL":
        return "high" if q10 > 0.020 else "medium"
    if verdict == "NOISE":
        return "high" if abs(mean) < 0.006 and interval_crosses_zero else "medium"
    return "low" if abs(mean) > 0.035 and interval_crosses_zero else "medium"


def pitcher_confidence(verdict: str, row: pd.Series) -> str:
    if verdict == "SIGNAL":
        return "medium"
    if verdict == "NOISE":
        return "medium"
    return "medium"


def fmt_woba(value: float) -> str:
    return f"{float(value):.3f}" if pd.notna(value) else "NA"


def fmt_delta(value: float) -> str:
    return f"{float(value):+.3f}" if pd.notna(value) else "NA"


def fmt_pct(value: float) -> str:
    return f"{100 * float(value):.1f}%" if pd.notna(value) else "NA"


def hitter_evidence(row: pd.Series) -> str:
    return (
        f"April wOBA {fmt_woba(row['woba_cutoff'])}, prior {fmt_woba(row['preseason_prior_woba'])}, "
        f"projected ROS {fmt_woba(row['pred_ros_woba'])} with raw 80% delta band "
        f"[{fmt_delta(row['delta_q10'])}, {fmt_delta(row['delta_q90'])}]."
    )


def pitcher_evidence(row: pd.Series) -> str:
    return (
        f"April K% {fmt_pct(row['k_rate_cutoff'])}, prior {fmt_pct(row['preseason_prior_k_rate'])}, "
        f"projected ROS {fmt_pct(row['pred_k_rate_mean'])} with raw 80% K% band "
        f"[{fmt_pct(row['k_rate_q10'])}, {fmt_pct(row['k_rate_q90'])}]."
    )


def plot_verdict_table(named: list[dict]) -> None:
    set_plot_style()
    rows = [[rec["player"], rec.get("r2_verdict", "NA"), rec["r3_verdict"], rec["confidence"]] for rec in named]
    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Player", "R2", "R3", "Confidence"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Named Starter Verdicts: R2 vs R3")
    fig.tight_layout()
    fig.savefig(R3_CHARTS_DIR / "r3_named_verdict_comparison.png")
    plt.close(fig)


def load_r2_hitter_verdicts() -> dict[int, str]:
    path = R3_TABLES_DIR.parent.parent / "round2" / "tables" / "r2_r1_sanity_hitters.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(row["batter"]): str(row.get("r2_verdict", "")) for _, row in df.iterrows()}


def load_r2_pitcher_verdicts() -> dict[int, str]:
    path = R3_TABLES_DIR.parent.parent / "round2" / "tables" / "r2_r1_sanity_reliever.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(row["pitcher"]): str(row.get("r2_verdict", "")) for _, row in df.iterrows()}


def main() -> dict:
    hitter_pred = pd.read_csv(R3_TABLES_DIR / "r3_persistence_predictions.csv")
    reliever_pred = pd.read_csv(R3_TABLES_DIR / "r3_reliever_board.csv")
    r2_hitters = load_r2_hitter_verdicts()
    r2_pitchers = load_r2_pitcher_verdicts()

    named_rows: list[dict] = []
    convergence: dict[str, dict[str, str]] = {}
    for key in HITTER_ORDER:
        mlbam = NAMED_HITTER_KEYS[key]
        row = hitter_pred[hitter_pred["batter"].eq(mlbam)].iloc[0]
        verdict = hitter_verdict(row)
        confidence = hitter_confidence(verdict, row)
        evidence = hitter_evidence(row)
        rec = {
            "key": key,
            "player": row["player"],
            "player_type": "hitter",
            "mlbam": int(mlbam),
            "r2_verdict": r2_hitters.get(int(mlbam)),
            "r3_verdict": verdict,
            "confidence": confidence,
            "evidence": evidence,
            **records_for_json(pd.DataFrame([row]), [
                "pa_cutoff",
                "woba_cutoff",
                "preseason_prior_woba",
                "pred_delta_mean",
                "delta_q10",
                "delta_q90",
                "pred_ros_woba",
                "q10_ros_woba",
                "q90_ros_woba",
                "xwoba_minus_prior_woba_22g",
            ])[0],
        }
        named_rows.append(rec)
        convergence[key] = {"verdict": verdict, "confidence": confidence, "evidence": evidence}

    for key in PITCHER_ORDER:
        mlbam = NAMED_PITCHER_KEYS[key]
        row = reliever_pred[reliever_pred["pitcher"].eq(mlbam)].iloc[0]
        verdict = reliever_verdict(row)
        confidence = pitcher_confidence(verdict, row)
        evidence = pitcher_evidence(row)
        rec = {
            "key": key,
            "player": row["player"],
            "player_type": "reliever",
            "mlbam": int(mlbam),
            "r2_verdict": r2_pitchers.get(int(mlbam)),
            "r3_verdict": verdict,
            "confidence": confidence,
            "evidence": evidence,
            **records_for_json(pd.DataFrame([row]), [
                "bf_cutoff",
                "ip_cutoff",
                "k_rate_cutoff",
                "preseason_prior_k_rate",
                "pred_k_rate_mean",
                "k_rate_q10",
                "k_rate_q90",
                "pred_k_delta_vs_prior",
            ])[0],
        }
        named_rows.append(rec)
        convergence[key] = {"verdict": verdict, "confidence": confidence, "evidence": evidence}

    pd.DataFrame(named_rows).to_csv(R3_TABLES_DIR / "r3_named_verdicts.csv", index=False)
    plot_verdict_table(named_rows)
    payload = {
        "named_starter_verdicts": convergence,
        "records": named_rows,
    }
    atomic_write_json(R3_MODELS_DIR / "r3_named_verdicts.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from common import DATASETS_DIR, HITTER_ANALOG_COLS, atomic_write_json
from r2_utils import R2_MODELS_DIR, R2_TABLES_DIR, add_r2_hitter_columns, load_name_map, pretty_player_name


MIN_SIM = 0.70


ANALOG_COLS = [
    "pa_22g",
    "bb_rate_22g",
    "k_rate_22g",
    "babip_22g",
    "iso_22g",
    "ops_22g",
    "ev_p90_22g",
    "hardhit_rate_22g",
    "barrel_rate_22g",
    "xwoba_22g",
    "xwoba_minus_woba_22g",
    "xwoba_minus_prior_woba_22g",
    "whiff_per_swing_22g",
    "chase_rate_22g",
]


def load_pick_pool() -> pd.DataFrame:
    pred = pd.read_parquet(DATASETS_DIR / "r2_persistence_predictions.parquet")
    tables = []
    for path, label in [
        (R2_TABLES_DIR / "r2_sleepers.csv", "sleeper"),
        (R2_TABLES_DIR / "r2_fake_hot.csv", "fake_hot"),
        (R2_TABLES_DIR / "r2_fake_cold.csv", "fake_cold"),
    ]:
        if path.exists() and path.stat().st_size:
            df = pd.read_csv(path)
            df["pick_type"] = label
            tables.append(df)
    if not tables:
        return pd.DataFrame(columns=["batter", "pick_type"])
    picks = pd.concat(tables, ignore_index=True)
    picks = picks.drop_duplicates(["batter", "pick_type"])
    return picks.merge(pred, on=[col for col in ["batter", "player"] if col in picks.columns and col in pred.columns], how="left", suffixes=("", "_pred"))


def compute_neighbors(hist: pd.DataFrame, target: pd.DataFrame, cols: list[str]) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_hist = scaler.fit_transform(imputer.fit_transform(hist[cols]))
    x_target = scaler.transform(imputer.transform(target[cols]))
    return cosine_similarity(x_target, x_hist)[0]


def main() -> dict:
    names = load_name_map()
    hist = pd.read_parquet(DATASETS_DIR / "hitter_features.parquet")
    hist = add_r2_hitter_columns(hist)
    pool = hist[hist["season"].between(2015, 2024) & hist["pa_22g"].ge(50) & hist["pa_ros"].ge(50)].copy()
    picks = load_pick_pool()
    if picks.empty:
        payload = {"min_cosine_similarity": MIN_SIM, "analogs": {}}
        atomic_write_json(R2_MODELS_DIR / "r2_analogs.json", payload)
        return payload
    for col in ANALOG_COLS:
        if col not in pool.columns:
            pool[col] = np.nan
        if col not in picks.columns:
            picks[col] = np.nan

    rows = []
    result = {}
    for _, pick in picks.iterrows():
        target = pd.DataFrame([pick])
        sims = compute_neighbors(pool, target, ANALOG_COLS)
        tmp = pool.copy()
        tmp["cosine_sim"] = sims
        top = tmp[tmp["cosine_sim"].ge(MIN_SIM)].sort_values("cosine_sim", ascending=False).head(5)
        key = f"{pick['pick_type']}:{int(pick['batter'])}"
        analogs = []
        for _, row in top.iterrows():
            rec = {
                "pick_type": pick["pick_type"],
                "target_player": pick.get("player", f"MLBAM {int(pick['batter'])}"),
                "target_mlbam": int(pick["batter"]),
                "analog_player": pretty_player_name(names.get(int(row["batter"]), f"MLBAM {int(row['batter'])}")),
                "analog_mlbam": int(row["batter"]),
                "season": int(row["season"]),
                "cosine_sim": float(row["cosine_sim"]),
                "first22_woba": float(row.get("woba_22g", np.nan)),
                "first22_xwoba_gap": float(row.get("xwoba_minus_woba_22g", np.nan)),
                "ros_woba": float(row.get("ros_woba", np.nan)),
                "ros_delta_vs_prior": float(row.get("ros_woba_delta_vs_prior", np.nan)),
                "ros_iso": float(row.get("ros_iso", np.nan)),
                "ros_k_rate": float(row.get("ros_k_rate", np.nan)),
            }
            rows.append(rec)
            analogs.append({k: v for k, v in rec.items() if not k.startswith("target_") and k != "pick_type"})
        result[key] = {
            "pick_type": pick["pick_type"],
            "target_player": pick.get("player", f"MLBAM {int(pick['batter'])}"),
            "target_mlbam": int(pick["batter"]),
            "n_analogs_ge_0_70": int(len(top)),
            "kill_gate_passed": bool(len(top) >= 5),
            "analogs": analogs,
        }
    pd.DataFrame(rows).to_csv(R2_TABLES_DIR / "r2_hitter_analogs.csv", index=False)
    payload = {"min_cosine_similarity": MIN_SIM, "feature_cols": ANALOG_COLS, "analogs": result}
    atomic_write_json(R2_MODELS_DIR / "r2_analogs.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

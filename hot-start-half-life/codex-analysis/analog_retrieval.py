from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from common import (
    CHARTS_DIR,
    DATASETS_DIR,
    HITTER_ANALOG_COLS,
    PITCHER_ANALOG_COLS,
    TABLES_DIR,
    atomic_write_json,
    read_json,
    set_plot_style,
)


MIN_SIM = 0.70


def load_names() -> dict[int, str]:
    path = DATASETS_DIR / "name_map.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("key_mlbam")) and pd.notna(row.get("name")):
            out[int(row["key_mlbam"])] = str(row["name"]).strip()
    return out


def compute_neighbors(hist: pd.DataFrame, target: pd.DataFrame, cols: list[str]) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_hist = scaler.fit_transform(imputer.fit_transform(hist[cols]))
    x_target = scaler.transform(imputer.transform(target[cols]))
    return cosine_similarity(x_target, x_hist)[0]


def plot_hitter_analog(player_key: str, player: str, analogs: list[dict], target_row: pd.Series) -> None:
    if not analogs:
        return
    set_plot_style()
    labels = [f"{a['player']} {a['year']}" for a in analogs]
    first = [a.get("first22_woba") for a in analogs]
    ros = [a.get("ros_woba") for a in analogs]
    x = np.arange(len(analogs))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, first, marker="o", color="#86bbd8", label="analog first 22")
    ax.plot(x, ros, marker="o", color="#2f4858", label="analog ROS")
    ax.axhline(target_row.get("woba_22g", np.nan), color="#f26419", linestyle="--", label=f"{player} first 22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("wOBA")
    ax.set_title(f"Nearest Analog Trajectories: {player}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / f"analog_{player_key}.png")
    plt.close(fig)


def plot_pitcher_analog(player_key: str, player: str, analogs: list[dict], target_row: pd.Series) -> None:
    if not analogs:
        return
    set_plot_style()
    labels = [f"{a['player']} {a['year']}" for a in analogs]
    first = [a.get("first22_ra9") for a in analogs]
    ros = [a.get("ros_ra9") for a in analogs]
    x = np.arange(len(analogs))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, first, marker="o", color="#86bbd8", label="analog first 22")
    ax.plot(x, ros, marker="o", color="#2f4858", label="analog ROS")
    ax.axhline(target_row.get("ra9_22g", np.nan), color="#f26419", linestyle="--", label=f"{player} first 22")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("RA9 proxy")
    ax.set_title(f"Nearest Analog Trajectories: {player}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / f"analog_{player_key}.png")
    plt.close(fig)


def hitter_analogs(names: dict[int, str]) -> dict:
    hist = pd.read_parquet(DATASETS_DIR / "hitter_features.parquet")
    target = pd.read_parquet(DATASETS_DIR / "hitter_features_2026.parquet")
    named = read_json(DATASETS_DIR / "named_players.json", {})["hitters"]
    pool = hist[hist["season"].between(2015, 2024) & hist["pa_22g"].ge(50) & hist["pa_ros"].ge(50)].copy()
    result = {}
    rows = []
    for key, info in named.items():
        mlbam = info.get("mlbam")
        if mlbam is None:
            result[key] = []
            continue
        t = target[target["batter"].eq(mlbam)]
        if t.empty or float(t.iloc[0].get("pa_22g", 0)) < 50:
            result[key] = []
            continue
        sims = compute_neighbors(pool, t, HITTER_ANALOG_COLS)
        tmp = pool.copy()
        tmp["cosine_sim"] = sims
        top = tmp[tmp["cosine_sim"].ge(MIN_SIM)].sort_values("cosine_sim", ascending=False).head(5)
        analogs = []
        for _, row in top.iterrows():
            rec = {
                "player": names.get(int(row["batter"]), f"MLBAM {int(row['batter'])}"),
                "year": int(row["season"]),
                "mlbam": int(row["batter"]),
                "first22_woba": float(row.get("woba_22g", np.nan)),
                "ros_woba": float(row.get("ros_woba", np.nan)),
                "ros_babip": float(row.get("ros_babip", np.nan)),
                "ros_iso": float(row.get("ros_iso", np.nan)),
                "cosine_sim": float(row["cosine_sim"]),
            }
            analogs.append(rec)
            rows.append({"player_key": key, "player": info["name"], **rec})
        result[key] = analogs
        plot_hitter_analog(key, info["name"], analogs, t.iloc[0])
    pd.DataFrame(rows).to_csv(TABLES_DIR / "hitter_analogs.csv", index=False)
    return result


def pitcher_analogs(names: dict[int, str]) -> dict:
    hist = pd.read_parquet(DATASETS_DIR / "pitcher_features.parquet")
    target = pd.read_parquet(DATASETS_DIR / "pitcher_features_2026.parquet")
    named = read_json(DATASETS_DIR / "named_players.json", {})["pitchers"]
    pool = hist[
        hist["season"].between(2015, 2024) & hist["bf_22g"].ge(25) & hist["bf_ros"].ge(25) & hist["ip_full"].le(95)
    ].copy()
    result = {}
    rows = []
    for key, info in named.items():
        mlbam = info.get("mlbam")
        t = target[target["pitcher"].eq(mlbam)]
        if t.empty or float(t.iloc[0].get("bf_22g", 0)) < 25:
            result[key] = []
            continue
        sims = compute_neighbors(pool, t, PITCHER_ANALOG_COLS)
        tmp = pool.copy()
        tmp["cosine_sim"] = sims
        top = tmp[tmp["cosine_sim"].ge(MIN_SIM)].sort_values("cosine_sim", ascending=False).head(5)
        analogs = []
        for _, row in top.iterrows():
            rec = {
                "player": names.get(int(row["pitcher"]), f"MLBAM {int(row['pitcher'])}"),
                "year": int(row["season"]),
                "mlbam": int(row["pitcher"]),
                "first22_ra9": float(row.get("ra9_22g", np.nan)),
                "ros_ra9": float(row.get("ros_ra9", np.nan)),
                "ros_k_rate": float(row.get("ros_k_rate", np.nan)),
                "ros_bb_rate": float(row.get("ros_bb_rate", np.nan)),
                "cosine_sim": float(row["cosine_sim"]),
            }
            analogs.append(rec)
            rows.append({"player_key": key, "player": info["name"], **rec})
        result[key] = analogs
        plot_pitcher_analog(key, info["name"], analogs, t.iloc[0])
    pd.DataFrame(rows).to_csv(TABLES_DIR / "pitcher_analogs.csv", index=False)
    return result


def main() -> dict:
    names = load_names()
    payload = {"hitters": hitter_analogs(names), "pitchers": pitcher_analogs(names), "min_cosine_similarity": MIN_SIM}
    atomic_write_json(TABLES_DIR / "analogs.json", payload)
    return payload


if __name__ == "__main__":
    main()

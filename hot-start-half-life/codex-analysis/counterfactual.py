from __future__ import annotations

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.impute import SimpleImputer

from common import (
    CHARTS_DIR,
    DATASETS_DIR,
    HITTER_FEATURE_COLS,
    SEED,
    TABLES_DIR,
    atomic_write_json,
    read_json,
    set_plot_style,
)


N_BOOT = 100


def fit_predict(train: pd.DataFrame, target_rows: pd.DataFrame, seed: int) -> np.ndarray:
    sampled = train.sample(frac=1.0, replace=True, random_state=seed)
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(sampled[HITTER_FEATURE_COLS])
    x_target = imputer.transform(target_rows[HITTER_FEATURE_COLS])
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=220,
        learning_rate=0.045,
        num_leaves=17,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=1,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(x_train, sampled["ros_woba"].to_numpy())
    return model.predict(x_target)


def main() -> dict:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    hitters = pd.read_parquet(DATASETS_DIR / "hitter_features.parquet")
    hitters_2026 = pd.read_parquet(DATASETS_DIR / "hitter_features_2026.parquet")
    named = read_json(DATASETS_DIR / "named_players.json", {})["hitters"]
    pool = hitters[hitters["pa_22g"].ge(50) & hitters["pa_ros"].ge(50) & hitters["ros_woba"].notna()].copy()
    broad = pool[pool["season"].between(2015, 2024)].copy()
    current = pool[pool["season"].between(2022, 2025)].copy()

    target_rows = []
    target_meta = []
    for key, info in named.items():
        mlbam = info.get("mlbam")
        if mlbam is None:
            continue
        row = hitters_2026[hitters_2026["batter"].eq(mlbam)]
        if row.empty or float(row.iloc[0].get("pa_22g", 0)) < 50:
            continue
        target_rows.append(row.iloc[0])
        target_meta.append({"player_key": key, "player": info["name"], "mlbam": int(mlbam)})
    if not target_rows:
        payload = {"avg_delta": None, "ci": [None, None], "n_boot": N_BOOT, "players": []}
        atomic_write_json(TABLES_DIR / "counterfactual_summary.json", payload)
        return payload
    targets = pd.DataFrame(target_rows)

    records = []
    for i in range(N_BOOT):
        seed = SEED + i
        broad_pred = fit_predict(broad, targets, seed)
        current_pred = fit_predict(current, targets, seed + 10000)
        for meta, b_pred, c_pred in zip(target_meta, broad_pred, current_pred):
            records.append(
                {
                    **meta,
                    "seed": seed,
                    "broad_2015_2024": float(b_pred),
                    "current_2022_2025": float(c_pred),
                    "delta_current_minus_broad": float(c_pred - b_pred),
                }
            )
    df = pd.DataFrame(records)
    df.to_csv(TABLES_DIR / "counterfactual_deltas.csv", index=False)

    players = []
    for key, grp in df.groupby("player_key"):
        players.append(
            {
                "player_key": key,
                "player": grp["player"].iloc[0],
                "mean_broad": float(grp["broad_2015_2024"].mean()),
                "mean_current": float(grp["current_2022_2025"].mean()),
                "mean_delta": float(grp["delta_current_minus_broad"].mean()),
                "ci": [
                    float(np.percentile(grp["delta_current_minus_broad"], 2.5)),
                    float(np.percentile(grp["delta_current_minus_broad"], 97.5)),
                ],
            }
        )
    avg_by_seed = df.groupby("seed")["delta_current_minus_broad"].mean()
    payload = {
        "avg_delta": float(avg_by_seed.mean()),
        "ci": [float(np.percentile(avg_by_seed, 2.5)), float(np.percentile(avg_by_seed, 97.5))],
        "n_boot": N_BOOT,
        "players": players,
    }
    atomic_write_json(TABLES_DIR / "counterfactual_summary.json", payload)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = pd.DataFrame(players).sort_values("mean_delta")
    y = np.arange(len(plot_df))
    ax.axvline(0, color="#333333", lw=1)
    ax.hlines(y, [c[0] for c in plot_df["ci"]], [c[1] for c in plot_df["ci"]], color="#86bbd8", lw=5)
    ax.scatter(plot_df["mean_delta"], y, color="#2f4858", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["player"])
    ax.set_xlabel("Current-era minus broad-era predicted ROS wOBA")
    ax.set_title("Era Counterfactual Deltas (100 Bootstrap Seeds)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "counterfactual_deltas.png")
    plt.close(fig)
    return payload


if __name__ == "__main__":
    main()

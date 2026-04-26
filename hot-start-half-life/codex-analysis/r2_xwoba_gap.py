from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATASETS_DIR, atomic_write_json, set_plot_style
from r2_utils import R2_CHARTS_DIR, R2_MODELS_DIR, R2_TABLES_DIR


def main() -> dict:
    pred = pd.read_parquet(DATASETS_DIR / "r2_persistence_predictions.parquet")
    pred["gap_xwoba_minus_woba"] = pred["xwoba_minus_woba_22g"]
    pred["abs_gap_xwoba_minus_woba"] = pred["gap_xwoba_minus_woba"].abs()
    pred["gap_side"] = np.where(
        pred["gap_xwoba_minus_woba"].gt(0),
        "under-performing xwOBA (actual wOBA below expected)",
        "over-performing xwOBA (actual wOBA above expected)",
    )
    top = pred.sort_values("abs_gap_xwoba_minus_woba", ascending=False).head(10).copy()
    top_cols = [
        "player",
        "batter",
        "team",
        "pa_cutoff",
        "woba_cutoff",
        "xwoba_22g",
        "gap_xwoba_minus_woba",
        "abs_gap_xwoba_minus_woba",
        "gap_side",
        "pred_delta_mean",
        "verdict",
        "in_mainstream_top20",
    ]
    top[top_cols].to_csv(R2_TABLES_DIR / "r2_xwoba_gap_top10.csv", index=False)

    crosstab = (
        pred.assign(gap_bucket=pd.qcut(pred["abs_gap_xwoba_minus_woba"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]))
        .groupby(["gap_bucket", "verdict"], observed=True)
        .size()
        .reset_index(name="n")
    )
    crosstab.to_csv(R2_TABLES_DIR / "r2_xwoba_gap_verdict_crosstab.csv", index=False)

    colors = np.where(top["gap_xwoba_minus_woba"].ge(0), "#2a9d8f", "#d62828")
    frame = top.sort_values("gap_xwoba_minus_woba")
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(frame["player"], frame["gap_xwoba_minus_woba"], color=np.where(frame["gap_xwoba_minus_woba"].ge(0), "#2a9d8f", "#d62828"))
    ax.axvline(0, color="#333333", lw=1)
    ax.set_xlabel("xwOBA - wOBA through cutoff")
    ax.set_title("Largest xwOBA/WOBA Gaps")
    fig.tight_layout()
    fig.savefig(R2_CHARTS_DIR / "r2_xwoba_gap_top10.png")
    plt.close(fig)

    fake_hot_overlap = pred[pred["verdict"].eq("fake_hot") & pred.index.isin(top.index)]["player"].tolist()
    fake_cold_overlap = pred[pred["verdict"].eq("fake_cold") & pred.index.isin(top.index)]["player"].tolist()
    payload = {
        "sign_convention": "gap = xwOBA - wOBA; positive means actual wOBA is below expected contact quality, negative means actual wOBA is above expected contact quality.",
        "top10_abs_gap": top[top_cols].to_dict(orient="records"),
        "overlaps": {
            "top10_gap_and_fake_hot": fake_hot_overlap,
            "top10_gap_and_fake_cold": fake_cold_overlap,
        },
        "crosstab": crosstab.to_dict(orient="records"),
    }
    atomic_write_json(R2_MODELS_DIR / "r2_xwoba_gap.json", payload)
    _ = colors
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))


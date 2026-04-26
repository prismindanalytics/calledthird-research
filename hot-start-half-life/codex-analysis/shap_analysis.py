from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import warnings
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from common import CHARTS_DIR, DATASETS_DIR, HITTER_FEATURE_COLS, MODELS_DIR, SEED, TABLES_DIR, atomic_write_json, set_plot_style


def main() -> dict:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    hitters = pd.read_parquet(DATASETS_DIR / "hitter_features.parquet")
    test = hitters[
        hitters["season"].eq(2025)
        & hitters["pa_22g"].ge(50)
        & hitters["pa_ros"].ge(50)
        & hitters["ros_woba"].notna()
    ].copy()
    model = joblib.load(MODELS_DIR / "lgbm_ros_woba.joblib")
    imputer = joblib.load(MODELS_DIR / "lgbm_ros_woba_imputer.joblib")
    x_test = imputer.transform(test[HITTER_FEATURE_COLS])
    perm = permutation_importance(
        model,
        x_test,
        test["ros_woba"].to_numpy(),
        n_repeats=30,
        random_state=SEED,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": HITTER_FEATURE_COLS,
            "permutation_importance": perm.importances_mean,
            "permutation_std": perm.importances_std,
        }
    ).sort_values("permutation_importance", ascending=False)
    perm_df.to_csv(TABLES_DIR / "permutation_importance.csv", index=False)

    sample_n = min(len(test), 1000)
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(np.arange(len(test)), size=sample_n, replace=False)
    x_sample = x_test[sample_idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    shap_abs = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": HITTER_FEATURE_COLS, "mean_abs_shap": shap_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    shap_df.to_csv(TABLES_DIR / "shap_importance.csv", index=False)

    merged = perm_df.merge(shap_df, on="feature")
    merged["perm_rank"] = merged["permutation_importance"].rank(ascending=False, method="average")
    merged["shap_rank"] = merged["mean_abs_shap"].rank(ascending=False, method="average")
    rho = float(spearmanr(merged["perm_rank"], merged["shap_rank"]).statistic)
    merged.sort_values("permutation_importance", ascending=False).to_csv(TABLES_DIR / "feature_importance_rank_check.csv", index=False)

    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    top = perm_df.head(10).iloc[::-1]
    ax.barh(top["feature"], top["permutation_importance"], xerr=top["permutation_std"], color="#33658a")
    ax.set_title("Permutation Importance: ROS wOBA")
    ax.set_xlabel("Increase in RMSE when permuted")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "permutation_importance_top10.png")
    plt.close(fig)

    shap.summary_plot(shap_values, features=x_sample, feature_names=HITTER_FEATURE_COLS, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close()

    payload = {
        "spearman_rank_correlation": rho,
        "top10_permutation": perm_df.head(10).to_dict(orient="records"),
        "investigation": (
            "SHAP and permutation ranks clear sanity threshold."
            if rho >= 0.6
            else "Rank correlation below threshold; compare feature_importance_rank_check.csv for interaction/surrogate effects."
        ),
    }
    if rho < 0.6:
        (TABLES_DIR / "shap_rank_investigation.md").write_text(
            "SHAP/permutation rank-correlation fell below the pre-committed 0.60 threshold.\n\n"
            "The permutation top group emphasizes marginal holdout damage from PA volume, preseason prior wOBA, EV p90, whiff rate, and xwOBA. "
            "TreeSHAP spreads attribution across correlated rate families, especially wOBA/OPS/SLG/ISO and contact-quality variables. "
            "I am treating the permutation top-10 as the headline ranking and the SHAP chart as an interaction/surrogate diagnostic rather than forcing an artificial agreement.\n",
            encoding="utf-8",
        )
    atomic_write_json(MODELS_DIR / "feature_importance_summary.json", payload)
    return payload


if __name__ == "__main__":
    main()

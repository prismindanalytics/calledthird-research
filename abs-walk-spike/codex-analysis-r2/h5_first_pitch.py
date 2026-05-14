from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from data_prep_r2 import (
    ARTIFACTS_DIR,
    CALLED_DESCRIPTIONS,
    CHARTS_DIR,
    COUNT_ORDER,
    GLOBAL_SEED,
    Round2Data,
    called_pitch_rows,
    encode_features,
    grouped_oof_predictions,
    lgbm_classifier,
    save_json,
    write_model_diagnostics,
)


@dataclass
class H5Result:
    summary: dict
    interaction_table: pd.DataFrame
    diagnostics: dict


def _h5_frame(data: Round2Data) -> pd.DataFrame:
    frame = pd.concat([called_pitch_rows(data.df_2025), called_pitch_rows(data.df_2026)], ignore_index=True)
    frame = frame[(frame["count_state"] == "0-0") | (frame["strikes"].astype(int) == 2)].copy()
    frame = frame[frame["zone_region"].isin(["heart", "top_edge", "bottom_edge", "side_edge", "waste"])].copy()
    frame["count_focus"] = np.where(frame["count_state"] == "0-0", "0-0", frame["count_state"].astype(str))
    frame["region_count"] = frame["zone_region"].astype(str) + "|" + frame["count_focus"].astype(str)
    return frame.dropna(subset=["plate_x", "plate_z", "game_pk", "is_called_strike"])


def _observed_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    obs = (
        frame.groupby(["year", "zone_region", "count_focus"], observed=True)
        .agg(called_pitches=("is_called_strike", "size"), called_strike_rate=("is_called_strike", "mean"))
        .reset_index()
    )
    wide = obs.pivot_table(index=["zone_region", "count_focus"], columns="year", values="called_strike_rate").reset_index()
    wide.columns = [str(c) if c not in ["zone_region", "count_focus"] else c for c in wide.columns]
    if "2025" not in wide:
        wide["2025"] = np.nan
    if "2026" not in wide:
        wide["2026"] = np.nan
    wide["yoy_delta_pp"] = (wide["2026"] - wide["2025"]) * 100
    wide.to_csv(ARTIFACTS_DIR / "h5_observed_called_strike_deltas.csv", index=False)
    return wide


def _shap_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = ["year_2026"]
    categorical = ["zone_region", "count_focus", "region_count"]
    X, columns = encode_features(frame, numeric, categorical)
    y = frame["is_called_strike"].to_numpy(dtype=int)
    model = lgbm_classifier(GLOBAL_SEED + 900, n_estimators=180)
    model.fit(X, y)

    sample_n = min(6000, len(X))
    sample = X.sample(n=sample_n, random_state=GLOBAL_SEED + 901)
    explainer = shap.TreeExplainer(model)
    interactions = explainer.shap_interaction_values(sample)
    if isinstance(interactions, list):
        interactions = interactions[1]
    year_idx = list(sample.columns).index("year_2026")
    rows = []
    for col_idx, col in enumerate(sample.columns):
        if not col.startswith("region_count_"):
            continue
        value = interactions[:, year_idx, col_idx]
        label = col.replace("region_count_", "", 1)
        if "|" not in label:
            continue
        region, count = label.split("|", 1)
        active = sample[col].to_numpy() > 0.5
        rows.append(
            {
                "zone_region": region,
                "count_focus": count,
                "mean_year_interaction": float(value[active].mean()) if active.any() else float(value.mean()),
                "mean_abs_year_interaction": float(np.abs(value[active]).mean()) if active.any() else float(np.abs(value).mean()),
                "n_sample_cell": int(active.sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACTS_DIR / "h5_shap_interactions.csv", index=False)
    return out


def _plot_interactions(interactions: pd.DataFrame) -> None:
    counts = ["0-0", "0-2", "1-2", "2-2", "3-2"]
    regions = ["heart", "top_edge", "bottom_edge", "side_edge", "waste"]
    pivot = interactions.pivot_table(index="zone_region", columns="count_focus", values="mean_year_interaction", fill_value=0.0)
    pivot = pivot.reindex(index=regions, columns=counts).fillna(0.0)
    vmax = max(0.001, float(np.nanmax(np.abs(pivot.to_numpy()))))
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    image = ax.imshow(pivot.to_numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts)
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(regions)
    for i in range(len(regions)):
        for j in range(len(counts)):
            ax.text(j, i, f"{pivot.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("H5: SHAP year x region-count interaction")
    ax.set_xlabel("Count state")
    ax.set_ylabel("Zone region")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Mean SHAP interaction (log-odds)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "h5_first_pitch_shap.png", dpi=220)
    fig.savefig(CHARTS_DIR / "h5_first_pitch_mechanism.png", dpi=220)
    plt.close(fig)


def run_h5(data: Round2Data) -> H5Result:
    frame = _h5_frame(data)
    y = frame["is_called_strike"].to_numpy(dtype=int)
    oof, _ = grouped_oof_predictions(
        frame,
        y,
        frame["game_pk"].to_numpy(),
        ["year_2026"],
        ["zone_region", "count_focus", "region_count"],
        GLOBAL_SEED + 920,
        n_splits=5,
        n_estimators=180,
    )
    diagnostics = write_model_diagnostics("h5_first_pitch_two_strike_classifier", y, oof)
    observed = _observed_deltas(frame)
    interactions = _shap_interactions(frame)
    _plot_interactions(interactions)

    def delta(region: str, count: str) -> float:
        row = observed[(observed["zone_region"] == region) & (observed["count_focus"] == count)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0]["yoy_delta_pp"])

    heart_00 = delta("heart", "0-0")
    top_twostrike_rows = observed[(observed["zone_region"] == "top_edge") & (observed["count_focus"].isin(["0-2", "1-2", "2-2", "3-2"]))]
    top_2k = float(top_twostrike_rows["yoy_delta_pp"].mean()) if not top_twostrike_rows.empty else float("nan")
    top_interaction = interactions[(interactions["zone_region"] == "top_edge") & (interactions["count_focus"].isin(["0-2", "1-2", "2-2", "3-2"]))]
    heart_interaction = interactions[(interactions["zone_region"] == "heart") & (interactions["count_focus"] == "0-0")]
    interaction_strength = float(interactions["mean_abs_year_interaction"].max()) if not interactions.empty else 0.0
    mechanism_supported = bool(
        np.isfinite(heart_00)
        and np.isfinite(top_2k)
        and heart_00 >= 0
        and top_2k < 0
        and interaction_strength > 0.002
    )
    summary = {
        "heart_zone_0_0_yoy_delta_pp": heart_00,
        "top_edge_2_strike_yoy_delta_pp": top_2k,
        "interaction_supported": mechanism_supported,
        "interaction_strength_max_abs": interaction_strength,
        "max_calibration_deviation": diagnostics["max_calibration_deviation"],
        "poor_calibration": diagnostics["poor_calibration"],
        "model": "LightGBM is_called_strike ~ year x zone_region x count_state via region_count interaction features",
    }
    save_json(summary, ARTIFACTS_DIR / "h5_summary.json")
    return H5Result(summary=summary, interaction_table=interactions, diagnostics=diagnostics)

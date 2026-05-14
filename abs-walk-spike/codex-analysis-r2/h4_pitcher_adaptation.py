from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.spatial.distance import jensenshannon

from data_prep_r2 import (
    ARTIFACTS_DIR,
    CHARTS_DIR,
    COUNT_ORDER,
    GLOBAL_SEED,
    Round2Data,
    encode_features,
    lgbm_classifier,
    save_json,
)


@dataclass
class H4Result:
    summary: dict
    leaderboard: pd.DataFrame
    shap_importance: pd.DataFrame


def _pitcher_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["pitcher", "player_name", "week_index"], observed=True)
        .agg(
            pitches=("pitch_type", "size"),
            mean_plate_z=("plate_z", "mean"),
            zone_rate=("in_fixed_zone", "mean"),
        )
        .reset_index()
    )


def _mix_vector(frame: pd.DataFrame, pitch_types: list[str]) -> np.ndarray:
    counts = frame["pitch_type"].value_counts(normalize=True)
    vec = np.array([counts.get(pt, 0.0) for pt in pitch_types], dtype=float)
    if vec.sum() == 0:
        return np.ones(len(pitch_types)) / len(pitch_types)
    return vec / vec.sum()


def _adaptation_leaderboard(df_2026: pd.DataFrame) -> pd.DataFrame:
    qualifiers = df_2026.groupby(["pitcher", "player_name"], observed=True).size().reset_index(name="pitches")
    qualifiers = qualifiers[qualifiers["pitches"] >= 200].copy()
    pitch_types = sorted(df_2026["pitch_type"].value_counts().head(12).index.tolist())
    max_week = int(df_2026["week_index"].max())
    rows = []
    for row in qualifiers.itertuples():
        p = df_2026[df_2026["pitcher"] == row.pitcher].copy()
        early = p[p["week_index"] <= 1]
        late = p[p["week_index"] >= max_week - 1]
        if len(early) < 40 or len(late) < 40:
            continue
        js = float(jensenshannon(_mix_vector(early, pitch_types), _mix_vector(late, pitch_types), base=2.0))
        vertical_shift = float(late["plate_z"].mean() - early["plate_z"].mean())
        zone_shift = float(late["in_fixed_zone"].mean() - early["in_fixed_zone"].mean())
        magnitude = js + abs(vertical_shift) / 0.25 + abs(zone_shift) / 0.05
        rows.append(
            {
                "pitcher": int(row.pitcher),
                "player_name": row.player_name,
                "pitches": int(row.pitches),
                "js_pitch_mix": js,
                "vertical_shift_ft": vertical_shift,
                "zone_rate_shift": zone_shift,
                "adaptation_magnitude": magnitude,
                "early_pitches": int(len(early)),
                "late_pitches": int(len(late)),
            }
        )
    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return leaderboard
    return leaderboard.sort_values("adaptation_magnitude", ascending=False).reset_index(drop=True)


def _plot_leaderboard(leaderboard: pd.DataFrame) -> None:
    top = leaderboard.head(10).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.barh(top["player_name"], top["adaptation_magnitude"], color="#2563eb")
    for i, row in enumerate(top.itertuples()):
        ax.text(
            row.adaptation_magnitude + 0.05,
            i,
            f"JS {row.js_pitch_mix:.2f} | z {row.vertical_shift_ft:+.2f} | zone {row.zone_rate_shift*100:+.1f}pp",
            va="center",
            fontsize=8.5,
        )
    ax.set_xlabel("Adaptation magnitude")
    ax.set_title("H4: top pitcher adaptation signals")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "h4_pitcher_adaptation_leaderboard.png", dpi=220)
    plt.close(fig)


def _pitcher_shap(df_2026: pd.DataFrame, leaderboard: pd.DataFrame) -> pd.DataFrame:
    rows = []
    max_week = int(df_2026["week_index"].max())
    numeric = ["plate_x", "plate_z", "in_fixed_zone", "balls", "strikes"]
    categorical = ["pitch_type", "count_state"]
    for rank, lead in enumerate(leaderboard.head(10).itertuples(), start=1):
        p = df_2026[df_2026["pitcher"] == lead.pitcher].copy()
        p = p[(p["week_index"] <= 1) | (p["week_index"] >= max_week - 1)].dropna(subset=["plate_x", "plate_z"])
        if len(p) < 100:
            continue
        p["late_period"] = (p["week_index"] >= max_week - 1).astype(int)
        if p["late_period"].nunique() < 2:
            continue
        X, columns = encode_features(p, numeric, categorical)
        y = p["late_period"].to_numpy(dtype=int)
        model = lgbm_classifier(GLOBAL_SEED + 700 + rank, n_estimators=180)
        model.fit(X, y)
        sample = X.sample(n=min(600, len(X)), random_state=GLOBAL_SEED + rank)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
        if isinstance(values, list):
            values = values[1]
        importance = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
        importance = importance.sort_values("mean_abs_shap", ascending=False).head(5)
        for feat in importance.itertuples():
            rows.append(
                {
                    "rank": rank,
                    "pitcher": int(lead.pitcher),
                    "player_name": lead.player_name,
                    "feature": feat.feature,
                    "mean_abs_shap": float(feat.mean_abs_shap),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ARTIFACTS_DIR / "h4_top_pitcher_shap_importance.csv", index=False)
    return out


def run_h4(data: Round2Data) -> H4Result:
    df_2026 = data.df_2026.dropna(subset=["pitcher", "plate_z", "pitch_type"]).copy()
    df_2025 = data.df_2025.dropna(subset=["plate_z", "pitch_type"]).copy()

    weekly = _pitcher_weekly(df_2026)
    weekly.to_csv(ARTIFACTS_DIR / "h4_pitcher_weekly_metrics.csv", index=False)
    leaderboard = _adaptation_leaderboard(df_2026)
    leaderboard.to_csv(ARTIFACTS_DIR / "h4_pitcher_adaptation_leaderboard.csv", index=False)
    if not leaderboard.empty:
        _plot_leaderboard(leaderboard)
    shap_importance = _pitcher_shap(df_2026, leaderboard) if not leaderboard.empty else pd.DataFrame()

    league_weekly = (
        df_2026.groupby("week_index", observed=True)
        .agg(pitches=("pitch_type", "size"), mean_plate_z=("plate_z", "mean"), zone_rate=("in_fixed_zone", "mean"))
        .reset_index()
    )
    league_weekly.to_csv(ARTIFACTS_DIR / "h4_league_weekly_metrics.csv", index=False)

    zone_2025 = float(df_2025["in_fixed_zone"].mean())
    zone_2026 = float(df_2026["in_fixed_zone"].mean())
    first_week = league_weekly.sort_values("week_index").iloc[0]
    last_week = league_weekly.sort_values("week_index").iloc[-1]
    summary = {
        "league_zone_rate_2025": zone_2025,
        "league_zone_rate_2026": zone_2026,
        "league_zone_rate_delta_pp": (zone_2026 - zone_2025) * 100,
        "week0_zone_rate_2026": float(first_week["zone_rate"]),
        "last_week_zone_rate_2026": float(last_week["zone_rate"]),
        "league_weekly_trend": "down" if last_week["zone_rate"] < first_week["zone_rate"] else "up",
        "qualified_pitchers": int(len(leaderboard)),
        "top_adapters": leaderboard.head(10).to_dict(orient="records") if not leaderboard.empty else [],
        "zone_definition": "fixed absolute proxy: |plate_x| <= 17/24 ft and 1.6 <= plate_z <= 3.5",
    }
    save_json(summary, ARTIFACTS_DIR / "h4_summary.json")
    return H4Result(summary=summary, leaderboard=leaderboard, shap_importance=shap_importance)

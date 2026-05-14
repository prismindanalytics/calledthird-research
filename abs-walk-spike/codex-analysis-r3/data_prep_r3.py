from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "codex-analysis-r3"
DATA_DIR = ANALYSIS_DIR / "data"
ARTIFACTS_DIR = ANALYSIS_DIR / "artifacts"
CHARTS_DIR = ANALYSIS_DIR / "charts"
DIAG_DIR = CHARTS_DIR / "model_diagnostics"

ROUND1_2026_PATH = ROOT / "data" / "statcast_2026_mar27_apr22.parquet"
R2_EXTENSION_2026_PATH = ROOT / "codex-analysis-r2" / "data" / "statcast_2026_apr23_may13.parquet"
FULL_2025_PATH = ROOT / ".." / "count-distribution-abs" / "data" / "statcast_2025_full.parquet"

START_2026 = pd.Timestamp("2026-03-27")
REQUESTED_END_2026 = pd.Timestamp("2026-05-13")
START_2025 = pd.Timestamp("2025-03-27")

GLOBAL_SEED = 20260514
WALK_EVENTS = {"walk", "intent_walk"}
CALLED_DESCRIPTIONS = {"called_strike", "ball"}
AUTO_DESCRIPTIONS = {"automatic_ball", "automatic_strike"}
BALL_LIKE_DESCRIPTIONS = {"ball", "blocked_ball", "pitchout", "automatic_ball"}
STRIKE_LIKE_DESCRIPTIONS = {"called_strike", "swinging_strike", "swinging_strike_blocked", "missed_bunt", "automatic_strike"}
FOUL_DESCRIPTIONS = {"foul"}
FOUL_TIP_DESCRIPTIONS = {"foul_tip"}
BUNT_FOUL_DESCRIPTIONS = {"foul_bunt", "bunt_foul_tip"}
IN_PLAY_DESCRIPTIONS = {"hit_into_play", "hit_into_play_score", "hit_into_play_no_out"}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
SWING_DESCRIPTIONS = WHIFF_DESCRIPTIONS | FOUL_DESCRIPTIONS | FOUL_TIP_DESCRIPTIONS | BUNT_FOUL_DESCRIPTIONS | IN_PLAY_DESCRIPTIONS

COUNT_ORDER = ["0-0", "1-0", "0-1", "2-0", "1-1", "0-2", "3-0", "2-1", "1-2", "3-1", "2-2", "3-2"]
COUNT_TO_SORT = {count: i for i, count in enumerate(COUNT_ORDER)}
PLATE_HALF_WIDTH_FT = 17.0 / 24.0
FIXED_ZONE_BOTTOM_FT = 1.6
FIXED_ZONE_TOP_FT = 3.5


@dataclass
class Round3Data:
    df_2025: pd.DataFrame
    df_2026: pd.DataFrame
    full_2025: pd.DataFrame
    effective_end_2026: pd.Timestamp
    effective_end_2025: pd.Timestamp
    metadata: dict[str, Any]


def ensure_dirs() -> None:
    for path in (DATA_DIR, ARTIFACTS_DIR, CHARTS_DIR, DIAG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def lgbm_classifier(seed: int, n_estimators: int = 70) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def lgbm_regressor(seed: int, n_estimators: int = 80) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=15,
        min_child_samples=12,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def zone_region(plate_x: pd.Series, plate_z: pd.Series) -> pd.Series:
    x = plate_x.astype(float)
    z = plate_z.astype(float)
    region = pd.Series("waste", index=plate_x.index, dtype="object")
    in_x = x.abs() <= 0.90
    region.loc[in_x & z.between(3.0, 3.8)] = "top_edge"
    region.loc[in_x & z.between(1.2, 2.0)] = "bottom_edge"
    region.loc[(x.abs() >= PLATE_HALF_WIDTH_FT) & (x.abs() <= 1.10) & z.between(2.0, 3.0)] = "side_edge"
    region.loc[(x.abs() <= 0.45) & z.between(2.0, 3.0)] = "heart"
    region.loc[x.isna() | z.isna()] = "missing"
    return region


def derive_columns(df: pd.DataFrame, year: int, season_start: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["year"] = year
    out["year_2026"] = int(year == 2026)
    out["pitcher"] = pd.to_numeric(out["pitcher"], errors="coerce")
    out["player_name"] = out["player_name"].fillna("Unknown").astype(str)
    out["pitch_type"] = out["pitch_type"].fillna("UNK").astype(str)
    out["description"] = out["description"].fillna("").astype(str)
    out["events"] = out["events"].fillna("").astype(str)
    out["balls"] = pd.to_numeric(out["balls"], errors="coerce").fillna(0).astype(int)
    out["strikes"] = pd.to_numeric(out["strikes"], errors="coerce").fillna(0).astype(int)
    out["count_state"] = out["balls"].astype(str) + "-" + out["strikes"].astype(str)
    out["week_index"] = ((out["game_date"] - season_start).dt.days // 7).astype(int)
    out["week_start"] = season_start + pd.to_timedelta(out["week_index"] * 7, unit="D")
    out["pa_id"] = (
        out["year"].astype(str)
        + "_"
        + out["game_pk"].astype(int).astype(str)
        + "_"
        + out["at_bat_number"].astype(int).astype(str)
    )
    out["is_called_strike"] = (out["description"] == "called_strike").astype(int)
    out["is_taken_zone_model"] = out["description"].isin(CALLED_DESCRIPTIONS)
    out["walk_event_pitch_row"] = out["events"].isin(WALK_EVENTS).astype(int)
    out["in_fixed_zone"] = (
        out["plate_x"].abs().le(PLATE_HALF_WIDTH_FT)
        & out["plate_z"].between(FIXED_ZONE_BOTTOM_FT, FIXED_ZONE_TOP_FT)
    ).fillna(False).astype(int)
    out["zone_region"] = zone_region(out["plate_x"], out["plate_z"])
    out["is_top_edge"] = (out["zone_region"] == "top_edge").astype(int)
    out["is_bottom_edge"] = (out["zone_region"] == "bottom_edge").astype(int)
    out["is_swing"] = out["description"].isin(SWING_DESCRIPTIONS).astype(int)
    out["is_whiff"] = out["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    return out


def terminal_pa_rows(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).copy()
    terminal = ordered[ordered["events"].notna() & (ordered["events"] != "")].copy()
    terminal = terminal.drop_duplicates(subset=["pa_id"], keep="last")
    terminal["walk_event"] = terminal["events"].isin(WALK_EVENTS).astype(int)
    terminal["terminal_count"] = terminal["count_state"]
    return terminal


def called_pitch_rows(df: pd.DataFrame) -> pd.DataFrame:
    called = df[df["description"].isin(CALLED_DESCRIPTIONS)].copy()
    called = called.dropna(subset=["plate_x", "plate_z", "count_state", "pitch_type", "game_pk"])
    return called


def walk_rate_from_terminal(terminal: pd.DataFrame, game_weights: dict[int, int] | None = None) -> float:
    if game_weights is None:
        return float(terminal["walk_event"].mean())
    weights = terminal["game_pk"].map(game_weights).fillna(0.0).to_numpy(dtype=float)
    denom = weights.sum()
    if denom <= 0:
        return float("nan")
    return float(np.dot(terminal["walk_event"].to_numpy(dtype=float), weights) / denom)


def encode_features(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    parts: list[pd.DataFrame] = []
    if numeric:
        parts.append(frame[numeric].astype(float).reset_index(drop=True))
    if categorical:
        cat = pd.get_dummies(frame[categorical].astype(str), columns=categorical, prefix=categorical, dtype=float)
        parts.append(cat.reset_index(drop=True))
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
        return X, columns
    return X, X.columns.tolist()


def sample_game_weights(games: np.ndarray, rng: np.random.Generator) -> dict[int, int]:
    games = np.asarray(games, dtype=int)
    sampled = rng.choice(games, size=len(games), replace=True)
    values, counts = np.unique(sampled, return_counts=True)
    return {int(g): int(w) for g, w in zip(values, counts, strict=False)}


def row_weights_from_games(frame: pd.DataFrame, game_weights: dict[int, int]) -> np.ndarray:
    return frame["game_pk"].map(game_weights).fillna(0.0).to_numpy(dtype=float)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denom = weights.sum()
    if denom <= 0:
        return float("nan")
    return float(np.dot(np.asarray(values, dtype=float), weights) / denom)


def percentile_ci(values: pd.Series | np.ndarray, q: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, q[0])), float(np.percentile(arr, q[1]))


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y_true, "p": np.clip(y_prob, 1e-6, 1 - 1e-6)})
    frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    tab = (
        frame.groupby("bin", observed=True)
        .agg(n=("y", "size"), predicted=("p", "mean"), empirical=("y", "mean"))
        .reset_index(drop=True)
    )
    tab["abs_deviation"] = (tab["empirical"] - tab["predicted"]).abs()
    return tab


def write_classifier_diagnostics(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    tab = calibration_table(y_true, y_prob)
    tab.to_csv(ARTIFACTS_DIR / f"{name}_calibration.csv", index=False)
    max_dev = float(tab["abs_deviation"].max())
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1)
    ax.plot(tab["predicted"], tab["empirical"], marker="o", color="#2563eb")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical rate")
    ax.set_title(f"{name} calibration")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / f"{name}_calibration.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, color="#dc2626")
    ax.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1)
    ax.set_title(f"{name} ROC")
    fig.tight_layout()
    fig.savefig(DIAG_DIR / f"{name}_roc.png", dpi=200)
    plt.close(fig)
    return {"auc": auc, "max_calibration_deviation": max_dev, "poor_calibration": bool(max_dev > 0.05)}


def grouped_oof_predictions(
    frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    seed: int,
    n_splits: int = 5,
    n_estimators: int = 80,
) -> tuple[np.ndarray, float]:
    preds = np.zeros(len(frame), dtype=float)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(frame, y, groups)):
        X_train, columns = encode_features(frame.iloc[train_idx], numeric, categorical)
        X_test, _ = encode_features(frame.iloc[test_idx], numeric, categorical, columns)
        model = lgbm_classifier(seed + fold_idx, n_estimators=n_estimators)
        model.fit(X_train, y[train_idx])
        preds[test_idx] = model.predict_proba(X_test)[:, 1]
    return preds, float(roc_auc_score(y, preds))


def load_round3_data() -> Round3Data:
    ensure_dirs()
    df_2026_old = pd.read_parquet(ROUND1_2026_PATH)
    df_2026_ext = pd.read_parquet(R2_EXTENSION_2026_PATH)
    full_2025_raw = pd.read_parquet(FULL_2025_PATH)
    for frame in (df_2026_old, df_2026_ext, full_2025_raw):
        frame["game_date"] = pd.to_datetime(frame["game_date"])

    common_2026 = sorted(set(df_2026_old.columns) & set(df_2026_ext.columns))
    df_2026 = pd.concat([df_2026_old[common_2026], df_2026_ext[common_2026]], ignore_index=True)
    df_2026 = df_2026.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"], keep="last")
    df_2026 = df_2026.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    observed_end_2026 = min(REQUESTED_END_2026, pd.Timestamp(df_2026["game_date"].max()))
    offset_days = int((observed_end_2026 - START_2026).days)
    effective_end_2025 = START_2025 + pd.Timedelta(days=offset_days)

    df_2026 = df_2026[(df_2026["game_date"] >= START_2026) & (df_2026["game_date"] <= observed_end_2026)].copy()
    df_2025_raw = full_2025_raw[
        (full_2025_raw["game_date"] >= START_2025) & (full_2025_raw["game_date"] <= effective_end_2025)
    ].copy()

    common = sorted(set(df_2025_raw.columns) & set(df_2026.columns))
    df_2025 = derive_columns(df_2025_raw[common], 2025, START_2025)
    df_2026 = derive_columns(df_2026[common], 2026, START_2026)
    full_2025 = derive_columns(full_2025_raw, 2025, pd.Timestamp("2025-03-20"))

    df_2025.to_parquet(DATA_DIR / "statcast_2025_matched_window.parquet", index=False)
    df_2026.to_parquet(DATA_DIR / "statcast_2026_window.parquet", index=False)

    metadata = {
        "requested_start_2026": str(START_2026.date()),
        "requested_end_2026": str(REQUESTED_END_2026.date()),
        "effective_end_2026": str(observed_end_2026.date()),
        "effective_end_2025": str(effective_end_2025.date()),
        "rows_2025_window": int(len(df_2025)),
        "rows_2026_window": int(len(df_2026)),
        "rows_2025_full": int(len(full_2025)),
        "statcast_2026_extension_rows": int(len(df_2026_ext)),
        "statcast_2026_extension_max_date": str(pd.Timestamp(df_2026_ext["game_date"].max()).date()),
        "may_13_available": bool(pd.Timestamp(df_2026_ext["game_date"].max()) >= REQUESTED_END_2026),
        "common_columns": len(common),
        "seed": GLOBAL_SEED,
    }
    save_json(metadata, DATA_DIR / "prep_metadata.json")
    return Round3Data(
        df_2025=df_2025,
        df_2026=df_2026,
        full_2025=full_2025,
        effective_end_2026=observed_end_2026,
        effective_end_2025=effective_end_2025,
        metadata=metadata,
    )

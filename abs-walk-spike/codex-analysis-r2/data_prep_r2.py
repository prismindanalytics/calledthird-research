from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pybaseball import statcast
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "codex-analysis-r2"
DATA_DIR = ANALYSIS_DIR / "data"
CHARTS_DIR = ANALYSIS_DIR / "charts"
ARTIFACTS_DIR = ANALYSIS_DIR / "artifacts"
DIAG_DIR = CHARTS_DIR / "model_diagnostics"

ROUND1_2026_PATH = ROOT / "data" / "statcast_2026_mar27_apr22.parquet"
FULL_2025_PATH = ROOT / ".." / "count-distribution-abs" / "data" / "statcast_2025_full.parquet"
EXTENSION_2026_PATH = DATA_DIR / "statcast_2026_apr23_may13.parquet"
SUBSET_2025_PATH = DATA_DIR / "statcast_2025_apr23_may13.parquet"
WEEKLY_PATH = DATA_DIR / "weekly_aggregation.csv"
PREP_META_PATH = DATA_DIR / "prep_metadata.json"

START_2026 = pd.Timestamp("2026-03-27")
REQUESTED_END_2026 = pd.Timestamp("2026-05-13")
START_2025 = pd.Timestamp("2025-03-27")
REQUESTED_END_2025 = pd.Timestamp("2025-05-13")
EXT_START_2026 = pd.Timestamp("2026-04-23")
EXT_START_2025 = pd.Timestamp("2025-04-23")

GLOBAL_SEED = 20260514
COUNT_ORDER = ["0-0", "1-0", "0-1", "2-0", "1-1", "0-2", "3-0", "2-1", "1-2", "3-1", "2-2", "3-2"]
COUNT_TO_SORT = {count: i for i, count in enumerate(COUNT_ORDER)}
WALK_EVENTS = {"walk", "intent_walk"}
CALLED_DESCRIPTIONS = {"called_strike", "ball"}
AUTO_DESCRIPTIONS = {"automatic_ball", "automatic_strike"}
PLATE_HALF_WIDTH_FT = 17.0 / 24.0
FIXED_ZONE_BOTTOM_FT = 1.6
FIXED_ZONE_TOP_FT = 3.5
HEART_X_FT = 0.45
HEART_Z_LOW_FT = 2.0
HEART_Z_HIGH_FT = 3.0

BALL_LIKE_DESCRIPTIONS = {"ball", "blocked_ball", "pitchout", "automatic_ball"}
STRIKE_LIKE_DESCRIPTIONS = {"called_strike", "swinging_strike", "swinging_strike_blocked", "missed_bunt", "automatic_strike"}
FOUL_DESCRIPTIONS = {"foul"}
FOUL_TIP_DESCRIPTIONS = {"foul_tip"}
BUNT_FOUL_DESCRIPTIONS = {"foul_bunt", "bunt_foul_tip"}
IN_PLAY_DESCRIPTIONS = {"hit_into_play", "hit_into_play_score", "hit_into_play_no_out"}


@dataclass
class Round2Data:
    df_2025: pd.DataFrame
    df_2026: pd.DataFrame
    effective_end_2026: pd.Timestamp
    effective_end_2025: pd.Timestamp
    metadata: dict[str, Any]


def ensure_dirs() -> None:
    for path in (DATA_DIR, CHARTS_DIR, ARTIFACTS_DIR, DIAG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def lgbm_classifier(seed: int, n_estimators: int = 180) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=70,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def fetch_statcast_extension(force: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if EXTENSION_2026_PATH.exists() and not force:
        df = pd.read_parquet(EXTENSION_2026_PATH)
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df

    print("Fetching 2026-04-23 to 2026-05-13 from Statcast...", file=sys.stderr)
    df = statcast(start_dt="2026-04-23", end_dt="2026-05-13")
    if df.empty:
        raise RuntimeError("pybaseball returned no rows for 2026-04-23 to 2026-05-13")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    df.to_parquet(EXTENSION_2026_PATH, index=False)
    return df


def derive_columns(df: pd.DataFrame, year: int, season_start: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    out["year"] = year
    out["year_2026"] = int(year == 2026)
    out["pitch_type"] = out["pitch_type"].fillna("UNK").astype(str)
    out["description"] = out["description"].fillna("").astype(str)
    out["count_state"] = out["balls"].astype(int).astype(str) + "-" + out["strikes"].astype(int).astype(str)
    out["week_index"] = ((out["game_date"] - season_start).dt.days // 7).astype(int)
    out["week_start"] = season_start + pd.to_timedelta(out["week_index"] * 7, unit="D")
    out["week_label"] = out["week_start"].dt.strftime("%b %d")
    out["pa_id"] = (
        out["year"].astype(str)
        + "_"
        + out["game_pk"].astype(int).astype(str)
        + "_"
        + out["at_bat_number"].astype(int).astype(str)
    )
    out["walk_event_pitch_row"] = out["events"].isin(WALK_EVENTS).astype(int)
    out["is_called_strike"] = (out["description"] == "called_strike").astype(int)
    out["is_taken_zone_model"] = out["description"].isin(CALLED_DESCRIPTIONS)
    in_zone = (
        out["plate_x"].abs().le(PLATE_HALF_WIDTH_FT)
        & out["plate_z"].between(FIXED_ZONE_BOTTOM_FT, FIXED_ZONE_TOP_FT)
    )
    out["in_fixed_zone"] = in_zone.fillna(False).astype(int)
    out["zone_region"] = zone_region(out["plate_x"], out["plate_z"])
    return out


def zone_region(plate_x: pd.Series, plate_z: pd.Series) -> pd.Series:
    x = plate_x.astype(float)
    z = plate_z.astype(float)
    region = pd.Series("waste", index=plate_x.index, dtype="object")
    in_x = x.abs() <= 0.90
    region.loc[in_x & (z >= 3.0) & (z <= 3.8)] = "top_edge"
    region.loc[in_x & (z <= 2.0) & (z >= 1.2)] = "bottom_edge"
    region.loc[(x.abs() >= PLATE_HALF_WIDTH_FT) & (x.abs() <= 1.10) & z.between(2.0, 3.0)] = "side_edge"
    region.loc[(x.abs() <= HEART_X_FT) & z.between(HEART_Z_LOW_FT, HEART_Z_HIGH_FT)] = "heart"
    region.loc[x.isna() | z.isna()] = "missing"
    return region


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


def walk_rate_from_terminal(terminal: pd.DataFrame) -> dict[str, Any]:
    return {
        "pas": int(len(terminal)),
        "walks": int(terminal["walk_event"].sum()),
        "walk_rate": float(terminal["walk_event"].mean()),
    }


def encode_features(
    frame: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    parts = []
    if numeric:
        parts.append(frame[numeric].astype(float).reset_index(drop=True))
    if categorical:
        cat = pd.get_dummies(frame[categorical].astype(str), columns=categorical, prefix=categorical, dtype=float)
        parts.append(cat.reset_index(drop=True))
    if not parts:
        X = pd.DataFrame(index=frame.index)
    else:
        X = pd.concat(parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
        return X, columns
    return X, X.columns.tolist()


def grouped_oof_predictions(
    frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    numeric: list[str],
    categorical: list[str],
    seed: int,
    n_splits: int = 5,
    n_estimators: int = 220,
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


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    quantiles = pd.qcut(pd.Series(y_prob), q=bins, duplicates="drop")
    tab = (
        pd.DataFrame({"y": y_true, "p": y_prob, "bin": quantiles})
        .groupby("bin", observed=True)
        .agg(n=("y", "size"), predicted=("p", "mean"), empirical=("y", "mean"))
        .reset_index(drop=True)
    )
    tab["abs_deviation"] = (tab["empirical"] - tab["predicted"]).abs()
    return tab


def write_model_diagnostics(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    tab = calibration_table(y_true, y_prob, bins=10)
    tab.to_csv(ARTIFACTS_DIR / f"{name}_calibration.csv", index=False)
    max_dev = float(tab["abs_deviation"].max())

    fig, ax = plt.subplots(figsize=(5.7, 5.2))
    ax.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1)
    ax.plot(tab["predicted"], tab["empirical"], marker="o", color="#2563eb")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical rate")
    ax.set_title(f"{name} calibration")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / f"{name}_calibration.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 5.2))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, color="#dc2626")
    ax.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1)
    ax.set_title(f"{name} ROC")
    fig.tight_layout()
    fig.savefig(DIAG_DIR / f"{name}_roc.png", dpi=200)
    plt.close(fig)

    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "max_calibration_deviation": max_dev,
        "poor_calibration": bool(max_dev > 0.05),
    }


def game_bootstrap_rates(terminal: pd.DataFrame, seed: int, n_boot: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    game_week = (
        terminal.groupby(["game_pk", "week_index"], observed=True)
        .agg(pas=("pa_id", "size"), walks=("walk_event", "sum"))
        .reset_index()
    )
    games = game_week["game_pk"].dropna().unique()
    rows = []
    for _ in range(n_boot):
        sampled_games = rng.choice(games, size=len(games), replace=True)
        weights = pd.Series(sampled_games).value_counts().rename_axis("game_pk").reset_index(name="weight")
        sampled = game_week.merge(weights, on="game_pk", how="inner")
        sampled["weighted_pas"] = sampled["pas"] * sampled["weight"]
        sampled["weighted_walks"] = sampled["walks"] * sampled["weight"]
        weekly = sampled.groupby("week_index", observed=True)[["weighted_pas", "weighted_walks"]].sum().reset_index()
        weekly["walk_rate"] = weekly["weighted_walks"] / weekly["weighted_pas"]
        rows.extend(weekly[["week_index", "walk_rate"]].to_dict(orient="records"))
    boot = pd.DataFrame(rows)
    return (
        boot.groupby("week_index", observed=True)["walk_rate"]
        .quantile([0.025, 0.975])
        .unstack()
        .rename(columns={0.025: "ci_low", 0.975: "ci_high"})
        .reset_index()
    )


def build_weekly_aggregation(df_2025: pd.DataFrame, df_2026: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for year, df in [(2025, df_2025), (2026, df_2026)]:
        terminal = terminal_pa_rows(df)
        weekly = (
            terminal.groupby(["year", "week_index", "week_start", "week_label"], observed=True)
            .agg(pas=("pa_id", "size"), walks=("walk_event", "sum"))
            .reset_index()
        )
        weekly["walk_rate"] = weekly["walks"] / weekly["pas"]
        ci = game_bootstrap_rates(terminal, GLOBAL_SEED + year)
        weekly = weekly.merge(ci, on="week_index", how="left")
        frames.append(weekly)
    out = pd.concat(frames, ignore_index=True).sort_values(["year", "week_index"])
    out.to_csv(WEEKLY_PATH, index=False)
    return out


def load_round2_data(force_fetch: bool = False) -> Round2Data:
    ensure_dirs()
    df_2026_old = pd.read_parquet(ROUND1_2026_PATH)
    df_2026_ext = fetch_statcast_extension(force=force_fetch)
    df_2025_full = pd.read_parquet(FULL_2025_PATH)

    for frame in (df_2026_old, df_2026_ext, df_2025_full):
        frame["game_date"] = pd.to_datetime(frame["game_date"])

    subset_2025_ext = df_2025_full[
        (df_2025_full["game_date"] >= EXT_START_2025) & (df_2025_full["game_date"] <= REQUESTED_END_2025)
    ].copy()
    subset_2025_ext.to_parquet(SUBSET_2025_PATH, index=False)

    common_2026 = sorted(set(df_2026_old.columns) & set(df_2026_ext.columns))
    df_2026 = pd.concat([df_2026_old[common_2026], df_2026_ext[common_2026]], ignore_index=True)
    df_2026 = df_2026.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"], keep="last")
    df_2026["game_date"] = pd.to_datetime(df_2026["game_date"])
    df_2026 = df_2026.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    observed_end_2026 = min(REQUESTED_END_2026, pd.Timestamp(df_2026["game_date"].max()))
    offset_days = int((observed_end_2026 - START_2026).days)
    effective_end_2025 = START_2025 + pd.Timedelta(days=offset_days)

    df_2026 = df_2026[(df_2026["game_date"] >= START_2026) & (df_2026["game_date"] <= observed_end_2026)].copy()
    df_2025 = df_2025_full[
        (df_2025_full["game_date"] >= START_2025) & (df_2025_full["game_date"] <= effective_end_2025)
    ].copy()

    common = sorted(set(df_2025.columns) & set(df_2026.columns))
    df_2025 = derive_columns(df_2025[common], 2025, START_2025)
    df_2026 = derive_columns(df_2026[common], 2026, START_2026)

    weekly = build_weekly_aggregation(df_2025, df_2026)
    meta = {
        "requested_start_2026": str(START_2026.date()),
        "requested_end_2026": str(REQUESTED_END_2026.date()),
        "effective_end_2026": str(observed_end_2026.date()),
        "effective_end_2025": str(effective_end_2025.date()),
        "statcast_extension_rows": int(len(df_2026_ext)),
        "statcast_extension_min_date": str(pd.Timestamp(df_2026_ext["game_date"].min()).date()),
        "statcast_extension_max_date": str(pd.Timestamp(df_2026_ext["game_date"].max()).date()),
        "may_13_available": bool(pd.Timestamp(df_2026_ext["game_date"].max()) >= REQUESTED_END_2026),
        "rows_2025": int(len(df_2025)),
        "rows_2026": int(len(df_2026)),
        "weekly_rows": int(len(weekly)),
        "common_columns": len(common),
        "seed": GLOBAL_SEED,
    }
    save_json(meta, PREP_META_PATH)
    return Round2Data(df_2025=df_2025, df_2026=df_2026, effective_end_2026=observed_end_2026, effective_end_2025=effective_end_2025, metadata=meta)

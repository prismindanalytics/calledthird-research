from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import BASE_DIR, CHARTS_DIR, DATASETS_DIR, DATA_DIR, MODELS_DIR, SEED, TABLES_DIR, atomic_write_json, safe_divide


ROUND2_DIR = BASE_DIR / "round2"
R2_MODELS_DIR = MODELS_DIR / "r2"
R2_CHARTS_DIR = CHARTS_DIR / "r2"
R2_DIAG_DIR = R2_CHARTS_DIR / "diag"
R2_TABLES_DIR = ROUND2_DIR / "tables"

for directory in [ROUND2_DIR, R2_MODELS_DIR, R2_CHARTS_DIR, R2_DIAG_DIR, R2_TABLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


R2_HITTER_FEATURE_COLS = [
    "pa_22g",
    "bb_rate_22g",
    "k_rate_22g",
    "babip_22g",
    "iso_22g",
    "ev_p90_22g",
    "hardhit_rate_22g",
    "barrel_rate_22g",
    "xwoba_22g",
    "xwoba_minus_woba_22g",
    "abs_xwoba_minus_woba_22g",
    "xwoba_minus_prior_woba_22g",
    "ev_p90_resid_22g",
    "hardhit_resid_22g",
    "barrel_resid_22g",
    "xwoba_resid_22g",
    "preseason_prior_woba",
    "preseason_prior_iso",
    "preseason_prior_k_rate",
    "league_woba",
    "league_bb_rate",
    "league_k_rate",
    "league_babip",
    "league_iso",
]

R2_RELIEVER_FEATURE_COLS = [
    "bf_22g",
    "ip_22g",
    "k_rate_22g",
    "bb_rate_22g",
    "hr_rate_22g",
    "ev_p90_allowed_22g",
    "hardhit_allowed_rate_22g",
    "pitches_per_bf_22g",
    "swing_rate_22g",
    "whiff_per_swing_22g",
    "zone_rate_22g",
    "preseason_prior_k_rate",
    "preseason_prior_bb_rate",
    "league_pitcher_k_rate",
    "league_pitcher_bb_rate",
    "league_pitcher_ra9",
]

NAMED_HITTER_KEYS = {
    "andy_pages": 681624,
    "ben_rice": 700250,
    "munetaka_murakami": 808959,
    "mike_trout": 545361,
}
NAMED_PITCHER_KEYS = {"mason_miller": 695243}


def metric_dict(y_true: Iterable[float], y_pred: Iterable[float]) -> dict:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else None,
        "n": int(len(y_true_arr)),
    }


def load_name_map() -> dict[int, str]:
    path = DATASETS_DIR / "name_map.csv"
    names: dict[int, str] = {}
    if path.exists():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            if pd.notna(row.get("key_mlbam")) and pd.notna(row.get("name")):
                names[int(row["key_mlbam"])] = str(row["name"]).strip()
    for key, mlbam in NAMED_HITTER_KEYS.items():
        names.setdefault(mlbam, key.replace("_", " ").title())
    for key, mlbam in NAMED_PITCHER_KEYS.items():
        names.setdefault(mlbam, key.replace("_", " ").title())
    return names


def pretty_player_name(name: str) -> str:
    text = str(name).strip()
    if not text:
        return text
    if text == text.lower() or text == text.upper():
        text = text.title()
    replacements = {
        " Ii": " II",
        " Iii": " III",
        " Iv": " IV",
        " Cj ": " CJ ",
        " Jp ": " JP ",
    }
    padded = f" {text} "
    for old, new in replacements.items():
        padded = padded.replace(old, new)
    return padded.strip()


def add_player_names(df: pd.DataFrame, id_col: str, name_col: str = "player") -> pd.DataFrame:
    names = load_name_map()
    out = df.copy()
    out[name_col] = out[id_col].map(
        lambda value: pretty_player_name(names.get(int(value), f"MLBAM {int(value)}")) if pd.notna(value) else "Unknown"
    )
    return out


def _season_reference_means(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ev_p90_22g", "hardhit_rate_22g", "barrel_rate_22g", "xwoba_22g"]
    rows = []
    for season, group in df.groupby("season"):
        row = {"season": season}
        for col in cols:
            row[f"{col}_league_22g"] = float(group[col].median(skipna=True)) if col in group else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_r2_hitter_columns(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    if reference is None:
        reference = _season_reference_means(out)
    if any(col.endswith("_league_22g") for col in reference.columns):
        out = out.merge(reference, on="season", how="left")
    for col in ["xwoba_minus_woba_22g", "xwoba_22g", "woba_22g", "preseason_prior_woba"]:
        if col not in out.columns:
            out[col] = np.nan
    out["abs_xwoba_minus_woba_22g"] = out["xwoba_minus_woba_22g"].abs()
    out["xwoba_minus_prior_woba_22g"] = out["xwoba_22g"] - out["preseason_prior_woba"]
    out["woba_minus_prior_woba_22g"] = out["woba_22g"] - out["preseason_prior_woba"]
    out["ros_woba_delta_vs_prior"] = out["ros_woba"] - out["preseason_prior_woba"] if "ros_woba" in out.columns else np.nan
    residual_pairs = {
        "ev_p90_resid_22g": ("ev_p90_22g", "ev_p90_22g_league_22g"),
        "hardhit_resid_22g": ("hardhit_rate_22g", "hardhit_rate_22g_league_22g"),
        "barrel_resid_22g": ("barrel_rate_22g", "barrel_rate_22g_league_22g"),
        "xwoba_resid_22g": ("xwoba_22g", "xwoba_22g_league_22g"),
    }
    for target, (col, ref_col) in residual_pairs.items():
        out[target] = out[col] - out[ref_col] if col in out.columns and ref_col in out.columns else np.nan
    for col in R2_HITTER_FEATURE_COLS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def add_pitcher_prior_columns(pitchers: pd.DataFrame) -> pd.DataFrame:
    out = pitchers.sort_values(["pitcher", "season"]).copy()
    for col in ["preseason_prior_k_rate", "preseason_prior_bb_rate"]:
        out[col] = np.nan
    league = out.groupby("season").agg(
        league_pitcher_k_rate=("k_full", lambda s: np.nan),
    )
    _ = league
    lookup = out.set_index(["pitcher", "season"])
    for idx, row in out.iterrows():
        pitcher = row["pitcher"]
        season = int(row["season"])
        weights = []
        vals_k = []
        vals_bb = []
        for lag, weight in [(1, 5.0), (2, 4.0), (3, 3.0)]:
            key = (pitcher, season - lag)
            if key in lookup.index:
                prior = lookup.loc[key]
                if pd.notna(prior.get("k_rate_full")):
                    vals_k.append(float(prior["k_rate_full"]))
                    weights.append(weight)
                if pd.notna(prior.get("bb_rate_full")):
                    vals_bb.append((float(prior["bb_rate_full"]), weight))
        if vals_k and weights:
            out.at[idx, "preseason_prior_k_rate"] = float(np.average(vals_k, weights=weights[: len(vals_k)]))
        elif pd.notna(row.get("league_pitcher_k_rate")):
            out.at[idx, "preseason_prior_k_rate"] = float(row["league_pitcher_k_rate"])
        clean_bb = [(v, w) for v, w in vals_bb if np.isfinite(v)]
        if clean_bb:
            out.at[idx, "preseason_prior_bb_rate"] = sum(v * w for v, w in clean_bb) / sum(w for _, w in clean_bb)
        elif pd.notna(row.get("league_pitcher_bb_rate")):
            out.at[idx, "preseason_prior_bb_rate"] = float(row["league_pitcher_bb_rate"])
    return out


def add_2026_pitcher_prior_columns(pitchers_2026: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    out = pitchers_2026.copy()
    hist = historical[historical["season"].between(2023, 2025)].copy()
    league_2025 = historical[historical["season"].eq(2025)].iloc[0]
    for col in ["preseason_prior_k_rate", "preseason_prior_bb_rate"]:
        out[col] = np.nan
    for idx, row in out.iterrows():
        vals = hist[hist["pitcher"].eq(row["pitcher"])].sort_values("season", ascending=False).head(3)
        weights = np.array([5.0, 4.0, 3.0])[: len(vals)]
        if len(vals):
            out.at[idx, "preseason_prior_k_rate"] = float(np.average(vals["k_rate_full"], weights=weights))
            out.at[idx, "preseason_prior_bb_rate"] = float(np.average(vals["bb_rate_full"], weights=weights))
        else:
            out.at[idx, "preseason_prior_k_rate"] = float(league_2025["league_pitcher_k_rate"])
            out.at[idx, "preseason_prior_bb_rate"] = float(league_2025["league_pitcher_bb_rate"])
    for col in R2_RELIEVER_FEATURE_COLS:
        if col not in out.columns:
            out[col] = np.nan
    return out


@dataclass
class LeafQuantileRegressor:
    n_estimators: int = 600
    min_samples_leaf: int = 5
    max_features: str | float = "sqrt"
    random_state: int = SEED
    n_jobs: int = -1

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LeafQuantileRegressor":
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(x, y)
        self.y_train_ = np.asarray(y, dtype=float)
        self.leaf_values_: list[dict[int, np.ndarray]] = []
        for estimator in self.model.estimators_:
            leaves = estimator.apply(x)
            mapping: dict[int, list[float]] = {}
            for leaf, target in zip(leaves, self.y_train_):
                mapping.setdefault(int(leaf), []).append(float(target))
            self.leaf_values_.append({leaf: np.asarray(vals, dtype=float) for leaf, vals in mapping.items()})
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def predict_quantiles(self, x: np.ndarray, quantiles: Iterable[float]) -> np.ndarray:
        quantiles = list(quantiles)
        rows = []
        for i in range(x.shape[0]):
            vals = []
            sample = x[i : i + 1]
            for estimator, mapping in zip(self.model.estimators_, self.leaf_values_):
                leaf = int(estimator.apply(sample)[0])
                leaf_vals = mapping.get(leaf)
                if leaf_vals is not None and len(leaf_vals):
                    vals.append(leaf_vals)
            if vals:
                joined = np.concatenate(vals)
            else:
                joined = self.model.predict(sample)
            rows.append(np.quantile(joined, quantiles))
        return np.asarray(rows)


def fit_leaf_qrf(train: pd.DataFrame, feature_cols: list[str], target: str, seed_offset: int = 0):
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(train[feature_cols])
    y = train[target].to_numpy(dtype=float)
    model = LeafQuantileRegressor(random_state=SEED + seed_offset).fit(x, y)
    return model, imputer


def predict_qrf_frame(model: LeafQuantileRegressor, imputer: SimpleImputer, df: pd.DataFrame, feature_cols: list[str], margin: float = 0.0) -> pd.DataFrame:
    x = imputer.transform(df[feature_cols])
    qs = model.predict_quantiles(x, [0.10, 0.50, 0.90])
    out = pd.DataFrame({"q10": qs[:, 0] - margin, "q50": qs[:, 1], "q90": qs[:, 2] + margin}, index=df.index)
    return out


def conformal_interval_margin(y: np.ndarray, q10: np.ndarray, q90: np.ndarray, alpha: float = 0.20) -> float:
    misses = np.maximum.reduce([q10 - y, y - q90, np.zeros_like(y, dtype=float)])
    if len(misses) == 0:
        return 0.0
    q = min(1.0, (1.0 - alpha) * (len(misses) + 1) / len(misses))
    return float(np.quantile(misses, q))


def interval_coverage(y: Iterable[float], q10: Iterable[float], q90: Iterable[float]) -> float:
    y_arr = np.asarray(list(y), dtype=float)
    lo = np.asarray(list(q10), dtype=float)
    hi = np.asarray(list(q90), dtype=float)
    return float(np.mean((y_arr >= lo) & (y_arr <= hi))) if len(y_arr) else float("nan")


def write_joblib(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

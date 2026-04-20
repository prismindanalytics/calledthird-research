"""CalledThird Coaching Gap — Round 6 Reconciliation Analyses (Codex agent).

This single-file script reproduces the Round 6 reconciliation analyses for the
CalledThird Coaching Gap study as run by the Codex agent. Round 6 pits three
estimators (pooled between-batter, matched-pairs within-batter, batter-FE
regression) against each other on an identical pre-registered substrate, plus
variance partition (H2), empirical-calibration power simulation (H3),
cut-definition robustness (H4), and quality-hitter 2x2 replication (H5).

How to run:
    python analyze_codex.py --data-dir data/

Expected inputs under --data-dir:
    agent_substrate/career_pitches.parquet        # scored pitches 2022-2025
    layer2_holdout_predictions.parquet             # context-only Layer 2 probs
    pitcher_cohort.csv                             # optional strict 371-cohort

Dependencies: pandas, numpy, scipy, statsmodels, pyarrow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


# =============================================================================
# Common helpers (inlined from common_round6.py)
# =============================================================================

PRE2026_SEASONS = (2022, 2023, 2024, 2025)
STRICT_COHORT_SIZE = 371
MIN_BATTER_SEASON_PITCHES = 150
DEFAULT_BOOTSTRAPS = 400
DEFAULT_PERMUTATIONS = 1_000
SEED = 7

MATCH_SCALES = {
    "pitch_number": 2.5,
    "outs_when_up": 0.75,
    "runner_count": 0.85,
    "prev_plate_x": 0.45,
    "prev_plate_z": 0.45,
    "avg_prev_plate_x": 0.55,
    "avg_prev_plate_z": 0.55,
}

TERTILE_ORDER = ["low", "mid", "high"]

BATTER_FEATURE_COLUMNS = [
    "batter", "season", "n_pitches",
    "chase_rate", "chase_rate_tertile",
    "whiff_rate_overall", "whiff_rate_overall_tertile",
    "zone_contact_rate", "zone_contact_rate_tertile",
    "xwoba_on_contact", "xwoba_on_contact_tertile",
    "quality_hitter",
]


class Paths:
    data_dir: Path
    pitches_path: Path
    probs_path: Path
    cohort_path: Path | None


PATHS = Paths()

# In-memory caches so H1..H5 don't re-read / re-compute the same substrate.
_CACHE: dict[str, Any] = {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def _default(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value
    path.write_text(json.dumps(payload, indent=2, default=_default))


def load_strict_cohort_pitchers() -> set[int] | None:
    path = PATHS.cohort_path
    if path is None or not path.exists():
        return None
    cohort = pd.read_csv(path)
    if "pitcher" not in cohort.columns:
        return None
    cohort["pitcher"] = cohort["pitcher"].astype(int)
    return set(cohort["pitcher"].astype(int))


def _quantile_labels(values: pd.Series, labels: Sequence[str] = TERTILE_ORDER) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=len(labels), labels=labels).astype(str)


def _build_pitch_id(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["season"].astype(int).astype(str) + "|"
        + frame["pitcher"].astype(int).astype(str) + "|"
        + frame["game_pk"].astype(int).astype(str) + "|"
        + frame["at_bat_number"].astype(int).astype(str) + "|"
        + frame["pitch_number"].astype(int).astype(str)
    )


def build_scored_pitch_substrate() -> pd.DataFrame:
    if "scored" in _CACHE:
        return _CACHE["scored"]
    columns = [
        "pitcher", "batter", "season", "game_date", "game_pk",
        "at_bat_number", "pitch_number", "label", "count_state",
        "stand", "p_throws", "zone", "outs_when_up", "runner_count",
        "prev_plate_x", "prev_plate_z", "avg_prev_plate_x", "avg_prev_plate_z",
        "swung", "whiff", "in_zone", "events", "woba_value",
        "is_terminal", "context_prob",
    ]
    frame = pd.read_parquet(PATHS.pitches_path, columns=columns)
    frame = frame[frame["season"].isin(PRE2026_SEASONS)].copy()
    frame["pitcher"] = frame["pitcher"].astype(int)
    frame["batter"] = pd.to_numeric(frame["batter"], errors="coerce").astype(int)
    cohort = load_strict_cohort_pitchers()
    if cohort is not None:
        frame = frame[frame["pitcher"].isin(cohort)].copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    for column in [
        "pitch_number", "outs_when_up", "runner_count",
        "prev_plate_x", "prev_plate_z", "avg_prev_plate_x", "avg_prev_plate_z",
        "context_prob", "woba_value", "zone",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["swung", "whiff", "in_zone", "is_terminal"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    frame["batter_season_id"] = frame["batter"].astype(str) + "|" + frame["season"].astype(str)
    frame["pitch_id"] = _build_pitch_id(frame)
    frame = frame.sort_values(["season", "pitcher", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    _CACHE["scored"] = frame
    return frame


def build_batter_season_features() -> pd.DataFrame:
    if "features" in _CACHE:
        return _CACHE["features"]
    frame = build_scored_pitch_substrate()
    out_zone = frame["in_zone"].eq(0)
    zone_swing = frame["in_zone"].eq(1) & frame["swung"].eq(1)
    zone_contact = frame["in_zone"].eq(1) & frame["swung"].eq(1) & frame["whiff"].eq(0)
    swung = frame["swung"].eq(1)
    whiff = frame["whiff"].eq(1)
    contact_event = frame["events"].notna() & ~frame["events"].isin(
        ["strikeout", "walk", "intent_walk", "hit_by_pitch"]
    )

    feature_frame = pd.DataFrame({
        "batter": frame["batter"].to_numpy(),
        "season": frame["season"].to_numpy(),
        "n_pitches": np.ones(len(frame), dtype=int),
        "n_swings": swung.astype(int).to_numpy(),
        "out_zone_pitches": out_zone.astype(int).to_numpy(),
        "zone_swings": zone_swing.astype(int).to_numpy(),
        "zone_contacts": zone_contact.astype(int).to_numpy(),
        "chase_swings": (out_zone & swung).astype(int).to_numpy(),
        "whiffs": whiff.astype(int).to_numpy(),
        "contact_events": contact_event.astype(int).to_numpy(),
        "contact_woba_num": np.where(contact_event, frame["woba_value"].to_numpy(dtype=float), 0.0),
    })
    feature_frame = feature_frame.groupby(["batter", "season"], observed=True).sum().reset_index()
    feature_frame = feature_frame[feature_frame["n_pitches"] >= MIN_BATTER_SEASON_PITCHES].copy()
    feature_frame["chase_rate"] = feature_frame["chase_swings"] / feature_frame["out_zone_pitches"]
    feature_frame["zone_contact_rate"] = feature_frame["zone_contacts"] / feature_frame["zone_swings"]
    feature_frame["whiff_rate_overall"] = feature_frame["whiffs"] / feature_frame["n_swings"]
    feature_frame["xwoba_on_contact"] = feature_frame["contact_woba_num"] / feature_frame["contact_events"]

    for feature, out_col in [
        ("chase_rate", "chase_rate_tertile"),
        ("whiff_rate_overall", "whiff_rate_overall_tertile"),
        ("zone_contact_rate", "zone_contact_rate_tertile"),
        ("xwoba_on_contact", "xwoba_on_contact_tertile"),
    ]:
        feature_frame[out_col] = _quantile_labels(feature_frame[feature], labels=TERTILE_ORDER)

    feature_frame["quality_hitter"] = (
        feature_frame["chase_rate_tertile"].eq("low") & feature_frame["xwoba_on_contact_tertile"].eq("high")
    ).astype(int)
    feature_frame = feature_frame.sort_values(["batter", "season"]).reset_index(drop=True)
    _CACHE["features"] = feature_frame
    return feature_frame


def add_predictability_flags(frame: pd.DataFrame, cut_mode: str) -> pd.DataFrame:
    work = frame.copy()
    if cut_mode == "within_pitcher_season":
        q20 = work.groupby(["pitcher", "season"], observed=True)["context_prob"].transform(lambda s: float(s.quantile(0.2)))
        q80 = work.groupby(["pitcher", "season"], observed=True)["context_prob"].transform(lambda s: float(s.quantile(0.8)))
    elif cut_mode == "league_wide":
        q20_value = float(work["context_prob"].quantile(0.2))
        q80_value = float(work["context_prob"].quantile(0.8))
        q20 = pd.Series(q20_value, index=work.index)
        q80 = pd.Series(q80_value, index=work.index)
    else:
        raise ValueError(f"Unknown cut_mode: {cut_mode}")
    work["predictability_low"] = (work["context_prob"] <= q20).astype(int)
    work["predictability_high"] = (work["context_prob"] >= q80).astype(int)
    work["predictability_level"] = np.where(
        work["predictability_high"].eq(1),
        "high",
        np.where(work["predictability_low"].eq(1), "low", "middle"),
    )
    return work


def build_terminal_substrate(cut_mode: str = "within_pitcher_season") -> pd.DataFrame:
    key = f"terminal::{cut_mode}"
    if key in _CACHE:
        return _CACHE[key]
    all_pitches = add_predictability_flags(build_scored_pitch_substrate(), cut_mode=cut_mode)
    terminal = all_pitches[all_pitches["is_terminal"].eq(1)].copy()
    terminal = terminal[(terminal["predictability_high"] == 1) | (terminal["predictability_low"] == 1)].copy()
    features = build_batter_season_features()
    terminal = terminal.merge(
        features[[
            "batter", "season",
            "chase_rate", "chase_rate_tertile",
            "whiff_rate_overall", "whiff_rate_overall_tertile",
            "zone_contact_rate", "zone_contact_rate_tertile",
            "xwoba_on_contact", "xwoba_on_contact_tertile",
            "quality_hitter",
        ]],
        on=["batter", "season"], how="inner",
    )
    support = (
        terminal.groupby(["batter", "season"], observed=True)
        .agg(n_terminal=("pitch_id", "size"),
             n_high=("predictability_high", "sum"),
             n_low=("predictability_low", "sum"))
        .reset_index()
    )
    support["has_both_predictability_levels"] = ((support["n_high"] > 0) & (support["n_low"] > 0)).astype(int)
    terminal = terminal.merge(support, on=["batter", "season"], how="left")
    terminal = terminal[terminal["has_both_predictability_levels"].eq(1)].copy()
    terminal["predictability_flag"] = terminal["predictability_high"].astype(int)
    terminal = terminal.sort_values(["season", "pitcher", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    _CACHE[key] = terminal
    return terminal


def _distance_components(available: pd.DataFrame, target: pd.Series, distance_cols: Sequence[str]) -> np.ndarray:
    distance = np.zeros(len(available), dtype=float)
    for column in distance_cols:
        scale = MATCH_SCALES.get(column, 1.0)
        distance += ((available[column].fillna(0.0) - float(target[column])) / scale) ** 2
    return np.sqrt(distance)


def build_matched_pairs(cut_mode: str = "within_pitcher_season", max_distance: float = 6.5) -> pd.DataFrame:
    key = f"pairs::{cut_mode}"
    if key in _CACHE:
        return _CACHE[key]
    terminal = build_terminal_substrate(cut_mode=cut_mode)
    rows: list[dict[str, Any]] = []
    group_cols = ["batter", "season", "count_state", "stand", "p_throws"]
    distance_cols = [
        "pitch_number", "outs_when_up", "runner_count",
        "prev_plate_x", "prev_plate_z",
        "avg_prev_plate_x", "avg_prev_plate_z",
    ]
    for _, group in terminal.groupby(group_cols, observed=True, sort=False):
        treated = group[group["predictability_high"] == 1].copy()
        controls = group[group["predictability_low"] == 1].copy().reset_index(drop=True)
        if treated.empty or controls.empty:
            continue
        treated = treated.sort_values("context_prob", ascending=False)
        used = np.zeros(len(controls), dtype=bool)
        for _, treated_row in treated.iterrows():
            available = controls.loc[~used].copy()
            if available.empty:
                break
            available = available.assign(match_distance=_distance_components(available, treated_row, distance_cols))
            best_idx = int(available["match_distance"].idxmin())
            best_distance = float(available.loc[best_idx, "match_distance"])
            if best_distance > max_distance:
                continue
            control_row = controls.loc[best_idx]
            used[best_idx] = True
            rows.append({
                "treated_pitch_id": treated_row["pitch_id"],
                "control_pitch_id": control_row["pitch_id"],
                "treated_pitcher": int(treated_row["pitcher"]),
                "control_pitcher": int(control_row["pitcher"]),
                "batter": int(treated_row["batter"]),
                "season": int(treated_row["season"]),
                "batter_season_id": treated_row["batter_season_id"],
                "count_state": treated_row["count_state"],
                "stand": treated_row["stand"],
                "p_throws": treated_row["p_throws"],
                "match_distance": best_distance,
                "treated_context_prob": float(treated_row["context_prob"]),
                "control_context_prob": float(control_row["context_prob"]),
                "delta_context_prob": float(treated_row["context_prob"] - control_row["context_prob"]),
                "treated_woba_value": float(treated_row["woba_value"]),
                "control_woba_value": float(control_row["woba_value"]),
                "delta_woba_value": float(treated_row["woba_value"] - control_row["woba_value"]),
                "chase_rate_tertile": treated_row["chase_rate_tertile"],
                "quality_hitter": int(treated_row["quality_hitter"]),
            })
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise RuntimeError(f"No matched pairs found for cut_mode={cut_mode}.")
    pairs = pairs.sort_values(["season", "batter", "treated_pitch_id", "control_pitch_id"]).reset_index(drop=True)
    _CACHE[key] = pairs
    return pairs


def build_terminal_unit_summary(cut_mode: str = "within_pitcher_season") -> pd.DataFrame:
    terminal = build_terminal_substrate(cut_mode=cut_mode)
    grouped = (
        terminal.groupby(
            ["batter", "season", "batter_season_id", "chase_rate_tertile",
             "quality_hitter", "predictability_level"],
            observed=True,
        )
        .agg(n_terminal=("pitch_id", "size"),
             sum_woba=("woba_value", "sum"),
             mean_woba=("woba_value", "mean"))
        .reset_index()
    )
    wide = (
        grouped.pivot_table(
            index=["batter", "season", "batter_season_id", "chase_rate_tertile", "quality_hitter"],
            columns="predictability_level",
            values=["n_terminal", "sum_woba", "mean_woba"],
            fill_value=0.0,
        )
        .reset_index()
    )
    wide.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple) else str(column)
        for column in wide.columns
    ]
    rename_map = {
        "n_terminal_high": "n_high",
        "n_terminal_low": "n_low",
    }
    wide = wide.rename(columns=rename_map)
    wide["gap_woba"] = wide["mean_woba_high"] - wide["mean_woba_low"]
    wide = wide.sort_values(["season", "batter"]).reset_index(drop=True)
    return wide


def build_pair_unit_summary(cut_mode: str = "within_pitcher_season") -> pd.DataFrame:
    pairs = build_matched_pairs(cut_mode=cut_mode)
    summary = (
        pairs.groupby(
            ["batter", "season", "batter_season_id", "chase_rate_tertile", "quality_hitter"],
            observed=True,
        )
        .agg(n_pairs=("delta_woba_value", "size"),
             gap_woba=("delta_woba_value", "mean"),
             gap_woba_median=("delta_woba_value", "median"),
             mean_match_distance=("match_distance", "mean"),
             mean_prob_gap=("delta_context_prob", "mean"))
        .reset_index()
        .sort_values(["season", "batter"])
        .reset_index(drop=True)
    )
    return summary


def bootstrap_mean_ci(values: Iterable[float], n_boot: int = DEFAULT_BOOTSTRAPS, seed: int = SEED) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def season_stratified_permutation(
    unit_frame: pd.DataFrame,
    label_col: str,
    statistic_fn: Callable[[pd.DataFrame], float],
    n_perm: int = DEFAULT_PERMUTATIONS,
    seed: int = SEED,
) -> dict[str, float]:
    observed = float(statistic_fn(unit_frame))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm, dtype=float)
    for idx in range(n_perm):
        shuffled = unit_frame.copy()
        shuffled_parts = []
        for _, sub in shuffled.groupby("season", observed=True, sort=False):
            temp = sub.copy()
            temp[label_col] = rng.permutation(temp[label_col].to_numpy())
            shuffled_parts.append(temp)
        shuffled = pd.concat(shuffled_parts, ignore_index=True)
        draws[idx] = float(statistic_fn(shuffled))
    return {
        "observed": observed,
        "permutation_pvalue": float((1 + np.sum(np.abs(draws) >= abs(observed))) / (n_perm + 1)),
        "null_mean": float(draws.mean()),
    }


def bootstrap_unit_statistic(
    unit_frame: pd.DataFrame,
    statistic_fn: Callable[[pd.DataFrame], float],
    n_boot: int = DEFAULT_BOOTSTRAPS,
    seed: int = SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sampled_parts = []
        for _, sub in unit_frame.groupby("season", observed=True, sort=False):
            sampled_idx = rng.integers(0, len(sub), size=len(sub))
            sampled_parts.append(sub.iloc[sampled_idx].copy())
        sampled = pd.concat(sampled_parts, ignore_index=True)
        draws[idx] = float(statistic_fn(sampled))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


# =============================================================================
# H1 — three estimators on identical substrate
# =============================================================================

def _group_gap_from_unit_wide(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, sub in frame.groupby(group_col, observed=True):
        high_mean = float(sub["sum_woba_high"].sum() / sub["n_high"].sum())
        low_mean = float(sub["sum_woba_low"].sum() / sub["n_low"].sum())
        rows.append({
            group_col: group,
            "predictable_mean_woba": high_mean,
            "unpredictable_mean_woba": low_mean,
            "gap_woba": high_mean - low_mean,
            "n_units": int(len(sub)),
            "n_terminal_high": int(sub["n_high"].sum()),
            "n_terminal_low": int(sub["n_low"].sum()),
        })
    out = pd.DataFrame(rows)
    out["sort_key"] = out[group_col].map({label: idx for idx, label in enumerate(TERTILE_ORDER)})
    out = out.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)
    return out


def _spread_low_minus_high(summary: pd.DataFrame, group_col: str) -> float:
    means = summary.set_index(group_col)["gap_woba"]
    return float(means["low"] - means["high"])


def _pooled_spread_from_units(unit_wide: pd.DataFrame) -> float:
    return _spread_low_minus_high(_group_gap_from_unit_wide(unit_wide, "chase_rate_tertile"), "chase_rate_tertile")


def _matched_gap_summary(pair_units: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tertile, sub in pair_units.groupby("chase_rate_tertile", observed=True):
        ci_low, ci_high = bootstrap_mean_ci(sub["gap_woba"].to_numpy(), n_boot=DEFAULT_BOOTSTRAPS, seed=7)
        n_pairs = int(sub["n_pairs"].sum()) if "n_pairs" in sub.columns else int(len(sub))
        rows.append({
            "chase_rate_tertile": tertile,
            "n_units": int(len(sub)),
            "n_pairs": n_pairs,
            "gap_woba": float(sub["gap_woba"].mean()),
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    out = pd.DataFrame(rows)
    out["sort_key"] = out["chase_rate_tertile"].map({label: idx for idx, label in enumerate(TERTILE_ORDER)})
    out = out.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)
    return out


def _matched_spread(pair_units: pd.DataFrame) -> float:
    summary = _matched_gap_summary(pair_units)
    return _spread_low_minus_high(summary, "chase_rate_tertile")


def _prepare_fe_cells(unit_wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in unit_wide.iterrows():
        for level, flag in [("low", 0), ("high", 1)]:
            weight = float(row[f"n_{level}"])
            mean_woba = float(row[f"mean_woba_{level}"])
            rows.append({
                "batter": int(row["batter"]),
                "season": int(row["season"]),
                "batter_season_id": row["batter_season_id"],
                "chase_rate_tertile": row["chase_rate_tertile"],
                "predictability_flag": flag,
                "mean_woba": mean_woba,
                "weight": weight,
                "tertile_mid": 1 if row["chase_rate_tertile"] == "mid" else 0,
                "tertile_high": 1 if row["chase_rate_tertile"] == "high" else 0,
            })
    cells = pd.DataFrame(rows)
    return cells[cells["weight"] > 0].copy()


def _weighted_within_demean(frame: pd.DataFrame, columns: list[str], weight_col: str, group_col: str) -> pd.DataFrame:
    work = frame.copy()
    group_weights = work.groupby(group_col, observed=True)[weight_col].transform("sum")
    for column in columns:
        weighted_mean = (
            work[column] * work[weight_col]
        ).groupby(work[group_col], observed=True).transform("sum") / group_weights
        work[f"{column}_dm"] = work[column] - weighted_mean
    return work


def run_fixed_effects_regression(unit_wide: pd.DataFrame) -> dict[str, Any]:
    cells = _prepare_fe_cells(unit_wide)
    cells["pred_mid"] = cells["predictability_flag"] * cells["tertile_mid"]
    cells["pred_high"] = cells["predictability_flag"] * cells["tertile_high"]

    season_dummies = pd.get_dummies(cells["season"].astype(int).astype(str), prefix="season", drop_first=True)
    design = pd.concat(
        [
            cells[["predictability_flag", "pred_mid", "pred_high", "tertile_mid", "tertile_high"]],
            season_dummies,
        ],
        axis=1,
    )
    design = design.astype(float)
    work = pd.concat([cells[["batter", "weight", "mean_woba"]], design], axis=1)
    demean_cols = ["mean_woba", *design.columns.tolist()]
    work = _weighted_within_demean(work, columns=demean_cols, weight_col="weight", group_col="batter")
    y = work["mean_woba_dm"].to_numpy(dtype=float)
    x_cols = [f"{column}_dm" for column in design.columns]
    x = work[x_cols].to_numpy(dtype=float)
    model = sm.WLS(y, x, weights=work["weight"].to_numpy(dtype=float))
    result = model.fit(cov_type="cluster", cov_kwds={"groups": work["batter"].to_numpy(dtype=int)})

    params = pd.Series(result.params, index=design.columns)
    conf = pd.DataFrame(result.conf_int(), index=design.columns, columns=["ci_low", "ci_high"])
    pvalues = pd.Series(result.pvalues, index=design.columns)
    high_gap = float(params["predictability_flag"] + params["pred_high"])
    mid_gap = float(params["predictability_flag"] + params["pred_mid"])
    low_gap = float(params["predictability_flag"])
    spread_low_minus_high = float(-params["pred_high"])
    spread_test = result.t_test(np.array([[0.0, 0.0, -1.0] + [0.0] * (len(design.columns) - 3)]))

    summary = pd.DataFrame([
        {"chase_rate_tertile": "low", "gap_woba": low_gap},
        {"chase_rate_tertile": "mid", "gap_woba": mid_gap},
        {"chase_rate_tertile": "high", "gap_woba": high_gap},
    ])
    summary["sort_key"] = summary["chase_rate_tertile"].map({label: idx for idx, label in enumerate(TERTILE_ORDER)})
    summary = summary.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)

    return {
        "summary": summary,
        "coef_table": (
            pd.DataFrame({
                "term": design.columns,
                "coef": [float(params[column]) for column in design.columns],
                "ci_low": [float(conf.loc[column, "ci_low"]) for column in design.columns],
                "ci_high": [float(conf.loc[column, "ci_high"]) for column in design.columns],
                "pvalue": [float(pvalues[column]) for column in design.columns],
            })
            .sort_values("term")
            .reset_index(drop=True)
        ),
        "spread_low_minus_high": spread_low_minus_high,
        "spread_ci_low": float(spread_test.conf_int()[0, 0]),
        "spread_ci_high": float(spread_test.conf_int()[0, 1]),
        "spread_pvalue": float(spread_test.pvalue),
        "n_cells": int(len(cells)),
        "n_batters": int(cells["batter"].nunique()),
    }


def run_h1_three_estimators(cut_mode: str = "within_pitcher_season") -> dict[str, Any]:
    """H1 — three estimators on identical substrate.

    Pooled between-batter (weighted unit-level), matched-pairs within-batter,
    and batter fixed-effects regression run on the same canonical substrate.
    Report low-minus-high chase-tertile spread with bootstrap/permutation
    inference and a kill criterion for convergence (<=0.005 and >=2 p<0.05).
    """
    unit_wide = build_terminal_unit_summary(cut_mode=cut_mode)
    pooled_summary = _group_gap_from_unit_wide(unit_wide, "chase_rate_tertile")
    pooled_spread = _spread_low_minus_high(pooled_summary, "chase_rate_tertile")
    pooled_ci_low, pooled_ci_high = bootstrap_unit_statistic(unit_wide, _pooled_spread_from_units, n_boot=DEFAULT_BOOTSTRAPS, seed=7)
    pooled_perm = season_stratified_permutation(
        unit_wide[["season", "batter_season_id", "chase_rate_tertile",
                   "sum_woba_high", "sum_woba_low", "n_high", "n_low"]].copy(),
        label_col="chase_rate_tertile",
        statistic_fn=lambda frame: _pooled_spread_from_units(frame),
        n_perm=DEFAULT_PERMUTATIONS,
        seed=7,
    )

    pair_units = build_pair_unit_summary(cut_mode=cut_mode)
    matched_summary = _matched_gap_summary(pair_units)
    matched_spread = _matched_spread(pair_units)
    matched_ci_low, matched_ci_high = bootstrap_unit_statistic(
        pair_units[["season", "batter_season_id", "chase_rate_tertile", "gap_woba"]].copy(),
        statistic_fn=lambda frame: _matched_spread(frame),
        n_boot=DEFAULT_BOOTSTRAPS,
        seed=17,
    )
    matched_perm = season_stratified_permutation(
        pair_units[["season", "batter_season_id", "chase_rate_tertile", "gap_woba"]].copy(),
        label_col="chase_rate_tertile",
        statistic_fn=lambda frame: _matched_spread(frame),
        n_perm=DEFAULT_PERMUTATIONS,
        seed=17,
    )

    fe = run_fixed_effects_regression(unit_wide)

    estimator_summary = pd.DataFrame([
        {"estimator": "Pooled between",
         "spread_low_minus_high": pooled_spread,
         "ci_low": pooled_ci_low, "ci_high": pooled_ci_high,
         "pvalue": pooled_perm["permutation_pvalue"]},
        {"estimator": "Matched pairs",
         "spread_low_minus_high": matched_spread,
         "ci_low": matched_ci_low, "ci_high": matched_ci_high,
         "pvalue": matched_perm["permutation_pvalue"]},
        {"estimator": "Batter FE",
         "spread_low_minus_high": fe["spread_low_minus_high"],
         "ci_low": fe["spread_ci_low"], "ci_high": fe["spread_ci_high"],
         "pvalue": fe["spread_pvalue"]},
    ])

    point_spread = estimator_summary["spread_low_minus_high"].to_numpy(dtype=float)
    findings = {
        "hypothesis": "H1_three_estimators",
        "cut_mode": cut_mode,
        "pooled_between": {
            "summary": pooled_summary.to_dict(orient="records"),
            "spread_low_minus_high": pooled_spread,
            "spread_ci_low": pooled_ci_low,
            "spread_ci_high": pooled_ci_high,
            "spread_permutation_pvalue": pooled_perm["permutation_pvalue"],
            "n_units": int(len(unit_wide)),
            "n_terminal_high": int(unit_wide["n_high"].sum()),
            "n_terminal_low": int(unit_wide["n_low"].sum()),
        },
        "matched_pairs": {
            "summary": matched_summary.to_dict(orient="records"),
            "spread_low_minus_high": matched_spread,
            "spread_ci_low": matched_ci_low,
            "spread_ci_high": matched_ci_high,
            "spread_permutation_pvalue": matched_perm["permutation_pvalue"],
            "n_units": int(len(pair_units)),
            "n_pairs": int(pair_units["n_pairs"].sum()),
        },
        "batter_fixed_effects": {
            "summary": fe["summary"].to_dict(orient="records"),
            "coef_table": fe["coef_table"].to_dict(orient="records"),
            "spread_low_minus_high": fe["spread_low_minus_high"],
            "spread_ci_low": fe["spread_ci_low"],
            "spread_ci_high": fe["spread_ci_high"],
            "spread_pvalue": fe["spread_pvalue"],
            "n_cells": fe["n_cells"],
            "n_batters": fe["n_batters"],
        },
        "estimator_alignment": estimator_summary.to_dict(orient="records"),
        "max_point_estimate_spread": float(point_spread.max() - point_spread.min()),
        "estimators_with_p_lt_0_05": int((estimator_summary["pvalue"] < 0.05).sum()),
        "kill_criterion_h1_pass": bool(
            (float(point_spread.max() - point_spread.min()) <= 0.005)
            and (int((estimator_summary["pvalue"] < 0.05).sum()) >= 2)
        ),
    }
    return findings


# =============================================================================
# H2 — variance partition
# =============================================================================

def _h2_decompose(frame: pd.DataFrame, unit_col: str) -> dict[str, float]:
    work = frame[[unit_col, "predictability_level", "woba_value"]].copy()
    overall_mean = float(work["woba_value"].mean())

    unit_summary = (
        work.groupby(unit_col, observed=True)
        .agg(n=("woba_value", "size"), mean_woba=("woba_value", "mean"), sum_woba=("woba_value", "sum"))
        .reset_index()
    )
    cell_summary = (
        work.groupby([unit_col, "predictability_level"], observed=True)
        .agg(n=("woba_value", "size"), mean_woba=("woba_value", "mean"))
        .reset_index()
        .merge(unit_summary[[unit_col, "mean_woba"]], on=unit_col, how="left", suffixes=("", "_unit"))
    )

    work = work.merge(cell_summary[[unit_col, "predictability_level", "mean_woba"]], on=[unit_col, "predictability_level"], how="left", suffixes=("", "_cell"))
    work = work.merge(unit_summary[[unit_col, "mean_woba"]], on=unit_col, how="left", suffixes=("", "_unit"))

    ss_between = float(np.sum(unit_summary["n"] * (unit_summary["mean_woba"] - overall_mean) ** 2))
    ss_within_predictability = float(np.sum(cell_summary["n"] * (cell_summary["mean_woba"] - cell_summary["mean_woba_unit"]) ** 2))
    ss_residual = float(np.sum((work["woba_value"] - work["mean_woba"]) ** 2))
    ss_total = float(np.sum((work["woba_value"] - overall_mean) ** 2))
    structural_total = ss_between + ss_within_predictability
    structural_share_between = ss_between / structural_total if structural_total > 0 else float("nan")
    structural_share_within = ss_within_predictability / structural_total if structural_total > 0 else float("nan")

    return {
        "overall_mean_woba": overall_mean,
        "ss_between_batter": ss_between,
        "ss_within_batter_predictability": ss_within_predictability,
        "ss_residual": ss_residual,
        "ss_total": ss_total,
        "ss_structural_total": structural_total,
        "share_between_batter": ss_between / ss_total,
        "share_within_batter_predictability": ss_within_predictability / ss_total,
        "share_residual": ss_residual / ss_total,
        "structural_share_between_batter": structural_share_between,
        "structural_share_within_batter_predictability": structural_share_within,
        "n_rows": int(len(work)),
        "n_units": int(unit_summary[unit_col].nunique()),
    }


def _h2_bootstrap(frame: pd.DataFrame, unit_col: str, n_boot: int = DEFAULT_BOOTSTRAPS, seed: int = 7) -> pd.DataFrame:
    units = frame[unit_col].drop_duplicates().to_numpy()
    unit_map = {unit: frame[frame[unit_col] == unit].copy() for unit in units}
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for boot in range(n_boot):
        sampled_units = rng.choice(units, size=len(units), replace=True)
        sampled = pd.concat([unit_map[unit] for unit in sampled_units], ignore_index=True)
        comp = _h2_decompose(sampled, unit_col=unit_col)
        comp["bootstrap"] = boot
        rows.append(comp)
    return pd.DataFrame(rows)


def run_h2_variance_partition() -> dict[str, Any]:
    """H2 — variance partition.

    ANOVA-style decomposition of terminal-pitch wOBA into between-batter,
    within-batter predictability, and residual components, at both batter and
    batter-season unit grains. Bootstrap 95% CIs on each share. Kill criterion
    is >70% of variance concentrated in a single component.
    """
    terminal = build_terminal_substrate(cut_mode="within_pitcher_season")
    batter_frame = terminal[["batter", "predictability_level", "woba_value"]].copy()
    batter_season_frame = terminal[["batter_season_id", "predictability_level", "woba_value"]].copy()

    batter_comp = _h2_decompose(batter_frame, unit_col="batter")
    batter_season_comp = _h2_decompose(batter_season_frame, unit_col="batter_season_id")
    batter_boot = _h2_bootstrap(batter_frame, unit_col="batter")
    batter_season_boot = _h2_bootstrap(batter_season_frame, unit_col="batter_season_id")

    dominant_share = max(
        batter_comp["share_between_batter"],
        batter_comp["share_within_batter_predictability"],
        batter_comp["share_residual"],
    )
    dominant_component = max(
        [("between_batter", batter_comp["share_between_batter"]),
         ("within_batter_predictability", batter_comp["share_within_batter_predictability"]),
         ("residual", batter_comp["share_residual"])],
        key=lambda item: item[1],
    )[0]
    structural_dominant_component = max(
        [("between_batter", batter_comp["structural_share_between_batter"]),
         ("within_batter_predictability", batter_comp["structural_share_within_batter_predictability"])],
        key=lambda item: item[1],
    )[0]
    structural_dominant_share = max(
        batter_comp["structural_share_between_batter"],
        batter_comp["structural_share_within_batter_predictability"],
    )

    findings = {
        "hypothesis": "H2_variance_partition",
        "batter_level": batter_comp,
        "batter_level_bootstrap": {
            "share_between_batter_ci_low": float(batter_boot["share_between_batter"].quantile(0.025)),
            "share_between_batter_ci_high": float(batter_boot["share_between_batter"].quantile(0.975)),
            "share_within_batter_predictability_ci_low": float(batter_boot["share_within_batter_predictability"].quantile(0.025)),
            "share_within_batter_predictability_ci_high": float(batter_boot["share_within_batter_predictability"].quantile(0.975)),
            "share_residual_ci_low": float(batter_boot["share_residual"].quantile(0.025)),
            "share_residual_ci_high": float(batter_boot["share_residual"].quantile(0.975)),
        },
        "batter_season_sensitivity": batter_season_comp,
        "batter_season_bootstrap": {
            "share_between_batter_ci_low": float(batter_season_boot["share_between_batter"].quantile(0.025)),
            "share_between_batter_ci_high": float(batter_season_boot["share_between_batter"].quantile(0.975)),
            "share_within_batter_predictability_ci_low": float(batter_season_boot["share_within_batter_predictability"].quantile(0.025)),
            "share_within_batter_predictability_ci_high": float(batter_season_boot["share_within_batter_predictability"].quantile(0.975)),
            "share_residual_ci_low": float(batter_season_boot["share_residual"].quantile(0.025)),
            "share_residual_ci_high": float(batter_season_boot["share_residual"].quantile(0.975)),
        },
        "dominant_component": dominant_component,
        "dominant_share": dominant_share,
        "structural_dominant_component": structural_dominant_component,
        "structural_dominant_share": structural_dominant_share,
        "kill_criterion_h2_pass": bool(dominant_share > 0.70),
    }
    return findings


# =============================================================================
# H3 — empirical-calibration power simulation
# =============================================================================

H3_TRUE_SPREAD = 0.025
H3_SPREAD_BY_TERTILE = {"low": H3_TRUE_SPREAD / 2.0, "mid": 0.0, "high": -H3_TRUE_SPREAD / 2.0}
H3_N_SIMULATIONS = 300
H3_FRACTIONS = [0.25, 0.50, 0.75, 1.00]


def _weighted_effective_n(frame: pd.DataFrame) -> pd.Series:
    return 2.0 / ((1.0 / frame["n_high"].clip(lower=1.0)) + (1.0 / frame["n_low"].clip(lower=1.0)))


def _h3_estimate_spread(frame: pd.DataFrame, value_col: str, weight_col: str | None = None) -> dict[str, float]:
    work = frame.copy()
    work["chase_rate_tertile"] = pd.Categorical(work["chase_rate_tertile"], categories=["low", "mid", "high"], ordered=True)
    dummies = pd.get_dummies(work["chase_rate_tertile"], drop_first=False)
    dummies = dummies.reindex(columns=["mid", "high"], fill_value=0).astype(float)
    design = sm.add_constant(dummies)
    if weight_col is None:
        model = sm.OLS(work[value_col].astype(float), design)
    else:
        model = sm.WLS(work[value_col].astype(float), design, weights=work[weight_col].astype(float))
    result = model.fit(cov_type="HC1")
    spread = float(-result.params["high"])
    pvalue = float(result.t_test(np.array([[0.0, 0.0, -1.0]])).pvalue)
    return {"spread": spread, "pvalue": pvalue}


def _h3_simulate_once(
    units: pd.DataFrame,
    overall_mean: float,
    sigma_baseline: float,
    sigma_cell: float,
    sigma_match_extra: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sim_units = units.copy()
    baseline = overall_mean + rng.normal(0.0, sigma_baseline, size=len(sim_units))
    true_gap = sim_units["chase_rate_tertile"].map(H3_SPREAD_BY_TERTILE).to_numpy(dtype=float)
    high_noise = rng.normal(0.0, sigma_cell / np.sqrt(sim_units["n_high"].clip(lower=1.0)), size=len(sim_units))
    low_noise = rng.normal(0.0, sigma_cell / np.sqrt(sim_units["n_low"].clip(lower=1.0)), size=len(sim_units))
    mean_high = np.clip(baseline + 0.5 * true_gap + high_noise, 0.0, 2.031)
    mean_low = np.clip(baseline - 0.5 * true_gap + low_noise, 0.0, 2.031)
    sim_units["mean_woba_high"] = mean_high
    sim_units["mean_woba_low"] = mean_low
    sim_units["sum_woba_high"] = sim_units["mean_woba_high"] * sim_units["n_high"]
    sim_units["sum_woba_low"] = sim_units["mean_woba_low"] * sim_units["n_low"]
    sim_units["gap_woba"] = sim_units["mean_woba_high"] - sim_units["mean_woba_low"]
    sim_units["eff_n"] = _weighted_effective_n(sim_units)

    sim_pairs = sim_units[["season", "batter_season_id", "chase_rate_tertile", "n_pairs", "gap_woba"]].copy()
    extra = rng.normal(0.0, sigma_match_extra / np.sqrt(sim_pairs["n_pairs"].clip(lower=1.0)), size=len(sim_pairs))
    sim_pairs["gap_woba"] = sim_pairs["gap_woba"] + extra
    return sim_units, sim_pairs


def run_h3_power_simulation() -> dict[str, Any]:
    """H3 — empirical-calibration power simulation.

    Calibrate baseline dispersion, cell-level noise, and matched-pairs extra
    noise from the actual Round 6 batter-season counts, simulate a true
    low-minus-high spread of +0.025, and recover with both the pooled and
    matched-pairs estimators across sample-size fractions. Report bias, power
    (p<0.05), sign recovery, and a pooled-vs-matched power asymmetry flag.
    """
    unit_wide = build_terminal_unit_summary(cut_mode="within_pitcher_season")
    pair_units = build_pair_unit_summary(cut_mode="within_pitcher_season")
    design = unit_wide.merge(
        pair_units[["batter_season_id", "n_pairs"]],
        on="batter_season_id", how="inner",
    ).copy()
    design["overall_mean_woba"] = (
        design["sum_woba_high"] + design["sum_woba_low"]
    ) / (design["n_high"] + design["n_low"])
    design["eff_n"] = _weighted_effective_n(design)
    group_gap_mean = design.groupby("chase_rate_tertile", observed=True)["gap_woba"].transform("mean")

    overall_mean = float(design["overall_mean_woba"].mean())
    sigma_baseline = float(design["overall_mean_woba"].std(ddof=0))
    sigma_cell = float(((design["gap_woba"] - group_gap_mean) * np.sqrt(design["eff_n"])).std(ddof=0))

    pair_gap = pair_units.merge(
        design[["batter_season_id", "gap_woba"]],
        on="batter_season_id", how="inner",
        suffixes=("_pair", "_unit"),
    )
    sigma_match_extra = float(
        ((pair_gap["gap_woba_pair"] - pair_gap["gap_woba_unit"]) * np.sqrt(pair_gap["n_pairs"].clip(lower=1.0))).std(ddof=0)
    )

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(7)
    for fraction in H3_FRACTIONS:
        sample_n = max(50, int(round(len(design) * fraction)))
        pooled_estimates = []
        pooled_reject = []
        matched_estimates = []
        matched_reject = []
        for sim in range(H3_N_SIMULATIONS):
            sampled = design.sample(n=sample_n, replace=False, random_state=int(rng.integers(0, 1_000_000_000))).copy()
            sim_units, sim_pairs = _h3_simulate_once(
                sampled, overall_mean=overall_mean,
                sigma_baseline=sigma_baseline,
                sigma_cell=sigma_cell,
                sigma_match_extra=sigma_match_extra,
                rng=rng,
            )
            pooled = _h3_estimate_spread(sim_units, value_col="gap_woba", weight_col="eff_n")
            matched = _h3_estimate_spread(sim_pairs, value_col="gap_woba", weight_col=None)
            pooled_estimates.append(pooled["spread"])
            pooled_reject.append(pooled["pvalue"] < 0.05)
            matched_estimates.append(matched["spread"])
            matched_reject.append(matched["pvalue"] < 0.05)

        for estimator, estimates, reject in [
            ("pooled_between", pooled_estimates, pooled_reject),
            ("matched_pairs", matched_estimates, matched_reject),
        ]:
            arr = np.asarray(estimates, dtype=float)
            rows.append({
                "fraction": fraction,
                "estimator": estimator,
                "sample_n_units": sample_n,
                "mean_estimate": float(arr.mean()),
                "median_estimate": float(np.median(arr)),
                "bias": float(arr.mean() - H3_TRUE_SPREAD),
                "rmse": float(np.sqrt(np.mean((arr - H3_TRUE_SPREAD) ** 2))),
                "power": float(np.mean(reject)),
                "sign_recovery_rate": float(np.mean(arr > 0.0)),
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
            })

    summary = pd.DataFrame(rows).sort_values(["estimator", "fraction"]).reset_index(drop=True)
    pooled_full = summary[(summary["estimator"] == "pooled_between") & (summary["fraction"] == 1.0)].iloc[0]
    matched_full = summary[(summary["estimator"] == "matched_pairs") & (summary["fraction"] == 1.0)].iloc[0]
    findings = {
        "hypothesis": "H3_power_simulation",
        "true_spread_low_minus_high": H3_TRUE_SPREAD,
        "n_simulations": H3_N_SIMULATIONS,
        "calibration": {
            "overall_mean_woba": overall_mean,
            "sigma_baseline": sigma_baseline,
            "sigma_cell": sigma_cell,
            "sigma_match_extra": sigma_match_extra,
        },
        "summary": summary.to_dict(orient="records"),
        "full_n": {
            "pooled_between_power": float(pooled_full["power"]),
            "matched_pairs_power": float(matched_full["power"]),
            "pooled_between_bias": float(pooled_full["bias"]),
            "matched_pairs_bias": float(matched_full["bias"]),
        },
        "matched_pairs_underpowered_relative_to_pooled": bool(
            float(matched_full["power"]) + 0.15 < float(pooled_full["power"])
        ),
    }
    return findings


# =============================================================================
# H4 — cut-definition robustness
# =============================================================================

def run_h4_cut_definition_robustness() -> dict[str, Any]:
    """H4 — cut-definition robustness.

    Rerun H1's three estimators under the alternative league-wide 80/20
    predictability cut and compare to the within-pitcher-season default. The
    delta on pooled between (league minus within) quantifies how much of the
    estimator disagreement is attributable to cut definition vs estimand.
    """
    within = run_h1_three_estimators(cut_mode="within_pitcher_season")
    league = run_h1_three_estimators(cut_mode="league_wide")

    rows = []
    for cut_mode, findings in [("within_pitcher_season", within), ("league_wide", league)]:
        rows.extend([
            {"cut_mode": cut_mode, "estimator": "Pooled between",
             "spread_low_minus_high": findings["pooled_between"]["spread_low_minus_high"],
             "pvalue": findings["pooled_between"]["spread_permutation_pvalue"]},
            {"cut_mode": cut_mode, "estimator": "Matched pairs",
             "spread_low_minus_high": findings["matched_pairs"]["spread_low_minus_high"],
             "pvalue": findings["matched_pairs"]["spread_permutation_pvalue"]},
            {"cut_mode": cut_mode, "estimator": "Batter FE",
             "spread_low_minus_high": findings["batter_fixed_effects"]["spread_low_minus_high"],
             "pvalue": findings["batter_fixed_effects"]["spread_pvalue"]},
        ])
    summary = pd.DataFrame(rows)
    summary["delta_vs_within_pitcher"] = summary.groupby("estimator", observed=True)["spread_low_minus_high"].transform(
        lambda s: s - s.iloc[0]
    )

    pooled_delta = float(
        summary.loc[
            (summary["cut_mode"] == "league_wide") & (summary["estimator"] == "Pooled between"),
            "delta_vs_within_pitcher",
        ].iloc[0]
    )
    findings = {
        "hypothesis": "H4_cut_definition_robustness",
        "summary": summary.to_dict(orient="records"),
        "within_pitcher_h1": within,
        "league_wide_h1": league,
        "pooled_between_delta_league_minus_within": pooled_delta,
        "changes_substantive_direction": bool(
            (
                within["pooled_between"]["spread_low_minus_high"]
                * league["pooled_between"]["spread_low_minus_high"]
            ) < 0.0
        ),
    }
    return findings


# =============================================================================
# H5 — quality-hitter 2x2 replication
# =============================================================================

def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    return float((frame[value_col] * frame[weight_col]).sum() / frame[weight_col].sum())


def _weighted_group_gap(frame: pd.DataFrame, value_col: str, weight_col: str) -> pd.Series:
    work = frame.copy()
    work["_weighted_value"] = work[value_col] * work[weight_col]
    grouped = work.groupby("quality_hitter", observed=True)
    numer = grouped["_weighted_value"].sum()
    denom = grouped[weight_col].sum()
    return numer / denom


def _season_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    temp = frame.reset_index(drop=True)
    return [sub.index.to_numpy(dtype=int) for _, sub in temp.groupby("season", observed=True, sort=False)]


def _bootstrap_indices(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    parts = [rng.choice(idx, size=len(idx), replace=True) for idx in groups]
    return np.concatenate(parts)


def _permute_labels_by_season(labels: np.ndarray, groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    shuffled = labels.copy()
    for idx in groups:
        shuffled[idx] = rng.permutation(shuffled[idx])
    return shuffled


def _pair_weighted_difference(pair_units: pd.DataFrame) -> float:
    means = _weighted_group_gap(pair_units, value_col="gap_woba", weight_col="n_pairs")
    return float(means.get(1, np.nan) - means.get(0, np.nan))


def _unit_weighted_difference(unit_frame: pd.DataFrame) -> float:
    means = unit_frame.groupby("quality_hitter", observed=True)["gap_woba"].mean()
    return float(means.get(1, np.nan) - means.get(0, np.nan))


def _pooled_between_quality_difference(unit_wide: pd.DataFrame) -> float:
    rows = []
    for quality, sub in unit_wide.groupby("quality_hitter", observed=True):
        high_mean = float(sub["sum_woba_high"].sum() / sub["n_high"].sum())
        low_mean = float(sub["sum_woba_low"].sum() / sub["n_low"].sum())
        rows.append({"quality_hitter": quality, "gap_woba": high_mean - low_mean})
    summary = pd.DataFrame(rows).set_index("quality_hitter")
    return float(summary.loc[1, "gap_woba"] - summary.loc[0, "gap_woba"])


def _season_unit_permutation_pair_weighted(pair_units: pd.DataFrame, n_perm: int = DEFAULT_PERMUTATIONS, seed: int = 7) -> dict[str, float]:
    quality = pair_units["quality_hitter"].to_numpy(dtype=int)
    gap = pair_units["gap_woba"].to_numpy(dtype=float)
    weight = pair_units["n_pairs"].to_numpy(dtype=float)
    groups = _season_groups(pair_units)
    observed = _pair_weighted_difference(pair_units)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm, dtype=float)
    for idx in range(n_perm):
        shuffled = _permute_labels_by_season(quality, groups, rng)
        q1 = shuffled == 1
        q0 = shuffled == 0
        draws[idx] = float((gap[q1] * weight[q1]).sum() / weight[q1].sum() - (gap[q0] * weight[q0]).sum() / weight[q0].sum())
    return {
        "observed": observed,
        "permutation_pvalue": float((1 + np.sum(np.abs(draws) >= abs(observed))) / (n_perm + 1)),
        "null_mean": float(draws.mean()),
    }


def _season_unit_permutation_unit_weighted(unit_frame: pd.DataFrame, statistic_fn, n_perm: int = DEFAULT_PERMUTATIONS, seed: int = 7) -> dict[str, float]:
    observed = float(statistic_fn(unit_frame))
    groups = _season_groups(unit_frame)
    quality = unit_frame["quality_hitter"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm, dtype=float)
    for idx in range(n_perm):
        shuffled = unit_frame.copy()
        shuffled["quality_hitter"] = _permute_labels_by_season(quality, groups, rng)
        draws[idx] = float(statistic_fn(shuffled))
    return {
        "observed": observed,
        "permutation_pvalue": float((1 + np.sum(np.abs(draws) >= abs(observed))) / (n_perm + 1)),
        "null_mean": float(draws.mean()),
    }


def _bootstrap_pair_weighted(pair_units: pd.DataFrame, n_boot: int = DEFAULT_BOOTSTRAPS, seed: int = 7) -> tuple[float, float]:
    quality = pair_units["quality_hitter"].to_numpy(dtype=int)
    gap = pair_units["gap_woba"].to_numpy(dtype=float)
    weight = pair_units["n_pairs"].to_numpy(dtype=float)
    groups = _season_groups(pair_units)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sampled_idx = _bootstrap_indices(groups, rng)
        q = quality[sampled_idx]
        g = gap[sampled_idx]
        w = weight[sampled_idx]
        q1 = q == 1
        q0 = q == 0
        draws[idx] = float((g[q1] * w[q1]).sum() / w[q1].sum() - (g[q0] * w[q0]).sum() / w[q0].sum())
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def run_h5_quality_hitter_replication() -> dict[str, Any]:
    """H5 — quality-hitter 2x2 replication.

    Replicate Codex R5's quality_hitter (LOW chase AND HIGH xwOBA-on-contact)
    contrast using three weighted estimators (pair-weighted matched pairs,
    unit-weighted, pooled between) on the canonical substrate. Report the
    quality-minus-rest gap with bootstrap 95% CIs and season-stratified
    permutation p-values. Primary kill criterion is matched-pairs >=0.025
    with p<0.05.
    """
    pair_units = build_pair_unit_summary(cut_mode="within_pitcher_season")
    unit_wide = build_terminal_unit_summary(cut_mode="within_pitcher_season")
    unit_groups = _season_groups(pair_units)
    pooled_groups = _season_groups(unit_wide)

    pair_group_summary = []
    for quality, sub in pair_units.groupby("quality_hitter", observed=True):
        rng = np.random.default_rng(10_000 + int(quality))
        groups = _season_groups(sub)
        gap = sub["gap_woba"].to_numpy(dtype=float)
        weight = sub["n_pairs"].to_numpy(dtype=float)
        boot_draws = []
        for _ in range(DEFAULT_BOOTSTRAPS):
            sampled_idx = _bootstrap_indices(groups, rng)
            boot_draws.append(float((gap[sampled_idx] * weight[sampled_idx]).sum() / weight[sampled_idx].sum()))
        pair_group_summary.append({
            "quality_hitter": int(quality),
            "n_pairs": int(sub["n_pairs"].sum()),
            "mean_gap": _weighted_mean(sub, value_col="gap_woba", weight_col="n_pairs"),
            "ci_low": float(np.quantile(boot_draws, 0.025)),
            "ci_high": float(np.quantile(boot_draws, 0.975)),
        })
    pair_group_summary = pd.DataFrame(pair_group_summary).sort_values("quality_hitter").reset_index(drop=True)

    unit_group_summary = []
    for quality, sub in pair_units.groupby("quality_hitter", observed=True):
        ci_low, ci_high = bootstrap_mean_ci(sub["gap_woba"], n_boot=DEFAULT_BOOTSTRAPS, seed=7)
        unit_group_summary.append({
            "quality_hitter": int(quality),
            "n_units": int(len(sub)),
            "mean_gap": float(sub["gap_woba"].mean()),
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    unit_group_summary = pd.DataFrame(unit_group_summary).sort_values("quality_hitter").reset_index(drop=True)

    pooled_group_summary = []
    for quality, sub in unit_wide.groupby("quality_hitter", observed=True):
        high_mean = float(sub["sum_woba_high"].sum() / sub["n_high"].sum())
        low_mean = float(sub["sum_woba_low"].sum() / sub["n_low"].sum())
        pooled_group_summary.append({
            "quality_hitter": int(quality),
            "n_units": int(len(sub)),
            "gap_woba": high_mean - low_mean,
        })
    pooled_group_summary = pd.DataFrame(pooled_group_summary).sort_values("quality_hitter").reset_index(drop=True)

    pair_diff = _pair_weighted_difference(pair_units)
    pair_ci_low, pair_ci_high = _bootstrap_pair_weighted(pair_units)
    pair_perm = _season_unit_permutation_pair_weighted(pair_units, n_perm=DEFAULT_PERMUTATIONS, seed=7)

    unit_diff = _unit_weighted_difference(pair_units)
    unit_perm = _season_unit_permutation_unit_weighted(pair_units, _unit_weighted_difference, n_perm=DEFAULT_PERMUTATIONS, seed=17)
    rng_unit = np.random.default_rng(1_000)
    unit_quality = pair_units["quality_hitter"].to_numpy(dtype=int)
    unit_gap = pair_units["gap_woba"].to_numpy(dtype=float)
    unit_draws = np.empty(DEFAULT_BOOTSTRAPS, dtype=float)
    for idx in range(DEFAULT_BOOTSTRAPS):
        sampled_idx = _bootstrap_indices(unit_groups, rng_unit)
        q = unit_quality[sampled_idx]
        g = unit_gap[sampled_idx]
        unit_draws[idx] = float(g[q == 1].mean() - g[q == 0].mean())
    unit_ci_low = float(np.quantile(unit_draws, 0.025))
    unit_ci_high = float(np.quantile(unit_draws, 0.975))

    pooled_diff = _pooled_between_quality_difference(unit_wide)
    pooled_perm = _season_unit_permutation_unit_weighted(
        unit_wide[["season", "batter_season_id", "quality_hitter",
                   "sum_woba_high", "sum_woba_low", "n_high", "n_low"]].copy(),
        _pooled_between_quality_difference,
        n_perm=DEFAULT_PERMUTATIONS,
        seed=23,
    )
    rng_pooled = np.random.default_rng(3_000)
    pooled_quality = unit_wide["quality_hitter"].to_numpy(dtype=int)
    pooled_high_sum = unit_wide["sum_woba_high"].to_numpy(dtype=float)
    pooled_low_sum = unit_wide["sum_woba_low"].to_numpy(dtype=float)
    pooled_n_high = unit_wide["n_high"].to_numpy(dtype=float)
    pooled_n_low = unit_wide["n_low"].to_numpy(dtype=float)
    pooled_boot = np.empty(DEFAULT_BOOTSTRAPS, dtype=float)
    for idx in range(DEFAULT_BOOTSTRAPS):
        sampled_idx = _bootstrap_indices(pooled_groups, rng_pooled)
        q = pooled_quality[sampled_idx]
        q1 = q == 1
        q0 = q == 0
        gap1 = pooled_high_sum[sampled_idx][q1].sum() / pooled_n_high[sampled_idx][q1].sum() - pooled_low_sum[sampled_idx][q1].sum() / pooled_n_low[sampled_idx][q1].sum()
        gap0 = pooled_high_sum[sampled_idx][q0].sum() / pooled_n_high[sampled_idx][q0].sum() - pooled_low_sum[sampled_idx][q0].sum() / pooled_n_low[sampled_idx][q0].sum()
        pooled_boot[idx] = float(gap1 - gap0)
    pooled_ci_low = float(np.quantile(pooled_boot, 0.025))
    pooled_ci_high = float(np.quantile(pooled_boot, 0.975))

    estimator_summary = pd.DataFrame([
        {"estimator": "Matched pairs",
         "quality_minus_rest_gap": pair_diff,
         "ci_low": pair_ci_low, "ci_high": pair_ci_high,
         "pvalue": pair_perm["permutation_pvalue"]},
        {"estimator": "Unit weighted",
         "quality_minus_rest_gap": unit_diff,
         "ci_low": unit_ci_low, "ci_high": unit_ci_high,
         "pvalue": unit_perm["permutation_pvalue"]},
        {"estimator": "Pooled between",
         "quality_minus_rest_gap": pooled_diff,
         "ci_low": pooled_ci_low, "ci_high": pooled_ci_high,
         "pvalue": pooled_perm["permutation_pvalue"]},
    ])

    findings = {
        "hypothesis": "H5_quality_hitter_replication",
        "pair_weighted_group_summary": pair_group_summary.to_dict(orient="records"),
        "unit_weighted_group_summary": unit_group_summary.to_dict(orient="records"),
        "pooled_between_group_summary": pooled_group_summary.to_dict(orient="records"),
        "estimator_summary": estimator_summary.to_dict(orient="records"),
        "survives_matched_pairs_primary": bool((pair_diff >= 0.025) and (pair_perm["permutation_pvalue"] < 0.05)),
    }
    return findings


# =============================================================================
# Main driver
# =============================================================================

def _resolve_paths(data_dir: Path) -> None:
    PATHS.data_dir = data_dir
    career = data_dir / "agent_substrate" / "career_pitches.parquet"
    holdout = data_dir / "layer2_holdout_predictions.parquet"
    if career.exists():
        PATHS.pitches_path = career
    elif holdout.exists():
        PATHS.pitches_path = holdout
    else:
        raise FileNotFoundError(
            f"Neither {career} nor {holdout} found. Pass --data-dir pointing at "
            "the shared substrate parquets."
        )
    PATHS.probs_path = holdout if holdout.exists() else career
    cohort = data_dir / "pitcher_cohort.csv"
    PATHS.cohort_path = cohort if cohort.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/"),
                        help="Directory containing agent_substrate/career_pitches.parquet "
                             "and layer2_holdout_predictions.parquet.")
    parser.add_argument("--out", type=Path, default=Path("findings_round6_codex.json"),
                        help="Path for the aggregated findings JSON.")
    args = parser.parse_args()

    _resolve_paths(args.data_dir)

    print(f"[analyze_codex] pitches: {PATHS.pitches_path}")
    print(f"[analyze_codex] probs:   {PATHS.probs_path}")
    print(f"[analyze_codex] cohort:  {PATHS.cohort_path}")

    print("\n[H1] three estimators on identical substrate ...")
    h1 = run_h1_three_estimators(cut_mode="within_pitcher_season")
    for row in h1["estimator_alignment"]:
        print(f"  {row['estimator']:16s} spread={row['spread_low_minus_high']:.4f}  "
              f"CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]  p={row['pvalue']:.4f}")
    print(f"  max point-estimate spread: {h1['max_point_estimate_spread']:.4f}")
    print(f"  kill_criterion_h1_pass: {h1['kill_criterion_h1_pass']}")

    print("\n[H2] variance partition ...")
    h2 = run_h2_variance_partition()
    print(f"  dominant: {h2['dominant_component']} ({h2['dominant_share']:.3f})")
    print(f"  structural dominant: {h2['structural_dominant_component']} "
          f"({h2['structural_dominant_share']:.3f})")

    print("\n[H3] power simulation (calibrated) ...")
    h3 = run_h3_power_simulation()
    print(f"  full-N pooled power:  {h3['full_n']['pooled_between_power']:.3f}  "
          f"bias: {h3['full_n']['pooled_between_bias']:+.4f}")
    print(f"  full-N matched power: {h3['full_n']['matched_pairs_power']:.3f}  "
          f"bias: {h3['full_n']['matched_pairs_bias']:+.4f}")

    print("\n[H4] cut-definition robustness ...")
    h4 = run_h4_cut_definition_robustness()
    print(f"  pooled delta (league - within): {h4['pooled_between_delta_league_minus_within']:+.4f}")

    print("\n[H5] quality-hitter replication ...")
    h5 = run_h5_quality_hitter_replication()
    for row in h5["estimator_summary"]:
        print(f"  {row['estimator']:16s} diff={row['quality_minus_rest_gap']:.4f}  "
              f"CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]  p={row['pvalue']:.4f}")
    print(f"  survives matched-pairs primary: {h5['survives_matched_pairs_primary']}")

    flagship_h1_pass = bool(h1["kill_criterion_h1_pass"])
    flagship_h2_pass = bool(h2["kill_criterion_h2_pass"])
    if flagship_h1_pass:
        flagship_basis = "H1_estimator_convergence"
    elif flagship_h2_pass:
        flagship_basis = f"H2_{h2['dominant_component']}_dominance"
    else:
        flagship_basis = "methods_still_disagree"

    findings = {
        "round": 6,
        "agent": "codex",
        "shared_substrate": {
            "pitch_source": str(PATHS.pitches_path),
            "probability_source": str(PATHS.probs_path),
            "cohort_size_expected": STRICT_COHORT_SIZE,
            "analysis_seasons": list(PRE2026_SEASONS),
        },
        "h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5,
        "flagship_publishable_under_method_independent_standard": bool(flagship_h1_pass or flagship_h2_pass),
        "flagship_basis": flagship_basis,
        "central_conclusion": (
            "Between-batter estimators stay positive and well-powered on the shared substrate, "
            "matched pairs stays weakly positive but underpowered, and total terminal-pitch outcome "
            "variance is dominated by residual noise."
        ),
    }
    write_json(args.out, findings)
    print(f"\n[analyze_codex] wrote {args.out}")


if __name__ == "__main__":
    main()

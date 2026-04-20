"""CalledThird Coaching Gap — Round 6 Reconciliation Analyses (Claude agent).

This single-file script reproduces the Round 6 reconciliation analyses for the
CalledThird Coaching Gap study as run by the Claude agent. Round 6 pits three
estimators (pooled between-batter, matched-pairs within-batter, batter-FE
regression) against each other on an identical pre-registered substrate, plus
variance partition (H2), power simulation (H3), cut-definition robustness (H4),
and quality-hitter 2x2 replication (H5).

How to run:
    python analyze_claude.py --data-dir data/

Expected inputs under --data-dir:
    agent_substrate/career_pitches.parquet        # scored pitches 2022-2025
    layer2_holdout_predictions.parquet             # context-only Layer 2 probs
    pitcher_cohort.csv                             # optional strict 371-cohort

Dependencies: pandas, numpy, scipy, statsmodels, pyarrow.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sstats


# =============================================================================
# Common helpers (inlined from _common.py)
# =============================================================================

SEED = 7
N_BOOT = 400          # >=300 per brief
N_PERM = 1000         # per brief
PRE2026_SEASONS = (2022, 2023, 2024, 2025)
MIN_BATTER_SEASON_PITCHES = 150
STRICT_COHORT_SIZE = 371

CANONICAL_FEATURES = ["chase_rate", "whiff_rate_overall", "zone_contact_rate", "xwoba_con_proxy"]
TERTILE_ORDER = ["low", "mid", "high"]
HIGH, LOW = "high", "low"


class Paths:
    """Resolved paths to input parquets. Set by main()."""
    data_dir: Path
    pitches_path: Path
    probs_path: Path
    cohort_path: Path | None


PATHS = Paths()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def _default(v):
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (pd.Series, pd.Index)):
            return v.tolist()
        if isinstance(v, pd.DataFrame):
            return v.to_dict(orient="records")
        return v
    path.write_text(json.dumps(payload, indent=2, default=_default))


def load_strict_cohort_pitchers() -> set[int] | None:
    """Return the strict 371-pitcher cohort if a cohort CSV is provided.

    If the cohort file is absent, return None — callers will then take all
    pitchers present in the pitches file as the working cohort.
    """
    path = PATHS.cohort_path
    if path is None or not path.exists():
        return None
    cohort = pd.read_csv(path)
    if "pitcher" not in cohort.columns:
        return None
    cohort["pitcher"] = cohort["pitcher"].astype(int)
    return set(cohort["pitcher"].astype(int).tolist())


def load_context_predictions(columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Load the scored pitch substrate (career_pitches joined with Layer 2 probs).

    The public release consolidates Codex's context_all_predictions table into
    a single parquet at data/agent_substrate/career_pitches.parquet. Filter to
    pre-2026 seasons and normalize id dtypes.
    """
    frame = pd.read_parquet(PATHS.pitches_path, columns=list(columns) if columns else None)
    frame = frame[frame["season"].isin(PRE2026_SEASONS)].copy()
    if "pitcher" in frame.columns:
        frame["pitcher"] = frame["pitcher"].astype(int)
    if "batter" in frame.columns:
        frame["batter"] = pd.to_numeric(frame["batter"], errors="coerce").fillna(-1).astype(int)
    return frame


def _quantile_labels(values: pd.Series, q: int, labels: Sequence[str]) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=q, labels=labels).astype(str)


def add_quantile_labels(
    frame: pd.DataFrame,
    value_col: str,
    out_col: str,
    q: int,
    labels: Sequence[str],
    within_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    work = frame.copy()
    output = pd.Series(index=work.index, dtype=object)
    if within_cols:
        for _, idx in work.groupby(list(within_cols), observed=True).groups.items():
            sub = work.loc[idx, value_col]
            clean = sub.dropna()
            if clean.empty:
                output.loc[idx] = np.nan
                continue
            output.loc[clean.index] = _quantile_labels(clean, q=q, labels=labels)
            missing = sub.index.difference(clean.index)
            if len(missing):
                output.loc[missing] = np.nan
    else:
        clean = work[value_col].dropna()
        output.loc[clean.index] = _quantile_labels(clean, q=q, labels=labels)
        missing = work.index.difference(clean.index)
        if len(missing):
            output.loc[missing] = np.nan
    work[out_col] = output.astype("string")
    return work


def build_batter_season_features() -> pd.DataFrame:
    """Byte-match Codex's R5 batter season feature recipe (Appendix A)."""
    columns = ["batter", "season", "zone", "swung", "whiff", "events", "woba_value"]
    frame = load_context_predictions(columns=columns)

    in_zone = pd.to_numeric(frame["zone"], errors="coerce").le(9).fillna(False)
    out_zone = ~in_zone
    swung = frame["swung"].eq(1)
    whiff = frame["whiff"].eq(1)
    contact = swung & ~whiff
    zone_swing = in_zone & swung
    zone_contact = in_zone & contact
    chase_swing = out_zone & swung
    contact_event = frame["events"].notna() & ~frame["events"].isin(
        ["strikeout", "walk", "intent_walk", "hit_by_pitch"]
    )

    ff = pd.DataFrame({
        "batter": frame["batter"].to_numpy(),
        "season": frame["season"].to_numpy(),
        "n_pitches": np.ones(len(frame), dtype=int),
        "n_swings": swung.astype(int).to_numpy(),
        "out_zone_pitches": out_zone.astype(int).to_numpy(),
        "zone_swings": zone_swing.astype(int).to_numpy(),
        "zone_contacts": zone_contact.astype(int).to_numpy(),
        "chase_swings": chase_swing.astype(int).to_numpy(),
        "whiffs": whiff.astype(int).to_numpy(),
        "contact_events": contact_event.astype(int).to_numpy(),
        "contact_woba_num": np.where(contact_event, pd.to_numeric(frame["woba_value"], errors="coerce"), 0.0),
    })
    ff = ff.groupby(["batter", "season"], observed=True).sum().reset_index()
    ff = ff[ff["n_pitches"] >= MIN_BATTER_SEASON_PITCHES].copy()
    ff["chase_rate"] = ff["chase_swings"] / ff["out_zone_pitches"].replace(0, np.nan)
    ff["zone_contact_rate"] = ff["zone_contacts"] / ff["zone_swings"].replace(0, np.nan)
    ff["whiff_rate_overall"] = ff["whiffs"] / ff["n_swings"].replace(0, np.nan)
    ff["xwoba_con_proxy"] = ff["contact_woba_num"] / ff["contact_events"].replace(0, np.nan)

    for feature in CANONICAL_FEATURES:
        ff[feature] = ff[feature].fillna(ff[feature].median())
        ff = add_quantile_labels(ff, value_col=feature, out_col=f"{feature}_tertile_pooled",
                                 q=3, labels=TERTILE_ORDER)
    return ff


def load_terminal_pitches_with_probs() -> pd.DataFrame:
    """Terminal-pitch dataframe with context_prob, woba_value, count_state, stand.

    Restrict to strict cohort (if provided), pre-2026, terminal pitches only.
    """
    cols = ["pitcher", "batter", "season", "game_pk", "at_bat_number", "pitch_number",
            "count_state", "stand", "p_throws", "context_prob", "is_terminal", "woba_value",
            "events"]
    f = load_context_predictions(columns=cols)
    cohort = load_strict_cohort_pitchers()
    if cohort is not None:
        f = f[f["pitcher"].isin(cohort)].copy()
    f = f[f["is_terminal"].eq(1)].copy()
    f["woba_value"] = pd.to_numeric(f["woba_value"], errors="coerce").fillna(0.0)
    return f


def bootstrap_mean_ci(values, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def permutation_spread_pvalue(df: pd.DataFrame, group_col: str, value_col: str,
                              within_cols=("season",), n_perm: int = N_PERM,
                              seed: int = SEED) -> dict[str, float]:
    work = df[[*within_cols, group_col, value_col]].dropna().copy()
    def _spread(frame):
        m = frame.groupby(group_col, observed=True)[value_col].mean()
        return float(m.max() - m.min())
    observed = _spread(work)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm)
    for i in range(n_perm):
        parts = []
        for _, sub in work.groupby(list(within_cols), observed=True):
            t = sub.copy()
            t[group_col] = rng.permutation(t[group_col].to_numpy())
            parts.append(t)
        draws[i] = _spread(pd.concat(parts, ignore_index=True))
    p = float((1 + (draws >= observed).sum()) / (n_perm + 1))
    return {"observed": observed, "p": p, "null_mean": float(draws.mean())}


# =============================================================================
# H1 — three estimators on identical substrate
# =============================================================================

def assign_predictable_flag(pitches: pd.DataFrame) -> pd.DataFrame:
    """Top-quintile context_prob within pitcher-season = predictable=1; bottom
    quintile = predictable=0 (unpredictable baseline). Middle 60% dropped."""
    out = pitches.copy()
    g = out.groupby(["pitcher", "season"], observed=True)["context_prob"]
    upper = g.transform(lambda s: float(s.quantile(0.80)))
    lower = g.transform(lambda s: float(s.quantile(0.20)))
    out["predictable"] = np.where(out["context_prob"] >= upper, 1,
                          np.where(out["context_prob"] <= lower, 0, -1)).astype(int)
    return out[out["predictable"] >= 0].copy()


def attach_tertile(pitches: pd.DataFrame, features: pd.DataFrame, feature: str) -> pd.DataFrame:
    col = f"{feature}_tertile_pooled"
    sub = features[["batter", "season", col]].rename(columns={col: "tertile"})
    return pitches.merge(sub, on=["batter", "season"], how="inner")


def pooled_between(pitches: pd.DataFrame) -> dict[str, float]:
    rows = []
    for t in TERTILE_ORDER:
        sub = pitches[pitches["tertile"] == t]
        if sub.empty:
            continue
        mp = sub.loc[sub["predictable"] == 1, "woba_value"].mean()
        mu = sub.loc[sub["predictable"] == 0, "woba_value"].mean()
        rows.append({"tertile": t, "gap": float(mp - mu),
                     "n_pred": int((sub["predictable"] == 1).sum()),
                     "n_unpred": int((sub["predictable"] == 0).sum())})
    df = pd.DataFrame(rows)
    spread = float(df.set_index("tertile").loc[HIGH, "gap"] -
                   df.set_index("tertile").loc[LOW, "gap"])
    return {"tertile_gaps": df.to_dict(orient="records"), "spread_high_minus_low": spread}


def pooled_between_bootstrap(pitches: pd.DataFrame, n_boot: int = N_BOOT,
                             seed: int = SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    spreads = np.empty(n_boot)
    batters = pitches["batter"].unique()
    bat_to_idx = {b: pitches.index[pitches["batter"] == b].to_numpy() for b in batters}
    for i in range(n_boot):
        pick = rng.choice(batters, size=len(batters), replace=True)
        idx = np.concatenate([bat_to_idx[b] for b in pick])
        sub = pitches.loc[idx]
        gaps = {}
        for t in TERTILE_ORDER:
            s = sub[sub["tertile"] == t]
            mp = s.loc[s["predictable"] == 1, "woba_value"].mean()
            mu = s.loc[s["predictable"] == 0, "woba_value"].mean()
            gaps[t] = mp - mu
        spreads[i] = gaps.get(HIGH, np.nan) - gaps.get(LOW, np.nan)
    spreads = spreads[np.isfinite(spreads)]
    return {"ci_low": float(np.quantile(spreads, 0.025)),
            "ci_high": float(np.quantile(spreads, 0.975)),
            "boot_mean": float(spreads.mean())}


def pooled_between_permutation(pitches: pd.DataFrame, n_perm: int = N_PERM,
                               seed: int = SEED) -> float:
    bs = pitches[["batter", "season", "tertile"]].drop_duplicates().reset_index(drop=True)
    def _spread(bs_labels):
        merged = pitches[["batter", "season", "predictable", "woba_value"]].merge(
            bs_labels, on=["batter", "season"], how="inner")
        gaps = []
        for t in TERTILE_ORDER:
            s = merged[merged["tertile"] == t]
            gaps.append(s.loc[s["predictable"] == 1, "woba_value"].mean()
                        - s.loc[s["predictable"] == 0, "woba_value"].mean())
        return gaps[TERTILE_ORDER.index(HIGH)] - gaps[TERTILE_ORDER.index(LOW)]
    observed = _spread(bs)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm)
    for i in range(n_perm):
        parts = []
        for _, sub in bs.groupby("season"):
            t = sub.copy()
            t["tertile"] = rng.permutation(t["tertile"].to_numpy())
            parts.append(t)
        draws[i] = _spread(pd.concat(parts, ignore_index=True))
    return float((1 + (np.abs(draws) >= abs(observed)).sum()) / (n_perm + 1))


def matched_pairs_estimator(pitches: pd.DataFrame, seed: int = SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    group_cols = ["batter", "season", "count_state", "stand", "p_throws"]
    deltas = []
    for (batter, season, count, stand, p_throws), grp in pitches.groupby(group_cols, sort=False, observed=True):
        pred = grp[grp["predictable"] == 1]["woba_value"].to_numpy()
        unpr = grp[grp["predictable"] == 0]["woba_value"].to_numpy()
        if len(pred) == 0 or len(unpr) == 0:
            continue
        rng.shuffle(pred); rng.shuffle(unpr)
        n = min(len(pred), len(unpr))
        for j in range(n):
            deltas.append({"batter": batter, "season": season,
                           "delta": float(pred[j] - unpr[j])})
    pairs_df = pd.DataFrame(deltas)
    t = pitches[["batter", "season", "tertile"]].drop_duplicates()
    pairs_df = pairs_df.merge(t, on=["batter", "season"], how="inner")
    per_tertile = pairs_df.groupby("tertile")["delta"].agg(["mean", "count"]).reset_index()
    per_tertile_mean = per_tertile.set_index("tertile")["mean"].to_dict()
    spread = float(per_tertile_mean.get(HIGH, np.nan) - per_tertile_mean.get(LOW, np.nan))
    return {"pairs": pairs_df, "per_tertile": per_tertile.to_dict(orient="records"),
            "spread_high_minus_low": spread}


def matched_pairs_bootstrap(pairs_df: pd.DataFrame, n_boot: int = N_BOOT,
                            seed: int = SEED) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    batters = pairs_df["batter"].unique()
    bat_idx = {b: pairs_df.index[pairs_df["batter"] == b].to_numpy() for b in batters}
    spreads = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(batters, size=len(batters), replace=True)
        idx = np.concatenate([bat_idx[b] for b in pick])
        sub = pairs_df.loc[idx]
        m = sub.groupby("tertile")["delta"].mean()
        spreads[i] = m.get(HIGH, np.nan) - m.get(LOW, np.nan)
    spreads = spreads[np.isfinite(spreads)]
    return {"ci_low": float(np.quantile(spreads, 0.025)),
            "ci_high": float(np.quantile(spreads, 0.975)),
            "boot_mean": float(spreads.mean())}


def matched_pairs_permutation(pairs_df: pd.DataFrame, n_perm: int = N_PERM,
                              seed: int = SEED) -> float:
    def _spread(df):
        m = df.groupby("tertile")["delta"].mean()
        return m.get(HIGH, np.nan) - m.get(LOW, np.nan)
    observed = _spread(pairs_df)
    bs = pairs_df[["batter", "season", "tertile"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm)
    for i in range(n_perm):
        parts = []
        for _, sub in bs.groupby("season"):
            t = sub.copy()
            t["tertile"] = rng.permutation(t["tertile"].to_numpy())
            parts.append(t)
        shuf = pd.concat(parts, ignore_index=True)
        merged = pairs_df.drop(columns="tertile").merge(shuf, on=["batter", "season"], how="left")
        draws[i] = _spread(merged)
    return float((1 + (np.abs(draws) >= abs(observed)).sum()) / (n_perm + 1))


def batter_fe_regression(pitches: pd.DataFrame) -> dict[str, Any]:
    """Batter-season FE absorbed via demeaning; woba_dm on predictable_dm and
    its interaction with tertile dummies (LOW is the reference category)."""
    df = pitches[["batter", "season", "tertile", "predictable", "woba_value"]].copy()
    grp = df.groupby(["batter", "season"], observed=True)
    df["y_dm"] = df["woba_value"] - grp["woba_value"].transform("mean")
    df["x_dm"] = df["predictable"] - grp["predictable"].transform("mean")
    df["is_high"] = (df["tertile"] == HIGH).astype(float)
    df["is_mid"] = (df["tertile"] == "mid").astype(float)
    df["x_high"] = df["x_dm"] * df["is_high"]
    df["x_mid"] = df["x_dm"] * df["is_mid"]
    X = df[["x_dm", "x_high", "x_mid"]].to_numpy()
    y = df["y_dm"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    batters = df["batter"].to_numpy()
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((k, k))
    for b, idx in pd.Series(np.arange(n)).groupby(batters).groups.items():
        Xi = X[idx]; ui = resid[idx]
        s = Xi.T @ ui
        meat += np.outer(s, s)
    var_beta = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(var_beta))
    t_stats = beta / se
    pvals = 2 * (1 - sstats.norm.cdf(np.abs(t_stats)))
    coef = {
        "predictable_low (base)": {"beta": float(beta[0]), "se": float(se[0]), "p": float(pvals[0])},
        "predictable_x_high": {"beta": float(beta[1]), "se": float(se[1]), "p": float(pvals[1])},
        "predictable_x_mid": {"beta": float(beta[2]), "se": float(se[2]), "p": float(pvals[2])},
    }
    spread = float(beta[1])
    ci = (float(beta[1] - 1.96 * se[1]), float(beta[1] + 1.96 * se[1]))
    return {"coef": coef, "spread_high_minus_low": spread, "ci": ci, "p_spread": float(pvals[1]),
            "n_obs": int(n)}


def run_h1_three_estimators() -> dict[str, Any]:
    """H1 — three estimators on IDENTICAL substrate.

    For each canonical feature f (chase, whiff, zone-contact, xwoba_con_proxy),
    compute the predictability-gap chase-tertile spread (HIGH minus LOW) under
    three estimators (pooled between-batter, matched-pairs within-batter,
    batter-FE regression). Report point estimates, bootstrap 95% CI, and
    permutation p-values. The primary feature is chase_rate.
    """
    features = build_batter_season_features()
    pitches = load_terminal_pitches_with_probs()
    pitches = assign_predictable_flag(pitches)

    out: dict[str, Any] = {"estimator_definitions": {
        "1_pooled_between_batter": "mean(wOBA|pred) - mean(wOBA|unpred) per tertile; spread = HIGH - LOW.",
        "2_matched_pairs_within_batter": "within batter-season, pair pred vs unpred pitches on count_state/stand/p_throws.",
        "3_batter_fe_regression": "woba ~ pred + pred*tertile with batter-season FE via demeaning.",
    }, "per_feature": {}}

    for feature in CANONICAL_FEATURES:
        p = attach_tertile(pitches, features, feature)
        e1_point = pooled_between(p)
        e1_boot = pooled_between_bootstrap(p)
        e1_p = pooled_between_permutation(p)

        mp = matched_pairs_estimator(p)
        pairs_df = mp["pairs"]
        e2_boot = matched_pairs_bootstrap(pairs_df)
        e2_p = matched_pairs_permutation(pairs_df)

        e3 = batter_fe_regression(p)

        out["per_feature"][feature] = {
            "estimator_1_pooled_between": {
                "spread": e1_point["spread_high_minus_low"],
                "tertile_gaps": e1_point["tertile_gaps"],
                "ci": [e1_boot["ci_low"], e1_boot["ci_high"]],
                "p": e1_p,
            },
            "estimator_2_matched_pairs": {
                "spread": mp["spread_high_minus_low"],
                "tertile_deltas": mp["per_tertile"],
                "ci": [e2_boot["ci_low"], e2_boot["ci_high"]],
                "p": e2_p,
                "n_pairs": int(len(pairs_df)),
            },
            "estimator_3_batter_fe": {
                "spread": e3["spread_high_minus_low"],
                "ci": list(e3["ci"]),
                "p": e3["p_spread"],
                "coef": e3["coef"],
                "n_obs": e3["n_obs"],
            },
        }
    prim = out["per_feature"]["chase_rate"]
    spreads = [prim["estimator_1_pooled_between"]["spread"],
               prim["estimator_2_matched_pairs"]["spread"],
               prim["estimator_3_batter_fe"]["spread"]]
    out["chase_spread_range"] = {"min": float(min(spreads)), "max": float(max(spreads)),
                                 "range": float(max(spreads) - min(spreads)),
                                 "within_0.005": bool((max(spreads) - min(spreads)) <= 0.005)}
    ps = [prim["estimator_1_pooled_between"]["p"],
          prim["estimator_2_matched_pairs"]["p"],
          prim["estimator_3_batter_fe"]["p"]]
    out["chase_significance"] = {"p_values": ps,
                                 "n_sig_at_0.05": int(sum(1 for p in ps if p < 0.05))}
    return out


# =============================================================================
# H2 — variance partitioning
# =============================================================================

def _variance_components(pitches: pd.DataFrame) -> dict[str, float]:
    df = pitches[["batter", "season", "predictable", "woba_value"]].copy()
    grand = df["woba_value"].mean()
    bs_mean = df.groupby(["batter", "season"], observed=True)["woba_value"].transform("mean")
    cell_mean = df.groupby(["batter", "season", "predictable"], observed=True)["woba_value"].transform("mean")

    ss_total = ((df["woba_value"] - grand) ** 2).sum()
    ss_between = ((bs_mean - grand) ** 2).sum()
    ss_within_between_pred = ((cell_mean - bs_mean) ** 2).sum()
    ss_residual = ((df["woba_value"] - cell_mean) ** 2).sum()

    n = len(df)
    return {
        "n_pitches": int(n),
        "ss_total": float(ss_total),
        "ss_between_batter": float(ss_between),
        "ss_within_between_predictability": float(ss_within_between_pred),
        "ss_residual": float(ss_residual),
        "pct_between_batter": float(ss_between / ss_total),
        "pct_within_between_pred": float(ss_within_between_pred / ss_total),
        "pct_residual": float(ss_residual / ss_total),
    }


def _gap_variance_decomposition(pitches: pd.DataFrame) -> dict[str, float]:
    df = pitches[["batter", "season", "predictable", "woba_value"]].copy()
    rows = []
    for (b, s), g in df.groupby(["batter", "season"], observed=True):
        p = g[g["predictable"] == 1]["woba_value"]
        u = g[g["predictable"] == 0]["woba_value"]
        if len(p) < 1 or len(u) < 1:
            continue
        rows.append({"batter": b, "season": s,
                     "gap": float(p.mean() - u.mean()),
                     "var_p": float(p.var(ddof=1)) if len(p) > 1 else 0.0,
                     "var_u": float(u.var(ddof=1)) if len(u) > 1 else 0.0,
                     "n_p": int(len(p)), "n_u": int(len(u)),
                     "sampling_var": float((p.var(ddof=1) if len(p) > 1 else 0.0) / max(len(p), 1)
                                          + (u.var(ddof=1) if len(u) > 1 else 0.0) / max(len(u), 1)),
                     })
    g = pd.DataFrame(rows)
    total_var = float(g["gap"].var(ddof=1))
    mean_sampling = float(g["sampling_var"].mean())
    signal_var = max(0.0, total_var - mean_sampling)
    return {
        "n_batter_seasons": int(len(g)),
        "total_var_gap": total_var,
        "mean_sampling_var": mean_sampling,
        "signal_var_between_batter_of_gap": signal_var,
        "pct_signal_of_total": float(signal_var / total_var) if total_var > 0 else float("nan"),
        "pct_noise_of_total": float(mean_sampling / total_var) if total_var > 0 else float("nan"),
        "mean_gap": float(g["gap"].mean()),
        "sd_gap": float(g["gap"].std(ddof=1)),
    }


def _per_tertile_gap_variance(pitches: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for t in TERTILE_ORDER:
        sub = pitches[pitches["tertile"] == t]
        if sub.empty:
            continue
        out[t] = _gap_variance_decomposition(sub)
    return out


def run_h2_variance_partition() -> dict[str, Any]:
    """H2 — variance partitioning of the pooled coaching gap.

    Partition pitch-level wOBA variance into between-batter, within-batter
    predictability, and residual components. Also decompose the per-batter GAP
    variance into between-batter signal vs sampling noise. Emits a dominance
    verdict on which component carries the signal.
    """
    features = build_batter_season_features()
    pitches = load_terminal_pitches_with_probs()
    pitches = assign_predictable_flag(pitches)

    result = {"overall_pitch_anova": _variance_components(pitches),
              "overall_gap_decomposition": _gap_variance_decomposition(pitches)}

    for feature in CANONICAL_FEATURES:
        p = attach_tertile(pitches, features, feature)
        result[f"{feature}_per_tertile_gap"] = _per_tertile_gap_variance(p)

    gap_dec = result["overall_gap_decomposition"]
    pct = gap_dec["pct_signal_of_total"]
    if pct >= 0.70:
        result["dominance_verdict"] = "between-batter dominant (>=70% signal)"
    elif pct <= 0.30:
        result["dominance_verdict"] = "within-batter/noise dominant (<=30% signal)"
    else:
        result["dominance_verdict"] = f"mixed ({pct*100:.1f}% between-batter signal)"
    return result


# =============================================================================
# H3 — simulation-based power calibration
# =============================================================================

BASE_MU = 0.32
BASE_GAP = 0.06
TRUE_SPREAD = -0.025
TERTILE_EFFECTS = {"low": 0.0125, "mid": 0.0, "high": -0.0125}  # HIGH - LOW = -0.025
SIGMA_BATTER = 0.06
SIGMA_ETA = 0.03
SIGMA_PITCH = 0.55
N_PRED = 40
N_UNPRED = 40


def _simulate(B: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mu = rng.normal(BASE_MU, SIGMA_BATTER, size=B)
    eta = rng.normal(0.0, SIGMA_ETA, size=B)
    tertile = rng.choice(["low", "mid", "high"], size=B)
    rows = []
    for i in range(B):
        gap_i = BASE_GAP + TERTILE_EFFECTS[tertile[i]] + eta[i]
        pw = rng.normal(mu[i] + gap_i, SIGMA_PITCH, size=N_PRED)
        uw = rng.normal(mu[i], SIGMA_PITCH, size=N_UNPRED)
        for w in pw:
            rows.append({"batter": i, "tertile": tertile[i], "predictable": 1, "woba_value": w})
        for w in uw:
            rows.append({"batter": i, "tertile": tertile[i], "predictable": 0, "woba_value": w})
    return pd.DataFrame(rows)


def _est_pooled(df: pd.DataFrame) -> float:
    gaps = {}
    for t in ["low", "high"]:
        s = df[df["tertile"] == t]
        gaps[t] = s.loc[s["predictable"] == 1, "woba_value"].mean() - s.loc[s["predictable"] == 0, "woba_value"].mean()
    return gaps["high"] - gaps["low"]


def _est_matched(df: pd.DataFrame, rng) -> float:
    deltas = []
    for b, grp in df.groupby("batter", sort=False):
        pred = grp[grp["predictable"] == 1]["woba_value"].to_numpy()
        unpr = grp[grp["predictable"] == 0]["woba_value"].to_numpy()
        rng.shuffle(pred); rng.shuffle(unpr)
        n = min(len(pred), len(unpr))
        for j in range(n):
            deltas.append({"batter": b, "delta": pred[j] - unpr[j],
                           "tertile": grp["tertile"].iloc[0]})
    pd_ = pd.DataFrame(deltas)
    m = pd_.groupby("tertile")["delta"].mean()
    return float(m["high"] - m["low"])


def _est_batter_fe(df: pd.DataFrame) -> tuple[float, float]:
    d = df.copy()
    bmean_y = d.groupby("batter")["woba_value"].transform("mean")
    bmean_p = d.groupby("batter")["predictable"].transform("mean")
    d["y_dm"] = d["woba_value"] - bmean_y
    d["x_dm"] = d["predictable"] - bmean_p
    d["is_high"] = (d["tertile"] == "high").astype(float)
    d["is_mid"] = (d["tertile"] == "mid").astype(float)
    d["x_high"] = d["x_dm"] * d["is_high"]
    d["x_mid"] = d["x_dm"] * d["is_mid"]
    X = d[["x_dm", "x_high", "x_mid"]].to_numpy()
    y = d["y_dm"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = len(d)
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros_like(XtX_inv)
    batters = d["batter"].to_numpy()
    for b, idx in pd.Series(np.arange(n)).groupby(batters).groups.items():
        Xi = X[idx]; ui = resid[idx]
        s = Xi.T @ ui
        meat += np.outer(s, s)
    var = XtX_inv @ meat @ XtX_inv
    return float(beta[1]), float(np.sqrt(var[1, 1]))


def _run_grid(B_list=(100, 200, 400, 800, 1600), n_reps: int = 200, seed: int = SEED) -> pd.DataFrame:
    rng_master = np.random.default_rng(seed)
    rows = []
    for B in B_list:
        est1_vals = np.empty(n_reps); est2_vals = np.empty(n_reps); est3_vals = np.empty(n_reps)
        est3_ps = np.empty(n_reps)
        for r in range(n_reps):
            sub_seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(sub_seed)
            df = _simulate(B, sub_seed)
            est1_vals[r] = _est_pooled(df)
            est2_vals[r] = _est_matched(df, rng)
            b_hat, b_se = _est_batter_fe(df)
            est3_vals[r] = b_hat
            est3_ps[r] = 2 * (1 - 0.5 * (1 + math.erf(abs(b_hat / b_se) / np.sqrt(2))))

        def pwr(vals: np.ndarray, p_from_z: np.ndarray | None = None):
            if p_from_z is not None:
                return float((p_from_z < 0.05).mean())
            se_rep = vals.std(ddof=1)
            z = vals / se_rep
            return float((np.abs(z) > 1.96).mean())

        rows.append({"B": B,
                     "est1_mean": float(est1_vals.mean()), "est1_bias": float(est1_vals.mean() - TRUE_SPREAD),
                     "est1_sd": float(est1_vals.std(ddof=1)), "est1_power": pwr(est1_vals),
                     "est2_mean": float(est2_vals.mean()), "est2_bias": float(est2_vals.mean() - TRUE_SPREAD),
                     "est2_sd": float(est2_vals.std(ddof=1)), "est2_power": pwr(est2_vals),
                     "est3_mean": float(est3_vals.mean()), "est3_bias": float(est3_vals.mean() - TRUE_SPREAD),
                     "est3_sd": float(est3_vals.std(ddof=1)), "est3_power": pwr(est3_vals, p_from_z=est3_ps),
                     })
    return pd.DataFrame(rows)


def run_h3_power_simulation() -> dict[str, Any]:
    """H3 — simulation-based power calibration.

    Generate synthetic data with a known true between-batter spread of -0.025
    across chase tertiles, run all three H1 estimators across a grid of
    batter counts, and report bias, sd, and power. Identifies the minimum N
    for 80% power per estimator.
    """
    grid = _run_grid(B_list=(100, 200, 400, 800, 1600), n_reps=150, seed=SEED)

    def min_N(col):
        hits = grid[grid[col] >= 0.80]
        return int(hits["B"].min()) if not hits.empty else None
    summary = {
        "true_spread": TRUE_SPREAD,
        "n_reps_per_B": 150,
        "grid": grid.to_dict(orient="records"),
        "min_B_for_80pct_power": {
            "est1_pooled_between": min_N("est1_power"),
            "est2_matched_pairs": min_N("est2_power"),
            "est3_batter_fe": min_N("est3_power"),
        },
        "observed_bias_at_B=400": {
            "est1": float(grid.loc[grid["B"] == 400, "est1_bias"].iloc[0]),
            "est2": float(grid.loc[grid["B"] == 400, "est2_bias"].iloc[0]),
            "est3": float(grid.loc[grid["B"] == 400, "est3_bias"].iloc[0]),
        },
    }
    p1 = grid.loc[grid["B"] == 400, "est1_power"].iloc[0]
    p2 = grid.loc[grid["B"] == 400, "est2_power"].iloc[0]
    p3 = grid.loc[grid["B"] == 400, "est3_power"].iloc[0]
    summary["power_at_B=400"] = {"est1": float(p1), "est2": float(p2), "est3": float(p3)}
    summary["interpretation"] = (
        "If est2_power << est1_power at B=400, matched-pairs has lost power relative "
        "to pooled between-batter under the same DGP."
    )
    return summary


# =============================================================================
# H4 — cut-definition robustness
# =============================================================================

def _assign_league_wide(pitches: pd.DataFrame) -> pd.DataFrame:
    out = pitches.copy()
    up = out["context_prob"].quantile(0.80)
    lo = out["context_prob"].quantile(0.20)
    out["predictable"] = np.where(out["context_prob"] >= up, 1,
                                  np.where(out["context_prob"] <= lo, 0, -1)).astype(int)
    return out[out["predictable"] >= 0].copy()


def run_h4_cut_definition_robustness() -> dict[str, Any]:
    """H4 — cut-definition robustness for the predictable-pitch flag.

    Compare the chase-tertile spread under (a) within-pitcher-season 80/20
    quantile cut (H1 default) and (b) league-wide pooled 80/20 cut. Reports
    both pooled-between and matched-pairs spreads with bootstrap 95% CIs.
    """
    features = build_batter_season_features()
    raw = load_terminal_pitches_with_probs()

    out = {}
    for label, prep in [("within_pitcher_80_20", assign_predictable_flag),
                        ("league_wide_80_20", _assign_league_wide)]:
        pitches = prep(raw)
        p = attach_tertile(pitches, features, "chase_rate")
        pooled_pt = pooled_between(p)
        pooled_ci = pooled_between_bootstrap(p)
        pooled_p = pooled_between_permutation(p)
        mp = matched_pairs_estimator(p)
        mp_ci = matched_pairs_bootstrap(mp["pairs"])
        mp_p = matched_pairs_permutation(mp["pairs"])
        out[label] = {
            "n_pitches_used": int(len(p)),
            "pooled_between": {
                "spread": pooled_pt["spread_high_minus_low"],
                "tertile_gaps": pooled_pt["tertile_gaps"],
                "ci": [pooled_ci["ci_low"], pooled_ci["ci_high"]],
                "p": pooled_p,
            },
            "matched_pairs": {
                "spread": mp["spread_high_minus_low"],
                "ci": [mp_ci["ci_low"], mp_ci["ci_high"]],
                "p": mp_p,
                "n_pairs": int(len(mp["pairs"])),
            },
        }

    a = out["within_pitcher_80_20"]["pooled_between"]["spread"]
    b = out["league_wide_80_20"]["pooled_between"]["spread"]
    out["pooled_spread_diff"] = float(a - b)
    out["directional_agreement"] = bool(np.sign(a) == np.sign(b))
    return out


# =============================================================================
# H5 — quality-hitter 2x2 replication
# =============================================================================

def _attach_quality(pitches: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    tmp = features[["batter", "season",
                    "chase_rate_tertile_pooled", "xwoba_con_proxy_tertile_pooled"]].copy()
    tmp["quality"] = ((tmp["chase_rate_tertile_pooled"] == "low") &
                      (tmp["xwoba_con_proxy_tertile_pooled"] == "high")).astype(int)
    return pitches.merge(tmp[["batter", "season", "quality"]], on=["batter", "season"], how="inner")


def _h5_pooled_diff(p: pd.DataFrame, seed: int = SEED, n_boot: int = N_BOOT) -> dict[str, Any]:
    gaps = {}
    for q in [0, 1]:
        s = p[p["quality"] == q]
        gaps[q] = s.loc[s["predictable"] == 1, "woba_value"].mean() - s.loc[s["predictable"] == 0, "woba_value"].mean()
    observed = float(gaps[1] - gaps[0])

    rng = np.random.default_rng(seed)
    batters = p["batter"].unique()
    bat_idx = {b: p.index[p["batter"] == b].to_numpy() for b in batters}
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(batters, size=len(batters), replace=True)
        idx = np.concatenate([bat_idx[b] for b in pick])
        sub = p.loc[idx]
        g = {}
        for q in [0, 1]:
            s = sub[sub["quality"] == q]
            g[q] = s.loc[s["predictable"] == 1, "woba_value"].mean() - s.loc[s["predictable"] == 0, "woba_value"].mean()
        draws[i] = g[1] - g[0]
    draws = draws[np.isfinite(draws)]
    return {"observed": observed, "ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "n_quality_pitches": int((p["quality"] == 1).sum()),
            "n_nonquality_pitches": int((p["quality"] == 0).sum()),
            "gap_quality": float(gaps[1]), "gap_nonquality": float(gaps[0])}


def _h5_matched_pairs_diff(p: pd.DataFrame, seed: int = SEED, n_boot: int = N_BOOT) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    group_cols = ["batter", "season", "count_state", "stand", "p_throws"]
    rows = []
    for (b, s, c, st, pt), grp in p.groupby(group_cols, sort=False, observed=True):
        pr = grp[grp["predictable"] == 1]["woba_value"].to_numpy()
        un = grp[grp["predictable"] == 0]["woba_value"].to_numpy()
        if len(pr) == 0 or len(un) == 0:
            continue
        rng.shuffle(pr); rng.shuffle(un)
        n = min(len(pr), len(un))
        for j in range(n):
            rows.append({"batter": b, "season": s, "delta": float(pr[j] - un[j])})
    pairs = pd.DataFrame(rows)
    q = p[["batter", "season", "quality"]].drop_duplicates()
    pairs = pairs.merge(q, on=["batter", "season"], how="inner")
    observed = float(pairs.loc[pairs["quality"] == 1, "delta"].mean()
                     - pairs.loc[pairs["quality"] == 0, "delta"].mean())

    rng = np.random.default_rng(seed + 1)
    batters = pairs["batter"].unique()
    bat_idx = {b: pairs.index[pairs["batter"] == b].to_numpy() for b in batters}
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(batters, size=len(batters), replace=True)
        idx = np.concatenate([bat_idx[b] for b in pick])
        sub = pairs.loc[idx]
        draws[i] = sub.loc[sub["quality"] == 1, "delta"].mean() - sub.loc[sub["quality"] == 0, "delta"].mean()
    draws = draws[np.isfinite(draws)]

    bs = pairs[["batter", "season", "quality"]].drop_duplicates().reset_index(drop=True)
    rng_p = np.random.default_rng(seed + 2)
    p_draws = np.empty(N_PERM)
    for i in range(N_PERM):
        parts = []
        for _, sub in bs.groupby("season"):
            t = sub.copy()
            t["quality"] = rng_p.permutation(t["quality"].to_numpy())
            parts.append(t)
        shuf = pd.concat(parts, ignore_index=True)
        merged = pairs.drop(columns="quality").merge(shuf, on=["batter", "season"], how="left")
        p_draws[i] = merged.loc[merged["quality"] == 1, "delta"].mean() - merged.loc[merged["quality"] == 0, "delta"].mean()
    p_two = float((1 + (np.abs(p_draws) >= abs(observed)).sum()) / (N_PERM + 1))

    return {"observed": observed, "ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "n_pairs": int(len(pairs)),
            "n_quality_batter_seasons": int(pairs[pairs["quality"] == 1]["batter"].nunique()),
            "p_permutation": p_two}


def _h5_batter_fe_diff(p: pd.DataFrame) -> dict[str, Any]:
    d = p[["batter", "season", "quality", "predictable", "woba_value"]].copy()
    grp = d.groupby(["batter", "season"], observed=True)
    d["y_dm"] = d["woba_value"] - grp["woba_value"].transform("mean")
    d["x_dm"] = d["predictable"] - grp["predictable"].transform("mean")
    d["x_q"] = d["x_dm"] * d["quality"]
    X = d[["x_dm", "x_q"]].to_numpy()
    y = d["y_dm"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = len(d)
    XtX_inv = np.linalg.inv(X.T @ X)
    batters = d["batter"].to_numpy()
    meat = np.zeros_like(XtX_inv)
    for b, idx in pd.Series(np.arange(n)).groupby(batters).groups.items():
        Xi = X[idx]; ui = resid[idx]
        s = Xi.T @ ui
        meat += np.outer(s, s)
    var = XtX_inv @ meat @ XtX_inv
    beta_q = float(beta[1]); se_q = float(np.sqrt(var[1, 1]))
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(beta_q / se_q) / np.sqrt(2))))
    return {"observed": beta_q, "se": se_q,
            "ci": [beta_q - 1.96 * se_q, beta_q + 1.96 * se_q],
            "p": float(p_val)}


def run_h5_quality_hitter_replication() -> dict[str, Any]:
    """H5 — quality-hitter 2x2 replication at matched N.

    Definition: quality_hitter = (chase_rate LOW tertile) AND
    (xwoba_con_proxy HIGH tertile). Tests the predictability gap for quality
    hitters vs the rest using all three estimators (pooled, matched-pairs,
    batter-FE) on the identical substrate.
    """
    features = build_batter_season_features()
    pitches = load_terminal_pitches_with_probs()
    pitches = assign_predictable_flag(pitches)
    p = _attach_quality(pitches, features)

    e1 = _h5_pooled_diff(p)
    e2 = _h5_matched_pairs_diff(p)
    e3 = _h5_batter_fe_diff(p)

    out = {
        "definition": "quality_hitter = (chase_rate LOW tertile) AND (xwoba_con_proxy HIGH tertile)",
        "contrast": "quality_hitter gap MINUS non-quality gap",
        "estimator_1_pooled_between": e1,
        "estimator_2_matched_pairs": e2,
        "estimator_3_batter_fe": e3,
        "codex_r5_matched_pairs_claim": {"spread": 0.025, "p": 0.004},
    }
    spreads = [e1["observed"], e2["observed"], e3["observed"]]
    out["three_estimator_range"] = {"min": float(min(spreads)), "max": float(max(spreads)),
                                     "range": float(max(spreads) - min(spreads)),
                                     "within_0.005": bool((max(spreads) - min(spreads)) <= 0.005)}
    return out


# =============================================================================
# Main driver
# =============================================================================

def _resolve_paths(data_dir: Path) -> None:
    PATHS.data_dir = data_dir
    # The public release layout: scored pitches at agent_substrate/ and Layer 2
    # predictions one level up. Both files carry the same pitch_id grain; we
    # read from career_pitches.parquet which is the merged scored substrate.
    career = data_dir / "agent_substrate" / "career_pitches.parquet"
    holdout = data_dir / "layer2_holdout_predictions.parquet"
    # Prefer the merged scored parquet if present; otherwise fall back.
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
    parser.add_argument("--out", type=Path, default=Path("findings_round6_claude.json"),
                        help="Path for the aggregated findings JSON.")
    args = parser.parse_args()

    _resolve_paths(args.data_dir)

    print(f"[analyze_claude] pitches: {PATHS.pitches_path}")
    print(f"[analyze_claude] probs:   {PATHS.probs_path}")
    print(f"[analyze_claude] cohort:  {PATHS.cohort_path}")

    print("\n[H1] three estimators on identical substrate ...")
    h1 = run_h1_three_estimators()
    prim = h1["per_feature"]["chase_rate"]
    for name, sub in prim.items():
        if name.startswith("estimator"):
            print(f"  {name}: spread={sub['spread']:.4f}  CI={sub['ci']}  p={sub['p']:.4f}")

    print("\n[H2] variance partition ...")
    h2 = run_h2_variance_partition()
    print(f"  verdict: {h2['dominance_verdict']}")
    print(f"  pitch ANOVA pct_residual={h2['overall_pitch_anova']['pct_residual']:.3f}")

    print("\n[H3] power simulation ...")
    h3 = run_h3_power_simulation()
    print(f"  power @ B=400: {h3['power_at_B=400']}")
    print(f"  min B for 80% power: {h3['min_B_for_80pct_power']}")

    print("\n[H4] cut-definition robustness ...")
    h4 = run_h4_cut_definition_robustness()
    print(f"  within-pitcher spread: {h4['within_pitcher_80_20']['pooled_between']['spread']:.4f}")
    print(f"  league-wide spread:    {h4['league_wide_80_20']['pooled_between']['spread']:.4f}")

    print("\n[H5] quality-hitter 2x2 replication ...")
    h5 = run_h5_quality_hitter_replication()
    print(f"  pooled  diff: {h5['estimator_1_pooled_between']['observed']:.4f}")
    print(f"  matched diff: {h5['estimator_2_matched_pairs']['observed']:.4f}  p={h5['estimator_2_matched_pairs']['p_permutation']:.4f}")
    print(f"  FE      diff: {h5['estimator_3_batter_fe']['observed']:.4f}  p={h5['estimator_3_batter_fe']['p']:.4f}")

    findings = {"round": 6, "agent": "claude",
                "h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5}
    write_json(args.out, findings)
    print(f"\n[analyze_claude] wrote {args.out}")


if __name__ == "__main__":
    main()

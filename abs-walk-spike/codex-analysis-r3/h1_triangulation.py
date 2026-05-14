from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

from data_prep_r3 import (
    ARTIFACTS_DIR,
    BALL_LIKE_DESCRIPTIONS,
    BUNT_FOUL_DESCRIPTIONS,
    CALLED_DESCRIPTIONS,
    CHARTS_DIR,
    COUNT_ORDER,
    DIAG_DIR,
    FOUL_DESCRIPTIONS,
    FOUL_TIP_DESCRIPTIONS,
    GLOBAL_SEED,
    IN_PLAY_DESCRIPTIONS,
    STRIKE_LIKE_DESCRIPTIONS,
    WALK_EVENTS,
    Round3Data,
    called_pitch_rows,
    encode_features,
    grouped_oof_predictions,
    lgbm_classifier,
    percentile_ci,
    row_weights_from_games,
    sample_game_weights,
    save_json,
    terminal_pa_rows,
    walk_rate_from_terminal,
    weighted_mean,
    write_classifier_diagnostics,
)


ZONE_NUMERIC = ["plate_x", "plate_z", "balls", "strikes"]
ZONE_CATEGORICAL = ["count_state", "pitch_type"]


@dataclass
class PitchStep:
    description: str
    event: str | None
    called_idx: int


@dataclass
class PASequence:
    pa_id: str
    game_pk: int
    actual_walk: int
    terminal_count: str
    steps: tuple[PitchStep, ...]


@dataclass
class H1Result:
    summary: dict
    methods: pd.DataFrame
    method_a_bootstrap: pd.DataFrame
    method_b_bootstrap: pd.DataFrame
    method_c_outer: pd.DataFrame
    per_count_method_c: pd.DataFrame
    per_edge_method_c: pd.DataFrame
    diagnostics: dict


def _build_sequences(df: pd.DataFrame) -> tuple[list[PASequence], pd.DataFrame]:
    ordered = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True).copy()
    terminal = terminal_pa_rows(ordered).set_index("pa_id")
    ordered = ordered[ordered["pa_id"].isin(terminal.index)].copy()
    ordered["called_idx"] = -1
    called_mask = ordered["description"].isin(CALLED_DESCRIPTIONS) & ordered[["plate_x", "plate_z"]].notna().all(axis=1)
    ordered.loc[called_mask, "called_idx"] = np.arange(int(called_mask.sum()))

    sequences: list[PASequence] = []
    for pa_id, group in ordered.groupby("pa_id", sort=False):
        terminal_row = terminal.loc[pa_id]
        steps = tuple(
            PitchStep(
                description=str(row.description),
                event=None if pd.isna(row.events) or row.events == "" else str(row.events),
                called_idx=int(row.called_idx),
            )
            for row in group.itertuples()
        )
        sequences.append(
            PASequence(
                pa_id=str(pa_id),
                game_pk=int(terminal_row.game_pk),
                actual_walk=int(terminal_row.walk_event),
                terminal_count=str(terminal_row.terminal_count),
                steps=steps,
            )
        )
    called_frame = ordered.loc[called_mask].copy().reset_index(drop=True)
    return sequences, called_frame


def _continuation_probs(df_2026: pd.DataFrame) -> dict[str, float]:
    terminal = terminal_pa_rows(df_2026)[["pa_id", "walk_event"]]
    reached = df_2026[["pa_id", "count_state"]].drop_duplicates()
    merged = reached.merge(terminal, on="pa_id", how="left")
    probs = merged.groupby("count_state", observed=True)["walk_event"].mean().to_dict()
    overall = float(terminal["walk_event"].mean())
    for count in COUNT_ORDER:
        probs.setdefault(count, overall)
    return probs


def _advance_non_called(description: str, event: str | None, balls: int, strikes: int) -> tuple[bool, float, int, int]:
    if event in WALK_EVENTS:
        return True, 1.0, balls, strikes
    if description in IN_PLAY_DESCRIPTIONS or description == "hit_by_pitch":
        return True, 0.0, balls, strikes
    if description in BALL_LIKE_DESCRIPTIONS:
        if balls == 3:
            return True, 1.0, 4, strikes
        return False, 0.0, balls + 1, strikes
    if description in STRIKE_LIKE_DESCRIPTIONS:
        if strikes == 2:
            return True, 0.0, balls, 3
        return False, 0.0, balls, strikes + 1
    if description in FOUL_TIP_DESCRIPTIONS or description in BUNT_FOUL_DESCRIPTIONS:
        if strikes == 2:
            return True, 0.0, balls, 3
        return False, 0.0, balls, strikes + 1
    if description in FOUL_DESCRIPTIONS:
        if strikes < 2:
            return False, 0.0, balls, strikes + 1
        return False, 0.0, balls, strikes
    if event is not None:
        return True, 0.0, balls, strikes
    return False, 0.0, balls, strikes


def replay_expected(sequences: list[PASequence], strike_probs: np.ndarray, continuation: dict[str, float]) -> np.ndarray:
    outcomes = np.zeros(len(sequences), dtype=float)
    for idx, seq in enumerate(sequences):
        state_probs: dict[tuple[int, int], float] = {(0, 0): 1.0}
        walk_expectation = 0.0
        for step in seq.steps:
            if not state_probs:
                break
            next_states: dict[tuple[int, int], float] = {}
            if step.called_idx >= 0:
                p_strike = float(strike_probs[step.called_idx])
                for (balls, strikes), weight in state_probs.items():
                    if strikes < 2:
                        next_states[(balls, strikes + 1)] = next_states.get((balls, strikes + 1), 0.0) + weight * p_strike
                    if balls == 3:
                        walk_expectation += weight * (1.0 - p_strike)
                    else:
                        next_states[(balls + 1, strikes)] = next_states.get((balls + 1, strikes), 0.0) + weight * (1.0 - p_strike)
            else:
                for (balls, strikes), weight in state_probs.items():
                    resolved, walk_value, nb, ns = _advance_non_called(step.description, step.event, balls, strikes)
                    if resolved:
                        walk_expectation += weight * walk_value
                    else:
                        next_states[(nb, ns)] = next_states.get((nb, ns), 0.0) + weight
            state_probs = next_states
        for (balls, strikes), weight in state_probs.items():
            walk_expectation += weight * continuation.get(f"{balls}-{strikes}", continuation.get("0-0", 0.0))
        outcomes[idx] = walk_expectation
    return outcomes


def _sequence_weights(sequences: list[PASequence], game_weights: dict[int, int] | None = None) -> np.ndarray:
    if game_weights is None:
        return np.ones(len(sequences), dtype=float)
    return np.array([game_weights.get(seq.game_pk, 0) for seq in sequences], dtype=float)


def _fit_predict(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    train_games: np.ndarray,
    game_weights: dict[int, int] | None,
    X_eval: pd.DataFrame,
    seed: int,
    n_estimators: int = 70,
) -> tuple[object, np.ndarray, dict]:
    if game_weights is None:
        fit_idx = np.arange(len(X_train))
        weights = None
        oob_idx = np.array([], dtype=int)
    else:
        weights_all = np.array([game_weights.get(int(g), 0) for g in train_games], dtype=float)
        fit_idx = np.flatnonzero(weights_all > 0)
        weights = weights_all[fit_idx]
        oob_idx = np.flatnonzero(weights_all == 0)
    model = lgbm_classifier(seed, n_estimators=n_estimators)
    model.fit(X_train.iloc[fit_idx], y_train[fit_idx], sample_weight=weights)
    probs = model.predict_proba(X_eval)[:, 1]
    audit = {"oob_n": int(len(oob_idx)), "oob_auc": np.nan, "oob_max_calibration_deviation": np.nan, "oob_poor_calibration": False}
    if len(oob_idx) >= 500 and len(np.unique(y_train[oob_idx])) == 2:
        oob_prob = model.predict_proba(X_train.iloc[oob_idx])[:, 1]
        from data_prep_r3 import calibration_table

        tab = calibration_table(y_train[oob_idx], oob_prob)
        audit["oob_auc"] = float(__import__("sklearn.metrics").metrics.roc_auc_score(y_train[oob_idx], oob_prob))
        audit["oob_max_calibration_deviation"] = float(tab["abs_deviation"].max())
        audit["oob_poor_calibration"] = bool(audit["oob_max_calibration_deviation"] > 0.05)
    return model, probs, audit


def _stat_from_probs(
    probs: np.ndarray,
    sequences: list[PASequence],
    continuation: dict[str, float],
    term_2025: pd.DataFrame,
    term_2026: pd.DataFrame,
    weights_2025: dict[int, int] | None,
    weights_2026: dict[int, int] | None,
) -> tuple[float, float, float, float]:
    pa = replay_expected(sequences, probs, continuation)
    pa_weights = _sequence_weights(sequences, weights_2026)
    cf_rate = weighted_mean(pa, pa_weights)
    actual_2025 = walk_rate_from_terminal(term_2025, weights_2025)
    actual_2026 = walk_rate_from_terminal(term_2026, weights_2026)
    denom = actual_2026 - actual_2025
    attribution = ((actual_2026 - cf_rate) / denom) * 100 if np.isfinite(denom) and abs(denom) > 1e-9 else np.nan
    return actual_2025, actual_2026, cf_rate, attribution


def _prob_without_features(model: object, X_eval: pd.DataFrame, feature_names: list[str], drop_features: set[str]) -> np.ndarray:
    contrib = model.booster_.predict(X_eval, pred_contrib=True)
    loc_idx = [i for i, name in enumerate(feature_names) if name in drop_features]
    raw_full = contrib[:, :-1].sum(axis=1) + contrib[:, -1]
    raw_no = raw_full - contrib[:, loc_idx].sum(axis=1) if loc_idx else raw_full
    return expit(raw_no)


def _shap_group_table(model: object, X_eval: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    contrib = model.booster_.predict(X_eval, pred_contrib=True)
    raw_full = contrib[:, :-1].sum(axis=1) + contrib[:, -1]
    rows = []
    groups = {
        "zone_location": {"plate_x", "plate_z"},
        "count_state": {"balls", "strikes"} | {c for c in feature_names if c.startswith("count_state_")},
        "pitch_type": {c for c in feature_names if c.startswith("pitch_type_")},
    }
    assigned = set().union(*groups.values())
    groups["other"] = set(feature_names) - assigned
    abs_values = {}
    for group, cols in groups.items():
        idx = [i for i, name in enumerate(feature_names) if name in cols]
        raw = contrib[:, idx].sum(axis=1) if idx else np.zeros(len(X_eval))
        abs_values[group] = float(np.mean(np.abs(raw)))
        rows.append(
            {
                "feature_group": group,
                "mean_raw_shap": float(np.mean(raw)),
                "mean_abs_raw_shap": abs_values[group],
                "share_abs_raw_shap": float(np.mean(np.abs(raw)) / np.mean(np.abs(raw_full - contrib[:, -1]))),
            }
        )
    out = pd.DataFrame(rows)
    denom = float(sum(abs_values.values())) or 1.0
    out["normalized_share_abs_raw_shap"] = out["feature_group"].map(lambda g: abs_values.get(g, 0.0) / denom)
    return out


def _write_learning_curve(called_2025: pd.DataFrame, y: np.ndarray) -> None:
    rng = np.random.default_rng(GLOBAL_SEED + 1300)
    games = called_2025["game_pk"].dropna().astype(int).unique()
    holdout_games = set(rng.choice(games, size=max(20, int(len(games) * 0.18)), replace=False))
    train = called_2025[~called_2025["game_pk"].isin(holdout_games)].copy()
    test = called_2025[called_2025["game_pk"].isin(holdout_games)].copy()
    y_train_all = train["is_called_strike"].to_numpy(dtype=int)
    y_test = test["is_called_strike"].to_numpy(dtype=int)
    X_train_all, columns = encode_features(train, ZONE_NUMERIC, ZONE_CATEGORICAL)
    X_test, _ = encode_features(test, ZONE_NUMERIC, ZONE_CATEGORICAL, columns)
    rows = []
    train_games = train["game_pk"].dropna().astype(int).unique()
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        use_games = set(rng.choice(train_games, size=max(8, int(len(train_games) * frac)), replace=False))
        idx = train["game_pk"].isin(use_games).to_numpy()
        model = lgbm_classifier(GLOBAL_SEED + int(frac * 1000), n_estimators=70)
        model.fit(X_train_all.loc[idx], y_train_all[idx])
        prob = model.predict_proba(X_test)[:, 1]
        auc = __import__("sklearn.metrics").metrics.roc_auc_score(y_test, prob)
        rows.append({"train_game_fraction": frac, "train_rows": int(idx.sum()), "holdout_auc": float(auc)})
    curve = pd.DataFrame(rows)
    curve.to_csv(ARTIFACTS_DIR / "h1_zone_classifier_learning_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(curve["train_game_fraction"], curve["holdout_auc"], marker="o", color="#2563eb")
    ax.set_xlabel("Training game fraction")
    ax.set_ylabel("Holdout AUC")
    ax.set_title("H1 zone classifier learning curve")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "h1_zone_classifier_learning_curve.png", dpi=200)
    plt.close(fig)


def _plot_h1_methods(methods: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ordered = methods.copy()
    x = np.arange(len(ordered))
    yerr = np.vstack(
        [
            np.maximum(ordered["point_estimate_pct"] - ordered["ci_low_pct"], 0.0),
            np.maximum(ordered["ci_high_pct"] - ordered["point_estimate_pct"], 0.0),
        ]
    )
    ax.bar(x, ordered["point_estimate_pct"], yerr=yerr, capsize=6, color=["#2563eb", "#10b981", "#7c3aed"])
    ax.axhline(40.46, color="#6b7280", linestyle="--", linewidth=1, label="R1 Codex 40.46%")
    ax.axhline(35.26, color="#111827", linestyle=":", linewidth=1, label="R2 Codex 35.3%")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["method_short"], rotation=0)
    ax.set_ylabel("Share of YoY walk spike (%)")
    ax.set_title("R3-H1 triangulated H3 attribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "h1_triangulated_attribution.png", dpi=220)
    plt.close(fig)


def run_h1_triangulation(
    data: Round3Data,
    n_method_a_boot: int = 200,
    n_method_c_outer: int = 100,
    n_method_c_inner: int = 10,
) -> H1Result:
    print("R3-H1: preparing zone classifier and PA sequences")
    called_2025 = called_pitch_rows(data.df_2025)
    sequences, called_2026 = _build_sequences(data.df_2026)
    continuation = _continuation_probs(data.df_2026)
    term_2025 = terminal_pa_rows(data.df_2025)
    term_2026 = terminal_pa_rows(data.df_2026)
    actual_probs = (called_2026["description"] == "called_strike").astype(float).to_numpy()

    y_train = called_2025["is_called_strike"].to_numpy(dtype=int)
    train_games = called_2025["game_pk"].to_numpy(dtype=int)
    X_train, columns = encode_features(called_2025, ZONE_NUMERIC, ZONE_CATEGORICAL)
    X_eval, _ = encode_features(called_2026, ZONE_NUMERIC, ZONE_CATEGORICAL, columns)
    games_2025 = data.df_2025["game_pk"].dropna().astype(int).unique()
    games_2026 = data.df_2026["game_pk"].dropna().astype(int).unique()

    oof, oof_auc = grouped_oof_predictions(
        called_2025, y_train, train_games, ZONE_NUMERIC, ZONE_CATEGORICAL, GLOBAL_SEED + 1100, n_estimators=70
    )
    diagnostics = write_classifier_diagnostics("h1_zone_classifier", y_train, oof)
    diagnostics["oof_auc"] = oof_auc
    _write_learning_curve(called_2025, y_train)

    print("R3-H1: full-sample baseline replay")
    base_model, base_probs, _ = _fit_predict(
        X_train, y_train, train_games, None, X_eval, GLOBAL_SEED + 1200, n_estimators=90
    )
    full_2025, full_2026, full_cf, full_attr = _stat_from_probs(
        base_probs, sequences, continuation, term_2025, term_2026, None, None
    )
    no_loc_probs = _prob_without_features(base_model, X_eval, columns, {"plate_x", "plate_z"})
    _, _, no_loc_cf, _ = _stat_from_probs(no_loc_probs, sequences, continuation, term_2025, term_2026, None, None)
    shap_point = ((no_loc_cf - full_cf) / (full_2026 - full_2025)) * 100
    shap_groups = _shap_group_table(base_model, X_eval, columns)
    shap_groups.to_csv(ARTIFACTS_DIR / "h1_method_b_shap_feature_groups.csv", index=False)
    location_share = float(
        shap_groups.loc[shap_groups["feature_group"] == "zone_location", "normalized_share_abs_raw_shap"].iloc[0]
    )
    shap_point = full_attr * location_share

    rng = np.random.default_rng(GLOBAL_SEED + 1400)
    method_a_rows = []
    method_b_rows = []
    audit_rows = []
    print(f"R3-H1: Method A/B game bootstrap with {n_method_a_boot} refit iterations")
    for b in range(n_method_a_boot):
        w25 = sample_game_weights(games_2025, rng)
        w26 = sample_game_weights(games_2026, rng)
        model, probs, audit = _fit_predict(
            X_train, y_train, train_games, w25, X_eval, GLOBAL_SEED + 2000 + b, n_estimators=70
        )
        a25, a26, cf, attr = _stat_from_probs(probs, sequences, continuation, term_2025, term_2026, w25, w26)
        no_loc = _prob_without_features(model, X_eval, columns, {"plate_x", "plate_z"})
        _, _, no_loc_rate, _ = _stat_from_probs(no_loc, sequences, continuation, term_2025, term_2026, w25, w26)
        denom = a26 - a25
        shap_attr = attr * location_share if np.isfinite(attr) else np.nan
        method_a_rows.append(
            {
                "iteration": b,
                "actual_2025": a25,
                "actual_2026": a26,
                "yoy_delta_pp": (a26 - a25) * 100,
                "counterfactual_walk_rate": cf,
                "attribution_pct": attr,
            }
        )
        method_b_rows.append(
            {
                "iteration": b,
                "counterfactual_no_location_walk_rate": no_loc_rate,
                "counterfactual_full_walk_rate": cf,
                "location_shap_attribution_pct": shap_attr,
                "location_shap_share": location_share,
            }
        )
        audit_rows.append({"method": "h1_method_a_b", "iteration": b, **audit})
        if (b + 1) % 25 == 0:
            print(f"  Method A/B bootstrap {b + 1}/{n_method_a_boot}")
    method_a = pd.DataFrame(method_a_rows)
    method_b = pd.DataFrame(method_b_rows)
    method_a.to_csv(ARTIFACTS_DIR / "h1_method_a_game_bootstrap.csv", index=False)
    method_b.to_csv(ARTIFACTS_DIR / "h1_method_b_shap_game_bootstrap.csv", index=False)

    print(f"R3-H1: Method C bootstrap-of-bootstrap {n_method_c_outer} outer x {n_method_c_inner} inner")
    count_masks = {count: (called_2026["count_state"] == count).to_numpy() for count in COUNT_ORDER}
    edge_masks = {
        "top_edge": (called_2026["zone_region"] == "top_edge").to_numpy(),
        "bottom_edge": (called_2026["zone_region"] == "bottom_edge").to_numpy(),
    }
    method_c_rows = []
    count_rows = []
    edge_rows = []
    for outer in range(n_method_c_outer):
        w25 = sample_game_weights(games_2025, rng)
        w26 = sample_game_weights(games_2026, rng)
        pa_weights = _sequence_weights(sequences, w26)
        a25 = walk_rate_from_terminal(term_2025, w25)
        a26 = walk_rate_from_terminal(term_2026, w26)
        denom = a26 - a25
        inner_stats = []
        inner_probs = []
        for inner in range(n_method_c_inner):
            seed = GLOBAL_SEED + 4000 + outer * 100 + inner
            _, probs, audit = _fit_predict(X_train, y_train, train_games, w25, X_eval, seed, n_estimators=70)
            pa = replay_expected(sequences, probs, continuation)
            cf = weighted_mean(pa, pa_weights)
            attr = ((a26 - cf) / denom) * 100 if np.isfinite(denom) and abs(denom) > 1e-9 else np.nan
            inner_stats.append(attr)
            inner_probs.append(probs)
            audit_rows.append({"method": "h1_method_c_inner", "iteration": outer, "inner": inner, **audit})
        outer_stat = float(np.nanmedian(inner_stats))
        median_probs = np.nanmedian(np.vstack(inner_probs), axis=0)
        cf_median_probs = replay_expected(sequences, median_probs, continuation)
        cf_median_rate = weighted_mean(cf_median_probs, pa_weights)
        method_c_rows.append(
            {
                "outer_iteration": outer,
                "actual_2025": a25,
                "actual_2026": a26,
                "yoy_delta_pp": denom * 100,
                "inner_median_attribution_pct": outer_stat,
                "median_prob_counterfactual_walk_rate": cf_median_rate,
                "inner_min_attribution_pct": float(np.nanmin(inner_stats)),
                "inner_max_attribution_pct": float(np.nanmax(inner_stats)),
            }
        )
        for count, mask in count_masks.items():
            variant = np.where(mask, median_probs, actual_probs)
            pa_variant = replay_expected(sequences, variant, continuation)
            cf_variant = weighted_mean(pa_variant, pa_weights)
            contribution_pp = (a26 - cf_variant) * 100
            count_rows.append(
                {
                    "outer_iteration": outer,
                    "count_state": count,
                    "counterfactual_walk_rate": cf_variant,
                    "zone_contribution_pp": contribution_pp,
                    "attribution_pct": (contribution_pp / (denom * 100)) * 100 if abs(denom) > 1e-9 else np.nan,
                }
            )
        for region, mask in edge_masks.items():
            variant = np.where(mask, median_probs, actual_probs)
            pa_variant = replay_expected(sequences, variant, continuation)
            cf_variant = weighted_mean(pa_variant, pa_weights)
            contribution_pp = (a26 - cf_variant) * 100
            edge_rows.append(
                {
                    "outer_iteration": outer,
                    "region": region,
                    "counterfactual_walk_rate": cf_variant,
                    "zone_contribution_pp": contribution_pp,
                    "attribution_pct": (contribution_pp / (denom * 100)) * 100 if abs(denom) > 1e-9 else np.nan,
                }
            )
        if (outer + 1) % 10 == 0:
            print(f"  Method C outer {outer + 1}/{n_method_c_outer}")
    method_c = pd.DataFrame(method_c_rows)
    method_c.to_csv(ARTIFACTS_DIR / "h1_method_c_bootstrap_of_bootstrap.csv", index=False)
    count_boot = pd.DataFrame(count_rows)
    edge_boot = pd.DataFrame(edge_rows)
    count_boot.to_csv(ARTIFACTS_DIR / "h1_method_c_per_count_bootstrap.csv", index=False)
    edge_boot.to_csv(ARTIFACTS_DIR / "h1_method_c_per_edge_bootstrap.csv", index=False)

    per_count = (
        count_boot.groupby("count_state", observed=True)
        .agg(
            zone_contribution_pp=("zone_contribution_pp", "median"),
            ci_low_pp=("zone_contribution_pp", lambda s: percentile_ci(s)[0]),
            ci_high_pp=("zone_contribution_pp", lambda s: percentile_ci(s)[1]),
            attribution_pct=("attribution_pct", "median"),
            attribution_ci_low=("attribution_pct", lambda s: percentile_ci(s)[0]),
            attribution_ci_high=("attribution_pct", lambda s: percentile_ci(s)[1]),
        )
        .reset_index()
    )
    per_count["sort"] = per_count["count_state"].map({c: i for i, c in enumerate(COUNT_ORDER)})
    per_count = per_count.sort_values("sort").drop(columns="sort")
    per_count.to_csv(ARTIFACTS_DIR / "h1_method_c_per_count_summary.csv", index=False)

    per_edge = (
        edge_boot.groupby("region", observed=True)
        .agg(
            zone_contribution_pp=("zone_contribution_pp", "median"),
            ci_low_pp=("zone_contribution_pp", lambda s: percentile_ci(s)[0]),
            ci_high_pp=("zone_contribution_pp", lambda s: percentile_ci(s)[1]),
            attribution_pct=("attribution_pct", "median"),
            attribution_ci_low=("attribution_pct", lambda s: percentile_ci(s)[0]),
            attribution_ci_high=("attribution_pct", lambda s: percentile_ci(s)[1]),
        )
        .reset_index()
    )
    per_edge.to_csv(ARTIFACTS_DIR / "h1_method_c_per_edge_summary.csv", index=False)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(ARTIFACTS_DIR / "h1_gbm_calibration_audit.csv", index=False)

    a_low, a_high = percentile_ci(method_a["attribution_pct"])
    b_low, b_high = percentile_ci(method_b["location_shap_attribution_pct"])
    c_low, c_high = percentile_ci(method_c["inner_median_attribution_pct"])
    methods = pd.DataFrame(
        [
            {
                "method": "Method A: expectation propagation with game-bootstrap refit",
                "method_short": "A replay",
                "point_estimate_pct": full_attr,
                "ci_low_pct": a_low,
                "ci_high_pct": a_high,
            },
            {
                "method": "Method B: SHAP location-share scaled replay",
                "method_short": "B SHAP",
                "point_estimate_pct": shap_point,
                "ci_low_pct": b_low,
                "ci_high_pct": b_high,
            },
            {
                "method": "Method C: bootstrap-of-bootstrap median ensemble",
                "method_short": "C boot-of-boot",
                "point_estimate_pct": float(method_c["inner_median_attribution_pct"].median()),
                "ci_low_pct": c_low,
                "ci_high_pct": c_high,
            },
        ]
    )
    methods.to_csv(ARTIFACTS_DIR / "h1_triangulated_methods.csv", index=False)
    _plot_h1_methods(methods)

    widest = methods.assign(width=methods["ci_high_pct"] - methods["ci_low_pct"]).sort_values("width", ascending=False).iloc[0]
    tri_point = float(np.median(methods["point_estimate_pct"]))
    summary = {
        "actual_2025_walk_rate": full_2025,
        "actual_2026_walk_rate": full_2026,
        "yoy_delta_pp": (full_2026 - full_2025) * 100,
        "method_a_point_pct": full_attr,
        "method_a_ci_low_pct": a_low,
        "method_a_ci_high_pct": a_high,
        "method_a_bootstrap_n": n_method_a_boot,
        "method_b_point_pct": shap_point,
        "method_b_ci_low_pct": b_low,
        "method_b_ci_high_pct": b_high,
        "method_b_bootstrap_n": n_method_a_boot,
        "method_b_location_shap_share": location_share,
        "method_b_note": "SHAP decomposes the 2025-zone classifier's raw predicted-strike movement; H3-scale Method B multiplies the Method A game-refit replay statistic by the normalized plate_x/plate_z SHAP share.",
        "method_c_point_pct": float(method_c["inner_median_attribution_pct"].median()),
        "method_c_ci_low_pct": c_low,
        "method_c_ci_high_pct": c_high,
        "method_c_outer_n": n_method_c_outer,
        "method_c_inner_n": n_method_c_inner,
        "triangulated_headline_pct": tri_point,
        "triangulated_ci_low_pct": float(widest["ci_low_pct"]),
        "triangulated_ci_high_pct": float(widest["ci_high_pct"]),
        "triangulated_ci_source": str(widest["method_short"]),
        "round1_codex_pct": 40.46,
        "round2_codex_pct": 35.25984526587193,
        "round2_codex_ci_was_seed_artifact": True,
        "zone_classifier_diagnostics": diagnostics,
        "bootstrap_calibration_max_deviation_max": float(audit["oob_max_calibration_deviation"].max(skipna=True)),
        "bootstrap_calibration_any_gt_5pp": bool((audit["oob_max_calibration_deviation"] > 0.05).any()),
    }
    save_json(summary, ARTIFACTS_DIR / "h1_summary.json")
    return H1Result(
        summary=summary,
        methods=methods,
        method_a_bootstrap=method_a,
        method_b_bootstrap=method_b,
        method_c_outer=method_c,
        per_count_method_c=per_count,
        per_edge_method_c=per_edge,
        diagnostics=diagnostics,
    )

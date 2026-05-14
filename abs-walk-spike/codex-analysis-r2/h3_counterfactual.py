from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data_prep_r2 import (
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
    Round2Data,
    called_pitch_rows,
    encode_features,
    grouped_oof_predictions,
    lgbm_classifier,
    save_json,
    terminal_pa_rows,
    walk_rate_from_terminal,
    write_model_diagnostics,
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
    actual_walk: int
    terminal_count: str
    steps: tuple[PitchStep, ...]


@dataclass
class H3Result:
    summary: dict
    pa_level: pd.DataFrame
    per_count_variants: pd.DataFrame
    per_edge_variants: pd.DataFrame
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
        event = terminal.loc[pa_id, "events"]
        terminal_count = terminal.loc[pa_id, "terminal_count"]
        actual_walk = int(event in WALK_EVENTS)
        steps = tuple(
            PitchStep(
                description=str(row.description),
                event=None if pd.isna(row.events) or row.events == "" else str(row.events),
                called_idx=int(row.called_idx),
            )
            for row in group.itertuples()
        )
        sequences.append(PASequence(pa_id=str(pa_id), actual_walk=actual_walk, terminal_count=str(terminal_count), steps=steps))

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
            next_states: dict[tuple[int, int], float] = {}
            if not state_probs:
                break
            if step.called_idx >= 0:
                p_strike = float(strike_probs[step.called_idx])
                for (balls, strikes), weight in state_probs.items():
                    if strikes == 2:
                        pass
                    else:
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


def _actual_called_probs(called_frame: pd.DataFrame) -> np.ndarray:
    return (called_frame["description"] == "called_strike").astype(float).to_numpy()


def _fit_zone_ensemble(called_2025: pd.DataFrame, called_2026: pd.DataFrame, n_models: int = 10) -> tuple[list[np.ndarray], dict]:
    y = called_2025["is_called_strike"].to_numpy(dtype=int)
    groups = called_2025["game_pk"].to_numpy()
    oof, auc = grouped_oof_predictions(
        called_2025,
        y,
        groups,
        ZONE_NUMERIC,
        ZONE_CATEGORICAL,
        GLOBAL_SEED + 300,
        n_splits=5,
        n_estimators=180,
    )
    diagnostics = write_model_diagnostics("h2_h3_zone_classifier", y, oof)
    diagnostics["oof_auc"] = auc

    X_train, columns = encode_features(called_2025, ZONE_NUMERIC, ZONE_CATEGORICAL)
    X_2026, _ = encode_features(called_2026, ZONE_NUMERIC, ZONE_CATEGORICAL, columns)
    pred_list = []
    seed_rows = []
    for i in range(n_models):
        seed = GLOBAL_SEED + 400 + i
        model = lgbm_classifier(seed, n_estimators=180)
        model.fit(X_train, y)
        probs = model.predict_proba(X_2026)[:, 1]
        pred_list.append(probs)
        seed_rows.append({"seed": seed, "mean_strike_prob_2026": float(probs.mean())})
    pd.DataFrame(seed_rows).to_csv(ARTIFACTS_DIR / "h3_zone_ensemble_seed_predictions.csv", index=False)
    return pred_list, diagnostics


def _interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) <= 1:
        return float(values[0]), float(values[0])
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    return mean - 1.96 * sd, mean + 1.96 * sd


def _plot_h3_summary(actual_2025: float, actual_2026: float, cf_values: np.ndarray, attribution_values: np.ndarray) -> None:
    cf_mean = float(cf_values.mean())
    cf_low, cf_high = _interval(cf_values)
    attr_mean = float(attribution_values.mean())
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    labels = ["Actual 2025", "Counterfactual 2026\n(2025 zone)", "Actual 2026"]
    vals = [actual_2025, cf_mean, actual_2026]
    err = np.array([[0.0, max(cf_mean - cf_low, 0.0), 0.0], [0.0, max(cf_high - cf_mean, 0.0), 0.0]]) * 100
    ax.bar(labels, np.array(vals) * 100, color=["#6b7280", "#2563eb", "#dc2626"], yerr=err, capsize=6)
    ax.axhline(actual_2025 * 100, color="#6b7280", linewidth=1, linestyle=":")
    ax.annotate(f"Round 2: {attr_mean:.1f}%\nRound 1: 40.46%", xy=(1, cf_mean * 100), xytext=(1.35, max(vals) * 100 + 0.25))
    ax.set_ylabel("Walk rate incl. IBB (%)")
    ax.set_title("H3: all-pitches zone counterfactual")
    ax.set_ylim(0, max(vals) * 100 + 0.9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "h3_counterfactual_attribution.png", dpi=220)
    plt.close(fig)


def _plot_per_count(per_count: pd.DataFrame) -> None:
    ordered = per_count.copy()
    ordered["sort"] = ordered["count_state"].map({c: i for i, c in enumerate(COUNT_ORDER)})
    ordered = ordered.sort_values("sort")
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    colors = ["#2563eb" if v >= 0 else "#dc2626" for v in ordered["zone_contribution_pp"]]
    yerr = np.vstack([
        ordered["zone_contribution_pp"] - ordered["ci_low_pp"],
        ordered["ci_high_pp"] - ordered["zone_contribution_pp"],
    ])
    ax.bar(ordered["count_state"], ordered["zone_contribution_pp"], color=colors, yerr=yerr, capsize=4)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_ylabel("Aggregate walk-rate contribution (pp)")
    ax.set_title("H3: zone attribution by called-pitch count")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "h3_per_count_attribution.png", dpi=220)
    plt.close(fig)


def _plot_edges(edges: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    yerr = np.vstack([
        edges["attribution_pct"] - edges["ci_low_pct"],
        edges["ci_high_pct"] - edges["attribution_pct"],
    ])
    ax.bar(edges["region"], edges["attribution_pct"], color=["#dc2626", "#2563eb"], yerr=yerr, capsize=6)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.axhline(40.46, color="#6b7280", linestyle="--", linewidth=1, label="Round 1 all-pitches")
    ax.set_ylabel("Share of YoY walk spike (%)")
    ax.set_title("H3: top-edge vs bottom-edge replay")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "h3_per_edge_attribution.png", dpi=220)
    plt.close(fig)


def _plot_ensemble_variance(attribution_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(np.arange(1, len(attribution_values) + 1), attribution_values, marker="o", color="#2563eb")
    ax.axhline(float(np.mean(attribution_values)), color="#dc2626", linestyle="--", label="ensemble mean")
    ax.axhline(40.46, color="#6b7280", linestyle=":", label="Round 1")
    ax.set_xlabel("Seed model")
    ax.set_ylabel("All-pitches attribution (%)")
    ax.set_title("Counterfactual model-ensemble variance")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "h3_ensemble_variance.png", dpi=200)
    plt.close(fig)


def run_h3(data: Round2Data, n_models: int = 10) -> H3Result:
    called_2025 = called_pitch_rows(data.df_2025)
    called_2026 = called_pitch_rows(data.df_2026)
    sequences, called_frame = _build_sequences(data.df_2026)
    continuation = _continuation_probs(data.df_2026)
    actual_probs = _actual_called_probs(called_frame)
    model_prob_list, diagnostics = _fit_zone_ensemble(called_2025, called_frame, n_models=n_models)

    term_2025 = terminal_pa_rows(data.df_2025)
    term_2026 = terminal_pa_rows(data.df_2026)
    actual_2025 = walk_rate_from_terminal(term_2025)["walk_rate"]
    actual_2026 = walk_rate_from_terminal(term_2026)["walk_rate"]
    denominator = actual_2026 - actual_2025

    full_rates = []
    full_attr = []
    pa_cols = {
        "pa_id": [seq.pa_id for seq in sequences],
        "actual_walk": [seq.actual_walk for seq in sequences],
        "terminal_count": [seq.terminal_count for seq in sequences],
    }
    per_count_rows = []
    edge_rows = []

    count_masks = {count: (called_frame["count_state"] == count).to_numpy() for count in COUNT_ORDER}
    edge_masks = {
        "top_edge_z_ge_3": (called_frame["plate_z"] >= 3.0).to_numpy(),
        "bottom_edge_z_le_2": (called_frame["plate_z"] <= 2.0).to_numpy(),
    }

    for model_idx, model_probs in enumerate(model_prob_list):
        full_pa = replay_expected(sequences, model_probs, continuation)
        full_rate = float(full_pa.mean())
        full_rates.append(full_rate)
        full_attr.append(((actual_2026 - full_rate) / denominator) * 100 if denominator else np.nan)
        pa_cols[f"cf_walk_prob_seed_{model_idx}"] = full_pa

        for count, mask in count_masks.items():
            variant_probs = np.where(mask, model_probs, actual_probs)
            variant_pa = replay_expected(sequences, variant_probs, continuation)
            contribution = actual_2026 - float(variant_pa.mean())
            per_count_rows.append(
                {
                    "seed_index": model_idx,
                    "count_state": count,
                    "counterfactual_walk_rate": float(variant_pa.mean()),
                    "zone_contribution_pp": contribution * 100,
                    "attribution_pct": (contribution / denominator) * 100 if denominator else np.nan,
                }
            )

        for region, mask in edge_masks.items():
            variant_probs = np.where(mask, model_probs, actual_probs)
            variant_pa = replay_expected(sequences, variant_probs, continuation)
            contribution = actual_2026 - float(variant_pa.mean())
            edge_rows.append(
                {
                    "seed_index": model_idx,
                    "region": region,
                    "counterfactual_walk_rate": float(variant_pa.mean()),
                    "contribution_pp": contribution * 100,
                    "attribution_pct": (contribution / denominator) * 100 if denominator else np.nan,
                }
            )

    pa_level = pd.DataFrame(pa_cols)
    seed_cols = [col for col in pa_level.columns if col.startswith("cf_walk_prob_seed_")]
    pa_level["cf_walk_prob_mean"] = pa_level[seed_cols].mean(axis=1)
    pa_level.to_parquet(ARTIFACTS_DIR / "h3_pa_level_counterfactual.parquet", index=False)

    per_count_seed = pd.DataFrame(per_count_rows)
    per_count_seed.to_csv(ARTIFACTS_DIR / "h3_per_count_by_seed.csv", index=False)
    per_count = (
        per_count_seed.groupby("count_state", observed=True)
        .agg(
            counterfactual_walk_rate=("counterfactual_walk_rate", "mean"),
            zone_contribution_pp=("zone_contribution_pp", "mean"),
            ci_low_pp=("zone_contribution_pp", lambda s: _interval(s.to_numpy())[0]),
            ci_high_pp=("zone_contribution_pp", lambda s: _interval(s.to_numpy())[1]),
            attribution_pct=("attribution_pct", "mean"),
        )
        .reset_index()
    )
    per_count.to_csv(ARTIFACTS_DIR / "h3_per_count_attribution.csv", index=False)

    edge_seed = pd.DataFrame(edge_rows)
    edge_seed.to_csv(ARTIFACTS_DIR / "h3_edge_by_seed.csv", index=False)
    edge = (
        edge_seed.groupby("region", observed=True)
        .agg(
            counterfactual_walk_rate=("counterfactual_walk_rate", "mean"),
            contribution_pp=("contribution_pp", "mean"),
            attribution_pct=("attribution_pct", "mean"),
            ci_low_pct=("attribution_pct", lambda s: _interval(s.to_numpy())[0]),
            ci_high_pct=("attribution_pct", lambda s: _interval(s.to_numpy())[1]),
        )
        .reset_index()
    )
    edge.to_csv(ARTIFACTS_DIR / "h3_edge_attribution.csv", index=False)

    full_rates_arr = np.array(full_rates)
    full_attr_arr = np.array(full_attr)
    cf_low, cf_high = _interval(full_rates_arr)
    attr_low, attr_high = _interval(full_attr_arr)
    _plot_h3_summary(actual_2025, actual_2026, full_rates_arr, full_attr_arr)
    _plot_per_count(per_count)
    _plot_edges(edge)
    _plot_ensemble_variance(full_attr_arr)

    summary = {
        "actual_2025_walk_rate": actual_2025,
        "actual_2026_walk_rate": actual_2026,
        "yoy_delta_pp": denominator * 100,
        "counterfactual_walk_rate": float(full_rates_arr.mean()),
        "counterfactual_ci_low": cf_low,
        "counterfactual_ci_high": cf_high,
        "zone_attribution_pct": float(full_attr_arr.mean()),
        "zone_attribution_ci_low": attr_low,
        "zone_attribution_ci_high": attr_high,
        "round1_benchmark_pct": 40.46,
        "models": n_models,
        "model_uncertainty": "10-seed LightGBM refit ensemble; intervals are mean +/- 1.96 cross-seed SD",
        "zone_classifier_diagnostics": diagnostics,
    }
    save_json(summary, ARTIFACTS_DIR / "h3_summary.json")
    return H3Result(summary=summary, pa_level=pa_level, per_count_variants=per_count, per_edge_variants=edge, diagnostics=diagnostics)

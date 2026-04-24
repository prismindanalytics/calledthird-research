from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    ARTIFACTS_DIR,
    CHARTS_DIR,
    BALL_LIKE_DESCRIPTIONS,
    BUNT_FOUL_DESCRIPTIONS,
    CALLED_DESCRIPTIONS,
    COUNT_ORDER,
    FOUL_DESCRIPTIONS,
    FOUL_TIP_DESCRIPTIONS,
    GLOBAL_SEED,
    IN_PLAY_DESCRIPTIONS,
    STRIKE_LIKE_DESCRIPTIONS,
    WALK_EVENTS,
    bootstrap_interval,
    ensure_output_dirs,
    save_json,
)
from zone_classifier import fit_zone_model


@dataclass
class PitchStep:
    description: str
    event: str | None
    called_idx: int


@dataclass
class PASequence:
    pa_id: str
    actual_walk: int
    actual_counts: tuple[str, ...]
    steps: tuple[PitchStep, ...]


@dataclass
class CounterfactualResult:
    counterfactual_walk_rate: float
    attribution_pct: float
    bootstrap_walk_rates: np.ndarray
    bootstrap_attribution_pct: np.ndarray
    unresolved_share_mean: float
    count_chart: pd.DataFrame


def _actual_count_rates(df: pd.DataFrame) -> pd.DataFrame:
    terminal_ids = set(df.loc[df["events"].notna() & (df["events"] != ""), "pa_id"])
    filtered = df[df["pa_id"].isin(terminal_ids)].copy()
    return (
        filtered[["count_state", "pa_id"]]
        .drop_duplicates()
        .merge(
            filtered[["pa_id", "events"]]
            .dropna()
            .drop_duplicates(subset=["pa_id"], keep="last")
            .assign(actual_walk=lambda frame: frame["events"].isin(WALK_EVENTS).astype(int))[["pa_id", "actual_walk"]],
            on="pa_id",
            how="left",
        )
        .groupby("count_state", observed=True)
        .agg(pas=("pa_id", "size"), walks=("actual_walk", "sum"))
        .assign(walk_rate=lambda frame: frame["walks"] / frame["pas"])
        .reset_index()
    )


def build_pa_sequences(df: pd.DataFrame) -> tuple[list[PASequence], pd.DataFrame]:
    ordered = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True).copy()
    terminal = ordered[ordered["events"].notna() & (ordered["events"] != "")].copy()
    outcomes = terminal.drop_duplicates(subset=["pa_id"], keep="last").set_index("pa_id")
    valid_pa_ids = set(outcomes.index)
    ordered = ordered[ordered["pa_id"].isin(valid_pa_ids)].reset_index(drop=True).copy()
    ordered["called_idx"] = -1
    called_mask = ordered["description"].isin(CALLED_DESCRIPTIONS) & ordered[["plate_x", "plate_z_norm"]].notna().all(axis=1)
    ordered.loc[called_mask, "called_idx"] = np.arange(int(called_mask.sum()))

    sequences: list[PASequence] = []
    for pa_id, group in ordered.groupby("pa_id", sort=False):
        steps = tuple(
            PitchStep(
                description=row.description,
                event=None if pd.isna(row.events) or row.events == "" else str(row.events),
                called_idx=int(row.called_idx),
            )
            for row in group.itertuples()
        )
        counts = tuple(group["count_state"].drop_duplicates().tolist())
        sequences.append(
            PASequence(
                pa_id=str(pa_id),
                actual_walk=int(outcomes.loc[pa_id, "events"] in WALK_EVENTS),
                actual_counts=counts,
                steps=steps,
            )
        )

    called_frame = ordered.loc[called_mask, ["pa_id", "count_state", "description", "plate_x", "plate_z_norm"]].reset_index(drop=True)
    return sequences, called_frame


def _advance_actual_pitch(description: str, event: str | None, balls: int, strikes: int) -> tuple[bool, float, int, int, bool]:
    if event in WALK_EVENTS:
        return True, 1.0, balls, strikes, False
    if description in IN_PLAY_DESCRIPTIONS or description == "hit_by_pitch" or (event is not None and event not in WALK_EVENTS):
        return True, 0.0, balls, strikes, False
    if description in BALL_LIKE_DESCRIPTIONS:
        if balls == 3:
            return True, 1.0, 4, strikes, False
        return False, 0.0, balls + 1, strikes, False
    if description in STRIKE_LIKE_DESCRIPTIONS:
        if strikes == 2:
            return True, 0.0, balls, 3, False
        return False, 0.0, balls, strikes + 1, False
    if description in FOUL_TIP_DESCRIPTIONS:
        if strikes == 2:
            return True, 0.0, balls, 3, False
        return False, 0.0, balls, strikes + 1, False
    if description in BUNT_FOUL_DESCRIPTIONS:
        if strikes == 2:
            return True, 0.0, balls, 3, False
        return False, 0.0, balls, strikes + 1, False
    if description in FOUL_DESCRIPTIONS:
        if strikes < 2:
            return False, 0.0, balls, strikes + 1, False
        return False, 0.0, balls, strikes, False
    raise ValueError(f"Unhandled description in replay: {description}")


def simulate_pa(
    sequence: PASequence,
    strike_probs: np.ndarray,
    continuation_probs: dict[str, float],
    rng: np.random.Generator,
) -> tuple[float, bool]:
    balls = 0
    strikes = 0

    for step in sequence.steps:
        if step.called_idx >= 0:
            is_strike = bool(rng.random() < strike_probs[step.called_idx])
            if is_strike:
                if strikes == 2:
                    return 0.0, False
                strikes += 1
            else:
                if balls == 3:
                    return 1.0, False
                balls += 1
            continue

        resolved, walk_value, balls, strikes, unresolved = _advance_actual_pitch(step.description, step.event, balls, strikes)
        if resolved:
            return walk_value, unresolved

    tail_state = f"{balls}-{strikes}"
    return float(continuation_probs.get(tail_state, continuation_probs["0-0"])), True


def simulate_dataset(
    sequences: list[PASequence],
    strike_probs: np.ndarray,
    continuation_probs: dict[str, float],
    draws: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    outcomes = np.zeros(len(sequences), dtype=float)
    unresolved = np.zeros(len(sequences), dtype=float)

    for draw_idx in range(draws):
        rng = np.random.default_rng(seed + draw_idx)
        for pa_idx, sequence in enumerate(sequences):
            walk_value, unresolved_flag = simulate_pa(sequence, strike_probs, continuation_probs, rng)
            outcomes[pa_idx] += walk_value
            unresolved[pa_idx] += float(unresolved_flag)

    outcomes /= draws
    unresolved /= draws
    return outcomes, float(unresolved.mean())


def _count_chart_frame(
    sequences: list[PASequence],
    pa_walk_probs: np.ndarray,
    actual_2025_counts: pd.DataFrame,
    actual_2026_counts: pd.DataFrame,
) -> pd.DataFrame:
    count_to_indices: dict[str, list[int]] = {count: [] for count in COUNT_ORDER}
    for idx, sequence in enumerate(sequences):
        reached = set(sequence.actual_counts)
        for count in reached:
            if count in count_to_indices:
                count_to_indices[count].append(idx)

    predicted_rows = []
    for count in COUNT_ORDER:
        indices = count_to_indices.get(count, [])
        if not indices:
            predicted_rows.append({"count_state": count, "counterfactual_walk_rate": np.nan})
            continue
        predicted_rows.append({"count_state": count, "counterfactual_walk_rate": float(pa_walk_probs[indices].mean())})
    predicted = pd.DataFrame(predicted_rows)

    merged = predicted.merge(
        actual_2025_counts[["count_state", "walk_rate"]].rename(columns={"walk_rate": "actual_2025_walk_rate"}),
        on="count_state",
        how="left",
    ).merge(
        actual_2026_counts[["count_state", "walk_rate"]].rename(columns={"walk_rate": "actual_2026_walk_rate"}),
        on="count_state",
        how="left",
    )
    return merged


def _plot_counterfactual_bars(
    actual_2025_walk_rate: float,
    counterfactual_walk_rate: float,
    actual_2026_walk_rate: float,
    bootstrap_walk_rates: np.ndarray,
    attribution_pct: float,
) -> None:
    labels = ["Actual 2025", "Counterfactual 2026\n(2025 zone)", "Actual 2026"]
    values = [actual_2025_walk_rate, counterfactual_walk_rate, actual_2026_walk_rate]
    cf_low, cf_high = bootstrap_interval(bootstrap_walk_rates)
    yerr = [[0.0, max(counterfactual_walk_rate - cf_low, 0.0), 0.0], [0.0, max(cf_high - counterfactual_walk_rate, 0.0), 0.0]]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#6b7280", "#3b82f6", "#ef4444"]
    ax.bar(labels, np.array(values) * 100, color=colors, yerr=np.array(yerr) * 100, capsize=6)
    ax.set_ylabel("Walk rate (%)")
    ax.set_title("Zone-Only Counterfactual Walk Rate")
    ax.annotate(
        f"Zone-only attribution: {attribution_pct:.1f}%",
        xy=(1, counterfactual_walk_rate * 100),
        xytext=(1, max(values) * 100 + 0.35),
        ha="center",
        arrowprops={"arrowstyle": "-", "color": "black"},
    )
    ax.set_ylim(0, max(values) * 100 + 0.7)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "counterfactual_walk_rate.png", dpi=220)
    plt.close(fig)


def _plot_walk_by_count(count_chart: pd.DataFrame) -> None:
    x = np.arange(len(count_chart))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.axvspan(len(count_chart) - 1 - 0.55, len(count_chart) - 1 + 0.55, color="#fde68a", alpha=0.35)
    ax.bar(x - width, count_chart["actual_2025_walk_rate"] * 100, width=width, label="Actual 2025", color="#6b7280")
    ax.bar(x, count_chart["counterfactual_walk_rate"] * 100, width=width, label="Counterfactual 2026", color="#3b82f6")
    ax.bar(x + width, count_chart["actual_2026_walk_rate"] * 100, width=width, label="Actual 2026", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(count_chart["count_state"])
    ax.set_ylabel("Walk rate after reaching count (%)")
    ax.set_title("Walk Rate by Count: Actual vs Counterfactual")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "walk_by_count.png", dpi=220)
    plt.close(fig)


def run_counterfactual_analysis(
    df_2025_primary: pd.DataFrame,
    df_2026_primary: pd.DataFrame,
    called_2025_primary: pd.DataFrame,
    called_2026_primary: pd.DataFrame,
    actual_2025_walk_rate: float,
    actual_2026_walk_rate: float,
    n_bootstrap: int = 100,
    draws_per_bootstrap: int = 12,
) -> CounterfactualResult:
    ensure_output_dirs()
    actual_2025_counts = _actual_count_rates(df_2025_primary)
    actual_2026_counts = _actual_count_rates(df_2026_primary)

    continuation_probs = dict(zip(actual_2026_counts["count_state"], actual_2026_counts["walk_rate"], strict=True))
    continuation_probs.setdefault("0-0", actual_2026_walk_rate)

    sequences, called_frame_2026 = build_pa_sequences(df_2026_primary)
    cf_features = called_frame_2026[["plate_x", "plate_z_norm"]].to_numpy(dtype=float)
    point_model = fit_zone_model(called_2025_primary, GLOBAL_SEED + 500)
    point_probs = point_model.predict_proba(cf_features)[:, 1]
    pa_walk_probs, unresolved_share_mean = simulate_dataset(
        sequences,
        point_probs,
        continuation_probs,
        draws=32,
        seed=GLOBAL_SEED + 700,
    )

    counterfactual_walk_rate = float(pa_walk_probs.mean())
    denominator = actual_2026_walk_rate - actual_2025_walk_rate
    attribution_pct = float(((actual_2026_walk_rate - counterfactual_walk_rate) / denominator) * 100.0)

    rng = np.random.default_rng(GLOBAL_SEED + 900)
    bootstrap_walk_rates = np.zeros(n_bootstrap, dtype=float)
    bootstrap_attribution = np.zeros(n_bootstrap, dtype=float)
    bootstrap_unresolved = np.zeros(n_bootstrap, dtype=float)

    for bootstrap_idx in range(n_bootstrap):
        sample_2025 = called_2025_primary.iloc[rng.choice(len(called_2025_primary), size=len(called_2025_primary), replace=True)].copy()
        boot_model = fit_zone_model(sample_2025, GLOBAL_SEED + 10000 + bootstrap_idx)
        boot_probs = boot_model.predict_proba(cf_features)[:, 1]
        boot_pa_probs, boot_unresolved = simulate_dataset(
            sequences,
            boot_probs,
            continuation_probs,
            draws=draws_per_bootstrap,
            seed=GLOBAL_SEED + 20000 + bootstrap_idx * 13,
        )
        bootstrap_walk_rates[bootstrap_idx] = float(boot_pa_probs.mean())
        bootstrap_unresolved[bootstrap_idx] = boot_unresolved
        bootstrap_attribution[bootstrap_idx] = float(((actual_2026_walk_rate - bootstrap_walk_rates[bootstrap_idx]) / denominator) * 100.0)

    count_chart = _count_chart_frame(sequences, pa_walk_probs, actual_2025_counts, actual_2026_counts)
    _plot_counterfactual_bars(
        actual_2025_walk_rate,
        counterfactual_walk_rate,
        actual_2026_walk_rate,
        bootstrap_walk_rates,
        attribution_pct,
    )
    _plot_walk_by_count(count_chart)

    diagnostics = {
        "counterfactual_walk_rate": counterfactual_walk_rate,
        "attribution_pct": attribution_pct,
        "bootstrap_walk_rate_ci": list(bootstrap_interval(bootstrap_walk_rates)),
        "bootstrap_attribution_ci": list(bootstrap_interval(bootstrap_attribution)),
        "unresolved_share_mean": unresolved_share_mean,
        "bootstrap_unresolved_share_mean": float(bootstrap_unresolved.mean()),
    }
    save_json(diagnostics, ARTIFACTS_DIR / "counterfactual_metrics.json")

    count_chart.to_csv(ARTIFACTS_DIR / "walk_by_count_chart_data.csv", index=False)
    called_frame_2026.to_csv(ARTIFACTS_DIR / "called_pitch_counterfactual_inputs.csv", index=False)

    return CounterfactualResult(
        counterfactual_walk_rate=counterfactual_walk_rate,
        attribution_pct=attribution_pct,
        bootstrap_walk_rates=bootstrap_walk_rates,
        bootstrap_attribution_pct=bootstrap_attribution,
        unresolved_share_mean=unresolved_share_mean,
        count_chart=count_chart,
    )

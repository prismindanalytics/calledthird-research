from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_prep_r2 import ARTIFACTS_DIR, CHARTS_DIR, COUNT_ORDER, Round2Data, save_json, terminal_pa_rows
from h3_counterfactual import H3Result


@dataclass
class H2Result:
    summary: dict
    table: pd.DataFrame


def _terminal_count_rates(terminal: pd.DataFrame) -> pd.DataFrame:
    out = (
        terminal.groupby("terminal_count", observed=True)
        .agg(pas=("pa_id", "size"), walks=("walk_event", "sum"))
        .reset_index()
        .rename(columns={"terminal_count": "count_state"})
    )
    out["share"] = out["pas"] / out["pas"].sum()
    out["walk_rate"] = out["walks"] / out["pas"]
    out["sort"] = out["count_state"].map({c: i for i, c in enumerate(COUNT_ORDER)})
    return out.sort_values("sort").drop(columns="sort")


def _interval(values: pd.Series) -> tuple[float, float]:
    arr = values.to_numpy(dtype=float)
    if len(arr) <= 1:
        return float(arr[0]), float(arr[0])
    m = float(arr.mean())
    sd = float(arr.std(ddof=1))
    return m - 1.96 * sd, m + 1.96 * sd


def _plot(table: pd.DataFrame) -> None:
    plot_df = table.copy()
    plot_df["sort"] = plot_df["count_state"].map({c: i for i, c in enumerate(COUNT_ORDER)})
    plot_df = plot_df.sort_values("sort")

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    x = np.arange(len(plot_df))
    width = 0.38
    ax.bar(x - width / 2, plot_df["rate_effect_pp"], width=width, color="#2563eb", label="Within-count rate effect")
    ax.bar(x + width / 2, plot_df["flow_effect_pp"], width=width, color="#f59e0b", label="Terminal-count traffic effect")
    ax.errorbar(
        x - width / 2,
        plot_df["zone_contribution_pp"],
        yerr=np.vstack([
            plot_df["zone_contribution_pp"] - plot_df["zone_ci_low_pp"],
            plot_df["zone_ci_high_pp"] - plot_df["zone_contribution_pp"],
        ]),
        fmt="o",
        color="#111827",
        capsize=4,
        label="Zone replay contribution",
    )
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["count_state"])
    ax.set_ylabel("Contribution to aggregate walk-rate delta (pp)")
    ax.set_title("H2: per-count decomposition of the walk spike")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "h2_per_count_contribution.png", dpi=220)
    plt.close(fig)


def run_h2(data: Round2Data, h3: H3Result) -> H2Result:
    term_2025 = terminal_pa_rows(data.df_2025)
    term_2026 = terminal_pa_rows(data.df_2026)
    rates_2025 = _terminal_count_rates(term_2025).rename(
        columns={"pas": "pas_2025", "walks": "walks_2025", "share": "share_2025", "walk_rate": "walk_rate_2025"}
    )
    rates_2026 = _terminal_count_rates(term_2026).rename(
        columns={"pas": "pas_2026", "walks": "walks_2026", "share": "share_2026", "walk_rate": "walk_rate_2026"}
    )
    table = pd.DataFrame({"count_state": COUNT_ORDER}).merge(rates_2025, on="count_state", how="left").merge(
        rates_2026, on="count_state", how="left"
    )
    fill_cols = [col for col in table.columns if col != "count_state"]
    table[fill_cols] = table[fill_cols].fillna(0.0)

    table["delta_pp"] = (table["walk_rate_2026"] - table["walk_rate_2025"]) * 100
    table["rate_effect_pp"] = table["share_2026"] * (table["walk_rate_2026"] - table["walk_rate_2025"]) * 100
    table["flow_effect_pp"] = (table["share_2026"] - table["share_2025"]) * table["walk_rate_2025"] * 100
    table["total_observed_contribution_pp"] = table["rate_effect_pp"] + table["flow_effect_pp"]

    pa = h3.pa_level.copy()
    seed_cols = [col for col in pa.columns if col.startswith("cf_walk_prob_seed_")]
    seed_rows = []
    for col in seed_cols:
        temp = (
            pa.groupby("terminal_count", observed=True)
            .agg(actual_walk_rate=("actual_walk", "mean"), counterfactual_walk_rate=(col, "mean"), pas=("pa_id", "size"))
            .reset_index()
            .rename(columns={"terminal_count": "count_state"})
        )
        temp["seed_col"] = col
        temp["share_2026"] = temp["pas"] / len(pa)
        temp["zone_contribution_pp"] = temp["share_2026"] * (temp["actual_walk_rate"] - temp["counterfactual_walk_rate"]) * 100
        seed_rows.append(temp)
    seed_df = pd.concat(seed_rows, ignore_index=True)
    seed_df.to_csv(ARTIFACTS_DIR / "h2_counterfactual_by_terminal_count_seed.csv", index=False)
    zone = (
        seed_df.groupby("count_state", observed=True)
        .agg(
            counterfactual_walk_rate=("counterfactual_walk_rate", "mean"),
            zone_contribution_pp=("zone_contribution_pp", "mean"),
            zone_ci_low_pp=("zone_contribution_pp", lambda s: _interval(s)[0]),
            zone_ci_high_pp=("zone_contribution_pp", lambda s: _interval(s)[1]),
        )
        .reset_index()
    )
    table = table.merge(zone, on="count_state", how="left").fillna(0.0)
    table["sort"] = table["count_state"].map({c: i for i, c in enumerate(COUNT_ORDER)})
    table = table.sort_values("sort").drop(columns="sort").reset_index(drop=True)
    table.to_csv(ARTIFACTS_DIR / "h2_per_count_decomposition.csv", index=False)
    _plot(table)

    summary = {
        "observed_total_delta_pp_from_components": float(table["total_observed_contribution_pp"].sum()),
        "within_count_rate_effect_pp": float(table["rate_effect_pp"].sum()),
        "terminal_count_flow_effect_pp": float(table["flow_effect_pp"].sum()),
        "zone_replay_contribution_pp": float(table["zone_contribution_pp"].sum()),
        "largest_observed_count": table.loc[table["total_observed_contribution_pp"].abs().idxmax(), "count_state"],
        "largest_zone_count": table.loc[table["zone_contribution_pp"].abs().idxmax(), "count_state"],
        "model_uncertainty": "Inherited 10-seed LightGBM zone ensemble from H3; no fixed-model row bootstrap.",
    }
    save_json(summary, ARTIFACTS_DIR / "h2_summary.json")
    return H2Result(summary=summary, table=table)

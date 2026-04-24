"""H3: count-leverage analysis — does 3-2 take the worst hit?

For 2025 vs 2026 (Mar 27 - Apr 14):

For each (PA, count_state at the time the PA-terminating pitch is faced) — actually
we need a per-PA "passed-through" count flag because a PA passes through many counts.
Two definitions, both reported:
  (A) "Plate appearance ever reached count C" — denominator = PAs that reach C
      Walk rate at count C = walks ending in PAs that reached C / PAs that reached C
      (this is identical to "what fraction of PAs that ever saw count C ended in BB")
  (B) "Pitch-level": for each pitch faced at count C, what fraction of those PAs
      eventually walk? (overlaps with A; we use A as the cleaner one.)

Per-count walk rate is:
  walks at count = sum over PAs that reached count C of indicator(events in walk)
  PAs at count = number of distinct PAs that reached count C

We then test whether the 2026-2025 delta at 3-2 is statistically larger than the
all-counts pooled delta using a binomial difference-of-differences test.

Outputs:
  - charts/walk_by_count.png (12 counts, 2025 vs 2026 with CI bars; 3-2 highlighted)
  - artifact dict with per-count rates, deltas, p-values
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from common import (
    PROJECT_ROOT, ALL_COUNTS,
    load_2025_samewin, load_2026, restrict_to_primary_window,
    plate_appearances, walk_flag,
)

OUT_DIR = PROJECT_ROOT / "claude-analysis"
CHART_DIR = OUT_DIR / "charts"
ART_DIR = OUT_DIR / "artifacts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)


def attach_pa_id(df: pd.DataFrame) -> pd.DataFrame:
    """Build a stable PA id from at_bat_number + game_pk if available, else fallback."""
    df = df.copy()
    if "at_bat_number" in df.columns and "game_pk" in df.columns:
        df["_pa_id"] = (df["game_pk"].astype("Int64").astype(str) + "_"
                        + df["at_bat_number"].astype("Int64").astype(str))
    else:
        # fallback - by game_date + batter + pitcher + inning + outs (less stable)
        df["_pa_id"] = (df["game_date"].astype(str) + "_"
                        + df["batter"].astype("Int64").astype(str) + "_"
                        + df.get("pitcher", pd.Series([0]*len(df))).astype("Int64").astype(str) + "_"
                        + df.get("inning", pd.Series([0]*len(df))).astype("Int64").astype(str))
    return df


def per_count_walk_rates(df_season: pd.DataFrame) -> pd.DataFrame:
    """For one season, per count_state compute (PAs reaching count, walk count, walk rate)."""
    df = attach_pa_id(df_season)
    df = df.dropna(subset=["balls", "strikes"])
    df["count_state"] = df["balls"].astype(int).astype(str) + "-" + df["strikes"].astype(int).astype(str)
    # PA terminating
    pa_term = df.dropna(subset=["events"])
    pa_term = pa_term.loc[pa_term["events"].astype(str).str.len() > 0]
    pa_walked = pa_term.assign(_walk=walk_flag(pa_term))[["_pa_id", "_walk"]]
    pa_walked = pa_walked.drop_duplicates(subset="_pa_id")

    # PAs that reached each count state - any pitch with that count is "reached"
    reached = df.drop_duplicates(subset=["_pa_id", "count_state"])[["_pa_id", "count_state"]]
    reached = reached.merge(pa_walked, on="_pa_id", how="left")
    reached["_walk"] = reached["_walk"].fillna(0).astype(int)
    g = reached.groupby("count_state").agg(
        pas_reached=("_pa_id", "count"),
        walks=("_walk", "sum"),
    )
    g["walk_rate"] = g["walks"] / g["pas_reached"]
    return g.reindex(list(ALL_COUNTS))


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom
    return (centre - half, centre + half)


def diff_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-proportion Z-test, two-sided p value."""
    if n1 == 0 or n2 == 0:
        return np.nan
    p1 = k1 / n1; p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p2 - p1) / se
    return 2 * (1 - stats.norm.cdf(abs(z)))


def compare_count_deltas(g25: pd.DataFrame, g26: pd.DataFrame, focus: str = "3-2"):
    """For each count, compute (delta_pp, CI on delta, two-prop test p)."""
    rows = []
    for c in ALL_COUNTS:
        n25 = int(g25.loc[c, "pas_reached"])
        k25 = int(g25.loc[c, "walks"])
        n26 = int(g26.loc[c, "pas_reached"])
        k26 = int(g26.loc[c, "walks"])
        p25 = k25 / max(n25, 1); p26 = k26 / max(n26, 1)
        delta = p26 - p25
        # Bootstrap CI on delta
        ci25 = wilson_ci(k25, n25); ci26 = wilson_ci(k26, n26)
        # Difference CI via Newcombe hybrid:
        if n25 > 0 and n26 > 0:
            l1, u1 = ci25; l2, u2 = ci26
            d_lo = (p26 - p25) - np.sqrt((p26 - l2)**2 + (u1 - p25)**2)
            d_hi = (p26 - p25) + np.sqrt((u2 - p26)**2 + (p25 - l1)**2)
        else:
            d_lo, d_hi = np.nan, np.nan
        p_val = diff_proportion_p(k25, n25, k26, n26)
        rows.append({
            "count": c, "pas25": n25, "walks25": k25, "rate25": p25,
            "ci25_lo": ci25[0], "ci25_hi": ci25[1],
            "pas26": n26, "walks26": k26, "rate26": p26,
            "ci26_lo": ci26[0], "ci26_hi": ci26[1],
            "delta_pp": delta, "delta_ci_lo": d_lo, "delta_ci_hi": d_hi,
            "p_val": p_val,
        })
    df = pd.DataFrame(rows).set_index("count")

    # All-counts pooled delta = total walks 26 / total PAs 26 minus same in 25.
    # The "all counts" baseline should use 0-0 (every PA reaches 0-0) for pure walk rate.
    # We'll report both: 0-0 (every PA) and all-12 weighted.
    pooled_n25 = int(g25.loc["0-0", "pas_reached"])
    pooled_k25 = int(g25.loc["0-0", "walks"])
    pooled_n26 = int(g26.loc["0-0", "pas_reached"])
    pooled_k26 = int(g26.loc["0-0", "walks"])
    pooled_p25 = pooled_k25 / pooled_n25
    pooled_p26 = pooled_k26 / pooled_n26
    pooled_delta = pooled_p26 - pooled_p25

    # Heterogeneity test: focus delta vs all-counts delta (difference of differences)
    # Test as: 3-2 walk rate 26 vs 25 vs 0-0 walk rate 26 vs 25 — equivalent to a
    # season:count interaction. Use a chi-square on a 2x2x2 contingency:
    # focus stratum vs reference stratum, year, walk/no-walk.
    f = df.loc[focus]
    fk25, fn25 = int(f.walks25), int(f.pas25); fk26, fn26 = int(f.walks26), int(f.pas26)
    rk25, rn25 = pooled_k25, pooled_n25; rk26, rn26 = pooled_k26, pooled_n26
    table = np.array([
        [[fk26, fn26 - fk26], [fk25, fn25 - fk25]],
        [[rk26, rn26 - rk26], [rk25, rn25 - rk25]],
    ])
    # Breslow-Day not in scipy. Use a simple Z test on the difference of deltas:
    # delta_focus - delta_ref ~ N(0, sqrt(SE_focus^2 + SE_ref^2))
    se_focus = np.sqrt(f.rate26 * (1 - f.rate26) / fn26 + f.rate25 * (1 - f.rate25) / fn25)
    se_ref = np.sqrt(pooled_p26 * (1 - pooled_p26) / pooled_n26 +
                     pooled_p25 * (1 - pooled_p25) / pooled_n25)
    diff_of_deltas = f.delta_pp - pooled_delta
    se_dod = np.sqrt(se_focus**2 + se_ref**2)
    z_dod = diff_of_deltas / se_dod if se_dod > 0 else np.nan
    p_dod = 2 * (1 - stats.norm.cdf(abs(z_dod))) if np.isfinite(z_dod) else np.nan

    # Cochran's Q across 12 strata for the season:walk interaction (Mantel-Haenszel-ish)
    # Cochran Q here = sum over counts of (delta_c - mean_delta)^2 / var_c
    deltas = df["delta_pp"].to_numpy()
    se_per = np.sqrt(df["rate25"] * (1 - df["rate25"]) / df["pas25"] +
                     df["rate26"] * (1 - df["rate26"]) / df["pas26"]).to_numpy()
    weights = 1 / np.maximum(se_per**2, 1e-12)
    weighted_mean = float(np.sum(weights * deltas) / np.sum(weights))
    Q = float(np.sum(weights * (deltas - weighted_mean)**2))
    df_q = len(deltas) - 1
    p_Q = float(1 - stats.chi2.cdf(Q, df_q))

    return df, {
        "pooled_p25": pooled_p25, "pooled_p26": pooled_p26,
        "pooled_delta_pp": pooled_delta,
        f"{focus}_delta_pp": float(f.delta_pp),
        f"{focus}_minus_pooled_pp": float(diff_of_deltas),
        f"{focus}_pooled_z": float(z_dod) if np.isfinite(z_dod) else None,
        f"{focus}_pooled_p_value": float(p_dod) if np.isfinite(p_dod) else None,
        "ratio_focus_over_pooled": float(f.delta_pp / pooled_delta) if pooled_delta != 0 else None,
        "cochran_Q": Q, "cochran_df": df_q, "cochran_p": p_Q,
    }


def plot_per_count(df: pd.DataFrame, focus: str, savepath: Path):
    fig, ax = plt.subplots(figsize=(11.0, 5.4), dpi=140)
    counts = list(df.index)
    x = np.arange(len(counts))
    w = 0.4
    r25 = df["rate25"].to_numpy() * 100
    r26 = df["rate26"].to_numpy() * 100
    err25 = np.array([
        (df["rate25"].to_numpy() - df["ci25_lo"].to_numpy()) * 100,
        (df["ci25_hi"].to_numpy() - df["rate25"].to_numpy()) * 100,
    ])
    err26 = np.array([
        (df["rate26"].to_numpy() - df["ci26_lo"].to_numpy()) * 100,
        (df["ci26_hi"].to_numpy() - df["rate26"].to_numpy()) * 100,
    ])
    ax.bar(x - w/2, r25, width=w, color="#1f77b4", label="2025", yerr=err25, capsize=2)
    ax.bar(x + w/2, r26, width=w, color="#d62728", label="2026", yerr=err26, capsize=2)
    # Highlight focus count
    idx = counts.index(focus)
    ax.axvspan(idx - 0.5, idx + 0.5, color="gold", alpha=0.18, zorder=-1)
    # Annotate deltas above each pair
    for i, c in enumerate(counts):
        d = df.loc[c, "delta_pp"] * 100
        sym = "+" if d >= 0 else ""
        ax.text(i, max(r25[i], r26[i]) + 1.5, f"{sym}{d:.1f}pp",
                ha="center", fontsize=8, color="0.2")
    ax.set_xticks(x); ax.set_xticklabels(counts)
    ax.set_xlabel("Count state"); ax.set_ylabel("Walk rate at PAs that reached this count (%)")
    ax.set_title(f"Walk rate by count state (Mar 27 - Apr 14), 2025 vs 2026 — {focus} highlighted")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(r26.max(), r25.max()) * 1.18)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)


def run() -> dict:
    df25 = load_2025_samewin()
    df26_pri = restrict_to_primary_window(load_2026())
    print(f"[count_leverage] 2025 rows: {len(df25):,}  2026 (Mar27-Apr14) rows: {len(df26_pri):,}")
    g25 = per_count_walk_rates(df25)
    g26 = per_count_walk_rates(df26_pri)
    print("[count_leverage] per-count summary 2025:"); print(g25)
    print("[count_leverage] per-count summary 2026:"); print(g26)

    table, summary = compare_count_deltas(g25, g26, focus="3-2")
    plot_per_count(table, focus="3-2", savepath=CHART_DIR / "walk_by_count.png")

    # Save table CSV
    table.to_csv(ART_DIR / "count_leverage.csv")
    return {"per_count_table_csv": str(ART_DIR / "count_leverage.csv"), **summary,
            "table_records": table.reset_index().to_dict(orient="records")}


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, default=str, indent=2))

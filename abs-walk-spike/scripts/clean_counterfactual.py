"""Clean third-implementation counterfactual to triangulate sign + magnitude.

Design choices to match Codex's adjudication.py:
  - Average across multiple draws per PA (lower variance)
  - Use empirical 2026 walk rate at the terminal count for unresolved tails
  - Same poly-2 + StandardScaler + LogReg (C=0.2) zone model
  - Keep all PAs (no dropna of plate coords on PA-counting)
  - Restrict to apples-to-apples Mar 27-Apr 14 window in both years

Differences from orchestrator's adjudicate_absolute_coords.py:
  - This implementation uses Codex-style design (averaging + continuation probs)
  - Sole purpose: confirm direction and magnitude of attribution

Output: research/abs-walk-spike/reviews/adjudication_results_clean.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
DATA_2026 = ROOT / "research" / "abs-walk-spike" / "data" / "statcast_2026_mar27_apr22.parquet"
DATA_2025 = ROOT / "research" / "count-distribution-abs" / "data" / "statcast_2025_mar27_apr14.parquet"
OUT = Path(__file__).resolve().parents[1] / "reviews" / "adjudication_results_clean.json"

WINDOW_END_2026 = pd.Timestamp("2026-04-14")
WALK_EVENTS = {"walk", "intent_walk"}
SEED = 20260423
N_DRAWS = 32  # match Codex's draws_per_bootstrap

CALLED = {"called_strike", "ball"}
IN_PLAY = {"hit_into_play"}
SWING_STRIKE = {"swinging_strike", "swinging_strike_blocked"}
FOUL = {"foul"}
FOUL_TIP = {"foul_tip", "bunt_foul_tip"}
BUNT_FOUL = {"foul_bunt"}
BLOCKED_BALL = {"blocked_ball"}
HBP = {"hit_by_pitch"}
ABS_ARTIFACT = {"automatic_ball", "automatic_strike", "pitchout", "missed_bunt"}


def load_window(path, year, end=None):
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date"].dt.year == year].copy()
    if end is not None:
        df = df[df["game_date"] <= end].copy()
    df["balls"] = df["balls"].astype(int)
    df["strikes"] = df["strikes"].astype(int)
    df["count_state"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    df["pa_id"] = df["game_pk"].astype(int).astype(str) + "_" + df["at_bat_number"].astype(int).astype(str)
    return df


def fit_zone_model(called):
    """Match Codex spec exactly: poly-2, StandardScaler, LogReg C=0.2."""
    X = called[["plate_x", "plate_z"]].to_numpy()
    y = (called["description"] == "called_strike").astype(int).to_numpy()
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=0.2, max_iter=4000, random_state=SEED)),
    ])
    pipe.fit(X, y)
    return pipe


def actual_walk_rate_by_count(df_26):
    """Empirical 2026 walk rate for PAs that PASS THROUGH each count state.
    Used as continuation probability for unresolved counterfactual tails.
    """
    pa_counts = df_26.groupby("pa_id")["count_state"].apply(set).to_dict()
    pa_walks = (
        df_26[df_26["events"].notna() & (df_26["events"] != "")]
        .drop_duplicates("pa_id", keep="last")
        .set_index("pa_id")["events"]
        .isin(WALK_EVENTS)
        .to_dict()
    )

    by_count = {}
    for pa_id, counts in pa_counts.items():
        if pa_id not in pa_walks:
            continue
        is_walk = pa_walks[pa_id]
        for c in counts:
            by_count.setdefault(c, []).append(is_walk)
    return {c: np.mean(walks) for c, walks in by_count.items() if walks}


def build_pa_steps(df, called_pitch_filter):
    """For each PA, build an ordered step list:
      step = (kind, payload)
      kind ∈ {'cf_called', 'actual_called', 'in_play', 'foul', 'foul_tip', 'swing_strike',
              'blocked_ball', 'hbp', 'abs', 'unknown'}
      For 'cf_called', payload is plate_x, plate_z (to score with 2025 model)
      For 'actual_called', payload is whether actual was a strike (1 or 0)
      For others, payload is the (description, events) tuple

    called_pitch_filter: function(row_dict) → bool. True means treat as 'cf_called',
    False means treat as 'actual_called' (use actual outcome).
    """
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    pa_steps = {}
    pa_actual_walk = {}

    for pa_id, group in df.groupby("pa_id", sort=False):
        steps = []
        for _, row in group.iterrows():
            desc = row["description"]
            events = row["events"] if pd.notna(row["events"]) else None
            if desc in CALLED:
                if pd.isna(row["plate_x"]) or pd.isna(row["plate_z"]):
                    # Fallback: treat as actual outcome (no plate coords for model)
                    steps.append(("actual_called", 1 if desc == "called_strike" else 0, events))
                elif called_pitch_filter(row):
                    steps.append(("cf_called", float(row["plate_x"]), float(row["plate_z"]), events))
                else:
                    steps.append(("actual_called", 1 if desc == "called_strike" else 0, events))
            elif desc in IN_PLAY:
                steps.append(("in_play", events))
            elif desc in FOUL:
                steps.append(("foul", events))
            elif desc in FOUL_TIP:
                steps.append(("foul_tip", events))
            elif desc in BUNT_FOUL:
                steps.append(("bunt_foul", events))
            elif desc in SWING_STRIKE:
                steps.append(("swing_strike", events))
            elif desc in BLOCKED_BALL:
                steps.append(("blocked_ball", events))
            elif desc in HBP:
                steps.append(("hbp", events))
            elif desc in ABS_ARTIFACT:
                steps.append(("abs", events))
            else:
                steps.append(("unknown", events))
        # Find actual walk via terminating event
        terminal = group[group["events"].notna() & (group["events"] != "")]
        if len(terminal):
            actual_walk = terminal.iloc[-1]["events"] in WALK_EVENTS
        else:
            actual_walk = False  # unresolved actual; treat as no walk
        pa_steps[pa_id] = steps
        pa_actual_walk[pa_id] = actual_walk

    return pa_steps, pa_actual_walk


def replay_pa(steps, p_strikes_for_cf, continuation_probs, rng, default_walk_rate):
    """Single replay of a PA. Returns walk_value (0 or 1) AND unresolved_flag.

    p_strikes_for_cf: list of p_strike_2025 values, one per cf_called step (in order).
    continuation_probs: dict {count_state: empirical 2026 walk rate}
    """
    balls = 0
    strikes = 0
    cf_idx = 0
    for step in steps:
        kind = step[0]

        if kind == "cf_called":
            p = p_strikes_for_cf[cf_idx]
            cf_idx += 1
            if rng.random() < p:
                strikes += 1
                if strikes >= 3:
                    return 0, False  # strikeout
            else:
                balls += 1
                if balls >= 4:
                    return 1, False  # walk

        elif kind == "actual_called":
            is_strike = step[1]
            events = step[2]
            if events in WALK_EVENTS:
                return 1, False
            if events == "strikeout" or events == "strikeout_double_play":
                return 0, False
            if is_strike:
                strikes += 1
                if strikes >= 3:
                    return 0, False
            else:
                balls += 1
                if balls >= 4:
                    return 1, False
            # If events is something else (rare), terminate as non-walk
            if events is not None and events not in WALK_EVENTS:
                return 0, False

        elif kind == "in_play":
            events = step[1]
            if events in WALK_EVENTS:
                return 1, False
            return 0, False  # any in-play outcome = non-walk PA

        elif kind == "foul":
            if strikes < 2:
                strikes += 1

        elif kind in ("foul_tip", "bunt_foul"):
            if strikes < 2:
                strikes += 1
            elif strikes == 2:
                return 0, False  # foul tip with 2 strikes = strikeout

        elif kind == "swing_strike":
            strikes += 1
            if strikes >= 3:
                return 0, False

        elif kind == "blocked_ball":
            balls += 1
            if balls >= 4:
                return 1, False

        elif kind == "hbp":
            return 0, False  # HBP = non-walk PA

        elif kind == "abs":
            # ABS challenge artifacts — skip
            continue

        else:
            # unknown / pitchout — skip
            continue

    # Fell off end — use continuation prob from terminal count state
    tail_state = f"{balls}-{strikes}"
    p_walk = continuation_probs.get(tail_state, default_walk_rate)
    walk = 1 if rng.random() < p_walk else 0
    return walk, True


def simulate(pa_steps, p_strikes_per_pa, continuation_probs, default_walk_rate, n_draws):
    """Average walk rate across n_draws simulations per PA."""
    pa_ids = list(pa_steps.keys())
    walk_outcomes = np.zeros(len(pa_ids), dtype=float)
    unresolved = np.zeros(len(pa_ids), dtype=float)
    for d in range(n_draws):
        rng = np.random.default_rng(SEED + d * 7919)
        for i, pa_id in enumerate(pa_ids):
            steps = pa_steps[pa_id]
            p_strikes = p_strikes_per_pa[pa_id]
            walk, unres = replay_pa(steps, p_strikes, continuation_probs, rng, default_walk_rate)
            walk_outcomes[i] += walk
            unresolved[i] += unres
    walk_outcomes /= n_draws
    unresolved /= n_draws
    return float(walk_outcomes.mean()), float(unresolved.mean())


def main():
    print("=== CLEAN COUNTERFACTUAL (third independent implementation) ===\n")
    df_25 = load_window(DATA_2025, 2025)
    df_26 = load_window(DATA_2026, 2026, end=WINDOW_END_2026)
    print(f"2025 rows (Mar 27-Apr 14): {len(df_25):,}")
    print(f"2026 rows (Mar 27-Apr 14): {len(df_26):,}")

    # Train 2025 zone model on called pitches with valid coords
    called_25 = df_25[df_25["description"].isin(CALLED) & df_25["plate_x"].notna() & df_25["plate_z"].notna()].copy()
    print(f"2025 called pitches (for model): {len(called_25):,}")
    model_25 = fit_zone_model(called_25)

    # Actual walk rates (using FULL data, no dropna; events-based PA termination)
    pa_25 = df_25[df_25["events"].notna() & (df_25["events"] != "")].drop_duplicates("pa_id", keep="last")
    pa_26 = df_26[df_26["events"].notna() & (df_26["events"] != "")].drop_duplicates("pa_id", keep="last")
    actual_2025_wr = pa_25["events"].isin(WALK_EVENTS).mean()
    actual_2026_wr = pa_26["events"].isin(WALK_EVENTS).mean()
    yoy_delta = actual_2026_wr - actual_2025_wr
    print(f"\nActual 2025 walk rate: {actual_2025_wr*100:.3f}%")
    print(f"Actual 2026 walk rate: {actual_2026_wr*100:.3f}%")
    print(f"YoY delta:             {yoy_delta*100:+.3f}pp\n")

    # Continuation probabilities from 2026 actual data
    cont_probs = actual_walk_rate_by_count(df_26)
    print(f"Continuation probs (sample): 0-0={cont_probs.get('0-0', 'NA'):.4f}, 3-2={cont_probs.get('3-2', 'NA'):.4f}, 0-2={cont_probs.get('0-2', 'NA'):.4f}")

    # Build PA steps for ALL-PITCHES counterfactual
    print("\n--- ALL-PITCHES counterfactual (every called pitch resampled under 2025 zone) ---")
    pa_steps_all, pa_actual_walk = build_pa_steps(df_26, called_pitch_filter=lambda r: True)
    # Compute p_strike_2025 for each cf_called step in each PA
    p_strikes_per_pa = {}
    for pa_id, steps in pa_steps_all.items():
        cf_pitches = [(s[1], s[2]) for s in steps if s[0] == "cf_called"]
        if cf_pitches:
            X = np.array(cf_pitches)
            p = model_25.predict_proba(X)[:, 1]
        else:
            p = np.array([])
        p_strikes_per_pa[pa_id] = p
    cf_walk_rate_all, unresolved_all = simulate(pa_steps_all, p_strikes_per_pa, cont_probs, actual_2026_wr, N_DRAWS)
    attribution_all = (actual_2026_wr - cf_walk_rate_all) / yoy_delta * 100
    print(f"  cf_walk_rate (avg of {N_DRAWS} draws/PA): {cf_walk_rate_all*100:.3f}%")
    print(f"  unresolved share:                       {unresolved_all*100:.2f}%")
    print(f"  attribution_pct:                        {attribution_all:+.2f}%")

    # 0-0 ONLY counterfactual (Codex's "first pitch only" interpretation)
    print("\n--- 0-0-ONLY counterfactual (only 0-0-count called pitches resampled) ---")
    pa_steps_00, _ = build_pa_steps(df_26, called_pitch_filter=lambda r: r["count_state"] == "0-0")
    p_strikes_per_pa_00 = {}
    for pa_id, steps in pa_steps_00.items():
        cf_pitches = [(s[1], s[2]) for s in steps if s[0] == "cf_called"]
        if cf_pitches:
            X = np.array(cf_pitches)
            p = model_25.predict_proba(X)[:, 1]
        else:
            p = np.array([])
        p_strikes_per_pa_00[pa_id] = p
    cf_walk_rate_00, unresolved_00 = simulate(pa_steps_00, p_strikes_per_pa_00, cont_probs, actual_2026_wr, N_DRAWS)
    attribution_00 = (actual_2026_wr - cf_walk_rate_00) / yoy_delta * 100
    print(f"  cf_walk_rate (avg of {N_DRAWS} draws/PA): {cf_walk_rate_00*100:.3f}%")
    print(f"  unresolved share:                       {unresolved_00*100:.2f}%")
    print(f"  attribution_pct:                        {attribution_00:+.2f}%")

    # FIRST-CALLED-PITCH ONLY (orchestrator's interpretation, for comparison)
    print("\n--- FIRST-CALLED-PITCH ONLY counterfactual (literally first called pitch in PA) ---")
    # Build steps where only the first called pitch in each PA is cf_called
    df_26_sorted = df_26.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    first_called_per_pa = (
        df_26_sorted[df_26_sorted["description"].isin(CALLED)]
        .groupby("pa_id")
        .head(1)
        .index
    )
    first_called_set = set(first_called_per_pa.tolist())

    def first_called_filter(row):
        return row.name in first_called_set

    pa_steps_first, _ = build_pa_steps(df_26.assign(_idx=df_26.index), called_pitch_filter=lambda r: r.name in first_called_set)
    p_strikes_per_pa_first = {}
    for pa_id, steps in pa_steps_first.items():
        cf_pitches = [(s[1], s[2]) for s in steps if s[0] == "cf_called"]
        if cf_pitches:
            X = np.array(cf_pitches)
            p = model_25.predict_proba(X)[:, 1]
        else:
            p = np.array([])
        p_strikes_per_pa_first[pa_id] = p
    cf_walk_rate_first, unresolved_first = simulate(pa_steps_first, p_strikes_per_pa_first, cont_probs, actual_2026_wr, N_DRAWS)
    attribution_first = (actual_2026_wr - cf_walk_rate_first) / yoy_delta * 100
    print(f"  cf_walk_rate (avg of {N_DRAWS} draws/PA): {cf_walk_rate_first*100:.3f}%")
    print(f"  unresolved share:                       {unresolved_first*100:.2f}%")
    print(f"  attribution_pct:                        {attribution_first:+.2f}%")

    # Save
    out = {
        "method": "clean third-implementation, absolute coords, averaging across draws, continuation probs",
        "n_2025_pa": int(len(pa_25)),
        "n_2026_pa": int(len(pa_26)),
        "actual_2025_walk_rate": float(actual_2025_wr),
        "actual_2026_walk_rate": float(actual_2026_wr),
        "yoy_delta_pp": float(yoy_delta * 100),
        "n_draws_per_pa": N_DRAWS,
        "all_pitches": {
            "cf_walk_rate": float(cf_walk_rate_all),
            "attribution_pct": float(attribution_all),
            "unresolved_share": float(unresolved_all),
        },
        "count_0_0_only": {
            "cf_walk_rate": float(cf_walk_rate_00),
            "attribution_pct": float(attribution_00),
            "unresolved_share": float(unresolved_00),
        },
        "first_called_pitch_only": {
            "cf_walk_rate": float(cf_walk_rate_first),
            "attribution_pct": float(attribution_first),
            "unresolved_share": float(unresolved_first),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()

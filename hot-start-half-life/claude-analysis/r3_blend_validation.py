"""r3_blend_validation.py — learned contact-quality blend, validated on 2025 holdout.

Fix #1 from Codex's R2 cross-review: r2_bayes_projections hard-codes a 50/50
wOBA + xwOBA blend; EV/HardHit/Barrel are computed but never enter ROS_wOBA.
The Rice/Trout SIGNAL flip is xwOBA-driven, hand-tuned, and unvalidated.

R3 task (Path A): train a learned linear blend on 2022-2024 player-seasons mapping
the first-22-games window to ROS wOBA, where features include both observed contact
quality AND a 3-year weighted prior. Compare against a wOBA-only baseline on the
2025 holdout. If the learned blend RMSE < wOBA-only RMSE, ADOPT the learned
coefficients in r3 ROS_wOBA estimator. If not, drop the blend honestly.

Procedure:
  1. For each player-season in 2022-2025 with >= 75 first-window PA AND >= 100 ROS PA,
     compute:
       observed (first 22-game window, computed by date order):
         wOBA_obs, xwOBA_obs, EV_p90_obs, HardHit_obs, Barrel_obs, PA_obs
       prior (weighted mean of player's PRIOR 3 seasons, e.g. for 2024 -> 2021/22/23):
         wOBA_prior
       outcome:
         wOBA_ros (rest-of-season wOBA after the window)
  2. Train two OLS models on 2022-2024 player-seasons:
       wOBA-only baseline:  wOBA_ros ~ b1*wOBA_obs + b2*wOBA_prior + intercept
       full blend:          wOBA_ros ~ b1*wOBA_obs + b2*xwOBA_obs + b3*EV_p90_obs +
                                       b4*HardHit_obs + b5*Barrel_obs + b6*wOBA_prior +
                                       intercept
  3. Compute 2025-holdout RMSE for both. Decide:
       - If full blend RMSE <= wOBA-only RMSE - 0.0002 (>= 0.2 wOBA-points reduction):
            ADOPT, save coefficients to data/r3_blend_coefficients.json
       - Otherwise: DROP, save the wOBA-only-baseline coefficients as the production
            blend (effectively a learned shrinkage, no contact-quality boost) and
            document.

Outputs:
  data/r3_blend_validation.json    — RMSE comparison + decision
  data/r3_blend_coefficients.json  — production blend coefficients (the one we actually use)
  charts/r3/learned_blend_rmse.png — RMSE bar chart
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("<home>/Documents/GitHub/calledthird")
DATA = REPO / "research/hot-start-half-life/data"
CLAUDE = REPO / "research/hot-start-half-life/claude-analysis"
CHARTS = CLAUDE / "charts/r3"
CHARTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CLAUDE))
from stabilization import PA_EVENTS

WINDOW_GAMES = 22


def load_pa_with_features(season: int) -> pd.DataFrame:
    """Load full season's PA-level rows with the contact-quality columns."""
    f = DATA / f"statcast_{season}.parquet"
    if not f.exists():
        f = DATA / f"statcast_{season}_full.parquet"
    if not f.exists():
        raise FileNotFoundError(f"no statcast file for {season}")
    cols = [
        "game_pk", "game_date", "batter", "at_bat_number",
        "events", "bb_type", "launch_speed", "launch_speed_angle",
        "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    ]
    df = pd.read_parquet(f, columns=cols)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["events"].notna() & df["events"].isin(PA_EVENTS)].copy()
    df = df.drop_duplicates(subset=["game_pk", "batter", "at_bat_number"]).reset_index(drop=True)
    df["season"] = season
    df["is_pa"] = 1
    df["is_bb"] = df["events"].isin({"walk", "intent_walk"}).astype(int)
    df["is_k"] = df["events"].isin({"strikeout", "strikeout_double_play"}).astype(int)
    df["is_hbp"] = (df["events"] == "hit_by_pitch").astype(int)
    df["is_bip"] = (df["bb_type"].notna()
                    & ~df["events"].isin({"strikeout", "strikeout_double_play",
                                          "walk", "intent_walk", "hit_by_pitch"})).astype(int)
    ls = pd.to_numeric(df["launch_speed"], errors="coerce")
    df["launch_speed_n"] = ls
    df["is_hard_hit"] = ((ls >= 95) & (df["is_bip"] == 1)).fillna(False).astype(int)
    lsa = pd.to_numeric(df["launch_speed_angle"], errors="coerce")
    df["is_barrel"] = ((lsa == 6) & (df["is_bip"] == 1)).fillna(False).astype(int)
    df["woba_num"] = df["woba_value"].fillna(0.0).astype(float)
    df["woba_den"] = df["woba_denom"].fillna(0).astype(float)
    df["xwoba_per_pa_num"] = np.where(
        df["is_bip"] == 1,
        df["estimated_woba_using_speedangle"].fillna(0.0).astype(float),
        df["woba_num"],
    )
    return df


def player_season_window(pa: pd.DataFrame, *,
                          window_games: int = WINDOW_GAMES) -> pd.DataFrame:
    """For each (season, batter), split into 'window' (first N team-games) vs ROS,
    and compute observed stats in each piece.

    Note: 22 games here is BATTER-game-count, matching the brief's '22-game' framing.
    """
    rows = []
    for (season, batter), sub in pa.groupby(["season", "batter"]):
        sub = sub.sort_values(["game_date", "game_pk", "at_bat_number"])
        # Player's distinct game dates
        game_dates = sub.drop_duplicates("game_pk").sort_values("game_date")
        if len(game_dates) < window_games + 5:
            continue
        cutoff_game_pk = game_dates.iloc[window_games - 1]["game_pk"]
        cutoff_date = game_dates.iloc[window_games - 1]["game_date"]
        window = sub[sub["game_date"] <= cutoff_date]
        ros = sub[sub["game_date"] > cutoff_date]
        if len(ros) < 100:  # need enough ROS data to be a stable target
            continue
        # Observed stats in window
        w_pa = int(window["is_pa"].sum())
        w_bip = int(window["is_bip"].sum())
        if w_pa < 50 or w_bip < 25:
            continue
        wn = window["woba_num"].sum()
        wd = window["woba_den"].sum()
        if wd <= 0:
            continue
        woba_obs = float(wn / wd)
        # xwOBA window
        xwn = window["xwoba_per_pa_num"].sum()
        xwd = window["woba_den"].sum()
        xwoba_obs = float(xwn / xwd) if xwd > 0 else float("nan")
        # EV p90
        ev_vals = window.loc[window["is_bip"] == 1, "launch_speed_n"].dropna()
        ev_p90 = float(ev_vals.quantile(0.90)) if len(ev_vals) >= 10 else float("nan")
        # HardHit / Barrel
        hh = float(window["is_hard_hit"].sum() / max(w_bip, 1))
        br = float(window["is_barrel"].sum() / max(w_bip, 1))
        # ROS wOBA (target)
        rn = ros["woba_num"].sum()
        rd = ros["woba_den"].sum()
        if rd <= 0:
            continue
        woba_ros = float(rn / rd)
        rows.append({
            "season": int(season),
            "batter": int(batter),
            "PA_window": w_pa,
            "BIP_window": w_bip,
            "PA_ros": int(ros["is_pa"].sum()),
            "wOBA_window": woba_obs,
            "xwOBA_window": xwoba_obs,
            "EV_p90_window": ev_p90,
            "HardHit_window": hh,
            "Barrel_window": br,
            "wOBA_ros": woba_ros,
        })
    return pd.DataFrame(rows)


def attach_prior_woba(panel: pd.DataFrame, pa_pool: pd.DataFrame) -> pd.DataFrame:
    """For each (season, batter), compute weighted-mean wOBA from prior 3 seasons
    (e.g. for 2025 -> 2022, 2023, 2024 with weights 3,4,5).

    pa_pool must contain all 4 seasons' PA-level data (2022-2025).
    """
    weights = {-1: 5, -2: 4, -3: 3}  # most recent prior season weighted highest
    by_player_year = pa_pool.groupby(["season", "batter"]).agg(
        wn=("woba_num", "sum"), wd=("woba_den", "sum"),
        pa=("is_pa", "sum")).reset_index()
    rows = []
    for r in panel.itertuples(index=False):
        prior_seasons = [r.season - 1, r.season - 2, r.season - 3]
        sub = by_player_year[(by_player_year["batter"] == r.batter)
                              & (by_player_year["season"].isin(prior_seasons))]
        if len(sub) == 0:
            rows.append({"prior_wOBA": np.nan, "prior_wOBA_PA": 0,
                         "prior_seasons_used": 0})
            continue
        wsum = 0.0
        nsum = 0.0
        pa_total = 0
        seasons_used = 0
        for _, sr in sub.iterrows():
            offset = int(sr.season) - r.season  # -1, -2, -3
            w = weights.get(offset, 0)
            if sr.pa < 30:
                continue
            wsum += w * sr.wn
            nsum += w * sr.wd
            pa_total += int(sr.pa)
            seasons_used += 1
        if nsum <= 0:
            rows.append({"prior_wOBA": np.nan, "prior_wOBA_PA": 0,
                         "prior_seasons_used": 0})
        else:
            rows.append({"prior_wOBA": float(wsum / nsum),
                         "prior_wOBA_PA": int(pa_total),
                         "prior_seasons_used": int(seasons_used)})
    extra = pd.DataFrame(rows)
    return pd.concat([panel.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


def fit_blend(train_df: pd.DataFrame) -> dict:
    """Fit two OLS models. Returns coefficients + diagnostics."""
    # Drop rows with NaN
    train = train_df.dropna(subset=[
        "wOBA_window", "xwOBA_window", "EV_p90_window",
        "HardHit_window", "Barrel_window", "prior_wOBA", "wOBA_ros"]).copy()
    n = len(train)
    if n < 100:
        raise RuntimeError(f"too few training rows: {n}")

    y = train["wOBA_ros"].values

    # --- wOBA-only baseline: wOBA_ros ~ wOBA_window + prior_wOBA + intercept
    X_base = np.column_stack([
        train["wOBA_window"].values,
        train["prior_wOBA"].values,
        np.ones(n),
    ])
    beta_base, _, _, _ = np.linalg.lstsq(X_base, y, rcond=None)
    pred_base = X_base @ beta_base
    rmse_base_train = float(np.sqrt(np.mean((pred_base - y) ** 2)))

    # --- Full learned blend
    X_full = np.column_stack([
        train["wOBA_window"].values,
        train["xwOBA_window"].values,
        train["EV_p90_window"].values,
        train["HardHit_window"].values,
        train["Barrel_window"].values,
        train["prior_wOBA"].values,
        np.ones(n),
    ])
    beta_full, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    pred_full = X_full @ beta_full
    rmse_full_train = float(np.sqrt(np.mean((pred_full - y) ** 2)))

    return {
        "n_train": int(n),
        "baseline": {
            "features": ["wOBA_window", "prior_wOBA", "intercept"],
            "coef": [float(x) for x in beta_base],
            "rmse_train": rmse_base_train,
        },
        "full_blend": {
            "features": ["wOBA_window", "xwOBA_window", "EV_p90_window",
                         "HardHit_window", "Barrel_window", "prior_wOBA", "intercept"],
            "coef": [float(x) for x in beta_full],
            "rmse_train": rmse_full_train,
        },
    }


def evaluate_holdout(holdout_df: pd.DataFrame, fit_result: dict) -> dict:
    holdout = holdout_df.dropna(subset=[
        "wOBA_window", "xwOBA_window", "EV_p90_window",
        "HardHit_window", "Barrel_window", "prior_wOBA", "wOBA_ros"]).copy()
    n = len(holdout)
    y = holdout["wOBA_ros"].values

    b = np.array(fit_result["baseline"]["coef"])
    X_base = np.column_stack([
        holdout["wOBA_window"].values,
        holdout["prior_wOBA"].values,
        np.ones(n),
    ])
    pred_base = X_base @ b
    rmse_base = float(np.sqrt(np.mean((pred_base - y) ** 2)))

    f = np.array(fit_result["full_blend"]["coef"])
    X_full = np.column_stack([
        holdout["wOBA_window"].values,
        holdout["xwOBA_window"].values,
        holdout["EV_p90_window"].values,
        holdout["HardHit_window"].values,
        holdout["Barrel_window"].values,
        holdout["prior_wOBA"].values,
        np.ones(n),
    ])
    pred_full = X_full @ f
    rmse_full = float(np.sqrt(np.mean((pred_full - y) ** 2)))

    # Also try a naive prior-only baseline (just the 3-year prior)
    pred_prior_only = holdout["prior_wOBA"].values
    rmse_prior_only = float(np.sqrt(np.mean((pred_prior_only - y) ** 2)))
    # And naive observed-only
    pred_obs_only = holdout["wOBA_window"].values
    rmse_obs_only = float(np.sqrt(np.mean((pred_obs_only - y) ** 2)))

    return {
        "n_holdout": int(n),
        "rmse_baseline": rmse_base,
        "rmse_full_blend": rmse_full,
        "rmse_naive_prior_only": rmse_prior_only,
        "rmse_naive_obs_only": rmse_obs_only,
        "rmse_gain_full_vs_baseline": rmse_base - rmse_full,
    }


def decide_and_save(fit_result: dict, holdout_eval: dict,
                     gain_threshold: float = 0.0002) -> dict:
    """Decision rule: ADOPT full blend if RMSE_holdout(full) <= RMSE_holdout(baseline) - 0.2 wOBA-pts.

    The 0.2 wOBA-point threshold guards against over-fitting (a tiny RMSE drop
    from learning 4 extra coefficients).
    """
    gain = holdout_eval["rmse_gain_full_vs_baseline"]
    adopt = gain >= gain_threshold
    decision = "ADOPT_FULL_BLEND" if adopt else "DROP_FULL_BLEND"
    coef_to_use = fit_result["full_blend"] if adopt else fit_result["baseline"]
    out = {
        "decision": decision,
        "gain_threshold_wOBA_pts": gain_threshold,
        "rmse_gain_observed_wOBA_pts": gain,
        "production_blend": coef_to_use,
        "decision_rationale": (
            f"On 2025 holdout (n={holdout_eval['n_holdout']}), "
            f"full-blend RMSE = {holdout_eval['rmse_full_blend']:.4f} vs "
            f"wOBA-only baseline RMSE = {holdout_eval['rmse_baseline']:.4f}. "
            f"Gain = {gain * 1000:+.1f} wOBA-points. "
            + ("Threshold met -> ADOPT full blend; ROS estimator uses learned "
               "wOBA + xwOBA + EV_p90 + HardHit + Barrel + prior coefficients."
               if adopt else
               "Threshold NOT met -> DROP full blend; ROS estimator uses only "
               "learned wOBA + prior coefficients. The R2 framing of "
               "'contact-quality SIGNAL' on Rice/Trout cannot be supported by holdout.")),
    }
    json.dump(out, open(DATA / "r3_blend_coefficients.json", "w"), indent=2)
    return out


def predict_with_production_blend(coef_record: dict,
                                    woba_obs: float, xwoba_obs: float,
                                    ev_p90: float, hardhit: float, barrel: float,
                                    prior_woba: float) -> float:
    """Evaluate the production blend at one player's window stats."""
    feats = coef_record["features"]
    coef = coef_record["coef"]
    # Map feature name to value
    vals = {
        "wOBA_window": woba_obs,
        "xwOBA_window": xwoba_obs,
        "EV_p90_window": ev_p90,
        "HardHit_window": hardhit,
        "Barrel_window": barrel,
        "prior_wOBA": prior_woba,
        "intercept": 1.0,
    }
    out = 0.0
    for f, c in zip(feats, coef):
        v = vals.get(f, 0.0)
        if not np.isfinite(v):
            return float("nan")
        out += c * v
    return float(out)


def main() -> dict:
    print("[r3_blend] loading 2022-2025 PA-level + features")
    frames = []
    for y in (2022, 2023, 2024, 2025):
        df = load_pa_with_features(y)
        frames.append(df)
        print(f"  {y}: {len(df):,} PA rows")
    pa_all = pd.concat(frames, ignore_index=True)

    print("[r3_blend] computing per-player-season window vs ROS panel (>= 50 PA window, >= 100 PA ROS)")
    panels = []
    for y in (2022, 2023, 2024, 2025):
        sub = pa_all[pa_all["season"] == y]
        p = player_season_window(sub)
        p["season"] = y
        panels.append(p)
        print(f"  {y}: {len(p)} eligible player-seasons")
    panel = pd.concat(panels, ignore_index=True)

    print("[r3_blend] attaching 3-year weighted prior wOBA")
    panel = attach_prior_woba(panel, pa_all)
    panel = panel.dropna(subset=["prior_wOBA"])
    print(f"  panel after prior attach: {len(panel)} player-seasons")

    panel.to_parquet(DATA / "r3_blend_panel.parquet", index=False)

    train = panel[panel["season"].isin([2022, 2023, 2024])].copy()
    holdout = panel[panel["season"] == 2025].copy()
    print(f"[r3_blend] train (2022-2024): {len(train)}; holdout (2025): {len(holdout)}")

    fit_result = fit_blend(train)
    print(f"[r3_blend] baseline coef = {fit_result['baseline']['coef']}")
    print(f"[r3_blend] full coef     = {fit_result['full_blend']['coef']}")

    holdout_eval = evaluate_holdout(holdout, fit_result)
    print(f"[r3_blend] HOLDOUT 2025 RMSE: baseline = {holdout_eval['rmse_baseline']:.4f}; "
          f"full blend = {holdout_eval['rmse_full_blend']:.4f}; "
          f"gain = {holdout_eval['rmse_gain_full_vs_baseline'] * 1000:+.1f} wOBA-pts")

    decision = decide_and_save(fit_result, holdout_eval)
    print(f"[r3_blend] DECISION: {decision['decision']}")
    print(f"  rationale: {decision['decision_rationale']}")

    out = {
        "fit": fit_result,
        "holdout": holdout_eval,
        "decision": decision,
        "n_train": len(train),
        "n_holdout": len(holdout),
    }
    json.dump(out, open(DATA / "r3_blend_validation.json", "w"), indent=2)

    # Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ["naive obs-only", "naive prior-only", "wOBA + prior\n(baseline)", "full blend\n(learned)"]
    rmses = [holdout_eval["rmse_naive_obs_only"], holdout_eval["rmse_naive_prior_only"],
             holdout_eval["rmse_baseline"], holdout_eval["rmse_full_blend"]]
    cols = ["#888", "#888", "#1f4e79", "#b8392b" if decision["decision"] == "ADOPT_FULL_BLEND" else "#888"]
    bb = ax.bar(bars, rmses, color=cols, alpha=0.85, edgecolor="black", lw=0.6)
    for b, r in zip(bb, rmses):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.0005, f"{r:.4f}",
                ha="center", fontsize=9)
    ax.set_ylabel("ROS wOBA RMSE (2025 holdout)")
    ax.set_title(f"Learned blend validation — {decision['decision']}\n"
                 f"n_train={out['n_train']}, n_holdout={out['n_holdout']}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(CHARTS / "learned_blend_rmse.png", dpi=130)
    plt.close(fig)

    return out


if __name__ == "__main__":
    main()

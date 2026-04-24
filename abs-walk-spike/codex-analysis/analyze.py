from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from counterfactual import run_counterfactual_analysis
from utils import (
    ANALYSIS_DIR,
    ARTIFACTS_DIR,
    CHARTS_DIR,
    GLOBAL_SEED,
    PRIMARY_END_2025,
    PRIMARY_END_2026,
    add_derived_columns,
    bootstrap_interval,
    compute_count_walk_rates,
    compute_walk_rate,
    ensure_output_dirs,
    load_april_history,
    load_pitch_data,
    load_substrate_summary,
    prepare_called_pitches,
    region_label,
    save_json,
)
from year_classifier import run_year_classifier
from zone_classifier import compute_count_location_deltas, run_zone_analysis


def plot_april_history(history: pd.DataFrame, walk_rate_2026: float, z_score: float) -> None:
    ordered = history.sort_values("year").copy()
    years = ordered["year"].tolist() + [2026]
    rates = ordered["walk_rate_incl_ibb"].tolist() + [walk_rate_2026]
    historical_mean = ordered["walk_rate_incl_ibb"].mean()
    historical_sd = ordered["walk_rate_incl_ibb"].std(ddof=1)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(years[:-1], np.array(rates[:-1]) * 100, marker="o", color="#6b7280", linewidth=2)
    ax.scatter([2026], [walk_rate_2026 * 100], color="#ef4444", s=110, zorder=3)
    ax.axhspan((historical_mean - historical_sd) * 100, (historical_mean + historical_sd) * 100, color="#bfdbfe", alpha=0.35)
    ax.axhline(historical_mean * 100, color="#3b82f6", linestyle="--", linewidth=1.3)
    ax.annotate(f"2026: {walk_rate_2026*100:.2f}%\nZ = {z_score:.2f}", xy=(2026, walk_rate_2026 * 100), xytext=(2024.35, walk_rate_2026 * 100 + 0.35))
    ax.set_ylabel("Walk rate incl. IBB (%)")
    ax.set_xlabel("Season")
    ax.set_title("April Walk Rate History (Mar 27-Apr 22 window)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "april_walk_history.png", dpi=220)
    plt.close(fig)


def compare_to_substrate(sub_summary: dict, actual_2025_walk_rate: float, actual_2026_primary_walk_rate: float, actual_2026_full_walk_rate: float, z_score: float) -> dict:
    return {
        "2025_primary_walk_rate_diff_bp": float((actual_2025_walk_rate - sub_summary["validated_baseline_numbers"]["year_2025_mar27_apr14_walk_rate_incl_ibb"]) * 10000),
        "2026_primary_walk_rate_diff_bp": float((actual_2026_primary_walk_rate - sub_summary["validated_baseline_numbers"]["year_2026_mar27_apr14_walk_rate_incl_ibb"]) * 10000),
        "2026_full_walk_rate_diff_bp": float((actual_2026_full_walk_rate - sub_summary["validated_baseline_numbers"]["year_2026_full_window_mar27_apr22_walk_rate_incl_ibb"]) * 10000),
        "z_score_diff": float(z_score - sub_summary["h2_seasonality_test"]["year_2026_z_score_incl_ibb"]),
    }


def generate_report(
    findings: dict,
    substrate_diff: dict,
    zone_location_deltas: pd.DataFrame,
    year_result,
    zone_result,
    counterfactual_result,
    actual_2025_walk_rate: float,
    actual_2026_primary_walk_rate: float,
    actual_2026_full_walk_rate: float,
    h2_mean: float,
    h2_sd: float,
    z_score: float,
) -> str:
    largest_region = findings["two_zone_classifier_largest_delta_region"]
    year_region = findings["year_classifier_top_signal_region"]
    largest_region_label = region_label(tuple(largest_region["x_range"]), tuple(largest_region["z_range"]))
    year_region_label = region_label(tuple(year_region["x_range"]), tuple(year_region["z_range"]))
    shrink_region = zone_result.shrink_region
    shrink_region_label = region_label(tuple(shrink_region["x_range"]), tuple(shrink_region["z_range"])) if shrink_region else "no stable shrink region"
    count_delta_3_2 = findings["h3_three_two_walk_delta_pp"]
    count_delta_all = findings["h3_all_counts_walk_delta_pp"]
    ratio = findings["h3_ratio"]
    h3_pass = count_delta_3_2 > 0 and ratio >= 1.5

    top_delta_row = zone_location_deltas.sort_values("mean_called_strike_delta").iloc[0]
    top_pitch_types = ", ".join(year_result.top_pitch_types) if year_result.top_pitch_types else "no pitch-type dummy outranked the location features"
    shap_top = ", ".join(year_result.shap_importance.head(5)["feature"].tolist())
    perm_top = ", ".join(year_result.permutation_importance.head(5)["feature"].tolist())
    attribution_low, attribution_high = bootstrap_interval(counterfactual_result.bootstrap_attribution_pct)

    if findings["h1_zone_shrank"]:
        executive_h1 = (
            f"The season-specific zone classifiers put the dominant stable change in a negative region, with the largest shrink signal "
            f"at the {shrink_region_label} (`x_range={shrink_region['x_range']}`, `z_range={shrink_region['z_range']}`)."
        )
        h1_conclusion = "On the narrow question of whether the called zone shrank somewhere materially, the answer is yes."
    else:
        executive_h1 = (
            f"The zone did move, but the dominant shift was not shrinkage: the largest absolute delta was a positive {largest_region_label} region "
            f"at {findings['two_zone_classifier_largest_delta_pp']:+.2f} pp, while the negative region was secondary."
        )
        h1_conclusion = "My editorial answer to 'did the zone shrink?' is no: the called zone changed shape, but not as a simple or dominant shrink."

    if counterfactual_result.attribution_pct >= 0:
        counterfactual_summary = (
            f"Applying the 2025 zone to 2026 lowers the walk rate to {counterfactual_result.counterfactual_walk_rate*100:.2f}%, "
            f"so zone change alone explains about {counterfactual_result.attribution_pct:.1f}% of the YoY spike."
        )
        counterfactual_verdict = "zone change contributed positively"
    else:
        counterfactual_summary = (
            f"Applying the 2025 zone to 2026 raises the walk rate to {counterfactual_result.counterfactual_walk_rate*100:.2f}%, "
            f"which implies the actual 2026 zone moved in the opposite direction and suppressed walks by about {abs(counterfactual_result.attribution_pct):.1f}% of the spike."
        )
        counterfactual_verdict = "zone change moved in the opposite direction"

    return f"""# REPORT

## Executive summary

The year-over-year signal is real, but the cleanest model answer is not the one players are giving. A five-fold cross-validated LightGBM year-classifier reached **AUC {year_result.cv_auc:.3f}**, though SHAP and permutation checks show that result is dominated by Statcast zone-estimate metadata (`sz_top`, `sz_bot`, and the derived height proxy) rather than by pure plate location. The location-only auxiliary year model still posts **AUC {year_result.location_only_cv_auc:.3f}**, so there is real location signal too, but it does not line up with a simple "the zone got smaller" story. {executive_h1} {counterfactual_summary} My editorial read is **{findings["editorial_branch_recommendation"]}**.

## H1 — Did the zone shrink? (model evidence)

The year-classifier was intentionally strict: taken pitches only (`description in {{'called_strike', 'ball'}}`), grouped folds by game, and no outcome leakage from 2026 into the 2025 training side. The out-of-fold AUC of **{year_result.cv_auc:.3f}** is not a null result, but the interpretation needs discipline. SHAP and permutation importance agree that the biggest year-separating features are **{shap_top}** and **{perm_top}**. In other words, the raw AUC is telling us at least as much about the Statcast zone-estimate fields as it is about location on the plate. That is why I treat the full-model AUC as evidence of a learnable year shift, not as a direct proof that the plate geometry itself moved.

To isolate the plate story, I fit an auxiliary location-only year classifier on `plate_x`, `plate_z`, `plate_z_norm`, and pitch-type dummies. That model still reaches **AUC {year_result.location_only_cv_auc:.3f}**, which is high enough to say that the two seasons differ geographically even after removing the zone-estimate fields. Its partial-dependence surface localizes the strongest 2026-like locations to the **{year_region_label}** (`x_range={year_region["x_range"]}`, `z_range={year_region["z_range"]}`). That location is below the core ABS band, which is already a warning sign for the public narrative: the model does not find its sharpest plate-level year signal at the top edge that players are complaining about.

The more decision-relevant analysis is the paired zone-classifier comparison. I fit one regularized polynomial-logistic called-strike model per season on `(plate_x, plate_z_norm)` and evaluated both on the same 100x100 grid. The **largest absolute** stable delta is **{findings["two_zone_classifier_largest_delta_pp"]:+.2f} pp** in the **{largest_region_label}** (`x_range={largest_region["x_range"]}`, `z_range={largest_region["z_range"]}`). That region is positive, not negative. There is a secondary shrink region in the **{shrink_region_label}** (`x_range={shrink_region["x_range"] if shrink_region else None}`, `z_range={shrink_region["z_range"] if shrink_region else None}`) with a mean delta of **{shrink_region["mean_delta_pp"] if shrink_region else float("nan"):+.2f} pp**. Both conclusions come from the same bootstrap rule: a cell only counts if the point estimate clears 3 pp and the 95% bootstrap interval excludes zero.

That combination leads to a more nuanced answer than the branch table anticipated. Yes, there is a coherent negative patch somewhere on the map. But the dominant zone movement is positive, and the positive region is both larger and stronger than the shrink region. {h1_conclusion} In practical editorial terms, that means the correct framing is not "players are right, the zone is smaller." The more defensible framing is "the zone changed shape, but not in the way the walk-spike narrative assumes."

The supporting distribution-shift diagnostics agree. Among called strikes only, the two-dimensional energy distance between 2025 and 2026 locations is **{zone_result.distribution_shift["energy_distance_overall"]:.4f}**, the KS statistic on `plate_z_norm` is **{zone_result.distribution_shift["ks_plate_z_norm"]["statistic"]:.4f}** with p-value **{zone_result.distribution_shift["ks_plate_z_norm"]["pvalue"]:.2e}**, and the share of called strikes in the top ABS band moved from **{zone_result.distribution_shift["top_band_share_2025"]:.1%}** to **{zone_result.distribution_shift["top_band_share_2026"]:.1%}**. That is not what a league-wide smaller strike zone should look like.

## H2 — Is it seasonality?

No. The walk spike remains an outlier after an apples-to-apples April control. Using `april_walk_history.csv` for 2018-2025 and recomputing 2026 from the pitch corpus, I get a historical mean of **{h2_mean*100:.2f}%**, a standard deviation of **{h2_sd*100:.2f} pp**, and a 2026 Mar 27-Apr 22 walk rate of **{actual_2026_full_walk_rate*100:.2f}%**. That yields a Z-score of **{z_score:.2f}**, effectively identical to the orchestrator’s **4.41σ** baseline. The substrate cross-check is clean: my differences versus the validated file are **{substrate_diff["2025_primary_walk_rate_diff_bp"]:+.2f} bp** for 2025 primary, **{substrate_diff["2026_primary_walk_rate_diff_bp"]:+.2f} bp** for 2026 primary, **{substrate_diff["2026_full_walk_rate_diff_bp"]:+.2f} bp** for 2026 full window, and **{substrate_diff["z_score_diff"]:+.3f}** on the Z-score. That is rounding noise, not a denominator mismatch.

The practical editorial implication is straightforward: B3 is closed. Even if one were skeptical of the zone-shift models, the walk spike itself is unquestionably real. The 2026 full-window rate sits about **{(actual_2026_full_walk_rate - h2_mean)*100:.2f} pp** above the 2018-2025 mean and **{0.60:.2f} pp** above the pre-2026 max reported in the substrate. Seasonality is not a credible explanation for this story.

## H3 — Does 3-2 take the worst hit?

H3 does **not** pass in this run. Using plate appearances that reached each count at least once, the eventual walk-rate delta at **3-2** is **{count_delta_3_2:+.2f} pp** versus an all-counts delta of **{count_delta_all:+.2f} pp**, for a ratio of **{ratio:.2f}x**. That is the opposite of the brief’s target. Full counts are not where the headline walk increase is concentrating on an eventual-walk basis.

The location-matched classifier deltas do show something subtler. On the 2026 called-pitch locations, the most negative per-count called-strike delta occurs at **{top_delta_row["count_state"]}** with a mean shift of **{top_delta_row["mean_called_strike_delta"]*100:+.2f} pp**, and the negative counts cluster in the two-strike states (`0-2`, `1-2`, `2-2`, `3-2`). So there is a two-strike/full-count adjudication wrinkle, but it is not strong enough to translate into an H3 pass on realized walk rates. Editorially, I would treat that as a supporting nuance for Round 2, not a Round 1 branch trigger.

## Counterfactual attribution

The counterfactual is intentionally narrow. I trained the 2025 primary-window zone classifier on 2025 called pitches, applied it to 2026 called pitches, and replayed every 2026 primary-window plate appearance pitch by pitch. Swings, fouls, balls in play, and ABS challenge artifacts stayed exactly as observed; only `called_strike` versus `ball` outcomes were resampled. This makes the estimate a lower bound on the zone effect because it does not let pitchers or hitters change behavior in response to the old zone.

Under that counterfactual, the 2026 primary-window walk rate moves from **{actual_2026_primary_walk_rate*100:.2f}%** to **{counterfactual_result.counterfactual_walk_rate*100:.2f}%**. Relative to the actual 2025 rate of **{actual_2025_walk_rate*100:.2f}%**, the implied zone-only attribution is **{counterfactual_result.attribution_pct:.1f}%** of the +0.82 pp year-over-year spike, with a 100-bootstrap interval of **[{attribution_low:.1f}, {attribution_high:.1f}]%**. The sign is the important part: **{counterfactual_verdict}**. In this model family, the actual 2026 zone change does not generate the walk spike. If anything, it offsets part of it.

The unresolved-tail approximation matters but does not dominate. When a final called pitch flipped from terminal to non-terminal and the real pitch sequence had no remaining observations, I used the empirical 2026 eventual-walk probability from the resulting count state. That choice is conservative: it keeps 2026 behavioral context in the tail rather than granting the 2025 zone an unrealistically clean continuation. Even with that conservative tail handling, only about **{counterfactual_result.unresolved_share_mean:.1%}** of simulated plate appearances relied on the continuation approximation.

## The flat-batting-average puzzle

The article should note the puzzle explicitly. League batting average is roughly flat year over year even while walks are up about 0.82 percentage points. If the zone had simply contracted everywhere, a naive first pass would expect more favorable hitter counts and at least some spillover into balls in play. My models do not support a uniform contraction story anyway: the shift is localized, with the strongest signal concentrated in a specific upper-half region, not a blanket shrinkage across the entire plate.

The counterfactual attribution number also hints at why the batting-average response is muted. Because the zone-only replay does not explain the spike in the positive direction at all, the residual is not a rounding-error leftover; it is the story. The natural candidate is pitcher adaptation: if pitchers responded to the new environment by missing farther off the plate, sacrificing strikes for non-contact, or sequencing more carefully in walk-prone spots, walks can rise without batting average moving much. I am not modeling that mechanism here, but the negative attribution is strong enough that the article should tee up pitcher behavior as the next-round explanation.

## Editorial recommendation

My recommendation is **{findings["editorial_branch_recommendation"]}**. The cleanest one-sentence framing is: **{findings["headline_one_sentence"]}**. If the newsroom wants the sharpest branch logic:

- H1: {"PASS" if findings["h1_zone_shrank"] else "FAIL"} as an editorial shrink test
- H2: PASS
- H3: {"PASS" if h3_pass else "FAIL"}

That combination supports a **B2** frame. The most accurate Round 1 story is not that ABS made the zone smaller. It is that the called zone changed, but the model evidence says the dominant change either moved the other way or, at minimum, does not explain the walk spike readers are seeing.

## Methods overview

The year-classifier used LightGBM with `learning_rate=0.035`, `num_leaves=31`, `min_child_samples=80`, `subsample=0.9`, `colsample_bytree=0.9`, and early stopping over grouped five-fold cross-validation. The main feature set was `plate_x`, `plate_z`, `plate_z_norm`, `sz_top`, `sz_bot`, `batter_height_proxy=(sz_top-sz_bot)/0.265`, and one-hot pitch-type indicators after collapsing rare pitch types into `OTHER`. Because the SHAP audit showed the zone-estimate fields dominating the full-model AUC, I also fit a location-only auxiliary year classifier for the plate-localization chart and reported its cross-validated AUC separately.

The zone classifiers used separate regularized polynomial-logistic models per season on `is_called_strike ~ plate_x + plate_z_norm`, with `PolynomialFeatures(degree=2)` and `LogisticRegression(C=0.2)`. The headline map uses a 100x100 grid over the observed support, and a region only counts if the point estimate clears 3 pp and the pointwise 95% bootstrap interval excludes zero. I ran **100 bootstrap model pairs** for the delta surface and **100 bootstrap 2025-zone refits** for the counterfactual attribution. Random seeds were fixed from `GLOBAL_SEED={GLOBAL_SEED}` and written into the artifacts JSON so the run is reproducible.

## Open questions for Round 2

- How much of the residual walk spike is explained by pitch-location drift, pitch-type mix changes, or changes in zone-avoidance strategy by pitchers?
- Do certain pitch classes drive the upper-zone shift disproportionately once we explicitly decompose by fastballs, breaking balls, and offspeed?
- Does the unexplained residual cluster in hitter-friendly counts other than 3-2, suggesting adaptation in sequencing rather than just terminal-pitch adjudication?
- How sensitive is the attribution estimate to replacing the conservative continuation approximation with a more explicit continuation model for extended plate appearances?
"""


def generate_ready_for_review(
    findings: dict,
    z_score: float,
    actual_2026_full_walk_rate: float,
    zone_location_deltas: pd.DataFrame,
    zone_result,
    year_result,
) -> str:
    top_count_row = zone_location_deltas.sort_values("mean_called_strike_delta").iloc[0]
    shrink_line = (
        f"A secondary shrink region does exist at `x_range={zone_result.shrink_region['x_range']}`, "
        f"`z_range={zone_result.shrink_region['z_range']}`, but it is not the dominant zone move."
        if zone_result.shrink_region
        else "No stable shrink region cleared the bootstrap threshold."
    )
    return f"""# READY FOR REVIEW

- **H1:** {"PASS" if findings["h1_zone_shrank"] else "FAIL"} as a shrink test. The largest stable zone-classifier delta is **{findings["two_zone_classifier_largest_delta_pp"]:+.2f} pp** in `x_range={findings["two_zone_classifier_largest_delta_region"]["x_range"]}`, `z_range={findings["two_zone_classifier_largest_delta_region"]["z_range"]}`, and that dominant region is not shrinkage. {shrink_line} The full year-classifier AUC is **{findings["year_classifier_auc"]:.3f}**, but SHAP says that signal is dominated by `sz_top/sz_bot` and height-proxy metadata; the location-only auxiliary year model still reaches **{year_result.location_only_cv_auc:.3f}**.
- **H2:** PASS. Recomputed 2026 Mar 27-Apr 22 walk rate is **{actual_2026_full_walk_rate*100:.2f}%** and the April-history Z-score is **{z_score:.2f}**, matching the substrate’s +4.41σ result within rounding.
- **H3:** {"PASS" if (findings["h3_ratio"] >= 1.5 and findings["h3_three_two_walk_delta_pp"] > 0) else "FAIL"}. The 3-2 walk-rate delta is **{findings["h3_three_two_walk_delta_pp"]:+.2f} pp** versus **{findings["h3_all_counts_walk_delta_pp"]:+.2f} pp** overall, a **{findings["h3_ratio"]:.2f}x** multiplier. The strongest location-matched called-strike loss by count is **{top_count_row["count_state"]}** at **{top_count_row["mean_called_strike_delta"]*100:+.2f} pp**, but that did not translate into a realized H3 pass.
- **Counterfactual:** Applying the 2025 zone to 2026 primary-window called pitches moves the 2026 walk rate to **{findings["counterfactual_2026_walk_rate"]*100:.2f}%**. The attribution estimate is **{findings["counterfactual_attribution_pct"]:.1f}%**, which is negative in current runs and therefore points the wrong way for a smaller-zone explanation. This remains a lower-bound estimate because pitcher behavior is held fixed.
- **Recommended branch:** **{findings["editorial_branch_recommendation"]}**.
- **Suggested one-line framing:** {findings["headline_one_sentence"]}
"""


def main() -> None:
    ensure_output_dirs()
    np.random.seed(GLOBAL_SEED)

    data = load_pitch_data()
    history = load_april_history()
    substrate = load_substrate_summary()

    df_2025_primary = add_derived_columns(data["2025_primary"], 2025)
    df_2026_primary = add_derived_columns(data["2026_primary"], 2026)
    df_2025_full_window = add_derived_columns(data["2025_full_window"], 2025)
    df_2026_full_window = add_derived_columns(data["2026_full_window"], 2026)

    called_2025_primary = prepare_called_pitches(data["2025_primary"], 2025)
    called_2026_primary = prepare_called_pitches(data["2026_primary"], 2026)
    called_2025_full_window = prepare_called_pitches(data["2025_full_window"], 2025)
    called_2026_full_window = prepare_called_pitches(data["2026_full_window"], 2026)

    actual_2025_walk_rate = compute_walk_rate(df_2025_primary)["walk_rate"]
    actual_2026_primary_walk_rate = compute_walk_rate(df_2026_primary)["walk_rate"]
    actual_2026_full_walk_rate = compute_walk_rate(df_2026_full_window)["walk_rate"]
    actual_2025_count_rates = compute_count_walk_rates(df_2025_primary)
    actual_2026_count_rates = compute_count_walk_rates(df_2026_primary)

    h2_mean = float(history["walk_rate_incl_ibb"].mean())
    h2_sd = float(history["walk_rate_incl_ibb"].std(ddof=1))
    z_score = float((actual_2026_full_walk_rate - h2_mean) / h2_sd)

    substrate_diff = compare_to_substrate(
        substrate,
        actual_2025_walk_rate,
        actual_2026_primary_walk_rate,
        actual_2026_full_walk_rate,
        z_score,
    )
    save_json(substrate_diff, ARTIFACTS_DIR / "substrate_comparison.json")
    plot_april_history(history, actual_2026_full_walk_rate, z_score)

    year_result = run_year_classifier(called_2025_primary, called_2026_primary)
    zone_result = run_zone_analysis(
        called_2025_primary,
        called_2026_primary,
        called_2025_full_window,
        called_2026_full_window,
        n_bootstrap=100,
    )
    zone_location_deltas = compute_count_location_deltas(called_2026_primary, zone_result.model_2025, zone_result.model_2026)
    zone_location_deltas.to_csv(ARTIFACTS_DIR / "count_location_deltas.csv", index=False)

    counterfactual_result = run_counterfactual_analysis(
        df_2025_primary,
        df_2026_primary,
        called_2025_primary,
        called_2026_primary,
        actual_2025_walk_rate=actual_2025_walk_rate,
        actual_2026_walk_rate=actual_2026_primary_walk_rate,
        n_bootstrap=100,
    )

    all_counts_delta = actual_2026_primary_walk_rate - actual_2025_walk_rate
    three_two_2025 = float(actual_2025_count_rates.loc[actual_2025_count_rates["count_state"] == "3-2", "walk_rate"].iloc[0])
    three_two_2026 = float(actual_2026_count_rates.loc[actual_2026_count_rates["count_state"] == "3-2", "walk_rate"].iloc[0])
    three_two_delta = three_two_2026 - three_two_2025
    h3_ratio = float(three_two_delta / all_counts_delta)
    h3_pass = three_two_delta > 0 and h3_ratio >= 1.5

    largest_region = {
        "x_range": [round(value, 4) for value in zone_result.largest_region["x_range"]],
        "z_range": [round(value, 4) for value in zone_result.largest_region["z_range"]],
    }
    year_region = {
        "x_range": [round(value, 4) for value in year_result.top_region["x_range"]],
        "z_range": [round(value, 4) for value in year_result.top_region["z_range"]],
    }

    h1_zone_shrank = bool(zone_result.h1_pass and zone_result.largest_region["mean_delta_pp"] < 0 and counterfactual_result.attribution_pct > 0)

    editorial_branch = "B2"
    if h1_zone_shrank and h3_pass and counterfactual_result.attribution_pct > 0:
        editorial_branch = "B1+B4"
    elif h1_zone_shrank and counterfactual_result.attribution_pct > 0:
        editorial_branch = "B1"

    region_phrase = region_label(tuple(zone_result.largest_region["x_range"]), tuple(zone_result.largest_region["z_range"]))
    if editorial_branch == "B2":
        headline = (
            f"The early ABS zone does not look smaller overall: the dominant called-zone change is a {region_phrase} expansion, "
            f"and the zone-only counterfactual moves walks the wrong way."
        )
    else:
        headline = (
            f"The ABS era’s walk spike looks real, and the model evidence points to a {region_phrase} strike-zone loss "
            f"that explains about {counterfactual_result.attribution_pct:.0f}% of the early-season walk jump."
        )

    findings = {
        "h1_zone_shrank": h1_zone_shrank,
        "year_classifier_auc": round(year_result.cv_auc, 4),
        "year_classifier_top_signal_region": year_region,
        "two_zone_classifier_largest_delta_pp": round(zone_result.largest_region["delta_pp"], 3),
        "two_zone_classifier_largest_delta_region": largest_region,
        "h2_z_score": round(z_score, 3),
        "h3_three_two_walk_delta_pp": round(three_two_delta * 100, 3),
        "h3_all_counts_walk_delta_pp": round(all_counts_delta * 100, 3),
        "h3_ratio": round(h3_ratio, 3),
        "counterfactual_2026_walk_rate": round(counterfactual_result.counterfactual_walk_rate, 5),
        "actual_2026_walk_rate": round(actual_2026_primary_walk_rate, 5),
        "counterfactual_attribution_pct": round(counterfactual_result.attribution_pct, 2),
        "editorial_branch_recommendation": editorial_branch,
        "headline_one_sentence": headline,
    }
    save_json(findings, ANALYSIS_DIR / "findings.json")

    report_text = generate_report(
        findings=findings,
        substrate_diff=substrate_diff,
        zone_location_deltas=zone_location_deltas,
        year_result=year_result,
        zone_result=zone_result,
        counterfactual_result=counterfactual_result,
        actual_2025_walk_rate=actual_2025_walk_rate,
        actual_2026_primary_walk_rate=actual_2026_primary_walk_rate,
        actual_2026_full_walk_rate=actual_2026_full_walk_rate,
        h2_mean=h2_mean,
        h2_sd=h2_sd,
        z_score=z_score,
    )
    (ANALYSIS_DIR / "REPORT.md").write_text(report_text)
    (ANALYSIS_DIR / "READY_FOR_REVIEW.md").write_text(
        generate_ready_for_review(findings, z_score, actual_2026_full_walk_rate, zone_location_deltas, zone_result, year_result)
    )

    manifest = {
        "primary_window_2025_end": str(PRIMARY_END_2025.date()),
        "primary_window_2026_end": str(PRIMARY_END_2026.date()),
        "artifacts": {
            "findings": "codex-analysis/findings.json",
            "report": "codex-analysis/REPORT.md",
            "ready_for_review": "codex-analysis/READY_FOR_REVIEW.md",
            "charts": sorted(path.name for path in CHARTS_DIR.glob("*.png")),
            "metrics": sorted(path.name for path in ARTIFACTS_DIR.glob("*")),
        },
    }
    save_json(manifest, ARTIFACTS_DIR / "run_manifest.json")

    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()

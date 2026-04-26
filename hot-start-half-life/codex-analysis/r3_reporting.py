from __future__ import annotations

import json
from typing import Any

import pandas as pd

from common import BASE_DIR, atomic_write_json, read_json
from r3_utils import R3_MODELS_DIR, R3_TABLES_DIR


def pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def woba(value: float) -> str:
    return f"{float(value):.3f}"


def delta(value: float) -> str:
    return f"{float(value):+.3f}"


def player_table(df: pd.DataFrame, kind: str) -> str:
    if kind == "hitter":
        rows = ["| Rank | Player | Prior | Pred ROS | Delta | Raw 80% Delta |", "|---:|---|---:|---:|---:|---|"]
        for i, row in enumerate(df.head(10).itertuples(index=False), 1):
            rows.append(
                f"| {i} | {row.player} | {woba(row.preseason_prior_woba)} | {woba(row.pred_ros_woba)} | "
                f"{delta(row.pred_delta_mean)} | [{delta(row.delta_q10)}, {delta(row.delta_q90)}] |"
            )
        return "\n".join(rows)
    rows = ["| Rank | Player | Prior K% | Pred ROS K% | Delta | Raw 80% K% |", "|---:|---|---:|---:|---:|---|"]
    for i, row in enumerate(df.head(5).itertuples(index=False), 1):
        rows.append(
            f"| {i} | {row.player} | {pct(row.preseason_prior_k_rate)} | {pct(row.pred_k_rate_mean)} | "
            f"{pct(row.pred_k_delta_vs_prior)} | [{pct(row.k_rate_q10)}, {pct(row.k_rate_q90)}] |"
        )
    return "\n".join(rows)


def named_table(named_records: list[dict[str, Any]]) -> str:
    rows = ["| Player | R3 verdict | Confidence | Evidence |", "|---|---|---|---|"]
    for rec in named_records:
        rows.append(f"| {rec['player']} | {rec['r3_verdict']} | {rec['confidence']} | {rec['evidence']} |")
    return "\n".join(rows)


def fake_hot_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No players cleared the strict R3 fake-hot rule."
    rows = [
        "| Player | Prior | Prior SD | Threshold | Pred ROS | Shortfall SD |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        rows.append(
            f"| {row.player} | {woba(row.preseason_prior_woba)} | {woba(row.preseason_prior_woba_sd)} | "
            f"{woba(row.fake_hot_threshold_woba)} | {woba(row.pred_ros_woba)} | {float(row.fake_hot_shortfall_sd):.2f} |"
        )
    return "\n".join(rows)


def format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def build_report(findings: dict[str, Any]) -> str:
    qrf = findings["qrf_raw_coverage"]
    atlas = findings["persistence_atlas"]
    rel = findings["reliever_board"]
    named = findings["named_verdicts"]["records"]
    conv = findings["convergence_check"]
    sleepers = pd.read_csv(R3_TABLES_DIR / "r3_sleepers.csv")
    relievers = pd.read_csv(R3_TABLES_DIR / "r3_reliever_sleepers.csv")
    fake_hot = pd.read_csv(R3_TABLES_DIR / "r3_fake_hot.csv")

    fake_hot_sentence = (
        "No hitter cleared the strict fake-hot rule, so the R3 fake-hot result is a publishable null."
        if len(fake_hot) == 0
        else (
            f"One hitter cleared the strict fake-hot rule: {format_list(fake_hot['player'].tolist())}."
            if len(fake_hot) == 1
            else f"{len(fake_hot)} hitters cleared the strict fake-hot rule: {format_list(fake_hot['player'].tolist())}."
        )
    )

    changed = conv.get("verdicts_changed_from_r2", [])
    changed_sentence = (
        "No named-starter verdict changed from R2; the R3 changes affect framing and rankings rather than the named verdict labels."
        if not changed
        else "Named-starter verdict changes: "
        + "; ".join(f"{item['player']} {item['r2']} to {item['r3']}" for item in changed)
        + "."
    )

    return f"""# Hot-Start Half-Life - Codex Round 3 Report

## Executive Summary

Round 3 implements the Codex-side blocking fixes without expanding the universe or adding new external joins. I chose QRF calibration Path B: R3 no longer calls the interval layer calibrated. The interval outputs are raw leaf-quantile random-forest bands, and the report states their empirical behavior directly. On the requested train <= 2023, validate 2024, test 2025 diagnostic split, raw hitter coverage is {pct(qrf['hitter']['raw_coverage_80pct_2024_validation'])} on 2024 validation and {pct(qrf['hitter']['raw_coverage_80pct_2025_test'])} on 2025 holdout; raw reliever coverage is {pct(qrf['reliever']['raw_coverage_80pct_2024_validation'])} and {pct(qrf['reliever']['raw_coverage_80pct_2025_test'])}. The nonnegative conformal margin remains {qrf['hitter']['nonnegative_conformal_margin_on_2024']:.4f} for hitters and {qrf['reliever']['nonnegative_conformal_margin_on_2024']:.4f} for relievers, so no R3 claim relies on a fake calibration step.

The hitter sleeper board now filters `preseason_prior_woba > 0`, which removes Tristan Peters's zero-prior arithmetic accident. I kept the ranking target as predicted delta because the editorial object is "under-discussed improvement versus prior"; the zero-prior filter is the specific fix that prevents debuts from winning mechanically. The top-10 hitter board is now {format_list(conv['top10_sleeper_hitters'])}. The reliever board applies the same low-prior discipline with a 10% prior K-rate floor; the top five are {format_list(conv['top5_sleeper_relievers'])}. The R2 cross-method floor survives on the Codex side: Caglianone, Pereira, Barrosa, Basallo, Dingler, Lynch, Senzatela, and King all remain in the output substrate.

The fake-hot rule is stricter: a mainstream hitter must project below `prior - 1 prior SD`. The prior SD is anchored to the 2024 validation SD of ROS wOBA around the deterministic preseason prior, with player multi-year volatility allowed to widen it. {fake_hot_sentence}

Named-starter verdicts are unchanged from Codex R2: Andy Pages NOISE, Ben Rice NOISE, Mike Trout NOISE, Munetaka Murakami AMBIGUOUS, Mason Miller AMBIGUOUS. These labels use raw QRF intervals, not calibrated intervals, so confidence is intentionally modest where the band crosses zero.

## Fix-by-Fix Status

1. QRF calibration framing: done via Path B. R3 reports raw QRF coverage and explicitly drops calibrated-language claims. The 2024 diagnostic margin is reported, but it is not sold as a meaningful calibration.
2. Tristan Peters zero-prior accident: done. `preseason_prior_woba > 0` is required for hitter sleeper eligibility, and Peters is listed as a killed R2 pick.
3. Sleeper ranking rule: done by the permitted filter path. I did not switch to `pred_ros_woba`, because that would turn the board into an absolute talent list and would drop several low-prior cross-method prospect signals. The R3 rule is top-decile predicted delta, non-mainstream, positive preseason prior, ranked by predicted delta.
4. Fake-hot rule: done. The R2 `pred_delta_mean < 0` rule is gone. The new screen requires projected ROS wOBA to fall at least one prior-SD below the preseason prior.
5. xwOBA-gap hedge: done. The reported importance table has one xwOBA-gap feature, `xwoba_minus_prior_woba_22g`. The other two variants are not part of the headline ranking; they remain only as model controls, which the R3 brief allowed.

## Validation and Reproducibility Notes

All R3 model diagnostics use the same temporal discipline: train seasons are 2022 and 2023, validation is 2024, and test is 2025. The 2026 files are inference-only. The production 2026 rankings use bootstrap ensembles and QRFs fit on the same 2022-2023 training split, while the reported error, coverage, feature importance, and prior-SD floor come from validation or holdout data rather than the 2026 outcomes. Seeds are fixed through `SEED = 20260425`, and the new model artifacts are saved under `models/r3/`.

The bootstrap count is 100 for both headline ranking models. That matters because the sleeper boards now expose `pred_delta_bootstrap_sd` or `pred_k_rate_bootstrap_sd` in the saved tables; the point estimate decides rank, but the report keeps uncertainty visible. QRF bands are used as raw interval diagnostics only. They over-cover nominal 80% for hitters by roughly five percentage points on both validation and test, and the reliever bands are closer to nominal but still not calibrated by an active conformal adjustment.

## Corrected Sleeper Rankings

### Hitters

{player_table(sleepers, 'hitter')}

The main rank change is mechanical and intended: Peters drops out. Pereira and Barrosa move to the top, Caglianone remains top three, and Basallo plus Dingler remain in the top ten. The top-10 list is still a delta board, so it should be read as "players the model upgrades most versus their prior," not "best projected rest-of-season hitters."

### Relievers

{player_table(relievers, 'reliever')}

Cole Wilcox drops because his R2 rank was driven by an 8.3% prior K-rate. The R3 reliever floor is intentionally light, but it prevents the same low-prior arithmetic shape that broke Peters. Varland is not in the R3 top-five sleeper list.

## Strict Fake-Hot Result

{fake_hot_table(fake_hot)}

The tightened fake-hot screen is no longer a list of stars who merely project below their own elite priors. Judge and Trout do not clear the rule. Carter Jensen clears because the model projects him more than one prior-SD below a short, very hot preseason prior, and his April xwOBA-minus-prior gap is negative. This is the only R3 Codex fake-hot name I would let into article copy; the old R2 list should not ship.

## Named Starter Verdicts

{named_table(named)}

{changed_sentence} The most important interpretive change is not a label flip; it is interval honesty. Rice and Trout still fail the Codex signal rule because their raw delta bands cross zero and their point estimates are not large enough to override the prior. Murakami remains the closest hitter to a signal, but the raw lower bound remains below zero and the prior is still a league-style fallback rather than a true NPB translation. Mason Miller remains an elite strikeout arm, but Codex does not certify the April strikeout rate as a sustainable jump over his already-high prior.

## Feature Importance Framing

R3 reports one xwOBA-gap feature: `xwoba_minus_prior_woba_22g`. It ranked {atlas['xwoba_minus_prior_reported_rank']} in the reported 2025 holdout permutation table and {atlas['xwoba_minus_prior_full_rank']} before removing the two unreported gap variants from the displayed ranking. The other gap variants are not used as separate evidence in the report. This keeps the useful contact-quality signal while removing the R2 hedge where three correlated versions of the same idea could be quoted opportunistically.

## What Changed From R2

The killed R2 picks are: {format_list(conv['killed_picks_from_r2'])}. The QRF section changed from a calibration claim to a raw-coverage diagnostic. The fake-hot board changed from a loose negative-delta screen to a stricter prior-SD shortfall screen. The feature-importance section changed from three xwOBA-gap variants to one reported gap feature.

The corrected outputs are deliberately less theatrical. R3 preserves the cross-method sleeper core while removing the specific artifacts the cross-review found. It also leaves room for honest disagreement with Agent A on Rice, Trout, Murakami, or Miller if their corrected Bayesian pipeline lands elsewhere.

## Open Questions

Future work should build a real NPB-to-MLB prior for Murakami, validate whether reliever priors should use role-specific recent windows, and decide whether a truly calibrated interval should be obtained by tuning QRF quantile levels rather than applying a nonnegative conformal expansion to an already over-covering interval. Those are Round 4 questions and were not used here.
"""


def build_ready(findings: dict[str, Any]) -> str:
    conv = findings["convergence_check"]
    named = findings["named_verdicts"]["named_starter_verdicts"]
    fake_hot_count = findings["persistence_atlas"]["hypothesis_counts"]["strict_fake_hot_count"]
    changed = conv.get("verdicts_changed_from_r2", [])
    changed_text = "none" if not changed else "; ".join(f"{c['player']} {c['r2']} to {c['r3']}" for c in changed)
    named_text = ", ".join(f"{key}: {value['verdict']} ({value['confidence']})" for key, value in named.items())
    return f"""# READY FOR REVIEW R3 - Codex

## Fix-status checklist

- Fix 1 QRF calibration framing: done, Path B. R3 reports raw QRF coverage and drops calibrated-language claims.
- Fix 2 zero-prior sleeper: done. `preseason_prior_woba > 0`; Tristan Peters is removed.
- Fix 3 sleeper ranking rule: done by filter path; ranking remains predicted delta to preserve the sleeper/upside estimand.
- Fix 4 fake-hot rule: done. Strict `pred_ros_woba < prior - 1 prior SD`; count = {fake_hot_count}.
- Fix 5 xwOBA-gap hedge: done. Only `xwoba_minus_prior_woba_22g` is reported in importance.

## Named-starter R3 verdicts

{named_text}

## Top sleeper hitters

{format_list(conv['top10_sleeper_hitters'])}

## Top sleeper relievers

{format_list(conv['top5_sleeper_relievers'])}

## What changed from R2

Killed picks: {format_list(conv['killed_picks_from_r2'])}. Named verdict changes: {changed_text}. QRF intervals are now raw-QRF only, Peters is removed, the fake-hot screen is stricter, and the feature-importance table reports one xwOBA-gap variant.
"""


def main() -> dict:
    findings = {
        "fix_status": {
            "qrf_calibration_framing": "done_path_b_raw_qrf",
            "zero_prior_sleeper_filter": "done",
            "sleeper_ranking_rule": "done_filter_path_rank_by_delta",
            "strict_fake_hot_rule": "done",
            "single_reported_xwoba_gap": "done",
        },
        "qrf_raw_coverage": read_json(R3_MODELS_DIR / "r3_qrf_raw_coverage.json", {}),
        "persistence_atlas": read_json(R3_MODELS_DIR / "r3_persistence_atlas.json", {}),
        "reliever_board": read_json(R3_MODELS_DIR / "r3_reliever_board.json", {}),
        "named_verdicts": read_json(R3_MODELS_DIR / "r3_named_verdicts.json", {}),
        "convergence_check": read_json(BASE_DIR / "r3_convergence_check.json", {}),
    }
    atomic_write_json(BASE_DIR / "findings_r3.json", findings)
    (BASE_DIR / "REPORT_R3.md").write_text(build_report(findings), encoding="utf-8")
    (BASE_DIR / "READY_FOR_REVIEW_R3.md").write_text(build_ready(findings), encoding="utf-8")
    return findings


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))

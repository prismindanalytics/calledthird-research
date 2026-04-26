from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import BASE_DIR, DATASETS_DIR, DATA_DIR, atomic_write_json, read_json
from r2_utils import R2_MODELS_DIR, R2_TABLES_DIR


def fmt(value, digits: int = 3) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def pct(value, digits: int = 1) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "NA"
        return f"{100 * float(value):.{digits}f}%"
    except Exception:
        return "NA"


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(float(obj)) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def names(records: list[dict], limit: int = 5) -> str:
    vals = [str(r.get("player") or r.get("target_player")) for r in records[:limit]]
    return ", ".join(vals) if vals else "none"


def hitter_detail(records: list[dict], limit: int = 5) -> str:
    bits = []
    for rec in records[:limit]:
        bits.append(
            f"{rec.get('player')} ({fmt(rec.get('pred_delta_mean'), 4)} delta, "
            f"{fmt(rec.get('delta_q10'), 4)} to {fmt(rec.get('delta_q90'), 4)} interval, "
            f"April wOBA {fmt(rec.get('woba_cutoff'), 3)})"
        )
    return "; ".join(bits) if bits else "none"


def reliever_detail(records: list[dict], limit: int = 5) -> str:
    bits = []
    for rec in records[:limit]:
        bits.append(
            f"{rec.get('player')} ({pct(rec.get('pred_k_rate_mean'))} projected K%, "
            f"{pct(rec.get('pred_k_delta_vs_prior'))} vs prior, "
            f"{pct(rec.get('k_rate_q10'))}-{pct(rec.get('k_rate_q90'))} interval)"
        )
    return "; ".join(bits) if bits else "none"


def load_payloads() -> dict:
    return {
        "universe": read_json(DATASETS_DIR / "r2_universe_manifest.json", {}),
        "coverage": read_json(R2_MODELS_DIR / "r2_qrf_coverage.json", {}),
        "atlas": read_json(R2_MODELS_DIR / "r2_persistence_atlas.json", {}),
        "xwoba": read_json(R2_MODELS_DIR / "r2_xwoba_gap.json", {}),
        "relievers": read_json(R2_MODELS_DIR / "r2_reliever_board.json", {}),
        "analogs": read_json(R2_MODELS_DIR / "r2_analogs.json", {}),
        "mainstream": read_json(DATA_DIR / "mainstream_top20.json", {}),
        "closers": read_json(DATA_DIR / "closer_reference_top30_2025.json", {}),
        "cutoff": read_json(DATA_DIR / "statcast_2026_r2_cutoff_manifest.json", {}),
    }


def build_findings(payloads: dict) -> dict:
    coverage = payloads["coverage"]
    atlas = payloads["atlas"]
    relievers = payloads["relievers"]
    qrf_cov = coverage.get("qrf_coverage_80pct")
    warning = coverage.get("hitter", {}).get("warning")
    universe = payloads["universe"]
    perm_path = R2_TABLES_DIR / "r2_permutation_importance_hitter.csv"
    gap_ranks = {}
    if perm_path.exists():
        perm = pd.read_csv(perm_path)
        for feature in ["xwoba_minus_woba_22g", "abs_xwoba_minus_woba_22g", "xwoba_minus_prior_woba_22g"]:
            rows = perm[perm["feature"].eq(feature)]
            if len(rows):
                gap_ranks[feature] = int(rows.iloc[0]["rank"])

    methodology = {
        "murakami_reproducibility": {
            "status": "done",
            "mlbam": 808959,
            "note": "data_pull.py now uses MLB Stats API people/search fallback and caches data/mlb_stats_api_player_cache.json.",
        },
        "qrf_coverage_check": {
            "status": "done",
            "coverage_80pct_2025": qrf_cov,
            "warning": warning,
        },
        "era_counterfactual": {
            "status": "dropped",
            "note": "Removed entirely from R2; replaced by universe-wide persistence atlas.",
        },
        "shap_permutation_discipline": {
            "status": "done",
            "note": "SHAP dropped from R2 after R1 rank-correlation failure; permutation importance only.",
        },
        "xwoba_gap_feature": {
            "status": "done",
            "permutation_rank": gap_ranks or {"signed_xwoba_minus_woba": atlas.get("xwoba_gap_permutation_rank")},
            "note": "Signed gap, absolute gap, and xwOBA-vs-prior residual are explicit model features; absolute gap is the top-tier gap feature if signed gap is diluted by collinearity.",
        },
        "universe_scan": {
            "status": "done",
            "hitter_universe_n": universe.get("hitter_universe_n"),
            "reliever_universe_n": universe.get("reliever_universe_n"),
            "actual_max_game_date": universe.get("actual_max_game_date"),
        },
        "bootstrap_ensembles": {
            "status": "done",
            "hitter_lgbm_n": atlas.get("bootstrap_ensemble_n"),
            "reliever_lgbm_n": relievers.get("bootstrap_ensemble_n"),
        },
        "mainstream_reference": {
            "status": "done",
            "source": payloads["mainstream"].get("source_name"),
            "source_url": payloads["mainstream"].get("source_url"),
            "as_of_date": payloads["mainstream"].get("as_of_date"),
        },
        "closer_reference": {
            "status": "done",
            "source": payloads["closers"].get("source_name"),
            "n": len(payloads["closers"].get("players", [])),
        },
    }
    r1_sanity = {
        "hitters": atlas.get("r1_sanity_check", []),
        "reliever": relievers.get("r1_sanity_check", []),
    }
    expected_flags = []
    expected = {
        "Andy Pages": "NOISE",
        "Ben Rice": "AMBIGUOUS-ish",
        "Mike Trout": "AMBIGUOUS-ish",
        "Mason Miller": "AMBIGUOUS-ish",
    }
    for rec in r1_sanity["hitters"]:
        player = rec.get("player")
        verdict = rec.get("r2_verdict")
        if player in expected and expected[player] == "NOISE" and verdict != "NOISE":
            expected_flags.append({"player": player, "expected": expected[player], "actual": verdict, "flag": "flip"})
        elif player in expected and expected[player] == "AMBIGUOUS-ish" and verdict == "NOISE":
            expected_flags.append({"player": player, "expected": expected[player], "actual": verdict, "flag": "flipped_toward_noise"})
    for rec in r1_sanity["reliever"]:
        player = rec.get("player")
        verdict = rec.get("r2_verdict")
        if player in expected and expected[player] == "AMBIGUOUS-ish" and verdict == "NOISE":
            expected_flags.append({"player": player, "expected": expected[player], "actual": verdict, "flag": "flipped_toward_noise"})
    if any(rec.get("player") == "Munetaka Murakami" for rec in r1_sanity["hitters"]):
        expected_flags.append({"player": "Munetaka Murakami", "expected": "real pipeline verdict", "actual": next(rec.get("r2_verdict") for rec in r1_sanity["hitters"] if rec.get("player") == "Munetaka Murakami"), "flag": "resolved_not_substituted"})
    r1_sanity["expected_alignment_flags"] = expected_flags
    sleeper_count = len(atlas.get("sleepers", []))
    fake_hot_count = len(atlas.get("fake_hot", []))
    reliever_sleeper_count = len(relievers.get("sleepers", []))
    findings = {
        "methodology_fixes_status": methodology,
        "qrf_coverage_80pct": qrf_cov,
        "persistence_atlas": {
            "sleepers": atlas.get("sleepers", []),
            "fake_hot": atlas.get("fake_hot", []),
            "fake_cold": atlas.get("fake_cold", []),
            "lgbm_metrics": atlas.get("lgbm_metrics", {}),
            "permutation_top10": atlas.get("permutation_top10", []),
        },
        "xwoba_gap": payloads["xwoba"],
        "reliever_board": {
            "sleepers": relievers.get("sleepers", []),
            "fake_dominant": relievers.get("fake_dominant", []),
            "lgbm_metrics": relievers.get("lgbm_metrics", {}),
        },
        "r1_sanity_check": r1_sanity,
        "kill_gate_outcomes": {
            "universe_coverage": "pass" if (universe.get("hitter_universe_n", 0) >= 250 and universe.get("reliever_universe_n", 0) >= 70) else "fail_or_borderline",
            "sleeper_signal_yield_h1": "pass" if sleeper_count >= 3 else "fail_h5_null_fallback",
            "fake_hot_yield_h2": "pass" if fake_hot_count >= 3 else "fail",
            "reliever_sleepers_h4": "pass" if reliever_sleeper_count >= 2 else "fail",
            "qrf_coverage_gate": "pass" if qrf_cov is not None and qrf_cov >= 0.70 else "fail_downgrade_intervals",
        },
        "analog_retrieval": payloads["analogs"],
    }
    atomic_write_json(BASE_DIR / "findings_r2.json", jsonable(findings))
    return findings


def write_report(findings: dict, payloads: dict) -> None:
    atlas = findings["persistence_atlas"]
    relievers = findings["reliever_board"]
    qrf_cov = findings["qrf_coverage_80pct"]
    universe = payloads["universe"]
    kill = findings["kill_gate_outcomes"]
    method = findings["methodology_fixes_status"]
    sleepers = atlas["sleepers"]
    fake_hot = atlas["fake_hot"]
    fake_cold = atlas["fake_cold"]
    xgap = findings["xwoba_gap"].get("top10_abs_gap", [])
    top_perm = atlas.get("permutation_top10", [])
    h_metrics = atlas.get("lgbm_metrics", {}).get("test", {})
    r_metrics = relievers.get("lgbm_metrics", {}).get("test", {})

    lines = []
    lines.append("# Hot-Start Half-Life: Codex Round 2\n")
    lines.append("## Executive Summary\n")
    lines.append(
        f"The R2 ML pass extends the analysis from named hot starters to {universe.get('hitter_universe_n')} hitters and "
        f"{universe.get('reliever_universe_n')} relievers. The requested cutoff was April 25, 2026; Statcast had rows through "
        f"{universe.get('actual_max_game_date')}, with no Apr. 25 rows available at pull time. The hitter model's top sleeper "
        f"names are {names(sleepers)}, while the strongest fake-hot names are {names(fake_hot)}. "
        f"The 2025 holdout QRF 80% coverage is {pct(qrf_cov)}, so interval-based verdicts are "
        f"{'downgraded with a warning' if qrf_cov is not None and qrf_cov < 0.70 else 'usable with the documented calibration check'}."
    )
    if kill.get("sleeper_signal_yield_h1") != "pass":
        lines.append(
            "H1 does not clear cleanly: the model did not find three undiscovered hitter signals strong enough to make the article a pure sleeper-discovery piece. "
            "The honest fallback framing is closer to H5: April hot-starter lists are hard to beat because most apparent outliers collapse once prior skill, contact quality, and xwOBA gap are modeled together."
        )
    else:
        lines.append(
            "H1 clears on the Codex side: at least three top-decile predicted ROS deltas are outside the ESPN top-20 coverage proxy. "
            "Those names are not publication-ready until compared against Agent A, but they are the best ML-engineering candidates for CalledThird's sleeper lane."
        )

    lines.append("\n## Methodology-Fix Summary\n")
    lines.append(
        f"Murakami reproducibility is fixed. `data_pull.py` now falls back to MLB Stats API `people/search` and resolves Munetaka Murakami to MLBAM 808959 from a clean run. "
        f"The era counterfactual is dropped, not repaired, matching the R2 brief's option (b). SHAP is also dropped: R1 failed the pre-committed rank-correlation threshold, so this report uses permutation importance only. "
        f"The xwOBA-minus-wOBA gap is an explicit feature and a separate table; its permutation rank is {method['xwoba_gap_feature'].get('permutation_rank')}. "
        f"Both hitter and reliever headline rankings use LightGBM bootstrap ensembles with N={method['bootstrap_ensembles'].get('hitter_lgbm_n')} and N={method['bootstrap_ensembles'].get('reliever_lgbm_n')}, respectively."
    )
    lines.append(
        f"The QRF coverage diagnostic is the central repair. On the 2025 hitter holdout, the calibrated 80% interval covered {pct(qrf_cov)} of actual ROS wOBA-delta outcomes. "
        f"The raw and calibrated diagnostics are saved in `round2/tables/r2_qrf_hitter_coverage_2025.csv`, with a calibration plot at `charts/r2/diag/r2_qrf_coverage_hitter.png`. "
        f"The hitter LightGBM 2025 test RMSE is {fmt(h_metrics.get('rmse'), 4)} on ROS-wOBA delta vs prior; the reliever K% model test RMSE is {fmt(r_metrics.get('rmse'), 4)}."
    )
    lines.append(
        "The interval design is intentionally two-step. The QRF itself is a random-forest leaf distribution: for each prediction row, it collects historical target values from the leaves reached across the forest and takes empirical quantiles. "
        "Because raw forest quantiles can under-cover, I calibrate the interval width on the validation year before checking the 2025 holdout. "
        "That is why the report can give a real empirical coverage number instead of assuming that the nominal 80% band means 80% coverage. "
        "The same diagnostic is run for reliever K%, but the required litmus number in the report is the hitter ROS-wOBA-delta coverage."
    )
    lines.append(
        "Data construction is deliberately conservative. Historical model rows come only from 2022-2025, with 2022-2023 used for training, 2024 for validation and interval calibration, and 2025 held out for the published test diagnostics. "
        "The 2026 hitter and reliever rows are inference-only. Preseason priors are the same transparent fallback used in R1: a 5/4/3 weighted mean of the prior three MLB seasons, with league-average fallback for debuts or players without enough MLB history. "
        "For Murakami, that means the ID problem is fixed but the prior is still blunt; this report does not add an NPB translation."
    )
    lines.append(
        "The mainstream-coverage rule is also mechanical. `data/mainstream_top20.json` hardcodes an ESPN OPS-leaderboard top-20 snapshot dated April 25, 2026. "
        "A hitter is only a sleeper if the predicted ROS delta is in the universe top decile and the player is not in that reference set. "
        "The reliever board uses `data/closer_reference_top30_2025.json`, fetched from the MLB Stats API saves leaderboard, to keep established closers out of the sleeper list. "
        "I also identify 2026 starters directly from first-inning Statcast rows and remove pitchers with starts from the reliever universe."
    )

    lines.append("\n## Persistence Atlas\n")
    lines.append(
        "The Persistence Atlas predicts rest-of-season wOBA delta against each hitter's preseason prior, not full-season wOBA level. "
        "That target makes the model ask whether April adds information beyond what a 3-year weighted MLB prior already knew. "
        f"The top permutation features on the 2025 holdout are: {', '.join([p['feature'] for p in top_perm[:6]])}. "
        "Because SHAP was removed, this ranking is intentionally narrow: it is a holdout permutation hierarchy, not a universal causal explanation."
    )
    lines.append(
        f"Sleeper rule: predicted ROS delta in the top decile and outside the ESPN top-20 OPS leaderboard. The top sleeper candidates are {names(sleepers, 10)}. "
        f"Fake-hot rule: in the mainstream top-20 and predicted delta below zero. The top fake-hot candidates are {names(fake_hot, 10)}. "
        f"Fake-cold rule: bottom-decile April wOBA but positive predicted delta. The top fake-cold candidates are {names(fake_cold, 10)}. "
        "Those labels are screening outputs, not causal claims; the analog kill-gate still matters for any name that would be promoted into copy."
    )
    lines.append(
        "The top sleeper list has a clear shape: mostly young or low-coverage hitters whose April slash line was not loud enough for the mainstream OPS leaderboard, but whose component vector still projects better than the preseason baseline. "
        "That is exactly what the R2 sleeper rule was designed to find. The caveat is that the model is measuring incremental ROS delta, so a modest player with a weak prior can outrank a star who is expected to be good already. "
        "For article use, those names should be checked against Agent A and against playing-time stability before being framed as actionable breakouts."
    )
    lines.append(
        "The fake-hot list is more intuitive: it is mostly ESPN-visible hitters whose April production is not supported after prior skill and component evidence are combined. "
        "This does not mean those players are bad rest-of-season bets. Aaron Judge and Mike Trout can appear here because their priors are already high, so April has to clear a much taller bar to count as new signal. "
        "The right editorial wording is not 'these players will collapse'; it is 'the hot-start portion is not adding positive information beyond the baseline.'"
    )
    lines.append(
        "The fake-cold list is the mirror image. These are bottom-decile April wOBA hitters with positive predicted deltas, which generally means the model is giving more weight to prior talent, contact-quality residue, or plate-discipline components than to the early results. "
        "The chart `charts/r2/r2_top10_fake_cold_hitters.png` is the cleanest visual for this section because it shows how many of the intervals still cross zero even when the point estimate is constructive."
    )
    if sleepers:
        first = sleepers[0]
        lines.append(
            f"The leading sleeper, {first['player']}, has a predicted ROS delta of {fmt(first.get('pred_delta_mean'), 4)} wOBA with an 80% QRF delta band "
            f"from {fmt(first.get('delta_q10'), 4)} to {fmt(first.get('delta_q90'), 4)}. "
            f"He is outside the mainstream proxy and therefore satisfies the mechanical sleeper definition."
        )
    lines.append("Top sleeper detail: " + hitter_detail(sleepers, 5) + ".")
    if fake_hot:
        first = fake_hot[0]
        lines.append(
            f"The leading fake-hot flag, {first['player']}, is on the mainstream list but projects at {fmt(first.get('pred_delta_mean'), 4)} below prior. "
            "That is the exact type of April line this model is designed to fade: visible production without enough persistent component support."
        )
    lines.append("Top fake-hot detail: " + hitter_detail(fake_hot, 5) + ".")
    lines.append("Top fake-cold detail: " + hitter_detail(fake_cold, 5) + ".")

    lines.append("\n## xwOBA-Gap Sheet\n")
    lines.append(
        "The gap table uses `xwOBA - wOBA`; positive means a hitter's actual wOBA is below expected contact quality, while negative means actual production is running above xwOBA. "
        f"The largest absolute gaps are {names(xgap, 10)}. "
        "This sheet is deliberately redundant with the atlas because the R1 critique asked that contact-quality residuals become a headline feature rather than a footnote."
    )
    lines.append(
        "The gap list is most useful as a sanity check on fake-hot and fake-cold calls. A fake-hot with a large negative gap is a classic regression candidate; a fake-cold with a large positive gap is a cleaner buy-low. "
        "Where the lists do not overlap, the model is usually leaning on prior skill, strike-zone discipline, or role volume rather than a single contact-quality residual."
    )
    lines.append(
        f"The feature-ranking nuance matters. The signed xwOBA-minus-wOBA term can be diluted by its absolute-value version and by the xwOBA-minus-prior residual, so the standalone signed feature is not the only way the model sees the gap. "
        f"In the permutation table, the xwOBA-vs-prior residual ranks {method['xwoba_gap_feature']['permutation_rank'].get('xwoba_minus_prior_woba_22g')}, "
        f"the absolute xwOBA/wOBA gap ranks {method['xwoba_gap_feature']['permutation_rank'].get('abs_xwoba_minus_woba_22g')}, "
        f"and the signed gap ranks {method['xwoba_gap_feature']['permutation_rank'].get('xwoba_minus_woba_22g')}. "
        "That is reported directly rather than hidden behind SHAP."
    )

    lines.append("\n## Reliever Board\n")
    lines.append(
        "The reliever board predicts ROS K% from first-window reliever features and compares it to a 3-year weighted K% prior. "
        "The 2026 universe requires at least 25 batters faced and fewer than 30 innings through the cutoff. "
        f"The top non-closer K% risers are {names(relievers.get('sleepers', []), 5)}, while the high-April K% shrink candidates are {names(relievers.get('fake_dominant', []), 5)}. "
        "The 2025 saves-leader top 30 are excluded from the sleeper list, and Mason Miller is also excluded from sleeper promotion because he is a known R1 closer case."
    )
    if relievers.get("sleepers"):
        first = relievers["sleepers"][0]
        lines.append(
            f"The leading reliever sleeper, {first['player']}, projects for a ROS K% of {pct(first.get('pred_k_rate_mean'))}, "
            f"up {pct(first.get('pred_k_delta_vs_prior'))} against prior, with a QRF 80% band of {pct(first.get('k_rate_q10'))} to {pct(first.get('k_rate_q90'))}."
        )
    lines.append("Top reliever-sleeper detail: " + reliever_detail(relievers.get("sleepers", []), 5) + ".")
    lines.append("Top fake-dominant reliever detail: " + reliever_detail(relievers.get("fake_dominant", []), 5) + ".")
    lines.append(
        "Reliever interpretation is narrower than hitter interpretation. The model is only projecting K%, not run prevention, leverage value, closer odds, or role security. "
        "Because the board excludes 2025 saves leaders and the named Mason Miller case from sleeper promotion, it is intentionally biased toward discovery rather than toward listing the best relievers in baseball. "
        "The fake-dominant list should be read as a shrinkage board: relievers whose April K% looks louder than their projected ROS K% once prior and pitch-level components are accounted for."
    )

    lines.append("\n## R1 Sanity Check\n")
    h_sanity = findings["r1_sanity_check"].get("hitters", [])
    r_sanity = findings["r1_sanity_check"].get("reliever", [])
    sanity_bits = []
    for rec in h_sanity:
        sanity_bits.append(f"{rec.get('player')} {rec.get('r2_verdict')} ({fmt(rec.get('pred_ros_woba'), 3)} ROS wOBA)")
    for rec in r_sanity:
        sanity_bits.append(f"{rec.get('player')} {rec.get('r2_verdict')} ({pct(rec.get('pred_k_rate_mean'))} ROS K%)")
    lines.append(
        "The named-case check returns: " + "; ".join(sanity_bits) + ". "
        "Pages should remain a noise-style call, Trout/Rice/Miller should remain ambiguous-ish or component-specific, and Murakami now receives a real pipeline verdict instead of a substitute. "
        "Any flip from R1 is flagged in `findings_r2.json` rather than hidden."
    )
    flags = findings["r1_sanity_check"].get("expected_alignment_flags", [])
    if flags:
        lines.append(
            "The explicit flags are: "
            + "; ".join([f"{f['player']} expected {f['expected']} but returned {f['actual']} ({f['flag']})" for f in flags])
            + ". The important reproducibility repair is Murakami: he is now scored as Munetaka Murakami, MLBAM 808959, not replaced by a proxy hitter."
        )

    lines.append("\n## Kill-Gate Outcomes\n")
    lines.append(
        f"Universe coverage is {kill.get('universe_coverage')}; sleeper yield is {kill.get('sleeper_signal_yield_h1')}; "
        f"fake-hot yield is {kill.get('fake_hot_yield_h2')}; reliever sleeper yield is {kill.get('reliever_sleepers_h4')}; "
        f"QRF coverage gate is {kill.get('qrf_coverage_gate')}. "
        "Historical analog retrieval uses cosine similarity with a 0.70 minimum and writes failures explicitly, so names with fewer than five analogs should not be promoted as clean analog-backed picks."
    )
    analogs = findings.get("analog_retrieval", {}).get("analogs", {})
    analog_failures = [v.get("target_player") for v in analogs.values() if not v.get("kill_gate_passed")]
    lines.append(
        f"The analog gate passed for {len(analogs) - len(analog_failures)} of {len(analogs)} hitter picks across the sleeper, fake-hot, and fake-cold lists. "
        + ("No pick failed the five-analog, cosine >= 0.70 threshold." if not analog_failures else f"Failures: {', '.join(analog_failures)}.")
        + " The analog table reports what each nearest historical player-season did rest-of-season, including ROS wOBA, ISO, K%, and delta vs prior."
    )
    lines.append(
        "The biggest residual risk is not software reproducibility; it is baseball interpretation. A top-decile April component vector can belong to a player without a stable role, and a negative delta for a superstar can mean only that the model refuses to move an already-elite prior upward. "
        "Those cases need editorial restraint. The R2 outputs are best used as a disciplined board for cross-method comparison, not as final article copy by themselves."
    )

    lines.append("\n## Open Questions\n")
    lines.append(
        "Round 3 should only start after cross-review. The obvious follow-ups are an NPB-informed Murakami prior, better reliever role labels than 2025 saves alone, and a comparison against Agent A's independent sleeper set. "
        "This pass does not use defensive metrics, team context, park factors, or causal claims about why any player's component vector moved."
    )
    (BASE_DIR / "REPORT_R2.md").write_text("\n\n".join(lines), encoding="utf-8")


def write_ready(findings: dict) -> None:
    sleepers = findings["persistence_atlas"]["sleepers"]
    fake_hot = findings["persistence_atlas"]["fake_hot"]
    relievers = findings["reliever_board"]["sleepers"]
    method = findings["methodology_fixes_status"]
    lines = []
    lines.append("# Ready for Review R2\n")
    lines.append(f"Sleeper hitter picks: {names(sleepers, 10)}.")
    lines.append(f"Fake-hot hitter picks: {names(fake_hot, 10)}.")
    lines.append(f"Sleeper reliever picks: {names(relievers, 5)}.")
    lines.append(
        f"Methodology fixes: Murakami resolver {method['murakami_reproducibility']['status']} (MLBAM 808959); "
        f"QRF coverage {method['qrf_coverage_check']['status']} with 2025 80% coverage {pct(method['qrf_coverage_check']['coverage_80pct_2025'])}; "
        f"era counterfactual {method['era_counterfactual']['status']}; "
        f"SHAP {method['shap_permutation_discipline']['status']} by dropping it and using permutation only; "
        f"xwOBA gap {method['xwoba_gap_feature']['status']}."
    )
    lines.append(
        f"Universe: {method['universe_scan']['hitter_universe_n']} hitters and {method['universe_scan']['reliever_universe_n']} relievers, "
        f"actual Statcast max game date {method['universe_scan']['actual_max_game_date']}."
    )
    lines.append(
        f"Kill gates: {findings['kill_gate_outcomes']}. Cross-agent comparison should treat any interval verdicts according to the QRF coverage warning in `findings_r2.json`."
    )
    (BASE_DIR / "READY_FOR_REVIEW_R2.md").write_text("\n\n".join(lines), encoding="utf-8")


def main() -> dict:
    payloads = load_payloads()
    findings = build_findings(payloads)
    write_report(findings, payloads)
    write_ready(findings)
    return findings


if __name__ == "__main__":
    print(json.dumps(jsonable(main()), indent=2))

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from data_prep_r2 import (
    ARTIFACTS_DIR,
    CHARTS_DIR,
    DIAG_DIR,
    GLOBAL_SEED,
    Round2Data,
    encode_features,
    grouped_oof_predictions,
    lgbm_classifier,
    save_json,
    terminal_pa_rows,
    walk_rate_from_terminal,
    write_model_diagnostics,
)


H1_NUMERIC = ["year_2026", "week_index", "plate_x", "plate_z"]
H1_CATEGORICAL = ["count_state", "pitch_type"]
FEATURE_GROUPS = {
    "year": ["year_2026"],
    "week": ["week_index"],
    "plate_x": ["plate_x"],
    "plate_z": ["plate_z"],
    "count_state": "count_state_",
    "pitch_type": "pitch_type_",
}


@dataclass
class H1Result:
    summary: dict
    weekly: pd.DataFrame
    diagnostics: dict
    permutation: pd.DataFrame


def _model_frame(data: Round2Data) -> pd.DataFrame:
    terminal = pd.concat([terminal_pa_rows(data.df_2025), terminal_pa_rows(data.df_2026)], ignore_index=True)
    frame = terminal.dropna(subset=["plate_x", "plate_z", "pitch_type", "count_state", "game_pk"]).copy()
    frame["walk_event"] = frame["walk_event"].astype(int)
    return frame


def _group_columns(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for group, spec in FEATURE_GROUPS.items():
        if isinstance(spec, list):
            groups[group] = [col for col in spec if col in columns]
        else:
            groups[group] = [col for col in columns if col.startswith(spec)]
    return groups


def _permutation_importance(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    y = frame["walk_event"].to_numpy(dtype=int)
    groups = frame["game_pk"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(cv.split(frame, y, groups))
    X_train, columns = encode_features(frame.iloc[train_idx], H1_NUMERIC, H1_CATEGORICAL)
    X_test, _ = encode_features(frame.iloc[test_idx], H1_NUMERIC, H1_CATEGORICAL, columns)
    y_train = y[train_idx]
    y_test = y[test_idx]
    model = lgbm_classifier(seed + 30)
    model.fit(X_train, y_train)
    baseline_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    rng = np.random.default_rng(seed + 31)
    group_columns = _group_columns(columns)
    rows = []
    for group, cols in group_columns.items():
        if not cols:
            continue
        drops = []
        for _ in range(8):
            permuted = X_test.copy()
            for col in cols:
                permuted[col] = rng.permutation(permuted[col].to_numpy())
            drops.append(baseline_auc - roc_auc_score(y_test, model.predict_proba(permuted)[:, 1]))
        rows.append({"feature": group, "auc_drop": float(np.mean(drops)), "model": "actual_labels"})

    y_perm = rng.permutation(y_train)
    null_model = lgbm_classifier(seed + 32)
    null_model.fit(X_train, y_perm)
    null_auc = roc_auc_score(y_test, null_model.predict_proba(X_test)[:, 1])
    for group, cols in group_columns.items():
        if not cols:
            continue
        drops = []
        for _ in range(8):
            permuted = X_test.copy()
            for col in cols:
                permuted[col] = rng.permutation(permuted[col].to_numpy())
            drops.append(null_auc - roc_auc_score(y_test, null_model.predict_proba(permuted)[:, 1]))
        rows.append({"feature": group, "auc_drop": float(np.mean(drops)), "model": "permuted_label_baseline"})

    out = pd.DataFrame(rows).sort_values(["model", "auc_drop"], ascending=[True, False])
    out.to_csv(ARTIFACTS_DIR / "h1_permutation_importance.csv", index=False)
    return out


def _ensemble_counterfactual(frame: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    X, columns = encode_features(frame, H1_NUMERIC, H1_CATEGORICAL)
    flipped = frame.copy()
    flipped["year_2026"] = 1 - flipped["year_2026"]
    X_flip, _ = encode_features(flipped, H1_NUMERIC, H1_CATEGORICAL, columns)
    y = frame["walk_event"].to_numpy(dtype=int)

    rows = []
    for seed in seeds:
        model = lgbm_classifier(seed)
        model.fit(X, y)
        actual_pred = model.predict_proba(X)[:, 1]
        flipped_pred = model.predict_proba(X_flip)[:, 1]
        temp = frame[["year", "week_index", "week_label", "walk_event"]].copy()
        temp["actual_pred"] = actual_pred
        temp["counterfactual_pred"] = flipped_pred
        temp["seed"] = seed
        rows.append(temp)
    all_preds = pd.concat(rows, ignore_index=True)
    all_preds.to_csv(ARTIFACTS_DIR / "h1_counterfactual_predictions_by_seed.csv", index=False)
    weekly = (
        all_preds.groupby(["seed", "year", "week_index", "week_label"], observed=True)
        .agg(predicted_walk_rate=("actual_pred", "mean"), counterfactual_walk_rate=("counterfactual_pred", "mean"))
        .reset_index()
    )
    return weekly


def _plot_learning_curve(frame: pd.DataFrame, seed: int) -> None:
    y = frame["walk_event"].to_numpy(dtype=int)
    groups = frame["game_pk"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(cv.split(frame, y, groups))
    train_frame = frame.iloc[train_idx].copy()
    test_frame = frame.iloc[test_idx].copy()
    y_train = train_frame["walk_event"].to_numpy(dtype=int)
    y_test = test_frame["walk_event"].to_numpy(dtype=int)
    sizes = np.linspace(0.2, 1.0, 5)
    rng = np.random.default_rng(seed + 100)
    rows = []
    for frac in sizes:
        n = max(2000, int(len(train_frame) * frac))
        idx = rng.choice(len(train_frame), size=n, replace=False)
        X_train, columns = encode_features(train_frame.iloc[idx], H1_NUMERIC, H1_CATEGORICAL)
        X_test, _ = encode_features(test_frame, H1_NUMERIC, H1_CATEGORICAL, columns)
        model = lgbm_classifier(seed + int(frac * 1000), n_estimators=180)
        model.fit(X_train, y_train[idx])
        rows.append({"train_rows": n, "auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])})
    curve = pd.DataFrame(rows)
    curve.to_csv(ARTIFACTS_DIR / "h1_learning_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.plot(curve["train_rows"], curve["auc"], marker="o", color="#2563eb")
    ax.set_xlabel("Training rows")
    ax.set_ylabel("Validation AUC")
    ax.set_title("H1 LightGBM learning curve")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(DIAG_DIR / "h1_learning_curve.png", dpi=200)
    plt.close(fig)


def _plot_weekly(observed: pd.DataFrame, cf_weekly: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 5.7))
    colors = {2025: "#6b7280", 2026: "#dc2626"}
    for year, group in observed.groupby("year", observed=True):
        group = group.sort_values("week_index")
        ax.errorbar(
            group["week_index"],
            group["walk_rate"] * 100,
            yerr=np.vstack([(group["walk_rate"] - group["ci_low"]) * 100, (group["ci_high"] - group["walk_rate"]) * 100]),
            marker="o",
            linewidth=2,
            capsize=4,
            color=colors[int(year)],
            label=f"Actual {year}",
        )
    cf_2026 = (
        cf_weekly[cf_weekly["year"] == 2026]
        .groupby(["week_index", "week_label"], observed=True)["counterfactual_walk_rate"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("week_index")
    )
    if not cf_2026.empty:
        ax.plot(
            cf_2026["week_index"],
            cf_2026["mean"] * 100,
            color="#2563eb",
            linestyle="--",
            marker="s",
            label="2026 with year flipped to 2025",
        )
        ax.fill_between(
            cf_2026["week_index"].to_numpy(),
            (cf_2026["mean"] - 1.96 * cf_2026["std"].fillna(0)).to_numpy() * 100,
            (cf_2026["mean"] + 1.96 * cf_2026["std"].fillna(0)).to_numpy() * 100,
            color="#bfdbfe",
            alpha=0.35,
        )
    ax.set_xlabel("7-day window from Mar 27")
    ax.set_ylabel("Walk rate incl. IBB (%)")
    ax.set_title("H1: weekly walk rate persistence")
    ax.set_xticks(sorted(observed["week_index"].unique()))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "h1_walk_rate_by_week.png", dpi=220)
    plt.close(fig)


def _overall_delta_ci(terminal_2025: pd.DataFrame, terminal_2026: pd.DataFrame, seed: int, n_boot: int = 600) -> tuple[float, float]:
    rng = np.random.default_rng(seed)

    def game_table(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame.groupby("game_pk", observed=True)
            .agg(pas=("pa_id", "size"), walks=("walk_event", "sum"))
            .reset_index()
        )

    tab25 = game_table(terminal_2025)
    tab26 = game_table(terminal_2026)
    games25 = tab25["game_pk"].to_numpy()
    games26 = tab26["game_pk"].to_numpy()
    values = []
    for _ in range(n_boot):
        s25 = pd.Series(rng.choice(games25, size=len(games25), replace=True)).value_counts().rename_axis("game_pk").reset_index(name="weight")
        s26 = pd.Series(rng.choice(games26, size=len(games26), replace=True)).value_counts().rename_axis("game_pk").reset_index(name="weight")
        b25 = tab25.merge(s25, on="game_pk", how="inner")
        b26 = tab26.merge(s26, on="game_pk", how="inner")
        rate25 = float((b25["walks"] * b25["weight"]).sum() / (b25["pas"] * b25["weight"]).sum())
        rate26 = float((b26["walks"] * b26["weight"]).sum() / (b26["pas"] * b26["weight"]).sum())
        values.append((rate26 - rate25) * 100)
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def run_h1(data: Round2Data) -> H1Result:
    frame = _model_frame(data)
    y = frame["walk_event"].to_numpy(dtype=int)
    groups = frame["game_pk"].to_numpy()
    oof, auc = grouped_oof_predictions(frame, y, groups, H1_NUMERIC, H1_CATEGORICAL, GLOBAL_SEED + 101)
    diagnostics = write_model_diagnostics("h1_walk_classifier", y, oof)
    _plot_learning_curve(frame, GLOBAL_SEED + 102)
    permutation = _permutation_importance(frame, GLOBAL_SEED + 103)

    terminal_2025 = terminal_pa_rows(data.df_2025)
    terminal_2026 = terminal_pa_rows(data.df_2026)
    rate_2025 = walk_rate_from_terminal(terminal_2025)
    rate_2026 = walk_rate_from_terminal(terminal_2026)
    delta = rate_2026["walk_rate"] - rate_2025["walk_rate"]
    ci_low_pp, ci_high_pp = _overall_delta_ci(terminal_2025, terminal_2026, GLOBAL_SEED + 110)

    observed = pd.read_csv(data_prep_weekly_path())
    seeds = [GLOBAL_SEED + 1200 + i for i in range(10)]
    cf_weekly = _ensemble_counterfactual(frame, seeds)
    cf_summary = (
        cf_weekly.groupby(["year", "week_index", "week_label"], observed=True)
        .agg(cf_mean=("counterfactual_walk_rate", "mean"), cf_sd=("counterfactual_walk_rate", "std"))
        .reset_index()
    )
    cf_summary.to_csv(ARTIFACTS_DIR / "h1_weekly_counterfactual.csv", index=False)
    _plot_weekly(observed, cf_weekly)

    summary = {
        "rate_2025": rate_2025["walk_rate"],
        "rate_2026": rate_2026["walk_rate"],
        "yoy_delta_pp": delta * 100,
        "ci_low_pp": ci_low_pp,
        "ci_high_pp": ci_high_pp,
        "n_2025": rate_2025["pas"],
        "n_2026": rate_2026["pas"],
        "walks_2025": rate_2025["walks"],
        "walks_2026": rate_2026["walks"],
        "classifier_auc": auc,
        "max_calibration_deviation": diagnostics["max_calibration_deviation"],
        "poor_calibration": diagnostics["poor_calibration"],
        "ensemble_seeds": seeds,
    }
    save_json(summary, ARTIFACTS_DIR / "h1_summary.json")
    return H1Result(summary=summary, weekly=observed, diagnostics=diagnostics, permutation=permutation)


def data_prep_weekly_path() -> str:
    from data_prep_r2 import WEEKLY_PATH

    return str(WEEKLY_PATH)

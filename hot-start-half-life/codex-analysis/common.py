from __future__ import annotations

import json
import math
import os
import random
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEED = 20260425
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
CHARTS_DIR = BASE_DIR / "charts"
DIAG_DIR = CHARTS_DIR / "diag"
TABLES_DIR = BASE_DIR / "tables"

for directory in [DATA_DIR, DATASETS_DIR, MODELS_DIR, CHARTS_DIR, DIAG_DIR, TABLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


STATCAST_COLUMNS = [
    "pitch_type",
    "game_date",
    "player_name",
    "batter",
    "pitcher",
    "events",
    "description",
    "des",
    "game_type",
    "stand",
    "p_throws",
    "home_team",
    "away_team",
    "type",
    "bb_type",
    "balls",
    "strikes",
    "game_year",
    "plate_x",
    "plate_z",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "launch_speed",
    "launch_angle",
    "estimated_woba_using_speedangle",
    "estimated_slg_using_speedangle",
    "woba_value",
    "woba_denom",
    "babip_value",
    "iso_value",
    "launch_speed_angle",
    "at_bat_number",
    "pitch_number",
    "game_pk",
    "bat_score",
    "post_bat_score",
    "home_score",
    "away_score",
]

MIN_REQUIRED_STATCAST_COLUMNS = {
    "pitch_type",
    "game_date",
    "batter",
    "pitcher",
    "events",
    "description",
    "launch_speed",
    "launch_angle",
    "bb_type",
    "pitch_type",
    "plate_x",
    "plate_z",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "balls",
    "strikes",
    "inning",
    "inning_topbot",
    "home_team",
    "away_team",
}

KNOWN_HITTERS = {
    "andy_pages": {"name": "Andy Pages", "mlbam": 681624},
    "ben_rice": {"name": "Ben Rice", "mlbam": 700250},
    "munetaka_murakami": {"name": "Munetaka Murakami", "mlbam": None},
    "mike_trout": {"name": "Mike Trout", "mlbam": 545361},
}

KNOWN_PITCHERS = {
    "mason_miller": {"name": "Mason Miller", "mlbam": 695243},
}

HIT_EVENTS = {"single", "double", "triple", "home_run"}
WALK_EVENTS = {"walk", "intent_walk"}
HBP_EVENTS = {"hit_by_pitch"}
K_EVENTS = {"strikeout", "strikeout_double_play"}
SF_EVENTS = {"sac_fly", "sac_fly_double_play"}
SAC_EVENTS = {"sac_bunt", "sac_fly", "sac_fly_double_play"}
NON_PA_EVENTS = {
    "caught_stealing_2b",
    "caught_stealing_3b",
    "caught_stealing_home",
    "pickoff_1b",
    "pickoff_2b",
    "pickoff_3b",
    "pickoff_caught_stealing_2b",
    "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
}

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "bunt_foul_tip",
    "missed_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}

HITTER_FEATURE_COLS = [
    "pa_22g",
    "ab_22g",
    "woba_22g",
    "avg_22g",
    "obp_22g",
    "slg_22g",
    "ops_22g",
    "babip_22g",
    "bb_rate_22g",
    "k_rate_22g",
    "iso_22g",
    "hr_rate_22g",
    "ev_mean_22g",
    "ev_p90_22g",
    "la_mean_22g",
    "hardhit_rate_22g",
    "barrel_rate_22g",
    "xwoba_22g",
    "xwoba_minus_woba_22g",
    "pitches_per_pa_22g",
    "swing_rate_22g",
    "whiff_per_swing_22g",
    "called_strike_rate_22g",
    "zone_rate_22g",
    "chase_rate_22g",
    "preseason_prior_woba",
    "preseason_prior_iso",
    "preseason_prior_k_rate",
    "league_woba",
    "league_bb_rate",
    "league_k_rate",
    "league_babip",
    "league_iso",
]

HITTER_ANALOG_COLS = [
    "pa_22g",
    "woba_22g",
    "babip_22g",
    "bb_rate_22g",
    "k_rate_22g",
    "iso_22g",
    "ops_22g",
    "ev_p90_22g",
    "hardhit_rate_22g",
    "barrel_rate_22g",
    "xwoba_minus_woba_22g",
    "swing_rate_22g",
    "whiff_per_swing_22g",
    "chase_rate_22g",
]

PITCHER_FEATURE_COLS = [
    "bf_22g",
    "ip_22g",
    "ra9_22g",
    "k_rate_22g",
    "bb_rate_22g",
    "hr_rate_22g",
    "ev_p90_allowed_22g",
    "hardhit_allowed_rate_22g",
    "pitches_per_bf_22g",
    "swing_rate_22g",
    "whiff_per_swing_22g",
    "zone_rate_22g",
    "league_pitcher_k_rate",
    "league_pitcher_bb_rate",
    "league_pitcher_ra9",
]

PITCHER_ANALOG_COLS = [
    "bf_22g",
    "ip_22g",
    "ra9_22g",
    "k_rate_22g",
    "bb_rate_22g",
    "hr_rate_22g",
    "ev_p90_allowed_22g",
    "hardhit_allowed_rate_22g",
    "whiff_per_swing_22g",
]


def slugify_player(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return ascii_name.lower().replace(".", "").replace("'", "").replace(" ", "_").replace("-", "_")


def safe_divide(num: Any, den: Any) -> Any:
    num_arr = np.asarray(num, dtype="float64")
    den_arr = np.asarray(den, dtype="float64")
    out = np.full(np.broadcast(num_arr, den_arr).shape, np.nan, dtype="float64")
    np.divide(num_arr, den_arr, out=out, where=den_arr != 0)
    if out.shape == ():
        return float(out)
    return out


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def clean_statcast_frame(df: pd.DataFrame) -> pd.DataFrame:
    for col in STATCAST_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    out = df.loc[:, STATCAST_COLUMNS].copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    if "game_type" in out.columns:
        out = out[(out["game_type"].isna()) | (out["game_type"] == "R")]
    numeric_cols = [
        "batter",
        "pitcher",
        "game_year",
        "plate_x",
        "plate_z",
        "outs_when_up",
        "inning",
        "launch_speed",
        "launch_angle",
        "estimated_woba_using_speedangle",
        "estimated_slg_using_speedangle",
        "woba_value",
        "woba_denom",
        "babip_value",
        "iso_value",
        "launch_speed_angle",
        "at_bat_number",
        "pitch_number",
        "game_pk",
        "bat_score",
        "post_bat_score",
        "home_score",
        "away_score",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def statcast_cache_valid(path: Path, min_rows: int = 1000) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing-or-empty"
    try:
        schema = pd.read_parquet(path, columns=[]).shape
        df_head = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive cache validation
        return False, f"read-error:{exc}"
    missing = sorted(MIN_REQUIRED_STATCAST_COLUMNS.difference(df_head.columns))
    if missing:
        return False, "missing-columns:" + ",".join(missing[:8])
    if len(df_head) < min_rows:
        return False, f"too-few-rows:{len(df_head)}"
    _ = schema
    return True, "ok"


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        value = float(value)
    except Exception:
        return default
    if math.isfinite(value):
        return value
    return default


def set_plot_style() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

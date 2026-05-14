from __future__ import annotations

import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from modeling_r2 import GLOBAL_SEED

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "codex-analysis-r2"
DATA_DIR = ANALYSIS_DIR / "data"
ARTIFACTS_DIR = ANALYSIS_DIR / "artifacts"
CHARTS_DIR = ANALYSIS_DIR / "charts"
DIAG_DIR = CHARTS_DIR / "model_diagnostics"

ROUND1_CODEX_DIR = ROOT / "codex-analysis"
ROUND1_ARTIFACTS_DIR = ROUND1_CODEX_DIR / "artifacts"
ROUND1_CALLED_PATH = ROUND1_ARTIFACTS_DIR / "called_pitches_full.parquet"
ROUND1_CHALLENGES_PATH = ROUND1_ARTIFACTS_DIR / "challenges_full.parquet"

STATCAST_2025_FULL_PATH = ROOT.parent / "count-distribution-abs" / "data" / "statcast_2025_full.parquet"
CHASE_RATE_PATH = DATA_DIR / "batter_chase_rate_2025.parquet"
PLAYER_LOOKUP_PATH = DATA_DIR / "player_lookup.parquet"

PREPARED_CALLED_R2_PATH = ARTIFACTS_DIR / "called_pitches_r2.parquet"
PREPARED_CHALLENGES_R2_PATH = ARTIFACTS_DIR / "challenges_r2.parquet"
RUN_MANIFEST_PATH = ARTIFACTS_DIR / "run_manifest_r2.json"

USER_AGENT = "CalledThird research bot; contact: research@calledthird.local"


def ensure_dirs() -> None:
    for path in [DATA_DIR, ARTIFACTS_DIR, CHARTS_DIR, DIAG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_json(payload: Any, path: Path) -> None:
    def default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if not np.isfinite(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return str(obj)

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default) + "\n")


def package_versions() -> dict[str, str]:
    out = {"python": sys.version.replace("\n", " ")}
    for name in ["pandas", "numpy", "lightgbm", "scikit-learn", "scipy", "matplotlib", "pyarrow"]:
        try:
            out[name] = version(name)
        except Exception:
            out[name] = "unknown"
    return out


def _request_people(ids: list[int]) -> list[dict[str, Any]]:
    if not ids:
        return []
    joined = ",".join(map(str, ids))
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={joined}"
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=45)
    response.raise_for_status()
    return response.json().get("people", [])


def build_player_lookup(called_pitches: pd.DataFrame, challenges: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    ensure_dirs()
    if PLAYER_LOOKUP_PATH.exists() and not force:
        return pd.read_parquet(PLAYER_LOOKUP_PATH)

    ids = set(pd.to_numeric(called_pitches.get("batter_id"), errors="coerce").dropna().astype(int).tolist())
    if "batter_id" in challenges.columns:
        ids.update(pd.to_numeric(challenges["batter_id"], errors="coerce").dropna().astype(int).tolist())
    rows: dict[int, dict[str, Any]] = {}
    if {"batter_id", "batter_name"}.issubset(challenges.columns):
        for rec in challenges[["batter_id", "batter_name"]].dropna().drop_duplicates().itertuples(index=False):
            rows[int(rec.batter_id)] = {"batter_id": int(rec.batter_id), "batter_name": str(rec.batter_name)}

    missing = sorted(ids - set(rows))
    for start in range(0, len(missing), 80):
        chunk = missing[start : start + 80]
        try:
            people = _request_people(chunk)
            for person in people:
                pid = int(person["id"])
                rows[pid] = {"batter_id": pid, "batter_name": person.get("fullName") or f"batter_{pid}"}
        except Exception:
            for pid in chunk:
                rows[pid] = {"batter_id": int(pid), "batter_name": f"batter_{pid}"}
        time.sleep(0.1)

    lookup = pd.DataFrame(rows.values()).drop_duplicates("batter_id")
    lookup.to_parquet(PLAYER_LOOKUP_PATH, index=False)
    return lookup


def load_prepared_data(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ensure_dirs()
    if PREPARED_CALLED_R2_PATH.exists() and PREPARED_CHALLENGES_R2_PATH.exists() and RUN_MANIFEST_PATH.exists() and not force:
        called = pd.read_parquet(PREPARED_CALLED_R2_PATH)
        challenges = pd.read_parquet(PREPARED_CHALLENGES_R2_PATH)
        manifest = json.loads(RUN_MANIFEST_PATH.read_text())
        return called, challenges, manifest

    if not ROUND1_CALLED_PATH.exists() or not ROUND1_CHALLENGES_PATH.exists():
        raise FileNotFoundError(
            "Round 1 prepared artifacts are required: "
            f"{ROUND1_CALLED_PATH} and {ROUND1_CHALLENGES_PATH}"
        )

    called = pd.read_parquet(ROUND1_CALLED_PATH).copy()
    challenges = pd.read_parquet(ROUND1_CHALLENGES_PATH).copy()
    for col in ["game_date"]:
        if col in called.columns:
            called[col] = pd.to_datetime(called[col])
        if col in challenges.columns:
            challenges[col] = pd.to_datetime(challenges[col])

    if "edge_distance_in" not in called.columns and "edge_distance_ft" in called.columns:
        called["edge_distance_in"] = pd.to_numeric(called["edge_distance_ft"], errors="coerce") * 12.0
    if "edge_distance_in" in called.columns and "edge_distance_ft" not in called.columns:
        called["edge_distance_ft"] = pd.to_numeric(called["edge_distance_in"], errors="coerce") / 12.0
    if "edge_distance_in" not in called.columns:
        called["edge_distance_in"] = pd.to_numeric(called["edge_distance_ft"], errors="coerce") * 12.0

    for frame in [called, challenges]:
        if "lineup_spot" in frame.columns:
            frame["lineup_spot"] = pd.to_numeric(frame["lineup_spot"], errors="coerce")
        if "is_pinch_hitter" in frame.columns:
            frame["is_pinch_hitter"] = frame["is_pinch_hitter"].fillna(False).astype(bool)

    player_lookup = build_player_lookup(called, challenges)
    called = called.merge(player_lookup, on="batter_id", how="left")
    called["batter_name"] = called["batter_name"].fillna(called["batter_id"].astype(str).map(lambda x: f"batter_{x}"))

    called.to_parquet(PREPARED_CALLED_R2_PATH, index=False)
    challenges.to_parquet(PREPARED_CHALLENGES_R2_PATH, index=False)

    manifest = {
        "seed": GLOBAL_SEED,
        "sources": {
            "called_pitches": str(ROUND1_CALLED_PATH),
            "challenges": str(ROUND1_CHALLENGES_PATH),
            "statcast_2025_full": str(STATCAST_2025_FULL_PATH),
            "player_lookup": str(PLAYER_LOOKUP_PATH),
        },
        "rows": {
            "called_pitches": int(len(called)),
            "borderline_called_pitches": int(called["is_borderline"].sum()) if "is_borderline" in called.columns else None,
            "challenges": int(len(challenges)),
            "catcher_challenges": int((challenges["challenger"].astype(str).str.lower() == "catcher").sum())
            if "challenger" in challenges.columns
            else None,
            "umpires_borderline": int(called.loc[called["is_borderline"].astype(bool), "umpire"].nunique())
            if {"is_borderline", "umpire"}.issubset(called.columns)
            else None,
        },
        "versions": package_versions(),
    }
    save_json(manifest, RUN_MANIFEST_PATH)
    return called, challenges, manifest

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codex-analysis"))

from utils import load_pitch_data, prepare_called_pitches  # noqa: E402
from zone_classifier import fit_zone_model  # noqa: E402


def summarize_year_specific_zone_fields() -> None:
    data = load_pitch_data()
    called25 = prepare_called_pitches(data["2025_primary"], 2025)
    called26 = prepare_called_pitches(data["2026_primary"], 2026)
    for year, frame in [(2025, called25), (2026, called26)]:
        zone_height = frame["sz_top"] - frame["sz_bot"]
        print(
            f"{year} called median sz_top={frame['sz_top'].median():.4f} "
            f"sz_bot={frame['sz_bot'].median():.4f} zone_height={zone_height.median():.4f}"
        )


def summarize_claude_normalized_grid() -> None:
    artifact = np.load(ROOT / "claude-analysis" / "artifacts" / "zone_grid_norm.npz")
    delta = artifact["delta"]
    lo = artifact["ci_lo"]
    hi = artifact["ci_hi"]
    n25 = artifact["n25"]
    n26 = artifact["n26"]
    x_lo = float(artifact["x_lo"])
    x_bin = float(artifact["x_bin"])
    z_lo = float(artifact["z_lo"])
    z_bin = float(artifact["z_bin"])

    x_centers = x_lo + x_bin * (np.arange(delta.shape[1]) + 0.5)
    z_centers = z_lo + z_bin * (np.arange(delta.shape[0]) + 0.5)
    xc, zc = np.meshgrid(x_centers, z_centers)
    inside_x = np.abs(xc) <= 17 / 24
    min_counts = np.minimum(n25, n26)

    masks = {
        "abs_top_band_in_human_norm_space": inside_x & (zc >= 0.485) & (zc <= 0.585),
        "abs_bottom_band_in_human_norm_space": inside_x & (zc >= 0.22) & (zc <= 0.32),
        "full_human_rulebook_box": inside_x & (zc >= 0.0) & (zc <= 1.0),
    }
    for label, mask in masks.items():
        valid = mask & ~np.isnan(delta) & (min_counts >= 30)
        if not valid.any():
            print(label, "no valid cells")
            continue
        weights = min_counts[valid]
        weighted_mean_pp = float(np.average(delta[valid], weights=weights) * 100)
        sig_neg = int(((hi < 0) & valid).sum())
        sig_pos = int(((lo > 0) & valid).sum())
        total = int(valid.sum())
        print(
            f"{label}: cells={total} weighted_mean_pp={weighted_mean_pp:.2f} "
            f"sig_neg_frac={sig_neg / total:.3f} sig_pos_frac={sig_pos / total:.3f}"
        )


def summarize_model_sign_on_first_pitches() -> None:
    data = load_pitch_data()
    called25 = prepare_called_pitches(data["2025_primary"], 2025)
    called26 = prepare_called_pitches(data["2026_primary"], 2026)
    model25 = fit_zone_model(called25, 20260423)
    model26 = fit_zone_model(called26, 20260424)

    for label, frame in [
        ("all", called26),
        ("0-0", called26[called26["count_state"] == "0-0"]),
        ("3-0", called26[called26["count_state"] == "3-0"]),
        ("3-1", called26[called26["count_state"] == "3-1"]),
        ("3-2", called26[called26["count_state"] == "3-2"]),
    ]:
        features = frame[["plate_x", "plate_z_norm"]].to_numpy(dtype=float)
        p25 = model25.predict_proba(features)[:, 1]
        p26 = model26.predict_proba(features)[:, 1]
        print(
            f"{label}: n={len(frame)} pred25_mean={p25.mean() * 100:.2f} "
            f"pred26_mean={p26.mean() * 100:.2f} delta_26_minus_25_pp={(p26.mean() - p25.mean()) * 100:.2f}"
        )


if __name__ == "__main__":
    print("[year-specific zone fields]")
    summarize_year_specific_zone_fields()
    print("\n[claude normalized grid]")
    summarize_claude_normalized_grid()
    print("\n[first-pitch sign check]")
    summarize_model_sign_on_first_pitches()

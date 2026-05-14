"""Did sz_top / sz_bot fields change semantics across 2025 -> 2026?

If yes, then Codex's plate_z_norm is NOT comparable across years and his
zone-classifier delta is contaminated by the schema shift, not by zone change.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_2026 = ROOT / "data" / "statcast_2026_mar27_apr22.parquet"
DATA_2025_PRIMARY = ROOT.parent / "count-distribution-abs" / "data" / "statcast_2025_mar27_apr14.parquet"

df_25 = pd.read_parquet(DATA_2025_PRIMARY)
df_26 = pd.read_parquet(DATA_2026)
df_25["game_date"] = pd.to_datetime(df_25["game_date"])
df_26["game_date"] = pd.to_datetime(df_26["game_date"])
df_26 = df_26[df_26["game_date"] <= "2026-04-14"]

print("==== sz_top distribution ====")
for label, d in [("2025", df_25), ("2026", df_26)]:
    s = d["sz_top"].dropna()
    print(f"  {label}: mean={s.mean():.3f} median={s.median():.3f} sd={s.std():.3f} n={len(s)}")
print()
print("==== sz_bot distribution ====")
for label, d in [("2025", df_25), ("2026", df_26)]:
    s = d["sz_bot"].dropna()
    print(f"  {label}: mean={s.mean():.3f} median={s.median():.3f} sd={s.std():.3f} n={len(s)}")
print()
print("==== sz_top - sz_bot (zone height) ====")
for label, d in [("2025", df_25), ("2026", df_26)]:
    h = (d["sz_top"] - d["sz_bot"]).dropna()
    print(f"  {label}: mean={h.mean():.3f} median={h.median():.3f} sd={h.std():.3f} n={len(h)}")
print()
print("==== Codex batter_height_proxy = (sz_top-sz_bot)/0.265 ====")
for label, d in [("2025", df_25), ("2026", df_26)]:
    h = ((d["sz_top"] - d["sz_bot"]) / 0.265).dropna()
    print(f"  {label}: mean={h.mean():.3f} median={h.median():.3f} sd={h.std():.3f} n={len(h)}")
print()
print("==== Implication for plate_z_norm ====")
print("If the same actual pitch (plate_z=3.2 ft) is binned with batter_height_proxy=6.83 in")
print("2025 vs 6.01 in 2026, its plate_z_norm is 0.469 in 2025 vs 0.532 in 2026.")
print("That's a 6 percentage-point shift in normalized vertical coordinate from a SCHEMA change,")
print("not a pitch-location change.")
print()
print("==== Variance of sz_top/sz_bot per batter (sanity) ====")
# A given batter's sz_top should be roughly constant. Big within-batter variance =
# an indicator that sz_top is being computed from per-pitch posture, not per-batter.
for label, d in [("2025", df_25), ("2026", df_26)]:
    by_b = d.groupby("batter")["sz_top"].agg(["mean", "std", "count"])
    by_b = by_b[by_b["count"] >= 30]
    print(f"  {label}: median within-batter std of sz_top = {by_b['std'].median():.4f} ft "
          f"({len(by_b)} batters)")
    by_b2 = d.groupby("batter")["sz_bot"].agg(["mean", "std", "count"])
    by_b2 = by_b2[by_b2["count"] >= 30]
    print(f"          median within-batter std of sz_bot = {by_b2['std'].median():.4f} ft")

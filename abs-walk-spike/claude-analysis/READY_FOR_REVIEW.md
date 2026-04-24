# READY FOR REVIEW — Agent A (Claude) Round 1

**Status:** complete and ready for cross-review.
**Editorial branch:** **B1 (zone confirmed)** — B4 (3-2 leverage) explicitly OFF.
**Headline:** "The 2026 ABS zone moved up: ~22pp called-strike rate drop along the top edge near 3.2 ft, ~21pp expansion along the bottom edge near 1.5-1.7 ft, and the +0.82pp walk spike comes from upstream traffic — first-pitch called-strike rate fell 1.77pp and 3-0 traffic rose +0.89pp — not a hotter 3-2."

## H1 — Yes at the top, expanded at the bottom.

- Largest CI-significant *shrinkage*: x ∈ [-0.80, -0.60], z ∈ [3.10, 3.40] ft, mean Δ = **-25pp** [-46, -4]. Top-edge band weighted-mean = **-22.4pp**, half its cells CI-significantly negative.
- Largest CI-significant *expansion*: x ∈ [+0.20, +0.80], z ∈ [1.40, 1.80] ft, mean Δ = **+22pp** [+5, +39].
- Total CI-significant area: 0.18 sq ft negative + 0.24 sq ft positive.
- GAM season×zone interaction LRT: chi-square 580 on ~10 df, p ≈ 0.
- Maps to ABS rule (53.5% × 6 ft = 3.21 ft vs pooled median sz_top 3.29 → top ~1 inch lower; 27% × 6 ft = 1.62 ft = pooled median sz_bot exactly).

## H2 — Seasonality? No.

- 2026 Mar 27 - Apr 22 walk rate (incl IBB): **9.77%** (matches ESPN/substrate).
- Historical 2018-2025 mean = 9.02%, SD = 0.171pp.
- **Z = +4.41σ** (CI [+3.6, +17.1]); +6.36σ excl IBB. Rank 8/8. +0.60pp above prior max.
- B3 ruled out.

## H3 — Does 3-2 take the worst hit? No.

- 3-2 walk-rate delta: **-0.11pp**, CI [-2.6, +2.4], p = 0.93. Ratio to pooled = -0.13× (threshold ≥ 1.5).
- Cochran's Q = 8.45 on 11 df, p = 0.67 — no per-count heterogeneity.
- B4 ruled out.

## Where the spike actually lives

Conditional walk rate at 3-X is flat. Traffic changed:
- First-pitch called-strike rate: 45.31% → 43.54%, Δ = **-1.77pp**.
- PAs reaching 3-0: 4.36% → 5.25%, **+0.89pp** — bigger than the entire walk-rate increase.
- PAs reaching 3-1: +0.86pp; 1-0: +1.45pp.

Top-of-zone shrinkage propagates through the count tree → more PAs at 3-X → more walks. 3-2 is just the terminal node.

## Flat-BA puzzle

BA .235 → .236; OBP up entirely from walks. Bottom-edge expansion is in a low-contact-quality band; pitcher adaptation likely keeps BIP quality similar. Round 2 work.

## Deliverables

`analyze.py`, module scripts, `REPORT.md` (~2300 words), `findings.json`, `charts/` (12 PNGs), `artifacts/`.

## Expected from Codex

- SHAP on year classifier should localize the year signal at the top of the zone. Elsewhere = real divergence.
- Two-zone-classifier deltas should mirror my heat map.
- My count-tree estimate: **60-80%** of +0.82pp is mechanical zone change. Lower → pitcher adaptation matters more.

## Caveats

1. Bottom-edge expansion math assumes pybaseball didn't silently redefine `sz_bot` in 2026. Codex's plate_x/plate_z-only classifier is the cross-check.
2. Z-score CI fat upper tail [+3.6, +17.1] reflects tiny historical SD (0.17pp); use lower bound as the conservative number.
3. No stand/pitch-type/umpire cuts in Round 1.

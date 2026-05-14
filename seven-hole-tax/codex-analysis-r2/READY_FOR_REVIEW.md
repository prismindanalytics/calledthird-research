# Ready For Review - Codex Round 2

H4: positive for per-umpire heterogeneity, not for the tax direction. 5 qualifying umpires cleared q<0.10 and |effect|>=2pp after the LightGBM umpire-lineup interaction model and paired bootstrap; all 5 are reverse-direction and 0 are pro-tax.

H5: null for the primary named-hitter tax. 0 hitters had positive FDR-significant residuals of at least 3pp after the no-batter-ID model; no-pinch creates a non-primary positive but it is not robust enough to publish as the main H5 result.

H6: null. Catcher-initiated spot-7 vs spot-3 challenge selection has energy-distance p=0.955.

H7: null. Low-chase 7-hole effect is -0.78 pp [-0.88, -0.66]; interaction p=0.852.

Recommended branch: **umpire-only**. Biggest concern: Actor-level samples are thin after the required per-umpire and per-hitter filters, so nulls are stronger than any borderline leaderboard rank.

Primary artifacts are `findings.json`, `REPORT.md`, and charts under `charts/`. I did not read `claude-analysis-r2/` before writing this file.

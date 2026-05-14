# READY FOR REVIEW

Round 2 Codex deliverables are complete in `codex-analysis-r2/`.

**H1:** Mixed. The spike persists but regressed: 2025 **8.80%** vs 2026 **9.46%**, delta **+0.66 pp** with game-bootstrap CI **[+0.27, +1.04] pp** through the effective endpoint **2026-05-12**. The May 13 Statcast pull returned no May 13 rows, so I matched through May 12.

**H2:** Terminal-count decomposition does not make 3-2 the whole story. Within-count rate effects sum to **+0.14 pp** and terminal-count traffic effects sum to **+0.52 pp**.

**H3:** Pass. All-pitches replay estimates **35.3%** zone attribution, CI **[34.6, 36.0]**, versus Round 1 **40.46%**. Model uncertainty uses a 10-seed LightGBM refit ensemble.

**H4:** Adaptation is descriptive but real enough for a sidebar. League fixed-zone rate moved **-0.29 pp** YoY; top-10 pitcher leaderboard and SHAP importance are written.

**H5:** Partial mechanism. Heart-zone 0-0 CS delta is **+0.00 pp**; top-edge two-strike delta is **-3.17 pp**. SHAP interactions support a count-dependent zone effect: **True**.

Recommended branch: **adaptation**. Biggest concern: Statcast returned no May 13 pitches at run time; the effective 2026 endpoint is 2026-05-12.

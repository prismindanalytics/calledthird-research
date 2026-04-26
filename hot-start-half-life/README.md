# Hot-Start Half-Life — Three Rounds

Through April 25, 2026, baseball Twitter spent the month talking about the same six MVP-pace 22-game hot starts: Andy Pages, Ben Rice, Mike Trout, Aaron Judge, Corbin Carroll, Max Muncy. Two independent statistical pipelines — one Bayesian, one ML — agreed within a week that all six collapse to noise; surfaced a list of six sleeper hitters and four sleeper relievers nobody was talking about; and forced one large mid-stream retraction we walked back honestly in the public article.

**Published article:** [The April Sell List](https://calledthird.com/analysis/april-sell-list) (April 26, 2026)

## The question

Given the brutal historical noise floor on April hot starts — only 2 of 20 top-5 22-game wOBA leaders from 2022-2025 sustained ≥85% of their April pace — which 2026 names have signal that survives both a Bayesian projection and a learned-coefficient ML pipeline, and which are noise that two independent methods can confidently retire?

## The approach

Two agents analyzed the same brief in parallel, with deliberately divergent methodologies and an editorial constraint of "ship only when both agree":

- **Agent A (Claude)** — interpretability-first: hierarchical Bayesian projection (NUTS in numpyro) with empirical-Bayes shrinkage; player-season cluster bootstrap for stabilization rates; PELT change-point detection on rate-stat trajectories; learned-coefficient contact-quality blend (wOBA + xwOBA + EV p90 + HardHit% + Barrel% + prior wOBA) trained on 2022-2024, validated on 2025 holdout.
- **Agent B (Codex)** — ML-engineering: LightGBM gradient-boosted projection with quantile regression forests for prediction intervals; permutation feature importance (SHAP dropped after R1 rank-correlation failure); historical-analog retrieval via cosine similarity over standardized 22-game feature vectors; honest "raw QRF, over-covering ~5pp" interval framing.

After each round, each agent peer-reviewed the other under FanGraphs-rejection standards. Three rounds of analysis, three rounds of cross-review, and the comparison memos are all in `reviews/`.

## Key findings (R3 final)

1. **The R1 "stabilization rates shifted" headline died.** Carleton 2007 stabilization rates still apply for BB%, K%, BABIP, and ISO once the bootstrap is done at the player-season level rather than within-player PA partitions. wOBA is the only stat that meaningfully shifted (451 PA half-stabilization point vs Carleton's 280, with non-overlapping CI), and even that's fragile.
2. **Pages, Rice, Trout, Judge, Carroll, Muncy — all NOISE.** Both methods independently agree. The R1/R2 SIGNAL claims on Rice and Trout were retracted in R3 when a learned contact-quality blend (RMSE 0.0359 on 2025 holdout vs 0.0371 wOBA-only baseline) replaced the R2 hand-tuned 50/50 wOBA+xwOBA blend.
3. **Six convergent sleeper hitters:** Caglianone, Pereira, Barrosa, Basallo, Mayo, House. All in the top decile of predicted ROS-wOBA-vs-prior delta from both methods, all outside the mainstream April hot-starter coverage, all clearing the analog kill-gate.
4. **Four convergent sleeper relievers:** Senzatela, King, Kilian, Lynch. R3 cross-review resolved the Lynch dispute — Codex's argument that his +9.7-pt K% lift on a 17.7% prior is the textbook sleeper shape (despite the April overshoot >0.30) was more defensible than Claude's coherence rule excluding him.
5. **Murakami AMBIGUOUS-ish, with explicit method-disagreement disclosure.** The only April performance both methods see signal in. Caveat: prior is a 60-PA league-average proxy because no NPB-translation factor was built. The right comparable would be an Ohtani-style NPB-to-MLB translation, but Murakami's NPB peak (56 HRs) is 2.5× Ohtani's NPB peak (22), so no clean analog exists.
6. **Mason Miller's K% rise is real (0.495 ROS projection).** Streak survival probabilities are intentionally not published — the only credible methodology requires a baserunner-state model we didn't build.

## What makes this project unusual

Three rounds of methodological pressure forced two named retractions:

- **R1 → R3:** "Three of five stabilization rates shifted vs Carleton" → "Only wOBA shifted; the within-player bootstrap was a flawed estimand."
- **R2 → R3:** "Rice and Trout are SIGNAL via contact-quality features" → "The contact-quality framing was a hand-tuned 50/50 blend that didn't validate. Once we trained the coefficients on holdout, both Rice and Trout collapse to NOISE."

The full editorial reasoning chain is in `reviews/COMPARISON_MEMO_R3.md`. Each round's brief, agent prompts, peer reviews, and synthesis memos are all preserved in `reviews/`.

## Files

| Path | Description |
|------|-------------|
| `RESEARCH_BRIEF.md` | R1 brief: hypotheses, data, methodology mandates per agent, kill criteria, scope fence |
| `ROUND2_BRIEF.md` | R2 brief: methodology fixes from R1 cross-review + universe scan + persistence atlas + reliever board |
| `ROUND3_BRIEF.md` | R3 brief: methodology fixes from R2 cross-review + ship-gate convergence test |
| `FLAGSHIP_DRAFT.md` | The article draft (the published version is on the website) |
| `claude-analysis/` | Agent A's R1+R2+R3 work: analyze*.py + module scripts, REPORT*.md, READY_FOR_REVIEW*.md, findings*.json |
| `codex-analysis/` | Agent B's R1+R2+R3 work: same structure, plus `round2/` and `round3/` subfolders with per-round tables |
| `reviews/` | Each round's cross-review documents and the three comparison memos (`COMPARISON_MEMO.md`, `COMPARISON_MEMO_R2.md`, `COMPARISON_MEMO_R3.md`) |
| `data/` | Small derived data: lookups, manifests, and JSON summary files. Statcast parquets are NOT included — pull them via `pybaseball` using the `data_pull.py` scripts in either analysis folder. |
| `requirements.txt` | Python dependencies covering both methodology lanes |

## Reproducing the analysis

```bash
# Set up Python environment (3.11+)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# R3 (the final / shipped version) — each agent runs from its own folder
cd claude-analysis && python analyze_r3.py
cd ../codex-analysis && python analyze_r3.py
```

Both agents share the `data/` folder for cached Statcast pulls. They check file existence before re-pulling, so concurrent runs are safe (the second one just consumes the first's cache).

The Statcast historical pulls (2015-2025) total ~30-50 GB; not included in the repo. The MLB Stats API resolver in `data_pull.py` (used for 2026 debuts like Murakami) uses `https://statsapi.mlb.com/api/v1/people/search` with no auth required.

## Caveats

- Codex's prediction intervals are raw QRF (over-cover by ~5pp on validation); deliberately conservative.
- Codex's sleeper floor required a positive preseason wOBA baseline; some sleepers (Pereira, Barrosa) have priors below the .250 line we'd ideally use.
- Codex's "fake hot" rule used a population residual standard deviation, not a per-player Bayesian SD.
- Claude's contact-quality blend was validated on a single 2022-2024 → 2025 holdout, not k-fold CV; coefficient stability across folds is not established.
- Murakami's prior is a 60-PA league-average proxy because no NPB-translation factor was built.
- No defensive metrics, no park-factor adjustments, no causal claims about why any individual player's profile looks the way it does.

## License

Same as the repo's [LICENSE](../LICENSE).

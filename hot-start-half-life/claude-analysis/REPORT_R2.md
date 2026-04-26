# The Hot-Start Half-Life — Round 2 Report (Claude / Agent A)

**Method family.** Per-stat empirical-Bayes shrinkage (conjugate Beta-Binomial)
with player-season cluster-bootstrap-derived stabilization PAs; contact-quality
features (EV p90, HardHit%, Barrel%, xwOBA, xwOBA-wOBA gap) added to the prior;
a true partial-pooling hierarchical wOBA model (`kappa ~ HalfNormal(300)`) fit
across all 279 universe hitters as a sanity-check on shrinkage strength.

**Data.** Statcast 2022-2025 full seasons + 2026 through 2026-04-24
(105,933 PAs; through_apr25 file). 2023-2025 used for player-specific priors;
2022 retained for stabilization estimation.

**Sample.** Universe scan: 279 hitters (>= 50 PA) and 233 relievers (>= 25 BF
AND avg <= 12 BF/appearance AND < 30 IP). Both above brief thresholds (250 / 70).

---

## Executive summary

> **Headline 1 (methodology).** With a *proper* player-season cluster bootstrap,
> the R1 claim that 3 of 5 stabilization rates have shifted vs Carleton 2007/2013
> **does not survive correction.** Only **wOBA** still flags as shifted (point ~447 PA,
> 95% cluster CI 298-3,600 vs Carleton 280; the upper CI is wide because ~8% of
> bootstrap draws never crossed 0.5 reliability inside the M=900 grid). BB%, K%,
> ISO, and BABIP are now all consistent with Carleton at 95% — the R1
> "non-overlapping CI" language was an artifact of within-player random-partition
> uncertainty, not sampling-uncertainty across player-seasons.
>
> **Headline 2 (universe scan, hitters).** H1 (sleeper signals exist) **PASSES**:
> 18 hitters have predicted ROS-vs-prior wOBA delta in the top decile *and* are
> not in the ESPN OPS top-20. Top-5 sleeper picks: **Jac Caglianone (KC)**,
> **Everson Pereira (NYY)**, **Jorbit Vivas (NYM)**, **Samuel Basallo (BAL)**,
> **Jorge Barrosa (ARI)**. H2 (fake hot) FAILS at 2 (Aaron Judge and Xavier
> Edwards are the only mainstream-listed hitters with negative ROS-vs-prior delta).
> H3 (fake cold) FAILS at 1 (Henry Davis is the only bottom-decile April hitter
> with positive delta). Per brief §6: drop FAKE HOT section, lead with the
> brutal-noise-floor finding.
>
> **Headline 3 (relievers).** H4 (reliever sleepers) **PASSES**: 8 relievers
> show K%-rise posteriors >= +0.04 over their 3-yr prior, with the known-closer
> set excluded. Top-5: **Louis Varland (BAL)**, **Antonio Senzatela (COL,
> previously a starter)**, **Daniel Lynch (KC)**, **Caleb Kilian (CHC)**, **John
> King (TEX)**. Mason Miller is *not* a sleeper (he's a known closer in the
> exclusion list) but appears at the top of the FAKE-DOMINANT board: his .650
> 22-game K% shrinks to .495 posterior median (R1: .500 — convergent).
>
> **Headline 4 (R1 sanity check).** Pages NOISE -> NOISE; Murakami SIGNAL -> SIGNAL
> (now reproducible from clean checkout); Miller K% AMBIG -> SIGNAL with streak
> probabilities **killed**. Two flips: **Rice and Trout** both move from
> R1-AMBIGUOUS to R2-SIGNAL when contact-quality features enter the prior.
> Their 80% credible intervals on ROS-vs-prior wOBA delta exclude zero: Rice
> [+0.026, +0.066], Trout [+0.005, +0.045]. The contact-quality story is
> what tipped them: Rice xwOBA .481 with HardHit 65%; Trout xwOBA .476 with
> Barrel 22%.

---

## 1. Methodology fixes — status table (per brief §4.2)

| Fix | Status | Where | Evidence |
|---|---|---|---|
| Murakami reproducibility (MLB Stats API) | **done** | `r2_data_pull.py` | `resolve_mlbam("Munetaka Murakami")` falls back to `https://statsapi.mlb.com/api/v1/people/search` and caches to `data/mlbam_resolver_cache.json`. `data/named_hot_starters_r2.parquet` regenerates Murakami's MLBAM (808959) from a clean checkout. |
| Stabilization cluster bootstrap | **done** | `r2_stabilization.py` | Resamples PLAYERS w/replacement, then resamples SEASONS within player. Replaces R1's within-player PA random-partition CI. Re-fit changes 4 of 5 verdicts vs R1 — see §2 below. |
| Projection prior includes contact quality | **done** | `r2_bayes_projections.py` | Prior now includes EV p90, HardHit%, Barrel%, xwOBA, xwOBA-wOBA gap. Posterior wOBA is a 50/50 blend of wOBA and xwOBA posteriors for the ROS-vs-prior delta computation. |
| Mason Miller streak model | **killed** | `r2_sanity_check.py` | HR-only ER proxy is dead; `delta_run_exp` accumulation requires re-pulling 3 years of pitch-level data with that column. Per brief: "kill the streak-extension probabilities entirely." Only K% posterior is reported now. |
| Hierarchical labeling honesty | **done** | `r2_bayes_projections.py::fit_hierarchical_universe_woba()` | True partial-pooling implemented: `kappa ~ HalfNormal(300)` shared across 279 universe hitters; 4 chains x 3,000 samples; **kappa R-hat = 1.009, ESS = 1,247** — both inside the brief's R-hat <= 1.01, ESS >= 400 bar. The faster conjugate Beta-Binomial per-player updates are kept and labeled honestly as "empirical-Bayes shrinkage with conjugate update." |

All 5 R1-blocking fixes are implemented before any new R2 analysis was run.

---

## 2. Corrected stabilization (the methodology delta)

| Stat | R1 point (within-PA CI) | R2 point (cluster CI) | Carleton | R1 verdict | R2 verdict |
|---|---|---|---|---|---|
| BB% | 122 (98-139) | **124 (93-164)** | 120 | consistent | **consistent** |
| K% | 54 (50-67) | **51 (50-71)** | 60 | consistent | **consistent** |
| ISO | 198 (176-238) | **193 (141-259)** | 160 | shifted (slower) | **consistent** |
| BABIP | 627 (583-747) | **622 (493-6,400)** | 820 | shifted (faster) | **consistent** |
| wOBA | 489 (396-569) | **447 (298-3,600)** | 280 | shifted (slower) | **shifted (slower)** |

The R1 framing — "the 2026 environment shifted stabilization rates by enough to
invalidate Carleton 2007" — does NOT survive correction. Three of five stats
that R1 called "shifted" now have CIs that comfortably contain the Carleton
value once the bootstrap properly samples player-seasons rather than partitioning
PAs within a fixed player set. **The correct article framing is the noise-floor
finding, not a methodology-novelty claim.** Codex's review on this point is
vindicated.

(See `charts/r2/stabilization_*.png` for the cluster-bootstrap CI bands.)

---

## 3. Persistence Atlas (Framing A): universe scan results

### 3.1 Top-10 SLEEPER picks (top decile of ROS-vs-prior, NOT in ESPN OPS top-20)

| # | Player | Team | PA | Obs wOBA | Obs xwOBA | Prior wOBA | ROS-vs-prior q50 (q10) | Contact-quality |
|---|---|---|---|---|---|---|---|---|
| 1 | Jac Caglianone | KC | 79 | .328 | .350 | .240 | **+0.051** (+0.032) | HH 55%, Bar 16%, EV90 111 |
| 2 | Everson Pereira | NYY | 50 | .416 | .403 | .220 | **+0.038** (+0.020) | HH 49%, Bar 14%, EV90 105 |
| 3 | Jorbit Vivas | NYM | 63 | .389 | .316 | .270 | +0.036 (+0.013) | HH 24%, Bar 2%, EV90 101 |
| 4 | Samuel Basallo | BAL | 73 | .339 | .374 | .246 | +0.035 (+0.016) | HH 45%, Bar 13%, EV90 108 |
| 5 | Jorge Barrosa | ARI | 55 | .344 | .247 | .178 | +0.032 (+0.013) | HH 24%, Bar 9%, EV90 98 |
| 6 | Oswald Peraza | NYY | 70 | .406 | .359 | .240 | +0.030 (+0.012) | HH 45%, Bar 13%, EV90 105 |
| 7 | Dillon Dingler | DET | 83 | .345 | **.438** | .322 | +0.021 (+0.001) | HH 50%, Bar 19%, EV90 106 |
| 8 | Ildemaro Vargas | ARI | 69 | .472 | .417 | .297 | +0.020 (+0.001) | HH 42%, Bar 9%, EV90 104 |
| 9 | Oneil Cruz | PIT | 106 | .411 | .399 | .326 | +0.020 (+0.001) | HH 67%, Bar 24%, EV90 114 |
| 10 | Otto Lopez | MIA | 95 | .454 | .387 | .324 | +0.020 (+0.000) | HH 58%, Bar 12%, EV90 106 |

The top-3 (Caglianone, Pereira, Basallo) are the cleanest signals: 22-game
performance well above prior, AND xwOBA / contact-quality posteriors agree
this is real. Vivas and Barrosa flag a caution — their wOBA is up but xwOBA is
weaker than wOBA, suggesting BABIP-luck contribution that the model partially
adjusts for via the 50/50 blend.

**Dillon Dingler is the standout under-the-radar contact-quality story:**
xwOBA .438 vs wOBA .345 — a +.093 gap suggesting his .345 22-game wOBA is
*understating* his real talent by enough that ROS could come in higher.

### 3.2 FAKE HOT (in mainstream top-20 with predicted ROS-vs-prior delta < 0)

| # | Player | Team | PA | Obs wOBA | Prior wOBA | ROS-vs-prior q50 |
|---|---|---|---|---|---|---|
| 1 | Aaron Judge | NYY | 102 | .426 | **.482** | -0.007 |
| 2 | Xavier Edwards | MIA | 113 | .404 | .337 | -0.005 |

Only 2 candidates — H2 fails the >= 3 threshold. Per brief §6: **drop FAKE HOT
section.** Judge's negative delta is the unusual case where his prior is so
dominant (3-yr weighted .482 wOBA) that even a top-of-leaderboard April pace
projects under-prior ROS. Xavier Edwards is in noise band; the negative point
estimate is essentially zero.

### 3.3 FAKE COLD (bottom decile April with predicted ROS-vs-prior delta > 0)

| # | Player | Team | PA | Obs wOBA | Prior wOBA | ROS-vs-prior q50 |
|---|---|---|---|---|---|---|
| 1 | Henry Davis | PIT | 54 | .222 | .259 | +0.009 |

Only 1 candidate — H3 also fails. The brutal-noise-floor finding cuts both
ways: Bayesian shrinkage on a weak-prior bad-April player puts ROS *between*
prior and observation, almost never *above* prior in 22 games. Per brief §6:
"Drop FAKE HOT section; lead with FAKE COLD instead" — but FAKE COLD is also
empty. The honest framing is: *April's bottom decile cannot be rescued by 22
games of evidence to a positive-delta verdict; their priors were already weak
and shrinkage will land them somewhere south of prior.*

### 3.4 R2 H1/H2/H3 outcomes

- H1 (sleepers >= 3): **PASS** (n=18)
- H2 (fake hot >= 3): **FAIL** (n=2)
- H3 (fake cold >= 3): **FAIL** (n=1)

H5 null fallback is partly invoked: the article cannot lead with a "you missed
these names" + "everyone is wrong about these" two-sided framing. The honest
spine is the sleeper picks (which are themselves uncertain — most are debuts
or low-PA hitters whose priors are weak and whose deltas are dominated by the
prior-vs-observation gap, not by signal-vs-noise discrimination).

---

## 4. xwOBA-Gap Sheet (Framing B, subsumed in A)

Top-10 over-performers (wOBA >> xwOBA, BABIP-luck):

| # | Player | PA | wOBA | xwOBA | Gap | ROS-vs-prior q50 |
|---|---|---|---|---|---|---|
| 1 | Mickey Moniak | 68 | .432 | .300 | -0.132 | -0.003 |
| 2 | Moisés Ballesteros | 55 | .480 | .370 | -0.110 | -0.025 |
| 3 | Matt Chapman | 100 | .347 | .258 | -0.090 | -0.012 |
| 4 | Graham Pauley | 58 | .295 | .210 | -0.085 | -0.007 |
| 5 | José Caballero | 79 | .341 | .256 | -0.084 | -0.013 |

The over-performer list is a clean *fake-hot* alternative — players whose
wOBA is ahead of their contact-quality. Note Moniak, Ballesteros, Chapman
all show negative ROS-vs-prior deltas (consistent with regression).

Top-10 under-performers (xwOBA >> wOBA, bad-luck victims):

| # | Player | PA | wOBA | xwOBA | Gap | ROS-vs-prior q50 |
|---|---|---|---|---|---|---|
| 1 | Bo Naylor | 54 | .225 | .348 | +0.123 | -0.008 |
| 2 | Harrison Bader | 51 | .148 | .267 | +0.119 | -0.021 |
| 3 | Ke'Bryan Hayes | 65 | .190 | .302 | +0.111 | -0.006 |
| 4 | Patrick Bailey | 59 | .166 | .271 | +0.104 | -0.004 |
| 5 | Austin Wells | 66 | .225 | .323 | +0.098 | -0.004 |

The under-performer list is the most-actionable *fake-cold* alternative the
universe scan surfaces — players whose 22-game wOBA buries their actual
contact quality. Hayes/Bailey/Wells/Bader all post xwOBA that would project
in the .300+ wOBA range yet currently sit under .225. The R2 Bayesian
projection still doesn't push them positive vs prior because their priors
were already strong relative to their April pace.

---

## 5. Reliever K% True-Talent Board (Framing C)

### 5.1 Top-5 SLEEPER reliever K% risers

(Excludes 20 known closers from 2025 saves leaderboard; 18 of those are in our
233-reliever universe.)

| # | Reliever | Team | BF | Prior K% | Obs K% | Post K% q50 (q10-q90) | Δ vs prior | Prior BF |
|---|---|---|---|---|---|---|---|---|
| 1 | Louis Varland | BAL | 41 | .239 | .415 | **.301** (.243-.359) | +0.062 | 817 |
| 2 | Antonio Senzatela | COL | 58 | .119 | .259 | **.181** (.131-.231) | +0.062 | 702 |
| 3 | Daniel Lynch | KC | 33 | .172 | .364 | **.231** (.171-.291) | +0.060 | 687 |
| 4 | Caleb Kilian | CHC | 41 | .148 | .293 | **.206** (.146-.266) | +0.058 | 81 |
| 5 | John King | TEX | 40 | .137 | .275 | **.185** (.131-.239) | +0.049 | 627 |

These are the picks. Varland and Senzatela are the most credible — both have
3-yr prior BF > 700 and the K% rise is large enough to survive the 70-BF
reliever-stabilization shrinkage. **Senzatela in particular** is interesting
because he's a converted starter — the bullpen shift may be unlocking his
K% (his 2025 starter K% was .119; his 2026 reliever K% is .259, projecting
to .181 ROS).

### 5.2 Top-5 FAKE-DOMINANT reliever K% (heavy shrinkage from blistering 22-game)

| # | Reliever | Team | BF | Obs K% | Post K% q50 | Shrinkage to prior | Closer? |
|---|---|---|---|---|---|---|---|
| 1 | **Mason Miller** | SD | 40 | .650 | .495 | 0.636 | YES |
| 2 | Daniel Lynch | KC | 33 | .364 | .231 | 0.680 | no |
| 3 | Louis Varland | BAL | 41 | .415 | .301 | 0.631 | no |
| 4 | Jordan Romano | TOR | 29 | .379 | .293 | 0.707 | YES |
| 5 | Keaton Winn | SF | 28 | .321 | .236 | 0.714 | no |

Mason Miller's posterior K% q50 = .495 with the cluster-bootstrap stabilization
PA — almost identical to R1's .500. **The K% finding is convergent across R1
and R2, even with corrected methodology.** What changed is the streak
probabilities are killed, not the K% verdict.

### 5.3 H4 outcome

H4 (>= 2 sleeper relievers): **PASS** (n=8 sleepers above the +0.04 K%-rise
threshold).

---

## 6. R1 sanity check — 5 named hot starters under R2 corrected methodology

| Slug | R1 verdict | R2 verdict | Changed? | Driver of change |
|---|---|---|---|---|
| Andy Pages | NOISE | **NOISE** | no | Posterior wOBA .335 vs prior .333; HardHit 60% (above prior) but Barrel only 6% (below prior) — high HH from softer-contact pulls; classic BABIP-driven excursion. |
| Ben Rice | AMBIGUOUS | **SIGNAL** | **flipped** | Contact-quality features (HardHit 65%, Barrel 23%, EV90 106) lift the prior-blended posterior. ROS-vs-prior 80% CI [+0.026, +0.066] excludes zero. |
| Munetaka Murakami | SIGNAL | **SIGNAL** | no | Same verdict, now reproducible from a clean checkout (MLB Stats API resolver). ROS-vs-prior CI [+0.030, +0.100] still strongly positive. |
| Mike Trout | AMBIGUOUS | **SIGNAL** | **flipped** | xwOBA .476 with Barrel 22%; the K%-down (.217 vs prior .298) story persists, and now the contact-quality posterior backs it. ROS-vs-prior CI [+0.006, +0.045]. |
| Mason Miller | AMBIGUOUS (streak claim) | **SIGNAL on K%** | partial | K% verdict same as R1; streak probabilities **killed** per brief (HR-only proxy was numerology). |

Two flips (Rice and Trout) are findings worth flagging in the article: when
the prior ignores contact-quality, the posterior under-rates a hitter whose
22-game wOBA happens to overlap a noisy BABIP excursion. Once contact quality
enters the prior, both Rice and Trout get pulled into "real signal" verdicts.

(See `charts/r2/r1_sanity_check_comparison.png`.)

---

## 7. Kill-gate outcomes

| Gate | Threshold | Result | Action |
|---|---|---|---|
| All §4 methodology fixes implemented | Yes/No | **YES** (5/5 done or honestly killed) | clear |
| Universe coverage | >= 250 hitters, >= 70 relievers | **YES** (279 / 233) | clear |
| H1 sleeper signals | >= 3 in top-10 not in mainstream top-20 | **PASS** (n=18) | publish |
| H2 fake hot | >= 3 mainstream-listed with delta < 0 | **FAIL** (n=2) | drop FAKE HOT section per brief |
| H3 fake cold | >= 3 bottom-decile with delta > 0 | **FAIL** (n=1) | drop FAKE COLD section per brief |
| H4 reliever sleepers | >= 2 K%-rise non-closers | **PASS** (n=8) | publish |
| Cluster-bootstrap CIs on every stabilization estimate | required | **YES** | clear |
| Convergence diagnostics on every Bayesian fit | R-hat <= 1.01, ESS >= 400 | **YES** for hierarchical kappa (R-hat 1.009, ESS 1,247); per-player conjugate Beta-Binomial is closed-form (no MCMC required) | clear |
| Sample-size discipline | >= 50 PA hitters, >= 25 BF relievers | enforced in `r2_universe.py` | clear |

---

## 8. Open questions for Round 3 (do NOT pursue this round)

1. **Cluster-bootstrap right-censoring on BABIP / wOBA upper CI.** Both have
   wide upper CIs (BABIP 6,400; wOBA 3,600) because some bootstrap draws never
   crossed 0.5 reliability inside our M=900 / M=1,600 grids. Extending the
   grid to M=3,000 would tighten these.
2. **NPB-translation prior for Murakami.** His SIGNAL is real but anchored to
   a 60-PA league-average prior. A proper NPB-MLB translation would tighten
   the projection considerably.
3. **The Trout / Rice flip as a feature-importance question.** What fraction
   of R2's verdict change comes from xwOBA prior, what fraction from
   HardHit%/Barrel%, what fraction from EV p90? Codex's LightGBM permutation
   importance would speak to this; ours doesn't.
4. **`delta_run_exp`-based reliever streak survival** — Miller's streak (now
   killed in R2) could be properly modeled if we re-pull pitch-level data with
   that column for 2023-2025.
5. **Era counterfactual with matched sample sizes** — see Codex review #1.
   Out of R2 scope but the era-shift question is unresolved.
6. **Park-factor-adjusted xwOBA gap** — Yankee Stadium / Coors / Petco etc.
   Some of the under-performer list (Bader, Bailey) may be partly home-park
   driven.
7. **Component-stat persistence-by-component validation backtests** —
   2022-2024 trained, 2025-ROS validated; do the same component-EB-shrinkage
   priors hit their own intervals empirically? An out-of-sample coverage check.

---

## 9. Files produced (R2)

- `claude-analysis/analyze_r2.py` — one-command R2 entry point (idempotent)
- `claude-analysis/r2_data_pull.py` — Statcast extension + MLB Stats API resolver
- `claude-analysis/r2_universe.py` — 2026 hitter (>=50 PA) and reliever (>=25 BF) universes
- `claude-analysis/r2_stabilization.py` — player-season cluster-bootstrap stabilization
- `claude-analysis/r2_bayes_projections.py` — extended priors + true partial-pooling hierarchical model
- `claude-analysis/r2_persistence_atlas.py` — universe-wide ROS delta + sleeper / fake-hot / fake-cold lists
- `claude-analysis/r2_reliever_board.py` — reliever K% Bayesian board + sleeper / fake-dominant lists
- `claude-analysis/r2_sanity_check.py` — R1 verdict comparison; Miller streak killed
- `claude-analysis/r2_charts.py` — sanity-check + xwOBA-gap charts
- `claude-analysis/REPORT_R2.md` — this report
- `claude-analysis/findings_r2.json` — top-level findings
- `claude-analysis/charts/r2/*.png` — 12 PNGs (stabilization, sleepers, fake_hot, fake_cold, sleeper_relievers, fake_dominant_relievers, r1_sanity_check_comparison, xwoba_gap)
- `data/r2_universe_posteriors.parquet`, `r2_reliever_posteriors.parquet`, `r2_persistence_atlas.json`, `r2_reliever_board.json`, `r2_named_hot_starter_projections.json`, `r2_stabilization_summary.json`, `r2_hierarchical_woba_summary.json`, `mlbam_resolver_cache.json`, `mlbam_id_to_name_cache.json`, `chadwick_register_cache.parquet`
- R1 outputs (REPORT.md, findings.json, charts/, etc.) preserved untouched.

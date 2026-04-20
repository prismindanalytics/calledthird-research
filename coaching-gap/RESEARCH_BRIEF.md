# The Coaching Gap — Research Brief (v3: Evolution-Centered)

**Status:** Week 1 data harmonized; hybrid labeling applied; thesis refined
post-probe findings. Career-history pull for 474 active pitchers in progress.
**Target:** 2-article flagship. `How Pitchers Evolve` leads; `The Coaching
Gap` follows and points back to it.
**Est. effort:** 6-8 weeks
**Principal axis:** *Career evolution archetypes × hitter response patterns*

---

## 1. What v2 Taught Us (Why v3 Exists)

The v0 probe + hybrid-labeling probe revealed three things that reshape the
thesis:

1. **Pitchers naturally split into three evolution tiers** by arsenal
   stability (measured via cross-season Adjusted Rand Index of physics-based
   pitch clusters):
   - **Stable (28%, ARI ≥ 0.90):** arsenal shapes rock-solid across seasons
   - **Middle (45%, 0.70 ≤ ARI < 0.90):** year-to-year drift, not total
     reinvention
   - **Evolvers (27%, ARI < 0.70):** genuinely different pitchers across
     seasons — added/dropped pitches, re-shaped existing ones

2. **Arsenal volatility as a model *feature* adds zero AUC lift** — but as
   a *stratifier* it reveals the main story.

3. **The coaching gap is 2× larger on stable pitchers than on volatile
   pitchers** (0.0042 vs 0.0033 xwOBA, thr=0.70). The hypothesis I expected
   — "teams fail to update on evolving pitchers" — got flipped. Reality:
   teams leave the most edge on the table against pitchers whose patterns
   *should* be maximally scoutable.

v3's central question becomes: **Why do hitters fail on the most-predictable
pitchers? And how does a pitcher's evolution shape — or mask — that
predictability?**

---

## 2. Thesis (v3)

**Primary thesis:** MLB pitchers follow identifiable *evolution archetypes*
traced across their careers — stable veterans, drift-pattern mid-career
arms, late-career transformers, rookie-phase explorers, post-injury
reinventors, and converted relievers. Each archetype produces a different
scouting challenge, and each elicits a systematically different hitter
response.

**Secondary thesis (the Coaching Gap):** The failure of hitters to exploit
predictability correlates with a pitcher's archetype — but *not* in the
direction naive scouting would predict. Stable veterans create the largest
gap despite being the most public-data-predictable. We believe this is
because: (a) teams pre-commit to plans against volatile pitchers but
improvise against stable ones, and (b) stable patterns include subtle
count-state habits that don't show up in video review but do in career
aggregates.

**Tertiary (methodological):** A per-pitcher-season physics-based pitch
clustering outperforms MLBAM labels for the 28% of pitchers whose shapes
are stable (ARI ≥ 0.90), but fails for the other 72% where arsenal
evolution drives real label drift. The right answer is hybrid labeling,
which we operationalize.

---

## 3. Hypotheses (Tightened)

**H1 — Archetype taxonomy exists and stabilizes.**
A clustering of career-trajectory features produces 4-8 distinct archetypes
whose membership is stable under bootstrap resampling (median cosine
similarity ≥ 0.85) and whose distinguishing features are interpretable
(velocity arc, arsenal size, role transitions, volatility tier, debut age).

**H2 — Predictability trajectory itself is a per-pitcher signature.**
Fit Layer 2 (contextual pitch model) per-season per-pitcher. Each pitcher
gets a *predictability trajectory* — AUC of their pitch-choice given
context, across seasons. Some pitchers start predictable and *become* more
varied (learn to deceive); some do the opposite; some stay flat. These
trajectories cluster into a second-order signature orthogonal to arsenal
volatility.

**H3 — Coaching gap is archetype-dependent.**
The team-level xwOBA gap on predictable pitches differs significantly
across archetypes. Specifically:
  - Stable-vet gap ≥ Middle gap ≥ Volatile-evolver gap (at matched
    predictability threshold)
  - The rank order is consistent across the 2024, 2025, and 2026-YTD seasons

**H4 — Evolution precedes measurable role / outcome change.**
Arsenal-volatility spikes cluster 6-18 months before observable performance
inflection points (role transitions, IL stints, velocity drops). This makes
volatility a *leading indicator*, and the most-overlooked scouting signal.

---

## 4. Data

### 4.1 Primary corpus (NEW in v3)
**Active-pitcher career history.** 474 pitchers who appeared in MLB 2026 to
date. For each, full Statcast history from their MLB debut (or 2015) to
2026-04-30.
  - ~8M pitches total expected (vs. 30M if we pulled all historical pitchers)
  - Per-pitcher parquet in `data/career_history/pitcher_<id>.parquet`
  - Pull driver: `scripts/pull_active_career_history.py`

### 4.2 Pre-existing corpus
  - Full-year parquets 2022, 2023, 2024 already pulled (`data/harmonized/raw/`)
  - Full 2025 parquet (739,820 pitches) — used for v0 probes
  - 2026 YTD via team-challenge-iq + schlittler daily pulls (474 pitchers,
    ~80K pitches so far)

### 4.3 Derived
  - `hybrid_labeled_pitches.parquet` — unified label column + arsenal
    volatility score per pitch
  - `ari_per_pitcher.csv` — ARI stability diagnostics per qualifying pitcher
  - Pitch Tunneling Atlas signatures (input to Layer 2)
  - Umpire personality dataset

### 4.4 External joins
  - MLB play-by-play (responsible-pitcher/catcher attribution — reuse from
    Fireman's Dilemma)
  - IL / injury records (Baseball Prospectus injury archive or MLB.com
    transactions scraping)
  - Role designations (starter / reliever / swing) — derivable from
    Statcast game logs

---

## 5. Feature Architecture

### 5.1 Evolution features (per pitcher, per season)
| Feature | Definition |
|---|---|
| `arsenal_volatility` | 1 - median cross-season ARI for this pitcher |
| `arsenal_entropy` | Shannon entropy of pitch-type usage distribution |
| `arsenal_size` | # distinct pitch types thrown with usage ≥ 5% |
| `primary_pitch_persistence` | % of years primary pitch type is unchanged |
| `new_pitch_this_season` | flag: pitch type used ≥5% but <2% year prior |
| `dropped_pitch_this_season` | flag: pitch type <2% but ≥5% year prior |
| `velocity_trajectory_slope` | OLS slope of annual velo, last 3 yrs |
| `velocity_peak_distance` | seasons since career peak velo |
| `release_point_drift` | cross-year std of release_pos_x, release_pos_z |
| `role_transition_this_season` | flag: starter↔reliever vs prior year |
| `age_days` | days since MLB debut (proxy for experience) |
| `post_injury_flag` | within 12 mo of IL stint |
| `seasons_at_role` | consecutive seasons in current role |

### 5.2 Predictability-trajectory features (new in v3)
Fit Layer 2 *per-season-per-pitcher* → extract:
| Feature | Definition |
|---|---|
| `pred_auc_career` | AUC of pitch-choice prediction, full career |
| `pred_auc_trajectory_slope` | OLS slope of per-season AUC |
| `pred_auc_trend_type` | ascending / descending / flat / u-shaped / inverted-u |
| `pred_auc_latest` | current-season AUC (the "how predictable are you RIGHT NOW") |
| `pred_auc_volatility` | std of AUC across seasons |

### 5.3 Contextual features (retained from v2)
Count state, pitch sequence, TTO, batter handedness, in-AB location
history, tunnel family, base-out state, leverage, catcher, within-game
velo decay, rest, batter-vs-pitcher history, umpire zone shape, park.

### 5.4 Batter features
Career whiff% vs each physics-defined pitch shape; batter's lagged
coaching-gap score; batter career arc vs pitcher archetype.

---

## 6. Archetype Taxonomy (Hypothesized)

We don't pre-commit to a fixed list — the clustering will tell us — but we
expect something like:

| Archetype (working name) | Volatility | Age phase | Velo trajectory | Role stability |
|---|---|---|---|---|
| **Stable Vet** | Low | Mid-late | Flat/declining | High |
| **Ascending Sophomore** | Mid | Early | Rising | Mid |
| **Drift Vet** | Mid | Mid-late | Declining | High |
| **Reinventor** | High | Any | Any | High |
| **Converted Reliever** | High | Mid | Rising peak velo | Low (transition) |
| **Post-Injury Rebuilder** | High | Any | Dropping then recovering | Mid |
| **Rookie Explorer** | High | Early | Stable (short window) | Low |
| **Late-Career Transformer** | High | Late | Declining | Mid-High |

**Deliverables:** per-archetype named case studies in the companion article,
preferably 1 case study per archetype (e.g., Zack Wheeler as Stable Vet,
Jared Jones as Ascending Sophomore, Corbin Burnes as Reinventor).

---

## 7. Methodology — Three Layers (Evolution-Centered)

### 7.1 Layer 1 — Career Trajectory Model
**Per-pitcher, using 2015-2026 per-pitcher parquets.**

1. Build monthly trajectory matrix for each pitcher:
   - Monthly usage % per physics cluster (if stable) OR per MLBAM pitch_type
   - Monthly mean velocity, mean pfx_x, mean pfx_z, mean spin per cluster
   - Role indicator (starter / reliever / swing)
2. Fit smoothing splines to each trajectory
3. Detect change-points via PELT algorithm (L2 cost, auto-scaled BIC penalty;
   synthetic gate passed: 100% recovery, 0% FDR)
4. Compute the 13 evolution features from §5.1
5. Embed each pitcher via LSTM autoencoder → 32-dim signature vector
   (bootstrap stability gate: median cosine ≥ 0.85)
6. GMM clustering on signature vectors → archetypes (BIC-selected K, cap 12)
7. Output per-pitcher: archetype probabilities, change-point locations,
   evolution-feature summary

### 7.2 Layer 2 — Contextual Pitch Model (unchanged architecture, new population)
Dual-agent; train on 2015-2025 with 2026 YTD as holdout.
  - **Agent A (Claude):** Hierarchical Bayesian multinomial logistic with
    GP temporal priors, archetype-level random effects nested within pitcher
    random effects, posterior uncertainty
  - **Agent B (Codex):** LightGBM multinomial with archetype × count
    pairwise interactions, SHAP attribution, bootstrap ensembles (N=100)
  - Train per-season-per-pitcher copies to extract predictability
    trajectories (§5.2)

### 7.3 Layer 3 — Sequential Within-AB
  - **Agent A:** Transformer (2 layers, 4 heads, d=64) over pitch sequence,
    batter history, pitcher recent form
  - **Agent B:** LSTM over same inputs
  - Fuse with Layer 2 logits via learned gating; kill Layer 3 if it doesn't
    add ≥ 0.005 AUC

### 7.4 Coaching Gap Decomposition (expanded from v2)
Compute four gap types per batter, aggregated to team:

1. **Archetype gap:** On predictable pitches from `Stable Vet` pitchers, do
   hitters adjust? Repeat per archetype. Produces an archetype × team
   matrix. *New in v3 — replaces v2's "archetype gap."*
2. **Recency gap:** Gap on pitches flagged predictable by career features
   but NOT by 30-day rolling features (i.e., a pitcher's habit you'd only
   know from history).
3. **Regime gap:** Gap in the 30 days after a detected Layer-1 change-point.
   Do hitters update to the new regime?
4. **Count-state gap:** Gap concentrated in counts where predictability
   varies most across pitchers (3-2, 0-0, 2-0).

For each: permutation test (500 iters); kill if team-level spread < 0.012
xwOBA OR p ≥ 0.05.

---

## 8. Kill Criteria (Updated)

**Pre-modeling gates:**
| Gate | Threshold | Status |
|---|---|---|
| Physics clustering ARI | ≥ 0.90 median | **FAILED (0.819)** — fallback: hybrid labels ✓ |
| Change-point synthetic recovery | ≥ 80% @ FDR ≤ 0.1 | **PASSED (100% / 0%)** ✓ |
| Active-pitcher career-history pull | ≥ 400 pitchers with ≥ 3 seasons of data | In progress |
| Archetype clustering stability | Bootstrap cosine ≥ 0.85 | TBD |
| Signature cosine stability | ≥ 0.85 across 10 bootstraps | TBD |

**Model-performance gates:**
| Gate | Threshold | Failure action |
|---|---|---|
| Full Layer 2 AUC lift over 30-day baseline | ≥ +0.02 | Kill longitudinal claim; publish v1-scale |
| Inter-agent top-50 stability | Spearman ρ ≥ 0.70 | Publish structure not rankings |
| Predictability-trajectory signal | ≥ 60% of pitchers have trend coef p < 0.1 | Drop §5.2 features |

**Publishability gates:**
| Gate | Threshold | Failure action |
|---|---|---|
| Archetype gap spread per archetype | ≥ 0.012 xwOBA best-worst team | Kill that archetype's chart |
| At least 1 of 4 gap types surviving | — | If zero, pivot to null-result article |
| Archetype interpretability | All K clusters nameable from top-3 distinguishing features | Reduce K |

**Null-result article pre-approved:** if no gap type survives, headline
becomes *"MLB Scouting Is Better Than the Data Suggests"* — still a
publishable finding.

---

## 9. Deliverables

### 9.1 Flagship article (~2,800 words) — NEW LEAD
**`How Pitchers Evolve: An Atlas of MLB Arsenal Trajectories`**
- Archetype taxonomy with interactive switcher
- 6-8 named case studies, one per archetype
- The physics-based pitch reclustering methodology (with hybrid
  fallback honesty)
- Per-pitcher trajectory visualizations (usage % × season × cluster)
- Predictability-trajectory sub-section

### 9.2 Secondary article (~2,200 words)
**`The Coaching Gap: Why Teams Miss What They Can See`**
- Team × archetype coaching-gap matrix (the money chart)
- Named hitter case studies: elite/weak at each archetype
- Honest acknowledgment that the biggest gap is on stable-vet pitchers —
  *opposite of the naive "teams don't update on evolvers" hypothesis*
- Links back to Evolution Atlas for methodology

### 9.3 Explore tools
- **Career Trajectory Card:** enter a pitcher, see trajectory + change-points +
  archetype + current predictability
- **Coaching Gap Matrix:** team × archetype heatmap, updated nightly once
  live

### 9.4 Public repo artifact (at launch, not before)
Full `coaching-gap/` folder merged into `calledthird-research`.

---

## 10. Timeline (Revised)

| Week | Focus | Gate |
|---|---|---|
| 1 (this week) | Hybrid label + active-pitcher pull + brief finalization | Pull complete |
| 2 | Layer 1 — evolution features + archetype clustering | Stability gates |
| 3 | Layer 2 Agent A + B in parallel | AUC lift ≥ +0.02 |
| 4 | Layer 3 + predictability-trajectory extraction | Layer 3 lift ≥ +0.005 |
| 5 | Coaching gap computation (4 types) + permutation tests | ≥1 gap type clears |
| 6 | Cross-review + comparison memo | — |
| 7 | Article drafting, charts, Explore build | — |
| 8 | Deploy + launch + public repo push | — |

Week 1 is substantially done. The revised thesis makes Week 2's Layer 1
the central story, not a supporting one.

---

## 11. Risks & Mitigations (Updated)

| Risk | Mitigation |
|---|---|
| Archetype clustering unstable | Use bootstrap gate at §8; fall back to 3-tier volatility bins |
| Predictability trajectories are noise | §5.2 features gated at 60% of pitchers having p<0.1 trends |
| Coaching gap signal doesn't scale | v0 already showed it exists (p=0.018, stable-pitcher subset p=0.006) |
| Case-study cherry-picking risk | Pre-commit to the top-archetype-probability pitcher per archetype as primary case study |
| "Teams already do this" concern | Our contribution is the *public measurement* + the volatility-tier operationalization + the counter-intuitive stable-vet finding |
| Post-injury archetype has few cases | Supplement with IL-stint data; keep as a tentative archetype, drop if sample < 15 |

---

## 12. How to Run (Current State)

```bash
cd research/coaching-gap

# ✓ Data harmonization done (2022-2024 full pulls)
# ✓ Hybrid labeling applied
# ✓ v0 + sensitivity + hybrid probes complete
# ⟳ Active-pitcher career-history pull in background (474 pitchers)

# Next (Week 2):
# 1. Layer 1 — evolution features + archetype clustering
python scripts/layer1_signatures.py --harmonized data/harmonized/hybrid_labeled_pitches.parquet

# 2. Layer 2 agent runs (Claude + Codex, independent)
# (agent prompts in claude-analysis/ and codex-analysis/)
```

---

## Appendix A — What changed from v2 to v3

| Dimension | v2 | v3 |
|---|---|---|
| Flagship article | The Coaching Gap | How Pitchers Evolve |
| Scope of corpus | 2015-2026, all pitchers | Active-2026 pitchers × full career |
| Central analytical object | Pitch prediction model | Evolution archetype taxonomy |
| Coaching gap framing | Recency / archetype / regime | Archetype / recency / regime / count-state |
| Stable vs volatile prediction | "Volatile shows bigger gap" | "Stable shows bigger gap" (empirically confirmed) |
| Physics clustering role | Primary labeling system | Hybrid (17% phys, 83% MLBAM) |
| Volatility as a feature | Input to Layer 2 | Stratifier and trajectory variable |
| Added feature families | — | Predictability-trajectory (§5.2) |

# Research Brief: Are Catchers the Real ABS Specialists?

## Hypothesis
Catchers are outperforming batters as ABS challengers in 2026, not just because they challenge more obvious misses, but because they have a better read on which taken pitches are worth sending to review.

## Why This Is the Other Standalone Package
- The latest folder `team-challenge-iq/` is built around team strategy and challenge efficiency.
- This package isolates a different question inside that same system: **which human role is actually best at using ABS?**

## Core Questions
1. How big is the raw catcher-vs-batter gap in overturn rate?
2. Do catchers challenge easier or harder pitches than batters?
3. Does the catcher edge hold in early, middle, and late count buckets?
4. Does the catcher edge survive when we control for count bucket and edge distance?
5. Are catcher-heavy teams automatically better, or is the role edge more subtle than that?

## Data
- `research/team-challenge-iq/data/all_challenges_detail.json`

## Planned Output
- A reproducible analysis package with report, charts, and tables
- Overall role comparison with confidence intervals
- Count-bucket and edge-bucket role breakdowns
- A controlled overturn model using role, count bucket, and edge distance
- Team role-mix context so we do not overstate the strategy takeaway

## Kill Criteria
- If catchers only win because they challenge much more obvious calls, the identity story weakens into a selection-only story.
- If the catcher advantage disappears once count bucket and edge distance are included, the headline must soften to “catchers pick better spots” rather than “catchers are better challengers.”
- If team-level role share strongly predicts results by itself, the story becomes about organizational role policy instead of catcher skill.

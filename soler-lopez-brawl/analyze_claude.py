"""
Post-Brawl Umpire Zone Analysis
================================
Do umpires call the zone differently after bench-clearing incidents?

Analyzes 7 Statcast-era bench-clearing incidents (2019-2026) comparing:
- Incident game zone metrics
- Next game (same umpire) zone metrics
- Season baseline zone metrics

Outputs: charts/ folder with visualizations + printed statistical summary.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
CHARTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(CHARTS, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#0f1117',
    'axes.edgecolor': '#333',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'grid.color': '#222',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'font.size': 11,
})

COLORS = {
    'incident': '#ef4444',   # red
    'next': '#f59e0b',       # amber
    'baseline': '#6366f1',   # indigo
    'accent': '#10b981',     # emerald
    'bg': '#0f1117',
    'card': '#1a1d28',
}

# ── Load incidents ──
with open(os.path.join(DATA, "incidents.json")) as f:
    incidents = json.load(f)

# ── Helpers ──
BALL_RADIUS_FT = 1.45 / 12  # baseball radius in feet (~0.121 ft)
ZONE_HALF_WIDTH = 17 / 12 / 2 + BALL_RADIUS_FT  # half-plate + ball radius
BORDERLINE_BUFFER = 0.5    # feet from edge to count as "borderline"

def zone_bounds(row):
    """Return expanded zone edges (plate + ball radius, sz_top/bot + ball radius)."""
    return (
        -ZONE_HALF_WIDTH,           # left
        ZONE_HALF_WIDTH,            # right
        row['sz_bot'] - BALL_RADIUS_FT,  # bottom
        row['sz_top'] + BALL_RADIUS_FT,  # top
    )

def is_in_zone(row):
    """True if the pitch is inside the rule-book zone (expanded by ball radius)."""
    left, right, bot, top = zone_bounds(row)
    return (row['plate_x'] >= left and row['plate_x'] <= right and
            row['plate_z'] >= bot and row['plate_z'] <= top)

def zone_distance(row):
    """Distance from zone edge (negative = inside, positive = outside).
    Uses Euclidean distance from the expanded zone boundary."""
    left, right, bot, top = zone_bounds(row)
    dx = max(0, left - row['plate_x'], row['plate_x'] - right)
    dz = max(0, bot - row['plate_z'], row['plate_z'] - top)
    if dx == 0 and dz == 0:
        # Inside zone — compute distance to nearest edge (negative)
        edge_dists = [
            row['plate_x'] - left,
            right - row['plate_x'],
            row['plate_z'] - bot,
            top - row['plate_z'],
        ]
        return -min(edge_dists)
    return np.hypot(dx, dz)

def is_borderline(row):
    """True if pitch is within BORDERLINE_BUFFER of zone edge."""
    return abs(zone_distance(row)) <= BORDERLINE_BUFFER

def compute_metrics(df):
    """Compute zone metrics for a dataframe of called pitches."""
    called = df[df['description'].isin(['called_strike', 'ball'])].copy()
    if len(called) == 0:
        return None

    called['in_zone'] = called.apply(is_in_zone, axis=1)
    called['zone_dist'] = called.apply(zone_distance, axis=1)
    called['borderline'] = called.apply(is_borderline, axis=1)
    called['is_strike_call'] = called['description'] == 'called_strike'

    # Correct calls: strike in zone or ball outside zone
    called['correct'] = ((called['is_strike_call'] & called['in_zone']) |
                         (~called['is_strike_call'] & ~called['in_zone']))

    n = len(called)
    n_strikes = called['is_strike_call'].sum()
    borderline = called[called['borderline']]
    n_borderline = len(borderline)
    borderline_strikes = borderline['is_strike_call'].sum() if n_borderline > 0 else 0

    return {
        'n_called': n,
        'called_strike_rate': n_strikes / n if n > 0 else 0,
        'accuracy': called['correct'].mean() if n > 0 else 0,
        'n_borderline': n_borderline,
        'borderline_strike_rate': borderline_strikes / n_borderline if n_borderline > 0 else 0,
        'avg_zone_dist_strikes': called.loc[called['is_strike_call'], 'zone_dist'].mean() if n_strikes > 0 else 0,
        'called_df': called,
    }

def load_csv(path):
    """Load a Statcast CSV, drop rows without location data."""
    df = pd.read_csv(path)
    df = df.dropna(subset=['plate_x', 'plate_z', 'sz_top', 'sz_bot'])
    return df

# ── Compute all metrics ──
results = []
all_incident_called = []
all_next_called = []
all_baseline_called = []

for inc in incidents:
    label = inc['label']
    slug = label.replace(' ', '-').replace('/', '-')

    # Incident game
    inc_path = os.path.join(DATA, "incident_games", f"{slug.replace('-', '_', 1)}".replace(slug, label.replace(' ', '_')) + ".csv")
    # More robust: find the file
    inc_files = [f for f in os.listdir(os.path.join(DATA, "incident_games")) if label.replace(' ', '_').replace('-', '_') in f.replace('-', '_').replace(' ', '_')]
    if not inc_files:
        # Try matching by first word
        inc_files = [f for f in os.listdir(os.path.join(DATA, "incident_games")) if label.split(' ')[0].split('-')[0] in f]

    next_files = [f for f in os.listdir(os.path.join(DATA, "next_games")) if label.split(' ')[0].split('-')[0] in f]
    base_files = [f for f in os.listdir(os.path.join(DATA, "baselines")) if label.split(' ')[0].split('-')[0] in f]

    row = {'label': label, 'umpire': inc['umpire_name'], 'date': inc['date']}

    # Incident game metrics
    if inc_files:
        inc_df = load_csv(os.path.join(DATA, "incident_games", inc_files[0]))
        m = compute_metrics(inc_df)
        if m:
            row['inc_csr'] = m['called_strike_rate']
            row['inc_acc'] = m['accuracy']
            row['inc_bsr'] = m['borderline_strike_rate']
            row['inc_n'] = m['n_called']
            row['inc_n_borderline'] = m['n_borderline']
            m['called_df']['game_type_label'] = 'incident'
            m['called_df']['incident_label'] = label
            all_incident_called.append(m['called_df'])

    # Next game metrics
    if next_files:
        next_df = load_csv(os.path.join(DATA, "next_games", next_files[0]))
        m = compute_metrics(next_df)
        if m:
            row['next_csr'] = m['called_strike_rate']
            row['next_acc'] = m['accuracy']
            row['next_bsr'] = m['borderline_strike_rate']
            row['next_n'] = m['n_called']
            row['next_n_borderline'] = m['n_borderline']
            m['called_df']['game_type_label'] = 'next'
            m['called_df']['incident_label'] = label
            all_next_called.append(m['called_df'])

    # Baseline metrics (excluding incident + next game dates)
    if base_files:
        base_df = load_csv(os.path.join(DATA, "baselines", base_files[0]))
        # Filter to called pitches only for baseline
        exclude_dates = [inc['date']]
        if 'next_game_date' in inc:
            exclude_dates.append(inc['next_game_date'])
        base_df = base_df[~base_df['game_date'].isin(exclude_dates)]
        m = compute_metrics(base_df)
        if m:
            row['base_csr'] = m['called_strike_rate']
            row['base_acc'] = m['accuracy']
            row['base_bsr'] = m['borderline_strike_rate']
            row['base_n'] = m['n_called']
            row['base_n_borderline'] = m['n_borderline']
            m['called_df']['game_type_label'] = 'baseline'
            m['called_df']['incident_label'] = label
            all_baseline_called.append(m['called_df'])

    results.append(row)

df_results = pd.DataFrame(results)

# ── Print Summary Table ──
print("=" * 90)
print("POST-BRAWL UMPIRE ZONE ANALYSIS — SUMMARY")
print("=" * 90)
print()

for _, r in df_results.iterrows():
    print(f"📋 {r['label']} — HP Ump: {r['umpire']} ({r['date']})")
    print(f"   {'Metric':<25} {'Incident':>10} {'Next Game':>10} {'Baseline':>10} {'Inc→Next Δ':>10}")
    print(f"   {'-'*65}")

    for metric, key in [('Called Strike Rate', 'csr'), ('Accuracy', 'acc'), ('Borderline Strike %', 'bsr')]:
        inc_val = r.get(f'inc_{key}', float('nan'))
        next_val = r.get(f'next_{key}', float('nan'))
        base_val = r.get(f'base_{key}', float('nan'))
        delta = next_val - inc_val if pd.notna(next_val) and pd.notna(inc_val) else float('nan')
        delta_str = f"{delta:+.1%}" if pd.notna(delta) else "N/A"
        next_str = f"{next_val:.1%}" if pd.notna(next_val) else "N/A"
        print(f"   {metric:<25} {inc_val:>9.1%} {next_str:>10} {base_val:>9.1%} {delta_str:>10}")

    n_inc = r.get('inc_n', 0)
    n_next = r.get('next_n', float('nan'))
    n_base = r.get('base_n', 0)
    next_n_str = f"{n_next:.0f}" if pd.notna(n_next) else "N/A"
    print(f"   {'Called Pitches':<25} {n_inc:>10.0f} {next_n_str:>10} {n_base:>10.0f}")
    print()

# Better formatted table
print("\n" + "=" * 90)
print("AGGREGATE COMPARISON")
print("=" * 90)

# Compute aggregates
has_next = df_results.dropna(subset=['next_csr'])

metrics_agg = {}
for key, label in [('csr', 'Called Strike Rate'), ('acc', 'Accuracy'), ('bsr', 'Borderline Strike %')]:
    inc_vals = df_results[f'inc_{key}'].dropna()
    next_vals = has_next[f'next_{key}'].dropna()
    # Use matched baseline (only cases that have next-game data) for next-vs-base tests
    matched_base_vals = has_next[f'base_{key}'].dropna()
    all_base_vals = df_results[f'base_{key}'].dropna()

    metrics_agg[key] = {
        'inc_mean': inc_vals.mean(),
        'inc_std': inc_vals.std(),
        'next_mean': next_vals.mean(),
        'next_std': next_vals.std(),
        'base_mean': all_base_vals.mean(),
        'base_std': all_base_vals.std(),
    }

    print(f"\n{label}:")
    print(f"  Incident games:  {inc_vals.mean():.1%} ± {inc_vals.std():.1%}  (n={len(inc_vals)})")
    print(f"  Next games:      {next_vals.mean():.1%} ± {next_vals.std():.1%}  (n={len(next_vals)})")
    print(f"  Season baseline: {all_base_vals.mean():.1%} ± {all_base_vals.std():.1%}  (n={len(all_base_vals)})")

    # PRIMARY TEST: one-sample t-test on (next - baseline) deltas
    # Tests whether the mean shift from baseline in next games differs from 0
    next_base_deltas = (has_next[f'next_{key}'] - has_next[f'base_{key}']).dropna()
    if len(next_base_deltas) >= 2:
        t_stat, p_val = stats.ttest_1samp(next_base_deltas, 0.0)
        ci_low, ci_high = stats.t.interval(0.95, len(next_base_deltas) - 1,
                                            loc=next_base_deltas.mean(),
                                            scale=stats.sem(next_base_deltas))
        print(f"  One-sample t-test (next vs baseline deltas): t={t_stat:.3f}, p={p_val:.4f}")
        print(f"    Mean delta: {next_base_deltas.mean():+.4f} ({next_base_deltas.mean()*100:+.1f}pp)")
        print(f"    95% CI: [{ci_low*100:+.1f}pp, {ci_high*100:+.1f}pp]")
        n_below = (next_base_deltas < 0).sum()
        print(f"    Direction: {n_below}/{len(next_base_deltas)} below baseline")

    # SECONDARY TEST: one-sample t-test on (incident - baseline) deltas
    inc_base_deltas = (df_results[f'inc_{key}'] - df_results[f'base_{key}']).dropna()
    if len(inc_base_deltas) >= 2:
        t_stat2, p_val2 = stats.ttest_1samp(inc_base_deltas, 0.0)
        print(f"  One-sample t-test (incident vs baseline deltas): t={t_stat2:.3f}, p={p_val2:.4f}")
        print(f"    Mean delta: {inc_base_deltas.mean()*100:+.1f}pp")

# ── Compute delta analysis ──
print("\n" + "=" * 90)
print("DELTA ANALYSIS: Incident Game vs Season Baseline")
print("=" * 90)

for _, r in df_results.iterrows():
    inc_csr = r.get('inc_csr', float('nan'))
    base_csr = r.get('base_csr', float('nan'))
    next_csr = r.get('next_csr', float('nan'))
    if pd.notna(inc_csr) and pd.notna(base_csr):
        delta_inc = inc_csr - base_csr
        print(f"  {r['label']:<30} Inc vs Base: {delta_inc:+.1%}", end="")
        if pd.notna(next_csr):
            delta_next = next_csr - base_csr
            print(f"   Next vs Base: {delta_next:+.1%}", end="")
        print()


# ══════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════

# ── Charts 1-3: Only include cases with data; skip missing next-game ──
# Filter to cases that have next-game data for grouped bar charts
df_complete = df_results.dropna(subset=['next_csr'])
labels_short = [r['label'].replace(' 20', '\n\'') for _, r in df_complete.iterrows()]
x = np.arange(len(labels_short))
width = 0.25

def make_grouped_bar(ax, inc_vals, next_vals, base_vals, ylabel, title, ylim=None):
    bars1 = ax.bar(x - width, inc_vals * 100, width, label='Incident Game', color=COLORS['incident'], alpha=0.9, zorder=3)
    bars2 = ax.bar(x, next_vals * 100, width, label='Next Game', color=COLORS['next'], alpha=0.9, zorder=3)
    bars3 = ax.bar(x + width, base_vals * 100, width, label='Season Baseline', color=COLORS['baseline'], alpha=0.9, zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=9)
    ax.legend(loc='upper right', framealpha=0.8, facecolor=COLORS['card'])
    ax.grid(axis='y', alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                        f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7, color='#ccc')

# Chart 1: Called Strike Rate
fig, ax = plt.subplots(figsize=(12, 6))
make_grouped_bar(ax, df_complete['inc_csr'].values, df_complete['next_csr'].values,
                 df_complete['base_csr'].values, 'Called Strike Rate (%)',
                 'Called Strike Rate: Incident vs Next vs Baseline (complete cases only)', ylim=(20, 50))
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "01_called_strike_rate_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Saved: charts/01_called_strike_rate_comparison.png")

# Chart 2: Accuracy
fig, ax = plt.subplots(figsize=(12, 6))
make_grouped_bar(ax, df_complete['inc_acc'].values, df_complete['next_acc'].values,
                 df_complete['base_acc'].values, 'Accuracy (%)',
                 'Umpire Accuracy: Incident vs Next vs Baseline (complete cases only)', ylim=(75, 100))
ax.legend(loc='lower right', framealpha=0.8, facecolor=COLORS['card'])
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "02_accuracy_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: charts/02_accuracy_comparison.png")

# Chart 3: Borderline Strike Rate
fig, ax = plt.subplots(figsize=(12, 6))
make_grouped_bar(ax, df_complete['inc_bsr'].values, df_complete['next_bsr'].values,
                 df_complete['base_bsr'].values, 'Borderline Strike Rate (%)',
                 'Borderline Pitch Strike Rate: Do Umpires Expand the Zone After a Brawl?')
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "03_borderline_strike_rate.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: charts/03_borderline_strike_rate.png")

# ── Chart 4: Delta from baseline (lollipop chart) ──
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

for idx, (key, title) in enumerate([
    ('csr', 'Called Strike Rate'),
    ('acc', 'Accuracy'),
    ('bsr', 'Borderline Strike %')
]):
    ax = axes[idx]
    labels = []
    inc_deltas = []
    next_deltas = []

    for _, r in df_results.iterrows():
        base = r.get(f'base_{key}', float('nan'))
        inc_v = r.get(f'inc_{key}', float('nan'))
        next_v = r.get(f'next_{key}', float('nan'))
        # Only include cases where both incident and next-game data exist
        if pd.notna(base) and pd.notna(inc_v) and pd.notna(next_v):
            labels.append(r['label'].split(' ')[0])
            inc_deltas.append((inc_v - base) * 100)
            next_deltas.append((next_v - base) * 100)

    y = np.arange(len(labels))

    ax.barh(y + 0.15, inc_deltas, 0.3, label='Incident vs Base', color=COLORS['incident'], alpha=0.85, zorder=3)
    ax.barh(y - 0.15, next_deltas, 0.3, label='Next vs Base', color=COLORS['next'], alpha=0.85, zorder=3)
    ax.axvline(0, color='#555', linewidth=1, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Δ from Baseline (pp)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.8, facecolor=COLORS['card'])
    ax.grid(axis='x', alpha=0.3)

fig.suptitle('Deviation from Season Baseline After Bench-Clearing Incidents', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "04_delta_from_baseline.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: charts/04_delta_from_baseline.png")

# ── Chart 5: Strike zone heatmaps — Incident vs Next vs Baseline (pooled) ──
if all_incident_called and all_baseline_called:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    datasets = [
        (pd.concat(all_incident_called), 'Incident Games', COLORS['incident']),
        (pd.concat(all_next_called) if all_next_called else None, 'Next Games', COLORS['next']),
        (pd.concat(all_baseline_called), 'Season Baseline', COLORS['baseline']),
    ]

    for idx, (df_pool, title, color) in enumerate(datasets):
        ax = axes[idx]

        if df_pool is None or len(df_pool) == 0:
            ax.set_title(title + '\n(No data)', fontsize=12)
            continue

        strikes = df_pool[df_pool['is_strike_call']]
        balls = df_pool[~df_pool['is_strike_call']]

        ax.scatter(balls['plate_x'], balls['plate_z'], s=8, alpha=0.15, c='#4488ff', label='Ball', zorder=2)
        ax.scatter(strikes['plate_x'], strikes['plate_z'], s=8, alpha=0.3, c='#ff4444', label='Strike', zorder=3)

        # Draw zone (expanded by ball radius)
        zone_rect = patches.Rectangle((-ZONE_HALF_WIDTH, 1.5 - BALL_RADIUS_FT),
                                       ZONE_HALF_WIDTH * 2, 2.0 + 2 * BALL_RADIUS_FT,
                                       linewidth=2, edgecolor='white', facecolor='none', zorder=4)
        ax.add_patch(zone_rect)

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
        ax.set_aspect('equal')
        ax.set_title(f"{title}\n{len(strikes)} strikes / {len(balls)} balls", fontsize=11, fontweight='bold')
        ax.set_xlabel('Plate X (ft)')
        if idx == 0:
            ax.set_ylabel('Plate Z (ft)')
        ax.legend(fontsize=8, loc='upper right', framealpha=0.7, facecolor=COLORS['card'])

    fig.suptitle('Called Pitch Locations: Pooled Across All Incidents', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, "05_zone_heatmaps.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: charts/05_zone_heatmaps.png")

# ── Chart 6: Per-incident CSR movement (slope chart) ──
fig, ax = plt.subplots(figsize=(10, 7))

for _, r in df_results.iterrows():
    inc_v = r.get('inc_csr', float('nan'))
    next_v = r.get('next_csr', float('nan'))
    base_v = r.get('base_csr', float('nan'))

    if pd.notna(inc_v) and pd.notna(base_v):
        points = [('Baseline', base_v * 100)]
        points.append(('Incident\nGame', inc_v * 100))
        if pd.notna(next_v):
            points.append(('Next\nGame', next_v * 100))

        xs = list(range(len(points)))
        ys = [p[1] for p in points]

        ax.plot(xs, ys, '-o', linewidth=2, markersize=8, alpha=0.8, label=r['label'])

        # Label endpoint
        ax.annotate(r['label'].split(' ')[0], (xs[-1] + 0.05, ys[-1]),
                    fontsize=8, color='#ccc', va='center')

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Season\nBaseline', 'Incident\nGame', 'Next\nGame'])
ax.set_ylabel('Called Strike Rate (%)')
ax.set_title('Called Strike Rate Trajectory: Baseline → Incident → Next Game', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "06_csr_trajectory.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: charts/06_csr_trajectory.png")

# ── Chart 7: Out-of-zone strike rate (correct denominator: strikes / all OOZ pitches) ──
fig, ax = plt.subplots(figsize=(12, 6))

ooz_data = []
for game_type, called_list in [('incident', all_incident_called), ('next', all_next_called), ('baseline', all_baseline_called)]:
    for df_c in called_list:
        label = df_c['incident_label'].iloc[0]
        outside_pitches = df_c[~df_c['in_zone']]
        if len(outside_pitches) > 0:
            ooz_strike_rate = outside_pitches['is_strike_call'].mean() * 100
            ooz_data.append({'label': label, 'game_type': game_type, 'ooz_strike_rate': ooz_strike_rate})

ooz_df = pd.DataFrame(ooz_data)

# Only include labels that have all three game types (complete cases)
complete_labels = []
for label in ooz_df['label'].unique():
    types_present = set(ooz_df[ooz_df['label'] == label]['game_type'])
    if {'incident', 'next', 'baseline'}.issubset(types_present):
        complete_labels.append(label)

ooz_complete = ooz_df[ooz_df['label'].isin(complete_labels)]

if len(complete_labels) > 0:
    x2 = np.arange(len(complete_labels))
    short_labels = [l.split(' ')[0] for l in complete_labels]

    inc_rates = [ooz_complete[(ooz_complete['label'] == l) & (ooz_complete['game_type'] == 'incident')]['ooz_strike_rate'].values[0] for l in complete_labels]
    next_rates = [ooz_complete[(ooz_complete['label'] == l) & (ooz_complete['game_type'] == 'next')]['ooz_strike_rate'].values[0] for l in complete_labels]
    base_rates = [ooz_complete[(ooz_complete['label'] == l) & (ooz_complete['game_type'] == 'baseline')]['ooz_strike_rate'].values[0] for l in complete_labels]

    ax.bar(x2 - width, inc_rates, width, label='Incident Game', color=COLORS['incident'], alpha=0.9, zorder=3)
    ax.bar(x2, next_rates, width, label='Next Game', color=COLORS['next'], alpha=0.9, zorder=3)
    ax.bar(x2 + width, base_rates, width, label='Season Baseline', color=COLORS['baseline'], alpha=0.9, zorder=3)

    ax.set_ylabel('Out-of-Zone Strike Rate (%)')
    ax.set_title('Out-of-Zone Strike Rate: % of Pitches Outside Zone Called as Strikes', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x2)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.legend(framealpha=0.8, facecolor=COLORS['card'])
    ax.grid(axis='y', alpha=0.3)

    for bars_vals, offset in [(inc_rates, -width), (next_rates, 0), (base_rates, width)]:
        for i, v in enumerate(bars_vals):
            ax.text(i + offset, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=7, color='#ccc')

fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "07_outside_zone_strikes.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: charts/07_outside_zone_strikes.png")

# ── Chart 8: Confidence interval forest plot ──
fig, ax = plt.subplots(figsize=(10, 6))

# For each metric, show the mean delta (incident - baseline) with bootstrapped CI
np.random.seed(42)

metric_labels = []
means = []
ci_lows = []
ci_highs = []
colors_plot = []

for key, label in [('csr', 'Called Strike Rate'), ('acc', 'Accuracy'), ('bsr', 'Borderline Strike %')]:
    # Incident vs baseline deltas
    deltas_inc = []
    deltas_next = []
    for _, r in df_results.iterrows():
        inc_v = r.get(f'inc_{key}', float('nan'))
        base_v = r.get(f'base_{key}', float('nan'))
        next_v = r.get(f'next_{key}', float('nan'))
        if pd.notna(inc_v) and pd.notna(base_v):
            deltas_inc.append((inc_v - base_v) * 100)
        if pd.notna(next_v) and pd.notna(base_v):
            deltas_next.append((next_v - base_v) * 100)

    for delta_list, suffix, color in [(deltas_inc, '(Inc)', COLORS['incident']), (deltas_next, '(Next)', COLORS['next'])]:
        if len(delta_list) >= 2:
            arr = np.array(delta_list)
            mean_d = arr.mean()
            # Bootstrap CI
            boot_means = [np.random.choice(arr, size=len(arr), replace=True).mean() for _ in range(5000)]
            ci_low = np.percentile(boot_means, 2.5)
            ci_high = np.percentile(boot_means, 97.5)

            metric_labels.append(f"{label}\n{suffix}")
            means.append(mean_d)
            ci_lows.append(ci_low)
            ci_highs.append(ci_high)
            colors_plot.append(color)

y_pos = np.arange(len(metric_labels))
ax.axvline(0, color='white', linewidth=1, alpha=0.5, zorder=1)

for i in range(len(metric_labels)):
    ax.plot([ci_lows[i], ci_highs[i]], [y_pos[i], y_pos[i]], linewidth=3, color=colors_plot[i], alpha=0.6, zorder=2)
    ax.scatter(means[i], y_pos[i], s=100, color=colors_plot[i], zorder=3, edgecolors='white', linewidth=1)
    ax.text(ci_highs[i] + 0.3, y_pos[i], f"{means[i]:+.1f}pp", va='center', fontsize=9, color='#ccc')

ax.set_yticks(y_pos)
ax.set_yticklabels(metric_labels, fontsize=10)
ax.set_xlabel('Δ from Season Baseline (percentage points)')
ax.set_title('Effect Size: How Much Do Zone Metrics Shift After a Brawl?\n(95% Bootstrap CI)', fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(CHARTS, "08_forest_plot.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: charts/08_forest_plot.png")


# ── Save results to CSV ──
df_results.to_csv(os.path.join(CHARTS, "..", "results_summary.csv"), index=False)
print(f"\n✅ Saved: results_summary.csv")

# ── Final verdict ──
print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)

avg_inc_csr = df_results['inc_csr'].mean()
avg_next_csr = df_results['next_csr'].dropna().mean()
avg_base_csr = df_results['base_csr'].mean()

delta_inc_base = (avg_inc_csr - avg_base_csr) * 100
delta_next_base = (avg_next_csr - avg_base_csr) * 100

print(f"\nAverage Called Strike Rate:")
print(f"  Incident games: {avg_inc_csr:.1%} ({delta_inc_base:+.1f}pp from baseline)")
print(f"  Next games:     {avg_next_csr:.1%} ({delta_next_base:+.1f}pp from baseline)")
print(f"  Season baseline: {avg_base_csr:.1%}")

if abs(delta_inc_base) < 1.0 and abs(delta_next_base) < 1.0:
    print("\n🔔 KILL CRITERIA MET: Deltas within 1pp → 'Umpires Don't Flinch' angle")
elif delta_next_base > 1.0:
    print("\n📰 FINDING: Umpires appear to EXPAND the zone after brawls (more called strikes)")
elif delta_next_base < -1.0:
    print("\n📰 FINDING: Umpires appear to TIGHTEN the zone after brawls (fewer called strikes)")
else:
    print("\n📊 MIXED RESULTS: Some movement but no clear pattern — needs more incidents")

print("\n✅ Analysis complete. See charts/ folder for visualizations.")

"""
Generate publication-quality figures for the thesis.
All figures use a consistent dark academic style with high DPI output.
Run: python3 results/generate_figures.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D

# ── Style config ──────────────────────────────────────────────────────────────
DARK_BG  = '#0e1117'
CARD_BG  = '#161b22'
GRID_CLR = '#21262d'
TEXT_CLR = '#c9d1d9'
DIM_CLR  = '#8b949e'

PALETTE = {
    'supplier':    '#22c55e',
    'factory':     '#3b82f6',
    'dc':          '#f59e0b',
    'customer':    '#ef4444',
    'road':        '#f97316',
    'rail':        '#3b82f6',
    'cost':        '#f85149',
    'emissions':   '#f59e0b',
    'resilience':  '#3fb950',
    'primary':     '#58a6ff',
    'purple':      '#bc8cff',
}

REP_COLORS = ['#3fb950', '#58a6ff', '#bc8cff', '#d29922']

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor': CARD_BG,
    'axes.edgecolor': GRID_CLR,
    'axes.labelcolor': TEXT_CLR,
    'text.color': TEXT_CLR,
    'xtick.color': DIM_CLR,
    'ytick.color': DIM_CLR,
    'grid.color': GRID_CLR,
    'grid.alpha': 0.4,
    'legend.facecolor': CARD_BG,
    'legend.edgecolor': GRID_CLR,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Inter', 'Helvetica Neue', 'Arial'],
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': DARK_BG,
})

FIG_DIR = './results/figures'
os.makedirs(FIG_DIR, exist_ok=True)


def load_results():
    with open('./results/experiment_summary.json') as f:
        summary = json.load(f)
    pareto = pd.read_csv('./results/pareto_front.csv')
    return summary, pareto


def fig_pareto_3d(pareto, reps):
    """3-D scatter plot of Pareto front with representative highlights."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(CARD_BG)
    fig.patch.set_facecolor(DARK_BG)

    # Main cloud
    sc = ax.scatter(
        pareto['cost'] / 1e6, pareto['emissions'] / 1e3, pareto['resilience'] * 100,
        c=pareto['resilience'], cmap='viridis', s=20, alpha=0.5, edgecolors='none'
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08, label='Resilience (%)')
    cb.ax.yaxis.label.set_color(TEXT_CLR)
    cb.ax.tick_params(colors=DIM_CLR)

    # Representative solutions
    for i, r in enumerate(reps):
        ax.scatter(
            [r['cost'] / 1e6], [r['emissions'] / 1e3], [r['resilience'] * 100],
            c=REP_COLORS[i % 4], s=120, marker='D', edgecolors='white', linewidths=1.5, zorder=5
        )
        ax.text(r['cost']/1e6, r['emissions']/1e3, r['resilience']*100 + 1.2,
                r['label'].split('(')[0].strip(), fontsize=8, color=REP_COLORS[i%4],
                ha='center', weight='bold')

    ax.set_xlabel('Cost ($M)', labelpad=10)
    ax.set_ylabel('Emissions (ton CO₂)', labelpad=10)
    ax.set_zlabel('Resilience (%)', labelpad=10)
    ax.set_title('Pareto Front — Cost × Emissions × Resilience', fontsize=14, weight='bold', pad=15)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GRID_CLR)
    ax.yaxis.pane.set_edgecolor(GRID_CLR)
    ax.zaxis.pane.set_edgecolor(GRID_CLR)
    ax.tick_params(colors=DIM_CLR)

    plt.savefig(f'{FIG_DIR}/pareto_front_3d.png')
    plt.close()
    print('  ✓ pareto_front_3d.png')


def fig_pareto_2d(pareto, reps):
    """2-D Pareto projections (3 subplots)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pairs = [
        ('cost', 'emissions', 'resilience', 'Cost ($M)', 'Emissions (ton CO₂)'),
        ('cost', 'resilience', 'emissions', 'Cost ($M)', 'Resilience (%)'),
        ('emissions', 'resilience', 'cost', 'Emissions (ton CO₂)', 'Resilience (%)'),
    ]

    for ax, (xc, yc, cc, xl, yl) in zip(axes, pairs):
        xd = pareto[xc] / 1e6 if 'cost' in xc else (pareto[xc] / 1e3 if 'emissions' in xc else pareto[xc] * 100)
        yd = pareto[yc] / 1e6 if 'cost' in yc else (pareto[yc] / 1e3 if 'emissions' in yc else pareto[yc] * 100)
        cd = pareto[cc]

        ax.scatter(xd, yd, c=cd, cmap='viridis', s=18, alpha=0.6, edgecolors='none')
        for i, r in enumerate(reps):
            rx = r[xc] / 1e6 if 'cost' in xc else (r[xc] / 1e3 if 'emissions' in xc else r[xc] * 100)
            ry = r[yc] / 1e6 if 'cost' in yc else (r[yc] / 1e3 if 'emissions' in yc else r[yc] * 100)
            ax.scatter(rx, ry, c=REP_COLORS[i%4], s=90, marker='D', edgecolors='white', linewidths=1.2, zorder=5)
            ax.annotate(r['label'].split('(')[0].strip(), (rx, ry), fontsize=7,
                        color=REP_COLORS[i%4], weight='bold',
                        textcoords='offset points', xytext=(5, 8))
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Pareto Front — 2-D Projections', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/pareto_front_2d.png')
    plt.close()
    print('  ✓ pareto_front_2d.png')


def fig_representative_solutions(reps):
    """Grouped bar chart + radar for representative solutions."""
    labels = [r['label'] for r in reps]
    n = len(labels)

    # Normalize for comparison
    costs = [r['cost'] for r in reps]
    emiss = [r['emissions'] for r in reps]
    resil = [r['resilience'] * 100 for r in reps]
    def norm(v):
        mn, mx = min(v), max(v)
        return [(x - mn) / (mx - mn + 1e-9) * 100 for x in v]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    subplot_kw={}, gridspec_kw={'width_ratios': [1.2, 1]})

    # ── Bar chart ──
    x = np.arange(n)
    w = 0.25
    ax1.bar(x - w, norm(costs),  w, color=PALETTE['cost'],      alpha=0.85, label='Cost (norm)')
    ax1.bar(x,     norm(emiss),  w, color=PALETTE['emissions'],  alpha=0.85, label='Emissions (norm)')
    ax1.bar(x + w, resil,        w, color=PALETTE['resilience'], alpha=0.85, label='Resilience (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace('Balanced ', 'Bal.\n') for l in labels], fontsize=9)
    ax1.set_ylabel('Normalized Score / %')
    ax1.set_title('Objective Comparison', fontsize=13, weight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)

    # ── Radar ──
    ax2 = fig.add_subplot(122, polar=True)
    ax2.set_facecolor(CARD_BG)
    categories = ['Cost\nEfficiency', 'Emission\nEfficiency', 'Resilience']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ce = [1 - (c - min(costs)) / (max(costs) - min(costs) + 1e-9) for c in costs]
    ee = [1 - (e - min(emiss)) / (max(emiss) - min(emiss) + 1e-9) for e in emiss]
    rn = [(r - min(resil)) / (max(resil) - min(resil) + 1e-9) for r in resil]

    for i, r in enumerate(reps):
        vals = [ce[i], ee[i], rn[i]]
        vals += vals[:1]
        ax2.plot(angles, vals, 'o-', color=REP_COLORS[i%4], linewidth=2, markersize=5, label=r['label'])
        ax2.fill(angles, vals, alpha=0.06, color=REP_COLORS[i%4])

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_title('Radar Comparison', fontsize=13, weight='bold', pad=20)
    ax2.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.35, 1.15))
    ax2.grid(color=GRID_CLR, alpha=0.4)
    ax2.tick_params(colors=DIM_CLR)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/representative_solutions.png')
    plt.close()
    print('  ✓ representative_solutions.png')


def fig_simulation_results(sim):
    """Gauge-style simulation results."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    metrics = [
        ('Average Fill Rate',    sim['average_fill_rate'] * 100,  '%', PALETTE['resilience']),
        ('Worst-Case Fill Rate', sim['min_fill_rate'] * 100,      '%', PALETTE['emissions']),
        ('Avg Disruptions/10yr', sim['average_n_disruptions'],    '',  PALETTE['cost']),
    ]

    for ax, (title, value, suffix, color) in zip(axes, metrics):
        # Draw a circular gauge
        theta = np.linspace(0, np.pi, 100)
        max_val = 100 if suffix == '%' else 10
        fill_angle = np.pi * min(value / max_val, 1.0)

        # Background arc
        ax.plot(np.cos(theta), np.sin(theta), color=GRID_CLR, linewidth=18, solid_capstyle='round')
        # Filled arc
        theta_fill = np.linspace(0, fill_angle, 100)
        ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, linewidth=18, solid_capstyle='round')

        ax.text(0, 0.15, f'{value:.1f}{suffix}', ha='center', va='center',
                fontsize=24, weight='bold', color=color)
        ax.text(0, -0.2, title, ha='center', va='center', fontsize=10, color=DIM_CLR)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.5, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')

    fig.suptitle('Resilience Simulation Results', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/simulation_results.png')
    plt.close()
    print('  ✓ simulation_results.png')


def fig_forecast_accuracy(forecast):
    """Forecast accuracy metrics bar chart."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    metrics = ['MAE', 'RMSE', 'MAPE']
    values  = [forecast['MAE'], forecast['RMSE'], forecast['MAPE']]
    colors  = [PALETTE['primary'], PALETTE['purple'], PALETTE['resilience']]

    bars = ax.bar(metrics, values, color=colors, width=0.5, edgecolor='none', alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.2f}', ha='center', fontsize=12, weight='bold', color=TEXT_CLR)

    ax.set_ylabel('Value')
    ax.set_title('Random Forest Forecast Accuracy', fontsize=14, weight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(values) * 1.25)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/forecast_accuracy.png')
    plt.close()
    print('  ✓ forecast_accuracy.png')


def fig_network_topology(suppliers, factories, dcs, customers):
    """Network topology map with node types."""
    fig, ax = plt.subplots(figsize=(10, 8))

    node_configs = [
        (suppliers,  'supplier_id', 'Suppliers',  PALETTE['supplier'], 's',  120),
        (factories,  'factory_id',  'Factories',  PALETTE['factory'],  '^',  120),
        (dcs,        'dc_id',       'DCs',        PALETTE['dc'],       'D',  100),
        (customers,  'customer_id', 'Customers',  PALETTE['customer'], 'o',   70),
    ]

    for df, id_col, label, color, marker, sz in node_configs:
        ax.scatter(df['longitude'], df['latitude'], c=color, s=sz, marker=marker,
                   edgecolors='white', linewidths=1, label=label, zorder=5)
        for _, row in df.iterrows():
            ax.annotate(row[id_col], (row['longitude'], row['latitude']),
                        textcoords='offset points', xytext=(6, 6),
                        fontsize=8, color=color, weight='bold')

    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Supply Chain Network Topology — India', fontsize=14, weight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/network_topology.png')
    plt.close()
    print('  ✓ network_topology.png')


def fig_demand_timeseries():
    """Demand time-series for all customers."""
    demand_path = './data/demand.csv'
    if not os.path.exists(demand_path):
        print('  ⚠ demand.csv not found, skipping demand plot')
        return
    demand = pd.read_csv(demand_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#58a6ff', '#3fb950', '#f59e0b', '#f85149', '#bc8cff']
    for i, cid in enumerate(sorted(demand['customer_id'].unique())):
        cdata = demand[demand['customer_id'] == cid].sort_values('period')
        ax.plot(cdata['period'], cdata['demand'], color=colors[i % len(colors)],
                linewidth=1.5, alpha=0.85, label=cid)

    ax.axvline(x=60, color='#f85149', linestyle='--', alpha=0.6, label='Forecast Start')
    ax.set_xlabel('Period')
    ax.set_ylabel('Demand (units)')
    ax.set_title('Customer Demand Time Series', fontsize=14, weight='bold')
    ax.legend(fontsize=9, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/demand_forecast.png')
    plt.close()
    print('  ✓ demand_forecast.png')


def fig_congestion_heatmaps():
    """Congestion heatmaps for each echelon."""
    echelons = [
        ('dist_supplier_factory', 'supplier_id', 'factory_id', 'S>F Congestion'),
        ('dist_factory_dc',       'factory_id',  'dc_id',      'F>DC Congestion'),
        ('dist_dc_customer',      'dc_id',       'customer_id','DC>C Congestion'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cmap = LinearSegmentedColormap.from_list('dark_heat', ['#161b22', '#f59e0b', '#f85149'])

    for ax, (fname, src, tgt, title) in zip(axes, echelons):
        path = f'./data/{fname}.csv'
        if not os.path.exists(path):
            ax.set_visible(False)
            continue
        df = pd.read_csv(path)
        # Aggregate across modes: take the max congestion
        agg = df.groupby([src, tgt])['congestion_factor'].max().reset_index()
        pivot = agg.pivot(index=src, columns=tgt, values='congestion_factor').fillna(1.0)

        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=1.0, vmax=1.8)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title(title, fontsize=11, weight='bold')

        # Annotate values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f'{pivot.values[i, j]:.2f}', ha='center', va='center',
                        fontsize=9, color='white', weight='bold')

    fig.suptitle('Traffic Congestion Factors', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/congestion_analysis.png')
    plt.close()
    print('  ✓ congestion_analysis.png')


def fig_distance_heatmaps():
    """Distance heatmaps for each echelon."""
    echelons = [
        ('dist_supplier_factory', 'supplier_id', 'factory_id', 'S>F Distance (km)'),
        ('dist_factory_dc',       'factory_id',  'dc_id',      'F>DC Distance (km)'),
        ('dist_dc_customer',      'dc_id',       'customer_id','DC>C Distance (km)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    cmap = LinearSegmentedColormap.from_list('dark_dist', ['#161b22', '#3b82f6', '#bc8cff'])

    for ax, (fname, src, tgt, title) in zip(axes, echelons):
        path = f'./data/{fname}.csv'
        if not os.path.exists(path):
            ax.set_visible(False)
            continue
        df = pd.read_csv(path)
        agg = df.groupby([src, tgt])['distance_km'].mean().reset_index()
        pivot = agg.pivot(index=src, columns=tgt, values='distance_km').fillna(0)

        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title(title, fontsize=11, weight='bold')

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f'{pivot.values[i, j]:.0f}', ha='center', va='center',
                        fontsize=8, color='white', weight='bold')

    fig.suptitle('Distance Matrices', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/distance_heatmaps.png')
    plt.close()
    print('  ✓ distance_heatmaps.png')


def fig_data_summary(suppliers, factories, dcs, customers):
    """Data summary panel with capacity distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Supplier capacities
    ax = axes[0, 0]
    bars = ax.barh(suppliers['supplier_id'], suppliers['capacity'], color=PALETTE['supplier'], alpha=0.85)
    ax.set_xlabel('Capacity (units)')
    ax.set_title('Supplier Capacities', fontsize=12, weight='bold')
    for bar, val in zip(bars, suppliers['capacity']):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.0f}', va='center', fontsize=9, color=TEXT_CLR)

    # Factory capacities
    ax = axes[0, 1]
    bars = ax.barh(factories['factory_id'], factories['capacity'], color=PALETTE['factory'], alpha=0.85)
    ax.set_xlabel('Capacity (units)')
    ax.set_title('Factory Capacities', fontsize=12, weight='bold')
    for bar, val in zip(bars, factories['capacity']):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.0f}', va='center', fontsize=9, color=TEXT_CLR)

    # DC capacities
    ax = axes[1, 0]
    bars = ax.barh(dcs['dc_id'], dcs['capacity'], color=PALETTE['dc'], alpha=0.85)
    ax.set_xlabel('Capacity (units)')
    ax.set_title('DC Capacities', fontsize=12, weight='bold')
    for bar, val in zip(bars, dcs['capacity']):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val:.0f}', va='center', fontsize=9, color=TEXT_CLR)

    # Cost comparison
    ax = axes[1, 1]
    x = np.arange(3)
    w = 0.3
    s_costs = suppliers['cost_per_unit'].values[:3]
    f_costs = factories['production_cost'].values[:2]
    d_costs = dcs['holding_cost'].values[:3]
    ax.bar([0, 1, 2], [s_costs.mean(), f_costs.mean(), d_costs.mean()],
           color=[PALETTE['supplier'], PALETTE['factory'], PALETTE['dc']], alpha=0.85, width=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Supplier\n($/unit)', 'Factory\n($/unit)', 'DC\nHolding ($/unit)'], fontsize=9)
    ax.set_ylabel('Avg Cost')
    ax.set_title('Average Unit Costs', fontsize=12, weight='bold')

    for a in axes.flat:
        a.grid(axis='x', alpha=0.3)

    fig.suptitle('Network Data Summary', fontsize=14, weight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/data_summary.png')
    plt.close()
    print('  ✓ data_summary.png')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('=' * 50)
    print('Generating publication-quality figures...')
    print('=' * 50)

    summary, pareto = load_results()
    reps = summary['representative_solutions']
    sim  = summary['simulation_results']
    forecast = summary['forecast_accuracy']

    suppliers  = pd.read_csv('./data/suppliers.csv')
    factories  = pd.read_csv('./data/factories.csv')
    dcs        = pd.read_csv('./data/dcs.csv')
    customers  = pd.read_csv('./data/customers.csv')

    fig_pareto_3d(pareto, reps)
    fig_pareto_2d(pareto, reps)
    fig_representative_solutions(reps)
    fig_simulation_results(sim)
    fig_forecast_accuracy(forecast)
    fig_network_topology(suppliers, factories, dcs, customers)
    fig_demand_timeseries()
    fig_congestion_heatmaps()
    fig_distance_heatmaps()
    fig_data_summary(suppliers, factories, dcs, customers)

    print('\n' + '=' * 50)
    print(f'All figures saved to {FIG_DIR}/')
    print('=' * 50)


if __name__ == '__main__':
    main()

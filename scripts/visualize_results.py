#!/usr/bin/env python3
"""
Visualization script for Green Resilient Supply Chain Network results.
Creates comprehensive plots for Pareto front, network flows, forecasts, and simulation results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

RESULTS_DIR = './results'
DATA_DIR = './data'
OUTPUT_DIR = './results/figures'


def create_output_dir():
    """Create output directory for figures."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_pareto_front_3d():
    """Plot 3D Pareto front (Cost vs Emissions vs Resilience)."""
    pareto_file = os.path.join(RESULTS_DIR, 'pareto_front.csv')
    if not os.path.exists(pareto_file):
        print(f"Warning: {pareto_file} not found. Skipping Pareto front plot.")
        return
    
    df = pd.read_csv(pareto_file)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize for better visualization
    cost_norm = (df['cost'] - df['cost'].min()) / (df['cost'].max() - df['cost'].min() + 1e-10)
    emissions_norm = (df['emissions'] - df['emissions'].min()) / (df['emissions'].max() - df['emissions'].min() + 1e-10)
    resilience_norm = df['resilience']
    
    scatter = ax.scatter(
        cost_norm, emissions_norm, resilience_norm,
        c=resilience_norm, cmap='viridis',
        s=50, alpha=0.6, edgecolors='black', linewidth=0.5
    )
    
    ax.set_xlabel('Normalized Cost', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Emissions', fontsize=12, fontweight='bold')
    ax.set_zlabel('Resilience Score', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Front: Cost vs Emissions vs Resilience', fontsize=14, fontweight='bold', pad=20)
    
    plt.colorbar(scatter, ax=ax, label='Resilience Score', shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pareto_front_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved 3D Pareto front: {OUTPUT_DIR}/pareto_front_3d.png")


def plot_pareto_front_2d():
    """Plot 2D projections of Pareto front."""
    pareto_file = os.path.join(RESULTS_DIR, 'pareto_front.csv')
    if not os.path.exists(pareto_file):
        return
    
    df = pd.read_csv(pareto_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Cost vs Emissions
    ax = axes[0]
    scatter = ax.scatter(df['cost'], df['emissions'], c=df['resilience'], 
                        cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Total Cost ($)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Emissions (kg CO2)', fontsize=11, fontweight='bold')
    ax.set_title('Cost vs Emissions', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Resilience')
    
    # Cost vs Resilience
    ax = axes[1]
    scatter = ax.scatter(df['cost'], df['resilience'], c=df['emissions'],
                        cmap='plasma', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Total Cost ($)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Resilience Score', fontsize=11, fontweight='bold')
    ax.set_title('Cost vs Resilience', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Emissions (kg CO2)')
    
    # Emissions vs Resilience
    ax = axes[2]
    scatter = ax.scatter(df['emissions'], df['resilience'], c=df['cost'],
                        cmap='coolwarm', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Total Emissions (kg CO2)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Resilience Score', fontsize=11, fontweight='bold')
    ax.set_title('Emissions vs Resilience', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cost ($)')
    
    plt.suptitle('Pareto Front: 2D Projections', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pareto_front_2d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved 2D Pareto projections: {OUTPUT_DIR}/pareto_front_2d.png")


def plot_representative_solutions():
    """Plot comparison of representative solutions."""
    summary_file = os.path.join(RESULTS_DIR, 'experiment_summary.json')
    if not os.path.exists(summary_file):
        return
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    reps = summary.get('representative_solutions', [])
    if not reps:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    labels = [sol['label'] for sol in reps]
    costs = [sol['cost'] for sol in reps]
    emissions = [sol['emissions'] for sol in reps]
    resilience = [sol['resilience'] for sol in reps]
    
    # Normalize for comparison
    cost_norm = [(c - min(costs)) / (max(costs) - min(costs) + 1e-10) for c in costs]
    em_norm = [(e - min(emissions)) / (max(emissions) - min(emissions) + 1e-10) for e in emissions]
    res_norm = resilience
    
    x = np.arange(len(labels))
    width = 0.25
    
    # Cost comparison
    ax = axes[0]
    ax.bar(x, cost_norm, width, label='Normalized Cost', color='steelblue', alpha=0.8)
    ax.set_xlabel('Solution Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized Value', fontsize=11, fontweight='bold')
    ax.set_title('Cost Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(' (', '\n(') for l in labels], rotation=0, ha='center')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Emissions comparison
    ax = axes[1]
    ax.bar(x, em_norm, width, label='Normalized Emissions', color='darkgreen', alpha=0.8)
    ax.set_xlabel('Solution Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Normalized Value', fontsize=11, fontweight='bold')
    ax.set_title('Emissions Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(' (', '\n(') for l in labels], rotation=0, ha='center')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Resilience comparison
    ax = axes[2]
    ax.bar(x, res_norm, width, label='Resilience Score', color='coral', alpha=0.8)
    ax.set_xlabel('Solution Type', fontsize=11, fontweight='bold')
    ax.set_ylabel('Resilience Score', fontsize=11, fontweight='bold')
    ax.set_title('Resilience Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(' (', '\n(') for l in labels], rotation=0, ha='center')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Representative Solutions Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'representative_solutions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved representative solutions comparison: {OUTPUT_DIR}/representative_solutions.png")


def plot_network_topology():
    """Plot supply chain network topology with locations."""
    suppliers_file = os.path.join(DATA_DIR, 'suppliers.csv')
    factories_file = os.path.join(DATA_DIR, 'factories.csv')
    dcs_file = os.path.join(DATA_DIR, 'dcs.csv')
    customers_file = os.path.join(DATA_DIR, 'customers.csv')
    
    if not all(os.path.exists(f) for f in [suppliers_file, factories_file, dcs_file, customers_file]):
        print("Warning: Missing data files. Skipping network topology plot.")
        return
    
    suppliers = pd.read_csv(suppliers_file)
    factories = pd.read_csv(factories_file)
    dcs = pd.read_csv(dcs_file)
    customers = pd.read_csv(customers_file)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot suppliers
    ax.scatter(suppliers['longitude'], suppliers['latitude'], 
              c='green', marker='s', s=300, label='Suppliers', 
              edgecolors='black', linewidth=2, zorder=5)
    for _, row in suppliers.iterrows():
        ax.annotate(row['supplier_id'], (row['longitude'], row['latitude']),
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Plot factories
    ax.scatter(factories['longitude'], factories['latitude'],
              c='blue', marker='^', s=400, label='Factories',
              edgecolors='black', linewidth=2, zorder=5)
    for _, row in factories.iterrows():
        ax.annotate(row['factory_id'], (row['longitude'], row['latitude']),
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Plot DCs
    ax.scatter(dcs['longitude'], dcs['latitude'],
              c='orange', marker='D', s=350, label='Distribution Centers',
              edgecolors='black', linewidth=2, zorder=5)
    for _, row in dcs.iterrows():
        ax.annotate(row['dc_id'], (row['longitude'], row['latitude']),
                   xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Plot customers
    ax.scatter(customers['longitude'], customers['latitude'],
              c='red', marker='o', s=150, label='Customers',
              edgecolors='black', linewidth=1, zorder=3, alpha=0.7)
    for _, row in customers.iterrows():
        ax.annotate(row['customer_id'], (row['longitude'], row['latitude']),
                   xytext=(3, 3), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('Supply Chain Network Topology', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'network_topology.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved network topology: {OUTPUT_DIR}/network_topology.png")


def plot_network_with_flows(solution_label='min_cost'):
    """Plot network with flow visualization for a specific solution."""
    # Load network data
    suppliers_file = os.path.join(DATA_DIR, 'suppliers.csv')
    factories_file = os.path.join(DATA_DIR, 'factories.csv')
    dcs_file = os.path.join(DATA_DIR, 'dcs.csv')
    customers_file = os.path.join(DATA_DIR, 'customers.csv')
    
    if not all(os.path.exists(f) for f in [suppliers_file, factories_file, dcs_file, customers_file]):
        return
    
    suppliers = pd.read_csv(suppliers_file)
    factories = pd.read_csv(factories_file)
    dcs = pd.read_csv(dcs_file)
    customers = pd.read_csv(customers_file)
    
    # Load flow data
    sf_flows_file = os.path.join(RESULTS_DIR, f'flows_supplier_factory_{solution_label}.csv')
    fd_flows_file = os.path.join(RESULTS_DIR, f'flows_factory_dc_{solution_label}.csv')
    dc_flows_file = os.path.join(RESULTS_DIR, f'flows_dc_customer_{solution_label}.csv')
    
    if not all(os.path.exists(f) for f in [sf_flows_file, fd_flows_file, dc_flows_file]):
        print(f"Warning: Flow files for {solution_label} not found. Skipping flow visualization.")
        return
    
    sf_flows = pd.read_csv(sf_flows_file)
    fd_flows = pd.read_csv(fd_flows_file)
    dc_flows = pd.read_csv(dc_flows_file)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot nodes
    ax.scatter(suppliers['longitude'], suppliers['latitude'], 
              c='green', marker='s', s=300, label='Suppliers', 
              edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(factories['longitude'], factories['latitude'],
              c='blue', marker='^', s=400, label='Factories',
              edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(dcs['longitude'], dcs['latitude'],
              c='orange', marker='D', s=350, label='Distribution Centers',
              edgecolors='black', linewidth=2, zorder=5)
    ax.scatter(customers['longitude'], customers['latitude'],
              c='red', marker='o', s=150, label='Customers',
              edgecolors='black', linewidth=1, zorder=3, alpha=0.7)
    
    # Plot flows - Supplier to Factory
    if len(sf_flows) > 0:
        max_flow = sf_flows['flow'].max()
        for _, flow in sf_flows.iterrows():
            try:
                supplier = suppliers[suppliers['supplier_id'] == flow['supplier']].iloc[0]
                factory = factories[factories['factory_id'] == flow['factory']].iloc[0]
                width = max(1, (flow['flow'] / max_flow) * 5)
                ax.plot([supplier['longitude'], factory['longitude']],
                       [supplier['latitude'], factory['latitude']],
                       'g-', alpha=0.4, linewidth=width, zorder=1)
            except (IndexError, KeyError):
                continue
    
    # Plot flows - Factory to DC
    if len(fd_flows) > 0:
        max_flow = fd_flows['flow'].max()
        for _, flow in fd_flows.iterrows():
            try:
                factory = factories[factories['factory_id'] == flow['factory']].iloc[0]
                dc = dcs[dcs['dc_id'] == flow['dc']].iloc[0]
                width = max(1, (flow['flow'] / max_flow) * 5)
                ax.plot([factory['longitude'], dc['longitude']],
                       [factory['latitude'], dc['latitude']],
                       'b-', alpha=0.5, linewidth=width, zorder=1)
            except (IndexError, KeyError):
                continue
    
    # Plot flows - DC to Customer
    if len(dc_flows) > 0:
        max_flow = dc_flows['flow'].max()
        for _, flow in dc_flows.iterrows():
            try:
                dc = dcs[dcs['dc_id'] == flow['dc']].iloc[0]
                customer = customers[customers['customer_id'] == flow['customer']].iloc[0]
                width = max(1, (flow['flow'] / max_flow) * 5)
                ax.plot([dc['longitude'], customer['longitude']],
                       [dc['latitude'], customer['latitude']],
                       'r--', alpha=0.3, linewidth=width, zorder=1)
            except (IndexError, KeyError):
                continue
    
    # Add labels
    for _, row in suppliers.iterrows():
        ax.annotate(row['supplier_id'], (row['longitude'], row['latitude']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    for _, row in factories.iterrows():
        ax.annotate(row['factory_id'], (row['longitude'], row['latitude']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    for _, row in dcs.iterrows():
        ax.annotate(row['dc_id'], (row['longitude'], row['latitude']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title(f'Network Flows: {solution_label.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'network_flows_{solution_label}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved network flows ({solution_label}): {OUTPUT_DIR}/network_flows_{solution_label}.png")


def plot_demand_forecast():
    """Plot historical and forecasted demand."""
    demand_file = os.path.join(DATA_DIR, 'demand.csv')
    if not os.path.exists(demand_file):
        return
    
    demand = pd.read_csv(demand_file)
    
    # Use first 60 periods as historical, rest as forecast (if available)
    historical = demand[demand['period'] <= 60].copy()
    forecast_periods = demand[demand['period'] > 60]
    
    fig, axes = plt.subplots(len(demand['customer_id'].unique()), 1, 
                           figsize=(14, 4 * len(demand['customer_id'].unique())))
    if len(demand['customer_id'].unique()) == 1:
        axes = [axes]
    
    for idx, customer_id in enumerate(sorted(demand['customer_id'].unique())):
        ax = axes[idx]
        cust_hist = historical[historical['customer_id'] == customer_id].sort_values('period')
        cust_forecast = forecast_periods[forecast_periods['customer_id'] == customer_id].sort_values('period')
        
        ax.plot(cust_hist['period'], cust_hist['demand'], 'o-', 
               label='Historical', color='steelblue', linewidth=2, markersize=4)
        if len(cust_forecast) > 0:
            ax.plot(cust_forecast['period'], cust_forecast['demand'], 's--',
                   label='Forecast', color='coral', linewidth=2, markersize=4)
        
        ax.axvline(x=60, color='red', linestyle='--', alpha=0.5, label='Forecast Start')
        ax.set_xlabel('Period', fontsize=10, fontweight='bold')
        ax.set_ylabel('Demand (units)', fontsize=10, fontweight='bold')
        ax.set_title(f'Customer {customer_id}: Demand Forecast', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Demand Forecasting: Historical vs Forecast', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'demand_forecast.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved demand forecast: {OUTPUT_DIR}/demand_forecast.png")


def plot_simulation_results():
    """Plot simulation resilience metrics."""
    summary_file = os.path.join(RESULTS_DIR, 'experiment_summary.json')
    if not os.path.exists(summary_file):
        return
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    sim_results = summary.get('simulation_results', {})
    if not sim_results:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Fill rate metrics
    ax = axes[0]
    metrics = ['average_fill_rate', 'min_fill_rate', 'resilience_score']
    values = [sim_results.get(m, 0) * 100 for m in metrics]  # Convert to percentage
    labels = ['Average Fill Rate', 'Min Fill Rate', 'Resilience Score']
    colors = ['steelblue', 'coral', 'darkgreen']
    
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('Resilience Metrics from Simulation', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Disruption statistics
    ax = axes[1]
    n_disruptions = sim_results.get('average_n_disruptions', 0)
    n_runs = sim_results.get('n_runs', 100)
    
    ax.bar(['Average Disruptions\nper Run'], [n_disruptions], 
          color='orange', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Number of Disruptions', fontsize=11, fontweight='bold')
    ax.set_title(f'Disruption Statistics (n={n_runs} runs)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value label
    ax.text(0, n_disruptions + 0.1, f'{n_disruptions:.2f}', 
           ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.suptitle('Simulation Results: Resilience Evaluation', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'simulation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved simulation results: {OUTPUT_DIR}/simulation_results.png")


def plot_forecast_accuracy():
    """Plot forecast accuracy metrics."""
    summary_file = os.path.join(RESULTS_DIR, 'experiment_summary.json')
    if not os.path.exists(summary_file):
        return
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    forecast_metrics = summary.get('forecast_accuracy', {})
    if not forecast_metrics:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = ['MAE', 'RMSE', 'MAPE']
    values = [forecast_metrics.get(m, 0) for m in metrics]
    colors = ['steelblue', 'darkgreen', 'coral']
    
    for ax, metric, val, color in zip(axes, metrics, values, colors):
        ax.bar([metric], [val], color=color, alpha=0.8, edgecolor='black', linewidth=1.5, width=0.5)
        ax.set_ylabel('Value' if metric != 'MAPE' else 'Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value label
        label = f'{val:.2f}%' if metric == 'MAPE' else f'{val:.2f}'
        ax.text(0, val + (max(values) * 0.05), label, 
               ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.suptitle('Forecast Accuracy Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'forecast_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved forecast accuracy: {OUTPUT_DIR}/forecast_accuracy.png")


def plot_data_summary():
    """Plot summary statistics of generated data."""
    suppliers_file = os.path.join(DATA_DIR, 'suppliers.csv')
    factories_file = os.path.join(DATA_DIR, 'factories.csv')
    dcs_file = os.path.join(DATA_DIR, 'dcs.csv')
    demand_file = os.path.join(DATA_DIR, 'demand.csv')
    
    if not all(os.path.exists(f) for f in [suppliers_file, factories_file, dcs_file, demand_file]):
        return
    
    suppliers = pd.read_csv(suppliers_file)
    factories = pd.read_csv(factories_file)
    dcs = pd.read_csv(dcs_file)
    demand = pd.read_csv(demand_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Supplier capacities and costs
    ax = axes[0, 0]
    x = np.arange(len(suppliers))
    width = 0.35
    ax.bar(x - width/2, suppliers['capacity'], width, label='Capacity', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, suppliers['cost_per_unit'] * 10, width, label='Cost (×10)', color='coral', alpha=0.8)
    ax.set_xlabel('Supplier', fontsize=10, fontweight='bold')
    ax.set_ylabel('Value', fontsize=10, fontweight='bold')
    ax.set_title('Supplier Capacities and Costs', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(suppliers['supplier_id'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Factory capacities and production costs
    ax = axes[0, 1]
    x = np.arange(len(factories))
    ax.bar(x - width/2, factories['capacity'], width, label='Capacity', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, factories['production_cost'] * 5, width, label='Production Cost (×5)', color='darkgreen', alpha=0.8)
    ax.set_xlabel('Factory', fontsize=10, fontweight='bold')
    ax.set_ylabel('Value', fontsize=10, fontweight='bold')
    ax.set_title('Factory Capacities and Production Costs', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(factories['factory_id'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # DC capacities and holding costs
    ax = axes[1, 0]
    x = np.arange(len(dcs))
    ax.bar(x - width/2, dcs['capacity'], width, label='Capacity', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, dcs['holding_cost'] * 50, width, label='Holding Cost (×50)', color='orange', alpha=0.8)
    ax.set_xlabel('Distribution Center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Value', fontsize=10, fontweight='bold')
    ax.set_title('DC Capacities and Holding Costs', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dcs['dc_id'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Demand distribution
    ax = axes[1, 1]
    ax.hist(demand['demand'], bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Demand (units)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title('Demand Distribution', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Generated Data Summary Statistics', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'data_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved data summary: {OUTPUT_DIR}/data_summary.png")


def plot_distance_heatmaps():
    """Plot distance matrices as heatmaps."""
    dist_sf_file = os.path.join(DATA_DIR, 'dist_supplier_factory.csv')
    dist_fd_file = os.path.join(DATA_DIR, 'dist_factory_dc.csv')
    dist_dc_file = os.path.join(DATA_DIR, 'dist_dc_customer.csv')
    
    if not all(os.path.exists(f) for f in [dist_sf_file, dist_fd_file, dist_dc_file]):
        return
    
    dist_sf = pd.read_csv(dist_sf_file)
    dist_fd = pd.read_csv(dist_fd_file)
    dist_dc = pd.read_csv(dist_dc_file)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Supplier-Factory distances
    ax = axes[0]
    sf_pivot = dist_sf.pivot(index='supplier_id', columns='factory_id', values='distance_km')
    sns.heatmap(sf_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Distance (km)'})
    ax.set_title('Supplier → Factory Distances', fontsize=12, fontweight='bold')
    ax.set_xlabel('Factory', fontsize=10, fontweight='bold')
    ax.set_ylabel('Supplier', fontsize=10, fontweight='bold')
    
    # Factory-DC distances
    ax = axes[1]
    fd_pivot = dist_fd.pivot(index='factory_id', columns='dc_id', values='distance_km')
    sns.heatmap(fd_pivot, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Distance (km)'})
    ax.set_title('Factory → DC Distances', fontsize=12, fontweight='bold')
    ax.set_xlabel('DC', fontsize=10, fontweight='bold')
    ax.set_ylabel('Factory', fontsize=10, fontweight='bold')
    
    # DC-Customer distances
    ax = axes[2]
    dc_pivot = dist_dc.pivot(index='dc_id', columns='customer_id', values='distance_km')
    sns.heatmap(dc_pivot, annot=True, fmt='.1f', cmap='RdPu', ax=ax, cbar_kws={'label': 'Distance (km)'})
    ax.set_title('DC → Customer Distances', fontsize=12, fontweight='bold')
    ax.set_xlabel('Customer', fontsize=10, fontweight='bold')
    ax.set_ylabel('DC', fontsize=10, fontweight='bold')
    
    plt.suptitle('Distance Matrices', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'distance_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved distance heatmaps: {OUTPUT_DIR}/distance_heatmaps.png")


def plot_congestion_analysis():
    """Plot congestion factors across network arcs."""
    dist_sf_file = os.path.join(DATA_DIR, 'dist_supplier_factory.csv')
    dist_fd_file = os.path.join(DATA_DIR, 'dist_factory_dc.csv')
    dist_dc_file = os.path.join(DATA_DIR, 'dist_dc_customer.csv')
    
    if not all(os.path.exists(f) for f in [dist_sf_file, dist_fd_file, dist_dc_file]):
        return
    
    dist_sf = pd.read_csv(dist_sf_file)
    dist_fd = pd.read_csv(dist_fd_file)
    dist_dc = pd.read_csv(dist_dc_file)
    
    # Check if congestion_factor column exists
    if 'congestion_factor' not in dist_sf.columns:
        print("Warning: congestion_factor not found in distance files. Skipping congestion analysis.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Supplier-Factory congestion
    ax = axes[0]
    sf_pivot = dist_sf.pivot(index='supplier_id', columns='factory_id', values='congestion_factor')
    sns.heatmap(sf_pivot, annot=True, fmt='.2f', cmap='Reds', ax=ax, 
               cbar_kws={'label': 'Congestion Factor'}, vmin=1.0, vmax=2.0)
    ax.set_title('Supplier → Factory Congestion', fontsize=12, fontweight='bold')
    ax.set_xlabel('Factory', fontsize=10, fontweight='bold')
    ax.set_ylabel('Supplier', fontsize=10, fontweight='bold')
    
    # Factory-DC congestion
    ax = axes[1]
    fd_pivot = dist_fd.pivot(index='factory_id', columns='dc_id', values='congestion_factor')
    sns.heatmap(fd_pivot, annot=True, fmt='.2f', cmap='Oranges', ax=ax,
               cbar_kws={'label': 'Congestion Factor'}, vmin=1.0, vmax=2.0)
    ax.set_title('Factory → DC Congestion', fontsize=12, fontweight='bold')
    ax.set_xlabel('DC', fontsize=10, fontweight='bold')
    ax.set_ylabel('Factory', fontsize=10, fontweight='bold')
    
    # DC-Customer congestion
    ax = axes[2]
    dc_pivot = dist_dc.pivot(index='dc_id', columns='customer_id', values='congestion_factor')
    sns.heatmap(dc_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
               cbar_kws={'label': 'Congestion Factor'}, vmin=1.0, vmax=2.0)
    ax.set_title('DC → Customer Congestion', fontsize=12, fontweight='bold')
    ax.set_xlabel('Customer', fontsize=10, fontweight='bold')
    ax.set_ylabel('DC', fontsize=10, fontweight='bold')
    
    plt.suptitle('Traffic Congestion Factors Across Network', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'congestion_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved congestion analysis: {OUTPUT_DIR}/congestion_analysis.png")


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    print()
    
    create_output_dir()
    
    # Generate all plots
    try:
        plot_pareto_front_3d()
        plot_pareto_front_2d()
        plot_representative_solutions()
        plot_network_topology()
        plot_demand_forecast()
        plot_simulation_results()
        plot_forecast_accuracy()
        plot_data_summary()
        plot_distance_heatmaps()
        plot_congestion_analysis()
        
        # Plot flows for each representative solution
        summary_file = os.path.join(RESULTS_DIR, 'experiment_summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            reps = summary.get('representative_solutions', [])
            for sol in reps:
                # Match the label format used in run_all.sh: replace(' ', '_').lower()
                label = sol['label'].replace(' ', '_').lower()
                plot_network_with_flows(label)
        
        print()
        print("=" * 60)
        print("✓ All visualizations generated successfully!")
        print("=" * 60)
        print(f"\nFigures saved in: {OUTPUT_DIR}/")
        print("\nGenerated files:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            if f.endswith('.png'):
                print(f"  - {f}")
        
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()


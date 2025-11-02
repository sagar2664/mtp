"""
Analysis and visualization module.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os


def analyze_solution(
    solution: Dict,
    suppliers: pd.DataFrame,
    facilities: pd.DataFrame,
    customers: pd.DataFrame,
    demand: pd.DataFrame
) -> Dict:
    """
    Analyze optimization solution and compute key metrics.
    
    Args:
        solution: Solution dictionary from optimizer
        suppliers: Supplier data
        facilities: Facility data
        customers: Customer data
        demand: Demand data
    
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Flow analysis
    if solution and 'flows_fc' in solution and len(solution['flows_fc']) > 0:
        flows_fc_df = pd.DataFrame(solution['flows_fc'])
        
        # Total flow by facility
        facility_flow = flows_fc_df.groupby('facility_id')['flow'].sum()
        analysis['facility_utilization'] = facility_flow.to_dict()
        
        # Total flow by period
        period_flow = flows_fc_df.groupby('period')['flow'].sum()
        analysis['period_total_flow'] = period_flow.to_dict()
        
        # Demand fulfillment
        demand_total = demand.groupby('period')['demand'].sum()
        fulfillment_rate = {}
        for period in demand_total.index:
            supplied = period_flow.get(period, 0)
            required = demand_total[period]
            fulfillment_rate[period] = (supplied / required * 100) if required > 0 else 100
        
        analysis['demand_fulfillment_pct'] = fulfillment_rate
    
    # Facility opening analysis
    if solution and 'facilities_open' in solution:
        analysis['facilities_opened'] = solution['facilities_open']
        analysis['n_facilities_opened'] = len(solution['facilities_open'])
    else:
        analysis['facilities_opened'] = []
        analysis['n_facilities_opened'] = 0
    
    # Inventory analysis
    if solution and 'inventory' in solution and len(solution['inventory']) > 0:
        inventory_df = pd.DataFrame(solution['inventory'])
        avg_inventory = inventory_df.groupby('facility_id')['inventory'].mean()
        analysis['avg_inventory_by_facility'] = avg_inventory.to_dict()
    else:
        analysis['avg_inventory_by_facility'] = {}
    
    return analysis


def plot_network(
    suppliers: pd.DataFrame,
    facilities: pd.DataFrame,
    customers: pd.DataFrame,
    solution: Optional[Dict] = None,
    save_path: Optional[str] = None
):
    """
    Plot supply chain network visualization.
    
    Args:
        suppliers: Supplier data
        facilities: Facility data
        customers: Customer data
        solution: Optional solution to highlight flows
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot suppliers
    ax.scatter(
        suppliers['longitude'],
        suppliers['latitude'],
        c='green',
        marker='s',
        s=100,
        label='Suppliers',
        zorder=3
    )
    
    # Plot facilities
    ax.scatter(
        facilities['longitude'],
        facilities['latitude'],
        c='blue',
        marker='^',
        s=150,
        label='Facilities',
        zorder=3
    )
    
    # Plot customers
    ax.scatter(
        customers['longitude'],
        customers['latitude'],
        c='red',
        marker='o',
        s=50,
        label='Customers',
        zorder=2,
        alpha=0.6
    )
    
    # Plot flows if solution provided
    if solution:
        if 'flows_sf' in solution:
            flows_sf_df = pd.DataFrame(solution['flows_sf'])
            if len(flows_sf_df) > 0:
                for _, flow in flows_sf_df.iterrows():
                    supplier = suppliers[suppliers['supplier_id'] == flow['supplier_id']].iloc[0]
                    facility = facilities[facilities['facility_id'] == flow['facility_id']].iloc[0]
                    ax.plot(
                        [supplier['longitude'], facility['longitude']],
                        [supplier['latitude'], facility['latitude']],
                        'g-',
                        alpha=0.3,
                        linewidth=flow['flow'] / 10,
                        zorder=1
                    )
        
        if 'flows_fc' in solution:
            flows_fc_df = pd.DataFrame(solution['flows_fc'])
            if len(flows_fc_df) > 0:
                # Aggregate flows by facility-customer pair
                fc_flows = flows_fc_df.groupby(['facility_id', 'customer_id'])['flow'].sum()
                for (f_id, c_id), flow_val in fc_flows.items():
                    facility = facilities[facilities['facility_id'] == f_id].iloc[0]
                    customer = customers[customers['customer_id'] == c_id].iloc[0]
                    ax.plot(
                        [facility['longitude'], customer['longitude']],
                        [facility['latitude'], customer['latitude']],
                        'b--',
                        alpha=0.2,
                        linewidth=min(flow_val / 10, 3),
                        zorder=1
                    )
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Supply Chain Network')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_demand_forecast(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot historical and forecasted demand.
    
    Args:
        historical: Historical demand DataFrame
        forecast: Forecasted demand DataFrame
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical demand
    for customer_id in historical['customer_id'].unique():
        customer_hist = historical[historical['customer_id'] == customer_id].sort_values('period')
        ax.plot(
            customer_hist['period'],
            customer_hist['demand'],
            'o-',
            alpha=0.5,
            label=f'Customer {customer_id} (Historical)'
        )
    
    # Plot forecast
    for customer_id in forecast['customer_id'].unique():
        customer_forecast = forecast[forecast['customer_id'] == customer_id].sort_values('period')
        ax.plot(
            customer_forecast['period'],
            customer_forecast['demand'],
            's--',
            alpha=0.7,
            label=f'Customer {customer_id} (Forecast)'
        )
    
    ax.set_xlabel('Period')
    ax.set_ylabel('Demand')
    ax.set_title('Demand Forecast')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_solution_metrics(
    analysis: Dict,
    save_path: Optional[str] = None
):
    """
    Plot solution metrics and KPIs.
    
    Args:
        analysis: Analysis results dictionary
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Facility utilization
    if 'facility_utilization' in analysis:
        ax = axes[0, 0]
        util_data = analysis['facility_utilization']
        ax.bar(util_data.keys(), util_data.values())
        ax.set_xlabel('Facility ID')
        ax.set_ylabel('Total Flow')
        ax.set_title('Facility Utilization')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Demand fulfillment over time
    if 'demand_fulfillment_pct' in analysis:
        ax = axes[0, 1]
        fulfillment = analysis['demand_fulfillment_pct']
        ax.plot(fulfillment.keys(), fulfillment.values(), 'o-')
        ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% Target')
        ax.set_xlabel('Period')
        ax.set_ylabel('Fulfillment %')
        ax.set_title('Demand Fulfillment Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Period total flow
    if 'period_total_flow' in analysis:
        ax = axes[1, 0]
        flow_data = analysis['period_total_flow']
        ax.bar(flow_data.keys(), flow_data.values())
        ax.set_xlabel('Period')
        ax.set_ylabel('Total Flow')
        ax.set_title('Total Flow by Period')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Average inventory
    if 'avg_inventory_by_facility' in analysis:
        ax = axes[1, 1]
        inv_data = analysis['avg_inventory_by_facility']
        if inv_data:
            ax.bar(inv_data.keys(), inv_data.values())
            ax.set_xlabel('Facility ID')
            ax.set_ylabel('Average Inventory')
            ax.set_title('Average Inventory by Facility')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def export_results(
    results: Dict,
    output_dir: str = './results'
):
    """
    Export results to CSV files.
    
    Args:
        results: Results dictionary
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Export solution flows
    if 'solution' in results and results['solution']:
        solution = results['solution']
        
        if 'flows_sf' in solution and solution['flows_sf']:
            pd.DataFrame(solution['flows_sf']).to_csv(
                f'{output_dir}/flows_supplier_facility.csv',
                index=False
            )
        
        if 'flows_fc' in solution and solution['flows_fc']:
            pd.DataFrame(solution['flows_fc']).to_csv(
                f'{output_dir}/flows_facility_customer.csv',
                index=False
            )
        
        if 'inventory' in solution and solution['inventory']:
            pd.DataFrame(solution['inventory']).to_csv(
                f'{output_dir}/inventory_levels.csv',
                index=False
            )
    
    # Export analysis
    if 'analysis' in results:
        analysis = results['analysis']
        # Convert dicts to DataFrames
        analysis_df = pd.DataFrame([analysis])
        analysis_df.to_csv(f'{output_dir}/analysis_summary.csv', index=False)
    
    # Export simulation results
    if 'simulation' in results:
        sim_df = pd.DataFrame([results['simulation']])
        sim_df.to_csv(f'{output_dir}/simulation_results.csv', index=False)


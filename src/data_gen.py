"""
Data generation module for creating synthetic supply chain data.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict


def generate_supply_chain_data(
    n_suppliers: int = 3,
    n_factories: int = 2,
    n_dcs: int = 3,
    n_customers: int = 5,
    n_periods: int = 60,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic 4-echelon supply chain data.
    Structure: Suppliers → Factories → Distribution Centers → Customers
    
    Args:
        n_suppliers: Number of suppliers (default: 3)
        n_factories: Number of factories (default: 2)
        n_dcs: Number of distribution centers (default: 3)
        n_customers: Number of customer zones (default: 5)
        n_periods: Number of time periods (months)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing DataFrames for suppliers, factories, DCs, customers, and demand
    """
    np.random.seed(seed)
    
    # Generate supplier data (S1, S2, S3)
    suppliers = pd.DataFrame({
        'supplier_id': [f'S{i+1}' for i in range(n_suppliers)],
        'capacity': np.random.uniform(200, 400, n_suppliers),
        'cost_per_unit': np.random.uniform(10, 30, n_suppliers),
        'latitude': np.random.uniform(20, 25, n_suppliers),  # India coordinates
        'longitude': np.random.uniform(75, 85, n_suppliers),
        'carbon_emission_factor': np.random.uniform(0.01, 0.03, n_suppliers)  # kg CO2 per unit
    })
    
    # Generate factory data (F1, F2)
    factories = pd.DataFrame({
        'factory_id': [f'F{i+1}' for i in range(n_factories)],
        'capacity': np.random.uniform(300, 500, n_factories),
        'production_cost': np.random.uniform(40, 60, n_factories),  # per unit
        'latitude': np.random.uniform(20, 25, n_factories),
        'longitude': np.random.uniform(75, 85, n_factories),
        'production_emission_factor': np.random.uniform(0.02, 0.05, n_factories)  # kg CO2 per unit
    })
    
    # Generate DC data (D1, D2, D3)
    dcs = pd.DataFrame({
        'dc_id': [f'D{i+1}' for i in range(n_dcs)],
        'capacity': np.random.uniform(250, 400, n_dcs),
        'holding_cost': np.random.uniform(2, 5, n_dcs),  # per unit per period
        'latitude': np.random.uniform(20, 25, n_dcs),
        'longitude': np.random.uniform(75, 85, n_dcs)
    })
    
    # Generate customer data (C1-C5)
    customers = pd.DataFrame({
        'customer_id': [f'C{i+1}' for i in range(n_customers)],
        'latitude': np.random.uniform(18, 28, n_customers),
        'longitude': np.random.uniform(72, 88, n_customers),
        'service_level': np.random.uniform(0.90, 0.98, n_customers)
    })
    
    # Generate demand data with richer, harder-to-forecast patterns
    # Base: demand = 100 + 2*t + 20*sin(2*pi*t/12) + noise
    # Add: customer-specific regime shifts, multiplicative shocks, occasional outliers
    demand = []
    # Customer-specific regime settings
    regime_shift_periods = {cid: np.random.choice(range(12, n_periods-12)) for cid in customers['customer_id']}
    regime_multipliers = {cid: np.random.uniform(0.7, 1.3) for cid in customers['customer_id']}
    for period in range(1, n_periods + 1):
        for customer_id in customers['customer_id']:
            # Base trend + seasonality
            t = period
            base = 100 + 2 * t
            seasonal = 20 * np.sin(2 * np.pi * t / 12)
            # Additive noise (higher variance)
            noise = 12 * np.random.randn()
            demand_value = base + seasonal + noise
            # Regime shift (multiplicative) after a change point
            if period >= regime_shift_periods[customer_id]:
                demand_value *= regime_multipliers[customer_id]
            # Occasional shocks (10% periods)
            if np.random.rand() < 0.1:
                shock = np.random.choice([0.6, 1.5])
                demand_value *= shock
            demand.append({
                'period': period,
                'customer_id': customer_id,
                'demand': max(10, demand_value)
            })
    
    demand_df = pd.DataFrame(demand)
    
    return {
        'suppliers': suppliers,
        'factories': factories,
        'dcs': dcs,
        'customers': customers,
        'demand': demand_df
    }


def generate_distance_matrix(
    suppliers: pd.DataFrame,
    factories: pd.DataFrame,
    dcs: pd.DataFrame,
    customers: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Generate distance matrices for 4-echelon network.
    Uses geodesic distance (haversine formula) via geopy.
    
    Args:
        suppliers: DataFrame with supplier locations
        factories: DataFrame with factory locations
        dcs: DataFrame with DC locations
        customers: DataFrame with customer locations
    
    Returns:
        Dictionary with distance matrices for all arcs
    """
    from geopy.distance import geodesic
    
    distances = {}
    
    # Supplier to Factory distances (x_ij)
    sf_distances = []
    for _, supplier in suppliers.iterrows():
        for _, factory in factories.iterrows():
            dist = geodesic(
                (supplier['latitude'], supplier['longitude']),
                (factory['latitude'], factory['longitude'])
            ).kilometers
            # Congestion factor > 1 inflates travel time/cost/emissions
            congestion = np.random.uniform(1.0, 1.5)
            sf_distances.append({
                'supplier_id': supplier['supplier_id'],
                'factory_id': factory['factory_id'],
                'distance_km': dist,
                'congestion_factor': congestion
            })
    distances['supplier_factory'] = pd.DataFrame(sf_distances)
    
    # Factory to DC distances (y_jk)
    fd_distances = []
    for _, factory in factories.iterrows():
        for _, dc in dcs.iterrows():
            dist = geodesic(
                (factory['latitude'], factory['longitude']),
                (dc['latitude'], dc['longitude'])
            ).kilometers
            congestion = np.random.uniform(1.0, 1.6)
            fd_distances.append({
                'factory_id': factory['factory_id'],
                'dc_id': dc['dc_id'],
                'distance_km': dist,
                'congestion_factor': congestion
            })
    distances['factory_dc'] = pd.DataFrame(fd_distances)
    
    # DC to Customer distances (z_kl)
    dc_distances = []
    for _, dc in dcs.iterrows():
        for _, customer in customers.iterrows():
            dist = geodesic(
                (dc['latitude'], dc['longitude']),
                (customer['latitude'], customer['longitude'])
            ).kilometers
            congestion = np.random.uniform(1.0, 1.7)
            dc_distances.append({
                'dc_id': dc['dc_id'],
                'customer_id': customer['customer_id'],
                'distance_km': dist,
                'congestion_factor': congestion
            })
    distances['dc_customer'] = pd.DataFrame(dc_distances)
    
    return distances


def save_supply_chain_data(
    data: Dict[str, pd.DataFrame],
    distances: Dict[str, pd.DataFrame],
    output_dir: str = './data'
) -> None:
    """
    Save generated supply chain data and distance matrices to CSV files.

    Args:
        data: Dictionary with DataFrames for 'suppliers', 'factories', 'dcs', 'customers', 'demand'
        distances: Dictionary with DataFrames for 'supplier_factory', 'factory_dc', 'dc_customer'
        output_dir: Directory to write CSV files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if 'suppliers' in data:
        data['suppliers'].to_csv(f"{output_dir}/suppliers.csv", index=False)
    if 'factories' in data:
        data['factories'].to_csv(f"{output_dir}/factories.csv", index=False)
    if 'dcs' in data:
        data['dcs'].to_csv(f"{output_dir}/dcs.csv", index=False)
    if 'customers' in data:
        data['customers'].to_csv(f"{output_dir}/customers.csv", index=False)
    if 'demand' in data:
        data['demand'].to_csv(f"{output_dir}/demand.csv", index=False)

    if distances and 'supplier_factory' in distances:
        distances['supplier_factory'].to_csv(f"{output_dir}/dist_supplier_factory.csv", index=False)
    if distances and 'factory_dc' in distances:
        distances['factory_dc'].to_csv(f"{output_dir}/dist_factory_dc.csv", index=False)
    if distances and 'dc_customer' in distances:
        distances['dc_customer'].to_csv(f"{output_dir}/dist_dc_customer.csv", index=False)


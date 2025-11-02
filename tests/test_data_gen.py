"""
Tests for data generation module (4-echelon structure).
"""
import pytest
import pandas as pd
import numpy as np
from src.data_gen import generate_supply_chain_data, generate_distance_matrix


def test_data_shapes():
    """Test that generated data has correct shapes and columns for 4-echelon network."""
    data = generate_supply_chain_data(
        n_suppliers=3,
        n_factories=2,
        n_dcs=3,
        n_customers=5,
        n_periods=24,
        seed=42
    )
    
    # Check suppliers
    assert 'suppliers' in data
    assert len(data['suppliers']) == 3
    required_cols = ['supplier_id', 'capacity', 'cost_per_unit', 'latitude', 'longitude', 'carbon_emission_factor']
    for col in required_cols:
        assert col in data['suppliers'].columns
    
    # Check factories
    assert 'factories' in data
    assert len(data['factories']) == 2
    required_cols = ['factory_id', 'capacity', 'production_cost', 'latitude', 'longitude', 'production_emission_factor']
    for col in required_cols:
        assert col in data['factories'].columns
    
    # Check DCs
    assert 'dcs' in data
    assert len(data['dcs']) == 3
    required_cols = ['dc_id', 'capacity', 'holding_cost', 'latitude', 'longitude']
    for col in required_cols:
        assert col in data['dcs'].columns
    
    # Check customers
    assert 'customers' in data
    assert len(data['customers']) == 5
    required_cols = ['customer_id', 'latitude', 'longitude', 'service_level']
    for col in required_cols:
        assert col in data['customers'].columns
    
    # Check demand
    assert 'demand' in data
    assert len(data['demand']) == 24 * 5  # n_periods * n_customers
    required_cols = ['period', 'customer_id', 'demand']
    for col in required_cols:
        assert col in data['demand'].columns
    
    # Check demand values are non-negative
    assert (data['demand']['demand'] >= 0).all()


def test_data_values():
    """Test that generated data has reasonable values."""
    data = generate_supply_chain_data(seed=42)
    
    # Check capacity values are positive
    assert (data['suppliers']['capacity'] > 0).all()
    assert (data['factories']['capacity'] > 0).all()
    assert (data['dcs']['capacity'] > 0).all()
    
    # Check cost values are positive
    assert (data['suppliers']['cost_per_unit'] > 0).all()
    assert (data['factories']['production_cost'] > 0).all()
    assert (data['dcs']['holding_cost'] > 0).all()


def test_distance_matrix():
    """Test distance matrix generation for 4-echelon network."""
    data = generate_supply_chain_data(seed=42)
    distances = generate_distance_matrix(
        data['suppliers'],
        data['factories'],
        data['dcs'],
        data['customers']
    )
    
    assert 'supplier_factory' in distances
    assert 'factory_dc' in distances
    assert 'dc_customer' in distances
    
    # Check supplier-factory distances
    sf_df = distances['supplier_factory']
    assert len(sf_df) == len(data['suppliers']) * len(data['factories'])
    assert 'distance_km' in sf_df.columns
    assert (sf_df['distance_km'] >= 0).all()
    
    # Check factory-DC distances
    fd_df = distances['factory_dc']
    assert len(fd_df) == len(data['factories']) * len(data['dcs'])
    assert 'distance_km' in fd_df.columns
    assert (fd_df['distance_km'] >= 0).all()
    
    # Check DC-customer distances
    dc_df = distances['dc_customer']
    assert len(dc_df) == len(data['dcs']) * len(data['customers'])
    assert 'distance_km' in dc_df.columns
    assert (dc_df['distance_km'] >= 0).all()


def test_reproducibility():
    """Test that same seed produces same results."""
    data1 = generate_supply_chain_data(seed=42)
    data2 = generate_supply_chain_data(seed=42)
    
    pd.testing.assert_frame_equal(data1['suppliers'], data2['suppliers'])
    pd.testing.assert_frame_equal(data1['factories'], data2['factories'])
    pd.testing.assert_frame_equal(data1['dcs'], data2['dcs'])
    pd.testing.assert_frame_equal(data1['customers'], data2['customers'])
    pd.testing.assert_frame_equal(data1['demand'], data2['demand'])

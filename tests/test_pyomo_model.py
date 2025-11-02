"""
Tests for Pyomo model feasibility and structure (4-echelon multi-objective).
"""
import pytest
from pyomo.environ import value
from src.data_gen import generate_supply_chain_data, generate_distance_matrix
from src.pyomo_model import create_multi_objective_model


def test_pyomo_model_creation():
    """Test that Pyomo model can be created successfully."""
    data = generate_supply_chain_data(
        n_suppliers=3,
        n_factories=2,
        n_dcs=3,
        n_customers=5,
        n_periods=6,
        seed=42
    )
    distances = generate_distance_matrix(
        data['suppliers'],
        data['factories'],
        data['dcs'],
        data['customers']
    )
    
    # Aggregate demand for single period
    demand_avg = data['demand'].groupby('customer_id')['demand'].mean().reset_index()
    
    model = create_multi_objective_model(
        data['suppliers'],
        data['factories'],
        data['dcs'],
        data['customers'],
        demand_avg,
        distances
    )
    
    # Check model has required components
    assert hasattr(model, 'I')  # Suppliers set
    assert hasattr(model, 'J')  # Factories set
    assert hasattr(model, 'K')  # DCs set
    assert hasattr(model, 'L')  # Customers set
    
    assert hasattr(model, 'cost_objective')
    assert hasattr(model, 'emission_objective_expr')
    assert hasattr(model, 'resilience_objective_expr')
    assert hasattr(model, 'supply_limit')
    assert hasattr(model, 'factory_balance')
    assert hasattr(model, 'dc_balance')
    assert hasattr(model, 'demand_satisfaction')


def test_pyomo_model_variables():
    """Test that model variables are defined correctly."""
    data = generate_supply_chain_data(
        n_suppliers=3,
        n_factories=2,
        n_dcs=3,
        n_customers=5,
        n_periods=3,
        seed=42
    )
    distances = generate_distance_matrix(
        data['suppliers'],
        data['factories'],
        data['dcs'],
        data['customers']
    )
    demand_avg = data['demand'].groupby('customer_id')['demand'].mean().reset_index()
    
    model = create_multi_objective_model(
        data['suppliers'],
        data['factories'],
        data['dcs'],
        data['customers'],
        demand_avg,
        distances
    )
    
    # Check variables exist
    assert hasattr(model, 'x')  # Supplier-factory flow
    assert hasattr(model, 'y')  # Factory-DC flow
    assert hasattr(model, 'z')  # DC-customer flow


def test_pyomo_model_parameters():
    """Test that model parameters are set correctly."""
    data = generate_supply_chain_data(seed=42)
    distances = generate_distance_matrix(
        data['suppliers'],
        data['factories'],
        data['dcs'],
        data['customers']
    )
    demand_avg = data['demand'].groupby('customer_id')['demand'].mean().reset_index()
    
    model = create_multi_objective_model(
        data['suppliers'],
        data['factories'],
        data['dcs'],
        data['customers'],
        demand_avg,
        distances
    )
    
    # Check parameter values
    for i in model.I:
        assert model.supply_capacity[i] > 0
        assert model.supplier_cost[i] > 0
    
    for j in model.J:
        assert model.factory_capacity[j] > 0
        assert model.production_cost[j] > 0
    
    for k in model.K:
        assert model.dc_capacity[k] > 0
        assert model.holding_cost[k] > 0
    
    # Check demand parameters
    for l in model.L:
        assert model.demand[l] >= 0

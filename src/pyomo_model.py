"""
Pyomo model definition for 4-echelon multi-objective supply chain optimization.
Network: Suppliers (I) → Factories (J) → Distribution Centers (K) → Customers (L)

Objectives:
1. Minimize total cost (supplier + production + transportation + inventory)
2. Minimize total carbon emissions (transportation + production)
3. Maximize resilience (minimize worst-case demand fulfillment under disruptions)
"""
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, Set, Param,
    minimize, NonNegativeReals
)
import pandas as pd
from typing import Dict
import numpy as np


def create_multi_objective_model(
    suppliers: pd.DataFrame,
    factories: pd.DataFrame,
    dcs: pd.DataFrame,
    customers: pd.DataFrame,
    demand: pd.DataFrame,
    distances: Dict[str, pd.DataFrame]
) -> ConcreteModel:
    """
    Create a multi-objective Pyomo optimization model for 4-echelon supply chain.
    
    Decision Variables:
    - x_ij: Flow from supplier i to factory j
    - y_jk: Flow from factory j to DC k
    - z_kl: Flow from DC k to customer l
    
    Objectives:
    - Cost: Minimize total supply chain cost
    - Emissions: Minimize total CO2 emissions
    - Resilience: Maximize minimum demand coverage under single-node failures
    
    Args:
        suppliers: Supplier data with capacity, cost_per_unit, carbon_emission_factor
        factories: Factory data with capacity, production_cost, production_emission_factor
        dcs: DC data with capacity, holding_cost
        customers: Customer data
        demand: Demand by customer (single period or aggregated)
        distances: Distance matrices for all arcs
    
    Returns:
        Pyomo ConcreteModel instance
    """
    model = ConcreteModel(name="MultiObjectiveGreenResilientSCN")
    
    # Sets
    model.I = Set(initialize=suppliers['supplier_id'].tolist())  # Suppliers
    model.J = Set(initialize=factories['factory_id'].tolist())   # Factories
    model.K = Set(initialize=dcs['dc_id'].tolist())              # Distribution Centers
    model.L = Set(initialize=customers['customer_id'].tolist())   # Customers
    
    # Parameters
    # Supplier parameters
    supplier_data = suppliers.set_index('supplier_id')
    model.supply_capacity = Param(
        model.I,
        initialize=supplier_data['capacity'].to_dict()
    )
    model.supplier_cost = Param(
        model.I,
        initialize=supplier_data['cost_per_unit'].to_dict()
    )
    model.supplier_emission_factor = Param(
        model.I,
        initialize=supplier_data['carbon_emission_factor'].to_dict()
    )
    
    # Factory parameters
    factory_data = factories.set_index('factory_id')
    model.factory_capacity = Param(
        model.J,
        initialize=factory_data['capacity'].to_dict()
    )
    model.production_cost = Param(
        model.J,
        initialize=factory_data['production_cost'].to_dict()
    )
    model.production_emission_factor = Param(
        model.J,
        initialize=factory_data['production_emission_factor'].to_dict()
    )
    
    # DC parameters
    dc_data = dcs.set_index('dc_id')
    model.dc_capacity = Param(
        model.K,
        initialize=dc_data['capacity'].to_dict()
    )
    model.holding_cost = Param(
        model.K,
        initialize=dc_data['holding_cost'].to_dict()
    )
    
    # Demand (aggregate or single period)
    if 'period' in demand.columns:
        # Use average demand across periods
        demand_dict = demand.groupby('customer_id')['demand'].mean().to_dict()
    else:
        demand_dict = demand.set_index('customer_id')['demand'].to_dict()
    
    model.demand = Param(
        model.L,
        initialize=demand_dict
    )
    
    # Distance parameters
    sf_dist_dict = {}
    for _, row in distances['supplier_factory'].iterrows():
        sf_dist_dict[(row['supplier_id'], row['factory_id'])] = row['distance_km']
    model.dist_sf = Param(
        model.I, model.J,
        initialize=sf_dist_dict,
        default=1000.0
    )
    # Congestion factor for supplier->factory
    sf_cong_dict = {}
    for _, row in distances['supplier_factory'].iterrows():
        sf_cong_dict[(row['supplier_id'], row['factory_id'])] = row.get('congestion_factor', 1.0)
    model.sf_congestion = Param(model.I, model.J, initialize=sf_cong_dict, default=1.0)
    
    fd_dist_dict = {}
    for _, row in distances['factory_dc'].iterrows():
        fd_dist_dict[(row['factory_id'], row['dc_id'])] = row['distance_km']
    model.dist_fd = Param(
        model.J, model.K,
        initialize=fd_dist_dict,
        default=1000.0
    )
    fd_cong_dict = {}
    for _, row in distances['factory_dc'].iterrows():
        fd_cong_dict[(row['factory_id'], row['dc_id'])] = row.get('congestion_factor', 1.0)
    model.fd_congestion = Param(model.J, model.K, initialize=fd_cong_dict, default=1.0)
    
    dc_dist_dict = {}
    for _, row in distances['dc_customer'].iterrows():
        dc_dist_dict[(row['dc_id'], row['customer_id'])] = row['distance_km']
    model.dist_dc = Param(
        model.K, model.L,
        initialize=dc_dist_dict,
        default=1000.0
    )
    dc_cong_dict = {}
    for _, row in distances['dc_customer'].iterrows():
        dc_cong_dict[(row['dc_id'], row['customer_id'])] = row.get('congestion_factor', 1.0)
    model.dc_congestion = Param(model.K, model.L, initialize=dc_cong_dict, default=1.0)
    
    # Cost and emission parameters
    model.transport_cost_per_km = Param(initialize=0.1)  # $ per unit per km
    model.transport_emission_factor = Param(initialize=0.02)  # kg CO2 per unit per km
    
    # Decision Variables (flows)
    model.x = Var(model.I, model.J, domain=NonNegativeReals)  # x_ij: supplier to factory
    model.y = Var(model.J, model.K, domain=NonNegativeReals)  # y_jk: factory to DC
    model.z = Var(model.K, model.L, domain=NonNegativeReals)   # z_kl: DC to customer
    
    # Constraints
    
    # Supply capacity constraints
    def supply_limit_rule(model, i):
        return sum(model.x[i, j] for j in model.J) <= model.supply_capacity[i]
    model.supply_limit = Constraint(model.I, rule=supply_limit_rule)
    
    # Factory flow balance: inflow = outflow
    def factory_balance_rule(model, j):
        inflow = sum(model.x[i, j] for i in model.I)
        outflow = sum(model.y[j, k] for k in model.K)
        return inflow == outflow
    model.factory_balance = Constraint(model.J, rule=factory_balance_rule)
    
    # Factory capacity constraints
    def factory_capacity_rule(model, j):
        return sum(model.x[i, j] for i in model.I) <= model.factory_capacity[j]
    model.factory_capacity_constraint = Constraint(model.J, rule=factory_capacity_rule)
    
    # DC flow balance: inflow = outflow
    def dc_balance_rule(model, k):
        inflow = sum(model.y[j, k] for j in model.J)
        outflow = sum(model.z[k, l] for l in model.L)
        return inflow == outflow
    model.dc_balance = Constraint(model.K, rule=dc_balance_rule)
    
    # DC capacity constraints
    def dc_capacity_rule(model, k):
        return sum(model.y[j, k] for j in model.J) <= model.dc_capacity[k]
    model.dc_capacity_constraint = Constraint(model.K, rule=dc_capacity_rule)
    
    # Demand satisfaction
    def demand_satisfaction_rule(model, l):
        return sum(model.z[k, l] for k in model.K) >= model.demand[l]
    model.demand_satisfaction = Constraint(model.L, rule=demand_satisfaction_rule)
    
    # Objectives (defined separately for multi-objective solver)
    
    # Objective 1: Total Cost
    def cost_objective_rule(model):
        supplier_cost = sum(
            model.supplier_cost[i] * model.x[i, j]
            for i in model.I for j in model.J
        )
        production_cost = sum(
            model.production_cost[j] * sum(model.x[i, j] for i in model.I)
            for j in model.J
        )
        transport_cost_sf = sum(
            model.transport_cost_per_km * model.dist_sf[i, j] * model.sf_congestion[i, j] * model.x[i, j]
            for i in model.I for j in model.J
        )
        transport_cost_fd = sum(
            model.transport_cost_per_km * model.dist_fd[j, k] * model.fd_congestion[j, k] * model.y[j, k]
            for j in model.J for k in model.K
        )
        transport_cost_dc = sum(
            model.transport_cost_per_km * model.dist_dc[k, l] * model.dc_congestion[k, l] * model.z[k, l]
            for k in model.K for l in model.L
        )
        holding_cost = sum(
            model.holding_cost[k] * sum(model.y[j, k] for j in model.J)
            for k in model.K
        )
        return supplier_cost + production_cost + transport_cost_sf + transport_cost_fd + transport_cost_dc + holding_cost
    
    model.cost_objective = Objective(expr=cost_objective_rule(model), sense=minimize)
    
    # Objective 2: Total Emissions
    def emission_objective_rule(model):
        supplier_emissions = sum(
            model.supplier_emission_factor[i] * model.x[i, j]
            for i in model.I for j in model.J
        )
        production_emissions = sum(
            model.production_emission_factor[j] * sum(model.x[i, j] for i in model.I)
            for j in model.J
        )
        transport_emissions_sf = sum(
            model.transport_emission_factor * model.dist_sf[i, j] * model.sf_congestion[i, j] * model.x[i, j]
            for i in model.I for j in model.J
        )
        transport_emissions_fd = sum(
            model.transport_emission_factor * model.dist_fd[j, k] * model.fd_congestion[j, k] * model.y[j, k]
            for j in model.J for k in model.K
        )
        transport_emissions_dc = sum(
            model.transport_emission_factor * model.dist_dc[k, l] * model.dc_congestion[k, l] * model.z[k, l]
            for k in model.K for l in model.L
        )
        return supplier_emissions + production_emissions + transport_emissions_sf + transport_emissions_fd + transport_emissions_dc
    
    # Note: Pyomo doesn't support multiple objectives directly.
    # For NSGA-II, we'll evaluate these separately.
    model.emission_objective_expr = emission_objective_rule(model)
    
    # Objective 3: Resilience (minimize negative of minimum coverage)
    # Resilience = min over all suppliers/factories of demand coverage if that node fails
    # For now, we approximate by penalizing single-source dependencies
    def resilience_approximation_rule(model):
        # Penalize solutions where customers depend on single DCs
        # Higher penalty = lower resilience
        penalty = sum(
            (model.z[k, l] / model.demand[l]) ** 2  # Square to penalize concentration
            for k in model.K for l in model.L
            if model.demand[l] > 0
        )
        return penalty
    model.resilience_objective_expr = resilience_approximation_rule(model)
    
    return model


def evaluate_objectives(model, flows_dict: Dict) -> Dict[str, float]:
    """
    Evaluate all three objectives for given flow values.
    
    Args:
        model: Pyomo model instance
        flows_dict: Dictionary with keys 'x', 'y', 'z' containing flow arrays
    
    Returns:
        Dictionary with 'cost', 'emissions', 'resilience' values
    """
    from pyomo.environ import value
    
    # Set flow values
    for i in model.I:
        for j in model.J:
            model.x[i, j].set_value(flows_dict['x'][i, j])
    for j in model.J:
        for k in model.K:
            model.y[j, k].set_value(flows_dict['y'][j, k])
    for k in model.K:
        for l in model.L:
            model.z[k, l].set_value(flows_dict['z'][k, l])
    
    cost = value(model.cost_objective)
    emissions = value(model.emission_objective_expr)
    resilience_penalty = value(model.resilience_objective_expr)
    
    # Convert resilience penalty to resilience score (inverse, normalized)
    # Higher resilience penalty = lower resilience score
    resilience_score = 1.0 / (1.0 + resilience_penalty)
    
    return {
        'cost': cost,
        'emissions': emissions,
        'resilience': resilience_score
    }

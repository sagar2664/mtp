"""
Discrete event simulation using SimPy for evaluating resilience under disruptions.
Simulates supplier and factory failures to measure demand fulfillment.
"""
import simpy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import random


class DisruptionSimulation:
    """
    Simulates supply chain operations under random disruptions.
    Measures resilience as demand fulfillment under supplier/factory failures.
    """
    
    def __init__(
        self,
        suppliers: pd.DataFrame,
        factories: pd.DataFrame,
        dcs: pd.DataFrame,
        customers: pd.DataFrame,
        flows: Dict,
        distances: Dict = None,
        mttf: float = 365.0,  # Mean Time To Failure (days)
        mttr: float = 180.0,   # Mean Time To Recovery (days)
        seed: int = 42
    ):
        """
        Initialize disruption simulation.
        
        Args:
            suppliers: Supplier data
            factories: Factory data
            dcs: DC data
            customers: Customer data
            flows: Flow solution dictionary with keys 'x', 'y', 'z'
            mttf: Mean Time To Failure (in days)
            mttr: Mean Time To Recovery (in days)
            seed: Random seed
        """
        self.env = simpy.Environment()
        self.suppliers = suppliers.set_index('supplier_id')
        self.factories = factories.set_index('factory_id')
        self.dcs = dcs.set_index('dc_id')
        self.customers = customers.set_index('customer_id')
        self.flows = flows
        self.distances = distances or {}
        self.mttf = mttf
        self.mttr = mttr
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # State tracking
        self.supplier_status = {s: True for s in self.suppliers.index}  # True = operational
        self.factory_status = {f: True for f in self.factories.index}
        
        # Baseline demand: use total planned deliveries (sum of z flows)
        self.baseline_total_demand = 0.0
        for (_, _), z_val in self.flows.get('z', {}).items():
            self.baseline_total_demand += float(z_val)

        # Rerouting efficiency (fraction of lost volume that can be reassigned to available DCs)
        self.reroute_efficiency = 0.5

        # Precompute baseline congestion per DC from inbound (factory->DC) and outbound (DC->customer) arcs
        self.dc_congestion_baseline = {k: 1.0 for k in self.dcs.index}
        try:
            if 'factory_dc' in self.distances:
                for _, row in self.distances['factory_dc'].iterrows():
                    k = row['dc_id']
                    self.dc_congestion_baseline[k] = max(self.dc_congestion_baseline.get(k, 1.0), float(row.get('congestion_factor', 1.0)))
            if 'dc_customer' in self.distances:
                for _, row in self.distances['dc_customer'].iterrows():
                    k = row['dc_id']
                    self.dc_congestion_baseline[k] = max(self.dc_congestion_baseline.get(k, 1.0), float(row.get('congestion_factor', 1.0)))
        except Exception:
            pass

        # Statistics
        self.stats = {
            'total_demand': 0,
            'total_fulfilled': 0,
            'disruption_events': [],
            'fill_rates': []
        }
    
    def supplier_process(self, supplier_id):
        """Process modeling supplier failure and recovery cycles."""
        while True:
            # Operational period
            uptime = np.random.exponential(self.mttf)
            yield self.env.timeout(uptime)
            
            # Failure event
            fail_time = self.env.now
            self.supplier_status[supplier_id] = False
            self.stats['disruption_events'].append({
                'time': fail_time,
                'node': supplier_id,
                'type': 'supplier',
                'event': 'failure'
            })
            
            # Recovery period
            downtime = np.random.exponential(self.mttr)
            yield self.env.timeout(downtime)
            
            # Recovery event
            recovery_time = self.env.now
            self.supplier_status[supplier_id] = True
            self.stats['disruption_events'].append({
                'time': recovery_time,
                'node': supplier_id,
                'type': 'supplier',
                'event': 'recovery'
            })
    
    def factory_process(self, factory_id):
        """Process modeling factory failure and recovery cycles."""
        while True:
            # Operational period
            uptime = np.random.exponential(self.mttf)
            yield self.env.timeout(uptime)
            
            # Failure event
            fail_time = self.env.now
            self.factory_status[factory_id] = False
            self.stats['disruption_events'].append({
                'time': fail_time,
                'node': factory_id,
                'type': 'factory',
                'event': 'failure'
            })
            
            # Recovery period
            downtime = np.random.exponential(self.mttr)
            yield self.env.timeout(downtime)
            
            # Recovery event
            recovery_time = self.env.now
            self.factory_status[factory_id] = True
            self.stats['disruption_events'].append({
                'time': recovery_time,
                'node': factory_id,
                'type': 'factory',
                'event': 'recovery'
            })
    
    def demand_fulfillment_monitor(self):
        """Monitor demand fulfillment continuously."""
        check_interval = 30.0  # Check every 30 days
        while True:
            yield self.env.timeout(check_interval)
            
            # Baseline total demand per window (proxy): sum of planned z flows
            total_demand = self.baseline_total_demand if self.baseline_total_demand > 0 else 1.0

            # Compute DC availability based on upstream x and y flows and node status
            x = self.flows.get('x', {})
            y = self.flows.get('y', {})
            z = self.flows.get('z', {})
            
            # For each factory j, compute available inflow fraction from suppliers
            factory_in_avail = {}
            for j in self.factories.index:
                total_in = sum(float(x.get((i, j), 0.0)) for i in self.suppliers.index)
                if total_in <= 0:
                    frac = 0.0
                else:
                    avail_in = sum(
                        float(x.get((i, j), 0.0)) if self.supplier_status.get(i, True) else 0.0
                        for i in self.suppliers.index
                    )
                    frac = avail_in / total_in
                # Factory must also be up
                factory_in_avail[j] = frac * (1.0 if self.factory_status.get(j, True) else 0.0)
            
            # For each DC k, compute available inflow via y_jk weighted by factory availability
            dc_in_avail = {}
            for k in self.dcs.index:
                total_in = sum(float(y.get((j, k), 0.0)) for j in self.factories.index)
                if total_in <= 0:
                    dc_in_avail[k] = 0.0
                else:
                    avail_in = sum(float(y.get((j, k), 0.0)) * factory_in_avail.get(j, 0.0) for j in self.factories.index)
                    dc_in_avail[k] = max(0.0, min(1.0, avail_in / total_in))

            # Apply congestion impact: sample a congestion factor around baseline and convert to availability factor
            for k in list(dc_in_avail.keys()):
                base_cong = float(self.dc_congestion_baseline.get(k, 1.0))
                # Random daily fluctuation
                fluct = np.random.normal(loc=1.0, scale=0.1)
                cong = max(1.0, min(2.0, base_cong * fluct))
                cong_avail = 1.0 / cong  # Higher congestion -> lower availability
                dc_in_avail[k] *= cong_avail
            
            # Available deliveries with simple rerouting per customer
            total_available = 0.0
            customers_list = list(self.customers.index)
            for l in customers_list:
                planned_total = sum(float(z.get((k, l), 0.0)) for k in self.dcs.index)
                if planned_total <= 0:
                    continue
                avail_contrib = sum(float(z.get((k, l), 0.0)) * dc_in_avail.get(k, 0.0) for k in self.dcs.index)
                lost = max(0.0, planned_total - avail_contrib)
                # If at least one DC available, recover part of lost via rerouting
                any_avail = any(dc_in_avail.get(k, 0.0) > 0.0 for k in self.dcs.index)
                recover = self.reroute_efficiency * lost if any_avail else 0.0
                total_available += min(planned_total, avail_contrib + recover)
            
            fill_rate = min(1.0, total_available / (total_demand + 1e-10))
            
            self.stats['fill_rates'].append({
                'time': self.env.now,
                'fill_rate': fill_rate
            })
            self.stats['total_fulfilled'] += fill_rate * total_demand
            self.stats['total_demand'] += total_demand
    
    def run(self, max_time: float = 1095.0):  # 3 years
        """
        Run simulation.
        
        Args:
            max_time: Maximum simulation time (days)
        """
        # Start supplier processes
        for supplier_id in self.suppliers.index:
            self.env.process(self.supplier_process(supplier_id))
        
        # Start factory processes
        for factory_id in self.factories.index:
            self.env.process(self.factory_process(factory_id))
        
        # Start demand monitoring
        self.env.process(self.demand_fulfillment_monitor())
        
        # Run simulation
        self.env.run(until=max_time)
    
    def get_resilience_metrics(self) -> Dict:
        """
        Calculate resilience metrics from simulation.
        
        Returns:
            Dictionary with resilience metrics
        """
        if len(self.stats['fill_rates']) == 0:
            return {
                'average_fill_rate': 0.0,
                'min_fill_rate': 0.0,
                'n_disruptions': 0,
                'worst_case_coverage': 0.0
            }
        
        fill_rates = [fr['fill_rate'] for fr in self.stats['fill_rates']]
        
        # Worst-case coverage (minimum fill rate) = resilience score
        worst_case_coverage = min(fill_rates) if fill_rates else 0.0
        
        return {
            'average_fill_rate': np.mean(fill_rates),
            'min_fill_rate': min(fill_rates) if fill_rates else 0.0,
            'n_disruptions': len([e for e in self.stats['disruption_events'] if e['event'] == 'failure']),
            'worst_case_coverage': worst_case_coverage,  # This is the resilience metric
            'total_fulfilled': self.stats['total_fulfilled'],
            'total_demand': self.stats['total_demand']
        }


def simulate_single_node_failure(
    suppliers: pd.DataFrame,
    factories: pd.DataFrame,
    dcs: pd.DataFrame,
    customers: pd.DataFrame,
    flows: Dict,
    failed_node: str,
    node_type: str = 'supplier'
) -> Dict:
    """
    Simulate a single node failure and measure demand coverage.
    Used for resilience calculation: R_i = fraction of demand met if node i fails.
    
    Args:
        suppliers: Supplier data
        factories: Factory data
        dcs: DC data
        customers: Customer data
        flows: Flow solution
        failed_node: ID of failed node
        node_type: 'supplier' or 'factory'
    
    Returns:
        Dictionary with coverage metrics
    """
    # Baseline total demand: sum of planned z flows
    total_demand = sum(float(v) for v in flows.get('z', {}).values())
    if total_demand <= 0:
        total_demand = 1.0
    
    # Remove flows from failed node
    available_flows = {}
    
    if node_type == 'supplier':
        # Remove all x_ij flows from this supplier
        x_flows = flows.get('x', {})
        available_flows['x'] = {
            (i, j): val for (i, j), val in x_flows.items()
            if i != failed_node
        }
        available_flows['y'] = flows.get('y', {})
        available_flows['z'] = flows.get('z', {})
    else:  # factory
        # Remove all x_ij and y_jk flows from this factory
        x_flows = flows.get('x', {})
        y_flows = flows.get('y', {})
        available_flows['x'] = {
            (i, j): val for (i, j), val in x_flows.items()
            if j != failed_node
        }
        available_flows['y'] = {
            (j, k): val for (j, k), val in y_flows.items()
            if j != failed_node
        }
        available_flows['z'] = flows.get('z', {})
    
    # Calculate total available flow to customers (simplified via z flows)
    total_available = 0
    for (k, l), z_val in available_flows['z'].items():
        total_available += float(z_val)
    
    # If some flows are missing, we might not satisfy all demand
    coverage = min(1.0, total_available / (total_demand + 1e-10))
    
    return {
        'failed_node': failed_node,
        'node_type': node_type,
        'demand_coverage': coverage,
        'total_demand': total_demand,
        'available_flow': total_available
    }


def calculate_resilience_score(
    suppliers: pd.DataFrame,
    factories: pd.DataFrame,
    dcs: pd.DataFrame,
    customers: pd.DataFrame,
    flows: Dict
) -> float:
    """
    Calculate resilience score as minimum demand coverage under single-node failures.
    R = min_i R_i, where R_i is coverage when node i fails.
    
    Args:
        suppliers: Supplier data
        factories: Factory data
        dcs: DC data
        customers: Customer data
        flows: Flow solution
    
    Returns:
        Resilience score (0-1, higher is better)
    """
    coverages = []
    
    # Test each supplier failure
    for supplier_id in suppliers['supplier_id']:
        result = simulate_single_node_failure(
            suppliers, factories, dcs, customers, flows,
            failed_node=supplier_id,
            node_type='supplier'
        )
        coverages.append(result['demand_coverage'])
    
    # Test each factory failure
    for factory_id in factories['factory_id']:
        result = simulate_single_node_failure(
            suppliers, factories, dcs, customers, flows,
            failed_node=factory_id,
            node_type='factory'
        )
        coverages.append(result['demand_coverage'])
    
    # Resilience = minimum coverage (worst-case scenario)
    resilience = min(coverages) if coverages else 0.0
    
    return resilience


def run_disruption_simulation(
    suppliers: pd.DataFrame,
    factories: pd.DataFrame,
    dcs: pd.DataFrame,
    customers: pd.DataFrame,
    flows: Dict,
    distances: Dict = None,
    n_runs: int = 100,
    mttf: float = 365.0,
    mttr: float = 180.0,
    max_time: float = 1095.0,
    seed: int = 42
) -> Dict:
    """
    Run multiple simulation runs to evaluate resilience.
    
    Args:
        suppliers: Supplier data
        factories: Factory data
        dcs: DC data
        customers: Customer data
        flows: Flow solution
        n_runs: Number of simulation runs
        mttf: Mean Time To Failure
        mttr: Mean Time To Recovery
        max_time: Maximum simulation time per run
        seed: Random seed
    
    Returns:
        Dictionary with aggregated resilience metrics
    """
    all_results = []
    
    for run in range(n_runs):
        sim = DisruptionSimulation(
            suppliers, factories, dcs, customers, flows, distances,
            mttf=mttf, mttr=mttr, seed=seed + run
        )
        sim.run(max_time=max_time)
        metrics = sim.get_resilience_metrics()
        all_results.append(metrics)
    
    # Aggregate results
    avg_fill_rate = np.mean([r['average_fill_rate'] for r in all_results])
    min_fill_rate = np.mean([r['min_fill_rate'] for r in all_results])
    avg_disruptions = np.mean([r['n_disruptions'] for r in all_results])
    
    return {
        'average_fill_rate': avg_fill_rate,
        'min_fill_rate': min_fill_rate,
        'average_n_disruptions': avg_disruptions,
        'n_runs': n_runs,
        'resilience_score': min_fill_rate  # Worst-case coverage across runs
    }

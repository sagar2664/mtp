"""
Multi-objective optimization using NSGA-II (pymoo).
Solves the 3-objective supply chain problem: minimize cost, minimize emissions, maximize resilience.
"""
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pyomo.environ import value


class SupplyChainProblem(Problem):
    """
    Multi-objective problem definition for pymoo NSGA-II.
    
    Decision variables: flows x_ij, y_jk, z_kl
    Objectives: minimize cost, minimize emissions, maximize resilience
    """
    
    def __init__(
        self,
        model,
        n_vars: int,
        n_obj: int = 3,
        n_constr: int = 0,
        xl: float = 0.0,
        xu: float = 1000.0
    ):
        """
        Initialize the problem.
        
        Args:
            model: Pyomo model instance
            n_vars: Number of decision variables (total flows)
            n_obj: Number of objectives (3: cost, emissions, resilience)
            n_constr: Number of constraints
            xl: Lower bound for variables
            xu: Upper bound for variables
        """
        super().__init__(
            n_var=n_vars,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu
        )
        self.model = model
        self.n_suppliers = len(model.I)
        self.n_factories = len(model.J)
        self.n_dcs = len(model.K)
        self.n_customers = len(model.L)
        
        # Calculate variable indices
        self.n_x = self.n_suppliers * self.n_factories
        self.n_y = self.n_factories * self.n_dcs
        self.n_z = self.n_dcs * self.n_customers
        
    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives for each solution in population X.
        
        Args:
            X: Population matrix (n_pop x n_vars)
            out: Output dictionary with 'F' for objectives
        """
        n_pop = X.shape[0]
        F = np.zeros((n_pop, 3))  # 3 objectives
        
        for i in range(n_pop):
            # Decode decision vector to flows
            flows = self._decode_solution(X[i])
            
            # Evaluate objectives
            objectives = self._evaluate_objectives(flows)
            
            F[i, 0] = objectives['cost']
            F[i, 1] = objectives['emissions']
            F[i, 2] = -objectives['resilience']  # Negate for minimization
            
        out["F"] = F
    
    def _decode_solution(self, x: np.ndarray) -> Dict:
        """
        Decode decision vector x into flow dictionaries.
        
        Args:
            x: Decision vector
        
        Returns:
            Dictionary with keys 'x', 'y', 'z' containing flow matrices
        """
        flows = {}
        
        # Reshape x into flows
        x_idx = 0
        
        # x_ij: supplier to factory flows
        x_flows = {}
        supplier_list = list(self.model.I)
        factory_list = list(self.model.J)
        for i_idx, i in enumerate(supplier_list):
            for j_idx, j in enumerate(factory_list):
                if x_idx < len(x):
                    x_flows[(i, j)] = max(0, x[x_idx])
                    x_idx += 1
        
        flows['x'] = x_flows
        
        # y_jk: factory to DC flows
        y_flows = {}
        dc_list = list(self.model.K)
        for j_idx, j in enumerate(factory_list):
            for k_idx, k in enumerate(dc_list):
                if x_idx < len(x):
                    y_flows[(j, k)] = max(0, x[x_idx])
                    x_idx += 1
        
        flows['y'] = y_flows
        
        # z_kl: DC to customer flows
        z_flows = {}
        customer_list = list(self.model.L)
        for k_idx, k in enumerate(dc_list):
            for l_idx, l in enumerate(customer_list):
                if x_idx < len(x):
                    z_flows[(k, l)] = max(0, x[x_idx])
                    x_idx += 1
        
        flows['z'] = z_flows
        
        return flows
    
    def _evaluate_objectives(self, flows: Dict) -> Dict[str, float]:
        """
        Evaluate all three objectives for given flows.
        
        Args:
            flows: Dictionary with 'x', 'y', 'z' flow matrices
        
        Returns:
            Dictionary with 'cost', 'emissions', 'resilience'
        """
        cost = 0.0
        emissions = 0.0
        penalty = 0.0
        
        # Cost calculation
        for (i, j), flow_val in flows['x'].items():
            if flow_val > 1e-6:
                cost += self.model.supplier_cost[i] * flow_val
                # Congestion inflates effective distance/time
                cost += self.model.transport_cost_per_km * self.model.dist_sf[i, j] * self.model.sf_congestion[i, j] * flow_val
        
        for (j, k), flow_val in flows['y'].items():
            if flow_val > 1e-6:
                # Production cost at factory
                total_into_j = sum(flows['x'].get((i, j), 0) for i in self.model.I)
                if total_into_j > 0:
                    cost += self.model.production_cost[j] * (flow_val / total_into_j) * sum(flows['x'].get((i, j), 0) for i in self.model.I)
                cost += self.model.transport_cost_per_km * self.model.dist_fd[j, k] * self.model.fd_congestion[j, k] * flow_val
        
        for (k, l), flow_val in flows['z'].items():
            if flow_val > 1e-6:
                cost += self.model.transport_cost_per_km * self.model.dist_dc[k, l] * self.model.dc_congestion[k, l] * flow_val
                cost += self.model.holding_cost[k] * flow_val
        
        # Emissions calculation
        for (i, j), flow_val in flows['x'].items():
            if flow_val > 1e-6:
                emissions += self.model.supplier_emission_factor[i] * flow_val
                emissions += self.model.transport_emission_factor * self.model.dist_sf[i, j] * self.model.sf_congestion[i, j] * flow_val
        
        for (j, k), flow_val in flows['y'].items():
            if flow_val > 1e-6:
                total_into_j = sum(flows['x'].get((i, j), 0) for i in self.model.I)
                if total_into_j > 0:
                    emissions += self.model.production_emission_factor[j] * flow_val
                emissions += self.model.transport_emission_factor * self.model.dist_fd[j, k] * self.model.fd_congestion[j, k] * flow_val
        
        for (k, l), flow_val in flows['z'].items():
            if flow_val > 1e-6:
                emissions += self.model.transport_emission_factor * self.model.dist_dc[k, l] * self.model.dc_congestion[k, l] * flow_val
        
        # Feasibility penalties
        # Supplier capacity
        for i in self.model.I:
            cap = float(self.model.supply_capacity[i])
            used = sum(float(flows['x'].get((i, j), 0.0)) for j in self.model.J)
            if used > cap:
                penalty += (used - cap)
        # Factory capacity and balance
        for j in self.model.J:
            cap = float(self.model.factory_capacity[j])
            inflow = sum(float(flows['x'].get((i, j), 0.0)) for i in self.model.I)
            outflow = sum(float(flows['y'].get((j, k), 0.0)) for k in self.model.K)
            if inflow > cap:
                penalty += (inflow - cap)
            # Balance violation
            penalty += abs(inflow - outflow)
        # DC capacity and balance
        for k in self.model.K:
            cap = float(self.model.dc_capacity[k])
            inflow = sum(float(flows['y'].get((j, k), 0.0)) for j in self.model.J)
            outflow = sum(float(flows['z'].get((k, l), 0.0)) for l in self.model.L)
            if inflow > cap:
                penalty += (inflow - cap)
            penalty += abs(inflow - outflow)
        # Demand satisfaction
        for l in self.model.L:
            demand_l = float(self.model.demand[l])
            supplied_l = sum(float(flows['z'].get((k, l), 0.0)) for k in self.model.K)
            if supplied_l < demand_l:
                penalty += (demand_l - supplied_l)

        # Scale penalties and add to objectives (soft constraint handling)
        PEN_COST = 1e3
        PEN_EM = 10.0
        cost += PEN_COST * penalty
        emissions += PEN_EM * penalty

        # Resilience calculation (approximate: minimize single-source dependencies)
        resilience_penalty = 0.0
        for l in self.model.L:
            total_to_l = sum(flows['z'].get((k, l), 0) for k in self.model.K)
            if total_to_l > 0 and self.model.demand[l] > 0:
                # Penalize concentration (prefer multiple DCs serving each customer)
                for k in self.model.K:
                    flow_share = flows['z'].get((k, l), 0) / total_to_l
                    resilience_penalty += flow_share ** 2  # Square to penalize concentration
        
        # Convert penalty to resilience score (higher penalty = lower resilience)
        resilience_score = 1.0 / (1.0 + resilience_penalty)
        
        return {
            'cost': cost,
            'emissions': emissions,
            'resilience': resilience_score
        }


def solve_with_nsga2(
    model,
    n_pop: int = 100,
    n_gen: int = 200,
    seed: int = 42
) -> Dict:
    """
    Solve multi-objective optimization problem using NSGA-II.
    
    Args:
        model: Pyomo model instance
        n_pop: Population size
        n_gen: Number of generations
        seed: Random seed
    
    Returns:
        Dictionary with Pareto front solutions and objectives
    """
    # Calculate number of decision variables
    n_suppliers = len(model.I)
    n_factories = len(model.J)
    n_dcs = len(model.K)
    n_customers = len(model.L)
    
    n_vars = (
        n_suppliers * n_factories +  # x_ij
        n_factories * n_dcs +        # y_jk
        n_dcs * n_customers          # z_kl
    )
    
    # Estimate upper bound (total demand * safety factor)
    total_demand = sum(value(model.demand[l]) for l in model.L)
    xu = total_demand * 2.0  # Upper bound
    
    # Create problem instance
    problem = SupplyChainProblem(
        model=model,
        n_vars=n_vars,
        n_obj=3,
        xl=0.0,
        xu=xu
    )
    
    # Initialize NSGA-II algorithm
    algorithm = NSGA2(
        pop_size=n_pop,
        eliminate_duplicates=True
    )
    
    # Solve
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        seed=seed,
        verbose=True
    )
    
    # Extract Pareto front
    pareto_solutions = []
    for i in range(len(res.X)):
        # Decode solution
        flows = problem._decode_solution(res.X[i])
        objectives = problem._evaluate_objectives(flows)
        
        pareto_solutions.append({
            'solution_id': i,
            'objectives': {
                'cost': objectives['cost'],
                'emissions': objectives['emissions'],
                'resilience': objectives['resilience']
            },
            'flows': flows
        })
    
    return {
        'pareto_front': pareto_solutions,
        'X': res.X,  # Decision variables
        'F': res.F,  # Objective values
        'algorithm': res.algorithm,
        'exec_time': res.exec_time
    }


def extract_representative_solutions(pareto_result: Dict, n_solutions: int = 3) -> List[Dict]:
    """
    Extract representative solutions from Pareto front.
    Typically: min cost, min emissions, max resilience (balanced).
    
    Args:
        pareto_result: Result from solve_with_nsga2
        n_solutions: Number of representative solutions to extract
    
    Returns:
        List of representative solution dictionaries
    """
    pareto_solutions = pareto_result['pareto_front']
    
    if len(pareto_solutions) == 0:
        return []
    
    # Rank lists
    cost_order = sorted(range(len(pareto_solutions)), key=lambda i: pareto_solutions[i]['objectives']['cost'])
    em_order = sorted(range(len(pareto_solutions)), key=lambda i: pareto_solutions[i]['objectives']['emissions'])
    res_order = sorted(range(len(pareto_solutions)), key=lambda i: -pareto_solutions[i]['objectives']['resilience'])

    # Pick distinct indices for Min Cost, Min Emissions, Max Resilience
    picked = []
    def pick_first_distinct(order_list):
        for idx in order_list:
            if idx not in picked:
                picked.append(idx)
                return idx
        return order_list[0]

    min_cost_idx = pick_first_distinct(cost_order)
    min_emissions_idx = pick_first_distinct(em_order)
    max_resilience_idx = pick_first_distinct(res_order)

    # Find balanced solutions in two ways:
    # 1) Knee: max distance from ideal-nadir line (high-curvature point)
    # 2) Utopia-distance: closest to (min cost, min emissions, max resilience)
    cost_vals = [s['objectives']['cost'] for s in pareto_solutions]
    emissions_vals = [s['objectives']['emissions'] for s in pareto_solutions]
    resilience_vals = [s['objectives']['resilience'] for s in pareto_solutions]
    
    cost_norm = (np.array(cost_vals) - min(cost_vals)) / (max(cost_vals) - min(cost_vals) + 1e-10)
    emissions_norm = (np.array(emissions_vals) - min(emissions_vals)) / (max(emissions_vals) - min(emissions_vals) + 1e-10)
    resilience_norm = 1 - (np.array(resilience_vals) - min(resilience_vals)) / (max(resilience_vals) - min(resilience_vals) + 1e-10)
    
    # Ideal (0,0,0) to Nadir (1,1,1) line
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([1.0, 1.0, 1.0])
    v = p1 - p0
    v_norm = v / (np.linalg.norm(v) + 1e-12)
    distances = []
    for a, b, c in zip(cost_norm, emissions_norm, resilience_norm):
        p = np.array([a, b, c])
        proj_len = np.dot(p - p0, v_norm)
        proj = p0 + proj_len * v_norm
        distances.append(np.linalg.norm(p - proj))
    order_balanced_knee = list(np.argsort(-np.array(distances)))  # descending distance = knee first
    balanced_knee_idx = pick_first_distinct(order_balanced_knee)

    # Utopia-distance (0,0,0) in normalized space
    utopia_dist = np.sqrt(cost_norm**2 + emissions_norm**2 + resilience_norm**2)
    order_utopia = list(np.argsort(utopia_dist))
    balanced_utopia_idx = pick_first_distinct(order_utopia)
    
    representative = [
        pareto_solutions[min_cost_idx].copy(),
        pareto_solutions[min_emissions_idx].copy(),
        pareto_solutions[balanced_utopia_idx].copy()
    ]
    
    # Add labels
    representative[0]['label'] = 'Min Cost'
    representative[1]['label'] = 'Min Emissions'
    representative[2]['label'] = 'Balanced (Utopia)'

    # Also include Knee as a fourth representative when requested by callers
    # If n_solutions >= 4, append the knee point
    if n_solutions >= 4:
        knee_rep = pareto_solutions[balanced_knee_idx].copy()
        # Avoid duplicate label if utopia picked the same index
        if knee_rep not in representative:
            knee_rep['label'] = 'Balanced (Knee)'
            representative.append(knee_rep)
    
    return representative

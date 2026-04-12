"""
Multi-objective optimization using NSGA-II (pymoo).
Solves the 3-objective supply chain problem: minimize cost, minimize emissions, maximize resilience.

Objective function sources:
  - Cost: Melo et al. (2009) + Jabbarzadeh et al. (2018) unmet demand penalty
  - Emissions: Pishvaee & Razmi (2012) LCA + Pishvaee, Torabi & Razmi (2012) congestion
  - Resilience: Hasani & Khosrojerdi (2016) HHI + Snyder & Daskin (2005) node failure
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
        self.n_modes = len(model.M)
        
        # Calculate variable indices
        self.n_x = self.n_suppliers * self.n_factories * self.n_modes
        self.n_y = self.n_factories * self.n_dcs * self.n_modes
        self.n_z = self.n_dcs * self.n_customers * self.n_modes
        
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
        mode_list = list(self.model.M)
        
        # x_ijm: supplier to factory flows
        x_flows = {}
        supplier_list = list(self.model.I)
        factory_list = list(self.model.J)
        for i_idx, i in enumerate(supplier_list):
            for j_idx, j in enumerate(factory_list):
                for m_idx, m in enumerate(mode_list):
                    if x_idx < len(x):
                        x_flows[(i, j, m)] = max(0, x[x_idx])
                        x_idx += 1
        flows['x'] = x_flows
        
        # y_jkm: factory to DC flows
        y_flows = {}
        dc_list = list(self.model.K)
        for j_idx, j in enumerate(factory_list):
            for k_idx, k in enumerate(dc_list):
                for m_idx, m in enumerate(mode_list):
                    if x_idx < len(x):
                        y_flows[(j, k, m)] = max(0, x[x_idx])
                        x_idx += 1
        flows['y'] = y_flows
        
        # z_klm: DC to customer flows
        z_flows = {}
        customer_list = list(self.model.L)
        for k_idx, k in enumerate(dc_list):
            for l_idx, l in enumerate(customer_list):
                for m_idx, m in enumerate(mode_list):
                    if x_idx < len(x):
                        z_flows[(k, l, m)] = max(0, x[x_idx])
                        x_idx += 1
        flows['z'] = z_flows
        
        return flows
    
    def _evaluate_objectives(self, flows: Dict) -> Dict[str, float]:
        """
        Evaluate all three objectives for given flows.
        
        Paper-sourced formulations:
          f1 (cost):       Melo et al. (2009) + Jabbarzadeh et al. (2018) unmet demand penalty
          f2 (emissions):  Pishvaee & Razmi (2012) LCA + Pishvaee, Torabi & Razmi (2012) congestion
          f3 (resilience): Hasani & Khosrojerdi (2016) HHI + Snyder & Daskin (2005) node failure
        
        Args:
            flows: Dictionary with 'x', 'y', 'z' flow matrices
        
        Returns:
            Dictionary with 'cost', 'emissions', 'resilience'
        """
        cost = 0.0
        emissions = 0.0
        penalty = 0.0
        
        # ================================================================
        # COST — Melo et al. (2009), EJOR 196(2), 401-412
        # ================================================================
        # Procurement + transport (Supplier → Factory)
        for (i, j, m), flow_val in flows['x'].items():
            if flow_val > 1e-6:
                cost += self.model.supplier_cost[i] * flow_val
                cost += self.model.unit_t_cost_sf[i, j, m] * flow_val
        
        # Transport (Factory → DC)
        for (j, k, m), flow_val in flows['y'].items():
            if flow_val > 1e-6:
                cost += self.model.unit_t_cost_fd[j, k, m] * flow_val
        
        # Transport (DC → Customer) + Holding
        for (k, l, m), flow_val in flows['z'].items():
            if flow_val > 1e-6:
                cost += self.model.unit_t_cost_dc[k, l, m] * flow_val
                cost += self.model.holding_cost[k] * flow_val
                
        # Production cost
        for j in self.model.J:
            total_into_j = sum(flows['x'].get((i, j, m), 0) for i in self.model.I for m in self.model.M)
            cost += self.model.production_cost[j] * total_into_j
        
        # Unmet demand penalty — Jabbarzadeh et al. (2018), IJPR 56(17), 5945-5968
        shortage_penalty = float(self.model.shortage_penalty.value) if hasattr(self.model, 'shortage_penalty') else 500.0
        for l in self.model.L:
            demand_l = float(self.model.demand[l])
            supplied_l = sum(float(flows['z'].get((k, l, m), 0.0)) for k in self.model.K for m in self.model.M)
            if supplied_l < demand_l:
                cost += shortage_penalty * (demand_l - supplied_l)
        
        # ================================================================
        # EMISSIONS — Pishvaee & Razmi (2012), AMM 36(8), 3433-3446
        #           + Pishvaee, Torabi & Razmi (2012), C&IE 62(2), 624-632
        # ================================================================
        # Supplier emissions
        for (i, j, m), flow_val in flows['x'].items():
            if flow_val > 1e-6:
                emissions += self.model.supplier_emission_factor[i] * flow_val
                emissions += self.model.unit_e_cost_sf[i, j, m] * flow_val
        
        # Transport emissions (Factory → DC)
        for (j, k, m), flow_val in flows['y'].items():
            if flow_val > 1e-6:
                emissions += self.model.unit_e_cost_fd[j, k, m] * flow_val
        
        # Transport emissions (DC → Customer)
        for (k, l, m), flow_val in flows['z'].items():
            if flow_val > 1e-6:
                emissions += self.model.unit_e_cost_dc[k, l, m] * flow_val
                
        # Production emissions
        for j in self.model.J:
            total_into_j = sum(flows['x'].get((i, j, m), 0) for i in self.model.I for m in self.model.M)
            emissions += self.model.production_emission_factor[j] * total_into_j
        
        # DC warehousing emissions — Pishvaee & Razmi (2012) LCA model
        for k in self.model.K:
            total_into_k = sum(float(flows['y'].get((j, k, m), 0.0)) for j in self.model.J for m in self.model.M)
            wh_ef = float(self.model.dc_warehousing_emission[k]) if hasattr(self.model, 'dc_warehousing_emission') else 0.01
            emissions += wh_ef * total_into_k
        
        # ================================================================
        # FEASIBILITY PENALTIES (soft constraints)
        # ================================================================
        # Supplier capacity
        for i in self.model.I:
            cap = float(self.model.supply_capacity[i])
            used = sum(float(flows['x'].get((i, j, m), 0.0)) for j in self.model.J for m in self.model.M)
            if used > cap:
                penalty += (used - cap)
        # Factory capacity and balance
        for j in self.model.J:
            cap = float(self.model.factory_capacity[j])
            inflow = sum(float(flows['x'].get((i, j, m), 0.0)) for i in self.model.I for m in self.model.M)
            outflow = sum(float(flows['y'].get((j, k, m), 0.0)) for k in self.model.K for m in self.model.M)
            if inflow > cap:
                penalty += (inflow - cap)
            penalty += abs(inflow - outflow)
        # DC capacity and balance
        for k in self.model.K:
            cap = float(self.model.dc_capacity[k])
            inflow = sum(float(flows['y'].get((j, k, m), 0.0)) for j in self.model.J for m in self.model.M)
            outflow = sum(float(flows['z'].get((k, l, m), 0.0)) for l in self.model.L for m in self.model.M)
            if inflow > cap:
                penalty += (inflow - cap)
            penalty += abs(inflow - outflow)
        # Demand satisfaction
        for l in self.model.L:
            demand_l = float(self.model.demand[l])
            supplied_l = sum(float(flows['z'].get((k, l, m), 0.0)) for k in self.model.K for m in self.model.M)
            if supplied_l < demand_l:
                penalty += (demand_l - supplied_l)

        # Scale penalties and add to objectives
        PEN_COST = 1e3
        PEN_EM = 10.0
        cost += PEN_COST * penalty
        emissions += PEN_EM * penalty

        # ================================================================
        # RESILIENCE — Composite metric
        # Component A: HHI Diversification — Hasani & Khosrojerdi (2016), TRE 87, 20-52
        # Component B: Node Failure Coverage — Snyder & Daskin (2005), Trans. Sci. 39(3)
        # ================================================================
        
        # Component A: HHI-based diversification index
        # HHI = Σ (share_i)^2; ranges from 1/n (perfect diversification) to 1.0 (monopoly)
        # We compute HHI for DC-to-Customer flows (downstream diversification)
        hhi_delivery = 0.0
        n_customers_with_demand = 0
        for l in self.model.L:
            total_to_l = sum(flows['z'].get((k, l, m), 0) for k in self.model.K for m in self.model.M)
            if total_to_l > 0 and self.model.demand[l] > 0:
                n_customers_with_demand += 1
                for k in self.model.K:
                    flow_share = sum(flows['z'].get((k, l, m), 0) for m in self.model.M) / total_to_l
                    hhi_delivery += flow_share ** 2
        
        # Normalize HHI: average across customers
        avg_hhi_delivery = hhi_delivery / max(n_customers_with_demand, 1)
        
        # HHI for Supplier-to-Factory flows (upstream diversification)
        hhi_supply = 0.0
        n_factories_with_flow = 0
        for j in self.model.J:
            total_to_j = sum(flows['x'].get((i, j, m), 0) for i in self.model.I for m in self.model.M)
            if total_to_j > 0:
                n_factories_with_flow += 1
                for i in self.model.I:
                    flow_share = sum(flows['x'].get((i, j, m), 0) for m in self.model.M) / total_to_j
                    hhi_supply += flow_share ** 2
        
        avg_hhi_supply = hhi_supply / max(n_factories_with_flow, 1)
        
        # Combined HHI: weighted average of upstream and downstream diversification
        avg_hhi = 0.5 * avg_hhi_supply + 0.5 * avg_hhi_delivery
        diversification_score = 1.0 - avg_hhi  # Higher = more diversified
        
        # Component B: Expected demand coverage under single-node failures
        # Snyder & Daskin (2005) expected failure cost approach
        total_demand = sum(float(self.model.demand[l]) for l in self.model.L)
        
        if total_demand > 0:
            coverage_ratios = []
            
            # Scenario: each supplier fails
            for i_fail in self.model.I:
                # Flow lost from this supplier
                flow_lost = sum(flows['x'].get((i_fail, j, m), 0) for j in self.model.J for m in self.model.M)
                total_supply = sum(flows['x'].get((i, j, m), 0) for i in self.model.I for j in self.model.J for m in self.model.M)
                coverage = 1.0 - (flow_lost / max(total_supply, 1e-6))
                coverage_ratios.append(coverage)
            
            # Scenario: each factory fails
            for j_fail in self.model.J:
                flow_lost = sum(flows['y'].get((j_fail, k, m), 0) for k in self.model.K for m in self.model.M)
                total_production = sum(flows['y'].get((j, k, m), 0) for j in self.model.J for k in self.model.K for m in self.model.M)
                coverage = 1.0 - (flow_lost / max(total_production, 1e-6))
                coverage_ratios.append(coverage)
            
            # Scenario: each DC fails
            for k_fail in self.model.K:
                flow_lost = sum(flows['z'].get((k_fail, l, m), 0) for l in self.model.L for m in self.model.M)
                total_delivery = sum(flows['z'].get((k, l, m), 0) for k in self.model.K for l in self.model.L for m in self.model.M)
                coverage = 1.0 - (flow_lost / max(total_delivery, 1e-6))
                coverage_ratios.append(coverage)
            
            min_coverage = min(coverage_ratios) if coverage_ratios else 0.0
        else:
            min_coverage = 0.0
        
        # Composite resilience score — weighted combination
        # w1=0.4 (diversification, P6), w2=0.6 (worst-case node failure, P4)
        w1, w2 = 0.4, 0.6
        resilience_score = w1 * diversification_score + w2 * min_coverage
        
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
    n_modes = len(model.M)
    
    n_vars = (
        n_suppliers * n_factories * n_modes +  # x_ijm
        n_factories * n_dcs * n_modes +        # y_jkm
        n_dcs * n_customers * n_modes          # z_klm
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

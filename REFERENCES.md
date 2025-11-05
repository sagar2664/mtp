# Required References for Thesis

## Core Algorithm Papers

### 1. NSGA-II (Multi-Objective Optimization)
**Paper:** Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.

**What we reference:**
- NSGA-II algorithm implementation for solving 3-objective optimization (cost, emissions, resilience)
- Non-dominated sorting and crowding distance for maintaining Pareto front diversity
- Fast convergence and elitism properties
- Used in `src/optimizer.py` - `solve_with_nsga2()` function

---

### 2. Random Forest (Demand Forecasting)
**Paper:** Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.

**What we reference:**
- Random Forest regression for demand forecasting
- Ensemble method with multiple decision trees
- Feature importance and robust predictions
- Used in `src/forecasting.py` - `forecast_demand_random_forest()` function

---

## Supply Chain Network Design Papers

### 3. Multi-Objective Supply Chain Network Design
**Paper:** (Example - find specific MDPI paper from your literature review)
**Title:** "Multi-objective optimization of green supply chain network design considering resilience"

**What we reference:**
- Multi-echelon supply chain network design (suppliers → factories → DCs → customers)
- Conflicting objectives: cost minimization, emission reduction, resilience maximization
- Flow-based optimization with capacity constraints
- Used in `src/pyomo_model.py` - `create_multi_objective_model()` function

---

### 4. Green Supply Chain Design
**Paper:** (Example - find MDPI Sustainability journal paper)
**Title:** "Carbon footprint optimization in multi-echelon supply chains"

**What we reference:**
- Integration of carbon emissions in supply chain optimization
- Transport emissions calculation (kg CO2 per unit-km)
- Production and supplier emission factors
- Trade-off between cost and environmental impact
- Used in cost and emissions objective functions in `src/pyomo_model.py`

---

### 5. Resilient Supply Chain Network Design
**Paper:** (Example - find MDPI paper on resilience)
**Title:** "Resilience metrics and disruption modeling in supply chain networks"

**What we reference:**
- Resilience as worst-case demand coverage under disruptions
- Single-node failure analysis
- Diversification strategies for improved resilience
- Used in `src/sim.py` - `calculate_resilience_score()` and `simulate_single_node_failure()`

---

## Demand Forecasting Papers

### 6. Machine Learning in Supply Chain Forecasting
**Paper:** (Example - MIT CTL or similar)
**Title:** "Machine learning approaches for demand forecasting in supply chain management"

**What we reference:**
- Lag features for time series forecasting
- Seasonal decomposition (sine/cosine terms)
- Rolling statistics for capturing trends
- MAPE, MAE, RMSE as accuracy metrics
- Used in `src/forecasting.py` with feature engineering

---

### 7. Demand Forecasting with Regime Shifts
**Paper:** (Find paper on change point detection or structural breaks)
**Title:** "Handling structural breaks and regime shifts in demand forecasting"

**What we reference:**
- Regime shift modeling (change points with multiplicative effects)
- Shock modeling (occasional anomalies)
- Used in `src/data_gen.py` - synthetic demand generation with regime shifts

---

## Simulation and Resilience Papers

### 8. Discrete Event Simulation for Supply Chains
**Paper:** (SimPy documentation or academic paper on DES)
**Title:** "Discrete event simulation for supply chain resilience evaluation"

**What we reference:**
- SimPy framework for discrete event simulation
- MTTF (Mean Time To Failure) and MTTR (Mean Time To Recovery) modeling
- Exponential distributions for failure and recovery times
- Fill rate calculation under disruptions
- Used in `src/sim.py` - `DisruptionSimulation` class

---

### 9. Resilience Metrics and Evaluation
**Paper:** Li, Y., et al. (Example - find specific paper on resilience metrics)
**Title:** "Resilience metrics for supply chain networks: area under performance curve"

**What we reference:**
- Worst-case coverage as resilience metric
- Performance degradation under disruptions
- Normalized resilience score calculation
- Used in `src/sim.py` - `get_resilience_metrics()` function

---

## Optimization and Modeling Papers

### 10. Pyomo Framework
**Paper:** Hart, W. E., Watson, J. P., & Woodruff, D. L. (2011). Pyomo: modeling and solving mathematical programs in Python. Mathematical Programming Computation, 3(3), 219-260.

**What we reference:**
- Pyomo algebraic modeling language
- Multi-objective model formulation
- Constraint and objective definition
- Used throughout `src/pyomo_model.py`

---

### 11. Multi-Objective Optimization in Logistics
**Paper:** (Example - find paper on multi-objective logistics optimization)
**Title:** "Pareto-optimal solutions in green logistics network design"

**What we reference:**
- Pareto front generation for trade-off analysis
- Representative solution selection (min-cost, min-emissions, balanced)
- Utopia point and knee point selection
- Used in `src/optimizer.py` - `extract_representative_solutions()`

---

## Traffic Congestion Papers

### 12. Traffic Congestion in Supply Chain Routing
**Paper:** (Find paper on congestion factors in logistics)
**Title:** "Congestion-aware routing in supply chain networks"

**What we reference:**
- Congestion factors affecting transport cost and emissions
- Dynamic congestion modeling in simulations
- Impact of congestion on delivery times and availability
- Used in `src/data_gen.py` (congestion factors) and `src/pyomo_model.py` (congestion in objectives)

---

## Additional References

### 13. Carbon Emission Factors
**Source:** IPCC Guidelines for National Greenhouse Gas Inventories, or industry standards

**What we reference:**
- Transport emission factors (kg CO2 per unit-km)
- Standard emission factors for logistics operations
- Used in `src/pyomo_model.py` - emission objective calculations

---

### 14. Geodesic Distance Calculations
**Paper/Standard:** Haversine formula or Vincenty's formulae

**What we reference:**
- Geodesic distance calculation between geographical coordinates
- Used in `src/data_gen.py` - `generate_distance_matrix()` using geopy library

---

### 15. Soft Constraints in Evolutionary Algorithms
**Paper:** (Find paper on penalty methods in genetic algorithms)
**Title:** "Penalty functions for constraint handling in evolutionary optimization"

**What we reference:**
- Feasibility penalty approach for constraint violations
- Soft constraint handling in NSGA-II
- Capacity, balance, and demand satisfaction penalties
- Used in `src/optimizer.py` - `_evaluate_objectives()` with penalty terms

---

## Software/Library References

### 16. pymoo Library
**Paper:** Blank, J., & Deb, K. (2020). pymoo: Multi-objective optimization in Python. IEEE Access, 8, 89497-89509.

**What we reference:**
- pymoo framework for multi-objective optimization
- NSGA-II implementation
- Used in `src/optimizer.py`

---

### 17. scikit-learn
**Paper:** Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.

**What we reference:**
- RandomForestRegressor implementation
- Mean absolute error, RMSE, MAPE metrics
- Used in `src/forecasting.py`

---

### 18. SimPy
**Paper/Reference:** SimPy Documentation or academic paper on discrete event simulation frameworks

**What we reference:**
- SimPy discrete event simulation framework
- Process-based simulation modeling
- Used in `src/sim.py`

---

## Summary Table

| # | Paper/Reference | What We Use It For | Where in Code |
|---|----------------|-------------------|---------------|
| 1 | Deb et al. (2002) - NSGA-II | Multi-objective genetic algorithm | `src/optimizer.py` |
| 2 | Breiman (2001) - Random Forest | Demand forecasting | `src/forecasting.py` |
| 3 | Multi-objective SCN design | Network design formulation | `src/pyomo_model.py` |
| 4 | Green SCN papers | Emissions objective | `src/pyomo_model.py` |
| 5 | Resilient SCN papers | Resilience metrics | `src/sim.py` |
| 6 | ML forecasting papers | Feature engineering | `src/forecasting.py` |
| 7 | Regime shift papers | Demand generation | `src/data_gen.py` |
| 8 | DES papers | Simulation framework | `src/sim.py` |
| 9 | Resilience metrics papers | Resilience calculation | `src/sim.py` |
| 10 | Hart et al. (2011) - Pyomo | Modeling framework | `src/pyomo_model.py` |
| 11 | Pareto optimization papers | Solution selection | `src/optimizer.py` |
| 12 | Congestion routing papers | Traffic modeling | `src/data_gen.py`, `src/pyomo_model.py` |
| 13 | IPCC Guidelines | Emission factors | `src/pyomo_model.py` |
| 14 | Haversine/Vincenty | Distance calculation | `src/data_gen.py` |
| 15 | Penalty methods papers | Constraint handling | `src/optimizer.py` |
| 16 | Blank & Deb (2020) - pymoo | Optimization library | `src/optimizer.py` |
| 17 | Pedregosa et al. (2011) - sklearn | ML library | `src/forecasting.py` |
| 18 | SimPy documentation | Simulation library | `src/sim.py` |

---

## Notes for Thesis

1. **Primary Algorithm:** NSGA-II (Deb et al., 2002) is the core method - cite prominently in Methodology section
2. **Forecasting:** Breiman (2001) Random Forest is the ML method - cite in Data Generation section
3. **Pyomo:** Hart et al. (2011) for modeling framework - cite in Problem Formulation
4. **MDPI Papers:** Replace placeholders with actual papers from your literature review on:
   - Green supply chain network design
   - Resilient supply chain optimization
   - Multi-objective optimization in logistics
5. **Industry Standards:** IPCC for emission factors, industry reports for congestion factors
6. **Software:** Cite pymoo, scikit-learn, SimPy as tools/libraries used

---

## How to Format in Thesis

Follow your institution's citation style (APA/IEEE/ACM). Example format:

**In-text citation:** "We use NSGA-II (Deb et al., 2002) to solve the multi-objective optimization problem..."

**Reference entry:**
- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.

---

**Next Steps:**
1. Search for actual MDPI papers on green/resilient supply chains
2. Find MIT CTL or similar papers on demand forecasting
3. Locate papers on congestion-aware routing
4. Format all references according to your institution's style guide


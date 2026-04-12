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

### 3. Multi-Echelon Supply Chain Cost Structure
**Paper:** Melo, M.T., Nickel, S., & Saldanha-da-Gama, F. (2009). Facility location and supply chain management — A review. *European Journal of Operational Research*, 196(2), 401–412.

**What we reference:**
- Multi-echelon SCND cost formulation (procurement + production + transport + holding)
- Foundational 4-echelon cost objective structure
- Used in `src/pyomo_model.py` — `cost_objective_rule()` and `src/optimizer.py` — `_evaluate_objectives()`

---

### 4. Green Supply Chain / LCA-Based Emission Model
**Paper:** Pishvaee, M.S. & Razmi, J. (2012). Environmental supply chain network design using multi-objective fuzzy mathematical programming. *Applied Mathematical Modelling*, 36(8), 3433–3446.

**What we reference:**
- Life Cycle Assessment (LCA) based emission model for supply chains
- Facility-level emissions (supplier, production, DC warehousing)
- DC warehousing emission factor concept
- Used in `src/pyomo_model.py` — `emission_objective_rule()` and `src/optimizer.py` — warehouse emission computation

**Supplementary Paper:** Pishvaee, M.S., Torabi, S.A., & Razmi, J. (2012). Credibility-based fuzzy mathematical programming model for green logistics design under uncertainty. *Computers & Industrial Engineering*, 62(2), 624–632.

**What we reference for supplementary:**
- Congestion-weighted transport emission factors
- Mode-specific (Road vs. Rail) emission modeling

---

### 5. Resilient Supply Chain — HHI Diversification
**Paper:** Hasani, A. & Khosrojerdi, A. (2016). Robust global supply chain network design under disruption and uncertainty considering resilience strategies: A parallel memetic algorithm for a real-life case study. *Transportation Research Part E*, 87, 20–52.

**What we reference:**
- Herfindahl-Hirschman Index (HHI) adapted to supply chain flow concentration
- HHI-based diversification score for supply resilience
- Multi-sourcing as resilience strategy
- Used in `src/optimizer.py` — HHI diversification computation (Component A of resilience)

**Supplementary Paper:** Snyder, L.V. & Daskin, M.S. (2005). Reliability models for facility location: The expected failure cost case. *Transportation Science*, 39(3), 400–416.

**What we reference for supplementary:**
- Expected demand coverage under single-node failure scenarios
- Level-r backup assignment strategy for failure modeling
- Used in `src/optimizer.py` — node failure coverage computation (Component B of resilience)

**Supplementary Paper:** Jabbarzadeh, A., Fahimnia, B., & Sabouhi, F. (2018). Resilient and sustainable supply chain design: Sustainability analysis under disruption risks. *International Journal of Production Research*, 56(17), 5945–5968.

**What we reference for supplementary:**
- Unmet demand penalty in cost objective (shortage cost)
- Integration of resilience into stochastic cost models
- Used in `src/pyomo_model.py` and `src/optimizer.py` — shortage penalty (π = 500 $/unit)

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
| 3 | Melo et al. (2009) - Cost structure | Multi-echelon cost formulation | `src/pyomo_model.py`, `src/optimizer.py` |
| 4 | Pishvaee & Razmi (2012) - LCA emissions | Emission objective (incl. DC warehousing) | `src/pyomo_model.py`, `src/optimizer.py` |
| 4b | Pishvaee, Torabi & Razmi (2012) - Congestion | Congestion-weighted transport emissions | `src/pyomo_model.py`, `src/optimizer.py` |
| 5 | Hasani & Khosrojerdi (2016) - HHI | HHI diversification for resilience | `src/optimizer.py` |
| 5b | Snyder & Daskin (2005) - Node failure | Expected node-failure demand coverage | `src/optimizer.py` |
| 5c | Jabbarzadeh et al. (2018) - Shortage penalty | Unmet demand penalty in cost | `src/pyomo_model.py`, `src/optimizer.py` |
| 6 | ML forecasting papers | Feature engineering | `src/forecasting.py` |
| 7 | Regime shift papers | Demand generation | `src/data_gen.py` |
| 8 | DES / SimPy | Simulation framework | `src/sim.py` |
| 9 | Pettit et al. (2010) | Resilience definition | `src/sim.py` |
| 10 | Hart et al. (2011) - Pyomo | Modeling framework | `src/pyomo_model.py` |
| 11 | Fattahi et al. (2017) | Responsive-resilient SCND | `src/optimizer.py` |
| 12 | IPCC Guidelines | Emission factors | `src/pyomo_model.py` |
| 13 | Haversine/Vincenty | Distance calculation | `src/data_gen.py` |
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


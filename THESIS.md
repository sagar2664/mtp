# AI-Based Multi-Objective Optimization of a Green and Resilient Supply Chain Network

## Declaration
I hereby declare that the work presented in this thesis, titled “AI-Based Multi-Objective Optimization of a Green and Resilient Supply Chain Network,” is my original work carried out under the guidance of my supervisor. This work has not been submitted elsewhere, in part or whole, for any other degree.

## Acknowledgements
I would like to thank my advisor, peers, and institution for their guidance and support. I also acknowledge the open-source community for tools used in this work (Pyomo, pymoo, scikit-learn, SimPy, etc.).

## Abstract
Modern supply chains must jointly achieve low cost, low carbon emissions, and high resilience to disruptions. These objectives conflict, requiring principled multi-objective optimization. This thesis develops and evaluates a reproducible framework for designing a small-scale, four-echelon supply chain (3 suppliers → 2 factories → 3 distribution centers → 5 customer zones). We integrate: (1) synthetic demand generation with realistic trends, seasonality, regime shifts and shocks; (2) machine learning demand forecasting (Random Forest with seasonal and rolling features); (3) a multi-objective network design formulated in Pyomo; (4) a genetic algorithm (NSGA-II) using pymoo to generate Pareto-optimal solutions balancing cost, emissions, and resilience; and (5) a discrete-event simulation in SimPy to evaluate operational resilience under disruptions.

Results show a rich Pareto front of 120 non-dominated solutions. Representative designs (Min Cost, Min Emissions, Balanced/Utopia) illustrate trade-offs: reducing emissions increases cost, while resilience improves via diversified routing at modest additional cost. Simulation with realistic mean-time-to-failure (MTTF) and mean-time-to-recovery (MTTR) settings, with rerouting efficiency, achieves high average fill rates (~97.5%) and robust worst-case performance (~77%) across scenarios. This work provides a reproducible, extensible template for data-driven supply chain design and decision support.

Keywords: supply chain network design, multi-objective optimization, NSGA-II, Pyomo, resilience, carbon emissions, machine learning forecasting, SimPy.

---

## Table of Contents
1. Introduction
2. Literature Review
3. Problem Definition and Mathematical Formulation
4. Data Generation and Forecasting
5. Multi-Objective Optimization Methodology
6. Discrete-Event Simulation for Resilience Evaluation
7. Experimental Setup
8. Results and Analysis
9. Discussion
10. Threats to Validity and Limitations
11. Conclusion
12. Future Work
13. References
14. Appendix A: Reproducibility and Code Structure
15. Appendix B: Parameter Tables
16. Detailed Mathematical Formulation
17. Algorithmic Details and Pseudocode
18. Feature Engineering and Forecasting Study
19. Sensitivity Analysis
20. Ablation Studies
21. Robustness and Stress Tests
22. Scalability and Complexity
23. Implementation Details
24. Ethical and Sustainability Considerations
25. Extended Literature Review (Pointers)
26. Additional Figures and Tables (Placeholders)
27. Author’s Contributions
28. Appendix C: Parameter Values (Example)
29. Appendix D: Replication Steps
30. Appendix E: Risk and Safety Considerations

---

## 1. Introduction
Global supply chains face pressure to optimize economic efficiency, environmental sustainability, and operational resilience. Regulatory demands (e.g., Scope 3 emissions reporting), stakeholder expectations, and frequent disruptions (e.g., pandemics, natural hazards, geopolitical risks) intensify the need for robust design. However, these goals often conflict: low-carbon sourcing may raise costs; redundancy to improve resilience can increase emissions.

This thesis addresses the multi-objective design of a four-echelon supply chain: 3 suppliers → 2 factories → 3 distribution centers (DCs) → 5 customers. We optimize continuous flows at each stage, aiming to minimize cost and emissions while maximizing resilience. We integrate four pillars:
- Synthetic data generation and ML-based demand forecasting.
- Pyomo-based multi-objective model capturing cost, emissions, and resilience.
- NSGA-II (genetic algorithm) to compute Pareto-optimal trade-offs.
- SimPy discrete-event simulation to evaluate resilience under disruptions with rerouting.

Contributions:
- A fully reproducible, modular framework for green and resilient network design.
- A practical resilience measure, combining worst-case coverage and simulation-based stress testing.
- Integration of demand forecasting into planning, reflecting uncertainty and seasonality.
- Empirical insights on trade-offs across cost, emissions, and resilience.

---

## 2. Literature Review
Multi-objective supply chain optimization is well-studied, with NSGA-II frequently used for its ability to approximate Pareto fronts in complex, non-convex spaces. Prior work has applied genetic algorithms to cold chains, green logistics, and risk-aware network design, reporting improved trade-offs over weighted-sum approaches. Demand forecasting in SCM has shifted toward machine learning, leveraging lag features, seasonal decompositions, and tree-based models to improve fill rates. Resilience is variously defined (e.g., worst-case coverage, time to recovery, area under performance curve), with discrete-event simulation an established tool to model disruptions.

Key takeaways from prior studies:
- NSGA-II is effective for balancing competing goals and maintaining solution diversity.
- Forecast accuracy materially influences planning outcomes.
- Redundancy and diversified sourcing improve resilience but increase cost and sometimes emissions.
- Simulation complements optimization, capturing dynamic operational effects under failures.

(References provided in Section 13.)

---

## 3. Problem Definition and Mathematical Formulation
### 3.1 Network and Decision Variables
We consider sets: suppliers \(I\), factories \(J\), DCs \(K\), and customers \(L\). Decision variables are continuous flows:
- \(x_{ij}\): Supplier \(i\) → Factory \(j\)
- \(y_{jk}\): Factory \(j\) → DC \(k\)
- \(z_{kl}\): DC \(k\) → Customer \(l\)

### 3.2 Objectives
- Cost: supplier cost + production cost + transport costs (3 arcs) + inventory/holding
- Emissions: supplier emission factors + production emission factors + transport emissions (unit-km)
- Resilience: maximize worst-case demand coverage (approximated in the model; evaluated via simulation)

### 3.3 Constraints
- Supply, factory, and DC capacity limits
- Flow balance at factories and DCs (inflow = outflow)
- Demand satisfaction at customers
- Non-negativity of flows

The Pyomo model in `src/pyomo_model.py` defines parameters (capacities, distances, costs, emissions), variables (\(x, y, z\)), objectives (cost, emissions as expressions), and constraints. Resilience is handled in two layers: an optimization-side proxy encouraging diversified deliveries; and a simulation-side evaluation using single-node failure coverage and stochastic runs.

---

## 4. Data Generation and Forecasting
### 4.1 Synthetic Demand Generation
We generate realistic synthetic demand with:
- Linear trend and seasonal (12-month) component.
- Higher-variance noise.
- Customer-specific regime shifts (change points with multiplicative effects).
- Occasional multiplicative shocks (e.g., 0.6× or 1.5×) to mimic anomalies.

This creates a more challenging forecasting environment, reducing unrealistically low errors common to simple synthetic series.

### 4.2 Feature Engineering and Model
We forecast using a Random Forest Regressor with:
- Lag features (6 months)
- Seasonal features (month-of-year, sine/cosine terms)
- Rolling statistics (means and standard deviations over 3/6/12 months)

Training uses historical periods (e.g., first 60 of 72 months) and forecasts the next horizon (e.g., 12 months). Accuracy is reported via MAE, RMSE, and MAPE. Forecasts feed into the planning stage by providing demand levels for optimization.

---

## 5. Multi-Objective Optimization Methodology
### 5.1 NSGA-II and Pareto Fronts
We implement NSGA-II (pymoo) with population sizes (e.g., 120) and generations (e.g., 300). The decision vector concatenates \(x, y, z\). Objective evaluation includes:
- Cost and emissions calculations derived from flows and distances.
- Feasibility penalties for capacity overruns, flow imbalance, and unmet demand (soft constraint handling), guiding the search toward valid solutions.

### 5.2 Representative Solutions
We extract representatives from the Pareto front:
- Min Cost
- Min Emissions
- Balanced (Utopia): closest to ideal point (low cost, low emissions, high resilience)
- Balanced (Knee): maximum curvature point (optional)

This dual balanced selection avoids selecting unintuitive extreme points as “balanced.”

---

## 6. Discrete-Event Simulation for Resilience Evaluation
We implement a SimPy simulation of supplier/factory failures with:
- MTTF (Mean Time To Failure) and MTTR (Mean Time To Recovery)
- Node availability processes for suppliers and factories
- Rerouting efficiency: proportion of lost delivery volume recovered via available DCs during disruptions
- Fill-rate computation from planned \(z\)-flows adjusted by upstream availability and rerouting

We compute:
- Average fill rate across runs
- Worst-case (minimum) fill rate across runs (conservative resilience proxy)
- Average number of disruptions

---

## 7. Experimental Setup
- Echelons: 3 suppliers, 2 factories, 3 DCs, 5 customers
- Time horizon: 72 months generated; 60 months used for training; 12 months forecasted
- Algorithms: NSGA-II (population 120, generations 300), Random Forest forecaster
- Simulation: 100 runs; MTTF=5 years; MTTR=45 days; reroute efficiency=0.5
- Environment: Python 3.10; packages: numpy, pandas, scikit-learn, pyomo, pymoo, simpy, matplotlib, seaborn, geopy

Reproducibility: `bash scripts/run_all.sh` orchestrates the full pipeline; all outputs saved under `results/`.

---

## 8. Results and Analysis
### 8.1 Forecasting
- Achieved MAPE ≈ 16–20% (depending on seed), reflecting increased difficulty from regime shifts and shocks. This is more realistic than overly smooth synthetic series.

### 8.2 Optimization and Pareto Front
- NSGA-II produced 120 Pareto-optimal solutions (example runtime ~5 seconds on a laptop for this scale).
- Trade-offs observed:
  - Lower emissions raise cost; modest resilience changes originate from routing diversification.
  - Balanced (Utopia) yields reasonable cost/emissions with improved resilience compared to Min Cost.

Refer to exported `results/pareto_front.csv` and `results/experiment_summary.json` for representative values.

### 8.3 Simulation and Resilience
- Average fill rate ≈ 97–98%; worst-case ≈ 77–80% under disruptions.
- Rerouting recovers a portion of lost capacity; increasing reroute efficiency or redundancy improves resilience at additional cost/emissions.

### 8.4 Visual and Tabular Summaries
- Pareto table: `results/pareto_front.csv`
- Representative flows: `results/flows_*_{min_cost|min_emissions|balanced_(utopia)|balanced_(knee)}.csv`
- Summary: `results/experiment_summary.json`

---

## 9. Discussion
- The framework successfully exposes actionable trade-offs. Decision-makers can prioritize cost, emissions, or resilience.
- Forecasting quality impacts planning; we inject realistic complexity to avoid underestimated uncertainty.
- The resilience proxy in optimization encourages diversification, but final evaluation via simulation is essential to capture dynamic behaviors under failures.
- The “Balanced (Utopia)” selection provides a more intuitive compromise than sole knee-based picks.

---

## 10. Threats to Validity and Limitations
- Synthetic data may not capture all real-world correlations and seasonal patterns.
- Feasibility penalties (soft constraints) may admit slightly infeasible individuals during search; post-processing or repair operators could tighten feasibility.
- Emission factors and cost coefficients are illustrative; absolute values depend on context.
- Simulation assumes simplified rerouting; real systems have lead times, transport modes, and inventory buffers.

---

## 11. Conclusion
This thesis presents a reproducible, end-to-end approach for multi-objective supply chain network design that balances cost, emissions, and resilience. By combining machine learning forecasting, Pyomo modeling, NSGA-II optimization, and SimPy simulation, we generate insightful Pareto fronts and robust resilience metrics. The design template is extensible for larger networks and richer constraints and provides a practical decision-support toolkit for modern supply chain planning.

---

## 12. Future Work
- Extend to larger, real datasets; integrate GIS routing and realistic transport modes.
- Add inventory dynamics, safety stock policies, and service-level agreements.
- Explore multi-period optimization with rolling-horizon control.
- Use repair operators and constraint handling to guarantee feasibility throughout evolution.
- Integrate carbon pricing, renewable energy investments, and probabilistic disruptions.

---

## 13. References
[1] Deb, K., et al. “A fast and elitist multiobjective genetic algorithm: NSGA-II.” (2002).  
[2] Pyomo Documentation. `https://pyomo.readthedocs.io`  
[3] pymoo: Multi-objective Optimization in Python. `https://pymoo.org`  
[4] Scikit-learn: Machine Learning in Python. `https://scikit-learn.org`  
[5] SimPy: Discrete Event Simulation. `https://simpy.readthedocs.io`  
[6] Geopy: Geodesic computations. `https://geopy.readthedocs.io`  
[7] Selected MDPI articles on green/resilient supply chains and NSGA-II (add exact citations per your style).  
[8] MIT CTL materials on demand forecasting (add exact citations per your style).  
[9] IPCC/industry sources for emission factors and logistics emissions (add exact citations).

(Note: Please replace placeholders with formal citations in your preferred bibliography style—APA/IEEE/ACM—before submission.)

---

## 14. Appendix A: Reproducibility and Code Structure
Project root contains:
- `src/`: data generation, forecasting, optimization (Pyomo/pymoo), simulation, analysis
- `scripts/`: `run_all.sh` pipeline executor
- `tests/`: unit tests for data, model, and optimizer
- `notebooks/`: optional demo notebook
- `results/`: exports from runs (`experiment_summary.json`, `pareto_front.csv`, `flows_*.csv`)

### How to Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh
```

---

## 15. Appendix B: Parameter Tables
- Cost coefficients (supplier cost, production cost, transport cost per km)
- Emission factors (supplier, production, transport per unit-km)
- Capacities (supplier, factory, DC)
- Simulation parameters (MTTF, MTTR, reroute efficiency)

(See `src/pyomo_model.py` for defaults and `scripts/run_all.sh` for experiment settings.)

---

## 16. Detailed Mathematical Formulation
### 16.1 Sets and Indices
- Suppliers: \( I = \{1, \dots, |I|\} \)
- Factories: \( J = \{1, \dots, |J|\} \)
- Distribution Centers (DCs): \( K = \{1, \dots, |K|\} \)
- Customers: \( L = \{1, \dots, |L|\} \)

### 16.2 Parameters
- \( S_i \): capacity of supplier \(i\) (units)
- \( F_j \): capacity of factory \(j\)
- \( D_k \): capacity of DC \(k\)
- \( c^{sup}_i \): material cost at supplier \(i\) ($/unit)
- \( c^{prod}_j \): production cost at factory \(j\) ($/unit)
- \( c^{trans} \): transport cost per unit per km ($/(unit·km))
- \( h_k \): holding cost at DC \(k\) ($/unit)
- \( d^{sf}_{ij} \), \( d^{fd}_{jk} \), \( d^{dc}_{kl} \): distances (km)
- \( e^{sup}_i \): supplier emission factor (kg CO2/unit)
- \( e^{prod}_j \): production emission factor (kg CO2/unit)
- \( e^{trans} \): transport emission factor (kg CO2/(unit·km))
- \( q_l \): demand of customer \(l\) (units)

### 16.3 Decision Variables
- \( x_{ij} \ge 0 \): flow from supplier \(i\) to factory \(j\)
- \( y_{jk} \ge 0 \): flow from factory \(j\) to DC \(k\)
- \( z_{kl} \ge 0 \): flow from DC \(k\) to customer \(l\)

### 16.4 Objectives
Cost (minimize):
\[
\min \; C = \sum_{i \in I}\sum_{j \in J} c^{sup}_i x_{ij} 
\; + \sum_{j \in J} c^{prod}_j \Big(\sum_{i \in I} x_{ij}\Big)
\; + c^{trans}\bigg(\sum_{i,j} d^{sf}_{ij} x_{ij} + \sum_{j,k} d^{fd}_{jk} y_{jk} + \sum_{k,l} d^{dc}_{kl} z_{kl}\bigg)
\; + \sum_{k \in K} h_k \Big(\sum_{j \in J} y_{jk}\Big)
\]

Emissions (minimize):
\[
\min \; E = \sum_{i \in I}\sum_{j \in J} e^{sup}_i x_{ij}
\; + \sum_{j \in J} e^{prod}_j \Big(\sum_{i \in I} x_{ij}\Big)
\; + e^{trans}\bigg(\sum_{i,j} d^{sf}_{ij} x_{ij} + \sum_{j,k} d^{fd}_{jk} y_{jk} + \sum_{k,l} d^{dc}_{kl} z_{kl}\bigg)
\]

Resilience (maximize): two-layer approach. In the optimization model we use a proxy encouraging diversification:
\[
R_{proxy} = - \sum_{l \in L}\sum_{k \in K} \Big( \frac{z_{kl}}{\sum_{k' \in K} z_{k'l} + \epsilon}\Big)^2
\]
In simulation we compute worst-case coverage under single-node failures and stochastic failures.

### 16.5 Constraints
- Supplier capacity: \( \sum_{j \in J} x_{ij} \le S_i, \; \forall i \in I \)
- Factory capacity: \( \sum_{i \in I} x_{ij} \le F_j, \; \forall j \in J \)
- Factory balance: \( \sum_{i \in I} x_{ij} = \sum_{k \in K} y_{jk}, \; \forall j \in J \)
- DC capacity: \( \sum_{j \in J} y_{jk} \le D_k, \; \forall k \in K \)
- DC balance: \( \sum_{j \in J} y_{jk} = \sum_{l \in L} z_{kl}, \; \forall k \in K \)
- Demand satisfaction: \( \sum_{k \in K} z_{kl} \ge q_l, \; \forall l \in L \)

Soft feasibility penalties in NSGA-II encourage satisfaction of these constraints during search.

---

## 17. Algorithmic Details and Pseudocode
### 17.1 NSGA-II Settings
- Population size: 120
- Generations: 300
- Selection: non-dominated sorting + crowding distance
- Variation: SBX crossover, polynomial mutation (pymoo defaults)

### 17.2 Pseudocode (High-Level)
```
Initialize population of flows (x,y,z)
For gen in 1..G:
    Evaluate objectives for each individual:
        cost ← C(x,y,z)
        emissions ← E(x,y,z)
        penalty ← feasibility_penalty(x,y,z)
        cost ← cost + α·penalty; emissions ← emissions + β·penalty
        resilience ← 1 / (1 + concentration_penalty(z))
    Non-dominated sort; compute crowding distance
    Select parents; apply crossover and mutation
    Merge and select next generation by rank and crowding distance
Return Pareto set
```

### 17.3 Balanced Selection
- Utopia-distance: closest to normalized (0,0,0)
- Knee: maximum distance from ideal–nadir line (optional)

---

## 18. Feature Engineering and Forecasting Study
Compare feature sets and errors; discuss effects of regime shifts, rolling windows, and seasonal terms. Explore model choices (RF vs. gradient boosting), temporal cross-validation, and uncertainty estimation.

---

## 19. Sensitivity Analysis
Vary transport cost, emission factors, and capacities; analyze Pareto front movement and representative solutions’ stability.

---

## 20. Ablation Studies
Remove feasibility penalties; set reroute efficiency to zero; compare convergence speed and resilience outcomes.

---

## 21. Robustness and Stress Tests
Test single-node and multi-node failures, prolonged MTTR, and reduced MTTF; report worst-case fill and recovery behavior.

---

## 22. Scalability and Complexity
Discuss decision variable growth, evaluation cost, and strategies for larger instances (parallelization, decomposition, MILP polishing).

---

## 23. Implementation Details
Python environment, package versions, seed control, and file structure; guidance for adaptation.

---

## 24. Ethical and Sustainability Considerations
Internalizing externalities, fair labor, lifecycle emissions, transparency in AI forecasting.

---

## 25. Extended Literature Review (Pointers)
Summaries on green logistics, multi-objective algorithms in SCM, resilience metrics, ML forecasting in supply chains (insert formal citations).

---

## 26. Additional Figures and Tables (Placeholders)
Pareto plots, flow maps, forecast vs. actual plots, sensitivity and simulation tables; link to `results/`.

---

## 27. Author’s Contributions
Modeling, implementation, experiments, analysis, and documentation.

---

## 28. Appendix C: Parameter Values (Example)
Transport cost, emission factors, cost ranges, regime multipliers, shock probabilities.

---

## 29. Appendix D: Replication Steps
Environment creation, running the script, and adjusting parameters.

---

## 30. Appendix E: Risk and Safety Considerations
Avoid overfitting, stress-test plans, and evaluate business continuity.

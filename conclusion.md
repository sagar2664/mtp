## 7. Conclusion

This thesis set out to design and evaluate an AI‑based, multi‑objective optimization framework for a green and resilient supply chain network spanning four echelons: suppliers, factories, distribution centers (DCs), and customer zones. The framework integrates synthetic yet realistic data generation, machine‑learning demand forecasting, a three‑objective network design model in Pyomo, multi‑objective optimization via NSGA‑II, and a discrete‑event simulation to assess resilience under stochastic disruptions and traffic congestion. The results demonstrate that meaningful trade‑offs among cost, carbon emissions, and resilience can be systematically explored and that data‑driven insights can guide robust network design choices.

### 7.1 Summary of Contributions
- Developed a complete, reproducible pipeline that: (i) generates multi‑echelon network data with spatial realism and congestion, (ii) forecasts demand using feature‑rich Random Forests, (iii) formulates a 3‑objective Pyomo model (minimize cost, minimize emissions, maximize resilience), (iv) solves with NSGA‑II using feasibility‑aware penalties, and (v) validates with a SimPy disruption simulation.
- Incorporated traffic congestion in both the optimization objectives (cost/emissions via distance×congestion) and the simulation (availability and flow propagation), aligning environmental and operational realism.
- Implemented representative‑solution selection from the Pareto set: Min Cost, Min Emissions, Balanced (Utopia), and Balanced (Knee), ensuring non‑duplicated managerial “picks.”
- Added a resilience proxy in optimization and a richer resilience validation in simulation with rerouting policies, realistic MTTF/MTTR, and congestion‑affected availability.

-### 7.2 Key Findings
- Pareto analysis confirms pronounced trade‑offs: solutions minimizing cost tend to increase emissions or reduce diversity/resilience; low‑emission solutions can incur higher cost or lower capacity headroom; balanced solutions spread flows more evenly, enhancing resilience at moderate cost/emission levels (see Figures 9 and 10).
- Representative solutions are distinct and interpretable:
  - Min Cost concentrates flows along the least‑cost routes and facilities, elevating exposure to node failures and congestion spikes.
  - Min Emissions prioritizes shorter, cleaner routes and lower emission factors, sometimes increasing transport legs or inventory to maintain service.
  - Balanced (Utopia) and Balanced (Knee) provide stable compromises with diversified sourcing and distribution, leading to stronger simulated fill rates (see Figure 11 for comparison; see Figures 12–15 for flow maps).
- Simulation validates resilience mechanisms (see Figure 17):
  - Balanced solutions exhibit higher average and minimum fill rates under disruptions due to diversified z‑flows and rerouting, despite congestion variability.
  - Realistic reliability (MTTF≈5 years, MTTR≈45 days) reveals meaningful performance dispersion across designs, underlining the value of explicit resilience optimization.
- Demand forecasting with enhanced features (seasonality, rolling stats, calendar effects) achieves robust accuracy on synthetic series with trends, seasonality, regime shifts, and shocks, supporting stable optimization inputs.
  - See Figure 5 for demand forecasts and Figure 6 for accuracy metrics.
- Congestion matters: incorporating congestion factors shifts the frontier, raising both costs and emissions and amplifying the benefit of diversified routing and facility usage.
  - See Figure 2 for congestion analysis and Figure 4 for distance heatmaps.

### 7.3 Managerial and Policy Implications
- Treat resilience as a first‑class objective: diversified flows and strategic headroom materially improve service continuity during failures or congestion spikes.
- Carbon‑aware routing and cleaner production choices can reduce emissions substantially; however, expect trade‑offs with cost and, at times, responsiveness. Balanced solutions often deliver strong overall value.
- Congestion‑adjusted planning is essential in urban and inter‑city logistics; ignoring it leads to optimistic cost/CO₂ estimates and brittle designs.
- Simulation complements optimization by revealing dynamic effects and validating resilience policies (e.g., rerouting effectiveness, recovery assumptions).

### 7.4 Limitations
- Synthetic data, while realistic, cannot capture all domain‑specific nuances (e.g., exact lead times, contractual terms, multimodal constraints). Results should be interpreted as methodological validation rather than company‑specific prescription.
- The resilience objective in the optimizer uses a proxy to incentivize diversification; simulation provides the ground‑truth stress test, but a gap may remain between proxy and true resilience.
- Single‑period or aggregated‑period design simplifies temporal inventory/production dynamics; multi‑period models could capture seasonality and ramp constraints more fully.
- Emission factors and transport cost coefficients are stylized; different geographies, fuels, and load factors can change magnitudes and the Pareto shape.

### 7.5 Future Work
- Richer resilience modeling in‑model (e.g., chance constraints, robust/ stochastic formulations, or two‑stage models with recourse), closing the gap between proxy and simulated resilience.
- Multi‑period and multi‑product extensions with inventory dynamics, capacity expansion decisions, and service‑level constraints (e.g., fill‑rate or late‑delivery penalties) embedded directly in the optimization.
- Calibrated emissions and congestion models using open real‑world datasets; inclusion of modal choices (road, rail, sea) and vehicle/load decisions.
- Hybrid ML‑OR approaches: demand scenarios from probabilistic forecasting; surrogate models to accelerate evaluation; active learning to guide NSGA‑II exploration.
- Decision support: interactive Pareto analytics, budget/carbon constraints exploration, and “what‑if” stress tests integrated into a dashboard.

### 7.6 Final Remarks
This work demonstrates that a unified, reproducible, and modular pipeline can illuminate the complex trade‑offs among cost, carbon, and resilience in supply chain network design. By uniting data generation, forecasting, mathematical optimization, and simulation under one roof—and by accounting explicitly for congestion and disruptions—the approach provides decision‑makers with transparent choices and stress‑tested designs. While several extensions remain, the current framework already enables evidence‑based selection of balanced solutions that meaningfully improve sustainability and robustness without losing sight of cost efficiency. As data availability and computational tools continue to advance, such integrated AI‑for‑OR approaches can become standard practice for building greener and more resilient supply chains.



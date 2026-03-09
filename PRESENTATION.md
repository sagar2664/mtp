# AI-Based Multi-Objective Optimization of a Green and Resilient Supply Chain Network

**M.Tech Thesis Presentation**

---

## Slide 1 – Motivation
- Cost pressure, carbon regulations, and repeated disruptions strain global supply chains.
- Lean, single-objective planning collapses under uncertainty and congestion.
- Need multi-objective decision support that balances efficiency, sustainability, resilience.
- **Figure placeholder:** `results/figures/network_topology.png`

## Slide 2 – Problem Statement
- Plan flows for a 4-echelon network: 3 suppliers → 2 factories → 3 DCs → 5 customers.
- Decision variables: `x_ij`, `y_jk`, `z_kl` material flows on each arc.
- Goal: minimize cost and emissions while maximizing resilience against node failures.
- **Figure placeholder:** four-echelon flow diagram.

## Slide 3 – Research Objectives
- Build a tri-objective mathematical model capturing cost, emissions, resilience.
- Apply NSGA-II to generate Pareto-optimal supply policies.
- Embed congestion factors into transport cost and emission calculations.
- Evaluate resilience via discrete-event simulation and extract representative solutions.

---

## Slide 4 – Literature Snapshot: Supply Chains
- Classical SCN models minimise cost; sustainability and resilience remain siloed.
- Multi-objective approaches exist but rarely combine AI, congestion, and simulation.
- Opportunity: integrated analytics pipeline for decision-ready trade-offs.

## Slide 5 – Literature Spotlight: NSGA-II
- NSGA-II (Deb et al., 2002) is the workhorse for multi-objective SCM problems.
- Strengths: fast non-dominated sorting, crowding-based diversity, elitism.
- Widely adopted for logistics, energy, and manufacturing trade-off studies.
- **Figure placeholder:** NSGA-II workflow graphic.

## Slide 6 – Literature Themes: Green & Resilient
- Green SCM emphasises transport/production emissions; congestion rarely modelled explicitly.
- Resilience studies rely on fill-rate/time-to-recover metrics, often via DES.
- Gap: joint treatment of emissions, resilience, and congestion within one optimizer.

---

## Slide 7 – Research Gap
- Few frameworks jointly optimise cost, emissions, and resilience on the same network.
- Congestion handling often absent or oversimplified.
- Resilience usually validated analytically, not with realistic stochastic simulation.
- Decision visualisation of Pareto trade-offs underdeveloped.

## Slide 8 – Our Contributions
- Unified pipeline: data generation → AI forecasting → Pyomo model → NSGA-II → SimPy → analytics.
- Congestion-aware objective and penalty design guiding feasible, realistic flows.
- Simulation-driven resilience metric propagated back into optimisation.
- Representative solution extraction for managerial storytelling (Cost, Emissions, Balanced Utopia/Knee).
- **Figure placeholder:** gap vs. contribution comparison.

---

## Slide 9 – Framework Overview
- Workflow orchestrated by `scripts/run_all.sh` for reproducibility.
- Modules: `data_gen`, `forecasting`, `pyomo_model`, `optimizer` (NSGA-II), `sim`, `analysis`.
- Outputs: Pareto front, flow tables, resilience reports, rich visualisations.
- **Figure placeholder:** architecture diagram.

## Slide 10 – Mathematical Model (Compact)
- Objective vector `[f1, f2, f3]`: cost, emissions, resilience coverage.
- Cost includes procurement, production, holding, congestion-adjusted transport.
- Emissions mirror transport congestion, supplier, and production factors.
- Resilience maximises minimum demand coverage across single-node failure scenarios.
- Constraints: supply/factory/DC capacities, flow balance, demand satisfaction, non-negativity.

---

## Slide 11 – NSGA-II Role
- Treats Pyomo model as evaluator inside evolutionary loop.
- Handles 27 decision variables (all flows) with feasibility penalties.
- Produces diverse Pareto front for managers to explore strategic trade-offs.

## Slide 12 – Solution Encoding
- Decision vector concatenates `[x_ij | y_jk | z_kl]` with bounds from forecasted demand.
- Decoding rebuilds echelon matrices ensuring non-negativity.
- **Figure placeholder:** encoding schematic showing vector-to-network mapping.

---

## Slide 13 – NSGA-II Steps (Part 1)
- Initialise random population (size 120) within realistic bounds.
- Evaluate cost, emissions, resilience, plus soft penalties for constraint violations.
- Rank via fast non-dominated sorting (Front 1 = Pareto set).
- Compute crowding distance to preserve diversity along the front.

## Slide 14 – NSGA-II Steps (Part 2)
- Tournament selection using (rank, crowding) dominance.
- Simulated binary crossover (rate 0.9) explores new flow combinations.
- Polynomial mutation (rate 0.1) maintains local search and feasibility repair.
- Combine parents + offspring, keep best 120, iterate for 300 generations.
- **Figure placeholder:** iterative Pareto evolution graphic.

---

## Slide 15 – Objective Evaluation Highlights
- Cost: supplier + production + transport×congestion + holding components.
- Emissions: supplier + production + transport×congestion emissions.
- Resilience: minimum demand coverage across supplier/factory outage simulations.
- Penalties: unmet demand, capacity overflow, flow imbalance.
- **Figure placeholder:** evaluation flowchart.

## Slide 16 – Algorithm Settings
- Population 120, generations 300, random seed 42.
- High crossover for exploration, moderate mutation for exploitation.
- Runtime 5–15 minutes depending on hardware; Pareto archive ~70 points.
- **Figure placeholder:** convergence / hypervolume plot.

## Slide 17 – Representative Solutions
- Min Cost: tight sourcing, higher emissions, lower simulated resilience.
- Min Emissions: prioritises short, low-congestion routes (cost increases).
- Balanced (Utopia): closest to ideal objective vector.
- Balanced (Knee): highest curvature; pragmatic compromise for managers.
- **Figure placeholder:** Pareto front annotated with representatives.

---

## Slide 18 – System Components
- Synthetic data: capacities, lat/long, congestion factors, tougher demand regimes.
- Forecasting: Random Forest with lags, calendar, rolling statistics (focus kept light).
- Optimisation: Pyomo model + NSGA-II in `optimizer.py` delivering Pareto set.
- Simulation: SimPy disruptions (MTTF 5 yrs, MTTR 45 days) with rerouting efficiency.
- Analysis: `analysis.py` + `visualize_results.py` generating plots and summaries.

## Slide 19 – NSGA-II Integration Details
- `SupplyChainProblem` extends `pymoo.core.problem.Problem`.
- `_evaluate` decodes flows, computes objectives, returns `[cost, emissions, -resilience]`.
- Soft penalties enforce supply, capacity, and demand fulfilment feasibility.
- `solve_with_nsga2` configures algorithm, exports Pareto archive + metadata.
- **Figure placeholder:** code structure diagram or pseudocode block.

---

## Slide 20 – Congestion Modelling
- Distances scaled by congestion factors (1.0–2.0) for each supplier→factory, factory→DC, DC→customer arc.
- Influences both cost and emissions objectives simultaneously.
- Simulation also reduces effective DC capacity when congestion spikes.
- Drives NSGA-II toward diversified routing and better resilience.
- **Figure placeholder:** `results/figures/congestion_analysis.png`.

## Slide 21 – Resilience Simulation
- SimPy processes model supplier/factory uptime/downtime (exponential MTTF/MTTR).
- Rerouting logic propagates upstream availability to downstream `z` flows.
- Metrics: fill rate trajectory, worst-case coverage, recovery duration.
- Resilience objective uses worst-case coverage across disruption scenarios.
- **Figure placeholder:** simulation workflow graphic.

---

## Slide 22 – Pareto Insights
- Clear cost–emission trade-off; resilience improves with network diversification.
- Balanced solutions occupy mid-front with manageable increases in cost/emissions.
- Pareto front gives leadership multiple viable operating points.
- **Figures:** `pareto_front_3d.png`, `pareto_front_2d.png`.

## Slide 23 – Representative Flow Patterns
- Min Cost concentrates volume on cheapest supplier–factory combinations.
- Min Emissions leans on proximity, reducing carbon but raising unit cost.
- Balanced solutions spread flows, preserving service during outages.
- **Figures:** network flow plots for Min Cost, Min Emissions, Balanced (Utopia/Knee).

## Slide 24 – Resilience Outcomes
- Balanced (Utopia) maintains >90% fill rate during simulated disruptions.
- Min Cost drops toward ~85% due to single-sourcing vulnerability.
- Recovery times align with MTTR but backlog differs by strategy.
- **Figure:** `results/figures/simulation_results.png`.

## Slide 25 – NSGA-II Performance
- Hypervolume/convergence stabilises after ~200 generations.
- Crowding distance prevents premature convergence; diversity retained.
- Execution profile reported in `experiment_summary.json` for reproducibility.
- **Figure placeholder:** convergence metric chart.

---

## Slide 26 – Key Findings
- Cost, emissions, resilience cannot be minimised/maximised simultaneously—trade-offs explicit.
- Congestion-aware modelling shifts Pareto front toward realistic economics.
- NSGA-II efficiently explores solution space, delivering ~70 distinct options.
- Simulation validates optimisation results under stochastic failures.

## Slide 27 – Managerial Takeaways
- Pick solution aligning with corporate priority: cost, carbon, or resilience.
- Balanced (Utopia) recommended for critical networks needing robustness and green performance.
- Pipeline enables rapid “what-if” analysis without manual solver tuning.
- Framework extensible to additional echelons, products, or geographies.

---

## Slide 28 – Conclusion & Future Scope
- Delivered reproducible, congestion-aware NSGA-II optimisation for green resilient SCN.
- Integrated forecasting, optimisation, simulation, analytics in `run_all.sh` workflow.
- Future work: multi-period planning, stochastic demand, facility location decisions, multi-product modeling.
- **Figure placeholder:** closing graphic or roadmap illustration.

## Slide 29 – References
- Deb K. et al. (2002) – NSGA-II core algorithm.
- Melo M.T. et al. (2009) – Facility location & supply chain review.
- Pettit T.J. et al. (2010) – Supply chain resilience framework.
- Wang F. et al. (2011) – Green supply chain optimisation.
- Additional citations listed in `REFERENCES.md`.

---

**End of Presentation**


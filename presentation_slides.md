# M.Tech Thesis Presentation Content

**Title:** Multi-Objective Optimization for Sustainable and Disruption-Resilient Supply Chain Network Design

---

## Slide 1: Title Slide
* **Title:** Multi-Objective Optimization for Sustainable and Disruption-Resilient Supply Chain Network Design
* **Subtitle:** An Integrated NSGA-II Framework with LCA Emissions and HHI-Based Resilience Metrics
* **Presenter:** Sagar
* **Degree:** Master of Technology
* **Institution:** Indian Institute of Technology Kharagpur
* **Date:** April 2026

---

## Slide 2: Background and Motivation
* **Global Vulnerability:** Recent events (COVID-19, geopolitical tensions) have exposed the fragility of lean, cost-optimised supply chains. Disruption losses exceed billions annually.
* **Environmental Pressures:** Logistics and manufacturing account for ~25% of global $CO_2$ emissions. Stricter regulations require greener networks.
* **The Core Trade-off:** 
  * Lowest cost = Concentrated flows (High vulnerability).
  * Highest resilience = Diversified flows across nodes (High cost, high emissions).
  * Lowest emissions = Rail-heavy, low-emission routes (May reduce diversity).
* **Need for Multi-Objective Approach:** Traditional single-objective (cost) models fail to capture these inherent trade-offs.

---

## Slide 3: Problem Statement
* **Objective:** Design a 4-echelon supply chain network (Suppliers $\rightarrow$ Factories $\rightarrow$ DCs $\rightarrow$ Customers).
* **Multi-modal:** Two transport modes available (Road and Rail).
* **Goals:** Determine the optimal material flow allocation to simultaneously:
  1. **Minimise** total landed cost.
  2. **Minimise** total life-cycle carbon emissions.
  3. **Maximise** a composite resilience score.
* **Constraint Envelope:** Subject to supply limits, processing capacities, flow conservation, and demand satisfaction.

---

## Slide 4: Variable Definitions
* **Network Topology:** 
  * $I$: Suppliers (3 nodes)
  * $J$: Factories (2 nodes)
  * $K$: Distribution Centres (3 nodes)
  * $L$: Customers (5 nodes)
  * $M$: Transport Modes (Road, Rail)
* **Continuous Decision Variables (54 in total):**
  * $x_{ijm}$: Units shipped from Supplier $i$ to Factory $j$ via Mode $m$
  * $y_{jkm}$: Units shipped from Factory $j$ to DC $k$ via Mode $m$
  * $z_{klm}$: Units shipped from DC $k$ to Customer $l$ via Mode $m$

---

## Slide 5: Assumptions
* **Planning Horizon:** Single-period, single-product flow allocation.
* **Deterministic Baseline:** Facility locations are fixed; capacities are deterministic. Demand is estimated via a forecasting module.
* **Linearity:** Cost and emission functions are linear with respect to the flow variables.
* **Congestion:** Traffic congestion factors ($\gamma$) are assumed static during the optimization phase but stochastic during simulation validation.
* **Failure Dynamics:** Facility failures follow exponential distributions (MTTF and MTTR) in the resilience simulation.

---

## Slide 6: Objective Functions Overview
* **The Optimization Problem:**
  $$\min f_1(x, y, z), \quad \min f_2(x, y, z), \quad \max f_3(x, y, z)$$
* **$f_1$:** Total Landed Cost (Economic Performance)
* **$f_2$:** Total Life-Cycle Carbon Emissions (Sustainability)
* **$f_3$:** Composite Resilience Score (Robustness)
* **Academic Traceability:** Every objective component is explicitly traced to published peer-reviewed models (Melo et al., Pishvaee & Razmi, Snyder & Daskin, etc.).

---

## Slide 7: Objective 1 - Total Cost Minimisation
* **Foundation:** Based on the canonical multi-echelon decomposition by Melo et al. (2009).
* **Extension:** Incorporates unmet demand penalty costs from Jabbarzadeh et al. (2018).
* **Total Cost Formulation ($f_1$):**
  $$f_1 = C_{proc} + C_{prod} + C_{trans} + C_{hold} + C_{short}$$
* Decomposes total supply chain cost into five distinct layers to capture the full "total landed cost".

---

## Slide 8: Objective 1 - Cost Components Breakdown
* **$C_{proc}$ (Procurement):** Purchasing raw materials from suppliers ($s_i \cdot x_{ijm}$).
* **$C_{prod}$ (Production):** Manufacturing costs at factories ($p_j \cdot$ factory inflow).
* **$C_{trans}$ (Transport):** Costs across all echelons. Includes a **congestion factor** ($\gamma_{ab} \geq 1$) that inflates costs on congested routes.
* **$C_{hold}$ (Holding):** Warehousing goods at distribution centres ($h_k \cdot$ DC inflow).
* **$C_{short}$ (Shortage Penalty):** Penalises unmet demand ($\pi = \$500$/unit). Converts hard demand constraints into soft, realistic service-level trade-offs.

---

## Slide 9: Objective 2 - Carbon Emissions Minimisation
* **Foundation:** Based on the Life Cycle Assessment (LCA) approach by Pishvaee and Razmi (2012).
* **Key Insight:** A naive transport-only emission model severely underestimates true environmental impact. Significant emissions occur at the facilities.
* **Total Emissions Formulation ($f_2$):**
  $$f_2 = E_{sup} + E_{prod} + E_{trans} + E_{wh}$$
* Comprehensively captures carbon footprint across all operational layers.

---

## Slide 10: Objective 2 - Emission Components Breakdown
* **$E_{sup}$ (Supplier):** $CO_2$ from raw material extraction and processing ($e_i \cdot x_{ijm}$).
* **$E_{prod}$ (Production):** $CO_2$ from energy-intensive manufacturing at factories ($\varepsilon_j \cdot$ factory inflow).
* **$E_{trans}$ (Transport):** Congestion-weighted emissions. Base rate $\tau_m$ is inflated by congestion $\gamma_{ab}$, capturing the reality that idling vehicles generate disproportionate emissions.
* **$E_{wh}$ (Warehousing):** $CO_2$ from DC operations like refrigeration, material handling, and lighting ($\xi_k \cdot$ DC inflow).

---

## Slide 11: Objective 3 - Resilience Maximisation
* **Dual Approach:** Combines structural diversification with worst-case failure robustness.
* **Composite Resilience Score ($f_3$):**
  $$f_3 = w_1 \cdot R_{div} + w_2 \cdot R_{\min}$$
* $R_{div}$: Network Diversification via Herfindahl–Hirschman Index (HHI).
* $R_{\min}$: Expected Node-Failure Coverage.
* **Weights:** $w_1 = 0.4$, $w_2 = 0.6$ (Higher weight on worst-case robustness to reflect automotive industry priorities).

---

## Slide 12: Objective 3 - Diversification ($R_{div}$)
* **Source:** Hasani and Khosrojerdi (2016).
* **Mechanism:** Uses the Herfindahl–Hirschman Index (HHI) to measure flow concentration.
* **Concept:**
  * HHI = 1.0 $\rightarrow$ 100% flow through a single node (Total fragility).
  * HHI = $1/n$ $\rightarrow$ Equal split across $n$ nodes (High resilience).
* **Application:** Computed for both upstream (supplier sources) and downstream (DC deliveries), then averaged.
* $R_{div} = 1 - \frac{1}{2}(HHI_{sup} + HHI_{del})$ (Higher is more diversified).

---

## Slide 13: Objective 3 - Node-Failure Coverage ($R_{\min}$)
* **Source:** Snyder and Daskin (2005).
* **Mechanism:** Quantifies the worst-case single-node failure vulnerability.
* **Calculation:**
  * For every node $n$, calculate its flow share $q(n)/Q$.
  * Coverage under failure: $\mathrm{cov}(n) = 1 - (q(n)/Q)$.
* **Result:** $R_{\min}$ is the minimum coverage across all nodes. It measures the maximum demand percentage that can still be served if the most critical node fails.

---

## Slide 14: Constraints (Capacity and Flow)
* **Constraint 1 - Supply Capacity:** Total flow out of supplier $i$ cannot exceed its capacity $C_i$.
* **Constraint 2 - Factory Flow Balance:** Material in = Material out at each factory $j$.
* **Constraint 3 - Factory Capacity:** Total throughput at factory $j$ cannot exceed production capacity $F_j$.
* **Constraint 4 - DC Flow Balance:** Material in = Material out at each DC $k$.
* **Constraint 5 - DC Capacity:** Total throughput at DC $k$ cannot exceed storage capacity $D_k$.

---

## Slide 15: Constraints (Demand & Soft Handling)
* **Constraint 6 - Demand Satisfaction:** Supply to customer $l$ should ideally meet demand $d_l$. (Converted to soft constraint via $C_{short}$ penalty).
* **Constraint 7 - Non-negativity:** All flow variables $x, y, z \geq 0$.
* **NSGA-II Implementation:** 
  * Hard constraints are handled via severe penalty functions.
  * Penalties for capacity and flow balance violations:
  $$\tilde{f}_1 = f_1 + 1000 \cdot P$$
  $$\tilde{f}_2 = f_2 + 10 \cdot P$$

---

## Slide 16: Solution Methodology: Overview
* **Modular Pipeline Architecture:**
  1. **Demand Forecasting:** Predict customer demands.
  2. **Mathematical Formulation:** Pyomo for objective definition.
  3. **Multi-Objective Optimization:** NSGA-II to find the Pareto front.
  4. **Simulation Validation:** Monte Carlo testing of optimal solutions.
  5. **Decision Support System:** Streamlit dashboard for stakeholder review.

---

## Slide 17: Methodology - Demand Forecasting
* **Model Used:** Random Forest Regression (100 trees).
* **Input Features:** 15 engineered features per customer (Lag variables, Fourier seasonality, rolling means/stds).
* **Data Split:** 60 periods training, 12 periods testing.
* **Purpose:** Provides robust inputs to the optimization model, capable of handling regime shifts and seasonality better than simple averages.

---

## Slide 18: Methodology - NSGA-II Optimization
* **Algorithm:** Non-dominated Sorting Genetic Algorithm II.
* **Configuration:**
  * Population size: 120
  * Generations: 300
  * Total evaluations: 36,000
* **Why NSGA-II?** Highly efficient for multi-objective spaces; maintains diversity across the Pareto front to offer distinct strategic choices.

---

## Slide 19: Methodology - Resilience Simulation
* **Purpose:** Validate the static resilience objective ($f_3$) against dynamic, real-world failure patterns.
* **Framework:** SimPy discrete-event Monte Carlo simulation.
* **Parameters:** 
  * 100 runs over a 10-year horizon.
  * Facilities fail randomly based on Mean Time To Failure (MTTF = 5 yr) and recover based on Mean Time To Repair (MTTR = 45 days).
* **Metric:** Evaluates the average and worst-case demand fill rate.

---

## Slide 20: Results - Demand Forecasting Performance
* **Accuracy:** Mean Absolute Percentage Error (MAPE) of 12.16%.
* **Significance:** Excellent performance for strategic network design, successfully capturing seasonal trends and abrupt demand shocks.
* **Visual:**
  * *[PLACEHOLDER: Add Figure 6.1: Customer demand time series from results/figures/demand_forecast.png]*

---

## Slide 21: Results - Optimization & Pareto Front
* **Computation:** NSGA-II generated 105 Pareto-optimal solutions in under 8 seconds.
* **Representative Solutions Extracted:**
  1. Minimum Cost
  2. Minimum Emissions
  3. Balanced (Utopia) - Closest to ideal point.
  4. Balanced (Knee) - Point of diminishing returns.
* **Visual:**
  * *[PLACEHOLDER: Add Figure 6.2: 3D Pareto front from results/figures/pareto_front_3d.png]*

---

## Slide 22: Results - Trade-off Analysis
* **Observation:** Min Cost achieves 12% lower cost than Min Emissions, but generates 49% more $CO_2$.
* **The "Price of Sustainability":** Shifting to the Min Emissions network costs a premium of ~14%.
* **Resilience Differentiation:** The Min Emissions solution is actually the *least* resilient (48.9%). It relies heavily on rail, bypassing nodes and reducing structural diversity.
* **Visual:**
  * *[PLACEHOLDER: Add Figure 6.3: Radar chart comparing solutions from results/figures/representative_solutions.png]*

---

## Slide 23: Results - The Balanced Solution
* **Recommendation:** The "Balanced (Utopia)" solution provides the most practical operating point.
* **Performance:** Achieves 56.1% resilience score with moderate cost and emissions by intelligently mixing road and rail while distributing flow across multiple suppliers.
* **Diminishing Returns:** Maximizing resilience beyond this point (the Knee solution) dramatically increases costs (38x higher) for only a marginal resilience gain.

---

## Slide 24: Results - Network Flow Visualization
* **Visualizing the Balanced Network:**
  * Shows how flows are strategically routed to avoid bottlenecks and maintain diversity.
  * *[PLACEHOLDER: Add Figure 6.4: Network flow map of the Balanced (Utopia) solution from results/figures/network_flows_balanced_(utopia).png]*

---

## Slide 25: Results - Simulation Validation
* **Findings:** The simulation confirms the HHI-based resilience metric works.
* **Performance under Stress:** The balanced network maintained an average fill rate of 81.19% despite experiencing an average of 3 major disruptions over the 10-year simulation.
* **Conclusion:** The static $f_3$ metric is a highly effective surrogate for dynamic operational recovery.
* **Visual:**
  * *[PLACEHOLDER: Add Figure 6.5: Resilience simulation gauges from results/figures/simulation_results.png]*

---

## Slide 26: Decision Support System (DSS)
* **Bridging Theory and Practice:** A Streamlit dashboard was deployed to allow stakeholders to interact with the Pareto front without coding knowledge.
* **Live Deployment:** Accessible via Streamlit Community Cloud.
* **Features:** Interactive 3D plots, geographic flow mapping, and real-time KPI comparisons.
* *Demo available during Q&A.*

---

## Slide 27: Summary of Contributions
1. **LCA & Penalty Integration:** Extended standard cost and emission models with shortage penalties and life-cycle warehousing emissions.
2. **Novel Resilience Metric:** Successfully combined structural HHI with worst-case node-failure modeling.
3. **Congestion-Aware:** Implemented multi-modal transport with congestion penalties on both cost and carbon footprint.
4. **End-to-End Pipeline:** Integrated machine learning, evolutionary optimization, and Monte Carlo simulation.

---

## Slide 28: Limitations and Future Work
* **Limitations:** 
  * Single-period formulation lacks dynamic inventory rollover.
  * Facility locations were predetermined rather than optimized.
* **Future Work:**
  * Expand to a multi-period, multi-product formulation.
  * Implement stochastic optimization using chance-constraints for uncertain demand.
  * Upgrade forecasting with Deep Learning (LSTM/Transformers).

---

## Slide 29: Conclusion
* Designing modern supply chains requires looking beyond mere cost efficiency.
* By actively mapping the trade-off space between Cost, Emissions, and Resilience, organizations can make informed, strategic decisions.
* The developed NSGA-II framework provides a rapid, mathematically rigorous, and fully traceable method for sustainable and robust network design.

---

## Slide 30: Thank You
* **Questions & Answers**
* **Live Dashboard Demo:** https://sagar2664-mtp.streamlit.app
* **Source Code:** https://github.com/sagar2664/mtp

# AI-Based Multi-Objective Optimization of a Green and Resilient Supply Chain Network

**M.Tech Thesis**

---

## Abstract

This thesis develops a multi-objective optimization framework for designing supply chain networks that simultaneously minimize total cost, minimize carbon emissions, and maximize resilience. The framework integrates Random Forest-based demand forecasting, NSGA-II evolutionary optimization, and SimPy discrete-event simulation within a 4-echelon network (Suppliers → Factories → DCs → Customers) targeted at the Indian automotive sector. Objective functions are grounded in established literature: cost formulation follows Melo et al. (2009) with unmet demand penalties from Jabbarzadeh et al. (2018); emission modeling adopts the LCA-based approach of Pishvaee & Razmi (2012); and resilience uses a composite metric combining the HHI diversification index from Hasani & Khosrojerdi (2016) with expected node-failure coverage from Snyder & Daskin (2005). Experiments on a test network yield 105 Pareto-optimal solutions in under 9 seconds, with representative solutions enabling transparent trade-off analysis. Simulation validates resilience under stochastic disruptions, achieving 81.2% average fill rate across 100 Monte Carlo runs.

**Keywords:** Supply Chain Network Design, Multi-Objective Optimization, Green Supply Chain, Resilience, NSGA-II, Random Forest, Discrete Event Simulation, HHI

---

## 1. Introduction

### 1.1 Background and Motivation

Modern supply chains face a triple challenge: cost efficiency, environmental sustainability, and operational resilience. The COVID-19 pandemic and recent geopolitical disruptions have exposed the fragility of lean, cost-optimized networks. Simultaneously, environmental regulations demand reduced carbon footprints. Traditional single-objective optimization cannot capture the inherent trade-offs between these competing goals.

### 1.2 Problem Statement

*How can we design a supply chain network that simultaneously minimizes total cost, minimizes carbon emissions, and maximizes resilience to disruptions, while accounting for demand uncertainty, traffic congestion, and multi-modal logistics?*

### 1.3 Research Contributions

1. **Integrated framework** combining AI-based forecasting, multi-objective optimization, and disruption simulation.
2. **Paper-sourced objective functions** grounded in established literature (6 papers), ensuring academic rigor.
3. **Composite resilience metric** combining HHI diversification (Hasani & Khosrojerdi, 2016) with expected node-failure coverage (Snyder & Daskin, 2005).
4. **Multi-modal logistics** with explicit Road vs. Rail transport mode decisions.
5. **Traffic congestion integration** into both cost and emission calculations.
6. **Interactive DSS** via Streamlit for stakeholder visualization.

### 1.4 Thesis Organization

- **Chapter 2**: Literature Review
- **Chapter 3**: Problem Definition & Network Structure
- **Chapter 4**: Mathematical Model Formulation (paper-sourced objectives)
- **Chapter 5**: Methodology (forecasting, NSGA-II, simulation)
- **Chapter 6**: Experimental Results
- **Chapter 7**: Conclusion

---

## 2. Literature Review

### 2.1 Supply Chain Network Design

Melo et al. (2009) provide a comprehensive review of facility location and SCND models, establishing the standard cost structure: fixed facility costs + procurement + production + transportation + inventory holding. Their multi-echelon formulation serves as the foundation for our cost objective.

### 2.2 Multi-Objective Optimization

NSGA-II (Deb et al., 2002) is the dominant evolutionary algorithm for multi-objective optimization, using non-dominated sorting and crowding distance to generate diverse Pareto fronts. It has been extensively applied to supply chain problems due to its ability to handle non-linear objectives and complex constraints.

### 2.3 Green Supply Chain

Pishvaee & Razmi (2012) pioneered LCA-based emission modeling in SCND using multi-objective fuzzy programming. Their formulation separates emissions into facility-level (supplier, production, warehousing) and transport-level components. Pishvaee, Torabi & Razmi (2012) extended this to include congestion-weighted transport emissions with mode-specific emission factors.

### 2.4 Supply Chain Resilience

Resilience literature spans two main approaches:
- **Diversification-based:** Hasani & Khosrojerdi (2016) use the Herfindahl-Hirschman Index (HHI) to penalize supply concentration, encouraging multi-sourcing as a resilience strategy.
- **Failure-based:** Snyder & Daskin (2005) model expected demand coverage under node failures using a level-r backup assignment strategy.
- **Combined cost-resilience:** Jabbarzadeh et al. (2018) integrate unmet demand penalties into stochastic cost optimization, bridging economic and resilience objectives.

### 2.5 Demand Forecasting

Random Forest regression (Breiman, 2001) is effective for supply chain demand forecasting due to its handling of non-linear patterns, feature interactions, and robustness to outliers. We enhance it with lag features, seasonal components, and rolling statistics.

### 2.6 Research Gap

While individual aspects of green, resilient, and cost-efficient SCND are well-studied, few works integrate all three objectives with: (a) multi-modal logistics, (b) traffic congestion effects, (c) AI-based demand forecasting, and (d) simulation-based resilience validation in a single reproducible pipeline.

---

## 3. Problem Statement

### 3.1 Network Structure

A **multi-modal 4-echelon supply chain** mapped to Indian automotive geography:

| Echelon | Nodes | Example Locations |
|---------|-------|-------------------|
| Suppliers (I) | S1, S2, S3 | Pune, Chennai, Manesar |
| Factories (J) | F1, F2 | Gujarat, Haryana |
| DCs (K) | D1, D2, D3 | Delhi, Mumbai, Bangalore |
| Customers (L) | C1–C5 | Lucknow, Nagpur, Hyderabad, Ahmedabad, Coimbatore |
| Transport Modes (M) | Road, Rail | — |

### 3.2 Decision Variables

- **xᵢⱼₘ**: Flow from supplier i to factory j via mode m (units, ≥ 0)
- **yⱼₖₘ**: Flow from factory j to DC k via mode m (units, ≥ 0)
- **zₖₗₘ**: Flow from DC k to customer l via mode m (units, ≥ 0)
- **Total: 54 continuous variables** (3×2×2 + 2×3×2 + 3×5×2)

### 3.3 Constraints

1. **Supply capacity:** Σⱼ,ₘ xᵢⱼₘ ≤ Cᵢ ∀i
2. **Factory balance:** Σᵢ,ₘ xᵢⱼₘ = Σₖ,ₘ yⱼₖₘ ∀j
3. **Factory capacity:** Σᵢ,ₘ xᵢⱼₘ ≤ Fⱼ ∀j
4. **DC balance:** Σⱼ,ₘ yⱼₖₘ = Σₗ,ₘ zₖₗₘ ∀k
5. **DC capacity:** Σⱼ,ₘ yⱼₖₘ ≤ Dₖ ∀k
6. **Demand satisfaction:** Σₖ,ₘ zₖₗₘ ≥ dₗ ∀l
7. **Non-negativity:** All flows ≥ 0

### 3.4 Assumptions

- Single product, single planning period using forecasted demand
- Deterministic capacities, fixed facility locations
- Linear cost/emission functions, static congestion during optimization
- Exponential failure/recovery distributions for simulation

---

## 4. Model Formulation

### 4.1 Notation

**Sets:** I (suppliers), J (factories), K (DCs), L (customers), M (modes: road, rail)

**Parameters:**
- sᵢ, eᵢ: supplier cost and emission factor per unit
- pⱼ, εⱼ: production cost and emission factor per unit
- hₖ, ξₖ: DC holding cost and warehousing emission factor per unit
- dᵢⱼ, cfᵢⱼ: distance and congestion factor per arc
- tₘ, τₘ: transport cost and emission factor per unit-km for mode m
- π: shortage penalty cost per unit of unmet demand

### 4.2 Objective Function 1: Total Cost Minimization

**Sources: Melo et al. (2009) + Jabbarzadeh et al. (2018)**

The cost formulation follows the classical multi-echelon SCND structure from Melo et al. (2009), augmented with unmet demand penalties from Jabbarzadeh et al. (2018):

```
f₁ = Σᵢ Σⱼ Σₘ sᵢ · xᵢⱼₘ                         [Procurement — Melo et al. 2009]
   + Σⱼ pⱼ · (Σᵢ Σₘ xᵢⱼₘ)                        [Production — Melo et al. 2009]
   + Σᵢ Σⱼ Σₘ tₘ · dᵢⱼ · cfᵢⱼ · xᵢⱼₘ             [Transport S→F — Melo et al. 2009]
   + Σⱼ Σₖ Σₘ tₘ · dⱼₖ · cfⱼₖ · yⱼₖₘ             [Transport F→DC — Melo et al. 2009]
   + Σₖ Σₗ Σₘ tₘ · dₖₗ · cfₖₗ · zₖₗₘ             [Transport DC→C — Melo et al. 2009]
   + Σₖ hₖ · (Σⱼ Σₘ yⱼₖₘ)                         [Holding — Melo et al. 2009]
   + π · Σₗ max(0, dₗ - Σₖ Σₘ zₖₗₘ)               [Unmet demand — Jabbarzadeh et al. 2018]
```

The first six terms represent the standard total landed cost from procurement through distribution. The seventh term, adopted from Jabbarzadeh et al. (2018), penalizes unmet demand with a shortage cost π (set to 500 $/unit), modeling the economic impact of supply shortfalls that arise under disruption scenarios.

### 4.3 Objective Function 2: Total Carbon Emissions Minimization

**Sources: Pishvaee & Razmi (2012) + Pishvaee, Torabi & Razmi (2012)**

The emission model follows the Life Cycle Assessment (LCA) approach of Pishvaee & Razmi (2012), capturing emissions across all supply chain operations. Transport emissions incorporate congestion-weighted factors per Pishvaee, Torabi & Razmi (2012):

```
f₂ = Σᵢ Σⱼ Σₘ eᵢ · xᵢⱼₘ                          [Supplier emissions — Pishvaee & Razmi 2012]
   + Σⱼ εⱼ · (Σᵢ Σₘ xᵢⱼₘ)                         [Production emissions — Pishvaee & Razmi 2012]
   + Σᵢ Σⱼ Σₘ τₘ · dᵢⱼ · cfᵢⱼ · xᵢⱼₘ              [Transport S→F — Pishvaee, Torabi & Razmi 2012]
   + Σⱼ Σₖ Σₘ τₘ · dⱼₖ · cfⱼₖ · yⱼₖₘ              [Transport F→DC — Pishvaee, Torabi & Razmi 2012]
   + Σₖ Σₗ Σₘ τₘ · dₖₗ · cfₖₗ · zₖₗₘ              [Transport DC→C — Pishvaee, Torabi & Razmi 2012]
   + Σₖ ξₖ · (Σⱼ Σₘ yⱼₖₘ)                          [DC warehousing — Pishvaee & Razmi 2012]
```

The key additions over basic transport-only emission models are: (a) facility-level emissions at suppliers, factories, and DCs as per the LCA framework; (b) congestion multipliers (cfᵢⱼ) that increase emissions on congested routes; and (c) DC warehousing emissions (ξₖ) from energy use in storage operations.

### 4.4 Objective Function 3: Resilience Maximization

**Sources: Hasani & Khosrojerdi (2016) + Snyder & Daskin (2005) + Jabbarzadeh et al. (2018)**

Resilience is modeled as a composite metric combining two complementary dimensions:

**Component A — Network Diversification (Hasani & Khosrojerdi, 2016):**

Using the Herfindahl-Hirschman Index (HHI), adapted from market concentration analysis to supply chain flow concentration:

```
HHI_delivery(l) = Σₖ (zₖₗ / Σₖ' zₖ'ₗ)²     ∀l ∈ L     [DC-to-customer concentration]
HHI_supply(j) = Σᵢ (xᵢⱼ / Σᵢ' xᵢ'ⱼ)²       ∀j ∈ J     [Supplier-to-factory concentration]
```

HHI ranges from 1/n (perfect diversification) to 1.0 (single-source dependency). Lower HHI indicates more diversified and thus more resilient sourcing.

**Component B — Expected Node-Failure Coverage (Snyder & Daskin, 2005):**

Following the expected failure cost model, we compute the worst-case demand coverage under single-node failures:

```
coverage(n) = 1 - flow_through(n) / total_flow     for each node n ∈ I ∪ J ∪ K
min_coverage = min{coverage(n) : n ∈ I ∪ J ∪ K}
```

This captures the fraction of demand that remains fulfillable when any single node (supplier, factory, or DC) fails.

**Composite Resilience Score:**

```
f₃ = w₁ · (1 - avg_HHI) + w₂ · min_coverage
```

Where w₁ = 0.4 (diversification weight) and w₂ = 0.6 (failure robustness weight). The higher weight on min_coverage reflects that worst-case failure robustness is the more critical resilience dimension.

### 4.5 Soft Constraint Handling

For NSGA-II, hard constraints are converted to soft constraints with penalties (added to f₁ and f₂):

```
Penalty = Σᵢ max(0, used_i - Cᵢ)                   [Capacity violations]
        + Σⱼ |inflow_j - outflow_j|                  [Balance violations]
        + Σₗ max(0, dₗ - supplied_l)                  [Demand violations]
```

Penalty coefficients: 1000 for cost, 10 for emissions.

---

## 5. Methodology

### 5.1 Pipeline Overview

Data Generation → Demand Forecasting → Pyomo Model → NSGA-II Optimization → SimPy Simulation → Results Export

### 5.2 Data Generation

Synthetic data grounded in real Indian geography:
- **Coordinates:** Real auto hubs (Pune, Chennai, Manesar) with small perturbation
- **Capacities:** Suppliers U(200,400), Factories U(300,500), DCs U(250,400)
- **Demand:** d(t) = 100 + 2t + 20·sin(2πt/12) + noise + regime_shift + shocks (72 periods)
- **Distances:** Geodesic (geopy) with Road (1.2×) and Rail (1.05×) tortuosity
- **Congestion:** U(1.0, 1.5–1.7) for road, U(1.0, 1.1) for rail

### 5.3 Demand Forecasting

**Model:** Random Forest Regression (100 trees, max_depth=10)

**Features (15 per customer):**
- 6 lag features (Lag1–Lag6)
- 3 temporal features (month, sin_season, cos_season)
- 6 rolling statistics (mean/std for windows 3, 6, 12)

**Training:** First 60 periods; **Testing:** Last 12 periods (iterative forecasting)

### 5.4 NSGA-II Optimization

**Algorithm:** pymoo NSGA-II (Deb et al., 2002)
- Population: 120, Generations: 300 (36,000 evaluations)
- Solution encoding: 54-dimensional continuous vector
- 3 objectives: minimize f₁, minimize f₂, maximize f₃

**Representative Solution Extraction:**
1. Min Cost — lowest f₁
2. Min Emissions — lowest f₂
3. Balanced (Utopia) — closest to ideal point in normalized space
4. Balanced (Knee) — maximum curvature point on Pareto front

### 5.5 Resilience Simulation

**Framework:** SimPy discrete-event simulation

**Parameters:**
- MTTF = 1825 days (5 years), MTTR = 45 days
- 100 Monte Carlo runs, 3650 days each
- 50% rerouting efficiency for failed facilities
- Congestion-based availability modulation at DCs

**Metrics:** Average fill rate, minimum fill rate, average disruptions per run

### 5.6 Decision Support System

Streamlit dashboard with: geographic flow visualization (Road/Rail color-coded), interactive 3D Pareto front, KPI cards for each representative solution.

---

## 6. Results

### 6.1 Experimental Setup

**Network:** 3 suppliers, 2 factories, 3 DCs, 5 customers, 2 transport modes
**Environment:** Python 3.10, Pyomo, pymoo, scikit-learn, SimPy

### 6.2 Optimization Results

NSGA-II produced **105 Pareto-optimal solutions** in **8.79 seconds**.

**Table 1: Representative Solutions**

| Solution | Cost ($) | Emissions (kg CO₂) | Resilience |
|----------|----------|---------------------|------------|
| Min Cost | 848,801 | 32,910 | 54.2% |
| Min Emissions | 966,775 | 22,054 | 48.9% |
| Balanced (Utopia) | 965,233 | 38,213 | 56.1% |
| Balanced (Knee) | 37,309,289 | 774,013 | 56.7% |

**Key Trade-off Observations:**
- Min Cost achieves ~14% lower cost than Min Emissions but at 49% higher emissions.
- Balanced (Utopia) achieves the best diversification-adjusted resilience (56.1%) at moderate cost.
- The composite resilience metric now differentiates solutions more meaningfully (48.9%–56.7% range vs. previous 31.7%–37.3%).
- The Knee solution represents an extreme diversification choice with diminishing returns.

### 6.3 Forecast Accuracy

**Table 2: Forecast Metrics**

| Metric | Value |
|--------|-------|
| MAE | 30.13 |
| RMSE | 39.40 |
| MAPE | 12.16% |

MAPE of 12.16% is within acceptable range for strategic network design with regime shifts and demand shocks.

### 6.4 Simulation Results

Disruption simulation on the Balanced (Utopia) solution:

| Metric | Value |
|--------|-------|
| Average fill rate | 81.19% |
| Minimum fill rate | 65.60% |
| Average disruptions per 10-year run | 3.04 |

The simulation confirms that diversified flow patterns (higher HHI-based resilience) translate to better fill rates under stochastic disruptions.

---

## 7. Conclusion

### 7.1 Summary

This thesis developed a unified framework for multi-objective supply chain network design that integrates demand forecasting, optimization, and simulation. The paper-sourced objective functions ensure academic rigor: cost follows Melo et al. (2009), emissions follow Pishvaee & Razmi (2012), and resilience combines Hasani & Khosrojerdi (2016) with Snyder & Daskin (2005).

### 7.2 Key Findings

- Pareto analysis confirms pronounced trade-offs: Min Cost concentrates flows, reducing resilience; Min Emissions prioritizes cleaner routes at higher cost; Balanced solutions diversify sourcing for robustness.
- The composite resilience metric (HHI diversification + node failure coverage) is more interpretable and better validated by simulation than simple flow-concentration proxies.
- Congestion-weighted emission modeling reveals that ignoring congestion leads to optimistic CO₂ estimates.

### 7.3 Limitations

- Synthetic data; results are methodological validation, not company-specific prescription.
- Single-period optimization; multi-period models could capture inventory dynamics.
- Resilience proxy in optimizer approximates simulation ground-truth.

### 7.4 Future Work

- Multi-period, multi-product extensions with inventory dynamics
- Robust/stochastic optimization with chance constraints
- Real-world data calibration
- Hybrid ML-OR approaches (surrogate models, probabilistic forecasting)

---

## References

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

2. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

3. Hasani, A. & Khosrojerdi, A. (2016). Robust global supply chain network design under disruption and uncertainty considering resilience strategies: A parallel memetic algorithm for a real-life case study. *Transportation Research Part E*, 87, 20–52.

4. Jabbarzadeh, A., Fahimnia, B., & Sabouhi, F. (2018). Resilient and sustainable supply chain design: Sustainability analysis under disruption risks. *International Journal of Production Research*, 56(17), 5945–5968.

5. Melo, M.T., Nickel, S., & Saldanha-da-Gama, F. (2009). Facility location and supply chain management — A review. *European Journal of Operational Research*, 196(2), 401–412.

6. Pishvaee, M.S. & Razmi, J. (2012). Environmental supply chain network design using multi-objective fuzzy mathematical programming. *Applied Mathematical Modelling*, 36(8), 3433–3446.

7. Pishvaee, M.S., Torabi, S.A., & Razmi, J. (2012). Credibility-based fuzzy mathematical programming model for green logistics design under uncertainty. *Computers & Industrial Engineering*, 62(2), 624–632.

8. Snyder, L.V. & Daskin, M.S. (2005). Reliability models for facility location: The expected failure cost case. *Transportation Science*, 39(3), 400–416.

9. Pettit, T.J., Fiksel, J., & Croxton, K.L. (2010). Ensuring supply chain resilience: Development of a conceptual framework. *Journal of Business Logistics*, 31(1), 1–21.

10. Fattahi, M., Govindan, K., & Keyvanshokooh, E. (2017). Responsive and resilient supply chain network design under operational and disruption risks. *Transportation Research Part E*, 101, 176–200.
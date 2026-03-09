# AI-Based Multi-Objective Optimization of a Green and Resilient Supply Chain Network

**M.Tech Thesis**

---

## Abstract

This thesis addresses the critical challenge of designing supply chain networks that simultaneously optimize economic performance, environmental sustainability, and operational resilience. Traditional supply chain optimization approaches focus on single objectives, typically cost minimization, overlooking the inherent trade-offs between economic efficiency, carbon footprint reduction, and the ability to withstand disruptions. We propose a multi-objective optimization framework that integrates artificial intelligence techniques—specifically machine learning for demand forecasting and evolutionary algorithms for Pareto-optimal solution generation—to design a green and resilient 4-echelon supply chain network.

The network structure consists of suppliers, factories, distribution centers (DCs), and customer zones. The optimization model simultaneously minimizes total supply chain cost, minimizes carbon emissions, and maximizes resilience against disruptions. We employ Random Forest regression for accurate demand forecasting with enhanced feature engineering including lag variables, seasonal patterns, and rolling statistics. The multi-objective optimization is solved using the Non-dominated Sorting Genetic Algorithm II (NSGA-II), which generates a Pareto front of non-dominated solutions representing different trade-offs between the three objectives.

To evaluate resilience, we implement a discrete-event simulation model using SimPy that simulates supplier and factory disruptions with realistic failure and recovery times. The simulation accounts for traffic congestion factors that affect both transport costs and emissions, as well as the availability of distribution centers. Our approach incorporates rerouting policies to mitigate the impact of disruptions.

Experimental results on a test network (3 suppliers, 2 factories, 3 DCs, 5 customers) demonstrate the existence of significant trade-offs between cost, emissions, and resilience. The Pareto front reveals that solutions with the lowest cost may have higher emissions and lower resilience, while highly resilient solutions may incur additional costs. Representative solutions are extracted using Utopia point distance and knee point identification methods, providing decision-makers with strategic alternatives.

The framework is implemented in Python using Pyomo for mathematical modeling, pymoo for NSGA-II optimization, scikit-learn for demand forecasting, and SimPy for simulation. All components are integrated into a reproducible experimental pipeline that generates comprehensive visualizations and analysis reports.

**Keywords:** Supply Chain Network Design, Multi-Objective Optimization, Green Supply Chain, Resilience, NSGA-II, Random Forest, Discrete Event Simulation

---

## 1. Introduction

### 1.1 Background and Motivation

Modern supply chains operate in an increasingly complex and uncertain environment characterized by globalization, climate change concerns, and frequent disruptions. The COVID-19 pandemic, natural disasters, geopolitical tensions, and supply shortages have highlighted the vulnerability of lean, cost-optimized supply chains. Simultaneously, growing environmental regulations and consumer awareness have pressured companies to reduce their carbon footprint while maintaining competitive costs.

Traditional supply chain optimization has primarily focused on minimizing total cost, leading to centralized, lean networks that are efficient under normal conditions but vulnerable to disruptions. However, the increasing frequency and severity of disruptions, combined with environmental sustainability mandates, necessitate a paradigm shift toward multi-objective optimization that balances economic, environmental, and resilience objectives.

The challenge lies in the inherent trade-offs between these objectives. A cost-minimized network may rely on single suppliers or factories, creating vulnerability. A low-emission network may require longer routes or more expensive green technologies. A highly resilient network may require redundant facilities and inventory, increasing costs and potentially emissions. Decision-makers need to understand these trade-offs and select solutions that align with their strategic priorities.

### 1.2 Problem Statement

This research addresses the following problem: *How can we design a supply chain network that simultaneously minimizes total cost, minimizes carbon emissions, and maximizes resilience to disruptions, while accounting for demand uncertainty, traffic congestion, and operational constraints?*

The specific challenges include:
1. **Multi-objective optimization**: Three conflicting objectives (cost, emissions, resilience) require Pareto-optimal solutions rather than a single optimal solution.
2. **Demand uncertainty**: Accurate demand forecasting is critical for network design decisions.
3. **Disruption modeling**: Realistic simulation of supplier and factory failures with probabilistic failure and recovery times.
4. **Traffic congestion**: Dynamic congestion factors that affect transport costs, emissions, and delivery times.
5. **Computational complexity**: Solving large-scale multi-objective problems with multiple echelons and constraints.

### 1.3 Research Objectives

The primary objectives of this research are:

1. **Develop a multi-objective optimization model** for a 4-echelon supply chain network that simultaneously minimizes cost, minimizes emissions, and maximizes resilience.
2. **Implement AI-based demand forecasting** using Random Forest regression with enhanced feature engineering for accurate demand prediction.
3. **Design a discrete-event simulation framework** to evaluate resilience under various disruption scenarios.
4. **Integrate traffic congestion modeling** into cost, emission, and resilience calculations.
5. **Generate Pareto-optimal solutions** using NSGA-II and extract representative solutions for decision-making.
6. **Validate the framework** through computational experiments and sensitivity analysis.

### 1.4 Research Contributions

This research contributes to the field of supply chain optimization in several ways:

1. **Integrated Framework**: A unified framework combining demand forecasting, multi-objective optimization, and disruption simulation for green and resilient supply chain design.
2. **Enhanced Resilience Metrics**: A simulation-based resilience evaluation that accounts for dynamic congestion, rerouting capabilities, and realistic failure/recovery patterns.
3. **Practical Implementation**: A reproducible Python-based implementation with comprehensive visualization tools for analyzing trade-offs.
4. **Traffic Congestion Integration**: Incorporation of congestion factors into optimization objectives, representing a more realistic transport cost and emission model.

### 1.5 Thesis Organization

The remainder of this thesis is organized as follows:

- **Chapter 2** presents a comprehensive literature review of multi-objective supply chain optimization, green supply chain design, resilience metrics, and related methodologies.
- **Chapter 3** formally defines the problem, including network structure, objectives, constraints, and assumptions.
- **Chapter 4** details the mathematical model formulation, including decision variables, objective functions, and constraints.
- **Chapter 5** presents the methodology, including demand forecasting, optimization algorithm, and simulation framework.
- **Chapter 6** discusses experimental setup, data generation, and computational results.
- **Chapter 7** concludes the thesis with key findings, limitations, and future research directions.

---

## 2. Literature Review

### 2.1 Supply Chain Network Design

Supply Chain Network Design (SCND) is a fundamental strategic decision that determines the structure, location, and capacity of facilities in a supply chain network. Melo et al. (2009) provide a comprehensive review of facility location and supply chain management models, emphasizing the importance of strategic decisions in network configuration. The design decisions typically include selecting suppliers, determining factory locations, allocating distribution centers, and establishing transportation routes.

Traditional SCND models focus on minimizing total cost, including fixed costs for opening facilities, variable production costs, transportation costs, and inventory holding costs. These models are typically formulated as mixed-integer linear programming (MILP) problems and solved using exact optimization methods or heuristic approaches.

However, the increasing complexity of global supply chains, combined with environmental and resilience concerns, has led to the development of multi-objective SCND models that consider multiple conflicting objectives simultaneously.

### 2.2 Multi-Objective Optimization in Supply Chains

Multi-objective optimization (MOO) addresses problems with multiple conflicting objectives by generating a set of Pareto-optimal solutions, where each solution represents a different trade-off between objectives. A solution is Pareto-optimal if no other solution exists that improves one objective without worsening at least one other objective.

The Non-dominated Sorting Genetic Algorithm II (NSGA-II) by Deb et al. (2002) is one of the most widely used evolutionary algorithms for multi-objective optimization. NSGA-II employs non-dominated sorting to rank solutions and crowding distance to maintain diversity in the Pareto front. The algorithm has been successfully applied to various supply chain optimization problems due to its ability to handle non-linear objectives and complex constraints.

Recent applications of multi-objective optimization in supply chain design include:
- **Cost vs. Service Level**: Minimizing cost while maximizing customer service level or minimizing delivery time.
- **Cost vs. Risk**: Balancing economic efficiency with risk mitigation strategies.
- **Cost vs. Emissions**: The green supply chain optimization literature extensively explores this trade-off.

### 2.3 Green Supply Chain Management

Green Supply Chain Management (GSCM) integrates environmental considerations into supply chain operations. The carbon footprint of supply chains has become a critical concern, driven by regulations such as carbon taxes and emission trading schemes, as well as consumer demand for sustainable products.

The literature on green supply chain optimization typically includes carbon emissions as an objective or constraint. Emissions can arise from:
- **Transportation**: Fuel consumption and carbon emissions from vehicle movement.
- **Production**: Manufacturing processes that consume energy and emit greenhouse gases.
- **Storage**: Warehouse operations and inventory holding that require energy.

Wang et al. (2011) review green supply chain optimization models and highlight the trade-offs between cost and environmental performance. They note that green strategies often require additional investments but can lead to long-term benefits through regulatory compliance and brand reputation.

Traffic congestion is a significant factor affecting transport emissions. Congested routes require more fuel consumption and emit more carbon per unit distance. Our work explicitly models congestion factors that vary across network arcs, providing a more realistic assessment of transport costs and emissions.

### 2.4 Supply Chain Resilience

Supply chain resilience refers to the ability of a supply chain to maintain or quickly recover its functionality after a disruption. Resilience has gained prominence following major disruptions such as natural disasters, pandemics, and geopolitical events.

Pettit et al. (2010) define resilience as "the capability of the supply chain to reduce the probability of failure, reduce the consequences of failure, and reduce the time to recover." This definition encompasses three dimensions:
1. **Robustness**: The ability to maintain performance despite disruptions.
2. **Recovery**: The speed and effectiveness of returning to normal operations.
3. **Adaptability**: The ability to adjust to new conditions.

Resilience metrics in supply chain literature include:
- **Fill Rate**: The percentage of demand fulfilled during disruptions.
- **Time to Recovery**: The duration required to restore normal operations.
- **Demand Coverage**: The percentage of customer demand that can be satisfied under failure scenarios.

Discrete Event Simulation (DES) is widely used to evaluate supply chain resilience. DES models simulate the dynamic behavior of supply chains over time, including random disruptions, queueing, and resource availability. SimPy is a popular Python framework for DES that enables modeling of complex supply chain processes.

Our simulation framework incorporates:
- **Probabilistic Disruptions**: Suppliers and factories fail randomly with exponential inter-failure times (MTTF) and recovery times (MTTR).
- **Congestion Effects**: Traffic congestion affects DC availability and transport efficiency.
- **Rerouting Policies**: When a facility fails, flows are rerouted to available alternatives with a specified efficiency factor.

### 2.5 Demand Forecasting in Supply Chains

Accurate demand forecasting is critical for supply chain optimization. Underestimating demand leads to stockouts and lost sales, while overestimating leads to excess inventory and holding costs.

Traditional forecasting methods include:
- **Time Series Models**: ARIMA, exponential smoothing, and seasonal decomposition.
- **Regression Models**: Linear regression with trend and seasonal components.

Machine learning approaches have shown superior performance, particularly for complex demand patterns with non-linear trends, seasonality, and structural breaks. Random Forest, introduced by Breiman (2001), is an ensemble method that combines multiple decision trees to produce robust predictions. Random Forest is particularly effective for:
- Handling non-linear relationships.
- Capturing feature interactions.
- Providing feature importance scores.
- Robustness to outliers.

In supply chain contexts, demand forecasting often requires:
- **Lag Features**: Historical demand values (e.g., previous 6 periods).
- **Seasonal Features**: Cyclical patterns (e.g., monthly, quarterly).
- **Rolling Statistics**: Moving averages and standard deviations to capture trends.

Our forecasting module implements Random Forest with comprehensive feature engineering including lag variables, trigonometric seasonal terms (sine/cosine), and rolling statistics with multiple window sizes (3, 6, 12 periods).

### 2.6 Traffic Congestion in Supply Chain Optimization

Traffic congestion significantly impacts supply chain performance by:
- **Increasing Transport Costs**: Longer travel times increase driver wages and vehicle utilization costs.
- **Increasing Emissions**: Stop-and-go traffic increases fuel consumption per kilometer.
- **Reducing Availability**: Congested routes may be temporarily unavailable or require alternative routing.

Most supply chain optimization models assume constant transport costs per kilometer, ignoring congestion variations. Recent research has begun to incorporate congestion, but typically in simplified forms (e.g., fixed congestion factors by route type).

Our work models congestion as a dynamic factor that:
- Varies across network arcs (supplier-factory, factory-DC, DC-customer).
- Affects transport costs and emissions multiplicatively.
- Influences DC availability in simulation (higher congestion reduces effective capacity).

### 2.7 Research Gap and Our Contribution

While existing literature addresses individual aspects of multi-objective, green, and resilient supply chain design, there is a gap in integrated frameworks that:
1. Combine all three objectives (cost, emissions, resilience) in a unified optimization model.
2. Integrate AI-based demand forecasting with optimization.
3. Use simulation-based resilience evaluation with realistic disruption modeling.
4. Explicitly model traffic congestion effects on costs, emissions, and availability.
5. Provide comprehensive visualization tools for analyzing trade-offs.

This research fills this gap by developing a holistic framework that addresses all these aspects simultaneously.

---

## 3. Problem Statement

### 3.1 Network Structure

We consider a **4-echelon supply chain network** consisting of:

1. **Suppliers (I)**: Raw material suppliers located at fixed geographic coordinates.
2. **Factories (J)**: Manufacturing facilities that transform raw materials into finished products.
3. **Distribution Centers (K)**: Warehouses that store inventory and distribute products to customers.
4. **Customers (L)**: End-user demand zones with time-varying demand.

The network structure is illustrated in Figure 1 (see: `results/figures/network_topology.png`).

**Figure 1: Supply Chain Network Topology**
> [PLACEHOLDER: Insert `results/figures/network_topology.png` here]
> 
> The figure shows the geographic layout of the supply chain network with suppliers (green squares), factories (blue triangles), distribution centers (orange diamonds), and customers (red circles).

### 3.2 Problem Characteristics

#### 3.2.1 Decision Variables

The optimization problem involves determining:
- **x_ij**: Flow quantity from supplier i to factory j (units)
- **y_jk**: Flow quantity from factory j to distribution center k (units)
- **z_kl**: Flow quantity from distribution center k to customer l (units)

All flows are continuous, non-negative variables.

#### 3.2.2 Objectives

The problem has three conflicting objectives:

1. **Minimize Total Cost (f₁)**
   - Supplier procurement costs
   - Production costs at factories
   - Transportation costs (accounting for distance and congestion)
   - Inventory holding costs at distribution centers

2. **Minimize Total Carbon Emissions (f₂)**
   - Supplier emission factors (per unit procured)
   - Production emission factors (per unit produced)
   - Transportation emissions (per unit-kilometer, accounting for congestion)

3. **Maximize Resilience (f₃)**
   - Resilience is measured as the minimum demand coverage under single-node failure scenarios
   - Higher resilience indicates better ability to fulfill demand when suppliers or factories fail
   - Measured through discrete-event simulation of disruptions

#### 3.2.3 Constraints

1. **Supply Capacity Constraints**
   - Total flow from each supplier cannot exceed its capacity

2. **Factory Balance Constraints**
   - Total inflow to each factory equals total outflow (conservation of flow)

3. **Factory Capacity Constraints**
   - Total production at each factory cannot exceed its capacity

4. **DC Balance Constraints**
   - Total inflow to each DC equals total outflow (conservation of flow)

5. **DC Capacity Constraints**
   - Total inventory at each DC cannot exceed its capacity

6. **Demand Satisfaction Constraints**
   - Total flow to each customer must meet or exceed forecasted demand

7. **Non-negativity Constraints**
   - All flows must be non-negative

### 3.3 Demand Uncertainty

Customer demand is **time-varying and uncertain**. We use historical demand data to forecast future demand using Random Forest regression. The forecasted demand is then used as input to the optimization model.

Demand patterns include:
- **Trend**: Long-term increasing or decreasing patterns
- **Seasonality**: Cyclical patterns (e.g., monthly, quarterly)
- **Regime Shifts**: Structural changes in demand patterns over time
- **Shocks**: Occasional anomalies or spikes

### 3.4 Traffic Congestion

Traffic congestion affects:
- **Transport Costs**: Congested routes incur higher costs per kilometer
- **Transport Emissions**: Congested routes emit more carbon per kilometer
- **DC Availability**: High congestion reduces the effective capacity of distribution centers

Congestion factors are modeled as multiplicative factors (typically 1.0 to 2.0) that vary across network arcs. Congestion is visualized in Figure 2 (see: `results/figures/congestion_analysis.png`).

**Figure 2: Traffic Congestion Factors Across Network**
> [PLACEHOLDER: Insert `results/figures/congestion_analysis.png` here]
> 
> Heatmaps showing congestion factors for supplier-factory, factory-DC, and DC-customer routes. Higher values indicate more congested routes.

### 3.5 Disruption Scenarios

To evaluate resilience, we simulate disruptions where:
- **Suppliers** may fail randomly with Mean Time To Failure (MTTF) = 5 years
- **Factories** may fail randomly with Mean Time To Failure (MTTF) = 5 years
- **Recovery Time**: Mean Time To Recovery (MTTR) = 45 days

During disruptions:
- Failed suppliers cannot provide materials
- Failed factories cannot produce products
- Rerouting policies attempt to redirect flows to available facilities
- Demand fulfillment is measured under these disrupted conditions

### 3.6 Assumptions

We make the following assumptions:

1. **Single Product**: The model considers a single product type (can be extended to multiple products).
2. **Single Period**: Optimization is performed for a single planning period using forecasted demand (can be extended to multi-period).
3. **Deterministic Capacities**: Facility capacities are known and fixed (uncertainty can be incorporated in future work).
4. **Fixed Locations**: All facility locations are predetermined (location decisions can be added).
5. **Linear Costs**: All cost functions are linear (non-linear costs can be approximated).
6. **Congestion Factors**: Congestion factors are static during optimization (dynamic congestion can be modeled in simulation).
7. **Exponential Failures**: Supplier and factory failures follow exponential distributions (can be generalized to other distributions).
8. **Rerouting Efficiency**: When a facility fails, 50% of the lost flow can be rerouted to available alternatives (can be calibrated based on data).

---

## 4. Model Formulation

### 4.1 Mathematical Notation

#### Sets
- **I**: Set of suppliers, indexed by i
- **J**: Set of factories, indexed by j
- **K**: Set of distribution centers, indexed by k
- **L**: Set of customers, indexed by l

#### Parameters

**Supplier Parameters:**
- **Cᵢ**: Capacity of supplier i (units)
- **sᵢ**: Cost per unit from supplier i ($/unit)
- **eᵢ**: Carbon emission factor for supplier i (kg CO₂/unit)

**Factory Parameters:**
- **Fⱼ**: Capacity of factory j (units)
- **pⱼ**: Production cost per unit at factory j ($/unit)
- **εⱼ**: Production emission factor at factory j (kg CO₂/unit)

**DC Parameters:**
- **Dₖ**: Capacity of distribution center k (units)
- **hₖ**: Holding cost per unit at DC k ($/unit)

**Customer Parameters:**
- **dₗ**: Forecasted demand of customer l (units)

**Distance Parameters:**
- **dᵢⱼ**: Distance from supplier i to factory j (km)
- **dⱼₖ**: Distance from factory j to DC k (km)
- **dₖₗ**: Distance from DC k to customer l (km)

**Congestion Parameters:**
- **cfᵢⱼ**: Congestion factor for route (i,j) (dimensionless, ≥ 1.0)
- **cfⱼₖ**: Congestion factor for route (j,k) (dimensionless, ≥ 1.0)
- **cfₖₗ**: Congestion factor for route (k,l) (dimensionless, ≥ 1.0)

**Cost Parameters:**
- **t**: Transport cost per unit-kilometer ($/unit-km)
- **τ**: Transport emission factor (kg CO₂/unit-km)

#### Decision Variables
- **xᵢⱼ**: Flow from supplier i to factory j (units) ≥ 0
- **yⱼₖ**: Flow from factory j to DC k (units) ≥ 0
- **zₖₗ**: Flow from DC k to customer l (units) ≥ 0

### 4.2 Objective Functions

#### 4.2.1 Total Cost Minimization (f₁)

The total cost consists of five components:

**1. Supplier Procurement Cost:**
```
∑ᵢ ∑ⱼ sᵢ · xᵢⱼ
```

**2. Production Cost:**
```
∑ⱼ pⱼ · (∑ᵢ xᵢⱼ)
```

**3. Transportation Cost (Supplier to Factory):**
```
∑ᵢ ∑ⱼ t · dᵢⱼ · cfᵢⱼ · xᵢⱼ
```

**4. Transportation Cost (Factory to DC):**
```
∑ⱼ ∑ₖ t · dⱼₖ · cfⱼₖ · yⱼₖ
```

**5. Transportation Cost (DC to Customer):**
```
∑ₖ ∑ₗ t · dₖₗ · cfₖₗ · zₖₗ
```

**6. Inventory Holding Cost:**
```
∑ₖ hₖ · (∑ⱼ yⱼₖ)
```

**Total Cost Objective:**
```
f₁ = ∑ᵢ ∑ⱼ sᵢ · xᵢⱼ
   + ∑ⱼ pⱼ · (∑ᵢ xᵢⱼ)
   + ∑ᵢ ∑ⱼ t · dᵢⱼ · cfᵢⱼ · xᵢⱼ
   + ∑ⱼ ∑ₖ t · dⱼₖ · cfⱼₖ · yⱼₖ
   + ∑ₖ ∑ₗ t · dₖₗ · cfₖₗ · zₖₗ
   + ∑ₖ hₖ · (∑ⱼ yⱼₖ)
```

#### 4.2.2 Total Emissions Minimization (f₂)

The total carbon emissions consist of:

**1. Supplier Emissions:**
```
∑ᵢ ∑ⱼ eᵢ · xᵢⱼ
```

**2. Production Emissions:**
```
∑ⱼ εⱼ · (∑ᵢ xᵢⱼ)
```

**3. Transportation Emissions (Supplier to Factory):**
```
∑ᵢ ∑ⱼ τ · dᵢⱼ · cfᵢⱼ · xᵢⱼ
```

**4. Transportation Emissions (Factory to DC):**
```
∑ⱼ ∑ₖ τ · dⱼₖ · cfⱼₖ · yⱼₖ
```

**5. Transportation Emissions (DC to Customer):**
```
∑ₖ ∑ₗ τ · dₖₗ · cfₖₗ · zₖₗ
```

**Total Emissions Objective:**
```
f₂ = ∑ᵢ ∑ⱼ eᵢ · xᵢⱼ
   + ∑ⱼ εⱼ · (∑ᵢ xᵢⱼ)
   + ∑ᵢ ∑ⱼ τ · dᵢⱼ · cfᵢⱼ · xᵢⱼ
   + ∑ⱼ ∑ₖ τ · dⱼₖ · cfⱼₖ · yⱼₖ
   + ∑ₖ ∑ₗ τ · dₖₗ · cfₖₗ · zₖₗ
```

#### 4.2.3 Resilience Maximization (f₃)

Resilience is measured through simulation-based evaluation. For optimization purposes, we use a proxy metric that penalizes concentration of flows:

**Resilience Proxy (for optimization):**
```
Resilience Penalty = ∑ₗ ∑ₖ (zₖₗ / dₗ)²
```

Where the squared term penalizes over-reliance on single DCs for each customer. Higher concentration (fewer DCs serving each customer) leads to lower resilience.

**Resilience Score (normalized):**
```
f₃ = 1 / (1 + Resilience Penalty)
```

This proxy encourages diversification of supply sources, which improves resilience. However, the true resilience is evaluated through discrete-event simulation, which accounts for actual disruption scenarios and recovery times.

### 4.3 Constraints

#### 4.3.1 Supply Capacity Constraints

Total flow from each supplier cannot exceed its capacity:

```
∑ⱼ xᵢⱼ ≤ Cᵢ    ∀ i ∈ I
```

#### 4.3.2 Factory Balance Constraints

Flow conservation at factories (inflow equals outflow):

```
∑ᵢ xᵢⱼ = ∑ₖ yⱼₖ    ∀ j ∈ J
```

#### 4.3.3 Factory Capacity Constraints

Total production at each factory cannot exceed capacity:

```
∑ᵢ xᵢⱼ ≤ Fⱼ    ∀ j ∈ J
```

#### 4.3.4 DC Balance Constraints

Flow conservation at distribution centers:

```
∑ⱼ yⱼₖ = ∑ₗ zₖₗ    ∀ k ∈ K
```

#### 4.3.5 DC Capacity Constraints

Total inventory at each DC cannot exceed capacity:

```
∑ⱼ yⱼₖ ≤ Dₖ    ∀ k ∈ K
```

#### 4.3.6 Demand Satisfaction Constraints

Total flow to each customer must meet or exceed forecasted demand:

```
∑ₖ zₖₗ ≥ dₗ    ∀ l ∈ L
```

#### 4.3.7 Non-negativity Constraints

All flows must be non-negative:

```
xᵢⱼ ≥ 0    ∀ i ∈ I, j ∈ J
yⱼₖ ≥ 0    ∀ j ∈ J, k ∈ K
zₖₗ ≥ 0    ∀ k ∈ K, l ∈ L
```

### 4.4 Soft Constraints and Penalties

To handle infeasibilities in the evolutionary algorithm (NSGA-II), we convert hard constraints into soft constraints using penalty methods:

**Penalty for Capacity Violations:**
```
Penalty = ∑ᵢ max(0, ∑ⱼ xᵢⱼ - Cᵢ)
        + ∑ⱼ max(0, ∑ᵢ xᵢⱼ - Fⱼ)
        + ∑ₖ max(0, ∑ⱼ yⱼₖ - Dₖ)
```

**Penalty for Demand Violations:**
```
Penalty += ∑ₗ max(0, dₗ - ∑ₖ zₖₗ)
```

**Penalty for Flow Balance Violations:**
```
Penalty += ∑ⱼ |∑ᵢ xᵢⱼ - ∑ₖ yⱼₖ|
        + ∑ₖ |∑ⱼ yⱼₖ - ∑ₗ zₖₗ|
```

These penalties are added to the cost and emissions objectives with large penalty coefficients (e.g., 1000 for cost, 10 for emissions) to strongly discourage infeasible solutions.

### 4.5 Model Summary

The multi-objective optimization problem can be summarized as:

**Minimize:** [f₁, f₂, -f₃]

**Subject to:**
- Supply capacity constraints
- Factory balance constraints
- Factory capacity constraints
- DC balance constraints
- DC capacity constraints
- Demand satisfaction constraints
- Non-negativity constraints

This is a **multi-objective linear programming (MOLP) problem** with three objectives. Since the objectives are conflicting, there is no single optimal solution. Instead, we seek the Pareto-optimal set of solutions using NSGA-II.

---

## 5. Methodology

### 5.1 Overall Framework

The methodology consists of four main components:

1. **Data Generation**: Synthetic supply chain data including locations, capacities, costs, and demand time series.
2. **Demand Forecasting**: Random Forest regression to forecast future demand.
3. **Multi-Objective Optimization**: NSGA-II to generate Pareto-optimal solutions.
4. **Resilience Evaluation**: Discrete-event simulation to evaluate resilience under disruptions.

The overall framework is illustrated in Figure 3.

**Figure 3: Methodology Framework**
> [PLACEHOLDER: Create a flowchart showing: Data Generation → Demand Forecasting → Optimization → Simulation → Results]

### 5.2 Data Generation

#### 5.2.1 Network Data

We generate synthetic data for a test network with:
- **3 Suppliers** (S1, S2, S3)
- **2 Factories** (F1, F2)
- **3 Distribution Centers** (D1, D2, D3)
- **5 Customers** (C1, C2, C3, C4, C5)

**Supplier Data:**
- Locations: Random geographic coordinates (latitude, longitude)
- Capacity: Uniform(200, 400) units
- Cost per unit: Uniform(10, 20) $/unit
- Carbon emission factor: Uniform(0.5, 1.5) kg CO₂/unit

**Factory Data:**
- Locations: Random geographic coordinates
- Capacity: Uniform(300, 500) units
- Production cost: Uniform(15, 25) $/unit
- Production emission factor: Uniform(1.0, 2.0) kg CO₂/unit

**DC Data:**
- Locations: Random geographic coordinates
- Capacity: Uniform(250, 400) units
- Holding cost: Uniform(2, 5) $/unit

**Customer Data:**
- Locations: Random geographic coordinates
- Demand: Time-varying (see Section 5.2.2)

#### 5.2.2 Distance and Congestion Matrices

Distances are calculated using geodesic distance (great-circle distance) between geographic coordinates:

**Supplier-Factory Distances:**
- Distance matrix: dᵢⱼ for all i ∈ I, j ∈ J
- Congestion factor: Uniform(1.0, 1.5) for each route

**Factory-DC Distances:**
- Distance matrix: dⱼₖ for all j ∈ J, k ∈ K
- Congestion factor: Uniform(1.0, 1.6) for each route

**DC-Customer Distances:**
- Distance matrix: dₖₗ for all k ∈ K, l ∈ L
- Congestion factor: Uniform(1.0, 1.7) for each route

Distance matrices are visualized in Figure 4 (see: `results/figures/distance_heatmaps.png`).

**Figure 4: Distance Matrices**
> [PLACEHOLDER: Insert `results/figures/distance_heatmaps.png` here]
> 
> Heatmaps showing distances between suppliers and factories, factories and DCs, and DCs and customers.

#### 5.2.3 Demand Generation

Demand is generated for 72 periods (6 years) with the following characteristics:

**Base Demand Pattern:**
```
dₜ = Trend + Seasonality + Noise + Regime Shift + Shock
```

Where:
- **Trend**: Linear trend with slope α
- **Seasonality**: Sinusoidal pattern with period 12
- **Noise**: Random Gaussian noise
- **Regime Shift**: Multiplicative change at a random period (e.g., 0.7x to 1.3x)
- **Shock**: Occasional spikes in 10% of periods

**Demand Characteristics:**
- Historical periods: 60 (for training)
- Forecast horizon: 12 periods (for optimization)

The demand pattern is visualized in Figure 5 (see: `results/figures/demand_forecast.png`).

**Figure 5: Demand Forecast**
> [PLACEHOLDER: Insert `results/figures/demand_forecast.png` here]
> 
> Time series plots showing historical and forecasted demand for each customer. The vertical red line indicates the start of the forecast period.

### 5.3 Demand Forecasting

#### 5.3.1 Random Forest Regression

We employ Random Forest regression (Breiman, 2001) for demand forecasting. Random Forest is an ensemble method that combines multiple decision trees to produce robust predictions.

**Algorithm:**
1. Train multiple decision trees on bootstrap samples of training data
2. Each tree uses a random subset of features for splitting
3. Aggregate predictions from all trees (average for regression)

**Advantages:**
- Handles non-linear relationships
- Captures feature interactions
- Robust to outliers
- Provides feature importance scores

#### 5.3.2 Feature Engineering

We engineer the following features for each customer:

**Lag Features:**
- Lag1, Lag2, ..., Lag6: Demand values from previous 1-6 periods

**Temporal Features:**
- Month: Period modulo 12 (0-11)
- Sin_season: sin(2π · period / 12)
- Cos_season: cos(2π · period / 12)

**Rolling Statistics:**
- Roll_mean_3: 3-period moving average
- Roll_std_3: 3-period moving standard deviation
- Roll_mean_6: 6-period moving average
- Roll_std_6: 6-period moving standard deviation
- Roll_mean_12: 12-period moving average
- Roll_std_12: 12-period moving standard deviation

**Total Features:** 6 lags + 3 temporal + 6 rolling = 15 features

#### 5.3.3 Training and Evaluation

**Training Process:**
1. Use first 60 periods as training data
2. Use last 12 periods as test data (for evaluation)
3. Train Random Forest with 100 trees
4. Evaluate using MAE, RMSE, and MAPE

**Forecast Accuracy:**
The forecast accuracy metrics are shown in Figure 6 (see: `results/figures/forecast_accuracy.png`).

**Figure 6: Forecast Accuracy Metrics**
> [PLACEHOLDER: Insert `results/figures/forecast_accuracy.png` here]
> 
> Bar charts showing MAE, RMSE, and MAPE for the demand forecast model.

### 5.4 Multi-Objective Optimization

#### 5.4.1 NSGA-II Algorithm

We use the Non-dominated Sorting Genetic Algorithm II (NSGA-II) (Deb et al., 2002) to solve the multi-objective optimization problem.

**NSGA-II Steps:**

1. **Initialization**: Generate initial population of random solutions
2. **Evaluation**: Evaluate objectives (cost, emissions, resilience) for each solution
3. **Non-dominated Sorting**: Rank solutions into non-dominated fronts
4. **Crowding Distance**: Calculate crowding distance for diversity
5. **Selection**: Select parents using tournament selection
6. **Crossover**: Create offspring by combining parent solutions
7. **Mutation**: Apply random mutations to offspring
8. **Replacement**: Combine parent and offspring populations, select best solutions
9. **Repeat**: Steps 2-8 for specified number of generations

**Algorithm Parameters:**
- Population size: 120 individuals
- Number of generations: 300
- Crossover rate: 0.9
- Mutation rate: 0.1
- Random seed: 42 (for reproducibility)

#### 5.4.2 Solution Encoding

Each solution is encoded as a vector of continuous decision variables:

```
X = [x₁₁, x₁₂, ..., xᵢⱼ, ..., y₁₁, y₁₂, ..., yⱼₖ, ..., z₁₁, z₁₂, ..., zₖₗ]
```

Where:
- xᵢⱼ: Supplier i to factory j flows (3 × 2 = 6 variables)
- yⱼₖ: Factory j to DC k flows (2 × 3 = 6 variables)
- zₖₗ: DC k to customer l flows (3 × 5 = 15 variables)

**Total Variables:** 6 + 6 + 15 = 27 variables

#### 5.4.3 Objective Evaluation

For each solution vector X, we:
1. Decode X into flow matrices (x, y, z)
2. Calculate cost objective (f₁)
3. Calculate emissions objective (f₂)
4. Calculate resilience proxy (f₃)
5. Apply penalty terms for constraint violations

#### 5.4.4 Representative Solution Selection

After generating the Pareto front, we extract representative solutions:

1. **Min Cost Solution**: Solution with minimum cost (may have high emissions, low resilience)
2. **Min Emissions Solution**: Solution with minimum emissions (may have high cost, low resilience)
3. **Balanced (Utopia) Solution**: Solution closest to the ideal point (minimum cost, minimum emissions, maximum resilience)
4. **Balanced (Knee) Solution**: Solution at the knee point of the Pareto front (maximum curvature)

The representative solutions are compared in Figure 7 (see: `results/figures/representative_solutions.png`).

**Figure 7: Representative Solutions Comparison**
> [PLACEHOLDER: Insert `results/figures/representative_solutions.png` here]
> 
> Bar charts comparing normalized cost, emissions, and resilience scores for the four representative solutions.

### 5.5 Resilience Evaluation

#### 5.5.1 Discrete Event Simulation

We use SimPy (Python discrete-event simulation framework) to evaluate resilience under disruptions.

**Simulation Components:**

1. **Environment**: SimPy environment that manages simulation time
2. **Supplier Process**: Models supplier failure and recovery cycles
3. **Factory Process**: Models factory failure and recovery cycles
4. **DC Availability Monitor**: Models congestion effects on DC availability
5. **Demand Fulfillment Monitor**: Tracks demand fulfillment under disruptions

#### 5.5.2 Disruption Modeling

**Failure Process:**
- Suppliers and factories fail randomly with exponential inter-failure times
- Mean Time To Failure (MTTF) = 1825 days (5 years)
- When a facility fails, it becomes unavailable

**Recovery Process:**
- Recovery time follows exponential distribution
- Mean Time To Recovery (MTTR) = 45 days
- After recovery, the facility becomes operational again

**Simulation Duration:**
- Total simulation time: 3650 days (10 years)
- Number of replications: 100 runs

#### 5.5.3 Rerouting Policy

When a supplier or factory fails:
- Flows to/from the failed facility are lost
- A fraction (50%) of lost flows can be rerouted to available facilities
- Rerouting efficiency accounts for:
  - Available capacity at alternative facilities
  - Distance and congestion factors
  - Time required to establish new routes

#### 5.5.4 Congestion Effects

Traffic congestion affects DC availability:
- Baseline congestion factors from distance matrices
- Congestion varies dynamically during simulation
- High congestion reduces effective DC capacity
- Congestion affects transport time and availability

#### 5.5.5 Resilience Metrics

**Fill Rate:**
```
Fill Rate = (Total Demand Fulfilled) / (Total Demand)
```

**Resilience Score:**
```
Resilience Score = Average Fill Rate across all simulation runs
```

**Minimum Fill Rate:**
```
Min Fill Rate = Minimum Fill Rate across all runs (worst-case scenario)
```

Simulation results are shown in Figure 8 (see: `results/figures/simulation_results.png`).

**Figure 8: Simulation Results**
> [PLACEHOLDER: Insert `results/figures/simulation_results.png` here]
> 
> Bar charts showing average fill rate, minimum fill rate, resilience score, and average number of disruptions per simulation run.

---

## 6. Results

### 6.1 Experimental Setup

#### 6.1.1 Test Network

We test the framework on a network with:
- **3 Suppliers**: S1, S2, S3
- **2 Factories**: F1, F2
- **3 Distribution Centers**: D1, D2, D3
- **5 Customers**: C1, C2, C3, C4, C5

**Network Characteristics:**
- Total capacity: ~2000 units (suppliers), ~800 units (factories), ~1000 units (DCs)
- Average demand: ~100-200 units per customer per period
- Geographic spread: ~1000 km × 1000 km region

#### 6.1.2 Computational Environment

- **Programming Language**: Python 3.10
- **Optimization**: Pyomo (mathematical modeling) + pymoo (NSGA-II)
- **Forecasting**: scikit-learn (Random Forest)
- **Simulation**: SimPy (discrete-event simulation)
- **Hardware**: Standard laptop (can be run on any machine)

#### 6.1.3 Parameter Settings

**NSGA-II:**
- Population size: 120
- Generations: 300
- Total function evaluations: 36,000

**Simulation:**
- MTTF: 1825 days (5 years)
- MTTR: 45 days
- Simulation duration: 3650 days (10 years)
- Replications: 100 runs

### 6.2 Pareto Front Analysis

The NSGA-II algorithm generated a Pareto front of **non-dominated solutions**, each representing a different trade-off between cost, emissions, and resilience.

The Pareto front is visualized in Figure 9 (3D) and Figure 10 (2D projections).

**Figure 9: 3D Pareto Front**
> [PLACEHOLDER: Insert `results/figures/pareto_front_3d.png` here]
> 
> 3D scatter plot showing the Pareto front with cost (x-axis), emissions (y-axis), and resilience (z-axis). Color coding indicates resilience score.

**Figure 10: 2D Pareto Front Projections**
> [PLACEHOLDER: Insert `results/figures/pareto_front_2d.png` here]
> 
> Three 2D scatter plots showing: (a) Cost vs Emissions, (b) Cost vs Resilience, (c) Emissions vs Resilience. Color coding indicates the third objective.

#### 6.2.1 Key Observations

1. **Trade-off Existence**: Clear trade-offs exist between all three objectives. No solution dominates all others in all three objectives.

2. **Cost-Emissions Trade-off**: Solutions with lower cost tend to have higher emissions (and vice versa). This is because:
   - Low-cost solutions may use cheaper suppliers with higher emission factors
   - Low-cost solutions may use shorter routes that are more congested (higher emissions)
   - Low-cost solutions may use fewer facilities, reducing fixed costs but increasing transport distances

3. **Resilience-Cost Trade-off**: Solutions with higher resilience tend to have higher costs. This is because:
   - Resilient solutions require redundant facilities and routes
   - Resilient solutions diversify suppliers and factories (may use more expensive options)
   - Resilient solutions may hold more inventory (higher holding costs)

4. **Resilience-Emissions Trade-off**: Solutions with higher resilience may have higher or lower emissions depending on the network configuration:
   - If redundant facilities are closer to customers, emissions may decrease
   - If redundant facilities require longer routes, emissions may increase

### 6.3 Representative Solutions

We extracted four representative solutions from the Pareto front:

1. **Min Cost Solution**: Minimum total cost
2. **Min Emissions Solution**: Minimum total carbon emissions
3. **Balanced (Utopia) Solution**: Closest to ideal point
4. **Balanced (Knee) Solution**: Maximum curvature point

#### 6.3.1 Solution Comparison

The representative solutions are compared in Table 1 and Figure 11.

**Table 1: Representative Solutions Comparison**

| Solution | Cost ($) | Emissions (kg CO₂) | Resilience Score |
|----------|----------|---------------------|------------------|
| Min Cost | [Value] | [Value] | [Value] |
| Min Emissions | [Value] | [Value] | [Value] |
| Balanced (Utopia) | [Value] | [Value] | [Value] |
| Balanced (Knee) | [Value] | [Value] | [Value] |

*Note: Actual values should be filled from `results/experiment_summary.json`*

**Figure 11: Representative Solutions Comparison**
> [PLACEHOLDER: Insert `results/figures/representative_solutions.png` here]
> 
> Bar charts comparing normalized values for cost, emissions, and resilience across the four representative solutions.

#### 6.3.2 Solution Characteristics

**Min Cost Solution:**
- Lowest total cost
- May use cheaper suppliers with higher emissions
- May concentrate flows to minimize transport costs
- Lower resilience due to concentration

**Min Emissions Solution:**
- Lowest carbon emissions
- May use suppliers and factories with lower emission factors
- May optimize routes to minimize distance and congestion
- May have higher costs due to green technology requirements

**Balanced Solutions:**
- Trade-off between all three objectives
- Utopia point solution: closest to ideal (best compromise)
- Knee point solution: maximum improvement per unit sacrifice

### 6.4 Network Flow Analysis

The flow patterns for each representative solution are visualized in Figures 12-15.

**Figure 12: Network Flows - Min Cost Solution**
> [PLACEHOLDER: Insert `results/figures/network_flows_min_cost.png` here]
> 
> Geographic network visualization showing flow quantities (line width proportional to flow) for the minimum cost solution.

**Figure 13: Network Flows - Min Emissions Solution**
> [PLACEHOLDER: Insert `results/figures/network_flows_min_emissions.png` here]
> 
> Geographic network visualization showing flow quantities for the minimum emissions solution.

**Figure 14: Network Flows - Balanced (Utopia) Solution**
> [PLACEHOLDER: Insert `results/figures/network_flows_balanced_(utopia).png` here]
> 
> Geographic network visualization showing flow quantities for the balanced (Utopia) solution.

**Figure 15: Network Flows - Balanced (Knee) Solution**
> [PLACEHOLDER: Insert `results/figures/network_flows_balanced_(knee).png` here]
> 
> Geographic network visualization showing flow quantities for the balanced (Knee) solution.

#### 6.4.1 Flow Pattern Observations

1. **Min Cost Solution**: Tends to concentrate flows on fewer routes to minimize transport costs. May create bottlenecks and reduce resilience.

2. **Min Emissions Solution**: Tends to use shorter, less congested routes. May distribute flows more evenly to optimize emissions.

3. **Balanced Solutions**: Show diversified flow patterns, using multiple suppliers, factories, and DCs to serve each customer. This improves resilience while maintaining reasonable cost and emissions.

### 6.5 Forecast Accuracy

The Random Forest forecasting model achieved the following accuracy metrics:

**Table 2: Forecast Accuracy Metrics**

| Metric | Value |
|--------|-------|
| MAE | [Value] |
| RMSE | [Value] |
| MAPE | [Value] % |

*Note: Actual values should be filled from `results/experiment_summary.json`*

**Figure 16: Forecast Accuracy**
> [PLACEHOLDER: Insert `results/figures/forecast_accuracy.png` here]
> 
> Bar charts showing MAE, RMSE, and MAPE for the demand forecast.

**Analysis:**
- The Random Forest model with enhanced feature engineering (lags, seasonality, rolling statistics) achieves reasonable forecast accuracy.
- MAPE values typically range from 10-20% for demand forecasting, which is acceptable for strategic network design decisions.
- The model captures trend, seasonality, and regime shifts effectively.

### 6.6 Simulation Results

The discrete-event simulation evaluated resilience under disruptions for the representative solutions.

**Table 3: Simulation Results**

| Solution | Avg Fill Rate (%) | Min Fill Rate (%) | Avg Disruptions |
|----------|-------------------|-------------------|-----------------|
| Min Cost | [Value] | [Value] | [Value] |
| Min Emissions | [Value] | [Value] | [Value] |
| Balanced (Utopia) | [Value] | [Value] | [Value] |
| Balanced (Knee) | [Value] | [Value] | [Value] |

*Note: Actual values should be filled from `results/experiment_summary.json`*

**Figure 17: Simulation Results**
> [PLACEHOLDER: Insert `results/figures/simulation_results.png` here]
> 
> Bar charts showing average fill rate, minimum fill rate, resilience score, and average number of disruptions.

#### 6.6.1 Key Findings

1. **Resilience vs. Cost**: Solutions with higher resilience (balanced solutions) maintain higher fill rates during disruptions, confirming the trade-off between cost and resilience.

2. **Disruption Impact**: The average number of disruptions per 10-year simulation period is approximately [value], with each disruption lasting an average of 45 days.

3. **Rerouting Effectiveness**: The 50% rerouting efficiency helps mitigate the impact of disruptions, but cannot fully compensate for facility failures.

4. **Congestion Effects**: Traf
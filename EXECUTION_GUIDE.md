# Step-by-Step Execution Guide

## Project Overview
**Green Resilient Supply Chain Network - Multi-Objective Optimization**
- 4-echelon network: 3 Suppliers → 2 Factories → 3 DCs → 5 Customers
- 3 Objectives: Minimize Cost, Minimize Emissions, Maximize Resilience
- Uses NSGA-II genetic algorithm for multi-objective optimization

---

## Prerequisites Check

Before starting, ensure you have:
- Python 3.10 installed
- Terminal/Command prompt access
- Internet connection (for package installation)

---

## STEP 1: Navigate to Project Directory

```bash
cd /Users/sagar/Desktop/mtp
```

**Verify you're in the right place:**
```bash
ls -la
```

You should see:
- `src/` directory
- `tests/` directory  
- `scripts/` directory
- `notebooks/` directory
- `requirements.txt`
- `environment.yml`
- `README.md`

---

## STEP 2: Create Virtual Environment (Recommended)

### Option A: Using venv (Python's built-in)

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify activation (you should see (venv) in prompt)
```

### Option B: Using conda

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate green_resilient_scn
```

---

## STEP 3: Install Dependencies

### Using pip:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Verify installations:

```bash
python -c "import numpy, pandas, pyomo, pymoo, sklearn, simpy, matplotlib, seaborn, geopy; print('✓ All packages installed')"
```

**Expected output:** `✓ All packages installed`

If you get import errors, install missing packages individually:
```bash
pip install numpy pandas pyomo pymoo scikit-learn simpy matplotlib seaborn geopy pytest
```

---

## STEP 4: Run Tests (Optional but Recommended)

Verify everything is set up correctly:

```bash
pytest tests/ -v
```

**Expected:** Tests should pass (some may skip if solvers not available - that's OK)

If tests fail, check error messages and install missing dependencies.

---

## STEP 5: Run the Complete Experiment

### Quick Run (Recommended):

```bash
bash scripts/run_all.sh
```

This will:
1. Generate 4-echelon supply chain data
2. Generate distance matrices
3. Forecast demand using Random Forest
4. Create multi-objective optimization model
5. Solve using NSGA-II (takes 5-15 minutes)
6. Evaluate resilience under disruptions
7. Export all results to `./results/`

### What to Expect:

```
==========================================
Green Resilient Supply Chain Network
Multi-Objective Optimization (NSGA-II)
==========================================

STEP 1: Generating 4-echelon supply chain data
  ✓ Generated 3 suppliers
  ✓ Generated 2 factories
  ✓ Generated 3 distribution centers
  ✓ Generated 5 customers

STEP 2: Generating distance matrices
  ✓ Supplier→Factory distances: 6 pairs
  ✓ Factory→DC distances: 6 pairs
  ✓ DC→Customer distances: 15 pairs

STEP 3: Forecasting demand using Random Forest
  ✓ Forecast MAPE: X.XX%
  ✓ Forecast MAE: X.XX
  ✓ Forecast RMSE: X.XX

STEP 4: Creating multi-objective optimization model
  ✓ Model created with 3 suppliers, 2 factories
  ✓ 3 DCs, 5 customers
  ✓ 3 objectives: Cost, Emissions, Resilience

STEP 5: Solving with NSGA-II (this may take several minutes)
  Running NSGA-II with population=50, generations=100...
  ✓ Found X Pareto-optimal solutions
  ✓ Execution time: XX.XX seconds

Representative Solutions:
Min Cost:
  Cost: $X,XXX.XX
  Emissions: XXX.XX kg CO2
  Resilience: XX.XX%

... (more solutions)

STEP 6: Evaluating resilience under disruptions
  ✓ Resilience score: XX.XX%
  Running disruption simulation (100 runs)...
  ✓ Average fill rate: XX.XX%
  ✓ Minimum fill rate: XX.XX%

STEP 7: Exporting results
  ✓ Saved Pareto front to results/pareto_front.csv
  ✓ Saved flow solutions
  ✓ Saved experiment summary to results/experiment_summary.json

EXPERIMENT COMPLETED SUCCESSFULLY!
```

**Time:** Expect 5-15 minutes for NSGA-II optimization (depends on your computer)

---

## STEP 6: View Results

After execution completes:

### View Summary:
```bash
cat results/experiment_summary.json
```

### View Pareto Front:
```bash
head -20 results/pareto_front.csv
```

### List All Results:
```bash
ls -lh results/
```

### View Flow Solutions:
```bash
# Minimum cost solution
cat results/flows_supplier_factory_min_cost.csv

# Minimum emissions solution
cat results/flows_supplier_factory_min_emissions.csv

# Balanced solution
cat results/flows_supplier_factory_balanced.csv
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pymoo'"

**Solution:**
```bash
pip install pymoo
```

### Issue: "NSGA-II taking too long"

**Solution:** The script uses `n_pop=50, n_gen=100`. For faster execution, edit `scripts/run_all.sh` and reduce:
- `n_pop=30` (smaller population)
- `n_gen=50` (fewer generations)

### Issue: "Error during optimization"

**Check:**
1. All dependencies installed: `pip list | grep pymoo`
2. Python version: `python --version` (should be 3.10+)
3. Virtual environment activated

### Issue: Tests failing

**Solution:** Some tests may fail if solvers aren't available. This is OK - the main functionality (NSGA-II) doesn't require external solvers.

---

## Alternative: Run Individual Components

If you want to run components separately:

### 1. Generate Data Only:
```bash
python -c "
from src.data_gen import generate_supply_chain_data, generate_distance_matrix
data = generate_supply_chain_data(n_suppliers=3, n_factories=2, n_dcs=3, n_customers=5, seed=42)
print('Data generated:', list(data.keys()))
"
```

### 2. Test Forecasting:
```bash
python -c "
from src.data_gen import generate_supply_chain_data
from src.forecasting import forecast_demand_random_forest
data = generate_supply_chain_data(n_periods=36, seed=42)
historical = data['demand'][data['demand']['period'] <= 30]
forecast, metrics = forecast_demand_random_forest(historical, forecast_horizon=6)
print('MAPE:', metrics['MAPE'])
"
```

### 3. Test NSGA-II (small problem):
```bash
python -c "
from src.data_gen import generate_supply_chain_data, generate_distance_matrix
from src.pyomo_model import create_multi_objective_model
from src.optimizer import solve_with_nsga2
data = generate_supply_chain_data(seed=42)
distances = generate_distance_matrix(data['suppliers'], data['factories'], data['dcs'], data['customers'])
demand_avg = data['demand'].groupby('customer_id')['demand'].mean().reset_index()
model = create_multi_objective_model(data['suppliers'], data['factories'], data['dcs'], data['customers'], demand_avg, distances)
result = solve_with_nsga2(model, n_pop=20, n_gen=10, seed=42)
print('Pareto solutions:', len(result['pareto_front']))
"
```

---

## Expected Output Files

After successful execution, `results/` directory contains:

1. **pareto_front.csv** - All Pareto-optimal solutions with cost, emissions, resilience
2. **flows_supplier_factory_*.csv** - Supplier→Factory flows for each solution
3. **flows_factory_dc_*.csv** - Factory→DC flows for each solution
4. **flows_dc_customer_*.csv** - DC→Customer flows for each solution
5. **experiment_summary.json** - Complete experiment summary with all metrics

---

## Performance Tips

- **Faster execution:** Reduce `n_pop` and `n_gen` in `scripts/run_all.sh`
- **Better results:** Increase `n_pop` and `n_gen` (but takes longer)
- **Parallel execution:** NSGA-II in pymoo can use multiple cores automatically

---

## Next Steps

1. Analyze Pareto front to understand trade-offs
2. Compare solutions (min-cost vs min-emissions vs balanced)
3. Visualize results (create custom plots using results CSV files)
4. Modify parameters (change network size, costs, capacities)
5. Experiment with different objectives or constraints

---

## Getting Help

If you encounter issues:
1. Check error messages carefully
2. Verify all dependencies: `pip list`
3. Try running tests: `pytest tests/ -v`
4. Check Python version: `python --version` (should be 3.10+)
5. Ensure virtual environment is activated

---

## Quick Reference Commands

```bash
# Setup
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run everything
bash scripts/run_all.sh

# Run tests
pytest tests/ -v

# View results
cat results/experiment_summary.json
ls -lh results/
```

Good luck with your MTech project! 🚀


# Green Resilient Supply Chain Network (green_resilient_scn)

A reproducible Python project for supply chain optimization and simulation with focus on green and resilient network design.

## Project Structure

```
green_resilient_scn/
├── src/                          # Source modules
│   ├── __init__.py
│   ├── data_gen.py              # Synthetic data generation
│   ├── forecasting.py           # Demand forecasting
│   ├── pyomo_model.py           # Pyomo optimization model
│   ├── optimizer.py             # Solver interface and solution extraction
│   ├── sim.py                   # SimPy discrete event simulation
│   └── analysis.py              # Analysis and visualization
├── notebooks/
│   └── notebook_demo.ipynb      # End-to-end pipeline demonstration
├── tests/                        # Unit tests
│   ├── test_data_gen.py         # Data generation tests
│   ├── test_pyomo_model.py      # Pyomo model tests
│   └── test_optimizer.py         # Optimizer output format tests
├── scripts/
│   └── run_all.sh               # Reproducible experiment runner
├── results/                      # Output directory (generated)
├── requirements.txt              # Python package dependencies
└── environment.yml               # Conda environment file
```

## Requirements

- Python 3.10
- Dependencies listed in `requirements.txt`:
  - numpy
  - pandas
  - pyomo
  - pymoo
  - scikit-learn
  - simpy
  - matplotlib
  - seaborn
  - geopy

## Setup Instructions

### Option 1: Using pip (Local or Codespaces)

1. **Install Python 3.10** (if not already installed):
   ```bash
   python3.10 --version  # Verify version
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Pyomo solver** (GLPK - open source):
   ```bash
   # Linux/macOS
   sudo apt-get install glpk-utils  # Ubuntu/Debian
   brew install glpk                # macOS
   
   # Or use CBC solver (alternative open source solver)
   # Instructions: https://github.com/coin-or/Cbc
   ```

### Option 2: Using Conda (Local or Codespaces)

1. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate environment**:
   ```bash
   conda activate green_resilient_scn
   ```

3. **Install Pyomo solver** (if not included):
   ```bash
   conda install -c conda-forge glpk  # or use CBC
   ```

## Quick Start

**For detailed step-by-step instructions, see [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)**

### Quick Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run complete experiment:**
   ```bash
   bash scripts/run_all.sh
   ```

This runs the full pipeline:
- Generates 4-echelon supply chain data (3 suppliers → 2 factories → 3 DCs → 5 customers)
- Forecasts demand using Random Forest
- Solves multi-objective optimization using NSGA-II
- Evaluates resilience under disruptions
- Exports all results to `./results/`

**Expected runtime:** 5-15 minutes (NSGA-II optimization takes most time)

This will:
1. Generate synthetic supply chain data
2. Generate distance matrices
3. Forecast demand
4. Build and solve optimization model
5. Run simulation
6. Analyze results
7. Generate visualizations
8. Export all results to `./results/`

### Run Tests

```bash
pytest tests/
```

Or with verbose output:

```bash
pytest tests/ -v
```

If tests fail, check the trace:

```bash
pytest tests/ -v --tb=short
```

### Run Jupyter Notebook

1. **Start Jupyter**:
   ```bash
   jupyter notebook notebooks/notebook_demo.ipynb
   ```

   Or using JupyterLab:
   ```bash
   jupyter lab notebooks/notebook_demo.ipynb
   ```

2. **Run all cells** to see the complete end-to-end pipeline.

## Running in GitHub Codespaces

1. **Open the repository in Codespaces**:
   - Navigate to your GitHub repository
   - Click "Code" → "Codespaces" → "Create codespace on main"

2. **In the Codespace terminal, run**:
   ```bash
   # Install system dependencies for GLPK
   sudo apt-get update
   sudo apt-get install -y glpk-utils
   
   # Create virtual environment
   python3.10 -m venv venv
   source venv/bin/activate
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Run the experiment**:
   ```bash
   bash scripts/run_all.sh
   ```

4. **Run tests**:
   ```bash
   pytest tests/
   ```

## Expected Output

After running `bash scripts/run_all.sh`, the `results/` directory will contain:

- `flows_supplier_facility.csv` - Supplier to facility flow decisions
- `flows_facility_customer.csv` - Facility to customer flow decisions
- `inventory_levels.csv` - Inventory levels by facility and period
- `analysis_summary.csv` - Summary analysis metrics
- `simulation_results.csv` - Simulation output statistics
- `experiment_summary.json` - Complete experiment summary
- `network_visualization.png` - Network topology plot
- `demand_forecast.png` - Demand forecasting visualization
- `solution_metrics.png` - Solution metrics and KPIs

## Troubleshooting

### Solver Issues

If you encounter solver errors:

1. **Check solver installation**:
   ```bash
   which glpsol  # Should return path to GLPK solver
   ```

2. **Try alternative solver**:
   - Edit `src/optimizer.py` and change `solver_name='cbc'` (if CBC is installed)
   - Or use `solver_name='cplex'` or `'gurobi'` if licensed solvers are available

3. **For testing without solver**: The tests are designed to work even if solvers are not available - they verify model structure and data formats.

### Import Errors

If you encounter import errors:

1. **Ensure you're in the project root directory**
2. **Check Python path**: The scripts add `src/` to the path automatically
3. **Verify installation**: `pip list | grep -E "(numpy|pandas|pyomo)"`

### Test Failures

If tests fail:

1. **Run with detailed output**:
   ```bash
   pytest tests/ -v --tb=long
   ```

2. **Run specific test file**:
   ```bash
   pytest tests/test_data_gen.py -v
   ```

3. **Check for missing dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

## Notes

- **Data**: This project uses only synthetic data generators - no external data files required
- **Reproducibility**: All random operations use fixed seeds (seed=42)
- **Open Source**: Uses open-source solvers (GLPK) by default - no commercial licenses required
- **Performance**: The demo uses smaller problem sizes for faster execution. Adjust parameters in scripts for larger problems.

## License

This is a scaffold/project template. Modify as needed for your specific use case.


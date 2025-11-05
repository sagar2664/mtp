#!/bin/bash

# Green Resilient Supply Chain Network - Reproducible Experiment Runner
# 4-echelon: Suppliers → Factories → DCs → Customers
# Multi-objective: Cost, Emissions, Resilience (NSGA-II)

set -e  # Exit on error

echo "=========================================="
echo "Green Resilient Supply Chain Network"
echo "Multi-Objective Optimization (NSGA-II)"
echo "=========================================="
echo ""

# Create results directory
RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"
echo "Created results directory: $RESULTS_DIR"
echo ""

# Create Python execution script
RUN_SCRIPT="$RESULTS_DIR/run_experiment.py"
cat > "$RUN_SCRIPT" << 'PYTHON_EOF'
"""Reproducible experiment script for 4-echelon multi-objective optimization."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
from src.data_gen import generate_supply_chain_data, generate_distance_matrix, save_supply_chain_data
from src.forecasting import forecast_demand_random_forest
from src.pyomo_model import create_multi_objective_model
from src.optimizer import solve_with_nsga2, extract_representative_solutions
from src.sim import calculate_resilience_score, run_disruption_simulation

def main():
    print("=" * 60)
    print("STEP 1: Generating 4-echelon supply chain data")
    print("=" * 60)
    data = generate_supply_chain_data(
        n_suppliers=3,
        n_factories=2,
        n_dcs=3,
        n_customers=5,
        n_periods=72,  # longer history to include regime shifts and noise
        seed=42
    )
    
    print(f"  ✓ Generated {len(data['suppliers'])} suppliers")
    print(f"  ✓ Generated {len(data['factories'])} factories")
    print(f"  ✓ Generated {len(data['dcs'])} distribution centers")
    print(f"  ✓ Generated {len(data['customers'])} customers")
    print(f"  ✓ Generated {len(data['demand'])} demand records")
    
    print("\n" + "=" * 60)
    print("STEP 2: Generating distance matrices")
    print("=" * 60)
    distances = generate_distance_matrix(
        data['suppliers'],
        data['factories'],
        data['dcs'],
        data['customers']
    )
    print(f"  ✓ Supplier→Factory distances: {len(distances['supplier_factory'])} pairs")
    print(f"  ✓ Factory→DC distances: {len(distances['factory_dc'])} pairs")
    print(f"  ✓ DC→Customer distances: {len(distances['dc_customer'])} pairs")
    
    # Save raw generated data and distances to ./data
    print("\n" + "=" * 60)
    print("STEP 2.1: Saving generated data to ./data")
    print("=" * 60)
    save_supply_chain_data(data, distances, output_dir='./data')
    print("  ✓ Data saved to ./data (CSV files)")
    
    print("\n" + "=" * 60)
    print("STEP 3: Forecasting demand using Random Forest")
    print("=" * 60)
    historical_periods = 60
    historical_demand = data['demand'][data['demand']['period'] <= historical_periods].copy()
    
    forecast, metrics = forecast_demand_random_forest(
        historical_demand,
        forecast_horizon=12,
        n_lags=6,
        random_state=42
    )
    
    print(f"  ✓ Forecast MAPE: {metrics['MAPE']:.2f}%")
    print(f"  ✓ Forecast MAE: {metrics['MAE']:.2f}")
    print(f"  ✓ Forecast RMSE: {metrics['RMSE']:.2f}")
    
    # Use average demand for optimization (aggregate across periods)
    demand_avg = data['demand'].groupby('customer_id')['demand'].mean().reset_index()
    demand_avg.columns = ['customer_id', 'demand']
    
    print("\n" + "=" * 60)
    print("STEP 4: Creating multi-objective optimization model")
    print("=" * 60)
    model = create_multi_objective_model(
        data['suppliers'],
        data['factories'],
        data['dcs'],
        data['customers'],
        demand_avg,
        distances
    )
    print(f"  ✓ Model created with {len(model.I)} suppliers, {len(model.J)} factories")
    print(f"  ✓ {len(model.K)} DCs, {len(model.L)} customers")
    print(f"  ✓ 3 objectives: Cost, Emissions, Resilience")
    
    print("\n" + "=" * 60)
    print("STEP 5: Solving with NSGA-II (this may take several minutes)")
    print("=" * 60)
    print("  Running NSGA-II with population=50, generations=100...")
    
    try:
        pareto_result = solve_with_nsga2(
            model,
            n_pop=120,
            n_gen=300,
            seed=42
        )
        
        print(f"  ✓ Found {len(pareto_result['pareto_front'])} Pareto-optimal solutions")
        print(f"  ✓ Execution time: {pareto_result['exec_time']:.2f} seconds")
        
        # Extract representative solutions
        representative = extract_representative_solutions(pareto_result, n_solutions=4)
        
        print("\n" + "=" * 60)
        print("Representative Solutions:")
        print("=" * 60)
        for sol in representative:
            print(f"\n{sol['label']}:")
            print(f"  Cost: ${sol['objectives']['cost']:,.2f}")
            print(f"  Emissions: {sol['objectives']['emissions']:.2f} kg CO2")
            print(f"  Resilience: {sol['objectives']['resilience']:.2%}")
        
        # Calculate resilience for one solution
        print("\n" + "=" * 60)
        print("STEP 6: Evaluating resilience under disruptions")
        print("=" * 60)
        
        # Use balanced solution for resilience testing (accept Knee label)
        candidates = [s for s in representative if s.get('label','').startswith('Balanced')]
        balanced_sol = candidates[0] if len(candidates) > 0 else representative[-1]
        flows = balanced_sol['flows']
        
        resilience_score = calculate_resilience_score(
            data['suppliers'],
            data['factories'],
            data['dcs'],
            data['customers'],
            flows
        )
        
        print(f"  ✓ Resilience score: {resilience_score:.2%}")
        
        # Run disruption simulation
        print("\n  Running disruption simulation (100 runs)...")
        sim_results = run_disruption_simulation(
            data['suppliers'],
            data['factories'],
            data['dcs'],
            data['customers'],
            flows,
            distances,
            n_runs=100,
            mttf=1825.0,  # 5 years mean time to failure
            mttr=45.0,    # 45 days mean time to recovery
            seed=42
        )
        
        print(f"  ✓ Average fill rate: {sim_results['average_fill_rate']:.2%}")
        print(f"  ✓ Minimum fill rate: {sim_results['min_fill_rate']:.2%}")
        print(f"  ✓ Average disruptions: {sim_results['average_n_disruptions']:.1f}")
        
        # Save results
        print("\n" + "=" * 60)
        print("STEP 7: Exporting results")
        print("=" * 60)
        
        # Save Pareto front
        pareto_data = []
        for sol in pareto_result['pareto_front']:
            pareto_data.append({
                'solution_id': sol['solution_id'],
                'cost': sol['objectives']['cost'],
                'emissions': sol['objectives']['emissions'],
                'resilience': sol['objectives']['resilience']
            })
        pd.DataFrame(pareto_data).to_csv('./results/pareto_front.csv', index=False)
        print("  ✓ Saved Pareto front to results/pareto_front.csv")
        
        # Save flows for representative solutions
        for sol in representative:
            label = sol['label'].replace(' ', '_').lower()
            
            # Save flows
            x_flows = []
            for (i, j), val in sol['flows']['x'].items():
                if val > 1e-6:
                    x_flows.append({'supplier': i, 'factory': j, 'flow': val})
            if x_flows:
                pd.DataFrame(x_flows).to_csv(f'./results/flows_supplier_factory_{label}.csv', index=False)
            
            y_flows = []
            for (j, k), val in sol['flows']['y'].items():
                if val > 1e-6:
                    y_flows.append({'factory': j, 'dc': k, 'flow': val})
            if y_flows:
                pd.DataFrame(y_flows).to_csv(f'./results/flows_factory_dc_{label}.csv', index=False)
            
            z_flows = []
            for (k, l), val in sol['flows']['z'].items():
                if val > 1e-6:
                    z_flows.append({'dc': k, 'customer': l, 'flow': val})
            if z_flows:
                pd.DataFrame(z_flows).to_csv(f'./results/flows_dc_customer_{label}.csv', index=False)
        
        print("  ✓ Saved flow solutions")
        
        # Save summary
        summary = {
            'experiment_name': 'green_resilient_scn_4echelon',
            'network_structure': {
                'n_suppliers': len(data['suppliers']),
                'n_factories': len(data['factories']),
                'n_dcs': len(data['dcs']),
                'n_customers': len(data['customers'])
            },
            'optimization': {
                'algorithm': 'NSGA-II',
                'n_pareto_solutions': len(pareto_result['pareto_front']),
                'execution_time_seconds': pareto_result['exec_time']
            },
            'representative_solutions': [
                {
                    'label': sol['label'],
                    'cost': sol['objectives']['cost'],
                    'emissions': sol['objectives']['emissions'],
                    'resilience': sol['objectives']['resilience']
                }
                for sol in representative
            ],
            'simulation_results': sim_results,
            'forecast_accuracy': metrics
        }
        
        with open('./results/experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("  ✓ Saved experiment summary to results/experiment_summary.json")
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nResults available in: ./results/")
        print("\nGenerated files:")
        print("  - pareto_front.csv (all Pareto-optimal solutions)")
        print("  - flows_*_min_cost.csv (minimum cost solution flows)")
        print("  - flows_*_min_emissions.csv (minimum emissions solution flows)")
        print("  - flows_*_balanced.csv (balanced solution flows)")
        print("  - experiment_summary.json (complete experiment summary)")
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        print("\nNote: NSGA-II requires pymoo library. Install with: pip install pymoo")
        raise

if __name__ == '__main__':
    main()
PYTHON_EOF

echo "Running experiment..."
echo ""

# Run the experiment with python3 (fallback to python)
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" "$RUN_SCRIPT"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Experiment completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results available in: $RESULTS_DIR"
    echo ""
    
    # Optionally run visualization
    if [ "${GENERATE_PLOTS:-1}" = "1" ]; then
        echo "Generating visualizations..."
        echo ""
        VIS_SCRIPT="$(dirname "$0")/visualize_results.py"
        if [ -f "$VIS_SCRIPT" ]; then
            "$PYTHON_BIN" "$VIS_SCRIPT" 2>/dev/null || echo "  (Visualization skipped - may require matplotlib/seaborn)"
        fi
        echo ""
    fi
    
    echo "Next steps:"
    echo "  1. Review results/experiment_summary.json"
    echo "  2. Check Pareto front: results/pareto_front.csv"
    echo "  3. Analyze flow solutions: results/flows_*.csv"
    if [ -d "$RESULTS_DIR/figures" ]; then
        echo "  4. View visualizations: results/figures/*.png"
    fi
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Experiment failed!"
    echo "=========================================="
    echo ""
    echo "Check the error messages above."
    echo "Common issues:"
    echo "  - Missing pymoo: pip install pymoo"
    echo "  - Missing other dependencies: pip install -r requirements.txt"
    echo ""
    exit $EXIT_CODE
fi

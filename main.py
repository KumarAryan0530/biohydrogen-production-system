"""
Biohydrogen Production Simulation & Optimization System
========================================================
Main entry point that orchestrates:
  1. Baseline biohydrogen simulation (ADM1 with disabled methanogenesis)
  2. pH × Temperature parameter sweep
  3. Scipy optimization for maximum H2 yield
  4. MPC-controlled fermentation
  5. Hydrogen production cost analysis (pyH2A-inspired)
  6. Results export to CSV and summary plots

Combines concepts from:
  - PyADM1 (anaerobic digestion model)
  - fermentation-mpc (fermentation control)
  - pyH2A (hydrogen cost analysis)

Usage:
    python main.py
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation.adm1_biohydrogen import BiohydrogenADM1, run_single_simulation
from optimization.parameter_sweep import run_parameter_sweep, find_best_from_sweep, optimize_conditions
from control.fermentation_control import FermentationController
from economics.h2_cost import calculate_h2_cost, estimate_annual_h2_from_simulation, h2_volume_to_mass


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def stage_1_baseline(results_dir):
    """Stage 1: Run baseline biohydrogen simulation."""
    print("\n" + "=" * 60)
    print("STAGE 1: Baseline Biohydrogen Simulation")
    print("=" * 60)
    print("  Model: ADM1 with methanogenesis DISABLED")
    print("  Conditions: pH=5.5, T=35°C, 30 days")

    model, results = run_single_simulation(
        target_ph=5.5,
        temperature_C=35.0,
        simulation_days=30.0,
        dt=0.5
    )

    h2_yield = model.get_total_h2_yield()
    avg_rate = model.get_average_h2_rate()
    h2_mass = h2_volume_to_mass(h2_yield, temperature_C=35.0)

    print(f"\n  Results:")
    print(f"    Cumulative H2 yield:  {h2_yield:,.2f} m³")
    print(f"    H2 mass:              {h2_mass:,.2f} kg")
    print(f"    Average H2 rate:      {avg_rate:,.2f} m³/day")
    print(f"    Final S_h2 (liquid):  {results['S_h2'].iloc[-1]:.6f} kgCOD/m³")
    print(f"    Final S_gas_h2 (gas): {results['S_gas_h2'].iloc[-1]:.6f} kgCOD/m³")

    # Save baseline time series
    results.to_csv(os.path.join(results_dir, 'baseline_simulation.csv'), index=False)

    return model, results


def stage_2_parameter_sweep(results_dir):
    """Stage 2: pH × Temperature parameter sweep."""
    print("\n" + "=" * 60)
    print("STAGE 2: Parameter Sweep (pH × Temperature)")
    print("=" * 60)
    print("  pH range:    5.0 — 8.0 (step 0.5)")
    print("  Temp range:  25 — 40°C (step 5)")
    print("  Simulation:  30 days per combination")
    print()

    sweep_results = run_parameter_sweep(
        ph_values=[5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
        temp_values=[25, 30, 35, 40],
        simulation_days=30.0,
        dt=1.0,
        verbose=True
    )

    # Save sweep results to CSV
    sweep_results.to_csv(
        os.path.join(results_dir, 'sweep_results.csv'),
        index=False
    )

    # Find best from grid
    best = find_best_from_sweep(sweep_results)
    print(f"\n  Best from grid search:")
    print(f"    pH = {best['pH']:.1f}")
    print(f"    Temperature = {best['temperature']:.0f}°C")
    print(f"    H2 yield = {best['hydrogen_yield']:,.2f} m³")

    return sweep_results, best


def stage_3_optimization(best_grid, results_dir):
    """Stage 3: Scipy optimization from best grid point."""
    print("\n" + "=" * 60)
    print("STAGE 3: Optimization (Nelder-Mead)")
    print("=" * 60)
    print(f"  Starting from grid best: pH={best_grid['pH']:.1f}, T={best_grid['temperature']:.0f}°C")

    opt_result = optimize_conditions(
        initial_ph=best_grid['pH'],
        initial_temp=best_grid['temperature'],
        ph_bounds=(5.0, 8.0),
        temp_bounds=(25.0, 40.0),
        simulation_days=30.0,
        dt=1.0,
        verbose=True
    )

    print(f"\n  Optimal conditions found:")
    print(f"    pH = {opt_result['optimal_pH']:.3f}")
    print(f"    Temperature = {opt_result['optimal_temperature']:.1f}°C")
    print(f"    H2 yield = {opt_result['hydrogen_yield']:,.2f} m³")
    print(f"    Avg H2 rate = {opt_result['avg_h2_rate']:,.2f} m³/day")

    return opt_result


def stage_4_mpc_control(opt_result, results_dir):
    """Stage 4: MPC-controlled fermentation at optimal conditions."""
    print("\n" + "=" * 60)
    print("STAGE 4: MPC Fermentation Control")
    print("=" * 60)
    print(f"  Operating at pH={opt_result['optimal_pH']:.2f}, T={opt_result['optimal_temperature']:.1f}°C")
    print(f"  Control: Optimizing feed rate (q_ad) every 5 days")

    controller = FermentationController(
        target_ph=opt_result['optimal_pH'],
        temperature_C=opt_result['optimal_temperature'],
        control_horizon=5.0,
        control_interval=5.0,
        total_time=30.0
    )

    control_results = controller.run_controlled(verbose=True)

    print(f"\n  Control Results:")
    print(f"    Uncontrolled H2: {control_results['uncontrolled_h2_yield']:,.2f} m³")
    print(f"    Controlled H2:   {control_results['controlled_h2_yield']:,.2f} m³")
    print(f"    Improvement:     {control_results['improvement_pct']:+.1f}%")

    # Save feed rate schedule
    control_results['feed_rates'].to_csv(
        os.path.join(results_dir, 'mpc_feed_rates.csv'),
        index=False
    )

    return control_results


def stage_5_economics(opt_result, control_results, results_dir):
    """Stage 5: Hydrogen production cost analysis."""
    print("\n" + "=" * 60)
    print("STAGE 5: Hydrogen Production Cost (pyH2A-inspired DCF)")
    print("=" * 60)

    # Use controlled H2 yield for cost analysis
    daily_h2_m3 = control_results['controlled_h2_yield'] / 30.0  # 30-day simulation
    annual_h2_kg = estimate_annual_h2_from_simulation(
        daily_h2_m3=daily_h2_m3,
        capacity_factor=0.90,
        temperature_C=opt_result['optimal_temperature']
    )

    print(f"  Input from simulation:")
    print(f"    Avg daily H2: {daily_h2_m3:,.2f} m³/day")
    print(f"    Annual H2:    {annual_h2_kg:,.1f} kg/year")

    cost_result = calculate_h2_cost(
        annual_h2_production_kg=annual_h2_kg,
        verbose=True
    )

    return cost_result


def stage_6_save_and_plot(sweep_results, best_grid, opt_result,
                          control_results, cost_result, results_dir):
    """Stage 6: Save final results and generate plots."""
    print("\n" + "=" * 60)
    print("STAGE 6: Save Results & Generate Plots")
    print("=" * 60)

    # Save summary
    summary = {
        'Parameter': [
            'Best pH (grid)', 'Best Temperature (grid)',
            'Best H2 yield (grid)',
            'Optimal pH', 'Optimal Temperature',
            'Optimal H2 yield',
            'Controlled H2 yield', 'Uncontrolled H2 yield',
            'MPC Improvement',
            'LCOH ($/kg H2)', 'Annual H2 (kg/yr)',
            'Total Capital ($)'
        ],
        'Value': [
            best_grid['pH'], best_grid['temperature'],
            best_grid['hydrogen_yield'],
            opt_result['optimal_pH'], opt_result['optimal_temperature'],
            opt_result['hydrogen_yield'],
            control_results['controlled_h2_yield'],
            control_results['uncontrolled_h2_yield'],
            control_results['improvement_pct'],
            cost_result['lcoh'], cost_result['annual_h2_kg'],
            cost_result['total_capital']
        ]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)
    print(f"  Saved: results/summary.csv")
    print(f"  Saved: results/sweep_results.csv")

    # === Plot 1: H2 Yield Heatmap ===
    try:
        successful = sweep_results[sweep_results['status'] == 'success']
        if len(successful) > 0:
            pivot = successful.pivot_table(
                values='hydrogen_yield',
                index='pH',
                columns='temperature',
                aggfunc='first'
            )

            fig, ax = plt.subplots(figsize=(10, 7))
            im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto',
                          origin='lower')

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{t}°C" for t in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{p:.1f}" for p in pivot.index])
            ax.set_xlabel('Temperature (°C)', fontsize=12)
            ax.set_ylabel('pH', fontsize=12)
            ax.set_title('Biohydrogen Yield (m³) — pH × Temperature', fontsize=14)

            # Add text annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.0f}',
                               ha='center', va='center',
                               color='black' if val < pivot.values.max() * 0.7 else 'white',
                               fontsize=10, fontweight='bold')

            # Mark optimal point
            opt_ph_idx = np.argmin(np.abs(np.array(pivot.index) - opt_result['optimal_pH']))
            opt_t_idx = np.argmin(np.abs(np.array(pivot.columns) - opt_result['optimal_temperature']))
            ax.plot(opt_t_idx, opt_ph_idx, 'w*', markersize=20, markeredgecolor='black')

            plt.colorbar(im, label='H₂ Yield (m³)', ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'h2_yield_heatmap.png'), dpi=150)
            plt.close()
            print(f"  Saved: results/h2_yield_heatmap.png")
    except Exception as e:
        print(f"  Warning: Could not generate heatmap: {e}")

    # === Plot 2: Baseline H2 production time series ===
    try:
        baseline_csv = os.path.join(results_dir, 'baseline_simulation.csv')
        if os.path.exists(baseline_csv):
            baseline = pd.read_csv(baseline_csv)

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            axes[0].plot(baseline['time'], baseline['q_h2'],
                        color='#2196F3', linewidth=2, label='H₂ flow rate')
            axes[0].set_ylabel('H₂ Flow Rate (m³/day)', fontsize=11)
            axes[0].set_title('Biohydrogen Production — Baseline Simulation', fontsize=14)
            axes[0].legend(fontsize=10)
            axes[0].grid(alpha=0.3)

            axes[1].plot(baseline['time'], baseline['cumulative_h2'],
                        color='#4CAF50', linewidth=2, label='Cumulative H₂')
            axes[1].set_xlabel('Time (days)', fontsize=11)
            axes[1].set_ylabel('Cumulative H₂ (m³)', fontsize=11)
            axes[1].legend(fontsize=10)
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'baseline_h2_production.png'), dpi=150)
            plt.close()
            print(f"  Saved: results/baseline_h2_production.png")
    except Exception as e:
        print(f"  Warning: Could not generate time series plot: {e}")

    # === Plot 3: Cost breakdown ===
    try:
        if cost_result['lcoh'] < float('inf'):
            fig, ax = plt.subplots(figsize=(8, 6))
            labels = ['Capital', 'Fixed OpEx', 'Feedstock', 'Utilities', 'Other']
            values = [
                max(cost_result['capital_contribution'], 0),
                max(cost_result['fixed_operating_contribution'], 0),
                max(cost_result['feedstock_contribution'], 0),
                max(cost_result['utilities_contribution'], 0),
                max(cost_result['other_contribution'], 0)
            ]
            colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']

            bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.6)
            ax.set_xlabel('Cost Contribution ($/kg H₂)', fontsize=12)
            ax.set_title(f'H₂ Cost Breakdown — LCOH: ${cost_result["lcoh"]:.2f}/kg', fontsize=14)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                       f'${val:.2f}', va='center', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'cost_breakdown.png'), dpi=150)
            plt.close()
            print(f"  Saved: results/cost_breakdown.png")
    except Exception as e:
        print(f"  Warning: Could not generate cost plot: {e}")


def main():
    """Main entry point — run all stages."""
    print("==========================================================")
    print("   Biohydrogen Production Simulation & Optimization      ")
    print("   ADM1 + MPC Control + H2A Economics                    ")
    print("==========================================================")

    start_time = time.time()
    results_dir = ensure_results_dir()

    # Stage 1: Baseline simulation
    baseline_model, baseline_results = stage_1_baseline(results_dir)

    # Stage 2: Parameter sweep
    sweep_results, best_grid = stage_2_parameter_sweep(results_dir)

    # Stage 3: Optimization
    opt_result = stage_3_optimization(best_grid, results_dir)

    # Stage 4: MPC Control
    control_results = stage_4_mpc_control(opt_result, results_dir)

    # Stage 5: Economics
    cost_result = stage_5_economics(opt_result, control_results, results_dir)

    # Stage 6: Save & Plot
    stage_6_save_and_plot(
        sweep_results, best_grid, opt_result,
        control_results, cost_result, results_dir
    )

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Best conditions:")
    print(f"    pH:          {opt_result['optimal_pH']:.2f}")
    print(f"    Temperature: {opt_result['optimal_temperature']:.1f}°C")
    print(f"    H2 yield:    {opt_result['hydrogen_yield']:,.2f} m³ (30 days)")
    print(f"  MPC control improvement: {control_results['improvement_pct']:+.1f}%")
    print(f"  LCOH: ${cost_result['lcoh']:.2f}/kg H2")
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Results saved to: {results_dir}")
    print("=" * 60)


def run_with_parameters(params_dict):
    """
    Run simulation with custom parameters from web form.
    
    Parameters (from web form):
    {
        'run_name': str,
        'baseline_ph': float,
        'baseline_temp': float,
        'baseline_days': float,
        'sweep_ph_min': float,
        'sweep_ph_max': float,
        'sweep_ph_step': float,
        'sweep_temp_min': float,
        'sweep_temp_max': float,
        'sweep_temp_step': float,
        'mpc_control_horizon': float,
        'mpc_control_interval': float,
        'mpc_total_time': float,
        'plant_capital_cost': float,
        'plant_life': int,
        'discount_rate': float,
        'capacity_factor': float
    }
    
    Returns:
    {
        'status': 'success' or 'error',
        'run_name': str,
        'optimal_ph': float,
        'optimal_temp': float,
        'optimal_h2_yield': float,
        'mpc_improvement': float,
        'lcoh': float,
        'annual_h2_kg': float,
        'total_capital': float,
        'error': str (only if status=='error')
    }
    """
    try:
        results_dir = ensure_results_dir()
        
        # Extract parameters
        baseline_ph = float(params_dict.get('baseline_ph', 5.5))
        baseline_temp = float(params_dict.get('baseline_temp', 35.0))
        baseline_days = float(params_dict.get('baseline_days', 30.0))
        
        sweep_ph_min = float(params_dict.get('sweep_ph_min', 5.0))
        sweep_ph_max = float(params_dict.get('sweep_ph_max', 8.0))
        sweep_ph_step = float(params_dict.get('sweep_ph_step', 0.5))
        
        sweep_temp_min = float(params_dict.get('sweep_temp_min', 25.0))
        sweep_temp_max = float(params_dict.get('sweep_temp_max', 40.0))
        sweep_temp_step = float(params_dict.get('sweep_temp_step', 5.0))
        
        mpc_control_horizon = float(params_dict.get('mpc_control_horizon', 5.0))
        mpc_control_interval = float(params_dict.get('mpc_control_interval', 5.0))
        mpc_total_time = float(params_dict.get('mpc_total_time', 30.0))
        
        plant_capital_cost = float(params_dict.get('plant_capital_cost', 5_000_000))
        plant_life = int(params_dict.get('plant_life', 20))
        discount_rate = float(params_dict.get('discount_rate', 0.08))
        capacity_factor = float(params_dict.get('capacity_factor', 0.90))
        
        # Build pH range as array
        ph_values = np.arange(sweep_ph_min, sweep_ph_max + sweep_ph_step/2, sweep_ph_step).tolist()
        temp_values = np.arange(sweep_temp_min, sweep_temp_max + sweep_temp_step/2, sweep_temp_step).tolist()
        
        print("\n" + "=" * 60)
        print(f"RUN: {params_dict.get('run_name', 'Custom Run')}")
        print("=" * 60)
        print(f"  Baseline: pH={baseline_ph:.1f}, T={baseline_temp:.0f}°C, {baseline_days:.0f} days")
        print(f"  Sweep: pH [{sweep_ph_min:.1f}-{sweep_ph_max:.1f}], T [{sweep_temp_min:.0f}-{sweep_temp_max:.0f}°C]")
        print(f"  MPC: horizon={mpc_control_horizon:.0f}d, interval={mpc_control_interval:.0f}d, total={mpc_total_time:.0f}d")
        
        # Stage 1: Baseline simulation
        print("\n[1/6] Running baseline simulation...")
        baseline_model, baseline_results = stage_1_baseline(results_dir)
        baseline_results.to_csv(os.path.join(results_dir, 'baseline_simulation.csv'), index=False)
        
        # Stage 2: Parameter sweep
        print("\n[2/6] Running parameter sweep...")
        sweep_results = run_parameter_sweep(
            ph_values=ph_values,
            temp_values=temp_values,
            simulation_days=baseline_days,
            dt=1.0,
            verbose=False
        )
        sweep_results.to_csv(os.path.join(results_dir, 'sweep_results.csv'), index=False)
        best_grid = find_best_from_sweep(sweep_results)
        print(f"  Best: pH={best_grid['pH']:.1f}, T={best_grid['temperature']:.0f}°C")
        
        # Stage 3: Optimization
        print("\n[3/6] Running optimization...")
        opt_result = optimize_conditions(
            initial_ph=best_grid['pH'],
            initial_temp=best_grid['temperature'],
            ph_bounds=(sweep_ph_min, sweep_ph_max),
            temp_bounds=(sweep_temp_min, sweep_temp_max),
            simulation_days=baseline_days,
            dt=1.0,
            verbose=False
        )
        print(f"  Optimal: pH={opt_result['optimal_pH']:.2f}, T={opt_result['optimal_temperature']:.1f}°C")
        
        # Stage 4: MPC Control
        print("\n[4/6] Running MPC control...")
        controller = FermentationController(
            target_ph=opt_result['optimal_pH'],
            temperature_C=opt_result['optimal_temperature'],
            control_horizon=mpc_control_horizon,
            control_interval=mpc_control_interval,
            total_time=mpc_total_time
        )
        control_results = controller.run_controlled(verbose=False)
        control_results['feed_rates'].to_csv(
            os.path.join(results_dir, 'mpc_feed_rates.csv'),
            index=False
        )
        print(f"  Improvement: {control_results['improvement_pct']:+.1f}%")
        
        # Stage 5: Economics
        print("\n[5/6] Running cost analysis...")
        daily_h2_m3 = control_results['controlled_h2_yield'] / mpc_total_time
        annual_h2_kg = estimate_annual_h2_from_simulation(
            daily_h2_m3=daily_h2_m3,
            capacity_factor=capacity_factor,
            temperature_C=opt_result['optimal_temperature']
        )
        
        # Create custom cost parameters (override defaults)
        from economics.h2_cost import DEFAULT_PLANT_PARAMS
        plant_params = DEFAULT_PLANT_PARAMS.copy()
        plant_params['total_capital_cost'] = plant_capital_cost
        plant_params['plant_life'] = plant_life
        plant_params['discount_rate'] = discount_rate
        plant_params['capacity_factor'] = capacity_factor
        
        cost_result = calculate_h2_cost(
            annual_h2_production_kg=annual_h2_kg,
            plant_params=plant_params,
            verbose=False
        )
        print(f"  LCOH: ${cost_result['lcoh']:.2f}/kg H2")
        
        # Stage 6: Save & Plot
        print("\n[6/6] Generating plots and summary...")
        stage_6_save_and_plot(
            sweep_results, best_grid, opt_result,
            control_results, cost_result, results_dir
        )
        
        # Return summary
        return {
            'status': 'success',
            'run_name': params_dict.get('run_name', 'Custom Run'),
            'optimal_ph': float(opt_result['optimal_pH']),
            'optimal_temp': float(opt_result['optimal_temperature']),
            'optimal_h2_yield': float(opt_result['hydrogen_yield']),
            'mpc_improvement': float(control_results['improvement_pct']),
            'lcoh': float(cost_result['lcoh']),
            'annual_h2_kg': float(cost_result['annual_h2_kg']),
            'total_capital': float(cost_result['total_capital'])
        }
    
    except Exception as e:
        import traceback
        print(f"\nError in run_with_parameters: {str(e)}")
        traceback.print_exc()
        return {
            'status': 'error',
            'error': str(e),
            'run_name': params_dict.get('run_name', 'Custom Run')
        }


if __name__ == '__main__':
    main()

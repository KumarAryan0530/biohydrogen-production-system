"""
Parameter Sweep & Optimization Module
======================================
Runs pH × Temperature grid search and scipy optimization
to find conditions that maximize biohydrogen yield.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys
import os
import concurrent.futures

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.adm1_biohydrogen import BiohydrogenADM1


def run_parameter_sweep(ph_values=None, temp_values=None,
                        simulation_days=30.0, dt=1.0,
                        verbose=True):
    """
    Run a grid search over pH and temperature combinations.

    Parameters
    ----------
    ph_values : array-like or None
        pH values to sweep. Default: [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    temp_values : array-like or None
        Temperature values in °C. Default: [25, 30, 35, 40]
    simulation_days : float
        Simulation duration for each run.
    dt : float
        Time step for each run.
    verbose : bool
        Print progress updates.

    Returns
    -------
    results : pd.DataFrame
        DataFrame with columns: pH, temperature, hydrogen_yield, avg_h2_rate
    """
    if ph_values is None:
        ph_values = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    if temp_values is None:
        temp_values = [25, 30, 35, 40]

    results = []
    
    # Generate all combinations
    combinations = [(ph, temp) for ph in ph_values for temp in temp_values]
    total = len(combinations)
    count = 0

    def evaluate_combination(params):
        ph, temp = params
        try:
            model = BiohydrogenADM1(
                target_ph=ph,
                temperature_C=float(temp),
                simulation_days=simulation_days,
                dt=dt
            )
            model.simulate()
            h2_yield = model.get_total_h2_yield()
            avg_rate = model.get_average_h2_rate()

            return {
                'pH': ph,
                'temperature': temp,
                'hydrogen_yield': h2_yield,
                'avg_h2_rate': avg_rate,
                'status': 'success'
            }
        except Exception as e:
            return {
                'pH': ph,
                'temperature': temp,
                'hydrogen_yield': 0.0,
                'avg_h2_rate': 0.0,
                'status': f'error: {str(e)}'
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(evaluate_combination, comb): comb for comb in combinations}
        for future in concurrent.futures.as_completed(futures):
            count += 1
            res = future.result()
            results.append(res)
            if verbose:
                ph, temp = res['pH'], res['temperature']
                if res['status'] == 'success':
                    print(f"  [{count}/{total}] pH={ph:.1f}, T={temp}°C → H2={res['hydrogen_yield']:.2f} m³", flush=True)
                else:
                    print(f"  [{count}/{total}] pH={ph:.1f}, T={temp}°C → ERROR: {res['status']}", flush=True)

    df = pd.DataFrame(results)
    return df


def find_best_from_sweep(sweep_results):
    """
    Find the best conditions from a parameter sweep.

    Parameters
    ----------
    sweep_results : pd.DataFrame
        Results from run_parameter_sweep().

    Returns
    -------
    best : dict
        Dictionary with best pH, temperature, and hydrogen_yield.
    """
    successful = sweep_results[sweep_results['status'] == 'success']
    if len(successful) == 0:
        return {'pH': 5.5, 'temperature': 35, 'hydrogen_yield': 0.0}

    best_idx = successful['hydrogen_yield'].idxmax()
    best = successful.loc[best_idx]
    return {
        'pH': float(best['pH']),
        'temperature': float(best['temperature']),
        'hydrogen_yield': float(best['hydrogen_yield']),
        'avg_h2_rate': float(best['avg_h2_rate'])
    }


def optimize_conditions(initial_ph=5.5, initial_temp=35.0,
                        ph_bounds=(5.0, 8.0), temp_bounds=(25.0, 40.0),
                        simulation_days=30.0, dt=1.0,
                        verbose=True):
    """
    Use scipy optimization to find the optimal pH and temperature
    that maximize hydrogen yield.

    Parameters
    ----------
    initial_ph : float
        Starting pH for optimizer.
    initial_temp : float
        Starting temperature for optimizer (°C).
    ph_bounds : tuple
        (min_ph, max_ph) bounds.
    temp_bounds : tuple
        (min_temp, max_temp) bounds.
    simulation_days : float
        Simulation duration.
    dt : float
        Time step.
    verbose : bool
        Print optimization progress.

    Returns
    -------
    result : dict
        Dictionary with optimal pH, temperature, hydrogen_yield, and
        optimization details.
    """
    eval_count = [0]

    def objective(x):
        """Negative H2 yield (minimize → maximize yield)."""
        ph, temp = x
        eval_count[0] += 1

        try:
            model = BiohydrogenADM1(
                target_ph=ph,
                temperature_C=temp,
                simulation_days=simulation_days,
                dt=dt
            )
            model.simulate()
            h2_yield = model.get_total_h2_yield()

            if verbose and eval_count[0] % 5 == 0:
                print(f"    Eval #{eval_count[0]}: pH={ph:.3f}, T={temp:.1f}°C → H2={h2_yield:.2f} m³")

            return -h2_yield  # Minimize negative = maximize yield

        except Exception:
            return 0.0  # Penalty for failed simulations

    x0 = [initial_ph, initial_temp]
    bounds = [ph_bounds, temp_bounds]

    if verbose:
        print(f"  Starting optimization from pH={initial_ph:.2f}, T={initial_temp:.1f}°C")

    opt_result = minimize(
        objective, x0,
        method='Nelder-Mead',
        options={
            'maxiter': 15,
            'xatol': 0.5,
            'fatol': 5.0,
            'adaptive': True
        }
    )

    # Clamp to bounds
    opt_ph = np.clip(opt_result.x[0], ph_bounds[0], ph_bounds[1])
    opt_temp = np.clip(opt_result.x[1], temp_bounds[0], temp_bounds[1])

    # Run final simulation at optimal point
    final_model = BiohydrogenADM1(
        target_ph=opt_ph,
        temperature_C=opt_temp,
        simulation_days=simulation_days,
        dt=dt
    )
    final_model.simulate()
    final_yield = final_model.get_total_h2_yield()

    result = {
        'optimal_pH': float(opt_ph),
        'optimal_temperature': float(opt_temp),
        'hydrogen_yield': float(final_yield),
        'avg_h2_rate': float(final_model.get_average_h2_rate()),
        'n_evaluations': eval_count[0],
        'optimizer_success': opt_result.success,
        'model': final_model
    }

    if verbose:
        print(f"  Optimization complete: pH={opt_ph:.3f}, T={opt_temp:.1f}°C, "
              f"H2={final_yield:.2f} m³ ({eval_count[0]} evaluations)")

    return result

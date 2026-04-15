"""
Fermentation Control Module
============================
MPC-inspired controller for biohydrogen fermentation.

Inspired by the fermentation-mpc repository (biosustain/fermentation-mpc),
adapted for ADM1-based biohydrogen production.

The controller optimizes the feed rate (q_ad) to maximize hydrogen
production rate using a receding horizon approach:
  - At each control interval, simulate N steps forward with different
    feed rates
  - Choose the feed rate that maximizes predicted H2 output
  - Apply and move to next interval

This avoids the tellurium dependency of the original fermentation-mpc
while maintaining the same MPC control philosophy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.adm1_biohydrogen import BiohydrogenADM1


class FermentationController:
    """
    MPC-inspired feed rate controller for biohydrogen production.

    Parameters
    ----------
    target_ph : float
        Target reactor pH.
    temperature_C : float
        Operating temperature (°C).
    base_q_ad : float
        Baseline feed rate (m³/d).
    q_ad_min : float
        Minimum allowed feed rate (m³/d).
    q_ad_max : float
        Maximum allowed feed rate (m³/d).
    control_horizon : float
        Prediction horizon in days for each control step.
    control_interval : float
        Time between control actions (days).
    total_time : float
        Total simulation time (days).
    """

    def __init__(self, target_ph=5.5, temperature_C=35.0,
                 base_q_ad=178.4674,
                 q_ad_min=50.0, q_ad_max=400.0,
                 control_horizon=5.0, control_interval=2.0,
                 total_time=30.0):
        self.target_ph = target_ph
        self.temperature_C = temperature_C
        self.base_q_ad = base_q_ad
        self.q_ad_min = q_ad_min
        self.q_ad_max = q_ad_max
        self.control_horizon = control_horizon
        self.control_interval = control_interval
        self.total_time = total_time

    def _evaluate_feed_rate(self, q_ad, current_state, horizon_days):
        """
        Evaluate a candidate feed rate by running a short prediction.

        Parameters
        ----------
        q_ad : float
            Feed rate to evaluate.
        current_state : list
            Current reactor state vector.
        horizon_days : float
            How far to predict.

        Returns
        -------
        float
            Predicted H2 production over the horizon (m³).
        """
        model = BiohydrogenADM1(
            target_ph=self.target_ph,
            temperature_C=self.temperature_C,
            simulation_days=horizon_days,
            dt=max(horizon_days / 5, 0.5),
            q_ad=q_ad
        )
        model.state = list(current_state)
        try:
            model.simulate()
            return model.get_total_h2_yield()
        except Exception:
            return 0.0

    def run_controlled(self, verbose=True):
        """
        Run the controlled fermentation simulation.

        Returns
        -------
        results : dict
            Dictionary with:
            - 'controlled_results': DataFrame of time series with MPC control
            - 'uncontrolled_results': DataFrame of baseline simulation
            - 'feed_rates': list of applied feed rates over time
            - 'controlled_h2_yield': total H2 with control
            - 'uncontrolled_h2_yield': total H2 without control
            - 'improvement_pct': percentage improvement
        """
        if verbose:
            print("  Running uncontrolled baseline...")

        # Uncontrolled baseline
        baseline_model = BiohydrogenADM1(
            target_ph=self.target_ph,
            temperature_C=self.temperature_C,
            simulation_days=self.total_time,
            dt=1.0,
            q_ad=self.base_q_ad
        )
        baseline_results = baseline_model.simulate()
        baseline_h2 = baseline_model.get_total_h2_yield()

        if verbose:
            print(f"    Baseline H2 yield: {baseline_h2:.2f} m³")
            print("  Running MPC-controlled simulation...")

        # MPC-controlled simulation
        n_controls = int(self.total_time / self.control_interval)
        current_state = list(baseline_model.state)  # Start from same initial state

        # Reset to original initial state
        from simulation.adm1_biohydrogen import get_default_initial_state
        current_state = get_default_initial_state()
        current_state[26] = 10 ** (-self.target_ph)

        controlled_dfs = []
        feed_rates = []
        cumulative_time = 0.0
        total_controlled_h2 = 0.0

        for step in range(n_controls):
            remaining = self.total_time - cumulative_time
            if remaining <= 0:
                break

            step_duration = min(self.control_interval, remaining)
            horizon = min(self.control_horizon, remaining)

            # Find optimal feed rate for this interval
            def neg_h2(q):
                return -self._evaluate_feed_rate(q, current_state, horizon)

            try:
                opt_result = minimize_scalar(
                    neg_h2,
                    bounds=(self.q_ad_min, self.q_ad_max),
                    method='bounded',
                    options={'maxiter': 10}
                )
                optimal_q = opt_result.x
            except Exception:
                optimal_q = self.base_q_ad

            optimal_q = np.clip(optimal_q, self.q_ad_min, self.q_ad_max)
            feed_rates.append({'time': cumulative_time, 'q_ad': optimal_q})

            if verbose:
                print(f"    Step {step+1}/{n_controls}: t={cumulative_time:.1f}d, "
                      f"optimal q_ad={optimal_q:.1f} m³/d")

            # Apply optimal feed rate for this interval
            step_model = BiohydrogenADM1(
                target_ph=self.target_ph,
                temperature_C=self.temperature_C,
                simulation_days=step_duration,
                dt=max(step_duration / 5, 0.5),
                q_ad=optimal_q
            )
            step_model.state = list(current_state)
            step_results = step_model.simulate()

            # Offset time
            step_results['time'] += cumulative_time
            controlled_dfs.append(step_results)

            # Update state for next interval
            current_state = [step_results[col].iloc[-1] for col in
                           ['S_su', 'S_aa', 'S_fa', 'S_va', 'S_bu', 'S_pro', 'S_ac',
                            'S_h2', 'S_ch4', 'S_IC', 'S_IN', 'S_I',
                            'X_xc', 'X_ch', 'X_pr', 'X_li', 'X_su', 'X_aa', 'X_fa',
                            'X_c4', 'X_pro', 'X_ac', 'X_h2', 'X_I',
                            'S_cation', 'S_anion', 'S_H_ion',
                            'S_va_ion', 'S_bu_ion', 'S_pro_ion', 'S_ac_ion',
                            'S_hco3_ion', 'S_co2', 'S_nh3', 'S_nh4_ion',
                            'S_gas_h2', 'S_gas_ch4', 'S_gas_co2']]

            step_h2 = step_model.get_total_h2_yield()
            total_controlled_h2 += step_h2
            cumulative_time += step_duration

        # Combine controlled results
        if controlled_dfs:
            controlled_results = pd.concat(controlled_dfs, ignore_index=True)
        else:
            controlled_results = baseline_results.copy()

        # Calculate improvement
        improvement_pct = 0.0
        if baseline_h2 > 0:
            improvement_pct = (total_controlled_h2 - baseline_h2) / baseline_h2 * 100

        if verbose:
            print(f"    Controlled H2 yield: {total_controlled_h2:.2f} m³")
            print(f"    Improvement: {improvement_pct:+.1f}%")

        return {
            'controlled_results': controlled_results,
            'uncontrolled_results': baseline_results,
            'feed_rates': pd.DataFrame(feed_rates),
            'controlled_h2_yield': total_controlled_h2,
            'uncontrolled_h2_yield': baseline_h2,
            'improvement_pct': improvement_pct
        }

"""
Hydrogen Cost Analysis Module
===============================
Simplified discounted cash flow (DCF) analysis for hydrogen production cost.

Inspired by pyH2A (jschneidewind/pyH2A) — a Python framework for hydrogen
production cost analysis based on the H2A model by DOE/NREL.

This module provides a standalone LCOH (Levelized Cost of Hydrogen) calculation
using standard financial parameters for dark fermentation biohydrogen plants.

Reference:
  Schneidewind, J. (2021). pyH2A: Python framework for hydrogen production
  cost analysis.  DOE H2A Hydrogen Analysis Production Models.
"""

import numpy as np


# ============================================================================
# Default Dark Fermentation Plant Parameters
# ============================================================================

DEFAULT_PLANT_PARAMS = {
    # Capital costs (USD)
    'total_capital_cost': 5_000_000,      # Total installed capital ($5M for small plant)
    'depreciable_fraction': 0.85,          # Fraction of capital that is depreciable

    # Operating costs (USD/year)
    'fixed_operating_cost': 200_000,       # Labor, insurance, maintenance ($/yr)
    'feedstock_cost_per_kg_h2': 2.50,      # Substrate cost per kg H2 produced
    'utilities_cost_per_kg_h2': 0.80,      # Electricity, water, pH buffer per kg H2

    # Financial parameters
    'plant_life': 20,                       # years
    'construction_time': 1,                 # years
    'discount_rate': 0.08,                  # 8% real after-tax IRR
    'inflation_rate': 0.019,                # 1.9% annual inflation
    'equity_fraction': 0.40,                # 40% equity financing
    'debt_interest_rate': 0.037,            # 3.7% interest on debt
    'federal_tax_rate': 0.21,               # 21% federal tax
    'state_tax_rate': 0.06,                 # 6% state tax
    'depreciation_period': 7,               # MACRS 7-year

    # Operating parameters
    'capacity_factor': 0.90,                # 90% annual uptime
    'startup_time': 0.5,                    # years
    'startup_production_fraction': 0.50,    # 50% production during startup
    'decommissioning_fraction': 0.10,       # 10% of capital
    'salvage_fraction': 0.10,               # 10% of capital
    'working_capital_fraction': 0.15,       # 15% of annual operating costs

    # Replacement costs
    'annual_replacement_fraction': 0.005,   # 0.5% of capital per year
}


def calculate_h2_cost(annual_h2_production_kg,
                      plant_params=None,
                      verbose=True):
    """
    Calculate the Levelized Cost of Hydrogen (LCOH) using simplified DCF.

    This follows the pyH2A / DOE H2A methodology:
    LCOH = (Annualized_Capital + Operating_Costs) / Annual_H2_Production

    Parameters
    ----------
    annual_h2_production_kg : float
        Annual H2 production in kg.
    plant_params : dict or None
        Plant parameters. If None, uses DEFAULT_PLANT_PARAMS.
    verbose : bool
        Print detailed cost breakdown.

    Returns
    -------
    result : dict
        Dictionary with LCOH and cost breakdown.
    """
    if plant_params is None:
        plant_params = DEFAULT_PLANT_PARAMS.copy()

    p = plant_params

    # Ensure minimum production
    if annual_h2_production_kg <= 0:
        if verbose:
            print("  WARNING: Zero or negative H2 production — cannot calculate cost")
        return {
            'lcoh': float('inf'),
            'annual_h2_kg': 0,
            'capital_contribution': float('inf'),
            'operating_contribution': float('inf'),
            'feedstock_contribution': 0,
            'total_capital': p['total_capital_cost'],
        }

    # ========================================================================
    # Capital cost annualization using CRF (Capital Recovery Factor)
    # ========================================================================
    r = p['discount_rate']
    n = p['plant_life']

    # Capital Recovery Factor: CRF = r(1+r)^n / ((1+r)^n - 1)
    crf = r * (1 + r) ** n / ((1 + r) ** n - 1)

    # Total depreciable capital
    depreciable_capital = p['total_capital_cost'] * p['depreciable_fraction']
    non_depreciable_capital = p['total_capital_cost'] * (1 - p['depreciable_fraction'])

    # Effective tax rate
    total_tax_rate = p['federal_tax_rate'] + p['state_tax_rate'] * (1 - p['federal_tax_rate'])

    # Tax shield from depreciation (simplified MACRS)
    # Assume straight-line approximation for simplicity
    annual_depreciation = depreciable_capital / p['depreciation_period']
    annual_tax_shield = annual_depreciation * total_tax_rate

    # Average annual tax shield over plant life (depreciation only applies
    # for depreciation_period years, averaged over plant_life)
    avg_annual_tax_shield = annual_tax_shield * min(p['depreciation_period'], n) / n

    # Annualized capital cost
    annualized_capital = p['total_capital_cost'] * crf

    # After-tax annualized capital (net of depreciation tax shield)
    net_annualized_capital = annualized_capital - avg_annual_tax_shield

    # ========================================================================
    # Operating costs
    # ========================================================================
    fixed_operating = p['fixed_operating_cost']

    # Variable operating costs scale with production
    feedstock_cost = p['feedstock_cost_per_kg_h2'] * annual_h2_production_kg
    utilities_cost = p['utilities_cost_per_kg_h2'] * annual_h2_production_kg

    # Replacement costs
    replacement_cost = p['total_capital_cost'] * p['annual_replacement_fraction']

    # Working capital (annualized cost of maintaining working capital)
    total_annual_operating = fixed_operating + feedstock_cost + utilities_cost + replacement_cost
    working_capital_cost = total_annual_operating * p['working_capital_fraction'] * r

    # Total annual operating cost (after tax)
    total_operating = (total_annual_operating + working_capital_cost) * (1 - total_tax_rate)

    # ========================================================================
    # Salvage and decommissioning (annualized)
    # ========================================================================
    decommissioning = p['total_capital_cost'] * p['decommissioning_fraction']
    salvage = p['total_capital_cost'] * p['salvage_fraction']
    net_end_of_life = (decommissioning - salvage) / ((1 + r) ** n)  # PV of net end-of-life
    annualized_eol = net_end_of_life * crf

    # ========================================================================
    # LCOH Calculation
    # ========================================================================
    total_annual_cost = net_annualized_capital + total_operating + annualized_eol
    lcoh = total_annual_cost / annual_h2_production_kg

    # Cost contributions per kg H2
    capital_per_kg = net_annualized_capital / annual_h2_production_kg
    fixed_op_per_kg = (fixed_operating * (1 - total_tax_rate)) / annual_h2_production_kg
    feedstock_per_kg = p['feedstock_cost_per_kg_h2'] * (1 - total_tax_rate)
    utilities_per_kg = p['utilities_cost_per_kg_h2'] * (1 - total_tax_rate)
    other_per_kg = lcoh - capital_per_kg - fixed_op_per_kg - feedstock_per_kg - utilities_per_kg

    result = {
        'lcoh': lcoh,
        'annual_h2_kg': annual_h2_production_kg,
        'total_capital': p['total_capital_cost'],
        'capital_contribution': capital_per_kg,
        'fixed_operating_contribution': fixed_op_per_kg,
        'feedstock_contribution': feedstock_per_kg,
        'utilities_contribution': utilities_per_kg,
        'other_contribution': other_per_kg,
        'annualized_capital': net_annualized_capital,
        'annual_operating': total_operating,
        'total_annual_cost': total_annual_cost,
        'crf': crf,
        'effective_tax_rate': total_tax_rate,
    }

    if verbose:
        print(f"\n  ================================================")
        print(f"  |  Hydrogen Production Cost Analysis (pyH2A)   |")
        print(f"  ================================================")
        print(f"  |  Annual H2 Production:  {annual_h2_production_kg:>12,.1f} kg/yr |")
        print(f"  |  Total Capital Cost:    ${p['total_capital_cost']:>12,.0f}       |")
        print(f"  |  Plant Life:            {p['plant_life']:>12d} years   |")
        print(f"  |  Discount Rate:         {p['discount_rate']*100:>12.1f}%       |")
        print(f"  ================================================")
        print(f"  |  Cost Breakdown ($/kg H2):                   |")
        print(f"  |    Capital:             ${capital_per_kg:>12.2f}       |")
        print(f"  |    Fixed Operating:     ${fixed_op_per_kg:>12.2f}       |")
        print(f"  |    Feedstock:           ${feedstock_per_kg:>12.2f}       |")
        print(f"  |    Utilities:           ${utilities_per_kg:>12.2f}       |")
        print(f"  |    Other:               ${other_per_kg:>12.2f}       |")
        print(f"  ================================================")
        print(f"  |  LCOH ($/kg H2):        ${lcoh:>12.2f}       |")
        print(f"  ================================================")

    return result


def h2_volume_to_mass(h2_volume_m3, temperature_C=35.0, pressure_atm=1.0):
    """
    Convert H2 gas volume (m³) to mass (kg) using ideal gas law.

    Parameters
    ----------
    h2_volume_m3 : float
        H2 volume in m³.
    temperature_C : float
        Gas temperature in °C.
    pressure_atm : float
        Gas pressure in atm.

    Returns
    -------
    float
        H2 mass in kg.
    """
    # H2 molar mass: 2.016 g/mol
    # Ideal gas: PV = nRT
    # n = PV / RT
    R = 8.314e-5  # m³·atm/(mol·K)
    T = temperature_C + 273.15
    n_moles = (pressure_atm * h2_volume_m3) / (R * T)
    mass_kg = n_moles * 2.016e-3  # Convert g to kg
    return mass_kg


def estimate_annual_h2_from_simulation(daily_h2_m3, capacity_factor=0.90,
                                       temperature_C=35.0):
    """
    Estimate annual H2 production in kg from daily simulation output.

    Parameters
    ----------
    daily_h2_m3 : float
        Average daily H2 production in m³/day from simulation.
    capacity_factor : float
        Annual capacity factor.
    temperature_C : float
        Gas temperature for density calculation.

    Returns
    -------
    float
        Annual H2 production in kg/year.
    """
    annual_h2_m3 = daily_h2_m3 * 365 * capacity_factor
    annual_h2_kg = h2_volume_to_mass(annual_h2_m3, temperature_C)
    return annual_h2_kg

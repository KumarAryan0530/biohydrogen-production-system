"""
Biohydrogen ADM1 Model
======================
Refactored from PyADM1 (Sadrimajd et al., 2021) for biohydrogen production.

Key modifications from standard ADM1:
  - Methanogenesis disabled (Rho_11 = Rho_12 = 0)
  - pH set as controllable parameter (buffered reactor assumption)
  - Temperature-dependent kinetics via Arrhenius correction
  - Synthetic constant influent for parameter studies
  - Class-based design replacing global state

Reference:
  Rosen, C., Jeppsson, U. (2006). Aspects on ADM1 implementation within
  the BSM2 framework. Dept. of Industrial Electrical Engineering and
  Automation, Lund University.
"""

import numpy as np
import scipy.integrate
import pandas as pd


# =============================================================================
# Default ADM1 Parameters (BSM2 values)
# =============================================================================

def get_default_params():
    """Return a dictionary of default ADM1 parameters from BSM2."""
    R = 0.083145  # bar.M^-1.K^-1
    T_base = 298.15  # K
    T_op = 308.15  # K (35°C default)

    params = {
        # Physical constants
        'R': R,
        'T_base': T_base,
        'T_op': T_op,
        'p_atm': 1.013,  # bar

        # Stoichiometric parameters
        'f_sI_xc': 0.1, 'f_xI_xc': 0.2, 'f_ch_xc': 0.2,
        'f_pr_xc': 0.2, 'f_li_xc': 0.3,
        'N_xc': 0.0376 / 14, 'N_I': 0.06 / 14,
        'N_aa': 0.007, 'N_bac': 0.08 / 14,
        'C_xc': 0.02786, 'C_sI': 0.03, 'C_ch': 0.0313,
        'C_pr': 0.03, 'C_li': 0.022, 'C_xI': 0.03,
        'C_su': 0.0313, 'C_aa': 0.03, 'C_fa': 0.0217,
        'C_bu': 0.025, 'C_pro': 0.0268, 'C_ac': 0.0313,
        'C_bac': 0.0313, 'C_va': 0.024, 'C_ch4': 0.0156,
        'f_fa_li': 0.95,
        'f_h2_su': 0.19, 'f_bu_su': 0.13,
        'f_pro_su': 0.27, 'f_ac_su': 0.41,
        'f_h2_aa': 0.06, 'f_va_aa': 0.23, 'f_bu_aa': 0.26,
        'f_pro_aa': 0.05, 'f_ac_aa': 0.40,
        'Y_su': 0.1, 'Y_aa': 0.08, 'Y_fa': 0.06,
        'Y_c4': 0.06, 'Y_pro': 0.04, 'Y_ac': 0.05, 'Y_h2': 0.06,

        # Biochemical rate parameters
        'k_dis': 0.5, 'k_hyd_ch': 10.0, 'k_hyd_pr': 10.0, 'k_hyd_li': 10.0,
        'K_S_IN': 1e-4,
        'k_m_su': 30.0, 'K_S_su': 0.5,
        'k_m_aa': 50.0, 'K_S_aa': 0.3,
        'k_m_fa': 6.0, 'K_S_fa': 0.4, 'K_I_h2_fa': 5e-6,
        'k_m_c4': 20.0, 'K_S_c4': 0.2, 'K_I_h2_c4': 1e-5,
        'k_m_pro': 13.0, 'K_S_pro': 0.1, 'K_I_h2_pro': 3.5e-6,
        'k_m_ac': 8.0, 'K_S_ac': 0.15, 'K_I_nh3': 0.0018,
        'k_m_h2': 35.0, 'K_S_h2': 7e-6,
        'k_dec_X_su': 0.02, 'k_dec_X_aa': 0.02, 'k_dec_X_fa': 0.02,
        'k_dec_X_c4': 0.02, 'k_dec_X_pro': 0.02,
        'k_dec_X_ac': 0.02, 'k_dec_X_h2': 0.02,

        # pH inhibition limits
        'pH_UL_aa': 5.5, 'pH_LL_aa': 4.0,
        'pH_UL_ac': 7.0, 'pH_LL_ac': 6.0,
        'pH_UL_h2': 6.0, 'pH_LL_h2': 5.0,

        # Physico-chemical
        'k_A_B_va': 1e10, 'k_A_B_bu': 1e10, 'k_A_B_pro': 1e10,
        'k_A_B_ac': 1e10, 'k_A_B_co2': 1e10, 'k_A_B_IN': 1e10,
        'k_L_a': 200.0,
        'k_p': 5e4,  # gas outlet friction

        # Reactor volumes
        'V_liq': 3400.0,  # m^3
        'V_gas': 300.0,   # m^3

        # Flow rate
        'q_ad': 178.4674,  # m^3.d^-1
    }

    # Derived temperature-dependent parameters (computed at T_op)
    params['K_a_va'] = 10 ** -4.86
    params['K_a_bu'] = 10 ** -4.82
    params['K_a_pro'] = 10 ** -4.88
    params['K_a_ac'] = 10 ** -4.76
    params['K_a_co2'] = 10 ** -6.35 * np.exp((7646 / (100 * R)) * (1 / T_base - 1 / T_op))
    params['K_a_IN'] = 10 ** -9.25 * np.exp((51965 / (100 * R)) * (1 / T_base - 1 / T_op))
    params['K_w'] = 10 ** -14.0 * np.exp((55900 / (100 * R)) * (1 / T_base - 1 / T_op))
    params['p_gas_h2o'] = 0.0313 * np.exp(5290 * (1 / T_base - 1 / T_op))
    params['K_H_co2'] = 0.035 * np.exp((-19410 / (100 * R)) * (1 / T_base - 1 / T_op))
    params['K_H_ch4'] = 0.0014 * np.exp((-14240 / (100 * R)) * (1 / T_base - 1 / T_op))
    params['K_H_h2'] = 7.8e-4 * np.exp(-4180 / (100 * R) * (1 / T_base - 1 / T_op))

    return params


def get_default_initial_state():
    """Return the default BSM2 initial state vector (38 states)."""
    return [
        0.012394,   # S_su
        0.0055432,  # S_aa
        0.10741,    # S_fa
        0.012333,   # S_va
        0.014003,   # S_bu
        0.017584,   # S_pro
        0.089315,   # S_ac
        2.51e-07,   # S_h2
        0.05549,    # S_ch4
        0.095149,   # S_IC
        0.094468,   # S_IN
        0.13087,    # S_I
        0.10792,    # X_xc
        0.020517,   # X_ch
        0.08422,    # X_pr
        0.043629,   # X_li
        0.31222,    # X_su
        0.93167,    # X_aa
        0.33839,    # X_fa
        0.33577,    # X_c4
        0.10112,    # X_pro
        0.67724,    # X_ac
        0.28484,    # X_h2
        17.2162,    # X_I
        1.08e-47,   # S_cation
        0.0052101,  # S_anion
        5.46e-08,   # S_H_ion
        0.012284,   # S_va_ion
        0.013953,   # S_bu_ion
        0.017511,   # S_pro_ion
        0.089035,   # S_ac_ion
        0.08568,    # S_hco3_ion
        0.0094689,  # S_co2
        0.001884,   # S_nh3
        0.092584,   # S_nh4_ion
        1.10e-05,   # S_gas_h2
        1.6535,     # S_gas_ch4
        0.01354,    # S_gas_co2
    ]


def get_default_influent():
    """
    Return a synthetic constant influent for biohydrogen studies.
    Glucose-rich feed representative of dark fermentation substrates.
    """
    return {
        'S_su': 5.0,     # High sugar (glucose) feed — kgCOD/m3
        'S_aa': 0.001,   # Minimal amino acids
        'S_fa': 0.001,
        'S_va': 0.0, 'S_bu': 0.0, 'S_pro': 0.0, 'S_ac': 0.0,
        'S_h2': 0.0, 'S_ch4': 0.0,
        'S_IC': 0.04, 'S_IN': 0.01, 'S_I': 0.02,
        'X_xc': 2.0, 'X_ch': 5.0, 'X_pr': 5.0, 'X_li': 3.0,
        'X_su': 0.0, 'X_aa': 0.0, 'X_fa': 0.0, 'X_c4': 0.0,
        'X_pro': 0.0, 'X_ac': 0.0, 'X_h2': 0.0, 'X_I': 10.0,
        'S_cation': 0.04, 'S_anion': 0.02,
    }


# State variable names in order
STATE_NAMES = [
    "S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac",
    "S_h2", "S_ch4", "S_IC", "S_IN", "S_I",
    "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa",
    "X_c4", "X_pro", "X_ac", "X_h2", "X_I",
    "S_cation", "S_anion", "S_H_ion",
    "S_va_ion", "S_bu_ion", "S_pro_ion", "S_ac_ion",
    "S_hco3_ion", "S_co2", "S_nh3", "S_nh4_ion",
    "S_gas_h2", "S_gas_ch4", "S_gas_co2"
]

INFLUENT_KEYS = [
    "S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac",
    "S_h2", "S_ch4", "S_IC", "S_IN", "S_I",
    "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa",
    "X_c4", "X_pro", "X_ac", "X_h2", "X_I",
    "S_cation", "S_anion"
]


# =============================================================================
# Core Model Class
# =============================================================================

class BiohydrogenADM1:
    """
    ADM1-based biohydrogen production model.

    Disables methanogenesis to simulate dark fermentation conditions
    where hydrogen accumulates instead of being consumed by methanogens.

    Parameters
    ----------
    target_ph : float
        Target reactor pH (buffered operation). Default 5.5.
    temperature_C : float
        Operating temperature in °C. Default 35.
    simulation_days : float
        Total simulation time in days. Default 50.
    dt : float
        Time step in days. Default 0.5.
    q_ad : float or None
        Volumetric flow rate (m³/d). If None, uses default.
    """

    def __init__(self, target_ph=5.5, temperature_C=35.0,
                 simulation_days=50.0, dt=0.5, q_ad=None):
        self.target_ph = target_ph
        self.temperature_C = temperature_C
        self.T_op = temperature_C + 273.15  # K
        self.simulation_days = simulation_days
        self.dt = dt

        # Load default parameters and adjust for temperature
        self.params = get_default_params()
        if q_ad is not None:
            self.params['q_ad'] = q_ad

        # Apply temperature corrections
        self._apply_temperature_corrections()

        # Influent
        self.influent = get_default_influent()
        self._build_influent_vector()

        # Initial state
        self.state = get_default_initial_state()

        # Override initial pH to match target
        self.state[26] = 10 ** (-self.target_ph)  # S_H_ion

        # Results storage
        self.results = None
        self.gas_results = None

    def _apply_temperature_corrections(self):
        """Apply Arrhenius-type temperature corrections to kinetic parameters."""
        p = self.params
        R = p['R']
        T_base = p['T_base']
        T_ref = 308.15  # Reference temperature (35°C)
        T = self.T_op

        # Temperature correction factor for kinetic rates
        # Using typical activation energy for anaerobic digestion (~50 kJ/mol)
        E_a = 50000.0  # J/mol (typical for AD processes)
        R_gas = 8.314  # J/(mol·K) universal gas constant
        temp_factor = np.exp((E_a / R_gas) * (1.0 / T_ref - 1.0 / T))

        # Scale all maximum uptake rates
        for key in ['k_m_su', 'k_m_aa', 'k_m_fa', 'k_m_c4', 'k_m_pro',
                     'k_m_ac', 'k_m_h2', 'k_dis', 'k_hyd_ch', 'k_hyd_pr', 'k_hyd_li']:
            p[key] *= temp_factor

        # Recompute temperature-dependent equilibrium constants
        p['K_a_co2'] = 10 ** -6.35 * np.exp((7646 / (100 * R)) * (1 / T_base - 1 / T))
        p['K_a_IN'] = 10 ** -9.25 * np.exp((51965 / (100 * R)) * (1 / T_base - 1 / T))
        p['K_w'] = 10 ** -14.0 * np.exp((55900 / (100 * R)) * (1 / T_base - 1 / T))
        p['p_gas_h2o'] = 0.0313 * np.exp(5290 * (1 / T_base - 1 / T))
        p['K_H_co2'] = 0.035 * np.exp((-19410 / (100 * R)) * (1 / T_base - 1 / T))
        p['K_H_ch4'] = 0.0014 * np.exp((-14240 / (100 * R)) * (1 / T_base - 1 / T))
        p['K_H_h2'] = 7.8e-4 * np.exp(-4180 / (100 * R) * (1 / T_base - 1 / T))

        # Store operating temperature
        p['T_op'] = T

    def _build_influent_vector(self):
        """Build the influent state vector from the influent dictionary."""
        self.state_input = [self.influent[k] for k in INFLUENT_KEYS]

    def _adm1_ode(self, t, state_zero):
        """
        ADM1 ODE system with methanogenesis DISABLED for biohydrogen.

        This is the core modification: Rho_11 (acetoclastic methanogenesis)
        and Rho_12 (hydrogenotrophic methanogenesis) are set to zero,
        causing hydrogen to accumulate in the system.
        """
        p = self.params

        # Unpack state variables
        S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac = state_zero[0:7]
        S_h2, S_ch4, S_IC, S_IN, S_I = state_zero[7:12]
        X_xc, X_ch, X_pr, X_li = state_zero[12:16]
        X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I = state_zero[16:24]
        S_cation, S_anion = state_zero[24:26]
        S_H_ion = state_zero[26]
        S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion = state_zero[27:31]
        S_hco3_ion, S_co2, S_nh3, S_nh4_ion = state_zero[31:35]
        S_gas_h2, S_gas_ch4, S_gas_co2 = state_zero[35:38]

        # Unpack influent
        si = self.state_input

        # Ensure non-negative concentrations
        S_su = max(S_su, 0); S_aa = max(S_aa, 0); S_fa = max(S_fa, 0)
        S_va = max(S_va, 0); S_bu = max(S_bu, 0); S_pro = max(S_pro, 0)
        S_ac = max(S_ac, 0); S_h2 = max(S_h2, 0); S_ch4 = max(S_ch4, 0)
        S_IN = max(S_IN, 1e-12); S_H_ion = max(S_H_ion, 1e-15)
        X_su = max(X_su, 0); X_aa = max(X_aa, 0); X_fa = max(X_fa, 0)
        X_c4 = max(X_c4, 0); X_pro = max(X_pro, 0)
        X_ac = max(X_ac, 0); X_h2 = max(X_h2, 0)
        S_gas_h2 = max(S_gas_h2, 0); S_gas_ch4 = max(S_gas_ch4, 0)
        S_gas_co2 = max(S_gas_co2, 0)

        # pH inhibition — use target pH for buffered reactor
        S_H_ion_eff = 10 ** (-self.target_ph)

        K_pH_aa = 10 ** (-1 * (p['pH_LL_aa'] + p['pH_UL_aa']) / 2.0)
        nn_aa = 3.0 / (p['pH_UL_aa'] - p['pH_LL_aa'])
        K_pH_ac = 10 ** (-1 * (p['pH_LL_ac'] + p['pH_UL_ac']) / 2.0)
        n_ac = 3.0 / (p['pH_UL_ac'] - p['pH_LL_ac'])
        K_pH_h2 = 10 ** (-1 * (p['pH_LL_h2'] + p['pH_UL_h2']) / 2.0)
        n_h2 = 3.0 / (p['pH_UL_h2'] - p['pH_LL_h2'])

        I_pH_aa = (K_pH_aa ** nn_aa) / (S_H_ion_eff ** nn_aa + K_pH_aa ** nn_aa)
        I_pH_ac = (K_pH_ac ** n_ac) / (S_H_ion_eff ** n_ac + K_pH_ac ** n_ac)
        I_pH_h2 = (K_pH_h2 ** n_h2) / (S_H_ion_eff ** n_h2 + K_pH_h2 ** n_h2)

        I_IN_lim = 1.0 / (1.0 + (p['K_S_IN'] / S_IN))
        I_h2_fa = 1.0 / (1.0 + (S_h2 / p['K_I_h2_fa']))
        I_h2_c4 = 1.0 / (1.0 + (S_h2 / p['K_I_h2_c4']))
        I_h2_pro = 1.0 / (1.0 + (S_h2 / p['K_I_h2_pro']))
        I_nh3 = 1.0 / (1.0 + (S_nh3 / p['K_I_nh3']))

        I_5 = I_pH_aa * I_IN_lim
        I_6 = I_5
        I_7 = I_pH_aa * I_IN_lim * I_h2_fa
        I_8 = I_pH_aa * I_IN_lim * I_h2_c4
        I_9 = I_8
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro
        I_11 = I_pH_ac * I_IN_lim * I_nh3
        I_12 = I_pH_h2 * I_IN_lim

        q_ad = p['q_ad']
        V_liq = p['V_liq']
        V_gas = p['V_gas']
        T_op = p['T_op']
        R = p['R']

        # Biochemical process rates
        Rho_1 = p['k_dis'] * X_xc
        Rho_2 = p['k_hyd_ch'] * X_ch
        Rho_3 = p['k_hyd_pr'] * X_pr
        Rho_4 = p['k_hyd_li'] * X_li
        Rho_5 = p['k_m_su'] * S_su / (p['K_S_su'] + S_su) * X_su * I_5
        Rho_6 = p['k_m_aa'] * (S_aa / (p['K_S_aa'] + S_aa)) * X_aa * I_6
        Rho_7 = p['k_m_fa'] * (S_fa / (p['K_S_fa'] + S_fa)) * X_fa * I_7
        Rho_8 = p['k_m_c4'] * (S_va / (p['K_S_c4'] + S_va)) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8
        Rho_9 = p['k_m_c4'] * (S_bu / (p['K_S_c4'] + S_bu)) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9
        Rho_10 = p['k_m_pro'] * (S_pro / (p['K_S_pro'] + S_pro)) * X_pro * I_10

        # =====================================================================
        # BIOHYDROGEN MODIFICATION: Disable methanogenesis
        # =====================================================================
        # Rho_11: Acetoclastic methanogenesis (acetate → CH4 + CO2) = DISABLED
        # Rho_12: Hydrogenotrophic methanogenesis (H2 + CO2 → CH4) = DISABLED
        # This causes hydrogen to accumulate rather than being consumed
        Rho_11 = 0.0  # DISABLED — no methane from acetate
        Rho_12 = 0.0  # DISABLED — no methane from hydrogen
        # =====================================================================

        Rho_13 = p['k_dec_X_su'] * X_su
        Rho_14 = p['k_dec_X_aa'] * X_aa
        Rho_15 = p['k_dec_X_fa'] * X_fa
        Rho_16 = p['k_dec_X_c4'] * X_c4
        Rho_17 = p['k_dec_X_pro'] * X_pro
        Rho_18 = p['k_dec_X_ac'] * X_ac
        Rho_19 = p['k_dec_X_h2'] * X_h2

        # Gas phase
        p_gas_h2 = S_gas_h2 * R * T_op / 16
        p_gas_ch4 = S_gas_ch4 * R * T_op / 64
        p_gas_co2 = S_gas_co2 * R * T_op

        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p['p_gas_h2o']
        q_gas = p['k_p'] * (p_gas - p['p_atm'])
        if q_gas < 0:
            q_gas = 0

        # Gas transfer
        Rho_T_8 = p['k_L_a'] * (S_h2 - 16 * p['K_H_h2'] * p_gas_h2)
        Rho_T_9 = p['k_L_a'] * (S_ch4 - 64 * p['K_H_ch4'] * p_gas_ch4)
        Rho_T_10 = p['k_L_a'] * (S_co2 - p['K_H_co2'] * p_gas_co2)

        S_nh4_ion = S_IN - S_nh3
        S_co2 = S_IC - S_hco3_ion

        # Differential equations — soluble matter (eq 1-12)
        diff_S_su = q_ad / V_liq * (si[0] - S_su) + Rho_2 + (1 - p['f_fa_li']) * Rho_4 - Rho_5
        diff_S_aa = q_ad / V_liq * (si[1] - S_aa) + Rho_3 - Rho_6
        diff_S_fa = q_ad / V_liq * (si[2] - S_fa) + p['f_fa_li'] * Rho_4 - Rho_7
        diff_S_va = q_ad / V_liq * (si[3] - S_va) + (1 - p['Y_aa']) * p['f_va_aa'] * Rho_6 - Rho_8
        diff_S_bu = q_ad / V_liq * (si[4] - S_bu) + (1 - p['Y_su']) * p['f_bu_su'] * Rho_5 + (1 - p['Y_aa']) * p['f_bu_aa'] * Rho_6 - Rho_9
        diff_S_pro = q_ad / V_liq * (si[5] - S_pro) + (1 - p['Y_su']) * p['f_pro_su'] * Rho_5 + (1 - p['Y_aa']) * p['f_pro_aa'] * Rho_6 + (1 - p['Y_c4']) * 0.54 * Rho_8 - Rho_10
        diff_S_ac = q_ad / V_liq * (si[6] - S_ac) + (1 - p['Y_su']) * p['f_ac_su'] * Rho_5 + (1 - p['Y_aa']) * p['f_ac_aa'] * Rho_6 + (1 - p['Y_fa']) * 0.7 * Rho_7 + (1 - p['Y_c4']) * 0.31 * Rho_8 + (1 - p['Y_c4']) * 0.8 * Rho_9 + (1 - p['Y_pro']) * 0.57 * Rho_10 - Rho_11

        # H2 balance (now includes production terms, no consumption by Rho_12)
        diff_S_h2 = q_ad / V_liq * (si[7] - S_h2) + (1 - p['Y_su']) * p['f_h2_su'] * Rho_5 + (1 - p['Y_aa']) * p['f_h2_aa'] * Rho_6 + (1 - p['Y_fa']) * 0.3 * Rho_7 + (1 - p['Y_c4']) * 0.15 * Rho_8 + (1 - p['Y_c4']) * 0.2 * Rho_9 + (1 - p['Y_pro']) * 0.43 * Rho_10 - Rho_12 - Rho_T_8

        diff_S_ch4 = q_ad / V_liq * (si[8] - S_ch4) + (1 - p['Y_ac']) * Rho_11 + (1 - p['Y_h2']) * Rho_12 - Rho_T_9

        # Carbon balance
        s_1 = (-p['C_xc'] + p['f_sI_xc'] * p['C_sI'] + p['f_ch_xc'] * p['C_ch'] + p['f_pr_xc'] * p['C_pr'] + p['f_li_xc'] * p['C_li'] + p['f_xI_xc'] * p['C_xI'])
        s_5 = (-p['C_su'] + (1 - p['Y_su']) * (p['f_bu_su'] * p['C_bu'] + p['f_pro_su'] * p['C_pro'] + p['f_ac_su'] * p['C_ac']) + p['Y_su'] * p['C_bac'])
        s_6 = (-p['C_aa'] + (1 - p['Y_aa']) * (p['f_va_aa'] * p['C_va'] + p['f_bu_aa'] * p['C_bu'] + p['f_pro_aa'] * p['C_pro'] + p['f_ac_aa'] * p['C_ac']) + p['Y_aa'] * p['C_bac'])
        s_7 = (-p['C_fa'] + (1 - p['Y_fa']) * 0.7 * p['C_ac'] + p['Y_fa'] * p['C_bac'])
        s_8 = (-p['C_va'] + (1 - p['Y_c4']) * 0.54 * p['C_pro'] + (1 - p['Y_c4']) * 0.31 * p['C_ac'] + p['Y_c4'] * p['C_bac'])
        s_9 = (-p['C_bu'] + (1 - p['Y_c4']) * 0.8 * p['C_ac'] + p['Y_c4'] * p['C_bac'])
        s_10 = (-p['C_pro'] + (1 - p['Y_pro']) * 0.57 * p['C_ac'] + p['Y_pro'] * p['C_bac'])
        s_11 = (-p['C_ac'] + (1 - p['Y_ac']) * p['C_ch4'] + p['Y_ac'] * p['C_bac'])
        s_12 = ((1 - p['Y_h2']) * p['C_ch4'] + p['Y_h2'] * p['C_bac'])
        s_2 = (-p['C_ch'] + p['C_su'])
        s_3 = (-p['C_pr'] + p['C_aa'])
        s_4 = (-p['C_li'] + (1 - p['f_fa_li']) * p['C_su'] + p['f_fa_li'] * p['C_fa'])
        s_13 = (-p['C_bac'] + p['C_xc'])

        Sigma = (s_1 * Rho_1 + s_2 * Rho_2 + s_3 * Rho_3 + s_4 * Rho_4 +
                 s_5 * Rho_5 + s_6 * Rho_6 + s_7 * Rho_7 + s_8 * Rho_8 +
                 s_9 * Rho_9 + s_10 * Rho_10 + s_11 * Rho_11 + s_12 * Rho_12 +
                 s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19))

        diff_S_IC = q_ad / V_liq * (si[9] - S_IC) - Sigma - Rho_T_10
        diff_S_IN = q_ad / V_liq * (si[10] - S_IN) + (p['N_xc'] - p['f_xI_xc'] * p['N_I'] - p['f_sI_xc'] * p['N_I'] - p['f_pr_xc'] * p['N_aa']) * Rho_1 - p['Y_su'] * p['N_bac'] * Rho_5 + (p['N_aa'] - p['Y_aa'] * p['N_bac']) * Rho_6 - p['Y_fa'] * p['N_bac'] * Rho_7 - p['Y_c4'] * p['N_bac'] * Rho_8 - p['Y_c4'] * p['N_bac'] * Rho_9 - p['Y_pro'] * p['N_bac'] * Rho_10 - p['Y_ac'] * p['N_bac'] * Rho_11 - p['Y_h2'] * p['N_bac'] * Rho_12 + (p['N_bac'] - p['N_xc']) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        diff_S_I = q_ad / V_liq * (si[11] - S_I) + p['f_sI_xc'] * Rho_1

        # Particulate matter (eq 13-24)
        diff_X_xc = q_ad / V_liq * (si[12] - X_xc) - Rho_1 + Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19
        diff_X_ch = q_ad / V_liq * (si[13] - X_ch) + p['f_ch_xc'] * Rho_1 - Rho_2
        diff_X_pr = q_ad / V_liq * (si[14] - X_pr) + p['f_pr_xc'] * Rho_1 - Rho_3
        diff_X_li = q_ad / V_liq * (si[15] - X_li) + p['f_li_xc'] * Rho_1 - Rho_4
        diff_X_su = q_ad / V_liq * (si[16] - X_su) + p['Y_su'] * Rho_5 - Rho_13
        diff_X_aa = q_ad / V_liq * (si[17] - X_aa) + p['Y_aa'] * Rho_6 - Rho_14
        diff_X_fa = q_ad / V_liq * (si[18] - X_fa) + p['Y_fa'] * Rho_7 - Rho_15
        diff_X_c4 = q_ad / V_liq * (si[19] - X_c4) + p['Y_c4'] * Rho_8 + p['Y_c4'] * Rho_9 - Rho_16
        diff_X_pro = q_ad / V_liq * (si[20] - X_pro) + p['Y_pro'] * Rho_10 - Rho_17
        diff_X_ac = q_ad / V_liq * (si[21] - X_ac) + p['Y_ac'] * Rho_11 - Rho_18
        diff_X_h2 = q_ad / V_liq * (si[22] - X_h2) + p['Y_h2'] * Rho_12 - Rho_19
        diff_X_I = q_ad / V_liq * (si[23] - X_I) + p['f_xI_xc'] * Rho_1

        # Ions
        diff_S_cation = q_ad / V_liq * (si[24] - S_cation)
        diff_S_anion = q_ad / V_liq * (si[25] - S_anion)

        # DAE states (set to 0 for ODE implementation, solved algebraically)
        diff_S_H_ion = 0
        diff_S_va_ion = 0
        diff_S_bu_ion = 0
        diff_S_pro_ion = 0
        diff_S_ac_ion = 0
        diff_S_hco3_ion = 0
        diff_S_co2 = 0
        diff_S_nh3 = 0
        diff_S_nh4_ion = 0

        # Gas phase
        diff_S_gas_h2 = (q_gas / V_gas * -1 * S_gas_h2) + (Rho_T_8 * V_liq / V_gas)
        diff_S_gas_ch4 = (q_gas / V_gas * -1 * S_gas_ch4) + (Rho_T_9 * V_liq / V_gas)
        diff_S_gas_co2 = (q_gas / V_gas * -1 * S_gas_co2) + (Rho_T_10 * V_liq / V_gas)

        return [
            diff_S_su, diff_S_aa, diff_S_fa, diff_S_va, diff_S_bu, diff_S_pro, diff_S_ac,
            diff_S_h2, diff_S_ch4, diff_S_IC, diff_S_IN, diff_S_I,
            diff_X_xc, diff_X_ch, diff_X_pr, diff_X_li, diff_X_su, diff_X_aa, diff_X_fa,
            diff_X_c4, diff_X_pro, diff_X_ac, diff_X_h2, diff_X_I,
            diff_S_cation, diff_S_anion, diff_S_H_ion,
            diff_S_va_ion, diff_S_bu_ion, diff_S_pro_ion, diff_S_ac_ion,
            diff_S_hco3_ion, diff_S_co2, diff_S_nh3, diff_S_nh4_ion,
            diff_S_gas_h2, diff_S_gas_ch4, diff_S_gas_co2
        ]

    def _dae_solve(self, state):
        """
        Solve algebraic (DAE) equations for ion states and hydrogen balance.
        Adapted from PyADM1 DAESolve() using Newton-Raphson.
        """
        p = self.params
        tol = 1e-12
        maxIter = 1000

        S_va = max(state[3], 0)
        S_bu = max(state[4], 0)
        S_pro = max(state[5], 0)
        S_ac = max(state[6], 0)
        S_h2 = max(state[7], 1e-15)
        S_IC = state[9]
        S_IN = max(state[10], 1e-12)
        S_cation = state[24]
        S_anion = state[25]
        S_H_ion = max(state[26], 1e-15)
        S_gas_h2 = max(state[35], 0)

        X_su = max(state[16], 0)
        X_aa = max(state[17], 0)
        X_fa = max(state[18], 0)
        X_c4 = max(state[19], 0)
        X_pro = max(state[20], 0)
        X_h2 = max(state[22], 0)

        # Override pH to target (buffered reactor)
        S_H_ion = 10 ** (-self.target_ph)

        # Compute ion states at target pH
        S_va_ion = p['K_a_va'] * S_va / (p['K_a_va'] + S_H_ion)
        S_bu_ion = p['K_a_bu'] * S_bu / (p['K_a_bu'] + S_H_ion)
        S_pro_ion = p['K_a_pro'] * S_pro / (p['K_a_pro'] + S_H_ion)
        S_ac_ion = p['K_a_ac'] * S_ac / (p['K_a_ac'] + S_H_ion)
        S_hco3_ion = p['K_a_co2'] * S_IC / (p['K_a_co2'] + S_H_ion)
        S_nh3 = p['K_a_IN'] * S_IN / (p['K_a_IN'] + S_H_ion)

        pH = -np.log10(S_H_ion)

        # DAE solver for S_h2 (Newton-Raphson) from Rosen et al. (2006)
        eps = 1e-7
        prevS_H_ion = S_H_ion

        K_pH_aa = 10 ** (-1 * (p['pH_LL_aa'] + p['pH_UL_aa']) / 2.0)
        nn_aa = 3.0 / (p['pH_UL_aa'] - p['pH_LL_aa'])
        K_pH_h2 = 10 ** (-1 * (p['pH_LL_h2'] + p['pH_UL_h2']) / 2.0)
        n_h2 = 3.0 / (p['pH_UL_h2'] - p['pH_LL_h2'])

        S_h2delta = 1.0
        j = 1
        while (abs(S_h2delta) > tol and j <= maxIter):
            I_pH_aa = (K_pH_aa ** nn_aa) / (prevS_H_ion ** nn_aa + K_pH_aa ** nn_aa)
            I_pH_h2 = (K_pH_h2 ** n_h2) / (prevS_H_ion ** n_h2 + K_pH_h2 ** n_h2)
            I_IN_lim = 1 / (1 + (p['K_S_IN'] / S_IN))
            I_h2_fa = 1 / (1 + (S_h2 / p['K_I_h2_fa']))
            I_h2_c4 = 1 / (1 + (S_h2 / p['K_I_h2_c4']))
            I_h2_pro = 1 / (1 + (S_h2 / p['K_I_h2_pro']))

            S_su_val = max(state[0], 0)
            S_aa_val = max(state[1], 0)
            S_fa_val = max(state[2], 0)

            Rho_5 = p['k_m_su'] * (S_su_val / (p['K_S_su'] + S_su_val)) * X_su * I_pH_aa * I_IN_lim
            Rho_6 = p['k_m_aa'] * (S_aa_val / (p['K_S_aa'] + S_aa_val)) * X_aa * I_pH_aa * I_IN_lim
            Rho_7 = p['k_m_fa'] * (S_fa_val / (p['K_S_fa'] + S_fa_val)) * X_fa * I_pH_aa * I_IN_lim * I_h2_fa
            Rho_8 = p['k_m_c4'] * (S_va / (p['K_S_c4'] + S_va)) * X_c4 * (S_va / (S_bu + S_va + eps)) * I_pH_aa * I_IN_lim * I_h2_c4
            Rho_9 = p['k_m_c4'] * (S_bu / (p['K_S_c4'] + S_bu)) * X_c4 * (S_bu / (S_bu + S_va + eps)) * I_pH_aa * I_IN_lim * I_h2_c4
            Rho_10 = p['k_m_pro'] * (S_pro / (p['K_S_pro'] + S_pro)) * X_pro * I_pH_aa * I_IN_lim * I_h2_pro
            # Rho_12 = 0  # DISABLED — no hydrogenotrophic methanogenesis

            p_gas_h2_local = S_gas_h2 * p['R'] * p['T_op'] / 16
            Rho_T_8 = p['k_L_a'] * (S_h2 - 16 * p['K_H_h2'] * p_gas_h2_local)

            S_h2delta = (p['q_ad'] / p['V_liq'] * (self.state_input[7] - S_h2) +
                        (1 - p['Y_su']) * p['f_h2_su'] * Rho_5 +
                        (1 - p['Y_aa']) * p['f_h2_aa'] * Rho_6 +
                        (1 - p['Y_fa']) * 0.3 * Rho_7 +
                        (1 - p['Y_c4']) * 0.15 * Rho_8 +
                        (1 - p['Y_c4']) * 0.2 * Rho_9 +
                        (1 - p['Y_pro']) * 0.43 * Rho_10 -
                        0 -  # Rho_12 disabled
                        Rho_T_8)

            S_h2gradeq = (-1.0 / p['V_liq'] * p['q_ad'] - p['k_L_a'] -
                         0.3 * (1 - p['Y_fa']) * p['k_m_fa'] * S_fa_val / (p['K_S_fa'] + S_fa_val) * X_fa * I_pH_aa / (1 + p['K_S_IN'] / S_IN) / ((1 + S_h2 / p['K_I_h2_fa']) ** 2) / p['K_I_h2_fa'])

            if abs(S_h2gradeq) > 1e-15:
                S_h2 = S_h2 - S_h2delta / S_h2gradeq
            if S_h2 <= 0:
                S_h2 = tol
            j += 1

        S_nh4_ion = S_IN - S_nh3
        S_co2 = S_IC - S_hco3_ion

        # Update state
        state[7] = S_h2
        state[26] = S_H_ion
        state[27] = S_va_ion
        state[28] = S_bu_ion
        state[29] = S_pro_ion
        state[30] = S_ac_ion
        state[31] = S_hco3_ion
        state[32] = S_co2
        state[33] = S_nh3
        state[34] = S_nh4_ion

        return state

    def simulate(self):
        """
        Run the biohydrogen simulation.

        Returns
        -------
        results : pd.DataFrame
            Time-series DataFrame with all state variables and gas outputs.
        """
        p = self.params
        dt = self.dt
        n_steps = int(self.simulation_days / dt)
        times = np.linspace(0, self.simulation_days, n_steps + 1)

        state = list(self.state)  # Copy initial state
        results_list = [list(state)]
        gas_h2_flow = [0.0]
        gas_total_flow = [0.0]
        q_h2_list = [0.0]

        import time
        start_t = time.time()

        for i in range(n_steps):
            if i % max(1, n_steps // 10) == 0:
                print(f"      Simulating step {i}/{n_steps} (t={times[i]:.2f}d), elapsed: {time.time()-start_t:.1f}s")
            t_span = [times[i], times[i + 1]]

            try:
                sol = scipy.integrate.solve_ivp(
                    self._adm1_ode, t_span, state,
                    method='BDF', max_step=dt,
                    rtol=1e-5, atol=1e-8
                )

                if sol.success and sol.y.shape[1] > 0:
                    state = [sol.y[j, -1] for j in range(len(state))]
                else:
                    # If integration fails, keep previous state
                    pass
            except Exception:
                # If integration fails, keep previous state
                pass

            # Solve DAE
            state = self._dae_solve(state)

            # Calculate gas outputs
            S_gas_h2 = max(state[35], 0)
            S_gas_ch4 = max(state[36], 0)
            S_gas_co2 = max(state[37], 0)

            p_gas_h2 = S_gas_h2 * p['R'] * p['T_op'] / 16
            p_gas_ch4 = S_gas_ch4 * p['R'] * p['T_op'] / 64
            p_gas_co2 = S_gas_co2 * p['R'] * p['T_op']
            p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + p['p_gas_h2o']

            q_gas = p['k_p'] * (p_gas - p['p_atm'])
            if q_gas < 0:
                q_gas = 0

            q_h2 = q_gas * (p_gas_h2 / max(p_gas, 1e-10))
            if q_h2 < 0:
                q_h2 = 0

            results_list.append(list(state))
            gas_h2_flow.append(q_h2)
            gas_total_flow.append(q_gas)
            q_h2_list.append(q_h2)

        # Build results DataFrame
        df = pd.DataFrame(results_list, columns=STATE_NAMES)
        df['time'] = times
        df['q_gas'] = gas_total_flow
        df['q_h2'] = q_h2_list
        df['pH'] = self.target_ph
        df['temperature_C'] = self.temperature_C

        # Cumulative H2 production (m³) — trapezoidal integration
        df['cumulative_h2'] = np.cumsum(np.array(q_h2_list) * dt)

        self.results = df
        return df

    def get_total_h2_yield(self):
        """
        Get total cumulative hydrogen yield in m³.

        Returns
        -------
        float
            Total H2 produced over the simulation period in m³.
        """
        if self.results is None:
            self.simulate()
        return float(self.results['cumulative_h2'].iloc[-1])

    def get_average_h2_rate(self):
        """
        Get average H2 production rate in m³/day.

        Returns
        -------
        float
            Average H2 flow rate in m³/day.
        """
        if self.results is None:
            self.simulate()
        return float(self.results['q_h2'].mean())


def run_single_simulation(target_ph=5.5, temperature_C=35.0,
                          simulation_days=50.0, dt=0.5, q_ad=None):
    """
    Convenience function to run a single biohydrogen simulation.

    Parameters
    ----------
    target_ph : float
        Target reactor pH.
    temperature_C : float
        Operating temperature in °C.
    simulation_days : float
        Simulation duration in days.
    dt : float
        Time step in days.
    q_ad : float or None
        Flow rate override.

    Returns
    -------
    model : BiohydrogenADM1
        The model instance with results.
    results : pd.DataFrame
        Simulation results DataFrame.
    """
    model = BiohydrogenADM1(
        target_ph=target_ph,
        temperature_C=temperature_C,
        simulation_days=simulation_days,
        dt=dt,
        q_ad=q_ad
    )
    results = model.simulate()
    return model, results

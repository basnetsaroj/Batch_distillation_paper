# =============================================================================
# Thermodynamic property correlations and interpolation for ethanol–water mixtures.
# =============================================================================

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator

# ============================================================================
# A. Density of ethanol-water mixture
''' Source: Washburn, E. W. (Ed.), International Critical Tables of Numerical Data
    of Physics, Chemistry and Technology, Vol. 3, McGraw-Hill, New York, 1926–1932.'''
# Rows: ethanol concentration (wt %)
# Columns: temperature (°C)
# Density units: g/cm³
# Temperature range: 10–40 °C
weight_percent = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], dtype=float)
temperatures = np.array([10, 15, 20, 25, 30, 35, 40], dtype=float)
density_data = np.array([
    [0.99970, 0.99910, 0.99820, 0.99705, 0.99565, 0.99403, 0.99222],
    [0.99095, 0.99029, 0.98935, 0.98814, 0.98667, 0.98498, 0.98308],
    [0.98390, 0.98301, 0.98184, 0.98040, 0.97872, 0.97682, 0.97472],
    [0.97797, 0.97666, 0.97511, 0.97331, 0.97130, 0.96908, 0.96667],
    [0.97249, 0.97065, 0.96861, 0.96636, 0.96392, 0.96131, 0.95853],
    [0.96662, 0.96421, 0.96165, 0.95892, 0.95604, 0.95303, 0.94988],
    [0.95974, 0.95683, 0.95379, 0.95064, 0.94738, 0.94400, 0.94052],
    [0.95159, 0.94829, 0.94491, 0.94143, 0.93787, 0.93422, 0.93048],
    [0.94235, 0.93879, 0.93515, 0.93145, 0.92767, 0.92382, 0.91989],
    [0.93223, 0.92849, 0.92469, 0.92082, 0.91689, 0.91288, 0.90881],
    [0.92159, 0.91773, 0.91381, 0.90982, 0.90577, 0.90165, 0.89747],
    [0.91052, 0.90656, 0.90255, 0.89847, 0.89434, 0.89013, 0.88586],
    [0.89924, 0.89520, 0.89110, 0.88696, 0.88275, 0.87848, 0.87414],
    [0.88771, 0.88361, 0.87945, 0.87524, 0.87097, 0.86664, 0.86224],
    [0.87599, 0.87184, 0.86763, 0.86337, 0.85905, 0.85467, 0.85022],
    [0.86405, 0.85985, 0.85561, 0.85131, 0.84695, 0.84254, 0.83806],
    [0.85194, 0.84769, 0.84341, 0.83908, 0.83470, 0.83027, 0.82576],
    [0.83948, 0.83522, 0.83093, 0.82658, 0.82218, 0.81772, 0.81320],
    [0.82652, 0.82225, 0.81795, 0.81360, 0.80920, 0.80476, 0.80026],
    [0.81276, 0.80850, 0.80422, 0.79989, 0.79553, 0.79112, 0.78668],
    [0.79782, 0.79358, 0.78932, 0.78504, 0.78073, 0.77639, 0.77201] ], dtype=float)

# Create a two-dimensional linear interpolator for density.
interp_func = RegularGridInterpolator((weight_percent, temperatures), 
                density_data, method="linear", bounds_error=False, fill_value=None)

# Density interpolation function 
def get_den(wt_percent, temperature):
    # Limit inputs to the range of the tabulated data.
    wt_percent = np.clip(wt_percent, weight_percent.min(), weight_percent.max() )
    temperature = np.clip(temperature, temperatures.min(), temperatures.max() )
    return float(interp_func((wt_percent, temperature)))

# Pure-component density functions
def get_pure_water_den(temperature):    # Density of pure water (0 wt% ethanol)
    return get_den(0.0, temperature)
def get_pure_ethanol_den(temperature):  # Density of pure ethanol (100 wt% ethanol)
    return get_den(100.0, temperature)


# ============================================================================
# B. Specific Heat Capacity and Heat of Vaporization
# ---------------------------------------------------------------------------
# 1. Water
''' Source: Osborne, N. S., Stimson, H. F., and Ginnings, D. C.,
    "Measurements of Heat Capacity and Heat of Vaporization of Water in the Range 0° to 100°C'''

# 1.1 Specific heat capacity of water (J/kg·K)
def Cp_water(T_cel):   
    CP_w = (4.169828 + 0.000364 * (T_cel + 100)**(5.26) * (10**(-10)) + 0.046709 * (10)**(-0.036 * T_cel))
    return CP_w * 1e3  

# 1.2 Latent Heat of vaporization of water (J/kg)
def Hvap_water(T_cel): 
    T_k = 273.16 + T_cel
    x = 5.1463 - 1540 / T_k
    # gamma is calculated in international joules per gram (J/g)
    gamma_int = 2500.5 - 2.3233 * T_cel - 10**x  
    # As the paper used, 1 international joule = 1.00019 absolute joules
    Lvw = gamma_int * 1000.19  # Convert to J/kg
    return Lvw


# 2. Ethanol ---------------------------------------------------------------------------
# 2.1 Specific heat capacity of ethanol (J/kg·K)
''' Source: Miyazawa, T., Kondo, S., Suzuki, T., and Sato, H., "Specific Heat Capacity at Constant 
    Pressure of Ethanol by Flow Calorimetry," Journal of Chemical and Engineering Data. '''
# Valid temperature range: 265–348 K.
# Data from Tables 6 and 7 are combined for interpolation.
T_e_k = np.array([
    # Table 7 (up to 320 K)
    265.005, 270.007, 273.004, 275.013, 280.000, 285.007, 290.009, 294.986, 300.028, 305.003, 310.006, 320.006,
    # Table 6 (above 320 K)
    325.003, 330.004, 333.008, 335.003, 338.013, 339.990, 342.997, 345.001, 348.003])
shc_e = np.array([
    # Table 7
    2.215, 2.234, 2.247, 2.265, 2.287, 2.312, 2.337, 2.356, 2.381, 2.417, 2.451, 2.545,
    # Table 6
    2.602, 2.638, 2.668, 2.692, 2.746, 2.767, 2.823, 2.841, 2.867])
# Linear interpolation of tabulated ethanol heat-capacity data.
fit_e = interp1d(T_e_k, shc_e, kind='linear', fill_value="extrapolate")
def Cp_ethanol(t_c):
    t_k = 273.15 + t_c 
    return fit_e(t_k)*1e3  # J/kg·K

# 2.2 Heat of vaporization of ethanol (J/kg)
''' Source: Fiock, E. F., Ginnings, D. C., and Holton, W. B. (1931), "Calorimetric Determinations of Thermal 
    Properties of Methyl Alcohol, Ethyl Alcohol, and Benzene," Bureau of Standards Journal of Research, 6(5), 881-900.'''
# Valid temperature range: 40–110°C.
def Hvap_ethanol(T_C): 
    # Empirical equation yielding Int. J/g
    L_e_g = -0.004067 * (240 - T_C)**2 + 2.198 * (240 - T_C) + 165.83 * (240 - T_C)**(1/4)
    # Convert legacy International Joules per gram to standard J/kg. 1 Int. J/g = 1000.165 J/kg
    Lve = L_e_g * 1000.165
    return Lve

# ============================================================================
# 3. Specific heat capacity of Ethanol-water mixture 
''' Source: Gaulhofer, A., Kolbe, B., and Gmehling, J., "Thermodynamic properties of ethanol and water — III.
    Description of the different excess functions using an empirical model," Fluid Phase Equilibria.'''
# Valid temperature range: 25–150°C.
# Columns correspond to k = 1, 2, 3, 4, 5, 6. Rows correspond to a1, a2, a3 parameters for Equation 9
A_PARAMS = {
    1: {"a1": 83.20264,   "a2": -101.1905,  "a3": 18.67453},
    2: {"a1": 9.015980,   "a2": 5.047225,   "a3": -2.620565},
    3: {"a1": -20.97006,  "a2": 10.61763,   "a3": -0.5380303},
    4: {"a1": 17.27665,   "a2": -13.74101,  "a3": 1.835535},
    5: {"a1": -15.28097,  "a2": 15.17921,   "a3": -2.596300},
    6: {"a1": -14.95465,  "a2": 14.57610,   "a3": -2.376966}}
R = 8.314               # Ideal gas constant, J/(mol*K)
T0 = 363.15             # Reference temperature in Kelvin
M_ethanol = 46.07e-3    # Molecular mass of ethanol (kg/mol)
M_water = 18.015e-3     # Molecular mass of water (kg/mol)
# Calculate Legendre polynomials used in the excess heat-capacity model.
def get_legendre_polynomials(x1, x2, p=7):
    z = x1 - x2
    Q = np.zeros(p)
    Q[0] = 1.0  
    if p > 1:
        Q[1] = z 
    for k in range(2, p): 
        Q[k] = ((2 * k - 1) * z * Q[k - 1] - (k - 1) * Q[k - 2]) / k
    return Q
# Calculate the excess heat capacity of the ethanol–water mixture.
def calculate_cp_excess(x1, x2, T_kelvin):
    Ti = T_kelvin / T0  
    Q = get_legendre_polynomials(x1, x2, p=7) 
    sum_term = 0.0
    # Loop from k = 1 to 6 to match A_PARAMS keys and Legendre polynomial orders directly
    for k in range(1, 7):
        a1 = A_PARAMS[k]["a1"]
        a2 = A_PARAMS[k]["a2"]
        a3 = A_PARAMS[k]["a3"]
        a_k_cp = a1 + 2.0 * a2 * Ti + 6.0 * a3 * (Ti ** 2)
        sum_term += a_k_cp * Q[k] 
    cp_E_over_R = -x1 * x2 * sum_term
    return cp_E_over_R * R
# Specific heat capacity of the ethanol–water mixture (J/kg·K)
def Cp_mixture(x_ethanol, T_celsius):
    x1 = x_ethanol
    x2 = 1.0 - x_ethanol
    T_k = T_celsius + 273.15
    # 1. Pure component properties
    cp1 = Cp_ethanol(T_celsius)*M_ethanol # J/(mol*K)
    cp2 = Cp_water(T_celsius)* M_water
    cp_ideal = x1 * cp1 + x2 * cp2
    # Non-ideal excess term 
    cp_excess = calculate_cp_excess(x1, x2, T_k)
    # Total real heat capacity 
    cp_real = cp_ideal + cp_excess
    M_mixture = x_ethanol * M_ethanol + (1.0 - x_ethanol) * M_water
    cp_real_mass = cp_real / M_mixture      # Convert J/(mol*K) to J/(kg*K)
    return cp_real_mass 

# ============================================================================
# 4. Latent heat of vaporization of Ethanol-water mixture
''' Source: R. H. Perry and D. W. Green, Eds., Perry's Chemical Engineers' Handbook, 7th ed. New York, NY, USA: McGraw-Hill, 1999.'''
# Pure ethanol saturated enthalpy (kJ/kg)
T_eth  = np.array([350, 360, 370, 380, 390, 400])
hf_eth = np.array([199.9, 230.1, 262.2, 295.1, 329.1, 364.2])
hg_eth = np.array([1161.9, 1178.4, 1193.9, 1208.4, 1221.5, 1233.6])

# Pure water saturated enthalpy (kJ/kg)
T_water  = np.array([350, 355, 360, 365, 370, 373.15, 375, 380, 385, 390, 400])
hf_water = np.array([321.7, 342.7, 363.7, 384.7, 405.8, 419.1, 426.8, 448.0, 469.2, 490.4, 532.9])
hg_water = np.array([2639, 2647, 2655, 2663, 2671, 2676, 2679, 2687, 2694, 2702, 2716])

# Cubic spline interpolation (kJ/kg)
hf_eth_spline   = CubicSpline(T_eth, hf_eth)
hg_eth_spline   = CubicSpline(T_eth, hg_eth)
hf_water_spline = CubicSpline(T_water, hf_water)
hg_water_spline = CubicSpline(T_water, hg_water)

def Mixture_Latent_Heat(x_liquid, y_vapour, temp_celsius):
    T_K = temp_celsius + 273.15
    # Pure liquid enthalpies (kJ/kg -> J/mol)
    hL_eth   = hf_eth_spline(T_K) * 1000 * M_ethanol
    hL_water = hf_water_spline(T_K) * 1000 * M_water
    # Pure vapour enthalpies (kJ/kg -> J/mol)
    hV_eth   = hg_eth_spline(T_K) * 1000 * M_ethanol
    hV_water = hg_water_spline(T_K) * 1000 * M_water
    
    # Liquid mixture enthalpy
    HL = x_liquid*hL_eth + (1-x_liquid)*hL_water
    # Vapour mixture enthalpy (ideal vapour)
    HV = y_vapour*hV_eth + (1-y_vapour)*hV_water
    Hlvm = HV - HL  #- h_excess(x_liquid, T_K)  # Include excess enthalpy for rigorous calculation
    return Hlvm  # J/mol



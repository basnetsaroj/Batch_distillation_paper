
# =============================================================================
# Batch Distillation with Reflux and Four Equilibrium Stages
# Ethanol–Water Separation Using the NRTL Model
# =============================================================================

import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve 
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
import Dist_thermo_properties as tp 

# ===================================================================================
# Constants ---------
P1      = 101325            # Atmospheric pressure at sea level (Pa)
g       = 9.81              # Acceleration due to gravity (m/s^2)
M_air   = 28.97e-3          # Molar mass of air (kg/mol)
R       = 8.314             # Universal gas constant (J/(mol·K))
Molw_e  = 46.07e-3          # Molecular weight of ethanol (kg/mol)
Molw_w  = 18.015e-3         # Molecular weight of water (kg/mol)

# ===================================================================================
# Input Parameters ------------------------------------------------------------------
# Experimental conditions and model parameters for Run 3
N_stages     = 4                    # Number of equilibrium stages
Initial_vol  = 3.5                  # Initial feed volume (L)
FM_i_t       = 17                   # Feed mixture initial temperature (°C)
C_e_initial  = 30                   # Initial ethanol concentration, v/v% 
# Average inlet temperature of reflux-condenser cooling water from three measurements (°C).
T_in_ref     = (23.4+25.3+25.5)/3   
T_out_ref    = (35+36.8+39.5)/3     # Outlet temp. (3 data points' average)
V_w_ref      = 900/((82+92+104)/3)  # Volumetric flow rate of reflux-condenser water (mL/s). (3 data points' average)
Total_testing_time  = 53            # Total experimental time from heating start to end of distillation (min).
Time_to_reach_bpt   = 15 + 40/60    # Time from heating start to initial boiling point (min).
h       = 1300                      # Elevation used for the atmospheric-pressure correction (m). # Source: Google Earth.
T_room  = 18                        # Ambient Room temperature (°C)

# =============================================================================
# Initial Feed Composition ------
# Determine the initial ethanol and water amounts while accounting for ethanol–water volume contraction during mixing.
V_final_target = Initial_vol                    # L, required final mixture volume
rho_w0      = tp.get_pure_water_den(FM_i_t)     # Density of pure water at initial temperature
rho_e0      = tp.get_pure_ethanol_den(FM_i_t)   # Density of pure ethanol at initial temperature
V_e_initial = Initial_vol * C_e_initial/100     # Initial ethanol volume (L)
M_e_initial = V_e_initial * rho_e0              # Initial ethanol mass (kg)

def calculate_final_volume(V_w):
    M_wi = V_w * rho_w0
    M_total = M_e_initial + M_wi
    ethanol_wt_percent = 100 * M_e_initial / M_total
    rho_mix = tp.get_den(ethanol_wt_percent, FM_i_t)
    V_f = M_total / rho_mix
    return V_f

tolerance = 1e-4 
low = 0.0
high = V_final_target
# Solve for the initial water volume that gives the target final mixture volume.
while high - low > tolerance:
    mid = (low + high) / 2
    if calculate_final_volume(mid) < V_final_target:
        low = mid
    else:
        high = mid
V_w_required = (low + high) / 2     
# Initial Feed Quantities ----------------
V_w_initial = V_w_required              # Initial water volume (L)
M_w_initial = V_w_initial * rho_w0      # Initial water mass (kg) 
M_total     = M_e_initial + M_w_initial # Total mass of the mixture (kg)
n_e_initial = M_e_initial / Molw_e      # Initial moles of ethanol
n_w_initial = M_w_initial / Molw_w      # Initial moles of water
n_total_i   = n_e_initial + n_w_initial # Total moles of the mixture
x_e0        = n_e_initial / n_total_i   # Initial mole fraction of ethanol in the feed mixture
print(f"Pure water volume: {V_w_required:.6f} L")
print(f"\nEthanol mole in liquid: {n_e_initial:.4f} mol") 
print(f"Water mole in liquid  : {n_w_initial:.4f} mol")
print(f"Total initial moles   : {n_total_i:.4f} mol")
print(f"Initial mole fraction of ethanol in liquid: {x_e0:.4f}")


# ===================================================================================
# Local Atmospheric Pressure and Pure-Component Boiling Points
# =================================================================================== 
T_k_r   = T_room + 273.15   # Convert to Kelvin 
# Estimate local atmospheric pressure using the barometric formula.
expon   = -(g * M_air * h) / (R * T_k_r)
P2      = P1 * math.exp(expon)      # Atmospheric pressure at Kathmandu (Pa)
P2_atm  = P2 / P1                   # Convert Pa to atm
P2_mmHg = P2_atm * 760              # Convert atm to mmHg
print(f"\nAtmospheric pressure at Kathmandu: {P2_mmHg:.2f} mmHg or {P2_atm:.2f} atm or {P2:.2f} Pa")

# Clausius-Clapeyron equation ---------------------------------------------
delta_Hvap_e = tp.Hvap_ethanol(T_room)*Molw_e  # Enthalpy of vap. of ethanol (J/mol)
T1_e = 351.45          # bpt of ethanol at sea level (K, 78.3°C)
delta_Hvap_w = tp.Hvap_water(T_room)*Molw_w    # Enthalpy of vap. of water (J/mol)
T1_w = 373.15          # bpt of water at sea level (K, 100°C)
# Compute lhs of Clausius-Clapeyron equation 
lhs = math.log(P2 / P1)
# Compute boiling point for ethanol
inv_T2_e = (1 / T1_e) - (lhs * R) / delta_Hvap_e
T2_e = 1 / inv_T2_e
T2_Celsius_e = T2_e - 273.15
# Compute boiling point for water
inv_T2_w = (1 / T1_w) - (lhs * R) / delta_Hvap_w
T2_w = 1 / inv_T2_w
T2_Celsius_w = T2_w - 273.15
# Display results
print(f"The boiling point of ethanol at Kathmandu is approximately: {T2_Celsius_e:.2f}°C.")
print(f"The boiling point of water at Kathmandu is approximately  : {T2_Celsius_w:.2f}°C.")


# ===================================================================================
# Vapor-Liquid Equilibrium (VLE) Curve ----------------------------------------------
# ===================================================================================
# Calculate ethanol vapor pressure using Antoine equation. These constants are for ethanol and water in mmHg and Celsius. 
# Antoine coefficients source: Gmehling and Onken, Vapor-Liquid Equilibrium Data Collection, 
# DECHEMA Chemistry Data Series, Vol. 1, Frankfurt, 1977.
def antoine_ethanol(T_C):
    a_e = 10 ** (7.58670 - 1281.590 / (T_C + 193.768)) # Valid temperature range: approximately 78–203 °C.
    return a_e
def antoine_water(T_C):
    a_w = 10 ** (8.07131 - 1730.630 / (T_C + 233.426)) # tempr: 1 to 100 °C
    return a_w

# NRTL binary interaction parameters for the ethanol-water system. Ethanol(1) + Water(2)
# Source: Guevara Luna et al., "Experimental Data and New Binary Interaction Parameters 
# for Ethanol Water VLE at Low Pressures Using NRTL and UNIQUAC," TECCIENCIA, Vol. 13, No. 24, pp. 17–26, 2018.
alpha = 0.30  
A12 = -0.801 
B12 = 246.2 
A21 = 3.458 
B21 = -586.1
def tau(T_C):
    T_K = T_C + 273.15       # Convert temperature from °C to K
    tau12 = A12 + B12 / T_K
    tau21 = A21 + B21 / T_K
    return tau12, tau21 

# NRTL Activity Coefficients ---------------------------
def activity_coeff_nrtl(x_etoh, tau12, tau21, alpha):
    x_water = 1.0 - x_etoh
    G12 = np.exp(-alpha * tau12)
    G21 = np.exp(-alpha * tau21)
    ln_gamma_etoh = (x_water**2 * (tau21 * (G21 / (x_etoh + x_water * G21))**2 + tau12 * G12 / (x_water + x_etoh * G12)**2 ) )
    ln_gamma_water = (x_etoh**2 * (tau12 * (G12 / (x_water + x_etoh * G12))**2 + tau21 * G21 / (x_etoh + x_water * G21)**2 ) )
    gamma_etoh = np.exp(ln_gamma_etoh)
    gamma_water = np.exp(ln_gamma_water)
    return gamma_etoh, gamma_water

# Calculate the ethanol vapor-phase mole fraction using NRTL and modified Raoult's law.
def calculate_y_etoh(x_etoh, T_C):
    x_water = 1.0 - x_etoh
    # Temperature-dependent NRTL interaction parameters.
    tau12, tau21 = tau(T_C)
    # Pure-component saturation pressures (mmHg)
    P_sat_etoh  = antoine_ethanol(T_C)
    P_sat_water = antoine_water(T_C)
    # NRTL activity coefficients
    gamma_etoh, gamma_water = activity_coeff_nrtl(x_etoh, tau12, tau21, alpha)
    # Total pressure from modified Raoult's law (mmHg).
    P_total = (gamma_etoh * x_etoh * P_sat_etoh + gamma_water * x_water * P_sat_water )
    # Ethanol vapor-phase mole fraction
    y_etoh = (gamma_etoh * x_etoh * P_sat_etoh) / P_total
    return y_etoh

# Determine mixture boiling temperature by solving modified Raoult's law equation at local atmospheric pressure. 
def boiling_temperature(x_etoh):
    x_water = 1 - x_etoh
    def equation(T):
        tau12, tau21 = tau(T)
        gamma_etoh, gamma_water = activity_coeff_nrtl(x_etoh, tau12, tau21, alpha)
        P_sat_etoh = antoine_ethanol(T)
        P_sat_water = antoine_water(T)
        return (gamma_etoh * x_etoh * P_sat_etoh + gamma_water * x_water * P_sat_water - P2_mmHg)
    # Initial temperature estimate based on mole-fraction-weighted boiling points of pure ethanol and water at local pressure.
    T_guess = (x_etoh*T2_Celsius_e + (1-x_etoh)*T2_Celsius_w)
    T_solution = fsolve(equation,T_guess)[0]
    return T_solution
T_bpt0 = boiling_temperature(x_e0)
print(f"Initial, boiling point of mixture: {T_bpt0:.2f}°C at {P2_mmHg:.2f} mmHg.")

# Vapor-Liquid Equilibrium Relation ---
def equilibrium(x):
    T = boiling_temperature(x)
    return calculate_y_etoh(x, T)


# ========================================================================================
# Heating Power Calculation
# ========================================================================================
# 1. Sensible heat required to heat the boiler pot.
m_boiler    = 6.056    # mass of the boiler pot
shc_boiler  = 511      # specific heat capacity of the boiler pot    
Q_pot = m_boiler * shc_boiler * (T_bpt0 - FM_i_t) 

# 2. Sensible heat required to heat the feed mixture, from the initial temperature to its initial boiling point.
T_range = np.append(np.arange(FM_i_t, T_bpt0, 1.0), T_bpt0)
Q_s = 0.0  
for i in range(len(T_range) - 1):
    T = T_range[i]
    dT = T_range[i + 1] - T
    # Calculate the mixture specific heat capacity at the current temperature (J/(kg·K)).
    cp_mix_real = tp.Cp_mixture(x_e0, T)
    # Calculate the sensible heat required over this temperature increment.
    dQ_step = M_total * cp_mix_real * dT
    Q_s += dQ_step

# 3. Average heating power required to reach the initial boiling point.
PowerI = (Q_s + Q_pot) / (Time_to_reach_bpt * 60)  
print(f"Sensible heat: {Q_s:.2f} J and Boiler pot heat: {Q_pot:.2f} J")

# Reflux Ratio Calculation --------------------------------------------------------------
# Water density evaluated at the average condenser-water temperature and converted from kg/L to kg/m³.
water_dens  = tp.get_pure_water_den((T_out_ref+T_in_ref)/2) * 1e3   
m_w_r       = 1e-6*V_w_ref* water_dens                  # Mass flow rate of water in condenser (kg/s)
Shc_water   = tp.Cp_water((T_out_ref+T_in_ref)/2)       # J/(kg·K)
H_water_ref = Shc_water*m_w_r*(T_out_ref - T_in_ref)    # Heat removed by the reflux-condenser cooling water (W).
# Reflux ratio estimated from the condenser heat duty and effective heating power.
RR          = H_water_ref/(PowerI - H_water_ref)                    


# ===================================================================================
# Distillation Column Analysis: Bottoms Composition–Distillate Composition Relationship
# ===================================================================================
def build_xw_to_xd_relation():
    # Distillation parameters
    N_p         = 400       # Number of points
    x_D_start   = 0.8943    # Starting distillate composition
    x_D_end     = 0.0001    # Ending distillate composition
    # Use nonlinear spacing to provide greater resolution near the initial distillate composition.
    s = np.linspace(0, 1, N_p)
    s_power = 3                     # >1 shifts density to start; <1 shifts density to end
    x_D_values = x_D_start + (x_D_end - x_D_start) * s**s_power
    x_W_values = np.zeros(N_p)      # To store bottoms composition
    # Pre-compute the isobaric VLE curve for interpolation during the stage-by-stage calculation.
    x_eq = np.linspace(0, 1, N_p) 
    y_eq = np.array([equilibrium(x)for x in x_eq])
    # Calculate relationship between bottoms composition (x_W) and distillate composition (x_D) using number of stages.
    for i, x_D in enumerate(x_D_values):
        m = RR / (RR + 1)       # Slope of the rectifying operating line.
        b = x_D / (RR + 1)      # Intercept of the rectifying operating line.
        current_x = x_D
        for _ in range(N_stages):
            current_y = m * current_x + b 
            current_x = np.interp(current_y, y_eq, x_eq)
        x_W_values[i] = current_x 
    # Sort the calculated data so that x_W is strictly increasing, as required by the cubic spline interpolator.
    sorted_indices  = np.argsort(x_W_values)
    x_W_sorted      = x_W_values[sorted_indices]
    x_D_sorted      = x_D_values[sorted_indices]
    # Remove duplicate x_W values before constructing the spline.
    x_W_unique, unique_indices = np.unique(x_W_sorted, return_index=True)
    x_D_unique = x_D_sorted[unique_indices]
    # Cubic-spline interpolation of x_D as a function of x_W.
    xw_to_xd_Cspline = CubicSpline(x_W_unique, x_D_unique) 
    return (xw_to_xd_Cspline, x_W_values, x_D_values, x_eq, y_eq)
# Build the bottoms-distillate composition relationship.
xw_to_xd_spline, x_W_values, x_D_values, x_eq, y_eq = build_xw_to_xd_relation()

# ===================================================================================
# Visualization 
# -----------------------------------------------------------------------------------
# Boiling temperature corresponding to the initial feed composition.
T_vis = boiling_temperature(x_e0)
# Select a representative point from the x_W-x_D relationship for the diagram.
idx = 280
x_D_vis = x_D_values[idx]
x_W_vis = x_W_values[idx]

# McCabe-Thiele diagram ----------------- 
m_vis = RR / (RR + 1)
b_vis = x_D_vis / (RR + 1)
x_vis = x_eq
y_eq_vis = y_eq
x_op_vis = np.linspace(0, x_D_vis, 100)
y_op_vis = m_vis * x_op_vis + b_vis

# Construct the stage-stepping points for the McCabe-Thiele diagram.
x_points, y_points = [x_D_vis], [x_D_vis]
current_x = x_D_vis
for _ in range(N_stages):
    # Step vertically to the operating line.
    current_y = m_vis * current_x + b_vis
    x_points.append(current_x)
    y_points.append(current_y)
    # Step horizontally to the equilibrium curve.
    current_x = np.interp(current_y, y_eq_vis, x_vis)
    x_points.append(current_x)
    y_points.append(current_y)
# Complete the final step to the bottoms composition.
x_points.append(current_x)
y_points.append(0)

# =============================================================================
# Vapor-Liquid Equilibrium (VLE) Curve
# Generate equilibrium data over the full liquid-composition range.
x_etoh_range = np.linspace(0, 1, 200)
T_r = np.array([boiling_temperature(x) for x in x_etoh_range])
y_etoh_range = np.array([calculate_y_etoh(x, T) for x, T in zip(x_etoh_range, T_r)])

plt.figure(figsize=(6, 6))
plt.plot(x_etoh_range,y_etoh_range,'g-',linewidth=2,label='Equilibrium Curve')
plt.plot([0,1],[0,1],'k--',label=r'$x_D=x_W$')
plt.xlabel('Mole Fraction of Ethanol in Liquid ($x_W$)', fontsize=12)
plt.ylabel('Mole Fraction of Ethanol in Vapor ($x_D$)', fontsize=12)
plt.title(f'Vapor-Liquid Equilibrium Curve for Ethanol-Water\n' f'at {P2_atm:.3f} atm', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
#plt.show()

# McCabe-Thiele Diagram -------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.plot(x_vis, y_eq_vis, label='Equilibrium curve', linewidth=2, color="#017F01")
plt.plot(x_op_vis, y_op_vis, label=f'Operating line (RR={RR:.3f})', linewidth=1.5, color="#1f77b4")
plt.plot(x_vis, x_vis, '--', label=r'$x_D = x_W$', alpha=1, color='black')
plt.plot(x_points, y_points, 'o-', markersize=6, linewidth=1.5, alpha=1,color='red', 
         markerfacecolor='white', markeredgewidth=1.5, label=f'{N_stages} stages')
plt.plot([x_points[-1], x_points[-1]], [0, y_points[-2]], '-', color='red')
# Circle marker for x_D
plt.scatter([x_D_vis], [x_D_vis], color='red', zorder=5, s=100,
            edgecolor='black', marker='o', label=r'$x_D$='+f'{x_D_vis:.3f}')
# sq marker for x_W
plt.scatter([x_points[-1]], [0], color='red', zorder=5, s=100, 
            edgecolor='black', marker='s', label=r'$x_W$='+f'{x_points[-1]:.3f}')
plt.xlabel('Mole Fraction of Ethanol in Liquid ($x_{W}$)', fontsize=12)
plt.ylabel('Mole Fraction of Ethanol in Vapor ($x_{D}$)', fontsize=12)
plt.title(f'McCabe-Thiele Diagram\n($x_D$ = {x_D_vis:.3f}, $x_W$ = {x_points[-1]:.3f})', fontsize=12)
plt.grid(True)
plt.legend(loc='lower right', fontsize=12)
#plt.show()

# Bottoms-Distillate Composition (xW-xD) Relationship -------------------------------------------
plt.figure(figsize=(6, 6))
plt.plot(x_W_values, x_D_values, 'b-', linewidth=2, label='Numerical Solution')
#plt.plot(x_W_unique, xw_to_xd_spline(x_W_unique), 'r--', linewidth=1.5, label='Cubic Spline Fit')
plt.scatter(x_W_vis, x_D_vis, color='red', s=100, zorder=5)
plt.xlabel('Mole Fraction of Ethanol in Liquid ($x_W$)', fontsize=12)
plt.ylabel('Mole Fraction of Ethanol in Vapor ($x_D$)', fontsize=12)
plt.title(f'Bottoms-Distillate Concentration Relationship\n' 
          f'({N_stages} Stages, RR = {RR:.3f},' f'T = {T_vis:.2f}°C)')
plt.grid(True)
plt.legend(loc='lower right', fontsize=12) 
plt.annotate(f'($x_W$={x_W_vis:.3f}, $x_D$={x_D_vis:.3f})', (x_W_vis, x_D_vis),  xytext=(0.8, 0.7), 
             textcoords='axes fraction',ha='right', va='center',  fontsize=12, bbox=dict(boxstyle='round,pad=0.5', 
             fc='white'), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.tight_layout()
#plt.show()


# ===================================================================================
# Batch Distillation Simulation
# ===================================================================================
# Simulation Parameters ----
# Separation period after the initial boiling point is reached (min).
Sim_time    = Total_testing_time - Time_to_reach_bpt    
total_time  = 60*Sim_time                       # Total simulation time (s). 
time_step_size = 1                              # Time step (s)
num_steps   = int(total_time / time_step_size)  # Number of time steps 

# Initialize simulation arrays -------------------
time = np.zeros(num_steps)            # Time array (s)
x_D  = np.zeros(num_steps)            # Distillate composition
x_w  = np.zeros(num_steps)            # Bottoms composition
T_b = np.zeros(num_steps)             # Boiling temperature of the remaining liquid (°C).

# Quantities remaining in the boiler.
n_e_remain = np.zeros(num_steps)      # Moles of ethanol remaining
n_w_remain = np.zeros(num_steps)      # Moles of water remaining
n_t_remain = np.zeros(num_steps)      # Total moles remaining
Mf_e_remain = np.zeros(num_steps)     # Mole fraction ethanol in bottom-liquid
Mf_w_remain = np.zeros(num_steps)     # Mole fraction water in liquid

# Quantities collected in the distillate.
n_e_collected = np.zeros(num_steps)   # Moles of ethanol collected
n_w_collected = np.zeros(num_steps)   # Moles of water collected
n_t_collected = np.zeros(num_steps)   # Total moles collected
Mf_e_collected= np.zeros(num_steps)   # Mole fraction ethanol in distillate
M_e_collected = np.zeros(num_steps)   # Mass of ethanol collected (kg)
M_w_collected = np.zeros(num_steps)   # Mass of water collected (kg)
M_t_collected = np.zeros(num_steps)   # Total mass collected (kg)
E_wt_Percent  = np.zeros(num_steps)   # Ethanol weight percent in distillate (%)
V_e_collected = np.zeros(num_steps)   # Volume of ethanol collected (L)
V_w_collected = np.zeros(num_steps)   # Volume of water collected (L)
V_t_collected = np.zeros(num_steps)   # Total volume collected (L)
C_e_collected = np.zeros(num_steps)   # Concentration in collected distillate (v/v)
instant_vv_p  = np.zeros(num_steps)   # Instant v/v% concentration

# Initialize the simulation with the feed conditions ----------
n_e_remain[0] = n_e_initial
n_w_remain[0] = n_w_initial
n_t_remain[0] = n_total_i
Mf_e_remain[0] = x_e0
T_b[0] = boiling_temperature(Mf_e_remain[0])

# ===================================================================================
# Simulation loop 
# -----------------------------------------------------------------------------------
for step in range(1, num_steps):
    time[step] = step * time_step_size          # Current simulation time (s).
    x_w[step] = Mf_e_remain[step-1]             # Ethanol mole fraction in the remaining boiler liquid.
    T_b[step] = boiling_temperature(x_w[step])  # Boiling temperature of the remaining liquid at the local pressure.
    # Calculate the distillate ethanol mole fraction after the specified number of equilibrium stages.
    x_D[step] = xw_to_xd_spline(x_w[step])      
    y_d1 = equilibrium(x_w[step])               # Ethanol vapor mole fraction leaving the first equilibrium stage.
    Energy = PowerI * time_step_size            # Energy supplied during the time step (J).
    # Effective latent heat of the vaporized mixture (J/mol).
    latent_heat = tp.Mixture_Latent_Heat(x_w[step],y_d1,T_b[step])
    n_dist = Energy/((1 + RR) * latent_heat)    # Total moles of distillate produced during the time step.

    ## Calculate ethanol and water moles in the distillate produced during the current time step.
    current_step_ethanol= n_dist * x_D[step]       # Moles ethanol
    current_step_water  = n_dist * (1 - x_D[step]) # Moles water

    # Prevent the calculated withdrawal from exceeding the amount remaining in the boiler.
    if current_step_ethanol > n_e_remain[step-1]:
        current_step_ethanol = n_e_remain[step-1]   
    if current_step_water > n_w_remain[step-1]:
        current_step_water = n_w_remain[step-1]

    # Update cumulative distillate quantities.
    n_e_collected[step] = n_e_collected[step-1] + current_step_ethanol
    n_w_collected[step] = n_w_collected[step-1] + current_step_water
    n_t_collected[step] = n_e_collected[step] + n_w_collected[step]
    Mf_e_collected[step] = n_e_collected[step] / n_t_collected[step]
    # Update quantities remaining in the boiler.
    n_e_remain[step] = n_e_remain[step-1] - current_step_ethanol
    n_w_remain[step] = n_w_remain[step-1] - current_step_water
    n_t_remain[step] = n_e_remain[step] + n_w_remain[step]
    Mf_e_remain[step] = n_e_remain[step] / n_t_remain[step] 

    # Update collected mass, volume, and concentration
    M_e_collected[step] = n_e_collected[step] * Molw_e
    M_w_collected[step] = n_w_collected[step] * Molw_w
    M_t_collected[step] = M_e_collected[step] + M_w_collected[step]
    E_wt_Percent[step]  = 100 * M_e_collected[step] / M_t_collected[step]
    V_e_collected[step] = M_e_collected[step] / rho_e0          # Ethanol volume collected (L)
    V_w_collected[step] = M_w_collected[step] / rho_w0          # Water volume collected (L)
    # Distillate mixture density evaluated at the initial feed temperature (kg/L).
    dens_mixture        = tp.get_den(E_wt_Percent[step],FM_i_t)  
    # Total distillate volume calculated from its mass and mixture density (L).
    V_t_collected[step] = M_t_collected[step]/ dens_mixture      
    # Instantaneous ethanol concentration of the distillate collected during the current time step (v/v %).
    C_e_collected[step] = 100 * V_e_collected[step] /V_t_collected[step]   
    instant_vv_p[step]  = 100 * (V_e_collected[step] - V_e_collected[step-1]) / (V_t_collected[step] - V_t_collected[step-1])
    # Stop the simulation when the remaining boiler liquid is effectively depleted.
    if n_t_remain[step] <= 1e-6:
        t_stop = time[step]/60
        print(f"Simulation stopped at "f"{t_stop:.1f} minutes because the boiler was depleted.")
        num_steps = step + 1
        time = time[:num_steps]
        break


# ===================================================================================
# Simulation Summary and Results ----------------------------------------------------
print("\nDistillation Results and Summary --------------")
print(f"Operation Time          : {time[num_steps-1]/60:.2f} mins")
print(f"Power Input             : {PowerI:.2f} W")
print(f'Ref. Condenser Heat Duty: {H_water_ref:.2f} W')
print(f"Reflux Ratio (R)        : {RR:.3f}")
print(f"Initial Eth m.f. liquid : {Mf_e_remain[0]:.5f}")
print(f"Dist. Conc. at start    : {instant_vv_p[1]:.2f}%")
print(f"Distillate Conc. at end : {instant_vv_p[num_steps-1]:.2f}%")
print(f"Collected Eth equivalent: {V_e_collected[num_steps-1]*1000:.2f} mL")
print(f"Collected Wat equivalent: {V_w_collected[num_steps-1]*1000:.2f} mL")
print(f"Final Distillate Conc.  : {C_e_collected[num_steps-1]:.2f}%")
print(f"Total Dist. Collected   : {V_t_collected[num_steps-1]*1000:.2f} mL\n")

print(f"Total ethanol mole collected: {n_e_collected[num_steps-1]:.4f} mol")
print(f"Total water mole collected  : {n_w_collected[num_steps-1]:.4f} mol")    
print(f"Total moles collected       : {n_t_collected[num_steps-1]:.4f} mol")
print(f"Final mole fr. eth in dist  : {Mf_e_collected[num_steps-1]:.4f}")

# ---------------------------------------------------------------------
print('\nBoiling Temperature vs. Time')
indices = [0]  # Initial point
# Include one point every 60 s.
indices.extend(range(60, num_steps, 60))
# Add the final simulation point if it is not already included.
if indices[-1] != num_steps - 1:
    indices.append(num_steps - 1)
# Create table
df_bpt = pd.DataFrame({
    "Time (min)": np.round(Time_to_reach_bpt + time[indices] / 60, 2),
    "Boiling Temperature (°C)": np.round(T_b[indices], 2)})

print(df_bpt)


# ===================================================================================
# Experimental Data for Plotting and Comparison
# -----------------------------------------------------------------------------------
# Experimental time points ---
# Collection point durations (min).
d1 = 2+49/60
d2 = 2+31/60
d3 = 2+39/60
d4 = 3+19/60
d5 = 3+53/60
d = np.array([d1, d2, d3, d4, d5])

# Experimental time points measured from the start of the test (min).
t0 = Time_to_reach_bpt # 15:40 
t1 = 18+29/60   # 18:29 || t0 + d1
t2 = 22+30/60   # 22:30 
t3 = 28         # 28:00
t4 = 35+3/60    # 35:03
t5 = 41         # 41:00
t6 = 53         # 53:00

# Experimental interval average distillate concentration samples (v/v %).
instant_start_times = np.array([t0, t2, t3, t4, t5])
instant_end_times   = instant_start_times + d
exp_sample_conc_rt  = np.array([87, 86, 85, 82, 75])     # v/v%

# Experimental interval-avg distillate collection rate.
exp_rate = np.array([90/d1, 90/d2, 90/d3, 90/d4, 90/d5])   # mL/min

# Cumulative distillate concentration measured at the end of each collection interval (v/v %).
exp_sample_times_cumu = np.array([t1, t2, t3, t4, t5, t6])
exp_sample_conc_cumu  = np.array([87, 86, 85, 84, 83, 79])     

# Cumulative volume collected (mL)
exp_interval_times = exp_sample_times_cumu
exp_interval_vol   = np.array([90, 244, 447, 655, 813, 1049])


# ===================================================================================
# Results Tables and Model–Experiment Comparison ------------------------------------
# ===================================================================================

# Calculate model-predicted instantaneous collection rates from the cumulative component and total distillate volumes.
time_minutes = Time_to_reach_bpt + time/60
total_volume_ml = 1000 * V_t_collected
ethanol_volume_ml = 1000 * V_e_collected
water_volume_ml   = 1000 * V_w_collected
total_rate   = np.gradient(total_volume_ml, time_minutes)
ethanol_rate = np.gradient(ethanol_volume_ml, time_minutes)
water_rate   = np.gradient(water_volume_ml, time_minutes)

# -----------------------------------------------------
# 1. Interval-average distillate ethanol concentration. Model–experiment comparison
# -----------------------------------------------------
def time_to_index(T):
    """Convert experimental elapsed time (min) to simulation array indices.""" 
    idx = ((T - Time_to_reach_bpt) * 60 / time_step_size).astype(int)
    return np.clip(idx, 0, num_steps - 1)
inst_start_idx = time_to_index(instant_start_times)
inst_end_idx = time_to_index(instant_end_times)

# Calculate the model-predicted average distillate concentration over the same interval as the experimental measurement.
delta_ethanol_volume = (V_e_collected[inst_end_idx] - V_e_collected[inst_start_idx])
delta_total_volume = (V_t_collected[inst_end_idx] - V_t_collected[inst_start_idx])
theo_inst_conc = (100 * delta_ethanol_volume / delta_total_volume)
# Absolute percentage deviation between model and experimental values.
inst_conc_error = 100 * np.abs(theo_inst_conc - exp_sample_conc_rt) / theo_inst_conc
inst_mid_times = (instant_start_times + instant_end_times) / 2
print("\nInterval-average concentration comparison:")
# Create the model–experiment comparison table.
table_inst_conc = pd.DataFrame({
    "Sample No.": np.arange(1, len(exp_sample_conc_rt) + 1),
    "Time Interval(min)": [f"{s:.2f}: {e:.2f}" for s, e in zip(instant_start_times, instant_end_times)],
    "Average Time(min)": np.round(inst_mid_times, 2),
    "Experimental(v/v %)": exp_sample_conc_rt,
    "Model(v/v %)": np.round(theo_inst_conc, 2),
    "Absolute Percentage Deviation (%)": np.round(inst_conc_error, 2)})
print(table_inst_conc.to_string(index=False))


# -----------------------------------------------
# 2. Interval-average distillate collection rate. Model–experiment comparison
# -----------------------------------------------
# Calculate the model-predicted average collection rate over the same experimental interval.
rate_start_idx = time_to_index(instant_start_times)
rate_end_idx = time_to_index(instant_end_times)
theo_rate = (1000* (V_t_collected[rate_end_idx] - V_t_collected[rate_start_idx])/ (instant_end_times - instant_start_times))
# Absolute percentage deviation between model and experimental rates.
rate_error = 100 * np.abs(theo_rate - exp_rate) / theo_rate
print("\nInterval-average collection rate comparison:")
# Create the model–experiment comparison table.
table_rate = pd.DataFrame({
    "Sample No.": np.arange(1, len(exp_rate) + 1),
    "Time Interval(min)": [f"{s:.2f}: {e:.2f}" for s, e in zip(instant_start_times, instant_end_times)],
    "Average Time(min)": np.round(inst_mid_times, 2),
    "Experimental(mL/min)": np.round(exp_rate, 2),
    "Model(mL/min)": np.round(theo_rate, 2),
    "Absolute Percentage Deviation (%)": np.round(rate_error, 2)})
print(table_rate.to_string(index=False))


# -------------------------------------------
# 3. Cumulative distillate volume comparison
# -------------------------------------------
cumu_idx = time_to_index(exp_interval_times)
theo_cumu_vol = (1000* V_t_collected[cumu_idx])
# Absolute percentage deviation between model and experimental volumes.
vol_error = 100 * np.abs(theo_cumu_vol - exp_interval_vol) / theo_cumu_vol
print("\nCumulative volume comparison:")
# Create the cumulative-volume comparison table.
table_cumu_vol = pd.DataFrame({
    "Time(min)": exp_interval_times,
    "Experimental(mL)": exp_interval_vol,
    "Model(mL)": np.round(theo_cumu_vol, 2),
    "Absolute Percentage Deviation (%)": np.round(vol_error, 2)})
print(table_cumu_vol.to_string(index=False))


# -----------------------------------------------
# 4. Cumulative ethanol concentration comparison
# -----------------------------------------------
cumu_conc_idx = time_to_index(exp_sample_times_cumu)
theo_cumu_conc = (C_e_collected[cumu_conc_idx]) 
# Absolute percentage deviation between model and experimental concentrations.
conc_error = 100 * np.abs(theo_cumu_conc - exp_sample_conc_cumu) / theo_cumu_conc
print("\nCumulative concentration comparison:")
table_cumu_conc = pd.DataFrame({
    "Time(min)": exp_sample_times_cumu, 
    "Experimental(v/v%)": exp_sample_conc_cumu,
    "Model(v/v%)": np.round(theo_cumu_conc, 2),
    "Absolute Percentage Deviation (%)": np.round(conc_error, 2)})
print(table_cumu_conc.to_string(index=False))

# Save results to Excel file -------------------------------
# Define output Excel file path
output_file = r"C:\Users\saroj\Desktop\Engineering\Project\Paper\Code 3 NRTL\Run 3\Results_table.xlsx"
# Use ExcelWriter to create multiple sheets
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    table_inst_conc.to_excel(writer, sheet_name='Interval Avg Conc', index=False)
    table_rate.to_excel(writer, sheet_name='Interval Avg Rate', index=False)
    table_cumu_vol.to_excel(writer, sheet_name='Cumulative Volume', index=False)
    table_cumu_conc.to_excel(writer, sheet_name='Cumulative Eth Conc', index=False)


# ===================================================================================
# Final Visualization
# ===================================================================================
# -----------------------------------------------------
# 1. Instantaneous Distillate Ethanol Concentration vs time
plt.figure(figsize=(6, 6))
plt.gca().set_axisbelow(True)
plt.grid(True, alpha=0.5)
# Experimental interval-average concentration
for start, end, conc in zip(instant_start_times, instant_end_times, exp_sample_conc_rt):
    plt.hlines(conc, start, end, colors='black', linewidth=2, zorder=4)
    plt.scatter((start+end)/2, conc, color='black', s=70, zorder=5)
# Instantaneous model-predicted concentration 
plt.plot(Time_to_reach_bpt + time[1:num_steps]/60, instant_vv_p[1:num_steps], linewidth=2, color='red', zorder=2)
# Model-predicted interval-average concentration
for start, end, conc in zip(instant_start_times, instant_end_times, theo_inst_conc):
    plt.hlines(conc, start, end, colors='darkblue', linewidth=2, linestyles='-.', zorder=3)
# Custom Legend handles
exp_handle = Line2D([0],[0], color='black', linewidth=2, marker='o', markersize=8)
theo_avg_handle = Line2D([0],[0], color='darkblue', linewidth=2, linestyle='-.')
theo_inst_handle = Line2D([0],[0], color='red', linewidth=2)
plt.legend(handles=[exp_handle, theo_inst_handle, theo_avg_handle],labels=[
    'Experimental interval average concentration', 'Instantaneous concentration (model)', 'Interval average concentration (model)'])
plt.xlabel('Distillation Time (min)', fontsize=12)
plt.ylabel('Ethanol Concentration (v/v%)', fontsize=12)
#plt.title('Distillate Ethanol Concentration Tracking', fontsize=12)
plt.ylim(50, 96)
plt.tight_layout()


# ---------------------------------------------------------
# 2. Distillate Collection Rate vs time
plt.figure(figsize=(6, 6))
plt.gca().set_axisbelow(True)
plt.grid(True, alpha=0.5)
# Plot experimental interval-average collection rates.
for start, end, r in zip(instant_start_times, instant_end_times, exp_rate):
    plt.hlines(r, start, end, colors='black', linewidth=2, zorder=3)
    plt.scatter([(start+end)/2], [r], color='black', s=70, zorder=4)
plt.plot(time_minutes, total_rate,   linewidth=2, color='red',   zorder=2)
plt.plot(time_minutes, ethanol_rate, linewidth=2, linestyle='--', color='green', zorder=1)
plt.plot(time_minutes, water_rate,   linewidth=2, linestyle=':', color='blue',  zorder=1)
for start, end, r in zip(instant_start_times, instant_end_times, theo_rate):
    plt.hlines(r, start, end, colors='darkblue', linewidth=2, linestyles='-.', zorder=2)
# Custom Legend handles
exp_handle   = Line2D([0],[0], color='black', linewidth=2, marker='o', markersize=8)
total_handle = Line2D([0],[0], color='red', linewidth=2)
eth_handle   = Line2D([0],[0], color='green', linewidth=2, linestyle='--')
wat_handle   = Line2D([0],[0], color='blue', linewidth=2, linestyle=':')
avg_handle   = Line2D([0],[0], color='darkblue', linewidth=2, linestyle='-.')
plt.legend(handles=[exp_handle, total_handle, eth_handle, wat_handle, avg_handle], labels=['Experimental interval average', 
        'Total distillate rate (model)', 'Ethanol contribution (model)', 'Water contribution (model)', 'Interval average rate (model)'])
plt.xlabel('Distillation Time (min)', fontsize=12)
plt.ylabel('Collection Rate (mL/min)', fontsize=12)
#plt.title('Distillate Collection Rate Tracking', fontsize=12)
plt.tight_layout()


# ------------------------------------------
# 3. Cumulative Distillate Volume vs. Time
plt.figure(figsize=(6, 6))
plt.gca().set_axisbelow(True)
plt.scatter(exp_interval_times, exp_interval_vol, color='black', s=70, marker='o', zorder=4, label='Experimental data') 
plt.plot(Time_to_reach_bpt+time[1:num_steps]/60, 1000*V_t_collected[1:num_steps], label='Total distillate volume (model)', 
         linewidth=2, zorder=3, color='red', linestyle='-')
plt.plot(Time_to_reach_bpt+time[1:num_steps]/60, 1000*V_e_collected[1:num_steps], label='Ethanol contribution (model)', 
         linewidth=2,zorder=2, color='green', linestyle='--')
plt.plot(Time_to_reach_bpt+time[1:num_steps]/60, 1000*V_w_collected[1:num_steps], label='Water contribution (model)', 
         zorder=1, linewidth=2, color='blue', linestyle=':')
plt.xlabel('Distillation Time (min)', fontsize=12)
plt.ylabel('Cumulative Distillate Volume (mL)', fontsize=12)
#plt.title('Cumulative Distillate Volume Tracking', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()


# -------------------------------------------------------
# 4. Cumulative Distillate Ethanol Concentration vs. Time
plt.figure(figsize=(6, 6))
plt.gca().set_axisbelow(True) 
plt.plot(Time_to_reach_bpt + time[1:num_steps]/60, C_e_collected[1:num_steps], 
         label='Cumulative ethanol concentration (model)',zorder=1, linewidth=2, color='red')
exp_cum_times = exp_interval_times
exp_cum_conc  = exp_sample_conc_cumu 
plt.scatter(exp_cum_times, exp_cum_conc,color='black', s=70, marker='o',zorder=2, label='Experimental data')
plt.xlabel('Distillation Time (min)', fontsize=12)
plt.ylabel('Cumulative Ethanol Concentration (v/v%)', fontsize=12)
#plt.title('Cumulative Distillate Concentration Tracking', fontsize=12)
plt.legend()
plt.grid(True)
plt.ylim(30, 96)
plt.tight_layout()
#plt.show()

# --------------------------------------------
# 5. Boiling Temperature vs. Time
plt.figure(figsize=(6, 6))
plt.gca().set_axisbelow(True)
plt.grid(True, alpha=0.5)
time_minutes = Time_to_reach_bpt + time[1:num_steps]/60
plt.plot(time_minutes, T_b[1:num_steps], linewidth=2, color='red', zorder=2, label='Boiling Temperature (model)')
plt.xlabel('Distillation Time (min)', fontsize=12)
plt.ylabel('Boiling Temperature (°C)', fontsize=12)
#plt.title('Boiling Point vs Time', fontsize=12)
plt.grid(True)
plt.legend()
#plt.ylim(T2_Celsius_e, T2_Celsius_w)
plt.tight_layout()
#plt.show()



# ===================================================================================
# ------------------------------------ END ------------------------------------------
# ==================================================================================='''

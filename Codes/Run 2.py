
# ========================================================================================================
# Batch distillation with reflux & 4 stages: Determining Ethanol volume and concentration in Distillate
# ========================================================================================================

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from sklearn.metrics import r2_score

# =========================================================================
#               Input Parameters
# =========================================================================

# Run 2: all variables
No_of_stages        = 4             # Number of equilibrium stages
Initial_volume      = 3             # Initial volume (L)
Total_testing_time  = 38            # Total testing time (min)
Time_to_reach_bpt   = 14+5/60            # minutes
T_in_ref            = (19.3+22.1)/2 # two data points average
T_out_ref           = (31.8+32.1)/2   
V_w_ref             = 900/(60+17)   # ml/sec
FM_initial_temp     = 15.1          # Initial feed mixture temperature (Celsius)
C_ethanonl_initial  = 27            # Initial ethanol concentration (v/v)%   


# ========================================================================================
# Pressure and Boiling point Calculation for Ethanol and Water in Kathmandu(testing site)
# ========================================================================================
# Constants
h       = 1300              # Elevation of Kathmandu (m) from google earth
P1      = 101325            # Atmospheric pressure at sea level (Pa)
g       = 9.81              # Acceleration due to gravity (m/s^2)
M_air   = 28.97e-3          # Molar mass of dry air (kg/mol)
R       = 8.314             # Universal gas constant (J/(mol·K))
T_room  = 17                # Room temperature (Celsius)
T_k     = T_room + 273.15   # Convert to Kelvin

# Barometric formula to calculate pressure at Kathmandu
exponent= -(g * M_air * h) / (R * T_k)
P2      = P1 * math.exp(exponent)   # Atmospheric pressure at Kathmandu (Pa)
P2_atm  = P2 / 101325               # Convert Pa to atm
P2_mmHg = P2_atm * 760              # Convert atm to mmHg
print(f"Atmospheric pressure at Kathmandu: {P2_mmHg:.2f} mmHg or {P2_atm:.2f} atm or {P2:.2f} Pa")

# For Clausius-Clapeyron equation
delta_Hvap_e = 38500   # Enthalpy of vap. of eth (J/mol)
T1_e = 351.45          # bpt of ethanol at sea level (K, 78.3°C)

delta_Hvap_w = 40700   # Enthalpy of vap. of water (J/mol)
T1_w = 373.15          # bpt of water at sea level (K, 100°C)

# Compute lhs of Clausius-Clapeyron equation (same for both)
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
#              Vapor-Liquid Equilibrium (VLE) Curve Calculation
# ===================================================================================
# Use boiling point of water as reference temperature
T = T2_Celsius_w   

# Calculate ethanol vapor pressure using Antoine equation
def antoine_ethanol(T):
    a_e = 10 ** (7.58670 - 1281.590 / (T + 193.768)) 
    return a_e

def antoine_water(T):
    a_w = 10 ** (8.07131 - 1730.630 / (T + 233.426))
    return a_w

# Margules equation for activity coefficients
def activity_coeff(x_etoh, A12=1.6022, A21=0.7947):
    x_w = 1 - x_etoh
    ln_gamma_etoh = x_w**2 * (A12 + 2 * x_etoh * (A21 - A12))
    ln_gamma_w = x_etoh**2 * (A21 + 2 * x_w * (A12 - A21))
    return np.exp(ln_gamma_etoh), np.exp(ln_gamma_w)

# Calculate vapor mole fraction of ethanol
def calculate_y_etoh(x_etoh, T):
    P_total = P2_mmHg                 # Total pressure in mmHg
    gamma_etoh, gamma_w = activity_coeff(x_etoh)
    P_sat_etoh = antoine_ethanol(T)
    P_sat_w = antoine_water(T)
    P_total = gamma_etoh * x_etoh * P_sat_etoh + gamma_w * (1 - x_etoh) * P_sat_w
    return (gamma_etoh * x_etoh * P_sat_etoh) / P_total

# Define VLE function
def equilibrium(x):
    return calculate_y_etoh(x, T) 

# Plot VLE curve
# Generate data for plotting
x_etoh_range = np.linspace(0, 1, 200)   
y_etoh_range = np.array([calculate_y_etoh(x, T) for x in x_etoh_range]) 

# -------Plot VLE curve--------- 
plt.figure(figsize=(6, 6))
plt.plot(x_etoh_range, y_etoh_range, 'g-', linewidth=2, label='Equilibrium Curve',)
plt.plot([0, 1], [0, 1], 'k--', label="$x_D = x_W$")
plt.xlabel('Mole Fraction of Ethanol in Liquid ($x_{W}$)', fontsize=12)
plt.ylabel('Mole Fraction of Ethanol in Vapor ($x_{D}$)', fontsize=12)
plt.title(f'Vapour-Liquid Equilibrium Curve for Ethanol-Water \n at {P2_atm:0.3f} atm and {T:0.2f} °C', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
#plt.show()


# ========================================================================================
#               Power input and RefluX Ratio Calculation
# ========================================================================================
# constants
rho_w1 = 998                        # Density of water at room temperature (kg/m^3)
rho_e1 = 789                        # Density of ethanol at room temperature (kg/m^3)
Specific_hc_water = 4184            # Specific heat capacity of water (J/(kg·K))
Specific_hc_ethanol = 2440          # Specific heat capacity of eth (J/(kg·K))
C_e_i_0 = C_ethanonl_initial/100    # Initial ethanol concentration (v/v)

# Power input (W)
PowerI = ((T2_Celsius_w - FM_initial_temp)*(rho_e1*C_e_i_0*Specific_hc_ethanol +
        rho_w1*(1 - C_e_i_0)*Specific_hc_water)*(Initial_volume/1000)) / (Time_to_reach_bpt*60)


# ----- Reflux Ratio Calculation--------
m_w_r = 1e-6*V_w_ref*rho_w1                                     # reflux condenser water mass flow rate (kg/s)
H_water_ref = Specific_hc_water*m_w_r*(T_out_ref - T_in_ref)    # condenser heat duty (W)
Reflux_r = H_water_ref/(PowerI - H_water_ref)                   # reflux ration, RR


# ===================================================================================
#   Distillation Column Analysis (x_W --- x_D Relationship)
# ===================================================================================

# Distillation parameters
RR          = Reflux_r  # Reflux ratio
N_p         = 400       # Number of points
x_D_start   = 0.9       # Starting distillate composition
x_D_end     = 0.0001    # Ending distillate composition

# Initialize arrays
# Custom nonlinear spacing: more points near x_D_start. Small change in x_W couse rapid change in x_D
s = np.linspace(0, 1, N_p)
s_power = 3                     # >1 shifts density to start, <1 shifts density to end
x_D_values = x_D_start + (x_D_end - x_D_start) * s**s_power
x_W_values = np.zeros(N_p)      # To store bottoms composition

# Pre-compute equilibrium curve for interpolation
x_eq = np.linspace(0, 1, N_p)
y_eq = equilibrium(x_eq)

# Generate xW-xD relationship
for i, x_D in enumerate(x_D_values):
    m = RR/(RR+1)           # Operating line slope
    b = x_D/(RR+1)          # Operating line intercept
    current_x = x_D
    for _ in range(No_of_stages):
        current_y = m * current_x + b
        current_x = np.interp(current_y, y_eq, x_eq)
    x_W_values[i] = current_x

# Create the cubic spline interpolator
# First, sort the data to ensure x_W_values is strictly increasing
sorted_indices  = np.argsort(x_W_values)
x_W_sorted      = x_W_values[sorted_indices]
x_D_sorted      = x_D_values[sorted_indices]

# Remove duplicate x-values to ensure strictly increasing sequence
x_W_unique, unique_indices = np.unique(x_W_sorted, return_index=True)
x_D_unique = x_D_sorted[unique_indices]
xw_to_xd_spline = CubicSpline(x_W_unique, x_D_unique) # Cubic Spline Interpolator

# Convert bottoms composition (x_W) to distillate composition (x_D) using cubic spline
def xw_to_xd_poly(x_W):
    return xw_to_xd_spline(x_W)

# Calculate R-squared value
y_pred2 = xw_to_xd_poly(x_W_unique)         # Spline prediction on training data
r_squared = r2_score(x_D_unique, y_pred2)   # Compare against training labels

# Select point for visualization; n-th point in the array
idx     = 260   
x_D_vis = x_D_values[idx]
x_W_vis = x_W_values[idx]

# McCabe-Thiele diagram components
m_vis = RR/(RR+1)
b_vis = x_D_vis/(RR+1)
x_vis = np.linspace(0, 1, N_p)
y_eq_vis = equilibrium(x_vis)
x_op_vis = np.linspace(0, x_D_vis, 100)
y_op_vis = m_vis * x_op_vis + b_vis

# Construct stage points
x_points, y_points = [x_D_vis], [x_D_vis]
current_x = x_D_vis

for _ in range(No_of_stages):
    # Step up to operating line
    current_y = m_vis * current_x + b_vis
    x_points.append(current_x)
    y_points.append(current_y)
    
    # Step over to equilibrium curve
    current_x = np.interp(current_y, y_eq_vis, x_vis)
    x_points.append(current_x)
    y_points.append(current_y)

# Final point
x_points.append(current_x)
y_points.append(0)


# -----------Visualization-------------
# McCabe-Thiele Diagram
plt.figure(figsize=(6, 6))
plt.plot(x_vis, y_eq_vis, label='Equilibrium curve', linewidth=2, color="#017F01")
plt.plot(x_op_vis, y_op_vis, label=f'Operating line (RR={RR:.3f})', linewidth=1.5, color="#1f77b4")
plt.plot(x_vis, x_vis, '--', label=r'$x_D = x_W$', alpha=1, color='black')

plt.plot(x_points, y_points, 'o-', markersize=6, linewidth=1.5, alpha=1,color='red', 
         markerfacecolor='white', markeredgewidth=1.5, label=f'{No_of_stages} stages')
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


# -------------- xW-xD relationship---------
plt.figure(figsize=(6, 6))
plt.plot(x_W_values, x_D_values, 'b-', linewidth=2, label='Numerical Solution')
plt.plot(x_W_unique, xw_to_xd_poly(x_W_unique), 'r--', linewidth=1.5, label='Cubic Spline Fit')
plt.scatter(x_W_vis, x_D_vis, color='red', s=100, zorder=5)

plt.xlabel('Bottoms Composition ($x_W$)', fontsize=12)
plt.ylabel('Distillate Composition ($x_D$)', fontsize=12)
plt.title(f'Bottoms-Distillate Composition Relationship\n({No_of_stages} Stages, RR = {RR:.3f})', fontsize=12)
plt.grid(True)
plt.legend(loc='lower right', fontsize=12)

# Add R-squared value to plot
plt.text(0.95, 0.5, f'$R^2$ = {r_squared:.4f}', transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=1), fontsize=12, ha='right', va='center') 
plt.annotate(f'($x_W$={x_W_vis:.3f}, $x_D$={x_D_vis:.3f})', (x_W_vis, x_D_vis),  xytext=(0.8, 0.7), 
             textcoords='axes fraction',ha='right', va='center',  fontsize=12, bbox=dict(boxstyle='round,pad=0.5', 
             fc='white'), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.tight_layout()
#plt.show()


# Print cubic spline information
print("\n---Cubic Spline Information---")
print(f"Original data points: {len(x_W_values)}")
print(f"x_W range           = [{x_W_unique.min():.6f}, {x_W_unique.max():.6f}]")
print(f"x_D range           = [{x_D_unique.min():.6f}, {x_D_unique.max():.6f}]")
print(f"Goodness of fit (R^2):{r_squared:.4f}")


# ===================================================================================
#         Distillation Simulation-Calculation
# ===================================================================================
# parameters
Sim_time    = Total_testing_time - Time_to_reach_bpt    # Simulation time, after reaching bpt (mins)
total_time  = 60*Sim_time                               # Total simulation time (s). 
time_step_size = 1                                      # Time step (s)
num_steps   = int(total_time / time_step_size)          # Number of time steps 

# Physical Constants
L_v_e   = 838000              # Latent heat of vaporization for ethanol (J/kg)
L_v_w   = 2260000             # Latent heat of vaporization for water (J/kg)
Molw_e  = 46.07e-3            # Molecular weight of ethanol (kg/mol)
Molw_w  = 18.015e-3           # Molecular weight of water (kg/mol)

# Initialize arrays
time = np.zeros(num_steps)            # Time array (s)
x_D = np.zeros(num_steps)             # Distillate composition
x_w = np.zeros(num_steps)             # Bottoms composition

# Container (remaining) quantities
n_e_remain = np.zeros(num_steps)      # Moles of ethanol remaining
n_w_remain = np.zeros(num_steps)      # Moles of water remaining
n_t_remain = np.zeros(num_steps)      # Total moles remaining
Mf_e_remain = np.zeros(num_steps)     # Mole fraction ethanol in bottom-liquid
Mf_w_remain = np.zeros(num_steps)     # Mole fraction water in liquid

# Collected quantities
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

# Initial values assignment
V_e_initial = Initial_volume * C_e_i_0          # Initial ethanol volume (L)
V_w_initial = Initial_volume - V_e_initial      # Initial water volume (L)
M_e_initial = V_e_initial * rho_e1/1000         # Initial ethanol mass (kg)
M_w_initial = V_w_initial * rho_w1/1000         # Initial water mass (kg)
 
n_e_remain[0] = M_e_initial / Molw_e            # Initial moles of ethanol
n_w_remain[0] = M_w_initial / Molw_w            # Initial moles of water
n_t_remain[0] = n_e_remain[0] + n_w_remain[0]
Mf_e_remain[0] = n_e_remain[0] / n_t_remain[0]  # Initial mole fraction ethanol


# ----------- Density of mixture ---------
# Data from the table at 20°C
ethanol_percent = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])            # ethanol weight percentage
density_20C     = np.array([998, 982, 969, 954, 935, 914, 891, 868, 843, 818, 789]) # mixture density

# Create cubic spline interpolation
density_spline = CubicSpline(ethanol_percent, density_20C, bc_type='natural')

def density_mix(ethanol_wt_percent):
    # Clip values to valid range [0, 100]
    if isinstance(ethanol_wt_percent, (list, np.ndarray)):
        ethanol_wt_percent = np.clip(ethanol_wt_percent, 0, 100)
    else:
        ethanol_wt_percent = np.clip(ethanol_wt_percent, 0, 100)
    
    return density_spline(ethanol_wt_percent)


# ---------------------------------------------------------
# -----------------Simulation loop-------------------------
for step in range(1, num_steps):
    # Update time
    time[step] = step * time_step_size
   
    # Current bottoms composition
    x_w[step] = Mf_e_remain[step-1]
    
    # Calculate distillate composition using the polynomial from previous section
    x_D[step] = xw_to_xd_poly(x_w[step])            # Using the pre-defined polynomial

    # Calculate moles of distillate collected
    Energy = PowerI * time_step_size                # Energy input (J)
    latent_heat = x_D[step] * Molw_e * L_v_e + (1 - x_D[step]) * Molw_w * L_v_w
    
    #c_f = 0 # No correction factor
    c_f = 1-593/719.69   # Diagnostic Validation: correction factor for mass loss from the 1st tray as liquid.

    no_dist =(1-c_f)* Energy / ((1 + RR) * latent_heat)     # Moles of distillate
    
    # Check if we have enough liquid left
    if no_dist > n_t_remain[step-1]:
        no_dist = n_t_remain[step-1]
        num_steps = step + 1
        time = time[:num_steps]
        break
    
    # Calculate component amounts
    current_step_ethanol= no_dist * x_D[step]       # Moles ethanol
    current_step_water  = no_dist * (1 - x_D[step]) # Moles water

    # Safety checks for component amounts
    # Moles of ethanol collected in this step
    if current_step_ethanol > n_e_remain[step-1]:
        current_step_ethanol = 0    

    # Moles of water collected in this step
    if current_step_water > n_w_remain[step-1]:
        current_step_water = 0

    # Update collected quantities
    n_e_collected[step] = n_e_collected[step-1] + current_step_ethanol
    n_w_collected[step] = n_w_collected[step-1] + current_step_water
    n_t_collected[step] = n_e_collected[step] + n_w_collected[step]
    
    # Update remaining quantities
    n_e_remain[step] = n_e_remain[step-1] - current_step_ethanol
    n_w_remain[step] = n_w_remain[step-1] - current_step_water
    n_t_remain[step] = n_e_remain[step] + n_w_remain[step]

    # Update compositions
    Mf_e_remain[step] = n_e_remain[step] / n_t_remain[step] if n_t_remain[step] > 0 else 0

    # Update collected mass & volume
    M_e_collected[step] = n_e_collected[step] * Molw_e
    M_w_collected[step] = n_w_collected[step] * Molw_w
    M_t_collected[step] = M_e_collected[step] + M_w_collected[step]
    E_wt_Percent[step]  = 100 * M_e_collected[step] / M_t_collected[step]

    V_e_collected[step] = M_e_collected[step] / rho_e1 * 1000           # Ethanol volume collected (L)
    V_w_collected[step] = M_w_collected[step] / rho_w1 * 1000           # Water volume collected (L)
    # Total mixed volume collected (L), including negative excess volume of mixing
    V_t_collected[step] = M_t_collected[step]/density_mix(E_wt_Percent[step])*1000  
    C_e_collected[step] = 100 * V_e_collected[step] /(V_e_collected[step]+V_w_collected[step]) # but, v/v --> conc is calculated using sum

    instant_vv_p[step] = 100 * (V_e_collected[step]-V_e_collected[step-1]) / ((V_e_collected[step]-
                                V_e_collected[step-1])+(V_w_collected[step]-V_w_collected[step-1])) 


# ===================================================================================
#         Summary and Results
# ===================================================================================
collected_ethanol = V_e_collected[num_steps-1]
recovery = (collected_ethanol / V_e_initial)
  
print("\nSimulation successfull.\n---- Distillation Results and Summary ----")
if num_steps > 2: 
    print(f"Operation Time          : {time[num_steps-1]/60:.1f} mins")
    print(f"Power Input             : {PowerI:.2f} W")
    print(f'Ref. Condenser Heat Duty: {H_water_ref:.2f} W')
    print(f"Reflux Ratio (R)        : {Reflux_r:.3f}")
    print(f"Initial Eth m.f. liquid : {Mf_e_remain[0]:.5f}")
    print(f"Dist. Conc. at start    : {instant_vv_p[1]:.2f}%")
    print(f"Distillate Conc. at end : {instant_vv_p[num_steps-1]:.2f}%")
    print(f"Final Distillate Conc.  : {C_e_collected[num_steps-1]:.2f}%")
    print(f"Collected Ethanol(pure) : {collected_ethanol*1000:.2f} mL")
    print(f"Collected Water(pure)   : {V_w_collected[num_steps-1]*1000:.2f} mL")
    print(f"Total Dist. Collected   : {V_t_collected[num_steps-1]*1000:.2f} mL")

else:
    print("Simulation unsuccessfull.")


# =================================================
#   ADDING EXPERIMENTAL MEASUREMENTS TO PLOTS
# =================================================
# INSTANTANEOUS RATE POINTS
d1 = 2+46/60
d2 = 2+51/60
d3 = 3+53/60
d = np.array([d1, d2, d3])

t0 = Time_to_reach_bpt # 
t1 = 16+46/60   # 16:46, t0+d1
t2 = 22         # 20:00
t3 = 30         # 30:00
t4 = 38         # 38:00

# --------------------------------------
# Instantaneous concentration samples 
instant_start_times = np.array([t0, t2, t3])
instant_end_times   = instant_start_times + d
exp_sample_conc_rt  = np.array([87, 86, 84])     # v/v%

# -----------------------------------------------------
# Instantaneous distillate collection rate (midpoints)
exp_rate_times = instant_start_times + d/2
exp_rate       = np.array([90/d1, 90/d2, 85/d3])   # mL/min

# -------------------------------------------------------
# Cumulative concentration (end of collection intervals)
exp_sample_times_cumu = np.array([t1, t2, t3, t4])
exp_sample_conc_cumu  = np.array([87, 86, 85, 84])        

# ---------------------------------
# Cumulative volume collected (mL)
exp_interval_times = exp_sample_times_cumu
exp_interval_vol   = np.array([90, 236, 472, 593])


# ===================================================================================
#         Final Visualization
# ===================================================================================

# -----------------------------------------------------
# Instantaneous Distillate Concentration (v/v%) vs time
plt.figure(figsize=(6, 6))
plt.gca().set_axisbelow(True)
plt.grid(True, alpha=0.5)

# Experimental horizontal bars + points 
for start, end, conc in zip(instant_start_times, instant_end_times, exp_sample_conc_rt):
    plt.hlines(conc, start, end, colors='black', linewidth=2, zorder=3)
    plt.scatter([(start+end)/2], [conc], color='black', s=70, zorder=4)
# Theoretical line 
plt.plot(Time_to_reach_bpt + time[1:num_steps]/60, instant_vv_p[1:num_steps], linewidth=2, color='red', zorder=2)
# Legend handles
exp_handle  = Line2D([0],[0], color='black', linewidth=2, marker='o', markersize=8)
theo_handle = Line2D([0],[0], color='red', linewidth=2)

plt.legend(handles=[exp_handle, theo_handle], labels=[ 'Experimental', 'Theoretical'])
plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('Ethanol Concentration (v/v%)', fontsize=12)
plt.title('Instantaneous Distillate Composition Tracking', fontsize=12)
plt.ylim(50, 90)
plt.tight_layout()


# ---------------------------------------------------------
# Instantaneous Distillate Collection Rate (mL/min) vs time
plt.figure(figsize=(6, 6))
# Make grid behind data
plt.gca().set_axisbelow(True)
plt.grid(True, alpha=0.5)

# Experimental horizontal bars + points
for start, end, r in zip(instant_start_times, instant_end_times, exp_rate):
    plt.hlines(r, start, end, colors='black', linewidth=2, zorder=3)
    plt.scatter([(start+end)/2], [r], color='black', s=70, zorder=4)

# Theoretical lines
time_minutes = Time_to_reach_bpt + time[1:num_steps]/60
total_volume_ml   = 1000 * V_t_collected[1:num_steps]
ethanol_volume_ml = 1000 * V_e_collected[1:num_steps]
water_volume_ml   = 1000 * V_w_collected[1:num_steps]

total_rate   = np.gradient(total_volume_ml, time_minutes)
ethanol_rate = np.gradient(ethanol_volume_ml, time_minutes)
water_rate   = np.gradient(water_volume_ml, time_minutes)

plt.plot(time_minutes, total_rate,   linewidth=2, color='red',   zorder=2)
plt.plot(time_minutes, ethanol_rate, linewidth=2, linestyle='--', color='green', zorder=1)
plt.plot(time_minutes, water_rate,   linewidth=2, linestyle=':', color='blue',  zorder=1)

# --- Legend handles ---
exp_handle = Line2D([0],[0], color='black', linewidth=2, marker='o', markersize=8)
total_handle = Line2D([0],[0], color='red', linewidth=2)
eth_handle   = Line2D([0],[0], color='green', linewidth=2, linestyle='--')
wat_handle   = Line2D([0],[0], color='blue', linewidth=2, linestyle=':')

plt.legend(handles=[exp_handle, total_handle, eth_handle, wat_handle], labels=[
    'Experimental', 'Theoretical', 'Pure Ethanol', 'Pure Water'])
plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('Collection Rate (mL/min)', fontsize=12)
plt.title('Distillate Collection Rate Tracking', fontsize=12)
plt.tight_layout()


# -------------------------------------
# Cumulative Volume Tracking (vs time)
plt.figure(figsize=(6, 6))
plt.scatter(exp_interval_times, exp_interval_vol, color='black', s=80, marker='o', zorder=4, label='Experimental') 
plt.plot(Time_to_reach_bpt+time[1:num_steps]/60, 1000*V_t_collected[1:num_steps], label='Theoritical', 
         linewidth=2, zorder=3, color='red', linestyle='-')
plt.plot(Time_to_reach_bpt+time[1:num_steps]/60, 1000*V_e_collected[1:num_steps], label='Pure Ethanol', 
         linewidth=2,zorder=2, color='green', linestyle='--')
plt.plot(Time_to_reach_bpt+time[1:num_steps]/60, 1000*V_w_collected[1:num_steps], label='Pure Water', 
         zorder=1, linewidth=2, color='blue', linestyle=':')

plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('Volume (mL)', fontsize=12)
plt.title('Cumulative Distillate Volume Tracking', fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()


# --------------------------------------------
# Cumulative Concentration Tracking (vs time)
plt.figure(figsize=(6, 6))
# Theoretical cumulative concentration (v/v%)
plt.plot(Time_to_reach_bpt + time[1:num_steps]/60, C_e_collected[1:num_steps], label='Theoretical',zorder=1, linewidth=2, color='red')

# Experimental cumulative conc
exp_cum_times = exp_interval_times
exp_cum_conc  = exp_sample_conc_cumu 
# plot experimental points
plt.scatter(exp_cum_times, exp_cum_conc,color='black', s=80, marker='o',zorder=2, label='Experimental')

plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('Ethanol Concentration (v/v%)', fontsize=12)
plt.title('Cumulative Distillate Composition Tracking', fontsize=12)
plt.legend(handles=[
    Line2D([], [], color='black', marker='o', linestyle='', markersize=8, label='Experimental'),
    Line2D([], [], color='red', linewidth=2, label='Theoretical')])
plt.grid(True)
plt.ylim(50, 90)
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------------
#  ---------------------------- END -----------------------------------------
#------------------------------------------------------------------------------
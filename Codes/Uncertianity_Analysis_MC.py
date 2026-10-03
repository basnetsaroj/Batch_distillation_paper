"""
Monte Carlo uncertainty propagation for effective heating power (P), reflux-condenser duty (Q_cond), reflux ratio (R), 
final distillate concentration, and final distillate volume"""


''' #Use this for google colab
from google.colab import files
uploaded = files.upload()'''

import math
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
import Dist_thermo_properties as tp
rng = np.random.default_rng(1)

# ===================================================================
# Fixed constants From Run 3
# ===================================================================
P1       = 101325
g        = 9.81
M_air    = 28.97e-3
Rgas     = 8.314
Molw_e   = 46.07e-3
Molw_w   = 18.015e-3
h        = 1300
T_room   = 18
N_stages = 4

alpha, A12, B12, A21, B21 = 0.30, -0.801, 246.2, 3.458, -586.1

def antoine_ethanol(T_C): return 10 ** (7.58670 - 1281.590/(T_C+193.768))
def antoine_water(T_C):   return 10 ** (8.07131 - 1730.630/(T_C+233.426))

def tau(T_C):
    T_K = T_C + 273.15
    return A12 + B12/T_K, A21 + B21/T_K

def activity_coeff_nrtl(x1, tau12, tau21):
    x2 = 1.0 - x1
    G12, G21 = np.exp(-alpha*tau12), np.exp(-alpha*tau21)
    ln_g1 = x2**2*(tau21*(G21/(x1+x2*G21))**2 + tau12*G12/(x2+x1*G12)**2)
    ln_g2 = x1**2*(tau12*(G12/(x2+x1*G12))**2 + tau21*G21/(x1+x2*G21)**2)
    return np.exp(ln_g1), np.exp(ln_g2)

def calculate_y_etoh(x1, T_C):
    x2 = 1.0 - x1
    t12, t21 = tau(T_C)
    g1, g2 = activity_coeff_nrtl(x1, t12, t21)
    Ps1, Ps2 = antoine_ethanol(T_C), antoine_water(T_C)
    Pt = g1*x1*Ps1 + g2*x2*Ps2
    return (g1*x1*Ps1)/Pt

def boiling_temperature(x1, P_mmHg, Tg_e, Tg_w):
    x2 = 1 - x1
    def eqn(T):
        t12, t21 = tau(T)
        g1, g2 = activity_coeff_nrtl(x1, t12, t21)
        return g1*x1*antoine_ethanol(T) + g2*x2*antoine_water(T) - P_mmHg
    T_guess = x1*Tg_e + x2*Tg_w
    return fsolve(eqn, T_guess)[0]

# ===================================================================
# Stage 1: heating power, condenser duty, reflux ratio 
# ===================================================================
def calc_P_R(Initial_vol, C_e, T0, t_bpt_s, V_w, t_fill, T_in, T_out):
    rho_w0 = tp.get_pure_water_den(T0)
    rho_e0 = tp.get_pure_ethanol_den(T0)
    V_e_initial = Initial_vol * C_e/100
    M_e_initial = V_e_initial * rho_e0

    def mixed_feed_volume(V_w_mix):
        M_wi = V_w_mix * rho_w0
        M_total = M_e_initial + M_wi
        wt = 100*M_e_initial/M_total
        return M_total/tp.get_den(wt, T0)

    lo, hi, tol = 0.0, Initial_vol, 1e-4
    while hi - lo > tol:
        mid = (lo+hi)/2
        if mixed_feed_volume(mid) < Initial_vol: lo = mid
        else: hi = mid
    V_w_initial = (lo+hi)/2
    M_w_initial = V_w_initial*rho_w0
    M_total = M_e_initial + M_w_initial
    n_e_initial = M_e_initial/Molw_e
    n_w_initial = M_w_initial/Molw_w
    x_e0 = n_e_initial/(n_e_initial+n_w_initial)

    T_k_r = T_room + 273.15
    P2 = P1*math.exp(-(g*M_air*h)/(Rgas*T_k_r))
    P2_mmHg = P2/P1*760

    dH_e = tp.Hvap_ethanol(T_room)*Molw_e
    dH_w = tp.Hvap_water(T_room)*Molw_w
    lhs = math.log(P2/P1)
    T2_e = 1/((1/351.45) - lhs*Rgas/dH_e) - 273.15
    T2_w = 1/((1/373.15) - lhs*Rgas/dH_w) - 273.15

    T_bpt0 = boiling_temperature(x_e0, P2_mmHg, T2_e, T2_w)

    m_boiler, shc_boiler = 6.056, 511
    Q_pot = m_boiler*shc_boiler*(T_bpt0-T0)
    T_range = np.append(np.arange(T0, T_bpt0, 1.0), T_bpt0)
    Q_s = sum(M_total*tp.Cp_mixture(x_e0, T_range[i])*(T_range[i+1]-T_range[i])
              for i in range(len(T_range)-1))
    PowerI = (Q_s+Q_pot)/t_bpt_s

    flow_mLs = V_w/t_fill
    water_dens = tp.get_pure_water_den((T_out+T_in)/2)*1e3
    m_w_r = 1e-6*flow_mLs*water_dens
    Shc_water = tp.Cp_water((T_out+T_in)/2)
    H_water_ref = Shc_water*m_w_r*(T_out-T_in)
    RR = H_water_ref/(PowerI-H_water_ref)
    return PowerI, H_water_ref, RR, n_e_initial, n_w_initial, rho_e0, rho_w0, P2_mmHg, T2_e, T2_w

# ===================================================================
# Stage 2: batch distillation simulation -> final conc. and volume
# ===================================================================
def run_simulation(PowerI, RR, n_e_initial, n_w_initial, rho_e0, rho_w0, P2_mmHg, T2_e, T2_w, Sim_time):
    # Pre-solve T(x) and y(x) once on a grid, reused by spline instead of
    # calling fsolve at every simulation time step (same physics, far faster).
    x_grid = np.linspace(1e-6, 1-1e-6, 150)
    T_grid = np.array([boiling_temperature(x, P2_mmHg, T2_e, T2_w) for x in x_grid])
    y_grid = np.array([calculate_y_etoh(x, T) for x, T in zip(x_grid, T_grid)])
    T_of_x = CubicSpline(x_grid, T_grid)
    y_of_x = CubicSpline(x_grid, y_grid)

    def build_xw_to_xd_relation():
        N_p, x_D_start, x_D_end = 400, 0.8943, 0.0001
        s = np.linspace(0, 1, N_p)
        x_D_values = x_D_start + (x_D_end-x_D_start)*s**3
        x_W_values = np.zeros(N_p)
        x_eq = np.linspace(0, 1, N_p)
        y_eq = y_of_x(x_eq)
        m = RR/(RR+1)
        for i, x_D in enumerate(x_D_values):
            b = x_D/(RR+1)
            cx = x_D
            for _ in range(N_stages):
                cy = m*cx+b
                cx = np.interp(cy, y_eq, x_eq)
            x_W_values[i] = cx
        order = np.argsort(x_W_values)
        xW, xD = x_W_values[order], x_D_values[order]
        xW_u, idx = np.unique(xW, return_index=True)
        return CubicSpline(xW_u, xD[idx])

    xw_to_xd_spline = build_xw_to_xd_relation()
    dt = 1.0        # sec
    num_steps = int(np.floor(Sim_time / dt))

    n_e_remain = n_e_initial
    n_w_remain = n_w_initial
    n_e_coll = 0.0
    n_w_coll = 0.0
    V_e_prev = 0.0
    V_t_prev = 0.0
    x_w = n_e_initial/(n_e_initial+n_w_initial)

    for step in range(num_steps):
        T_b   = float(T_of_x(x_w))
        x_D   = float(xw_to_xd_spline(x_w))
        y_d1  = float(y_of_x(x_w))
        Energy = PowerI * dt
        latent_heat = tp.Mixture_Latent_Heat(x_w, y_d1, T_b)
        n_dist = Energy/((1+RR)*latent_heat)

        step_e = min(n_dist*x_D, n_e_remain)
        step_w = min(n_dist*(1-x_D), n_w_remain)

        n_e_coll += step_e; n_w_coll += step_w
        n_e_remain -= step_e; n_w_remain -= step_w
        n_t_remain = n_e_remain + n_w_remain
        if n_t_remain <= 1e-6:
            break
        x_w = n_e_remain/n_t_remain

        M_e_coll = n_e_coll*Molw_e
        M_w_coll = n_w_coll*Molw_w
        M_t_coll = M_e_coll + M_w_coll
        E_wt_pct = 100*M_e_coll/M_t_coll
        V_e_coll = M_e_coll/rho_e0
        V_w_coll = M_w_coll/rho_w0
        dens_mix = tp.get_den(E_wt_pct, 17)          # distillate density at feed ref. temp (FM_i_t)
        V_t_coll = M_t_coll/dens_mix
        V_e_prev, V_t_prev = V_e_coll, V_t_coll

    C_e_final = 100*V_e_prev/V_t_prev    # final distillate concentration, v/v %
    V_t_final = V_t_prev*1000            # L -> mL
    return C_e_final, V_t_final

# ===================================================================
# Monte Carlo driver
# ===================================================================
# Nominal values and standard uncertainties, Run 3.
# Initial_vol, C_e, T0: single readings -> Type B (resolution) only.
# T_in, T_out, t_fill : 3 logged readings -> Type A (repeatability of the
#   mean, s/sqrt(3)) combined in quadrature with Type B (resolution).
nom = dict(Initial_vol=3.5, C_e=30, T0=17, t_bpt_s=940, t_total_s =53*60, V_w=900, t_fill=92.67, T_in=24.73, T_out=37.1)

# Instrument resolution
resolution = dict(
    Initial_vol = 0.005,   # L (5 mL)
    C_e         = 1.0,     # % v/v
    T0          = 0.1,     # °C
    t_bpt_s     = 1.0,     # sec
    t_total_s   = 1.0,     # s
    V_w         = 5.0,     # mL
    t_fill      = 1.0,     # s
    T_in        = 0.1,     # °C
    T_out       = 0.1      # °C
)
# Standard uncertainty for a rectangular distribution
u = {key: value / np.sqrt(12) for key, value in resolution.items()}

if __name__ == "__main__":
    print("Standard uncertainties used:")
    for k in nom:
        print(f"  {k:12s} nominal={nom[k]:8.3f}   u={u[k]:.4f}")

    N = 1000
    out, n_discarded = [], 0
    for k in range(N):
        print(f"\rMC run {k+1}/{N}", end="")
        s = {key: rng.uniform(nom[key] - resolution[key]/2, nom[key] + resolution[key]/2) for key in nom}
        t_sim_s = s.pop("t_total_s") - s["t_bpt_s"]
        P_s, Q_s, R_s, ne_s, nw_s, rho_e0_s, rho_w0_s, Pmm_s, Te_s, Tw_s = calc_P_R(**s)
        if P_s <= Q_s:
            n_discarded += 1
            continue
        C_f, V_f = run_simulation(P_s, R_s, ne_s, nw_s, rho_e0_s, rho_w0_s, Pmm_s, Te_s, Tw_s, t_sim_s)
        out.append((P_s, Q_s, R_s, C_f, V_f))
    print()

    out = np.array(out)
    print(f"Discarded draws (P <= Q_cond): {n_discarded}/{N}")
    for name, col in zip(
        ["P (W)", "Q_cond (W)", "R", "Final conc (v/v%)", "Total dist (mL)"],
        out.T,):
        print(f"{name:20s}: {col.mean():9.3f} +/- {col.std(ddof=1):7.3f}  "
              f"({100*col.std(ddof=1)/col.mean():5.2f} %)")




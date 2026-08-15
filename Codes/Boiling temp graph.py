import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

# -------------------------------------------------------
# Aspen Plus vs Present Model: Transient Temperature
# -------------------------------------------------------

# Run 1
distillation_time_1 = np.array([12, 18, 24, 30, 36, 40])
temp_aspen_1        = np.array([84.37, 86.31, 88.54, 90.85, 92.86, 93.84])
temp_model_1        = np.array([84.49, 86.43, 88.82, 91.35, 93.48, 94.45])

# Run 2
distillation_time_2 = np.array([14.08, 20.08, 26.08, 32.08, 38.08])
temp_aspen_2        = np.array([82.39, 83.49, 84.92, 86.67, 88.71])
temp_model_2        = np.array([82.49, 83.60, 85.10, 87.10, 89.45])

# Run 3
distillation_time_3 = np.array([15.67, 21.67, 27.67, 33.67, 39.67, 45.67, 51.67, 53.67])
temp_aspen_3        = np.array([81.71, 82.52, 83.54, 84.79, 86.30, 88.06, 89.98, 90.41])
temp_model_3        = np.array([81.84, 82.64, 83.67, 85.01, 86.73, 88.76, 90.90, 91.36])


# -------------------------------------------------------
# Function for plotting
# -------------------------------------------------------
def plot_temperature_comparison(distillation_time, temp_aspen, temp_model, run_number):
    plt.figure(figsize=(6, 6))
    plt.gca().set_axisbelow(True)
    # Smooth time points for visualization
    time_smooth = np.linspace(distillation_time.min(), distillation_time.max(),300)
    # Shape-preserving interpolation
    aspen_smooth = PchipInterpolator(distillation_time, temp_aspen)(time_smooth)
    model_smooth = PchipInterpolator( distillation_time, temp_model)(time_smooth)
    # Aspen Plus 
    plt.plot(time_smooth, aspen_smooth, color='blue', linewidth=2, zorder=2)
    # Present Model  
    plt.plot(time_smooth, model_smooth, color='red', linewidth=2, zorder=3)
    # Original Aspen Plus data points
    plt.scatter(distillation_time, temp_aspen, color='blue', s=45, marker='o', zorder=4)
    # Original Model data points
    plt.scatter(distillation_time, temp_model, color='red', s=45, marker='s', zorder=5)

    # Legend with line + marker
    legend_handles = [
        Line2D([0], [0],color='blue',linewidth=2,marker='o',markersize=6,label='Aspen Plus'),
        Line2D([0], [0],color='red',linewidth=2,marker='s',markersize=5,label='Present Model')]
    plt.legend(handles=legend_handles)
    plt.xlabel('Distillation Time (min)', fontsize=12)
    plt.ylabel('Boiling Temperature (°C)', fontsize=12)
    plt.grid(True)
    #plt.ylim(80, 96)
    #plt.title(f'Run {run_number}', fontsize=12)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# Generate the three figures
# -------------------------------------------------------
plot_temperature_comparison(distillation_time_1, temp_aspen_1, temp_model_1, 1)
plot_temperature_comparison(distillation_time_2, temp_aspen_2, temp_model_2, 2)
plot_temperature_comparison(distillation_time_3, temp_aspen_3, temp_model_3, 3)


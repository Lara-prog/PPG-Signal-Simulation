# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 11:57:51 2025

@author: laram
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

# Global storage for past pressure values (for DDE simulation)
t_history = []
P_history = []

# Windkessel Model with Delayed Wave Reflection
def windkessel_dde(t, P, R, C, Zc, I0, Ts, HR, alpha=0.07, L=0.5, c=3.5):
    """
    Windkessel Model with Delayed Wave Reflection.
    """
    T = 60 / HR  # Cardiac cycle duration
    tau = L / c  # Delay time for wave reflections

    # More realistic blood ejection profile using a sinusoidal function
    if (t % T) < Ts:
        Q = I0 * np.sin(np.pi * (t % T) / Ts) ** 2  # Smoothly varying inflow
    else:
        Q = 0  # Diastolic phase

    # Delayed pressure reflection using cubic spline interpolation
    if len(t_history) == 0 or (t - tau) < min(t_history):
        P_delayed = 0  # No reflection in the first cycle
    else:
        # Ensure strictly increasing time history before interpolation
        sorted_data = sorted(set(zip(t_history, P_history)))
        t_sorted, P_sorted = zip(*sorted_data)

        spline = CubicSpline(t_sorted, P_sorted, extrapolate=True)
        P_delayed = spline(t - tau)

    # Windkessel ODE with damping for wave reflection
    dPdt = (Q - P / Zc) / C - (P / (R * C)) + alpha * (P_delayed / (R * C))
    return dPdt

# Solve the Delayed Differential Equation (DDE)
def solve_windkessel_dde(duration=10, P0=90, R=1.3, C=1.2, Zc=0.1, 
                          I0=1.2, Ts=0.3, HR=60, alpha=0.07, L=0.5, c=3.5):
    """
    Solve the Windkessel Model with Delayed Reflections for multiple cardiac cycles.
    """
    global t_history, P_history
    t_history, P_history = [], []
    T = 60 / HR  # Duration of one cardiac cycle
    num_cycles = int(duration / T)  # Number of heartbeats in the duration
    t_full = []
    PPG_full = []

    for cycle in range(num_cycles):
        t_start = cycle * T  # Start time of this cycle
        t_eval = np.linspace(t_start, t_start + T, int(T * 2000))  # Increase resolution

        # Solve ODE for one cardiac cycle with delay
        sol = solve_ivp(windkessel_dde, [t_start, t_start + T], [P0],
                        args=(R, C, Zc, I0, Ts, HR, alpha, L, c), t_eval=t_eval,
                        method='LSODA')  # LSODA solver for stiff systems

        # Ensure strictly increasing history
        if len(t_history) == 0 or sol.t[0] > t_history[-1]:  
            t_history.extend(sol.t)
            P_history.extend(sol.y[0])

        # Append results to full timeline
        t_full.extend(sol.t)
        PPG_full.extend(C * sol.y[0])  # Assume PPG is proportional to arterial pressure

    return np.array(t_full), np.array(PPG_full)

# **High-pass filter for baseline correction**
def high_pass_filter(signal, cutoff=1, fs=2000, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)


# Solve for 10 seconds using the corrected model
t_dde, PPG_dde = solve_windkessel_dde(duration=10)


# Apply high-pass filtering for baseline correction
PPG_dde_filtered = high_pass_filter(PPG_dde)

t_dde = t_dde[t_dde > 1]  
PPG_dde_filtered = PPG_dde_filtered[len(PPG_dde_filtered) - len(t_dde):]



# Plot PPG waveform from the corrected model
plt.figure(figsize=(12, 4))
plt.plot(t_dde, PPG_dde_filtered, label="PPG Signal (Corrected Model)", color="black")
plt.xlabel("Time (s)")
plt.ylabel("PPG Signal")
plt.title("PPG Signal with Corrected Delayed Wave Reflection")
plt.legend()
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

# Global storage for past pressure values (for DDE simulation)
t_history = []
P_history = []

# Heart Rate Variability Model
def heart_rate_variability(t, HR_base=60, HRV_amplitude=5, HRV_frequency=0.1):
    """Generates a dynamically varying heart rate."""
    return HR_base + HRV_amplitude * np.sin(2 * np.pi * HRV_frequency * t)

# Pulse Transit Time Variability
def pulse_transit_time_variability(C, L_base=0.5, c_base=3.5):
    """Adjust pulse transit time based on vascular compliance changes."""
    c_dynamic = c_base * np.sqrt(C / 1.2)  # Compliance-dependent velocity
    return L_base / c_dynamic

# Physiological Aortic Inflow Waveform
def inflow_waveform(t, I0=1.2, Ts=0.3, HR=60, k=10):
    """Pulsatile inflow using a sigmoid ramp-up for gradual systolic rise."""
    T = 60 / HR
    t_phase = t % T  # Time within the cardiac cycle
    if t_phase < Ts:
        return I0 * (1 / (1 + np.exp(-k * (t_phase - Ts / 4))))  # Smooth rise
    return 0  # Diastolic phase


# Windkessel Model with Variable Parameters
def windkessel_dde(t, P, R, C, Zc, I0, Ts, HR_base, alpha=0.07, L=0.5, c=3.5):
    """Windkessel Model with HR variability and dynamic pulse transit time."""
    HR = heart_rate_variability(t, HR_base)  # Dynamic HR
    T = 60 / HR  # Cardiac cycle duration
    tau = pulse_transit_time_variability(C, L, c)  # Dynamic delay

    # Inflow function with HR adjustments
    Q = inflow_waveform(t, I0, Ts, HR)

    # Delayed pressure reflection using cubic spline interpolation
    if len(t_history) == 0 or (t - tau) < min(t_history):
        P_delayed = 0  # No reflection in the first cycle
    else:
        sorted_data = sorted(set(zip(t_history, P_history)))
        t_sorted, P_sorted = zip(*sorted_data)
        spline = CubicSpline(t_sorted, P_sorted, extrapolate=True)
        P_delayed = spline(t - tau)

    # Windkessel ODE with damping and dynamic HR
    dPdt = (Q - P / Zc) / C - (P / (R * C)) + alpha * (P_delayed / (R * C))
    return dPdt

# Solve the Windkessel Model with Dynamic Components
def solve_windkessel_dde(duration=10, P0=90, R=1.3, C=1.2, Zc=0.1,
                          I0=1.2, Ts=0.3, HR_base=60, alpha=0.07, L=0.5, c=3.5):
    """Solves the ODE while incorporating HRV and dynamic transit time."""
    global t_history, P_history
    t_history, P_history = [], []
    T_avg = 60 / HR_base  
    num_cycles = int(duration / T_avg)  # Number of heartbeats in duration
    t_full, PPG_full = [], []

    for cycle in range(num_cycles):
        t_start = cycle * T_avg
        t_eval = np.linspace(t_start, t_start + T_avg, int(T_avg * 2000))  

        sol = solve_ivp(windkessel_dde, [t_start, t_start + T_avg], [P0],
                        args=(R, C, Zc, I0, Ts, HR_base, alpha, L, c),
                        t_eval=t_eval, method='LSODA')

        if len(t_history) == 0 or sol.t[0] > t_history[-1]:  
            t_history.extend(sol.t)
            P_history.extend(sol.y[0])

        t_full.extend(sol.t)
        PPG_full.extend(C * sol.y[0])  

    return np.array(t_full), np.array(PPG_full)

# High-pass filter for baseline correction
def high_pass_filter(signal, cutoff=0.8, fs=2000, order=2):
    """Applies a high-pass filter to remove low-frequency drifts."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)

# Solve for 10 seconds with dynamic HR and transit time
t_dde, PPG_dde = solve_windkessel_dde(duration=10)

# Apply high-pass filtering for baseline correction
PPG_dde_filtered = high_pass_filter(PPG_dde)

# Remove first second for stability
t_dde = t_dde[t_dde > 1]  
PPG_dde_filtered = PPG_dde_filtered[len(PPG_dde_filtered) - len(t_dde):]

# Plot PPG waveform with improved model
plt.figure(figsize=(12, 4))
plt.plot(t_dde, PPG_dde_filtered, label="PPG Signal (Improved Model)", color="black")
plt.xlabel("Time (s)")
plt.ylabel("PPG Signal")
plt.title("PPG Signal with Dynamic HR & Wave Reflection")
plt.legend()
plt.grid()
plt.show()



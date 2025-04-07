# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 17:15:36 2025

@author: laram
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Differential Equation-Based PPG Model
def windkessel_ode(t, P, R, C, I0, Ts, HR):
    """ODE for arterial pressure dynamics based on the Windkessel model."""
    T = 60 / HR  # Duration of one cardiac cycle    
    # Define cardiac input flow (Q) based on systolic phase
    if (t % T) < Ts:
        Q = I0 * np.sin(np.pi * (t % T) / Ts) ** 2  # Systolic phase
    else:
        Q = 0  # Diastolic phase
    # Windkessel model ODE: dP/dt = (Q / C) - (P / (R * C))
    dPdt = (Q / C) - (P / (R * C))
    return dPdt
# Function to solve the ODE over a given time period
def solve_ppg_periodic(duration=10, P0=100, R=1.0, C=1.0, I0=1.2, Ts=0.3, HR=60):
    """Solve the Windkessel ODE periodically for multiple cardiac cycles."""
    T = 60 / HR  # Duration of one cardiac cycle
    num_cycles = int(duration / T)  # Number of heartbeats in the duration
    t_full = []
    PPG_full = []    
    # Solve for each heartbeat separately
    for cycle in range(num_cycles):
        t_start = cycle * T  # Start time of this cycle
        t_eval = np.linspace(t_start, t_start + T, int(T * 1000))  # High-resolution time grid for one cycle        
        # Solve ODE for one cardiac cycle
        sol = solve_ivp(windkessel_ode, [t_start, t_start + T], [P0], args=(R, C, I0, Ts, HR),
t_eval=t_eval, method='RK45')        
        # Append the results to the full timeline
        t_full.extend(sol.t)
        PPG_full.extend(C * sol.y[0])  # Assume PPG is proportional to pressure    
    return np.array(t_full), np.array(PPG_full)
# Solve for 10 seconds to observe periodic output
t_periodic, PPG_periodic = solve_ppg_periodic()
# Plot periodic PPG waveform
plt.figure(figsize=(12, 4))
plt.plot(t_periodic, PPG_periodic, label="Periodic PPG Signal (10 sec)", color="black")
plt.xlabel("Time (s)")
plt.ylabel("PPG Signal")
plt.title("PPG Signal Derived from Windkessel Model (Periodic)")
plt.legend()
plt.grid()
plt.show()

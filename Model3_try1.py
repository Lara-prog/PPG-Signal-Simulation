# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 17:17:05 2025

@author: laram
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

    
# 3-Element Windkessel Model with Characteristic Impedance
def windkessel_3_element(t, P, R, C, Zc, I0, Ts, HR):
    """
    3-Element Windkessel Model ODE- Includes characteristic impedance (Zc) to model fast arterial pressure changes.
    - Provides more realistic wave propagation and reflections.
    """
    T = 60 / HR  # Cardiac cycle duration
    # Define input cardiac flow (Q) based on systolic phase
    if (t % T) < Ts:
        Q = I0 * np.sin(np.pi * (t % T) / Ts) ** 2  # Systolic phase
    else:
        Q = 0  # Diastolic phase
    # 3-Element Windkessel equation: dP/dt = (Q - P/Zc)/C - P/(R*C)
    dPdt = (Q - P / Zc) / C - (P / (R * C))
    return dPdt
# Solve the 3-element Windkessel ODE for periodic PPG simulation
def solve_3_element_windkessel(duration=10, P0=100, R=1.0, C=1.0, Zc=0.1, I0=1.2, Ts=0.3, HR=60):
    """
    Solve the 3-Element Windkessel Model for multiple cardiac cycles.
    - Zc models fast wave reflections and pressure transients.
    """
    T = 60 / HR  # Duration of one cardiac cycle
    num_cycles = int(duration / T)  # Number of heartbeats in the duration
    t_full = []
    PPG_full = []
        # Solve cycle-by-cycle to ensure periodicity
    for cycle in range(num_cycles):
        t_start = cycle * T  # Start time of this cycle
        t_eval = np.linspace(t_start, t_start + T, int(T * 1000))  # High-resolution time grid        
        # Solve ODE for one cardiac cycle
        sol = solve_ivp(windkessel_3_element, [t_start, t_start + T], [P0], args=(R, C, Zc, I0, Ts, HR), t_eval=t_eval, method='RK45')        
        # Append results to full timeline
        t_full.extend(sol.t)
        PPG_full.extend(C * sol.y[0])  # Assume PPG is proportional to arterial pressure    
    return np.array(t_full), np.array(PPG_full)
# Solve for 10 seconds using 3-Element Windkessel Model
t_3_element, PPG_3_element = solve_3_element_windkessel()
# Plot PPG waveform from 3-element Windkessel model
plt.figure(figsize=(12, 4))
plt.plot(t_3_element, PPG_3_element, label="PPG Signal (3-Element Windkessel, 10 sec)",
color="black")
plt.xlabel("Time (s)")
plt.ylabel("PPG Signal")
plt.title("PPG Signal Derived from 3-Element Windkessel Model with Characteristic Impedance")
plt.legend()
plt.grid()
plt.show()

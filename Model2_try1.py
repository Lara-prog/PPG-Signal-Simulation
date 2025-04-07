# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 17:14:54 2025

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
def solve_ppg_ode(duration=10, P0=100, R=1.0, C=1.0, I0=1.2, Ts=0.3, HR=60):
    """Solve the Windkessel ODE for PPG simulation."""
    t_eval = np.linspace(0, duration, duration * 1000)  # High-resolution time grid    
    # Solve the ODE using numerical integration
    sol = solve_ivp(windkessel_ode, [0, duration], [P0], args=(R, C, I0, Ts, HR), 
t_eval=t_eval, method='RK45')    
    # Extract pressure solution and assume PPG is proportional to pressure
    PPG = C * sol.y[0]  
    return sol.t, PPG
# Solve for 10 seconds to observe waveform details
t_ode, PPG_ode = solve_ppg_ode()

# Plot PPG waveform derived from ODE
plt.figure(figsize=(12, 4))
plt.plot(t_ode, PPG_ode, label="PPG Signal from Windkessel ODE", color="black")
plt.xlabel("Time (s)")
plt.ylabel("PPG Signal")
plt.title("PPG Signal Derived from Windkessel Model ODE")
plt.legend()
plt.grid()
plt.show()

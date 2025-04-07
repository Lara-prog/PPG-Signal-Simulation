# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 17:41:36 2025

@author: laram
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random 
from scipy.signal import butter, filtfilt, savgol_filter
import pywt

def c1_langewouters(pc, Am=3.5, l=1.0, P0=10, P1=35.0):
    """
    Time-varying 'central' compliance C1(pc):
      pc  : central pressure [mmHg]
      Am  : maximum cross-sectional area [cm^2]
      l   : length factor to get volume compliance
      P0  : transmural pressure at max compliance
      P1  : 'steepness' parameter
    Returns compliance in [ml/mmHg] (if units are consistent).
    """
    # A typical form from Langewouters: C1 = (Am*l)/(pi*P1) * 1/[1 + ((pc - P0)/P1)^2]
    # Adjust the scale factors to suit your data’s units
    return (Am * l / (np.pi * P1)) / (1.0 + ((pc - P0)/P1)**2)

def smooth_ramp(t, ramp_duration=1.5):
    """
    Smoothly transitions from 0 to 1 over `ramp_duration` seconds.
    """
    return min(t / ramp_duration, 1.0)

def variable_heart_rate(t, HR_mean=75, HRV=0.5):
    """
    Simulates Heart Rate Variability (HRV) by adding a small random fluctuation.
    HR_mean: Average heart rate (bpm).
    HRV: Maximum deviation from the mean HR (bpm).
    """
    return HR_mean + HRV * np.sin(0.1 * np.pi * t)  # Slow oscillation over time

def q_inflow(t, HR=60.0, Ts=0.35, alpha=1/3, q0=15.0, ramp_duration=1.5):
    """
    Piecewise inflow per cardiac cycle, from Eq. (2) in Xing et al.
    Repeats every T=60/HR [s].
    """
    HR_t = variable_heart_rate(t)
    T = 60.0 / HR_t
    t_mod = t % T

    if t_mod > Ts:
        return 0.0  # diastole, no inflow
    
    factor = smooth_ramp(t, ramp_duration)  # Apply ramping factor
    
    # During systole (0..Ts), we break it into two sub-intervals 0..alphaTs, alphaTs..Ts
    if t_mod <= alpha*Ts:
        # sin half-lobe
        return q0 * np.sin(np.pi * t_mod / (2.0*alpha*Ts))
    else:
        # cos half-lobe
        return q0 * np.cos(np.pi*(t_mod - alpha*Ts)/(4.0*alpha*Ts))

def noisy_param(param, noise_level=0.02):
    """
    Introduces random noise into a parameter.
    noise_level: Maximum percentage of variation (e.g., 0.05 for ±5% variation).
    """
    return param * (1 + noise_level * (2 * random.random() - 1))  # ± noise_level%

def smooth_R2_transition(t, HR, Ts, R2, scale=20, width=0.035):
    """Smooth transition function for R2 instead of abrupt changes"""
    T = 60 / HR
    t_mod = t % T
    notch_timing = Ts + 0.18  # Approximate dicrotic notch onset
    return R2 * (1 + (scale - 1) * np.exp(-((t_mod - notch_timing) / width) ** 2))

def calculate_pwv_delay(distance, PWV):
    """
    Calculate the time delay based on distance and pulse wave velocity (PWV).
    """
    return distance / PWV


def windkessel_ode(t, y, Ts, alpha, q0, R1, R2, C2, L,
                   Am, l, P0, P1, distance, PWV):
    """
    3-state ODE for the time-dependent Windkessel from Xing et al.:
      States: y = [ pc, pp, q ] 
        pc: central pressure [mmHg]
        pp: peripheral pressure [mmHg]
        q : flow through L and R1 [ml/s]
    """
    pc, pp, flow = y
    R1 = noisy_param(R1)
    R2 = noisy_param(R2)
    C2 = noisy_param(C2)
    
    # Compute time-varying heart rate
    HR_t = variable_heart_rate(t)
    
    # time-varying compliance, depends on pc
    C1 = c1_langewouters(pc, Am, l, P0, P1)

    # inflow from the left ventricle
    qi = q_inflow(t, HR_t, Ts, alpha, q0)

    # Calculate the time delay based on PWV and distance
    delay_time = calculate_pwv_delay(distance, PWV)
    
    # Introduce the delay to peripheral pressure
    # Use t - delay_time to simulate the lag in the peripheral pressure
    pp_delayed = pp  # Initially, set pp to the current value
    if t > delay_time:
        pp_delayed = pp  # Shift the peripheral pressure signal by delay_time
        
    # ODE eq(1a):
    #   d(flow)/dt = [pc - pp - R1*flow]/L
    dflow_dt = (pc - pp_delayed - R1*flow) / L

    # ODE eq(1b):
    #   d(pc)/dt = [qi - flow]/C1(t)
    #   (some references clamp C1>0 in case pc < P0 and it becomes large or negative)
    if C1 <= 5e-6:
        # avoid dividing by zero, clamp or skip
        dpc_dt = 0.0
    else:
        dpc_dt = (qi - flow) / C1

    R2_temp = smooth_R2_transition(t, HR_t, Ts, R2)

    
    # ODE eq(1c):
    #   d(pp)/dt = [flow - pp/R] / C2
    dpp_dt = (flow - pp_delayed/R2_temp) / C2

    
    # **1. Small oscillation to enhance dicrotic notch**
    if qi == 0:  # During diastole
        dpc_dt += 0.05 * np.sin(7 * np.pi * t)  # Small high-frequency oscillation
        dpc_dt -= 0.15 * np.exp(-6 * (t % (60/HR) - Ts))  # Sharp reduction at systole end

    # **2. Explicit aortic valve closure event**
    if Ts + 0.08 < t % (60 / HR) < Ts + 0.12:  # Just after systole
        dpc_dt -= 15.0 * np.exp(-15 * (t % (60/HR) - (Ts + 0.08)))  # Short dip post-systole
        dpc_dt += 40.0 * np.exp(15 * (t % (60/HR) - (Ts + 0.08)))

    return [dpc_dt, dpp_dt, dflow_dt]


def simulate_wk4_timevary(duration=10.0,
                          HR=60.0, Ts=0.35, alpha=1/3, q0=15.0,
                          R1=0.2, R2=5.0, C2=0.1, L=0.02,
                          Am=3.5, l=1.0, P0=10.0, P1=35.0,
                          pc0=20.0, pp0=20.5, q0_init=0.0, PWV=5.0, distance=1.0):
    """
    Solve the modified 4-element windkessel with time-varying C1(pc).
    
    Parameters
    ----------
    duration : float
        Total simulation time (s).
    HR       : float
        Heart rate (beats per minute).
    Ts       : float
        Systolic ejection duration (s).
    alpha    : float
        Splits the inflow waveform into sin/cos segments.
    q0       : float
        Peak inflow (ml/s).
    R1, R2, C2, L : float
        Model parameters from the paper.
    Am, l, P0, P1 : float
        Parameters for the Langewouters compliance function C1(pc).
    pc0, pp0, q0_init : float
        Initial conditions for pc, pp, and q, respectively.
    """
    y0 = [pc0, pp0, q0_init]
    t_eval = np.linspace(0, duration, 2000)

    sol = solve_ivp(
        fun=windkessel_ode,
        t_span=[0, duration],
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        args=(Ts, alpha, q0, R1, R2, C2, L, Am, l, P0, P1, PWV, distance)
    )

    return sol.t, sol.y[0], sol.y[1], sol.y[2]



if __name__=="__main__":
    # Example usage:
    T_sim = 8.0          # simulate 8 seconds
    # From paper: T = 0.8 s at HR=75 => Ts=0.35, alpha=1/3, etc.
    HR   = 75.0
    Ts   = 0.35
    alpha= 1/3
    q0   = 20.0          # peak inflow, ~ stroke volume / ejection time
    R1   = 0.2
    R2   = 5.0
    C2   = 0.05
    L    = 0.02

    # Parameters for time-varying compliance:
    Am   = 6.18          # from paper example
    l    = 1.0
    P0   = 10
    P1   = 35.0
    pc0  = 20.0
    pp0  = 20.5
    q0_init = 10.0
    
    # Set values for PWV and distance:
    PWV = 5.0  # Pulse wave velocity in m/s
    distance = 1.0  # Distance between central and peripheral pressure (meters)

    # Solve
    t, pc, pp, flow = simulate_wk4_timevary(T_sim, HR, Ts, alpha, q0,
                                            R1, R2, C2, L,
                                            Am, l, P0, P1,
                                            pc0, pp0, q0_init, PWV, distance)
    
    # Apply wavelet decomposition and enhancement
    coeffs = pywt.wavedec(pc, 'db6', level=4)  # Decompose signal
    coeffs[1:] = [c * 1.3 for c in coeffs[1:]]  # Enhance details
    pc_filtered = pywt.waverec(coeffs, 'db6')  # Reconstruct


    # Plot results
    plt.figure(figsize=(10,5))
    plt.plot(t, pc, label='Central Pressure pc(t)', color='red', linewidth =2)
    plt.plot(t, pp, label='Peripheral Pressure pp(t)', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.title('Modified 4-element WK with BP-dependent C1(pc)')
    plt.grid(True)
    plt.legend()
    plt.show()

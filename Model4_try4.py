# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 17:40:42 2025

@author: laram
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random 
from scipy.signal import butter, filtfilt, savgol_filter
import pywt

def c1_langewouters(pc, Am=3.5, l=1.0, P0=10, P1=42.3):
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

def noisy_param(param, noise_level=0.01):
    """
    Introduces random noise into a parameter.
    noise_level: Maximum percentage of variation (e.g., 0.05 for ±5% variation).
    """
    return param * (1 + noise_level * (2 * random.random() - 1))  # ± noise_level%

def windkessel_ode(t, y, HR, Ts, alpha, q0, R1, R2, C2, L,
                   Am, l, P0, P1):
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
    
    # time-varying compliance, depends on pc
    C1 = c1_langewouters(pc, Am, l, P0, P1)

    # inflow from the left ventricle
    qi = q_inflow(t, HR, Ts, alpha, q0)

    # ODE eq(1a):
    #   d(flow)/dt = [pc - pp - R1*flow]/L
    dflow_dt = (pc - pp - R1*flow) / L

    # ODE eq(1b):
    #   d(pc)/dt = [qi - flow]/C1(t)
    #   (some references clamp C1>0 in case pc < P0 and it becomes large or negative)
    if C1 <= 5e-6:
        # avoid dividing by zero, clamp or skip
        dpc_dt = 0.0
    else:
        dpc_dt = (qi - flow) / C1

    if Ts + 0.08 < t % (60 / HR) < Ts + 0.18:
        R2_temp = R2 * 20  # Increase temporarily R2
    else:
        R2_temp = R2
    
    # ODE eq(1c):
    #   d(pp)/dt = [flow - pp/R] / C2
    dpp_dt = (flow - pp/R2_temp) / C2

    
    # **1. Small oscillation to enhance dicrotic notch**
    if qi == 0:  # During diastole
        dpc_dt += 0.03 * np.sin(5 * np.pi * t)  # Small high-frequency oscillation
        dpc_dt -= 0.2 * np.exp(-5 * (t % (60/HR) - Ts))  # Sharp reduction at systole end

    # **2. Explicit aortic valve closure event**
    if Ts + 0.08 < t % (60 / HR) < Ts + 0.12:  # Just after systole
        dpc_dt -= 4.0 * np.exp(-5 * (t % (60/HR) - (Ts + 0.04)))  # Short dip post-systole
        dpc_dt += 2.0 * np.exp(-5 * (t % (60/HR) - (Ts + 0.04)))
        #Ãdpc_dt += 5.0 * np.sin(50 * np.pi * (t % (60/HR) - Ts - 0.08))

    return [dpc_dt, dpp_dt, dflow_dt]


def simulate_wk4_timevary(duration=10.0,
                          HR=60.0, Ts=0.35, alpha=1/3, q0=15.0,
                          R1=0.2, R2=5.0, C2=0.1, L=0.02,
                          Am=3.5, l=1.0, P0=10.0, P1=42.3,
                          pc0=17.5, pp0=17.0, q0_init=0.0):
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
        args=(HR, Ts, alpha, q0, R1, R2, C2, L, Am, l, P0, P1)
    )

    return sol.t, sol.y[0], sol.y[1], sol.y[2]

# Filtrage avec Butterworth
def butter_lowpass_filter(data, cutoff=5, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Filtrage avec Savitzky-Golay
def savgol_smoothing(data, window_length=11, polyorder=3):
    return savgol_filter(data, window_length, polyorder)

# Filtrage avec ondelettes
def wavelet_denoising(data, wavelet='db4', level=2):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, np.std(c)/2, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

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
    P1   = 42.3
    pc0  = 17.0
    pp0  = 17.5
    q0_init = 10.0

    # Solve
    t, pc, pp, flow = simulate_wk4_timevary(T_sim, HR, Ts, alpha, q0,
                                            R1, R2, C2, L,
                                            Am, l, P0, P1,
                                            pc0, pp0, q0_init)
    
    # Application of filters on pressions
    pc_filtered_butter = butter_lowpass_filter(pc)
    pc_filtered_savgol = savgol_smoothing(pc)
    pc_filtered_wavelet = wavelet_denoising(pc)
    
    pp_filtered_butter = butter_lowpass_filter(pp)
    pp_filtered_savgol = savgol_smoothing(pp)
    pp_filtered_wavelet = wavelet_denoising(pp)
    
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

    #Plot Results of filters
    plt.figure(figsize=(12, 6))
    plt.plot(t, pc, label="Original pc", alpha=0.7)
    plt.plot(t, pc_filtered_butter, label="Butterworth", linestyle="--")
    plt.plot(t, pc_filtered_savgol, label="Savitzky-Golay", linestyle="-.")
    plt.plot(t, pc_filtered_wavelet, label="Wavelet", linestyle=":")
    plt.legend()
    plt.title("Filtrage du signal de pression centrale")
    plt.xlabel("Temps (s)")
    plt.ylabel("Pression (mmHg)")
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, pp, label="Original pp", alpha=0.7)
    plt.plot(t, pp_filtered_butter, label="Butterworth", linestyle="--")
    plt.plot(t, pp_filtered_savgol, label="Savitzky-Golay", linestyle="-.")
    plt.plot(t, pp_filtered_wavelet, label="Wavelet", linestyle=":")
    plt.legend()
    plt.title("Filtrage du signal de pression périphérique")
    plt.xlabel("Temps (s)")
    plt.ylabel("Pression (mmHg)")
    plt.grid()
    plt.show()

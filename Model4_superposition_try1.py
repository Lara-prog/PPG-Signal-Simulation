# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 17:42:14 2025

@author: laram
"""

# Code real ppg signal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# Define the GitHub raw CSV URL
url = "https://raw.githubusercontent.com/godamartonaron/GODA_pyPPG/main/sample_data/Sample_PPG_CSV_125Hz.csv"

# Load CSV (single-column data, no header)
ppg_data = pd.read_csv(url, header=None)

# Extract PPG signal
ppg_signal = ppg_data.iloc[:, 0]

# Define Sampling Rate (Extracted from File Name: 125Hz)
fs = 125  # Hz

# Compute the number of samples for the first 10 seconds
num_samples = fs * 10  # 10 seconds of data

# Limit data to first 10 seconds
ppg_signal = ppg_signal[:num_samples]

# Generate time axis
time = np.arange(len(ppg_signal)) / fs  # Time in seconds

# Plot the first 10 seconds of the PPG signal
plt.figure(figsize=(12, 5))
plt.plot(time, ppg_signal, label="PPG Signal (First 10s)", color="b")
plt.title("PPG Signal from GitHub Sample Data (First 10 Seconds, 125 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("PPG Amplitude")
plt.legend()
plt.grid()
plt.show()

# My code
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random 
from scipy.signal import butter, filtfilt, savgol_filter
import pywt
from scipy.signal import correlate
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import interp1d

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
    return max(0, param * (1 + noise_level * (2 * random.random() - 1)))

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
    t_eval = np.linspace(0, duration, 1250)
    

    sol = solve_ivp(
        fun=windkessel_ode,
        t_span=[0, duration],
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        args=(Ts, alpha, q0, R1, R2, C2, L, Am, l, P0, P1, PWV, distance)
    )

    return sol.t, sol.y[0], sol.y[1], sol.y[2]

def lowpass_filter(signal, cutoff=6, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

if __name__=="__main__":
    # Example usage:
    T_sim = 9.8         # simulate 8 seconds
    # From paper: T = 0.8 s at HR=75 => Ts=0.35, alpha=1/3, etc.
    HR   = 70.0
    Ts   = 0.30
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
    coeffs = pywt.wavedec(pc, 'db6', level=4)
    coeffs[1:] = [c * 1.2 for c in coeffs[1:]]
    pc_filtered = pywt.waverec(coeffs, 'db6')
    
    # synthetic signal filter
    pc_filtered = lowpass_filter(pc_filtered, cutoff=6, fs=125, order=4)
    
    scale_factor = np.max(ppg_signal) / np.max(pc_filtered)
    pc_filtered *= scale_factor
    time_simulated = np.linspace(0, T_sim, 1250)
    ppg_signal_normalized = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
    pc_filtered_normalized = (pc_filtered - np.mean(pc_filtered)) / np.std(pc_filtered)
    
    # Maximum cross-correlation to estimate the best time lag
    correlation = correlate(ppg_signal, pc_filtered, mode='full')
    best_shift = np.argmax(correlation) - len(ppg_signal) + 1
    print(f'Décalage optimal : {best_shift / fs:.4f} s')

    # Synthetic signal time alignment
    time_shift = best_shift / fs
    time_simulated_shifted = time_simulated + time_shift
    
    interp_func = interp1d(time_simulated_shifted, pc_filtered, kind='cubic', fill_value="extrapolate")
    pc_filtered_shifted = interp_func(time)



    # RMSE
    min_len = min(len(ppg_signal_normalized), len(pc_filtered_shifted))
    rmse = np.sqrt(mean_squared_error(ppg_signal_normalized[:min_len], pc_filtered_shifted[:min_len]))
    print(f'RMSE : {rmse:.4f}')
    
    # Pearson Correlation
    pearson_corr, _ = pearsonr(ppg_signal, pc_filtered)
    print(f'Corrélation de Pearson : {pearson_corr:.4f}')
    
    # MAE
    mae = mean_absolute_error(ppg_signal, pc_filtered)
    print(f'MAE : {mae:.4f}')

      
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
    
    # Signal superposition
    plt.figure(figsize=(12, 5))
    plt.plot(time, ppg_signal_normalized, label="Real PPG", color='b', alpha=0.7)
    plt.plot(time_simulated_shifted, pc_filtered_normalized[:len(time)], label="Synthetic PPG", color='r', alpha=0.7)
    plt.title("Superposition of Real and Synthetic PPG")
    plt.xlabel("Time (s)")
    plt.ylabel("PPG Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

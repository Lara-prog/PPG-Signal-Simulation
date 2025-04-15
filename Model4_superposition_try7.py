# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 17:22:03 2025

@author: laram
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 16:24:16 2025

@author: laram
"""
# Code real ppg signal
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
from fastdtw import fastdtw
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq


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

def variable_heart_rate(t, HR_mean=75, HRV=0.8):
    """
    Simulates heart rate variability with more pronounced oscillation
    HR_mean: Average heart rate (bpm)
    HRV: Maximum amplitude of variation (bpm)
    """
    # Addition of a more complex component to match the real signal
    return HR_mean + HRV * np.sin(0.1 * np.pi * t) + 0.5 * HRV * np.sin(0.25 * np.pi * t)


def q_inflow(t, HR=60.0, Ts=0.35, alpha=1/3, q0=15.0, ramp_duration=1.5):
    """
    Piecewise inflow per cardiac cycle, from Eq. (2) in Xing et al.
    Repeats every T=60/HR [s].
    """
    HR_t = variable_heart_rate(t, HR_mean=HR)
    T = 60.0 / HR_t # Variable period
    t_mod = t % T  #  Modulo in relation to the variable period


    if t_mod > Ts:
        return 0.0  # Diastole, no inflow

    factor = smooth_ramp(t, ramp_duration)  # Apply ramping factor

    # During systole (0..Ts), we break it into two sub-intervals 0..alphaTs, alphaTs..Ts
    if t_mod <= alpha * Ts:
        # sin half-lobe
        return q0 * factor * np.sin(np.pi * t_mod / (2.0 * alpha * Ts))
    else:
        # cos half-lobe (corrected expression)
        return q0 * factor * np.cos(np.pi * (t_mod - alpha * Ts) / (2.0 * (1 - alpha) * Ts))


def noisy_param(param, noise_level=0.02):
    """
    Introduces random noise into a parameter.
    noise_level: Maximum percentage of variation (e.g., 0.05 for ±5% variation).
    """
    return max(0, param * (1 + noise_level * (2 * random.random() - 1)))

def smooth_R2_transition(t, HR, Ts, R2, scale=15, width=0.028):
    """
    Enhanced transition for R2 with synchronized timing between central and peripheral
    """
    T = 60 / HR
    t_mod = t % T
    
    # Common timing for both central and peripheral notches
    notch_timing = Ts + 0.04  # Earlier timing
    
    # Very dramatic scale factor
    transition = R2 * (1 + (scale * 2.5 - 1) * np.exp(-((t_mod - notch_timing) / (width * 0.5)) ** 2))
    
    # Enhanced secondary component with same timing as central notch
    secondary_timing = notch_timing + 0.025
    if t_mod > secondary_timing:
        # Significantly increased amplitude for dramatic notch effect
        transition += 0.8 * R2 * np.exp(-((t_mod - secondary_timing) / 0.018) ** 2)
        
    return transition


def calculate_pwv_delay(distance, PWV):
    """
    Calculate realistic delay between central and peripheral pressures
    """
    # Simple physical model with realistic delay
    return distance / PWV  # This should give delay in seconds

def windkessel_ode(t, y, Ts, alpha, q0, R1, R2, C2, L,
                  Am, l, P0, P1, distance, PWV, HR):
    """
    Modified ODE with synchronized and pronounced dicrotic notches
    for both central and peripheral pressure
    """
    pc, pp, flow = y
    R1 = noisy_param(R1, noise_level=0.003)  # Reduced noise for cleaner notches
    R2 = noisy_param(R2, noise_level=0.003)
    C2 = noisy_param(C2, noise_level=0.003)

    # Variable heart rate
    HR_t = variable_heart_rate(t, HR_mean=HR, HRV=0.5)
    T = 60.0 / HR_t

    # Variable compliance
    C1 = c1_langewouters(pc, Am, l, P0, P1)

    # Inflow
    qi = q_inflow(t, HR=HR_t, Ts=Ts, alpha=alpha, q0=q0)

    # Almost eliminate delay to synchronize notches
    delay_time = calculate_pwv_delay(distance, PWV)  # Significantly reduced delay
    pp_delayed = pp

    # ODE for flow
    dflow_dt = (pc - pp_delayed - R1 * flow) / L

    # ODE for central pressure
    if C1 <= 5e-6:
        dpc_dt = 0.0
    else:
        dpc_dt = (qi - flow) / C1 * min(1.0, t/(t+0.05))

    # Enhanced R2 transition with synchronized timing
    R2_temp = smooth_R2_transition(t, HR_t, Ts, R2, scale=30, width=0.018)  # Further increased scale

    # ODE for peripheral pressure with strong notch effect
    dpp_dt = (flow - pp_delayed / R2_temp) / C2

    # Reduce background oscillations
    if qi == 0 and t > 1.0:
        dpc_dt += 0.01 * np.sin(9 * np.pi * t)
        dpc_dt -= 0.08 * np.exp(-8 * (t % T - Ts))

    # Common timing for valve closure effect in both signals
    valve_close_start = Ts + 0.03 * (T / 1.0)
    valve_close_end = Ts + 0.055 * (T / 1.0)

    t_cycle = t % T
    if valve_close_start < t_cycle < valve_close_end and t > 0.5:
        # Dramatic central pressure notch effect
        dpc_dt -= 55.0 * np.exp(-30 * (t_cycle - valve_close_start))
        dpc_dt += 100.0 * np.exp(40 * (t_cycle - (Ts + 0.04 * (T / 1.0))))
        
        # Matching peripheral pressure notch effect with identical timing
        dpp_dt -= 55.0 * np.exp(-30 * (t_cycle - valve_close_start))
        dpp_dt += 100.0 * np.exp(40 * (t_cycle - (Ts + 0.04 * (T / 1.0))))

    # Very strong coupling between central and peripheral for synchronized features
    # Replace the strong coupling with more realistic values
    if t > 0.2:
        # Reduced coupling to allow natural timing differences
        dpp_dt += 0.05 * dpc_dt  # Central affects peripheral (reduced)
        dpc_dt += 0.01 * dpp_dt  # Peripheral affects central (minimal)


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
        args=(Ts, alpha, q0, R1, R2, C2, L, Am, l, P0, P1, PWV, distance, HR)
    )

    return sol.t, sol.y[0], sol.y[1], sol.y[2]

def lowpass_filter(signal, cutoff=6, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

def introduce_synthesis_randomness(signal):
    """
    Enhanced signal randomization that preserves synchronized features
    """
    # Use wavelet decomposition that preserves sharp features
    coeffs = pywt.wavedec(signal, 'db6', level=7)
    
    # Enhanced coefficient adjustments to maintain notch characteristics
    coeffs[1] = coeffs[1] * 1.3   # Significantly boost high frequencies for sharper notches
    coeffs[2] = coeffs[2] * 1.2   # Enhance level 2 details
    coeffs[3] = coeffs[3] * 1.15  # Enhance level 3 details
    coeffs[4] = coeffs[4] * 1.05
    coeffs[5] = coeffs[5] * 0.95
    coeffs[6:] = [c * 0.90 for c in coeffs[6:]]
    
    # Minimal random perturbations to preserve notch synchronization
    for i in range(1, 5):
        # Use same random seed for consistent randomization
        np.random.seed(42 + i)
        noise = np.random.normal(0, 0.025, len(coeffs[i]))
        coeffs[i] = coeffs[i] + noise * np.std(coeffs[i])
    
    return pywt.waverec(coeffs, 'db6')

def process_synthetic_signal(pc, pp, ppg_signal, fs=125):
    """
    Enhanced signal processing with synchronization of central and peripheral pressures,
    preserving sharp features like the dicrotic notch.
    """
    # Apply the same controlled randomness to both signals
    pc_randomized = introduce_synthesis_randomness(pc)
    pp_randomized = introduce_synthesis_randomness(pp)

    # Filtering parameters chosen to preserve sharp features
    filter_cutoff = 10
    filter_order = 3
    pc_filtered = lowpass_filter(pc_randomized, cutoff=filter_cutoff, fs=fs, order=filter_order)
    pp_filtered = lowpass_filter(pp_randomized, cutoff=filter_cutoff, fs=fs, order=filter_order)

    # Normalize both signals using the same scale factor to retain their relative shape
    scale_factor = 1.05 * np.max(ppg_signal) / np.max(pc_filtered)
    pc_filtered *= scale_factor
    pp_filtered *= scale_factor

    # Create a time vector matching the synthetic signal
    time_simulated = np.linspace(0, len(pc_filtered) / fs, len(pc_filtered))

    # Truncate all signals to the minimum common length
    min_len = min(len(time), len(ppg_signal), len(pc_filtered))
    time_common = time[:min_len]
    ppg_signal_common = ppg_signal[:min_len]
    pc_filtered_common = pc_filtered[:min_len]
    pp_filtered_common = pp_filtered[:min_len]

    # Normalize all signals for DTW alignment
    ppg_signal_normalized = (ppg_signal_common - np.mean(ppg_signal_common)) / np.std(ppg_signal_common)
    pc_filtered_normalized = (pc_filtered_common - np.mean(pc_filtered_common)) / np.std(pc_filtered_common)
    pp_filtered_normalized = (pp_filtered_common - np.mean(pp_filtered_common)) / np.std(pp_filtered_common)

    # Align central pressure with PPG signal and apply the same shift to peripheral pressure
    start_idx = int(0.2 * fs)
    if start_idx < min_len:
        distance, path = fastdtw(ppg_signal_normalized[start_idx:], pc_filtered_normalized[start_idx:])
        index_shifts = [p1 - p2 for p1, p2 in path]
        best_shift = float(np.median(index_shifts)) / fs


        time_simulated_shifted = time_simulated[:min_len] + best_shift - 0.02

        # Use cubic interpolation to resample both signals on the common time base
        interp_func_pc = interp1d(time_simulated_shifted, pc_filtered_common, kind='cubic', fill_value="extrapolate")
        interp_func_pp = interp1d(time_simulated_shifted, pp_filtered_common, kind='cubic', fill_value="extrapolate")

        pc_filtered_shifted = interp_func_pc(time_common)
        pp_filtered_shifted = interp_func_pp(time_common)
    else:
        best_shift = 0
        time_simulated_shifted = time_simulated[:min_len]
        pc_filtered_shifted = pc_filtered_common
        pp_filtered_shifted = pp_filtered_common

    # Return synchronized and normalized signals with auxiliary information
    return (
        pc_filtered_shifted,               # Aligned central pressure
        pp_filtered_shifted,               # Aligned peripheral pressure
        ppg_signal_normalized,             # Normalized PPG signal
        best_shift,                        # Estimated time shift
        time_simulated_shifted,            # Shifted time axis
        pc_filtered_normalized,            # Normalized central pressure
        pp_filtered_normalized,            # Normalized peripheral pressure
        pp_randomized[:min_len],           # Randomized peripheral pressure (pre-filter)
        pc_filtered_common,                # Truncated filtered central pressure (pre-alignment)
        pc_randomized[:min_len]            # Randomized central pressure (pre-filter)
    )


def compute_error_metrics(ppg_signal, pc_filtered_shifted):
    min_len = min(len(ppg_signal), len(pc_filtered_shifted))
    rmse = np.sqrt(mean_squared_error(ppg_signal[:min_len], pc_filtered_shifted[:min_len]))
    pearson_corr, _ = pearsonr(ppg_signal[:min_len], pc_filtered_shifted[:min_len])
    mae = mean_absolute_error(ppg_signal[:min_len], pc_filtered_shifted[:min_len])

    print(f'RMSE: {rmse:.4f}')
    print(f'Pearson Correlation: {pearson_corr:.4f}')
    print(f'MAE: {mae:.4f}')


if __name__=="__main__":
    # Example usage:
    T_sim = 9.8         # simulate 8 seconds
    # From paper: T = 0.8 s at HR=75 => Ts=0.35, alpha=1/3, etc.
    HR   = 102.0
    Ts   = 0.31
    alpha= 0.36
    q0   = 17.0          # peak inflow, ~ stroke volume / ejection time
    R1   = 0.16
    R2   = 0.70
    C2   = 0.18
    L    = 0.011

    # Parameters for time-varying compliance:
    Am   = 6.0          # from paper example
    l    = 1.0
    P0   = 13.5
    P1   = 21.0
    pc0  = 10.0
    pp0  = 10.0
    q0_init = 0.0
    
    # Set values for PWV and distance:
    PWV = 6  # Pulse wave velocity in m/s
    distance = 1  # Distance between central and peripheral pressure (meters)

    # ---- MODEL SIMULATION ----
    t, pc, pp, flow = simulate_wk4_timevary(T_sim, HR, Ts, alpha, q0,
                                            R1, R2, C2, L,
                                            Am, l, P0, P1,
                                            pc0, pp0, q0_init, PWV, distance)
    
    # ---- SIGNAL PROCESSING ----
    # Process the synthetic signal and peripheral pressure
    pc_filtered_shifted, pp_filtered_shifted, ppg_signal_normalized, time_shift, time_simulated_shifted, \
    pc_filtered_normalized, pp_filtered_normalized, pp_randomized, pc_filtered_common, pc_randomized = \
    process_synthetic_signal(pc, pp, ppg_signal)



    print(f'Optimal shift: {time_shift:.4f} s')

    
    # ---- ERROR METRICS ----
    compute_error_metrics(ppg_signal, pc_filtered_shifted)
    
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

 
    # Plot Original and Filtered Signals (including pp)
    plt.figure(figsize=(12, 5))
    plt.plot(t, pc, label="Original Central Pressure pc(t)", color='gray', linestyle='dashed', alpha=0.7)
    plt.plot(t, pp, label="Original Peripheral Pressure pp(t)", color='blue', linestyle='dotted', alpha=0.7)  # Original pp
    plt.plot(t, pp_filtered_normalized[:len(time)], label="Filtered Peripheral Pressure pp(t)", color='g', alpha=0.7)  # Filtered pp
    plt.plot(t, pc_randomized, label="Randomized Central Pressure pc(t)", color='purple', alpha=0.7)
    plt.plot(t, pc_filtered_normalized[:len(time)], label="Filtered Central Pressure pc(t)", color='red', linewidth=2)
    plt.title("Comparison of Original, Randomized, Filtered Synthetic PPG, and Peripheral Pressure pp(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Signal Superposition (Real vs Synthetic PPG including pp)
    plt.figure(figsize=(12, 5))
    plt.plot(time, ppg_signal_normalized, label="Real PPG", color='b', alpha=0.7)
    plt.plot(time_simulated_shifted, pc_filtered_normalized[:len(time)], label="Filtered Central Pressure pc(t)", color='r', alpha=0.7)
    plt.plot(t, pp_filtered_normalized, label="Filtered Peripheral Pressure pp(t)", color='g', linestyle='dashed', alpha=0.7)  # Normalized pp
    plt.title("Superposition of Real, Synthetic PPG, and Filtered Peripheral Pressure pp(t)")
    plt.xlabel("Time (s)")
    plt.ylabel("PPG Amplitude")
    plt.legend()
    plt.grid()
    plt.show()

    
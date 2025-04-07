# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 17:18:06 2025

@author: laram
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def c1_langewouters(pc, Am=3.5, l=1.0, P0=50.4, P1=42.3):
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

def q_inflow(t, HR=60.0, Ts=0.35, alpha=1/3, q0=15.0):
    """
    Piecewise inflow per cardiac cycle, from Eq. (2) in Xing et al.
    Repeats every T=60/HR [s].
    """
    T = 60.0 / HR
    t_mod = t % T

    if t_mod > Ts:
        return 0.0  # diastole, no inflow

    # During systole (0..Ts), we break it into two sub-intervals 0..alphaTs, alphaTs..Ts
    if t_mod <= alpha*Ts:
        # sin half-lobe
        return q0 * np.sin(np.pi * t_mod / (2.0*alpha*Ts))
    else:
        # cos half-lobe
        return q0 * np.cos(np.pi*(t_mod - alpha*Ts)/(4.0*alpha*Ts))

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

    # time-varying compliance, depends on pc
    C1 = c1_langewououters = c1_langewouters(pc, Am, l, P0, P1)

    # inflow from the left ventricle
    qi = q_inflow(t, HR, Ts, alpha, q0)

    # ODE eq(1a):
    #   d(flow)/dt = [pc - pp - R1*flow]/L
    dflow_dt = (pc - pp - R1*flow) / L

    # ODE eq(1b):
    #   d(pc)/dt = [qi - flow]/C1(t)
    #   (some references clamp C1>0 in case pc < P0 and it becomes large or negative)
    if C1 <= 1e-6:
        # avoid dividing by zero, clamp or skip
        dpc_dt = 0.0
    else:
        dpc_dt = (qi - flow) / C1

    # ODE eq(1c):
    #   d(pp)/dt = [flow - pp/R] / C2
    dpp_dt = (flow - pp/R2) / C2

    return [dpc_dt, dpp_dt, dflow_dt]

def simulate_wk4_timevary(duration=10.0,
                          HR=60.0, Ts=0.35, alpha=1/3, q0=15.0,
                          R1=0.05, R2=1.0, C2=0.1, L=0.02,
                          Am=3.5, l=1.0, P0=50.4, P1=42.3,
                          pc0=80.0, pp0=70.0, q0_init=0.0):
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

if __name__=="__main__":
    # Example usage:
    T_sim = 8.0          # simulate 8 seconds
    # From paper: T = 0.8 s at HR=75 => Ts=0.35, alpha=1/3, etc.
    HR   = 75.0
    Ts   = 0.35
    alpha= 1/3
    q0   = 20.0          # peak inflow, ~ stroke volume / ejection time
    R1   = 0.05
    R2   = 1.0
    C2   = 0.1
    L    = 0.02

    # Parameters for time-varying compliance:
    Am   = 6.18          # from paper example
    l    = 1.0
    P0   = -2.3
    P1   = 21.6
    pc0  = 80.0
    pp0  = 70.0
    q0_init = 0.0

    # Solve
    t, pc, pp, flow = simulate_wk4_timevary(T_sim, HR, Ts, alpha, q0,
                                            R1, R2, C2, L,
                                            Am, l, P0, P1,
                                            pc0, pp0, q0_init)
    

    # Plot results
    plt.figure(figsize=(10,5))
    plt.plot(t, pc, label='Central Pressure pc(t)', color='red')
    plt.plot(t, pp, label='Peripheral Pressure pp(t)', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (mmHg)')
    plt.title('Modified 4-element WK with BP-dependent C1(pc)')
    plt.grid(True)
    plt.legend()
    plt.show()

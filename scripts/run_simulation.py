"""
SIRA Simulation Execution Script.
Runs the deterministic ODE and exact Gillespie simulator side-by-side.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sira.simulation.deterministic import DeterministicEpidemic
from sira.simulation.gillespie import GillespieSimulator

def main():
                      # 1. Epidemic Configuration
    N = 1000          # Small population ensures highly visible stochastic noise
    I0 = 5            # Patient Zeros
    R0 = 0
    S0 = N - I0 - R0
    initial_state = [S0, I0, R0]
    
    params = {'beta': 0.3, 'gamma': 0.1}
    t_max = 100.0

    print(f"Running SIR Simulation (N={N}, beta={params['beta']}, gamma={params['gamma']})...")
    t_ode = np.linspace(0, t_max, 1000)
    y_ode = odeint(
        DeterministicEpidemic.sir, 
        initial_state, 
        t_ode, 
        args=(N, params['beta'], params['gamma'])
    )
    S_ode, I_ode, R_ode = y_ode.T

    simulator = GillespieSimulator(model_type="SIR", N=N, params=params)
    t_stoch, y_stoch = simulator.simulate(initial_state, t_max=t_max)
    
    S_stoch, I_stoch, R_stoch = y_stoch.T

    # 4. Visualization & Benchmarking
    plt.figure(figsize=(10, 6))
    
    # Plot Continuous ODE (Smooth Dashed Lines)
    plt.plot(t_ode, S_ode, 'b--', alpha=0.8, linewidth=2, label='S (ODE)')
    plt.plot(t_ode, I_ode, 'r--', alpha=0.8, linewidth=2, label='I (ODE)')
    plt.plot(t_ode, R_ode, 'g--', alpha=0.8, linewidth=2, label='R (ODE)')
    
    # Plot Discrete CTMC (Jagged Step Lines)
    # plt.step() ensures the graph jumps vertically, representing exact integer changes
    plt.step(t_stoch, S_stoch, 'b', alpha=0.4, where='post', label='S (Stochastic)')
    plt.step(t_stoch, I_stoch, 'r', alpha=0.7, linewidth=1.5, where='post', label='I (Stochastic)')
    plt.step(t_stoch, R_stoch, 'g', alpha=0.4, where='post', label='R (Stochastic)')
    
    plt.title(f"SIRA Engine: Deterministic Vector Field vs. Exact CTMC (N={N})", fontsize=14)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Population", fontsize=12)
    plt.legend(loc="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Render to screen
    plt.show()

if __name__ == "__main__":
    main()
"""
Epidemiological Simulation Module

Provides functions to generate both deterministic and stochastic 
data for the Susceptible-Infected-Removed (SIR) model. 

Implements the exact Gillespie Stochastic Simulation Algorithm (SSA) 
to generate realistic, noisy observation ensembles for system identification.
"""

import numpy as np
from scipy.integrate import solve_ivp
import warnings

# Suppress minor integration warnings during standard evaluation
warnings.filterwarnings("ignore", category=UserWarning)


def calculate_r0(beta: float, gamma: float) -> float:
    """
    Calculates the Basic Reproduction Number (R0).
    """
    if gamma == 0:
        return float('inf')
    return beta / gamma


def is_epidemic_active(beta: float, gamma: float) -> bool:
    """
    Determines if the epidemic parameters result in active spread (R0 > 1.1).
    Used to filter out configurations where the initial infection dies out immediately.
    """
    return calculate_r0(beta, gamma) > 1.1


def sir_deterministic(t: float, y: list, beta: float, gamma: float) -> list:
    """
    Defines the standard Ordinary Differential Equations (ODEs) for the SIR model.
    Used to establish the true deterministic baseline for accuracy validation.
    
    Args:
        t: Current time.
        y: State vector containing population proportions [S, I].
        beta: Infection rate parameter.
        gamma: Recovery rate parameter.
        
    Returns:
        State derivatives [dS/dt, dI/dt].
    """
    S, I = y[0], y[1]
    dS_dt = -beta * S * I
    dI_dt = beta * S * I - gamma * I
    
    return [dS_dt, dI_dt]


def run_gillespie_ensemble(N_pop=1000, I0=10, beta=1.2, gamma=0.3, t_max=20, num_sims=50, sensor_noise=0.0):
    """
    Generates stochastic epidemic trajectories using the Gillespie Algorithm.
    
    Simulates discrete Markov jump processes for individual infection and 
    recovery events, capturing the inherent demographic noise.
    
    Args:
        N_pop (int): Total population size.
        I0 (int): Initial infected population count.
        beta (float): True infection rate.
        gamma (float): True recovery rate.
        t_max (int): Duration of simulation.
        num_sims (int): Number of independent stochastic runs to ensemble.
        sensor_noise (float): Standard deviation of Gaussian noise added to simulate measurement error.
        
    Returns:
        t_eval (np.ndarray): Uniform time grid (1000 steps).
        x_ensemble (np.ndarray): Stacked array of [S_mean, I_mean] averaged across runs.
    """
    print(f"Running {num_sims} Gillespie simulations (beta={beta:.2f}, gamma={gamma:.2f})...")
    
    # Define a uniform evaluation grid for downstream derivative approximation
    t_eval = np.linspace(0, t_max, 1000)
    
    S_ensemble = np.zeros((num_sims, len(t_eval)))
    I_ensemble = np.zeros((num_sims, len(t_eval)))

    for sim in range(num_sims):
        S, I = N_pop - I0, I0
        t = 0
        
        times, S_path, I_path = [t], [S / N_pop], [I / N_pop]

        while t < t_max and I > 0:
            rate_inf = beta * (S * I) / N_pop
            rate_rec = gamma * I
            total_rate = rate_inf + rate_rec

            if total_rate == 0:
                break

            # Sample time to next event from an exponential distribution
            dt = np.random.exponential(1 / total_rate)
            t += dt

            # Determine the event type based on relative transition probabilities
            if np.random.rand() < (rate_inf / total_rate):
                S -= 1
                I += 1
            else:
                I -= 1

            times.append(t)
            S_path.append(S / N_pop)
            I_path.append(I / N_pop)

        # Interpolate the non-uniform jump times onto the uniform grid
        S_ensemble[sim, :] = np.interp(t_eval, times, S_path)
        I_ensemble[sim, :] = np.interp(t_eval, times, I_path)

    # Compute the expected trajectory across the ensemble
    S_mean = np.mean(S_ensemble, axis=0)
    I_mean = np.mean(I_ensemble, axis=0)

    # Apply synthetic measurement noise if specified
    if sensor_noise > 0:
        S_mean += np.random.normal(0, sensor_noise, len(t_eval))
        I_mean += np.random.normal(0, sensor_noise, len(t_eval))

    return t_eval, np.vstack((S_mean, I_mean)).T


def get_deterministic_truth(beta=1.2, gamma=0.3, t_max=20):
    """
    Generates the analytical solution using standard ODE integration.
    Assumes an initial condition of 99% susceptible, 1% infected.
    """
    t_eval = np.linspace(0, t_max, 1000)
    solution = solve_ivp(sir_deterministic, [0, t_max], [0.99, 0.01], args=(beta, gamma), t_eval=t_eval)
    return t_eval, solution.y.T
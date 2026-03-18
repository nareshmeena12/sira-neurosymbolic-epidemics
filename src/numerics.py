"""
Numerical Validation Module

This module provides implementations of numerical integration schemes 
to simulate the discovered differential equations. Custom implementations 
allow for fine-grained stability testing and comparison across various 
explicit and multi-step methods (e.g., Euler, RK4, Predictor-Corrector).
"""

import numpy as np

# ============================================================
# SINGLE-STEP METHODS
# ============================================================

def euler_step(func, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    Performs a single Forward Euler step.
    
    Explicit first-order method. 
    Truncation error: O(dt^2) per step, O(dt) globally.
    Note: Provided primarily as a baseline. Stiff systems may cause 
    Euler integration to exhibit numerical instability.
    """
    dy_dt = np.array(func(t, y))
    return y + dt * dy_dt

def rk2_step(func, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    Performs a 2nd Order Runge-Kutta (Midpoint) step.
    
    Truncation error: O(dt^3) per step, O(dt^2) globally.
    """
    k1 = np.array(func(t, y))
    k2 = np.array(func(t + dt / 2.0, y + dt * k1 / 2.0))
    return y + dt * k2

def rk4_step(func, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    Performs a 4th Order Runge-Kutta step.
    
    Truncation error: O(dt^5) per step, O(dt^4) globally.
    """
    k1 = np.array(func(t, y))
    k2 = np.array(func(t + dt / 2.0, y + dt * k1 / 2.0))
    k3 = np.array(func(t + dt / 2.0, y + dt * k2 / 2.0))
    k4 = np.array(func(t + dt, y + dt * k3))
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================
# MULTI-STEP & HYBRID METHODS
# ============================================================

def adams_bashforth_2_step(func, t_curr: float, y_curr: np.ndarray, y_prev: np.ndarray, dt: float) -> np.ndarray:
    """
    Performs a 2-step Adams-Bashforth (AB2) step.
    
    Explicit multi-step method utilizing the history of the previous 
    time step to predict the subsequent state.
    Truncation error: O(dt^2) globally.
    """
    f_curr = np.array(func(t_curr, y_curr))
    f_prev = np.array(func(t_curr - dt, y_prev))
    
    return y_curr + dt * (1.5 * f_curr - 0.5 * f_prev)

def predictor_corrector_step(func, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """
    Performs a Predictor-Corrector step (Heun's Method).
    
    Uses Forward Euler as the predictor and the Trapezoidal Rule 
    as the corrector to refine the approximation.
    """
    # Predictor
    f_curr = np.array(func(t, y))
    y_predict = y + dt * f_curr
    
    # Corrector
    f_predict = np.array(func(t + dt, y_predict))
    y_correct = y + (dt / 2.0) * (f_curr + f_predict)
    
    return y_correct

# ============================================================
# MAIN SIMULATION ENGINE
# ============================================================

def simulate_discovered_physics(beta: float, gamma: float, t_eval: np.ndarray, y0: list, method: str = 'rk4') -> np.ndarray:
    """
    Takes the discovered mathematical parameters and integrates the 
    SIR vector field forward in time using the specified numerical method.
    
    Args:
        beta: Estimated infection rate.
        gamma: Estimated recovery rate.
        t_eval: Time grid for simulation.
        y0: Initial conditions [S0, I0].
        method: Integration method ('euler', 'rk2', 'rk4', 'ab2', 'pred_corr').
        
    Returns:
        y_sim: 2D numpy array of the simulated [S, I] trajectories.
    """
    dt = t_eval[1] - t_eval[0]
    y_sim = np.zeros((len(t_eval), len(y0)))
    y_sim[0] = y0
    
    def sir_vector_field(t, y):
        S, I = y[0], y[1]
        dS_dt = -beta * S * I
        dI_dt = beta * S * I - gamma * I
        return [dS_dt, dI_dt]

    for i in range(1, len(t_eval)):
        t_prev = t_eval[i-1]
        y_prev = y_sim[i-1]
        
        if method == 'rk4':
            y_sim[i] = rk4_step(sir_vector_field, t_prev, y_prev, dt)
        elif method == 'rk2':
            y_sim[i] = rk2_step(sir_vector_field, t_prev, y_prev, dt)
        elif method == 'euler':
            y_sim[i] = euler_step(sir_vector_field, t_prev, y_prev, dt)
        elif method == 'pred_corr':
            y_sim[i] = predictor_corrector_step(sir_vector_field, t_prev, y_prev, dt)
        elif method == 'ab2':
            if i == 1:
                y_sim[i] = rk2_step(sir_vector_field, t_prev, y_prev, dt)
            else:
                y_sim[i] = adams_bashforth_2_step(sir_vector_field, t_prev, y_prev, y_sim[i-2], dt)
        else:
            raise ValueError(f"Unknown numerical method: {method}")
            
    return y_sim

# ============================================================
# SANITY CHECKS & METRICS
# ============================================================

def check_physical_conservation(y_sim: np.ndarray) -> float:
    """
    Verifies population conservation in a closed SIR system.
    
    Checks if S(t) + I(t) > 1.0, which indicates a physical violation 
    often associated with numerical instability during integration.
    """
    total_active_population = y_sim[:, 0] + y_sim[:, 1]
    max_violation = np.max(total_active_population) - 1.0
    
    if max_violation > 1e-5:
        print(f"Warning: Population exceeded 1.0 by {max_violation:.5f}. Numerical method may be unstable.")
    
    return max_violation

def calculate_recovery_metrics(y_true: np.ndarray, y_sim: np.ndarray):
    """
    Calculates the Mean Squared Error (MSE) and Maximum Absolute Error 
    between the true trajectories and the recovered trajectories.
    """
    mse = np.mean((y_true - y_sim)**2)
    max_err = np.max(np.abs(y_true - y_sim))
    return mse, max_err
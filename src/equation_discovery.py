"""
Equation Discovery Module

Uses PySINDy (Sparse Identification of Nonlinear Dynamics) to extract the 
underlying differential equations from the data. 

Implements two approaches:
- Neural-Symbolic: Uses torch.autograd on a trained neural manifold.
- Weak Formulation: Uses integral windows over noisy data directly.
"""

import torch
import numpy as np
import pysindy as ps
import warnings

# Suppress minor PySINDy deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)


def discover_via_autograd(ml_model: torch.nn.Module, t_data: np.ndarray, print_equations: bool = True):
    """
    Extracts analytical derivatives from the continuous neural manifold using 
    PyTorch Autograd, then performs sparse regression via PySINDy.
    
    Args:
        ml_model: Trained PyTorch manifold model.
        t_data: 1D numpy array of time steps.
        print_equations: Toggle for console output.
        
    Returns:
        beta_est (float): Estimated infection rate.
        gamma_est (float): Estimated recovery rate.
        preds (np.ndarray): The smoothed [S, I] trajectory.
    """
    if print_equations:
        print("Extracting analytical derivatives via torch.autograd...")

    t_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1).requires_grad_(True)
    preds = ml_model(t_tensor)
    S_pred, I_pred = preds[:, 0], preds[:, 1]

    # Compute exact analytical derivatives
    dS_dt = torch.autograd.grad(S_pred, t_tensor, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    dI_dt = torch.autograd.grad(I_pred, t_tensor, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]

    # Note: Only passing S and I to avoid collinearity issues (since S + I + R = 1)
    X_sindy = preds[:, :2].detach().numpy()
    X_dot_sindy = torch.cat([dS_dt, dI_dt], dim=1).detach().numpy()

    if print_equations:
        print("Running symbolic discovery via PySINDy...")

    library = ps.PolynomialLibrary(degree=2, include_bias=False)

    # Threshold of 0.15 is used to filter out minor autograd artifacts
    optimizer = ps.STLSQ(threshold=0.15, alpha=0.05, normalize_columns=True)

    sindy_model = ps.SINDy(feature_library=library, optimizer=optimizer)
    sindy_model.fit(X_sindy, t=t_data, x_dot=X_dot_sindy)

    if print_equations:
        print("\nDiscovered Deterministic Equations (x0=S, x1=I):")
        sindy_model.print()

    coefs = sindy_model.coefficients()
    
    # Library order for degree 2 without bias: ['x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']
    # S*I term is at index 3. I term is at index 1.
    try:
        beta_est = abs(coefs[0, 3])  
        gamma_est = abs(coefs[1, 1]) 
    except IndexError:
        if print_equations:
            print("Warning: Threshold pruned necessary physical terms. Returning zeros.")
        beta_est, gamma_est = 0.0, 0.0

    return beta_est, gamma_est, X_sindy


def discover_via_weak_formulation(t_data: np.ndarray, x_noisy: np.ndarray, print_equations: bool = True):
    """
    Bypasses the neural network to handle noise using the Weak Formulation,
    which relies on integration over local spatial-temporal windows.
    """
    if print_equations:
        print("Running Weak SINDy on noisy data...")

    x_input = x_noisy[:, :2]

    weak_lib = ps.WeakPDELibrary(
        library_functions=[lambda x: x, lambda x, y: x * y, lambda x: x**2],
        function_names=[lambda x: x, lambda x, y: x + ' ' + y, lambda x: x + '^2'],
        spatiotemporal_grid=t_data,
        is_uniform=True,
        K=100
    )

    optimizer = ps.STLSQ(threshold=0.25, alpha=0.05, normalize_columns=True)

    model = ps.SINDy(feature_library=weak_lib, optimizer=optimizer)
    model.fit(x_input, t=t_data)

    if print_equations:
        print("\nWeak SINDy Equations (x0=S, x1=I):")
        model.print()

    coefs = model.coefficients()
    
    try:
        # Expected library order: x0, x1, x0*x1, x0^2, x1^2
        beta_est = abs(coefs[0, 2])
        gamma_est = abs(coefs[1, 1])
    except Exception:
        beta_est, gamma_est = 0.0, 0.0

    return beta_est, gamma_est, model
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
    preds    = ml_model(t_tensor)
    S_pred, I_pred = preds[:, 0], preds[:, 1]

    # compute exact analytical derivatives for both compartments
    dS_dt = torch.autograd.grad(S_pred, t_tensor, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    dI_dt = torch.autograd.grad(I_pred, t_tensor, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]

    # only pass S and I to SINDy — passing R too causes collinearity since S+I+R=1
    X_sindy     = preds[:, :2].detach().numpy()
    X_dot_sindy = torch.cat([dS_dt, dI_dt], dim=1).detach().numpy()

    if print_equations:
        print("Running symbolic discovery via PySINDy...")

    library   = ps.PolynomialLibrary(degree=2, include_bias=False)
    optimizer = ps.STLSQ(threshold=0.2, alpha=0.05, normalize_columns=True)

    sindy_model = ps.SINDy(feature_library=library, optimizer=optimizer)
    sindy_model.fit(X_sindy, t=t_data, x_dot=X_dot_sindy)

    if print_equations:
        print("\nDiscovered Deterministic Equations (x0=S, x1=I):")
        sindy_model.print()

    coefs      = sindy_model.coefficients()
    feat_names = sindy_model.get_feature_names()

    # look up term positions by name rather than hardcoding indices —
    # this way the code stays correct even if the library order ever shifts
    try:
        si_idx = next(
            j for j, name in enumerate(feat_names)
            if set(name.lower().split()) == {"x0", "x1"}
        )
        i_idx = next(
            j for j, name in enumerate(feat_names)
            if name.strip().lower() == "x1"
        )
        beta_est  = abs(coefs[0, si_idx])
        gamma_est = abs(coefs[1, i_idx])
    except StopIteration:
        if print_equations:
            print("Warning: Threshold pruned necessary physical terms. Returning zeros.")
        beta_est, gamma_est = 0.0, 0.0

    return beta_est, gamma_est, X_sindy


def discover_via_weak_formulation(t_data: np.ndarray, x_noisy: np.ndarray, print_equations: bool = True):
    if print_equations:
        print("Running Weak SINDy on noisy data...")

    x_input = x_noisy[:, :2]

    # function_library replaces the old library_functions parameter
    # pass a PolynomialLibrary object instead of lambda list
    weak_lib = ps.WeakPDELibrary(
        function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
        spatiotemporal_grid=t_data,
        K=100,
    )

    optimizer = ps.STLSQ(threshold=0.25, alpha=0.05, normalize_columns=True)
    model     = ps.SINDy(feature_library=weak_lib, optimizer=optimizer)
    model.fit(x_input, t=t_data)

    if print_equations:
        print("\nWeak SINDy Equations (x0=S, x1=I):")
        model.print()

    coefs      = model.coefficients()
    feat_names = model.get_feature_names()

    try:
        si_idx    = next(j for j, n in enumerate(feat_names)
                         if set(n.lower().split()) == {"x0", "x1"})
        i_idx     = next(j for j, n in enumerate(feat_names)
                         if n.strip().lower() == "x1")
        beta_est  = abs(coefs[0, si_idx])
        gamma_est = abs(coefs[1, i_idx])
    except StopIteration:
        beta_est, gamma_est = 0.0, 0.0

    return beta_est, gamma_est, model
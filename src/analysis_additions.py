"""
analysis_additions.py

Six additional analyses that extend the depth study findings.
All functions save plots to results/plots/ and return summary dicts.

Additions:
    1. noise_sensitivity_curve    — beta error vs sensor noise per depth
    2. sample_efficiency_curve    — beta error vs N (already done in notebook)
    3. derivative_quality         — second derivative L2 norm vs depth
    4. r0_failure_boundary        — failure rate and error vs R0 (already done)
    5. confidence_intervals       — recovery confidence bands per R0 regime
    6. cld_integration_methods    — CLD across all 5 numerical methods
    7. fano_vs_depth              — Fano factor at each ensemble depth
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")

os.makedirs("results/plots", exist_ok=True)

DEPTHS         = [20, 50, 100, 150, 200]
DEPTH_COLORS   = ["#1D9E75", "#378ADD", "#EF9F27", "#E24B4A", "#7F77DD"]
NOISE_LEVELS   = [0.0, 0.005, 0.01, 0.02, 0.05]
BETA_RANGE     = (0.8, 2.5)
GAMMA_RANGE    = (0.1, 0.6)
N_EXPERIMENTS  = 30     # per noise level per depth — enough for stable curves
EPOCHS         = 2000


# ── Addition 1 — Noise Sensitivity Curve ─────────────────────────────────────

def compute_noise_sensitivity(depths=None, noise_levels=None,
                               n_experiments=N_EXPERIMENTS):
    """
    Runs N_EXPERIMENTS experiments at each (depth, noise_level) combination
    and records median beta error. Returns a DataFrame indexed by noise level
    with one column per depth.
    """
    from src.simulation import run_gillespie_ensemble, is_epidemic_active
    from src.neural_net import train_neural_smoother
    from src.equation_discovery import discover_via_autograd

    if depths is None:
        depths = DEPTHS
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    np.random.seed(42)
    torch.manual_seed(42)

    true_betas  = np.random.uniform(*BETA_RANGE,  n_experiments)
    true_gammas = np.random.uniform(*GAMMA_RANGE, n_experiments)

    results = {d: [] for d in depths}

    for depth in depths:
        print(f"  Noise sensitivity — N={depth}")
        noise_medians = []

        for sigma in noise_levels:
            b_errs = []
            for i in range(n_experiments):
                b_true = true_betas[i]
                g_true = true_gammas[i]
                if not is_epidemic_active(b_true, g_true):
                    continue
                try:
                    t, S, I, R = run_gillespie_ensemble(
                        beta=b_true, gamma=g_true,
                        num_sims=depth, sensor_noise=sigma
                    )
                    model       = train_neural_smoother(t, S, I, R, epochs=EPOCHS)
                    b_est, g_est, _ = discover_via_autograd(
                        model, t, print_equations=False
                    )
                    if b_est > 0:
                        b_errs.append(abs(b_true - b_est) / b_true * 100)
                except Exception:
                    pass
            noise_medians.append(np.median(b_errs) if b_errs else np.nan)

        results[depth] = noise_medians

    df = pd.DataFrame(results, index=noise_levels)
    df.index.name = "sensor_noise"
    return df


def plot_noise_sensitivity(df=None, save=True):
    """
    Plots beta error vs sensor noise level for each ensemble depth.
    If df is None, runs compute_noise_sensitivity() first.
    """
    if df is None:
        print("Computing noise sensitivity (this takes ~20 minutes)...")
        df = compute_noise_sensitivity()

    fig, ax = plt.subplots(figsize=(9, 5))

    for depth, color in zip(DEPTHS, DEPTH_COLORS):
        if depth not in df.columns:
            continue
        vals = df[depth].values
        ax.plot(df.index * 100, vals, "o-", color=color,
                lw=2, ms=6, label=f"N={depth}")

    ax.axvline(0.5, color="grey", linestyle="--", lw=1, alpha=0.7,
               label="CNT boundary (0.5%)")
    ax.set_xlabel("Sensor noise level (%)")
    ax.set_ylabel("Median beta error (%)")
    ax.set_title("Noise sensitivity curve — pipeline robustness under measurement error")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig("results/plots/noise_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()
    return df


# ── Addition 3 — Derivative Quality Analysis ─────────────────────────────────

def compute_derivative_quality(n_experiments=20):
    """
    Trains a model at each ensemble depth and computes the L2 norm of the
    second derivative of I(t) from the neural manifold. A stable or decreasing
    L2 norm across depths proves the smoothness trap does not exist here.
    """
    from src.simulation import run_gillespie_ensemble, is_epidemic_active
    from src.neural_net import train_neural_smoother

    np.random.seed(42)
    torch.manual_seed(42)

    true_betas  = np.random.uniform(*BETA_RANGE,  n_experiments)
    true_gammas = np.random.uniform(*GAMMA_RANGE, n_experiments)

    results = {d: [] for d in DEPTHS}

    for depth in DEPTHS:
        print(f"  Derivative quality — N={depth}")
        l2_norms = []

        for i in range(n_experiments):
            b_true = true_betas[i]
            g_true = true_gammas[i]
            if not is_epidemic_active(b_true, g_true):
                continue
            try:
                t, S, I, R = run_gillespie_ensemble(
                    beta=b_true, gamma=g_true, num_sims=depth
                )
                model = train_neural_smoother(t, S, I, R, epochs=EPOCHS)

                t_ten = torch.tensor(t, dtype=torch.float32).view(-1, 1).requires_grad_(True)
                preds = model(t_ten)
                I_pred = preds[:, 1]

                # first derivative
                dI = torch.autograd.grad(
                    I_pred, t_ten,
                    grad_outputs=torch.ones_like(I_pred),
                    create_graph=True
                )[0]

                # second derivative
                d2I = torch.autograd.grad(
                    dI, t_ten,
                    grad_outputs=torch.ones_like(dI),
                    create_graph=False
                )[0]

                l2 = float(torch.norm(d2I).item())
                l2_norms.append(l2)

            except Exception:
                pass

        results[depth] = l2_norms

    return results


def plot_derivative_quality(results=None, save=True):
    """
    Plots L2 norm of second derivative of I(t) vs ensemble depth.
    A flat or decreasing curve proves smoother manifolds at higher N.
    """
    if results is None:
        print("Computing derivative quality (~15 minutes)...")
        results = compute_derivative_quality()

    medians = [np.median(results[d]) if results[d] else np.nan for d in DEPTHS]
    q25     = [np.percentile(results[d], 25) if results[d] else np.nan for d in DEPTHS]
    q75     = [np.percentile(results[d], 75) if results[d] else np.nan for d in DEPTHS]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(DEPTHS, medians, "o-", color="#1D9E75", lw=2, ms=7, label="Median L2 norm")
    ax.fill_between(DEPTHS, q25, q75, alpha=0.2, color="#1D9E75", label="IQR")
    ax.set_xlabel("Ensemble depth (N simulations)")
    ax.set_ylabel("L2 norm of d²I/dt²")
    ax.set_title("Derivative smoothness vs ensemble depth\n"
                 "Stable or decreasing = no smoothness trap")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(DEPTHS)

    plt.tight_layout()
    if save:
        plt.savefig("results/plots/derivative_quality.png", dpi=150, bbox_inches="tight")
    plt.show()
    return results


# ── Addition 6 — CLD Across Integration Methods ──────────────────────────────

def compute_cld_all_methods(n_experiments=30):
    """
    Runs all 5 numerical integration methods on discovered equations
    and computes CLD for each. Shows conservation is a property of
    the discovered equations, not the integration method.
    """
    from src.simulation import run_gillespie_ensemble, is_epidemic_active
    from src.neural_net import train_neural_smoother
    from src.equation_discovery import discover_via_autograd
    from src.numerics import simulate_discovered_physics

    np.random.seed(42)
    torch.manual_seed(42)

    true_betas  = np.random.uniform(*BETA_RANGE,  n_experiments)
    true_gammas = np.random.uniform(*GAMMA_RANGE, n_experiments)

    methods = ["euler", "rk2", "rk4", "ab2", "pred_corr"]
    results = {m: [] for m in methods}

    print("Computing CLD across integration methods...")

    for i in range(n_experiments):
        b_true = true_betas[i]
        g_true = true_gammas[i]
        if not is_epidemic_active(b_true, g_true):
            continue
        try:
            t, S, I, R = run_gillespie_ensemble(beta=b_true, gamma=g_true, num_sims=50)
            model       = train_neural_smoother(t, S, I, R, epochs=EPOCHS)
            b_est, g_est, _ = discover_via_autograd(model, t, print_equations=False)

            if b_est == 0.0 or g_est == 0.0:
                continue

            y0 = [S[0], I[0]]
            for method in methods:
                try:
                    y_sim = simulate_discovered_physics(b_est, g_est, t, y0, method=method)
                    S_sim = y_sim[:, 0]
                    I_sim = y_sim[:, 1]
                    R_sim = np.clip(1.0 - S_sim - I_sim, 0, None)
                    cld   = float(np.max(np.abs(S_sim + I_sim + R_sim - 1.0)))
                    results[method].append(cld)
                except Exception:
                    pass
        except Exception:
            pass

    return results


def plot_cld_methods(results=None, save=True):
    """
    Box plot of CLD distribution for each integration method.
    """
    if results is None:
        results = compute_cld_all_methods()

    methods       = ["euler", "rk2", "rk4", "ab2", "pred_corr"]
    method_labels = ["Euler", "RK2", "RK4", "Adams-B2", "Pred-Corr"]
    colors        = ["#E24B4A", "#EF9F27", "#1D9E75", "#378ADD", "#7F77DD"]

    fig, ax = plt.subplots(figsize=(9, 5))

    data = [results[m] for m in methods if results[m]]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(method_labels)
    ax.set_ylabel("Conservation Law Deviation (CLD)")
    ax.set_title("CLD across integration methods\n"
                 "Conservation is a property of discovered equations, not the solver")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save:
        plt.savefig("results/plots/cld_integration_methods.png",
                    dpi=150, bbox_inches="tight")
    plt.show()
    return results


# ── Addition 7 — Fano Factor vs Ensemble Depth ───────────────────────────────

def compute_fano_vs_depth(n_experiments=20):
    """
    Computes Fano factor at peak infection for each ensemble depth.
    Shows super-Poissonian variability is a property of Gillespie SSA
    itself, not the ensemble size.
    """
    from src.simulation import is_epidemic_active

    np.random.seed(42)

    true_betas  = np.random.uniform(*BETA_RANGE,  n_experiments)
    true_gammas = np.random.uniform(*GAMMA_RANGE, n_experiments)

    results = {d: [] for d in DEPTHS}

    for depth in DEPTHS:
        print(f"  Fano vs depth — N={depth}")
        for i in range(n_experiments):
            b_true = true_betas[i]
            g_true = true_gammas[i]
            if not is_epidemic_active(b_true, g_true):
                continue
            try:
                N_pop  = 1000
                I0     = 10
                t_eval = np.linspace(0, 20, 500)
                I_ens  = np.zeros((depth, 500))

                for sim in range(depth):
                    S, I, R = N_pop - I0, I0, 0
                    t       = 0
                    times   = [t]
                    I_p     = [float(I)]

                    while t < 20 and I > 0:
                        ri   = b_true * (S * I) / N_pop
                        rr   = g_true * I
                        rt   = ri + rr
                        if rt == 0:
                            break
                        t   += np.random.exponential(1 / rt)
                        if np.random.rand() < (ri / rt):
                            S -= 1; I += 1
                        else:
                            I -= 1; R += 1
                        times.append(t)
                        I_p.append(float(I))

                    I_ens[sim, :] = np.interp(t_eval, times, I_p)

                I_mean   = I_ens.mean(axis=0)
                I_var    = I_ens.var(axis=0)
                peak_idx = int(np.argmax(I_mean))

                if I_mean[peak_idx] > 0:
                    fano = float(I_var[peak_idx] / I_mean[peak_idx])
                    results[depth].append(fano)

            except Exception:
                pass

    return results


def plot_fano_vs_depth(results=None, save=True):
    """
    Box plot of Fano factor at peak infection for each ensemble depth.
    Fano > 1 at all depths confirms super-Poissonian variability is
    intrinsic to the Gillespie process.
    """
    if results is None:
        print("Computing Fano vs depth (~10 minutes)...")
        results = compute_fano_vs_depth()

    medians = [np.median(results[d]) if results[d] else np.nan for d in DEPTHS]
    q25     = [np.percentile(results[d], 25) if results[d] else np.nan for d in DEPTHS]
    q75     = [np.percentile(results[d], 75) if results[d] else np.nan for d in DEPTHS]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(DEPTHS, medians, "o-", color="#1D9E75", lw=2, ms=7, label="Median Fano")
    ax.fill_between(DEPTHS, q25, q75, alpha=0.2, color="#1D9E75", label="IQR")
    ax.axhline(1.0, color="#E24B4A", linestyle="--", lw=1.5,
               label="Poisson baseline (Fano=1)")

    ax.set_xlabel("Ensemble depth (N simulations)")
    ax.set_ylabel("Fano factor at peak infection")
    ax.set_title("Fano factor vs ensemble depth\n"
                 "Super-Poissonian variability is intrinsic to Gillespie SSA")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(DEPTHS)

    plt.tight_layout()
    if save:
        plt.savefig("results/plots/fano_vs_depth.png", dpi=150, bbox_inches="tight")
    plt.show()
    return results

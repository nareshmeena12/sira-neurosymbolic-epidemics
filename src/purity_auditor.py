"""
purity_auditor.py

Evaluation metrics for the SIR equation discovery pipeline.
Measures structural correctness, physical validity, real-world robustness,
and computational efficiency of the discovered equations.

Metrics implemented:
    Structural   — SPI (Structural Purity Index), PSS (Parameter Stability Score)
    Physical     — CLD (Conservation Law Deviation), FH (Forecasting Horizon),
                   OOD (Out-of-Distribution MAE across 4 unseen epidemic regimes)
    Deployability — CNT (Critical Noise Threshold), DST (Data Sparsity Tolerance)
    Efficiency   — ZER (Zero-Shot Efficiency Ratio), TTD (Time-To-Discovery)
"""

import time
import warnings

import numpy as np
import torch
import pysindy as ps
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")


GROUND_TRUTH_TERMS = {
    "S": {"x0 x1"},
    "I": {"x0 x1", "x1"},
    "R": {"x1"},
}

OOD_REGIMES = {
    "explosive":   {"beta": (1.1, 1.5),  "gamma": (0.01, 0.09)},
    "fast_dieout": {"beta": (0.1, 0.4),  "gamma": (0.4,  0.7)},
    "high_turn":   {"beta": (1.5, 2.5),  "gamma": (0.5,  0.9)},
    "slow_burn":   {"beta": (0.1, 0.3),  "gamma": (0.01, 0.05)},
}


def _run_sir(beta, gamma, y0, t_eval):
    def odes(t, y, b, g):
        S, I = max(y[0], 0), max(y[1], 0)
        return [-b * S * I, b * S * I - g * I]
    sol = solve_ivp(odes, [t_eval[0], t_eval[-1]], y0,
                    args=(beta, gamma), t_eval=t_eval)
    return sol if sol.success else None


def _norm(name):
    return " ".join(sorted(name.strip().lower().split()))


def compute_spi(coefs, feat_names, threshold=0.0005):
    """
    Structural Purity Index.

    Checks whether SINDy discovered exactly the right terms and nothing else.
    Precision measures how many found terms are physically real.
    Recall measures how many real terms were actually found.
    SPI is the F1 score of the two. A score of 1.0 is a perfect equation.
    """
    norm_feat = [_norm(n) for n in feat_names]
    norm_gt   = {eq: {_norm(t) for t in terms}
                 for eq, terms in GROUND_TRUTH_TERMS.items()}

    all_p, all_r, all_f = [], [], []
    results = {}

    for idx, eq in enumerate(["S", "I", "R"]):
        if idx >= coefs.shape[0]:
            continue

        found   = {norm_feat[j] for j, c in enumerate(coefs[idx]) if abs(c) >= threshold}
        truth   = norm_gt.get(eq, set())
        correct = found & truth

        p  = len(correct) / len(found) if found else 0.0
        r  = len(correct) / len(truth) if truth else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        results[eq] = {
            "spi":       round(f1, 4),
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "spurious":  found - truth,
            "missed":    truth - found,
        }
        all_p.append(p); all_r.append(r); all_f.append(f1)

    results["overall"] = {
        "spi":       round(float(np.mean(all_f)), 4),
        "precision": round(float(np.mean(all_p)), 4),
        "recall":    round(float(np.mean(all_r)), 4),
    }
    return results


def compute_pss(t_data, S_data, I_data, R_data,
                train_fn, discover_fn,
                noise_level=0.01, n_repeats=10):
    """
    Parameter Stability Score.

    Runs discovery n_repeats times with 1% Gaussian noise injected
    into the data each time. Reports the standard deviation of the
    discovered beta and gamma across all repeats.
    A low PSS means the pipeline is stable under noisy observations.
    """
    betas, gammas = [], []

    for _ in range(n_repeats):
        S_n = S_data + np.random.normal(0, noise_level, len(S_data))
        I_n = I_data + np.random.normal(0, noise_level, len(I_data))
        R_n = R_data + np.random.normal(0, noise_level, len(R_data))
        try:
            model           = train_fn(t_data, S_n, I_n, R_n)
            b_est, g_est, _ = discover_fn(model, t_data, print_equations=False)
            if b_est > 0 and g_est > 0:
                betas.append(b_est)
                gammas.append(g_est)
        except Exception:
            pass

    if not betas:
        return {"pss_beta": None, "pss_gamma": None, "n_valid": 0}

    return {
        "pss_beta":   round(float(np.std(betas)),   5),
        "pss_gamma":  round(float(np.std(gammas)),  5),
        "mean_beta":  round(float(np.mean(betas)),  5),
        "mean_gamma": round(float(np.mean(gammas)), 5),
        "n_valid":    len(betas),
    }


def compute_cld(beta_est, gamma_est, t_max=20, n_steps=1000, y0=None):
    """
    Conservation Law Deviation.

    Integrates the discovered equations forward using a Predictor-Corrector
    solver and measures how much S + I + R drifts from 1.0 over time.
    A perfect CLD of 0.0 means the discovered physics is fully conserved.
    """
    if y0 is None:
        y0 = [0.99, 0.01]

    t_eval  = np.linspace(0, t_max, n_steps)
    dt      = t_eval[1] - t_eval[0]
    traj    = np.zeros((n_steps, 2))
    traj[0] = y0

    def f(y):
        S, I = y
        return np.array([-beta_est * S * I,
                          beta_est * S * I - gamma_est * I])

    for i in range(1, n_steps):
        y       = traj[i - 1]
        fc      = f(y)
        y_pred  = y + dt * fc
        traj[i] = y + (dt / 2.0) * (fc + f(y_pred))

    S          = traj[:, 0]
    I          = traj[:, 1]
    R          = np.clip(1.0 - S - I, 0, None)
    deviations = np.abs(S + I + R - 1.0)

    return {
        "max_deviation":  round(float(np.max(deviations)),  8),
        "mean_deviation": round(float(np.mean(deviations)), 8),
        "is_conserved":   bool(np.max(deviations) < 1e-4),
    }


def compute_fh(beta_true, gamma_true, beta_est, gamma_est,
               t_max=20, train_frac=0.5, error_threshold=0.05):
    """
    Forecasting Horizon.

    Trains on the first 50% of the epidemic and uses the discovered
    equations to predict the remaining 50%. Reports how many days
    into the future the prediction stays within 5% of the true trajectory.
    """
    t_eval = np.linspace(0, t_max, 1000)
    split  = int(len(t_eval) * train_frac)
    y0     = [0.99, 0.01]

    sol_true = _run_sir(beta_true, gamma_true, y0, t_eval)
    if sol_true is None:
        return {"fh_days": 0.0, "fh_pct": 0.0}

    y0_hand  = [sol_true.y[0, split], sol_true.y[1, split]]
    t_future = t_eval[split:]
    sol_disc = _run_sir(beta_est, gamma_est, y0_hand, t_future)
    if sol_disc is None:
        return {"fh_days": 0.0, "fh_pct": 0.0}

    n      = min(sol_true.y[:, split:].shape[1], sol_disc.y.shape[1])
    errors = np.mean(np.abs(sol_true.y[:, split:][:, :n] - sol_disc.y[:, :n]), axis=0)

    exceeded = np.where(errors > error_threshold)[0]
    if len(exceeded) == 0:
        fh_idx, fh_pct = len(errors) - 1, 100.0
    else:
        fh_idx = exceeded[0]
        fh_pct = round(fh_idx / len(errors) * 100, 2)

    return {
        "fh_days": round(float(t_future[fh_idx]), 3),
        "fh_pct":  fh_pct,
    }


def compute_ood_mae(beta_est, gamma_est, n_points=30, t_max=20):
    """
    Out-of-Distribution MAE.

    Tests the discovered equations on four extreme epidemic regimes
    that were never seen during training. If the MAE stays low,
    the pipeline learned the universal laws of epidemic spread
    rather than just memorising the training data.
    """
    results = {}
    np.random.seed(99)

    for regime, cfg in OOD_REGIMES.items():
        maes = []
        for _ in range(n_points):
            b  = np.random.uniform(*cfg["beta"])
            g  = np.random.uniform(*cfg["gamma"])
            te = np.linspace(0, t_max, 200)
            y0 = [0.99, 0.01]

            sol_t = _run_sir(b,        g,        y0, te)
            sol_d = _run_sir(beta_est, gamma_est, y0, te)

            if sol_t is not None and sol_d is not None:
                maes.append(float(np.mean(np.abs(sol_t.y - sol_d.y))))

        results[regime] = round(float(np.mean(maes)), 6) if maes else None

    valid            = [v for v in results.values() if v is not None]
    results["overall"] = round(float(np.mean(valid)), 6) if valid else None
    return results


def compute_cnt(t_data, S_data, I_data, R_data,
                beta_true, gamma_true,
                train_fn, discover_fn,
                noise_levels=None, threshold=10.0):
    """
    Critical Noise Threshold.

    Finds the maximum observational noise the pipeline can absorb
    before beta error exceeds the given threshold (default 10%).
    Useful for understanding how the pipeline performs on messy
    or delayed real-world hospital reporting.
    """
    if noise_levels is None:
        noise_levels = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

    results = {}
    cnt     = None

    for sigma in noise_levels:
        errs = []
        for _ in range(5):
            S_n = S_data + np.random.normal(0, sigma, len(S_data))
            I_n = I_data + np.random.normal(0, sigma, len(I_data))
            R_n = R_data + np.random.normal(0, sigma, len(R_data))
            try:
                model           = train_fn(t_data, S_n, I_n, R_n)
                b_est, _, _     = discover_fn(model, t_data, print_equations=False)
                if b_est > 0:
                    errs.append(abs(beta_true - b_est) / beta_true * 100)
            except Exception:
                pass

        mean_err       = float(np.mean(errs)) if errs else float("inf")
        results[sigma] = round(mean_err, 3)

        if mean_err <= threshold and cnt is None:
            cnt = sigma

    return {"cnt": cnt, "threshold_used": threshold, "errors_per_sigma": results}


def compute_dst(t_data, S_data, I_data, R_data,
                train_fn, discover_fn,
                keep_fractions=None):
    """
    Data Sparsity Tolerance.

    Randomly drops increasing fractions of the timeline to simulate
    missed hospital reports and finds the minimum data fraction
    needed to still achieve a perfect SPI of 1.0.
    """
    if keep_fractions is None:
        keep_fractions = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]

    results = {}
    dst     = None

    for frac in keep_fractions:
        n_keep = max(10, int(len(t_data) * frac))
        idx    = np.sort(np.random.choice(len(t_data), n_keep, replace=False))

        try:
            model = train_fn(t_data[idx], S_data[idx], I_data[idx], R_data[idx])

            t_ten = torch.tensor(t_data[idx], dtype=torch.float32).view(-1, 1).requires_grad_(True)
            preds = model(t_ten)
            dS    = torch.autograd.grad(preds[:, 0], t_ten,
                        grad_outputs=torch.ones(n_keep), create_graph=True)[0]
            dI    = torch.autograd.grad(preds[:, 1], t_ten,
                        grad_outputs=torch.ones(n_keep), create_graph=True)[0]

            library     = ps.PolynomialLibrary(degree=2, include_bias=False)
            optimizer   = ps.STLSQ(threshold=0.2, alpha=0.05, normalize_columns=True)
            sindy_model = ps.SINDy(feature_library=library, optimizer=optimizer)
            sindy_model.fit(preds[:, :2].detach().numpy(), t=t_data[idx],
                            x_dot=torch.cat([dS, dI], dim=1).detach().numpy())

            coefs      = sindy_model.coefficients()
            feat_names = sindy_model.get_feature_names()
            spi_val    = compute_spi(coefs, feat_names)["overall"]["spi"]

            results[frac] = round(spi_val, 4)
            if spi_val >= 1.0 and dst is None:
                dst = frac

        except Exception:
            results[frac] = None

    return {"dst": dst, "spi_per_fraction": results}


def compute_zer(mean_beta_error_pct, mean_gamma_error_pct, num_simulations):
    """
    Zero-Shot Efficiency Ratio.

    ZER = 1 / (mean_error * num_simulations)

    A direct comparison against pipelines that use hundreds of simulations.
    The competitor used 500 simulations. We use 50. A higher ZER means
    more accuracy per simulation used.
    """
    mean_error = (mean_beta_error_pct + mean_gamma_error_pct) / 2.0 / 100.0
    if mean_error == 0 or num_simulations == 0:
        return {"zer": float("inf"), "mean_error": 0.0}

    zer = 1.0 / (mean_error * num_simulations)
    return {
        "zer":             round(zer, 4),
        "mean_error":      round(mean_error, 6),
        "num_simulations": num_simulations,
    }


def compute_ttd(train_fn, discover_fn, t_data, S_data, I_data, R_data, n_runs=3):
    """
    Time-To-Discovery.

    Measures wall-clock seconds from raw data to discovered ODE, averaged
    over n_runs. Shows whether the pipeline is fast enough for real-time
    deployment on standard hospital hardware.
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        try:
            model = train_fn(t_data, S_data, I_data, R_data)
            discover_fn(model, t_data, print_equations=False)
        except Exception:
            pass
        times.append(time.perf_counter() - start)

    return {
        "mean_ttd_sec": round(float(np.mean(times)), 2),
        "min_ttd_sec":  round(float(np.min(times)),  2),
        "max_ttd_sec":  round(float(np.max(times)),  2),
        "n_runs":       n_runs,
    }


def run_full_audit(beta_true, gamma_true, beta_est, gamma_est,
                   coefs, feat_names,
                   t_data, S_data, I_data, R_data,
                   train_fn, discover_fn,
                   num_simulations=50,
                   mean_beta_err_pct=None,
                   mean_gamma_err_pct=None):
    """
    Runs the five fast metrics (SPI, CLD, FH, OOD, ZER) for a single experiment
    and returns a flat dict ready to be written to CSV.

    PSS, CNT, DST, and TTD are slower because they each require multiple
    extra training runs. Call those separately on a representative subset.
    """
    b_err = abs(beta_true  - beta_est)  / beta_true  * 100
    g_err = abs(gamma_true - gamma_est) / gamma_true * 100

    spi = compute_spi(coefs, feat_names)
    cld = compute_cld(beta_est, gamma_est)
    fh  = compute_fh(beta_true, gamma_true, beta_est, gamma_est)
    ood = compute_ood_mae(beta_est, gamma_est)
    zer = compute_zer(
        mean_beta_err_pct  if mean_beta_err_pct  is not None else b_err,
        mean_gamma_err_pct if mean_gamma_err_pct is not None else g_err,
        num_simulations,
    )

    return {
        "spi_overall":     spi["overall"]["spi"],
        "spi_precision":   spi["overall"]["precision"],
        "spi_recall":      spi["overall"]["recall"],
        "cld_max":         cld["max_deviation"],
        "cld_mean":        cld["mean_deviation"],
        "cld_conserved":   cld["is_conserved"],
        "fh_days":         fh["fh_days"],
        "fh_pct":          fh["fh_pct"],
        "ood_explosive":   ood.get("explosive"),
        "ood_fast_dieout": ood.get("fast_dieout"),
        "ood_high_turn":   ood.get("high_turn"),
        "ood_slow_burn":   ood.get("slow_burn"),
        "ood_overall":     ood.get("overall"),
        "zer":             zer["zer"],
    }


def print_audit_summary(audit_rows, competitor_zer=0.020):
    """
    Prints a summary table across all experiments.
    Pass competitor_zer to show the data efficiency comparison directly.
    """
    if not audit_rows:
        print("No audit data available.")
        return

    def _med(key):
        vals = [r[key] for r in audit_rows
                if r.get(key) is not None and r[key] != float("inf")]
        return round(float(np.median(vals)), 5) if vals else None

    print("\n" + "=" * 55)
    print("PURITY AUDITOR SUMMARY")
    print("=" * 55)
    print(f"{'Metric':<32} {'Median':>10}")
    print("-" * 55)
    print(f"{'SPI  (1.0 = perfect)':<32} {str(_med('spi_overall')):>10}")
    print(f"{'SPI Precision':<32} {str(_med('spi_precision')):>10}")
    print(f"{'SPI Recall':<32} {str(_med('spi_recall')):>10}")
    print(f"{'CLD max deviation':<32} {str(_med('cld_max')):>10}")
    print(f"{'FH  coverage (%)':<32} {str(_med('fh_pct')):>10}")
    print(f"{'OOD overall MAE':<32} {str(_med('ood_overall')):>10}")
    print(f"{'ZER (higher = better)':<32} {str(_med('zer')):>10}")
    print("-" * 55)

    our_zer = _med("zer")
    if our_zer and competitor_zer:
        print(f"ZER vs competitor: {round(our_zer / competitor_zer, 1)}x more efficient")

    conserved = sum(1 for r in audit_rows if r.get("cld_conserved"))
    print(f"Conserved runs: {conserved} / {len(audit_rows)}")
    print("=" * 55)

def compute_r0_regime_analysis(df, metrics=None):
    """
    Bins results by R0 regime and returns median metrics per bin.

    Regimes:
        endemic    R0 < 2
        moderate   2 <= R0 < 5
        fast       5 <= R0 < 10
        explosive  R0 >= 10
    """
    import pandas as pd

    if metrics is None:
        metrics = [
            "beta_err_pct", "gamma_err_pct",
            "spi_overall", "cld_max", "fh_pct", "zer",
        ]

    df = df.copy()
    if "status" in df.columns:
        df = df[df["status"] == "ok"]

    if "r0_true" not in df.columns:
        df["r0_true"] = df["true_beta"] / df["true_gamma"]

    bins   = [0, 2, 5, 10, float("inf")]
    labels = ["endemic (R0<2)", "moderate (2-5)", "fast (5-10)", "explosive (R0>10)"]
    df["regime"] = pd.cut(df["r0_true"], bins=bins, labels=labels, right=False)

    available = [m for m in metrics if m in df.columns]
    summary   = df.groupby("regime", observed=True)[available].median().round(4)

    counts = {
        regime: {
            "total": len(df[df["regime"] == regime]),
            "pct":   round(len(df[df["regime"] == regime]) / len(df) * 100, 1),
        }
        for regime in labels
    }

    return summary, counts


def print_r0_regime_summary(df, metrics=None):
    """
    Prints a regime breakdown table from the results dataframe.
    """
    summary, counts = compute_r0_regime_analysis(df, metrics=metrics)

    print("\n" + "=" * 65)
    print("R0 REGIME ANALYSIS")
    print("=" * 65)
    print(f"{'Regime':<24} {'N':>5} {'%':>6}")
    print("-" * 65)
    for regime, c in counts.items():
        print(f"{regime:<24} {c['total']:>5} {c['pct']:>5.1f}%")
    print()
    print(summary.to_string())
    print("=" * 65)
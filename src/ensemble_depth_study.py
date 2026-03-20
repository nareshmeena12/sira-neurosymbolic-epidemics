"""
ensemble_depth_study.py

Runs 500 experiments across 5 ensemble depths (N=20,50,100,150,200)
measuring all purity metrics at each depth.

Two entry points:
    run_depth_study()   — fast metrics on all 2500 runs (~2 hours on 44 cores)
    run_slow_metrics()  — PSS, CNT, DST, TTD on 50 representative experiments
                          reads the CSV produced by run_depth_study()

Scientific questions:
    Does equation purity degrade as ensemble size grows?
    Does conservation hold regardless of ensemble depth?
    What is the optimal N for accuracy vs efficiency?
    Where does the pipeline struggle across R0 regimes?

Results saved incrementally — safe to resume after disconnects
using BATCH_START.

"""

import os
import csv
import time
import warnings

import numpy as np
import pandas as pd
import torch
import pysindy as ps
from tqdm.notebook import tqdm

from src.simulation import run_gillespie_ensemble, is_epidemic_active
from src.neural_net import train_neural_smoother
from src.equation_discovery import discover_via_autograd
from src.purity_auditor import (
    compute_spi, compute_cld, compute_fh,
    compute_ood_mae, compute_zer,
    compute_pss, compute_cnt, compute_dst, compute_ttd,
)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="pysindy")


# ── configuration ─────────────────────────────────────────────────────────────

DEPTHS        = [20, 50, 100, 150, 200]
N_EXPERIMENTS = 500
EPOCHS        = 2000
BETA_RANGE    = (0.8, 2.5)
GAMMA_RANGE   = (0.1, 0.6)
BATCH_START   = 0
N_SLOW        = 10      # slow metric experiments per depth (10×5 = 50 total)

RESULTS_DIR       = "results/depth_study"
FAST_CSV_PATH     = f"{RESULTS_DIR}/depth_study_results.csv"
SLOW_CSV_PATH     = f"{RESULTS_DIR}/slow_metrics_results.csv"

FAST_COLUMNS = [
    "depth", "experiment_id",
    "true_beta", "true_gamma", "r0_true",
    "est_beta", "est_gamma", "r0_est",
    "beta_err_pct", "gamma_err_pct",
    "spi_overall", "spi_precision", "spi_recall",
    "cld_max", "cld_mean", "cld_conserved",
    "fh_days", "fh_pct",
    "ood_overall", "ood_explosive", "ood_fast_dieout",
    "ood_high_turn", "ood_slow_burn",
    "zer", "train_time_sec", "status",
]

SLOW_COLUMNS = [
    "depth", "experiment_id",
    "true_beta", "true_gamma", "r0_true",
    "beta_err_pct", "gamma_err_pct",
    "pss_beta", "pss_gamma", "pss_n_valid",
    "cnt", "cnt_threshold",
    "dst",
    "ttd_mean_sec", "ttd_min_sec", "ttd_max_sec",
    "status",
]


# ── csv helpers ───────────────────────────────────────────────────────────────

def _setup_csv(path, columns):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(columns)
        print(f"CSV created at {path}")
    else:
        print(f"Appending to existing CSV at {path}")


def _write_row(path, columns, row_dict):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([row_dict.get(col) for col in columns])


# ── single fast experiment ────────────────────────────────────────────────────

def _run_fast(depth, exp_id, b_true, g_true):
    row = {
        "depth": depth, "experiment_id": exp_id,
        "true_beta": round(b_true, 5), "true_gamma": round(g_true, 5),
        "r0_true": round(b_true / g_true, 4), "status": "pending",
    }

    try:
        t_start = time.perf_counter()

        t, S, I, R = run_gillespie_ensemble(
            beta=b_true, gamma=g_true, num_sims=depth
        )
        model       = train_neural_smoother(t, S, I, R, epochs=EPOCHS)
        b_est, g_est, _ = discover_via_autograd(model, t, print_equations=False)

        train_time = round(time.perf_counter() - t_start, 2)

        if b_est == 0.0 or g_est == 0.0:
            row["status"] = "pruned"
            return row

        b_err = abs(b_true - b_est) / b_true * 100
        g_err = abs(g_true - g_est) / g_true * 100

        t_ten = torch.tensor(t, dtype=torch.float32).view(-1, 1).requires_grad_(True)
        preds = model(t_ten)
        dS    = torch.autograd.grad(preds[:, 0], t_ten,
                    grad_outputs=torch.ones(len(t)), create_graph=True)[0]
        dI    = torch.autograd.grad(preds[:, 1], t_ten,
                    grad_outputs=torch.ones(len(t)), create_graph=True)[0]

        library     = ps.PolynomialLibrary(degree=2, include_bias=False)
        sindy_opt   = ps.STLSQ(threshold=0.2, alpha=0.05, normalize_columns=True)
        sindy_model = ps.SINDy(feature_library=library, optimizer=sindy_opt)
        sindy_model.fit(
            preds[:, :2].detach().numpy(), t=t,
            x_dot=torch.cat([dS, dI], dim=1).detach().numpy()
        )
        coefs      = sindy_model.coefficients()
        feat_names = sindy_model.get_feature_names()

        spi = compute_spi(coefs, feat_names)
        cld = compute_cld(b_est, g_est)
        fh  = compute_fh(b_true, g_true, b_est, g_est)
        ood = compute_ood_mae(b_est, g_est)
        zer = compute_zer(b_err, g_err, depth)

        row.update({
            "est_beta": round(b_est, 5), "est_gamma": round(g_est, 5),
            "r0_est": round(b_est / g_est, 4),
            "beta_err_pct": round(b_err, 4), "gamma_err_pct": round(g_err, 4),
            "spi_overall": spi["overall"]["spi"],
            "spi_precision": spi["overall"]["precision"],
            "spi_recall": spi["overall"]["recall"],
            "cld_max": cld["max_deviation"], "cld_mean": cld["mean_deviation"],
            "cld_conserved": cld["is_conserved"],
            "fh_days": fh["fh_days"], "fh_pct": fh["fh_pct"],
            "ood_overall": ood.get("overall"),
            "ood_explosive": ood.get("explosive"),
            "ood_fast_dieout": ood.get("fast_dieout"),
            "ood_high_turn": ood.get("high_turn"),
            "ood_slow_burn": ood.get("slow_burn"),
            "zer": zer["zer"],
            "train_time_sec": train_time, "status": "ok",
        })

    except Exception as e:
        row["status"] = f"error: {type(e).__name__}: {str(e)[:80]}"

    return row


# ── single slow experiment ────────────────────────────────────────────────────

def _run_slow(depth, exp_id, b_true, g_true, b_err, g_err):
    row = {
        "depth": depth, "experiment_id": exp_id,
        "true_beta": b_true, "true_gamma": g_true,
        "r0_true": round(b_true / g_true, 4),
        "beta_err_pct": b_err, "gamma_err_pct": g_err,
        "status": "pending",
    }

    try:
        np.random.seed(exp_id + depth * 10000)
        t, S, I, R = run_gillespie_ensemble(
            beta=b_true, gamma=g_true, num_sims=depth
        )

        pss = compute_pss(t, S, I, R, train_neural_smoother,
                          discover_via_autograd, noise_level=0.01, n_repeats=10)
        cnt = compute_cnt(t, S, I, R, b_true, g_true,
                          train_neural_smoother, discover_via_autograd)
        dst = compute_dst(t, S, I, R, train_neural_smoother, discover_via_autograd)
        ttd = compute_ttd(train_neural_smoother, discover_via_autograd,
                          t, S, I, R, n_runs=3)

        row.update({
            "pss_beta": pss.get("pss_beta"), "pss_gamma": pss.get("pss_gamma"),
            "pss_n_valid": pss.get("n_valid"),
            "cnt": cnt.get("cnt"), "cnt_threshold": cnt.get("threshold_used"),
            "dst": dst.get("dst"),
            "ttd_mean_sec": ttd.get("mean_ttd_sec"),
            "ttd_min_sec": ttd.get("min_ttd_sec"),
            "ttd_max_sec": ttd.get("max_ttd_sec"),
            "status": "ok",
        })

    except Exception as e:
        row["status"] = f"error: {type(e).__name__}: {str(e)[:80]}"

    return row


# ── main: fast depth study ────────────────────────────────────────────────────

def run_depth_study():
    """
    Runs 500 experiments at each of 5 ensemble depths.
    Measures SPI, CLD, FH, OOD, ZER at every experiment.
    Saves results incrementally to FAST_CSV_PATH.
    """
    _setup_csv(FAST_CSV_PATH, FAST_COLUMNS)

    np.random.seed(42)
    torch.manual_seed(42)

    true_betas  = np.random.uniform(*BETA_RANGE,  N_EXPERIMENTS)
    true_gammas = np.random.uniform(*GAMMA_RANGE, N_EXPERIMENTS)

    print(f"Depth study: {N_EXPERIMENTS} × {len(DEPTHS)} depths = "
          f"{N_EXPERIMENTS * len(DEPTHS)} total runs")
    print(f"Depths: {DEPTHS}  |  Batch start: {BATCH_START}")
    print("=" * 55)

    completed, failed = 0, 0

    for depth in DEPTHS:
        print(f"\nN={depth}")
        depth_ok, depth_failed = 0, 0

        for exp_id in tqdm(range(BATCH_START, N_EXPERIMENTS), desc=f"N={depth}"):
            b_true = true_betas[exp_id]
            g_true = true_gammas[exp_id]

            if not is_epidemic_active(b_true, g_true):
                _write_row(FAST_CSV_PATH, FAST_COLUMNS, {
                    "depth": depth, "experiment_id": exp_id,
                    "true_beta": b_true, "true_gamma": g_true,
                    "r0_true": round(b_true / g_true, 4),
                    "status": "skipped_r0"
                })
                depth_failed += 1
                continue

            row = _run_fast(depth, exp_id, b_true, g_true)
            _write_row(FAST_CSV_PATH, FAST_COLUMNS, row)

            if row["status"] == "ok":
                depth_ok += 1; completed += 1
            else:
                depth_failed += 1; failed += 1

        print(f"  done — valid: {depth_ok}  failed: {depth_failed}")

    print(f"\nStudy complete — valid: {completed}  failed: {failed}")
    print(f"Saved to {FAST_CSV_PATH}")


# ── main: slow metrics ────────────────────────────────────────────────────────

def run_slow_metrics():
    """
    Reads the fast study CSV and runs PSS, CNT, DST, TTD on
    N_SLOW representative experiments per depth (spread across R0 range).
    Saves to SLOW_CSV_PATH.
    """
    if not os.path.exists(FAST_CSV_PATH):
        print(f"Fast results not found at {FAST_CSV_PATH}")
        print("Run run_depth_study() first.")
        return

    _setup_csv(SLOW_CSV_PATH, SLOW_COLUMNS)

    df     = pd.read_csv(FAST_CSV_PATH)
    df_ok  = df[df["status"] == "ok"].copy()

    selected = []
    for depth in DEPTHS:
        subset  = df_ok[df_ok["depth"] == depth].sort_values("r0_true")
        indices = np.linspace(0, len(subset) - 1, N_SLOW, dtype=int)
        selected.append(subset.iloc[indices])

    subset_df = pd.concat(selected, ignore_index=True)
    n_total   = len(subset_df)

    print(f"Running slow metrics on {n_total} experiments "
          f"({N_SLOW} per depth across R0 range)")
    print("=" * 55)

    completed, failed = 0, 0

    for _, r in tqdm(subset_df.iterrows(), total=n_total, desc="Slow metrics"):
        depth  = int(r["depth"])
        exp_id = int(r["experiment_id"])
        b_true = float(r["true_beta"])
        g_true = float(r["true_gamma"])
        b_err  = float(r["beta_err_pct"]) if pd.notna(r.get("beta_err_pct")) else None
        g_err  = float(r["gamma_err_pct"]) if pd.notna(r.get("gamma_err_pct")) else None

        row = _run_slow(depth, exp_id, b_true, g_true, b_err, g_err)
        _write_row(SLOW_CSV_PATH, SLOW_COLUMNS, row)

        if row["status"] == "ok":
            completed += 1
        else:
            failed += 1

    print(f"\nDone — valid: {completed}  failed: {failed}")
    print(f"Saved to {SLOW_CSV_PATH}")


# ── summary printers ──────────────────────────────────────────────────────────

def print_depth_summary():
    if not os.path.exists(FAST_CSV_PATH):
        print("No fast results found. Run run_depth_study() first.")
        return

    df    = pd.read_csv(FAST_CSV_PATH)
    df_ok = df[df["status"] == "ok"].copy()

    metrics = [
        ("beta_err_pct",  "Beta Error (%)"),
        ("gamma_err_pct", "Gamma Error (%)"),
        ("spi_overall",   "SPI Overall"),
        ("cld_max",       "CLD Max"),
        ("fh_pct",        "FH Coverage (%)"),
        ("ood_overall",   "OOD Overall MAE"),
        ("zer",           "ZER"),
        ("train_time_sec","Train Time (s)"),
    ]

    print("\n" + "=" * 72)
    print("ENSEMBLE DEPTH STUDY — FAST METRICS SUMMARY")
    print("=" * 72)
    header = f"{'Metric':<22}" + "".join(f"  N={d:<7}" for d in DEPTHS)
    print(header)
    print("-" * 72)

    for col, label in metrics:
        if col not in df_ok.columns:
            continue
        row_str = f"{label:<22}"
        for depth in DEPTHS:
            val = df_ok[df_ok["depth"] == depth][col].dropna().median()
            row_str += f"  {val:<9.4f}"
        print(row_str)

    print("-" * 72)
    for depth in DEPTHS:
        n_ok  = len(df_ok[df_ok["depth"] == depth])
        n_all = len(df[df["depth"] == depth])
        print(f"  N={depth}: {n_ok}/{n_all} valid runs")
    print("=" * 72)


def print_slow_summary():
    if not os.path.exists(SLOW_CSV_PATH):
        print("No slow results found. Run run_slow_metrics() first.")
        return

    df    = pd.read_csv(SLOW_CSV_PATH)
    df_ok = df[df["status"] == "ok"].copy()

    metrics = [
        ("pss_beta",     "PSS Beta (std)"),
        ("pss_gamma",    "PSS Gamma (std)"),
        ("cnt",          "CNT (max noise)"),
        ("dst",          "DST (min fraction)"),
        ("ttd_mean_sec", "TTD mean (sec)"),
    ]

    print("\n" + "=" * 72)
    print("ENSEMBLE DEPTH STUDY — SLOW METRICS SUMMARY")
    print("=" * 72)
    header = f"{'Metric':<24}" + "".join(f"  N={d:<7}" for d in DEPTHS)
    print(header)
    print("-" * 72)

    for col, label in metrics:
        if col not in df_ok.columns:
            continue
        row_str = f"{label:<24}"
        for depth in DEPTHS:
            subset = df_ok[df_ok["depth"] == depth][col].dropna()
            val    = subset.median() if not subset.empty else float("nan")
            row_str += f"  {val:<9.4f}"
        print(row_str)

    print("-" * 72)
    for depth in DEPTHS:
        n = len(df_ok[df_ok["depth"] == depth])
        print(f"  N={depth}: {n} experiments with slow metrics")
    print("=" * 72)


if __name__ == "__main__":
    run_depth_study()
    print_depth_summary()
    run_slow_metrics()
    print_slow_summary()
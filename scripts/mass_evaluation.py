import os
import sys
import csv
import numpy as np
import torch
from tqdm import tqdm

# Add src to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation import run_gillespie_ensemble, is_epidemic_active
from src.neural_net import train_neural_smoother
from src.equation_discovery import discover_via_autograd

def run_mass_evaluation(num_experiments=250, epochs=1500):
    print(f"\nStarting benchmark: {num_experiments} runs")
    print("-" * 50)

    os.makedirs("results", exist_ok=True)
    csv_path = "results/mass_evaluation_metrics.csv"
    
    # Setup CSV headers
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Experiment_ID", "True_Beta", "True_Gamma", 
            "Est_Beta", "Est_Gamma", "Beta_Error_Pct", "Gamma_Error_Pct"
        ])

    # Locking seeds for reproducible benchmark results
    np.random.seed(42)
    torch.manual_seed(42)

    # Randomize the epidemic parameters
    test_betas = np.random.uniform(1.0, 3.0, num_experiments)
    test_gammas = np.random.uniform(0.1, 1.0, num_experiments)

    beta_errors = []
    gamma_errors = []
    failed_runs = 0

    for i in tqdm(range(num_experiments), desc="Evaluating"):
        b_true = test_betas[i]
        g_true = test_gammas[i]

        # skip if R0 < 1.1 (disease dies out too fast to measure)
        if not is_epidemic_active(b_true, g_true):
            continue

        try:
            # 50 ensembles is usually enough to average out the Gillespie noise
            t, x_noisy = run_gillespie_ensemble(
                beta=b_true, gamma=g_true, num_sims=50, sensor_noise=0.015
            )

            ml_model = train_neural_smoother(t, x_noisy, epochs=epochs)

            # set print_equations=False so it doesn't flood the terminal
            b_est, g_est, _ = discover_via_autograd(ml_model, t, print_equations=False)

            # SINDy thresholding might have killed the terms entirely
            if b_est == 0.0 or g_est == 0.0:
                failed_runs += 1
                continue

            # absolute percentage errors
            b_err = abs(b_true - b_est) / b_true * 100
            g_err = abs(g_true - g_est) / g_true * 100

            beta_errors.append(b_err)
            gamma_errors.append(g_err)

            # save incrementally so we don't lose data if script crashes
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i, b_true, g_true, b_est, g_est, b_err, g_err])

        except Exception as e:
            # Catching generic exception to keep loop alive, just log the failure
            failed_runs += 1

    valid_runs = len(beta_errors)
    
    # print final stats
    print("\n--- Benchmark Results ---")
    print(f"Valid Runs: {valid_runs}")
    print(f"Failed (Pruned): {failed_runs}")
    print("-" * 25)
    
    if valid_runs > 0:
        print(f"Beta (Infection) Error:")
        print(f"  Mean:   {np.mean(beta_errors):.2f}%")
        print(f"  Median: {np.median(beta_errors):.2f}%")
        print(f"  90th %: {np.percentile(beta_errors, 90):.2f}%")
        
        print(f"\nGamma (Recovery) Error:")
        print(f"  Mean:   {np.mean(gamma_errors):.2f}%")
        print(f"  Median: {np.median(gamma_errors):.2f}%")
        print(f"  90th %: {np.percentile(gamma_errors, 90):.2f}%")
    else:
        print("No valid runs to report.")
        
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    run_mass_evaluation(num_experiments=250)
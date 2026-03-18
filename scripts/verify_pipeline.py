import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure Python can find your 'src' folder even if you haven't pip installed it yet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simulation import run_gillespie_ensemble, get_deterministic_truth
from src.neural_net import train_neural_smoother
from src.equation_discovery import discover_via_autograd
from src.numerics import simulate_discovered_physics, check_physical_conservation

def verify():
    print("="*60)
    print("🚀 INITIATING END-TO-END NEURO-SYMBOLIC PIPELINE VERIFICATION")
    print("="*60)

    # 1. Ground Truth Setup (Let's use an aggressive virus for the test)
    TRUE_BETA = 1.8
    TRUE_GAMMA = 0.5
    print(f"\n🎯 Target Parameters: Beta={TRUE_BETA}, Gamma={TRUE_GAMMA}")

    # 2. Generate Noisy Data
    # Adding a 1% sensor noise to the Gillespie mean to make it challenging
    t, x_noisy = run_gillespie_ensemble(
        beta=TRUE_BETA, gamma=TRUE_GAMMA, num_sims=30, sensor_noise=0.01
    )
    S_noisy, I_noisy = x_noisy[:, 0], x_noisy[:, 1]

    # 3. Train Neural ODE Manifold
    # Using 1500 epochs for a quick verification test (normally 2500)
    ml_model = train_neural_smoother(t, x_noisy, epochs=1500)

    # 4. Discover Equations via PySINDy + Autograd
    beta_est, gamma_est, x_smoothed = discover_via_autograd(ml_model, t)

    # Calculate Errors
    b_err = abs(TRUE_BETA - beta_est) / TRUE_BETA * 100
    g_err = abs(TRUE_GAMMA - gamma_est) / TRUE_GAMMA * 100

    print(f"\n✅ PIPELINE COMPLETE!")
    print(f"Discovered Beta:  {beta_est:.3f} (Error: {b_err:.1f}%)")
    print(f"Discovered Gamma: {gamma_est:.3f} (Error: {g_err:.1f}%)")

    # 5. Numerical Validation (RK4 vs Euler)
    y0 = [S_noisy[0], I_noisy[0]]
    
    print("\n⚙️ Testing Discovered Physics via Custom Integrators...")
    y_rk4 = simulate_discovered_physics(beta_est, gamma_est, t, y0, method='rk4')
    y_euler = simulate_discovered_physics(beta_est, gamma_est, t, y0, method='euler')

    # Physics Sanity Check
    check_physical_conservation(y_rk4)

    # 6. Plotting the Ultimate Verification Graph
    plt.figure(figsize=(12, 6))
    
    # Layer 1: The Raw Input
    plt.scatter(t[::10], I_noisy[::10], color='gray', alpha=0.5, label="Noisy Sensor Data (I)")
    
    # Layer 2: The ML Processing
    plt.plot(t, x_smoothed[:, 1], 'r--', linewidth=2, label="AI Manifold (Filtered)")
    
    # Layer 3: The Physics Verification
    plt.plot(t, y_rk4[:, 1], 'b-', linewidth=3, label="RK4 Simulated Physics (Stable)")
    plt.plot(t, y_euler[:, 1], 'g:', linewidth=2, label="Euler Step (Showing Drift)")

    plt.title(f"Neuro-Symbolic Recovery | True: β={TRUE_BETA}, γ={TRUE_GAMMA} | Found: β={beta_est:.3f}, γ={gamma_est:.3f}")
    plt.xlabel("Days")
    plt.ylabel("Infected Proportion")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot so we can see it even if running from a headless terminal
    plt.savefig("pipeline_verification.png")
    print("\n📊 Graph saved as 'pipeline_verification.png'. Displaying now...")
    plt.show()

if __name__ == "__main__":
    verify()
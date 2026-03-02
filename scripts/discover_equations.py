import torch
import numpy as np
from sira.simulation.gillespie import GillespieSimulator
from sira.neural_ode.vector_field import ODEFunc
from sira.neural_ode.solver import NeuralODESystem
from sira.symbolic.library import Library
from sira.symbolic.stlsq import STLSQ

def main():
    # 1. Config & Data Generation
    N = 1000
    params = {'beta': 0.3, 'gamma': 0.1}
    sim = GillespieSimulator(model_type="SIR", N=N, params=params)
    t_raw, x_raw = sim.simulate([995, 5, 0], t_max=40)
    
    # Scale and prepare data
    x_train = torch.tensor(x_raw, dtype=torch.float32) / N
    y0 = x_train[0]
    t_train = torch.tensor(t_raw, dtype=torch.float32)

    # 2. Train Neural ODE (Filtering the noise)
    print("Training Neural ODE...")
    func = ODEFunc(latent_dim=3, hidden_dim=64)
    model = NeuralODESystem(func)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(3):
        optimizer.zero_grad()
        pred = model(y0, t_train)
        loss = torch.nn.MSELoss()(pred, x_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # 3. Discovery (The SINDy Step)
    print("\nDiscovering Equations...")
    with torch.no_grad():
        x_smooth = model(y0, t_train)
        dxdt_neural = func(0, x_smooth).numpy()
    
    theta = Library.poly_library(x_smooth).numpy()
    discoverer = STLSQ(threshold=0.02)
    coeffs = discoverer.fit(theta, dxdt_neural)

    # 4. Results
    terms = ['1', 'S', 'I', 'R', 'S^2', 'SI', 'SR', 'I^2', 'IR', 'R^2']
    labels = ['dS/dt', 'dI/dt', 'dR/dt']
    
    for i in range(3):
        eq = " + ".join([f"({coeffs[j,i]:.3f})*{terms[j]}" for j in range(10) if abs(coeffs[j,i]) > 0.01])
        print(f"{labels[i]} = {eq}")

if __name__ == "__main__":
    main()
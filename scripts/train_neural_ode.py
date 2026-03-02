import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from sira.simulation.gillespie import GillespieSimulator
from sira.neural_ode.vector_field import ODEFunc
from sira.neural_ode.solver import NeuralODESystem

def main():
    # 1. Generate Training Data (The Jagged Reality)
    N = 1000
    params = {'beta': 0.3, 'gamma': 0.1}
    initial_state = [995.0, 5.0, 0.0]
    
    sim = GillespieSimulator(model_type="SIR", N=N, params=params)
    t_raw, x_raw = sim.simulate(initial_state, t_max=50)
    
    # Convert to Tensors for PyTorch
    # We normalize by N so the network sees values between 0 and 1
    t_train = torch.tensor(t_raw, dtype=torch.float32)
    x_train = torch.tensor(x_raw, dtype=torch.float32) / N 
    y0 = x_train[0]

    # 2. Initialize the AI
    func = ODEFunc(latent_dim=3, hidden_dim=64)
    model = NeuralODESystem(func)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    print("Starting Training... Learning the hidden physics.")

    # 3. Training Loop
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Predict the whole trajectory
        pred_x = model(y0, t_train)
        
        # Compare prediction to jagged reality
        loss = loss_fn(pred_x, x_train)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # 4. Visualize the "Neural Smoothing"
    with torch.no_grad():
        final_pred = model(y0, t_train).numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(t_raw, x_raw[:, 1]/N, 'r.', alpha=0.3, label='Jagged Data (I)')
    plt.plot(t_raw, final_pred[:, 1], 'k', linewidth=2, label='Neural ODE (Learned)')
    plt.title("Neural ODE Smoothing the Stochastic Noise")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
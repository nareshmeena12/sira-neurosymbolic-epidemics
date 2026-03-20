"""
Neural Manifold Learning Module

Implements a PyTorch-based neural network to learn a continuous, 
differentiable manifold from stochastic Gillespie simulation data.
This continuous representation enables analytical differentiation 
via `torch.autograd` during the equation discovery phase.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SIRContinuousManifold(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for symbolic regression.
    
    Uses Tanh activations exclusively to ensure the learned manifold is 
    infinitely smooth. This prevents discontinuities in higher-order 
    derivatives when computing dS/dt and dI/dt via autograd.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Maps time (t) -> population fractions [S, I, R]
        """
        return self.net(t)


def train_neural_smoother(t_data: np.ndarray, S_data: np.ndarray,
                           I_data: np.ndarray, R_data: np.ndarray,
                           epochs: int = 2000, lr: float = 0.005) -> SIRContinuousManifold:
    """
    Trains the neural network to fit the stochastic simulation data.
    
    Args:
        t_data:  1D numpy array of time steps.
        S_data:  1D numpy array of mean susceptible fractions.
        I_data:  1D numpy array of mean infected fractions.
        R_data:  1D numpy array of mean recovered fractions.
        epochs:  Number of training iterations.
        lr:      Learning rate for the Adam optimizer.
        
    Returns:
        Trained PyTorch model moved to CPU.
    """
    # print("Training neural manifold on stochastic data...")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = SIRContinuousManifold().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    t_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1).to(device)
    target   = torch.tensor(
        np.vstack((S_data, I_data, R_data)).T,
        dtype=torch.float32
    ).to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean((model(t_tensor) - target) ** 2)
        loss.backward()
        optimizer.step()

    model.eval()
    return model.cpu()
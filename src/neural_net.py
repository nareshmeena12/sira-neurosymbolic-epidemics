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
        Maps Time (t) -> Population Fractions [S, I, R]
        """
        return self.net(t)


def train_neural_smoother(t_data: np.ndarray, x_noisy: np.ndarray, epochs: int = 2500, lr: float = 0.005) -> SIRContinuousManifold:
    """
    Trains the neural network to fit the stochastic data.
    
    Args:
        t_data: 1D numpy array of time steps.
        x_noisy: 2D numpy array containing the noisy [S, I] (and optionally R) data.
        epochs: Number of training iterations.
        lr: Learning rate for the Adam optimizer.
        
    Returns:
        Trained PyTorch model.
    """
    print("Training neural manifold on stochastic data...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIRContinuousManifold().to(device)
    
    # Use Weight Decay (L2 Regularization) to act as a low-pass filter,
    # encouraging smooth curve fitting rather than memorizing stochastic noise.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    t_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1).to(device)
    
    # Calculate R if only S and I are provided to ensure a 3-column target
    if x_noisy.shape[1] == 2:
        R_data = 1.0 - (x_noisy[:, 0] + x_noisy[:, 1])
        target = torch.tensor(np.column_stack((x_noisy, R_data)), dtype=torch.float32).to(device)
    else:
        target = torch.tensor(x_noisy, dtype=torch.float32).to(device)

    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(t_tensor)
        
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()

    model.eval()
    return model.cpu()
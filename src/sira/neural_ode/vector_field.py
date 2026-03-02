import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    The 'Brain' of the Neural ODE.
    Instead of predicting the next state, it learns to predict the 
    derivative: dy/dt = f(y, t, theta)
    """
    def __init__(self, latent_dim=3, hidden_dim=64):
        super(ODEFunc, self).__init__()
        
        # This network represents the right-hand side of our ODE system
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Softplus(),  # Softplus is C-infinity smooth; crucial for ODE solvers
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Initialize weights with small values to prevent 'gradient explosion' 
        # in the ODE solver during the first few epochs.
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        """
        t: The current time (required by torchdiffeq, even if not used)
        y: The current state [S, I, R]
        """
        # We return the predicted derivative
        return self.net(y)
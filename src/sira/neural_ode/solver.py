import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
class NeuralODESystem(nn.Module):
    """
    The full Neural ODE system that wraps the vector field
    and the integration solver.
    """
    def __init__(self, vector_field):
        super(NeuralODESystem, self).__init__()
        self.f = vector_field

    def forward(self, y0, t):
        """
        y0: Initial state [S, I, R] at t=0
        t:  The time points we want predictions for
        """
        # We use the 'dopri5' solver (Dormand-Prince), 
        # which is an adaptive Runge-Kutta 4th/5th order method.
        # It's the gold standard for accuracy in physical systems.
        predicted_trajectory = odeint(self.f, y0, t, method='dopri5')
        return predicted_trajectory
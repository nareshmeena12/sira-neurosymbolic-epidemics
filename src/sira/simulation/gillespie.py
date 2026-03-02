"""
Exact Stochastic Simulation Algorithm (Gillespie SSA) for SIRA.
Generates exact sample paths from the Continuous-Time Markov Chain (CTMC).
"""
import numpy as np

class GillespieSimulator:
    """
    A highly optimized, vectorized exact stochastic simulator.
    Supports SIR, SEIR, and SIRS models dynamically via stoichiometry matrices.
    """
    def __init__(self,model_type='SIR',N=100,params=None):
        self.model_type=model_type.upper()
        self.N=N
        self.params=params
        self._setup_stoichiometry()
    def _setup_stoichiometry(self):
        """
        Defines the state-change vectors for every possible event.
        Rows represent events, columns represent compartments.
        """
        if self.model_type=='SIR':
            self.V=np.array([
                [-1,1,0],
                [0,-1,1]
            ])
            self.num_compartments=3

        elif self.model_type=='SEIR':
            # State: [S, E, I, R]
            # Event 0: Infection S->E | Event 1: Incubation E->I | Event 2: Recovery I->R
            self.V=np.array([
                [-1,0,1,0],
                [0,-1,0,1],
                [0,1,-1,0]
            ])
            self.num_compartments=4
        elif self.model_type == "SIRS":
            # State: [S, I, R]
            # Event 0: Infection S->I | Event 1: Recovery I->R | Event 2: Waning R->S
            self.V = np.array([
                [-1,  1,  0],
                [ 0, -1,  1],
                [ 1,  0, -1]
            ])
            self.num_compartments = 3
        else:
            raise ValueError(f"Model {self.model_type} not supported.")
        
    def _propensities(self,state):
        """Calculates the instantaneous rates (probabilities) of events."""
        if self.model_type == "SIR":
            S, I, R = state
            return np.array([
                (self.params.get('beta', 0.3) * S * I) / self.N,
                self.params.get('gamma', 0.1) * I
            ])
        elif self.model_type == "SEIR":
            S, E, I, R = state
            return np.array([
                (self.params.get('beta', 0.3) * S * I) / self.N,
                self.params.get('sigma', 0.2) * E,
                self.params.get('gamma', 0.1) * I
            ])
            
        elif self.model_type == "SIRS":
            S, I, R = state
            return np.array([
                (self.params.get('beta', 0.3) * S * I) / self.N,
                self.params.get('gamma', 0.1) * I,
                self.params.get('omega', 0.05) * R
            ])
    def simulate(self, initial_state, t_max=100.0, max_steps=1000000):
        """
        Runs the exact Monte Carlo SSA.
        Returns: time array (t), state array (X)
        """
        t = np.zeros(max_steps)
        X = np.zeros((max_steps, self.num_compartments), dtype=int)
        current_t = 0.0
        current_X = np.array(initial_state, dtype=int)
        
        t[0] = current_t
        X[0] = current_X
        step = 0
        
        while current_t < t_max and step < max_steps - 1:
            props = self._propensities(current_X)
            a0 = np.sum(props)
    
            if a0 == 0.0:
                break
                
            r1, r2 = np.random.rand(2)
            tau = (1.0 / a0) * np.log(1.0 / r1)
            
            event_idx = np.searchsorted(np.cumsum(props), r2 * a0)
            current_t += tau
            current_X = current_X + self.V[event_idx]
            
            step += 1
            t[step] = current_t
            X[step] = current_X
        return t[:step+1], X[:step+1]

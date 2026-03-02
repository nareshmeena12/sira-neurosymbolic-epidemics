import numpy as np

class STLSQ:
    """
    Sequential Thresholded Least Squares (STLSQ).
    The algorithm that 'discovers' the sparsest (simplest) equation.
    """
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.coefficients = None

    def fit(self, Theta, dXdt):
        """
        Theta: Dictionary matrix from Library.poly_library
        dXdt:  The slopes (derivatives) from our Neural ODE
        """
        Xi = np.linalg.lstsq(Theta, dXdt, rcond=None)[0]
        
        for _ in range(10):
            small_indices = np.abs(Xi) < self.threshold
            Xi[small_indices] = 0 
            
            for j in range(dXdt.shape[1]):
                big_indices = ~small_indices[:, j]
                Xi[big_indices, j] = np.linalg.lstsq(Theta[:, big_indices], dXdt[:, j], rcond=None)[0]
        
        self.coefficients = Xi
        return Xi
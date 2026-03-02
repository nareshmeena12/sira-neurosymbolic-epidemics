import torch

class Library:
    """
    Constructs a dictionary (Theta) of candidate functions.
    Example: If input is [S, I], it builds [1, S, I, S^2, SI, I^2]
    """
    @staticmethod
    def poly_library(X,degree=2):
        """
        X: State matrix [TimePoints, 3] (S, I, R)
        Returns: Library matrix [TimePoints, LibrarySize]
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        n_samples, n_dims = X.shape
        library = [torch.ones(n_samples, 1)]

        for i in range(n_dims):
            library.append(X[:, i:i+1])
        if degree >= 2:
            for i in range(n_dims):
                for j in range(i, n_dims):
                    library.append(X[:, i:i+1] * X[:, j:j+1])
        return torch.cat(library, dim=1)
    
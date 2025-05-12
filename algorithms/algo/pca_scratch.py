import numpy as np

class PCAScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        idxs = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idxs[:self.n_components]]

        return np.dot(X_centered, self.components)

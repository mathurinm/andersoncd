import numpy as np
from numba.experimental import jitclass
from numba import float64

spec_quadratic = [
    ('Xty', float64[:]),
    ('lipschitz', float64[:]),
]


@jitclass(spec_quadratic)
class Quadratic():
    def __init__(self):
        pass

    def initialize(self, X, y):
        self.Xty = X.T @ y
        self.lipschitz = (X ** 2).sum(axis=0) / len(y)

    def initialize_sparse(
            self, data, indptr, indices, y, n_features):
        self.Xty = np.empty(n_features)
        self.lipschitz = np.empty(n_features)
        for j in range(n_features):
            Xj = data[indptr[j]:indptr[j+1]]
            idx_nz = indices[indptr[j]:indptr[j+1]]
            self.Xty[j] = Xj @ y[idx_nz]
            self.lipschitz[j] = (Xj ** 2).sum() / len(y)

    def value(self, y, w, Xw):
        return np.sum((y - Xw) ** 2) / (2 * len(Xw))

    def gradient_scalar(self, X, w, Xw, j):
        return (X[:, j] @ Xw - self.Xty[j]) / len(Xw)

    def gradient_scalar_sparse(self, Xj, idx_nz, Xw, j):
        return (Xj @ Xw[idx_nz] - self.Xty[j]) / len(Xw)

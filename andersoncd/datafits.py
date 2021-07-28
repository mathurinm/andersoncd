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
            self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1
        self.Xty = np.zeros(n_features)
        self.lipschitz = np.empty(n_features)
        for j in range(n_features):
            Xj = X_data[X_indptr[j]:X_indptr[j+1]]
            idx_nz = X_indices[X_indptr[j]:X_indptr[j+1]]
            for i, idx_i in enumerate(idx_nz):
                self.Xty[j] += Xj[i] * y[idx_i]
            self.lipschitz[j] = (Xj ** 2).sum() / len(y)

    def value(self, y, w, Xw):
        return np.sum((y - Xw) ** 2) / (2 * len(Xw))

    def gradient_scalar(self, X, w, Xw, j):
        return (X[:, j] @ Xw - self.Xty[j]) / len(Xw)

    def gradient_scalar_sparse(self, Xj, idx_nz, Xw, j):
        XjTXw = 0
        for i, idx_i in enumerate(idx_nz):
            XjTXw += Xj[i] * Xw[idx_i]
        return (XjTXw - self.Xty[j]) / len(Xw)

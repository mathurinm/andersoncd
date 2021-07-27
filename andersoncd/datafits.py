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

    def value(self, X, y, w, Xw):
        return np.sum((y - Xw) ** 2) / (2 * len(Xw))

    def gradient(self, X, y, w, Xw):
        return (X.T @ Xw - self.Xty) / len(Xw)

    def gradient_scalar(self, X, w, Xw, j):
        # TODO remove w unused in argument
        return (X[:, j] @ Xw - self.Xty[j]) / len(Xw)

    def gradient_scalar_sparse(self, Xj, idx_nz, Xw, y):
        return (Xj @ (Xw[idx_nz] - y[idx_nz])) / len(Xw)

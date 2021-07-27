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
        self.Xty = X.T @ y  # TODO handle centering
        self.lipschitz = (X ** 2).sum(axis=0)  # TODO beware of n_samples

    def value(self, X, y, w, Xw):
        return np.sum((y - Xw) ** 2) / (2 * len(Xw))

    # @staticmethod
    def gradient(self, X, y, w, Xw):
        return X.T @ (Xw - y)

    # @staticmethod
    def gradient_scalar(self, X, w, Xw, j):
        return (X[:, j] @ Xw - self.Xty[j])

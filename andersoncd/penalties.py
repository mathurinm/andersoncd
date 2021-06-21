import numpy as np
from numba import float64
from abc import abstractmethod
from numba.experimental import jitclass

from andersoncd.utils import ST


spec_L1 = [
    ('alpha', float64),
]


class Penalty():
    @abstractmethod
    def value(self, w):
        """Value of penalty at vector w."""

    @abstractmethod
    def prox_1d(self, value, stepsize, j):
        """Proximal operator of penalty for feature j."""

    @abstractmethod
    def subdiff_distance(self, w, grad, ws):
        """Distance of gradient to subdifferential of penalty for feature j."""

    @abstractmethod
    def is_penalized(self, n_features):
        """Mask corresponding to penalized features."""


@jitclass(spec_L1)
class L1(Penalty):
    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, w):
        return self.alpha * np.sum(np.abs(w))

    def prox_1d(self, value, stepsize, j):
        return ST(value, self.alpha * stepsize)

    def subdiff_distance(self, w, grad, ws):
        res = np.zeros_like(grad)
        for idx in range(ws.shape[0]):
            j = ws[idx]
            if w[j] == 0:
                # distance of grad to alpha * [-1, 1]
                res[idx] = max(0, np.abs(grad[j]) - self.alpha)
            else:
                # distance of grad_j to alpha  * sign(w[j])
                res[idx] = np.abs(np.abs(grad[j]) - self.alpha)
        return res

    def is_penalized(self, n_features):
        return np.ones(n_features, dtype=bool)

    def prox(self):
        pass
        # TODO needed ?


spec_WeightedL1 = [
    ('alpha', float64),
    ('weights', float64[:]),
]


@jitclass(spec_WeightedL1)
class WeightedL1(Penalty):
    def __init__(self, alpha, weights):
        self.alpha = alpha
        self.weights = weights

    def value(self, w):
        return self.alpha * np.sum(np.abs(w) * self.weights)

    def prox_1d(self, value, stepsize, j):
        return ST(value, self.alpha * stepsize * self.weights[j])

    def subdiff_distance(self, w, grad, ws):
        res = np.zeros_like(grad)
        for idx in range(ws.shape[0]):
            j = ws[idx]
            if w[j] == 0:
                # distance of grad to alpha * weights[j] * [-1, 1]
                res[idx] = max(0, np.abs(grad[j]) -
                               self.alpha * self.weights[j])
            else:
                # distance of grad_j to alpha * weights[j] * sign(w[j])
                res[idx] = np.abs(np.abs(grad[j]) -
                                  self.alpha * self.weights[j])
        return res

    def is_penalized(self, n_features):
        return self.weights != 0

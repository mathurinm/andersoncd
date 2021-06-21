import numpy as np
from numba import float64
from abc import abstractmethod
from numba.experimental import jitclass
from numba.types import bool_

from andersoncd.utils import ST


class Penalty():
    @abstractmethod
    def value(self, w):
        """Value of penalty at vector w."""

    @abstractmethod
    def prox_1d(self, value, stepsize, j):
        """Proximal operator of penalty for feature j."""

    @abstractmethod
    def subdiff_distance(self, w, neg_grad, ws):
        """Distance of gradient to subdifferential of penalty for feature j.

        w : array, shape (n_features,)
            Coefficient vector.
        neg_grad: array, shape (n_features,)
            Minus the value of the gradient of the datafit at w.
        ws: array
            Features in the working set.
        """

    @abstractmethod
    def is_penalized(self, n_features):
        """Mask corresponding to penalized features."""


spec_L1 = [
    ('alpha', float64),
]


@jitclass(spec_L1)
class L1(Penalty):
    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, w):
        return self.alpha * np.sum(np.abs(w))

    def prox_1d(self, value, stepsize, j):
        return ST(value, self.alpha * stepsize)

    def subdiff_distance(self, w, neg_grad, ws):
        res = np.zeros_like(neg_grad)
        for idx in range(ws.shape[0]):
            j = ws[idx]
            if w[j] == 0:
                # distance of grad to alpha * [-1, 1]
                res[idx] = max(0, np.abs(neg_grad[idx]) - self.alpha)
            else:
                # distance of grad_j to alpha  * sign(w[j])
                res[idx] = np.abs(np.abs(neg_grad[idx]) - self.alpha)
        return res

    def is_penalized(self, n_features):
        return np.ones(n_features).astype(bool_)


spec_L1_plus_L2 = [
    ('alpha', float64),
    ('l1_ratio', float64),
]


# TODO find a better name
@jitclass(spec_L1_plus_L2)
class L1_plus_L2(Penalty):
    def __init__(self, alpha, l1_ratio):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def value(self, w):
        res = self.l1_ratio * self.alpha * np.sum(np.abs(w))
        res += (1 - self.l1_ratio) * self.alpha / 2 * np.sum(w ** 2)
        return res

    def prox_1d(self, value, stepsize, j):
        res = ST(value, self.l1_ratio * self.alpha * stepsize)
        res /= (1 + stepsize * (1 - self.l1_ratio) * self.alpha)
        return res

    def subdiff_distance(self, w, neg_grad, ws):
        res = np.zeros_like(neg_grad)
        for idx in range(ws.shape[0]):
            j = ws[idx]
            if w[j] == 0:
                # distance of grad_j to alpha * [-1, 1]
                tmp = neg_grad[idx]
                tmp -= (1 - self.l1_ratio) * self.alpha * w[j]
                tmp = np.abs(tmp)
                tmp -= self.l1_ratio * self.alpha
                res[idx] = max(0, tmp)
            else:
                # distance of grad_j to alpha  * sign(w[j])
                tmp = neg_grad[idx]
                tmp -= (1 - self.l1_ratio) * self.alpha * w[j]
                tmp = np.abs(tmp)
                tmp -= self.l1_ratio * self.alpha
                res[idx] = np.abs(tmp)
        return res

    def is_penalized(self, n_features):
        return np.ones(n_features).astype(bool_)


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
                res[idx] = max(0, np.abs(grad[idx]) -
                               self.alpha * self.weights[j])
            else:
                # distance of grad_j to alpha * weights[j] * sign(w[j])
                res[idx] = np.abs(np.abs(grad[idx]) -
                                  self.alpha * self.weights[j])
        return res

    def is_penalized(self, n_features):
        return self.weights != 0

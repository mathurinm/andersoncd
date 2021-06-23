# Author: Quentin Bertrand <quentin.bertrand@inria.fr>
#         Mathurin Massias <mathurin.massias@gmail.com>
#         Salim Benchelabi
# License: BSD 3 clause

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
        """ alpha * ||w||_1
        """
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


@jitclass(spec_L1_plus_L2)
class L1_plus_L2(Penalty):
    def __init__(self, alpha, l1_ratio):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def value(self, w):
        """ alpha * (l1_ratio * ||w||_1 + (1 - l1_ratio) * ||w||_2^2)
        """
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
        """ sum_{j=1}^{n_features} alpha * weights[j] |w_j|
        """
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


spec_MCP = [
    ('alpha', float64),
    ('gamma', float64),
]


@jitclass(spec_MCP)
class MCP_pen(Penalty):
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma

    def value(self, w):
        """
        With x >= 0
        pen(x) = alpha * x - x^2 / (2 * gamma) if x =< gamma * alpha
                 gamma * alpha ** 2 / 2        if x > gamma * alpha
        value = sum_{j=1}^{n_features} pen(|w_j|)

        For more details see
        Coordinate descent algorithms for nonconvex penalized regression,
        with applications to biological feature selection, Breheny and Huang.
        """
        s0 = np.abs(w) < self.gamma * self.alpha
        res = np.full_like(w, self.gamma * self.alpha ** 2 / 2.)
        res[s0] = self.alpha * np.abs(w[s0]) - w[s0]**2 / (2 * self.gamma)
        return np.sum(res)

    def prox_1d(self, value, stepsize, j):
        tau = self.alpha * stepsize
        g = self.gamma / stepsize
        if np.abs(value) <= tau:
            return 0.
        if np.abs(value) > g * tau:
            return value
        return np.sign(value) * (np.abs(value) - tau) / (1. - 1./g)

    def subdiff_distance(self, w, grad, ws):
        res = np.zeros_like(grad)
        for idx in range(ws.shape[0]):
            j = ws[idx]
            if w[j] == 0:
                # distance of grad to alpha * [-1, 1]
                res[idx] = max(0, np.abs(grad[idx]) - self.alpha)
            elif np.abs(w[j]) < self.alpha * self.gamma:
                # distance of grad_j to (alpha - abs(w[j])/gamma) * sign(w[j])
                res[idx] = np.abs(np.abs(grad[idx]) - (
                    self.alpha - np.abs(w[j])/self.gamma))
            else:
                # distance of grad to 0
                res[idx] = np.abs(grad[idx])
        return res

    def is_penalized(self, n_features):
        return np.ones(n_features).astype(bool_)

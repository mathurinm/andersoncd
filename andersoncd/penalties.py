import numpy as np
from numba import float64
from numba.experimental import jitclass

from andersoncd.utils import ST


spec_L1 = [
    ('alpha', float64),
    # ('weights', float64[:]),
]


@jitclass(spec_L1)
class L1():
    def __init__(self, alpha):
        self.alpha = alpha
        # self.weights = weights

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
                # distance of grad_j to alpha * weight * sign(w[j])
                res[idx] = np.abs(np.abs(grad[j]) - self.alpha)
        return res

    def prox(self):
        pass
        # TODO needed ?

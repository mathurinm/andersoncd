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

    def prox_scalar(self, value, stepsize):
        return ST(value, self.alpha * stepsize)

    def prox_scalar(self):
        pass
        # TODO needed ?

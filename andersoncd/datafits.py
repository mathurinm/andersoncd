# Author: Quentin Bertrand <quentin.bertrand@inria.fr>
#         Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

from numba import njit
from abc import abstractmethod

import numpy as np
from numpy.linalg import norm
from numba import float64
from numba.experimental import jitclass


class BaseDatafit():
    @abstractmethod
    def initialize(self, X, y):
        """Computations before fitting on new data X and y."""

    @abstractmethod
    def initialize_sparse(
            self, X_data, X_indptr, X_indices, y):
        """Computations before fitting on new data X and y, with sparse X."""

    @abstractmethod
    def value(self, y, w, Xw):
        """Value of datafit at vector w."""

    @abstractmethod
    def gradient_scalar(self, X, y, w, Xw, j):
        """Gradient with respect to j-th coordinate of w."""

    @abstractmethod
    def gradient_scalar_sparse(self, Xj, idx_nz, y, Xw, j):
        """Gradient with respect to j-th coordinate of w when X is sparse."""


spec_quadratic = [
    ('Xty', float64[:]),
    ('lipschitz', float64[:]),
]


@jitclass(spec_quadratic)
class Quadratic(BaseDatafit):
    def __init__(self):
        pass

    def initialize(self, X, y):
        self.Xty = X.T @ y
        n_samples, n_features = X.shape
        self.lipschitz = np.zeros(n_features)
        for j in range(n_features):
            self.lipschitz[j] = norm(X[:, j]) ** 2 / len(y)
            # for i in range(n_samples):
            #     self.lipschitz[j] += X[i, j] ** 2 / len(y)

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

    def gradient_scalar(self, X, y, w, Xw, j):
        # res = 0
        # n_samples = len(Xw)
        # for i in range(n_samples):
        #     if X[i, j] != 0 and Xw[i] != 0:
        #         res += X[i, j] * Xw[i]
        # res -= self.Xty[j]
        # res /= len(Xw)
        # return res
        # Remark: the following lin is faster than expanding the computation
        return (X[:, j] @ Xw - self.Xty[j]) / len(Xw)

    def gradient_scalar_sparse(self, Xj, idx_nz, y, Xw, j):
        # TODO j is not needed since we pass Xj
        XjTXw = 0
        for i, idx_i in enumerate(idx_nz):
            XjTXw += Xj[i] * Xw[idx_i]
        return (XjTXw - self.Xty[j]) / len(Xw)


@njit
def sigmoid(x):
    """Vectorwise sigmoid."""
    return 1. / (1. + np.exp(- x))


@jitclass(spec_quadratic)
class Logistic(BaseDatafit):
    def __init__(self):
        pass

    def initialize(self, X, y):
        self.lipschitz = (X ** 2).sum(axis=0) / (len(y) * 4)

    def initialize_sparse(self, X_data, X_indptr, X_indices, y):
        n_features = len(X_indptr) - 1
        self.lipschitz = np.zeros(n_features)
        for j in range(n_features):
            Xj = X_data[X_indptr[j]:X_indptr[j+1]]
            self.lipschitz[j] = (Xj ** 2).sum() / len(y) / 4

    def value(self, y, w, Xw):
        return np.log(1. + np.exp(- y * Xw)).sum() / len(y)

    def gradient_scalar(self, X, y, w, Xw, j):
        return (- X[:, j] @ (y * sigmoid(- y * Xw))) / len(y)

    def gradient_scalar_sparse(self, Xj, idx_nz, y, Xw, j):
        grad = 0.
        for i, idx_i in enumerate(idx_nz):
            grad -= Xj[i] * y[idx_i] * sigmoid(- y[idx_i] * Xw[idx_i])
        return grad / len(Xw)

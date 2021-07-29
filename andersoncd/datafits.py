# Author: Quentin Bertrand <quentin.bertrand@inria.fr>
#         Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

from numba import njit
from abc import abstractmethod

import numpy as np
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
    def value(self, w, Xw):
        """Value of datafit at vector w."""

    @abstractmethod
    def gradient_scalar(self, w, Xw, j):
        """Gradient with respect to j-th coordinate of w."""

    # @abstractmethod
    # def gradient_scalar_sparse(self, Xj, idx_nz, y, Xw, j):
    #     """Gradient with respect to j-th coordinate of w when X is sparse."""


common_spec = [
    ('X', float64[:, :]),
    ('X_data', float64[:]),
    ('X_indices', float64[:]),
    ('X_indptr', float64[:]),
    ('y', float64[:]),
]

spec_quadratic = [
    ('Xty', float64[:]),
    ('lipschitz', float64[:]),
]


@jitclass(common_spec + spec_quadratic)
class Quadratic(BaseDatafit):
    def __init__(self):
        pass

    def initialize(self, X, y):
        self.Xty = X.T @ y
        self.lipschitz = (X ** 2).sum(axis=0) / len(y)
        self.X = X
        self.is_sparse = False

        self.X_data = np.ones(1, dtype=np.float64)
        self.X_indices = np.ones(1, dtype=np.int32)
        self.X_indptr = np.ones(1, dtype=np.int32)

    def initialize_sparse(
            self, X_data, X_indptr, X_indices, y):
        self.X_data = X_data
        self.X_indices = X_indices
        self.X_indptr = X_indptr
        self.is_sparse = True
        self.X = np.ones((1, 1), dtype=np.float64)

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

    def gradient_scalar(self, w, Xw, j):
        if self.is_sparse:
            XjTXw = 0
            Xj = self.X_data[self.X_indptr[j]:self.X_indptr[j+1]]
            idx_nz = self.X_indices[self.X_indptr[j]:self.X_indptr[j+1]]
            for i, idx_i in enumerate(idx_nz):
                XjTXw += Xj[i] * Xw[idx_i]
            return (XjTXw - self.Xty[j]) / len(Xw)
        else:
            return (self.X[:, j] @ Xw - self.Xty[j]) / len(Xw)

    # def gradient_scalar_sparse(self, Xj, idx_nz, y, Xw, j):
    #     # TODO j is not needed since we pass Xj


# @njit
# def sigmoid(x):
#     """Vectorwise sigmoid."""
#     return 1. / (1. + np.exp(- x))


# @jitclass(spec_logistic)
# class Logistic(BaseDatafit):
#     def __init__(self):
#         pass

#     def initialize(self, X, y):
#         self.lipschitz = (X ** 2).sum(axis=0) / len(y) / 4
#         self.X = X

#     def initialize_sparse(self, X_data, X_indptr, X_indices, y):
#         n_features = len(X_indptr) - 1
#         self.lipschitz = np.zeros(n_features)
#         for j in range(n_features):
#             Xj = X_data[X_indptr[j]:X_indptr[j+1]]
#             self.lipschitz[j] = (Xj ** 2).sum() / len(y) / 4

#     def value(self, y, w, Xw):
#         return np.log(1. + np.exp(- y * Xw)).sum() / len(y)

#     def gradient_scalar(self, X, y, w, Xw, j):
#         return (- X[:, j] @ (y * sigmoid(- y * Xw))) / len(y)

#     def gradient_scalar_sparse(self, Xj, idx_nz, y, Xw, j):
#         grad = 0.
#         for i, idx_i in enumerate(idx_nz):
#             grad -= Xj[i] * y[idx_i] * sigmoid(- y[idx_i] * Xw[idx_i])
#         return grad / len(Xw)

import scipy
import numpy as np
import seaborn as sns
from scipy.sparse import issparse

from numpy.linalg import norm


C_LIST = sns.color_palette()


def get_gd_mat_gram(A):
    """A is the Hessian."""
    n_features = A.shape[0]
    it = np.eye(n_features) - A / norm(A, ord=2)
    return it


def get_gd_mat(X):
    """X is the design matrix."""
    n_features = X.shape[1]
    return np.eye(n_features) - X.T @ X / norm(X, ord=2) ** 2


def get_gd_mat_dual(X):
    """X is the design matrix."""
    n_samples = X.shape[0]
    return np.eye(n_samples) - X @ X.T / norm(X, ord=2) ** 2


def get_cd_mat_gram(A):
    """A is the Hessian."""
    n_features = A.shape[0]
    L = np.diag(A)
    it = np.identity(n_features)

    for j in np.where(L != 0)[0]:
        line = np.zeros(n_features)
        line[j] = 1
        line -= A[j, :] / L[j]
        it[j, :] = it.T @ line
    return it


def get_cd_mat(X, sym=False, rho=0):
    n_features = X.shape[1]
    if issparse(X):
        L = scipy.sparse.linalg.norm(X, axis=0) ** 2
    else:
        L = norm(X, axis=0) ** 2
    mat = np.eye(X.shape[1])
    for j in range(n_features):
        mat[j:j+1, :] -= (X[:, j].T @ X) @ mat / L[j]
        mat[j:j+1, :] -= ((X[:, j].T @ X) @ mat / L[j] +
                          rho * mat[j, :]) / (1 + rho)
    if sym:
        for j in list(range(n_features))[::-1]:
            mat[j, :] -= (X[:, j].T @ X) @ mat / L[j]
    return mat


def get_cd_mat_dual(X):
    n_features = X.shape[1]
    L = norm(X, axis=0) ** 2
    mat = np.eye(X.shape[0])
    for j in range(n_features):
        mat -= np.outer(X[:, j], X[:, j]) @ mat / L[j]
    return mat


def get_kaczmarz_mat(X):
    n_samples, n_features = X.shape
    mat = np.eye(n_features)
    for i in range(n_samples):
        mat -= np.outer(X[i, :], X[i, :] @ mat) / norm(X[i, :]) ** 2
    return mat

import scipy
import numpy as np
import seaborn as sns
from scipy.sparse import issparse

from numpy.linalg import norm
from scipy.linalg import toeplitz
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state


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


def power_method(A, max_iter=100, tol=1e-6):
    n, d = A.shape
    np.random.seed(1)
    u = np.random.randn(n)
    v = np.random.randn(d)
    spec_norm = 0
    for _ in range(max_iter):
        spec_norm_old = spec_norm
        u = A @ v
        u /= norm(u)
        v = A.T @ u
        spec_norm = norm(v)
        v /= spec_norm
        if np.abs(spec_norm - spec_norm_old) < tol:
            break
    return spec_norm


def fetch_leukemia():
    data = fetch_openml("leukemia")
    X = np.asfortranarray(data.data.astype(float))
    y = np.array([- 1. if y == "ALL" else 1. for y in data.target])
    return X, y


def make_sparse_data(n, p, rho=0.5, s=None, snr=None, w_type="randn"):
    """
    Generate X and y as follows:
    - A has shape (n, p), has Gaussain entries with Toeplitz correlation
        with parameter rho.
    - noisy or exact measurements: y = A @ w_true + Gaussian noise
    """
    assert w_type in ("ones", "randn")

    rng = check_random_state(seed=24)
    corr = rho ** np.arange(p)
    cov = toeplitz(corr)
    X = rng.multivariate_normal(np.zeros(p), cov, size=n)
    if s is None:
        s = p // 10
    supp = rng.choice(p, s, replace=False)
    w_true = np.zeros(p)

    if w_type == "randn":
        w_true[supp] = rng.randn(s)
    else:
        w_true[supp] = 1.

    y = X @ w_true
    if snr is not None:
        noise = rng.randn(n)
        y += noise / norm(noise) * norm(y) / snr
    return np.asfortranarray(X), y, w_true

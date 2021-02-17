import pytest
import numpy as np
from scipy import sparse
from numpy.linalg import norm

from sklearn.linear_model import ElasticNet

from andersoncd.lasso import solver_enet, apcg_enet
from andersoncd.data.synthetic import simu_linreg


l1_ratios = [0.9, 0.7, 0.6]


@pytest.mark.parametrize(
    "algo, use_acc", [
        ("cd", True), ("pgd", True), ("fista", False), ("apcg", False)])
@pytest.mark.parametrize("l1_ratio", l1_ratios)
def test_enet_solver(algo, use_acc, l1_ratio):
    X, y = simu_linreg(n_samples=30, n_features=40)
    n_samples = X.shape[0]

    alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
    alpha = 0.8 * alpha_max
    tol = 1e-14
    # enet from sklearn
    estimator = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, tol=tol)
    estimator.fit(X, y)
    coef_sk = estimator.coef_

    if algo == "apcg":
        coef_ours = apcg_enet(
            X, y, alpha * l1_ratio * n_samples,
            rho=alpha * (1 - l1_ratio) * n_samples, tol=tol, verbose=True,
            max_iter=100_000)[0]
    else:
        coef_ours = solver_enet(
            X, y, alpha * l1_ratio * n_samples,
            rho=alpha * (1 - l1_ratio) * n_samples, tol=tol, algo=algo,
            use_acc=use_acc, max_iter=1_000_000)[0]

    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)


@pytest.mark.parametrize("sparse_X", [False, True])
def test_apcg(sparse_X):
    X, y = simu_linreg(n_samples=30, n_features=40)
    if sparse_X:
        X = sparse.csc_matrix(X)

    tol = 1e-8
    alpha = np.max(np.abs(X.T @ y)) / 20
    w, E, gaps = apcg_enet(
        X, y, alpha, tol=tol, f_gap=50, max_iter=1000000, verbose=1)
    np.testing.assert_array_less(gaps[-1], tol)

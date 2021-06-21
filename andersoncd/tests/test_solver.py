import pytest
import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import Lasso, ElasticNet

from andersoncd.solver import solver
from andersoncd.data.synthetic import simu_linreg
from andersoncd.penalties import L1, L1_plus_L2

X, y = simu_linreg(n_samples=30, n_features=40)
n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = 0.05 * alpha_max
tol = 1e-14


@pytest.mark.parametrize("use_acc", [True, False])
def test_solver(use_acc):
    # lasso from sklearn
    estimator = Lasso(
        alpha=alpha, fit_intercept=False, tol=tol)
    estimator.fit(X, y)
    coef_sk = estimator.coef_

    penalty = L1(alpha)
    w = np.zeros(n_features)
    R = y - X @ w
    norms_X_col = norm(X, axis=0)
    coef_ours = solver(
        X, y, penalty, w, R, norms_X_col, verbose=2, max_iter=10,
        max_epochs=1_000, tol=1e-10)[0]

    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)


@pytest.mark.parametrize("use_acc", [True, False])
def test_solver2(use_acc=True):
    # elastic net from sklearn
    estimator = ElasticNet(
        alpha=2 * alpha, l1_ratio=0.5, fit_intercept=False, tol=tol)
    estimator.fit(X, y)
    coef_sk = estimator.coef_

    penalty = L1_plus_L2(alpha, alpha)
    w = np.zeros(n_features)
    R = y - X @ w
    norms_X_col = norm(X, axis=0)
    coef_ours = solver(
        X, y, penalty, w, R, norms_X_col, verbose=2, max_iter=20,
        max_epochs=1_000, tol=1e-10)[0]

    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)


if __name__ == '__main__':
    test_solver2(True)

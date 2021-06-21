import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import ElasticNet

from andersoncd.solver import solver
from andersoncd.data.synthetic import simu_linreg
from andersoncd.penalties import L1


def test_enet_solver(use_acc=True):
    X, y = simu_linreg(n_samples=30, n_features=40)
    n_samples, n_features = X.shape

    alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
    alpha = 0.8 * alpha_max
    tol = 1e-14
    # enet from sklearn
    estimator = ElasticNet(
        alpha=alpha, l1_ratio=1, fit_intercept=False, tol=tol)
    estimator.fit(X, y)
    coef_sk = estimator.coef_

    penalty = L1(alpha)
    w = np.zeros(n_features)
    R = y.copy() - X @ w
    norms_X_col = norm(X, axis=0)
    coef_ours = solver(
        X, y, penalty, w, R, norms_X_col, max_iter=10_000)[0]

    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)


if __name__ == '__main__':
    test_enet_solver(True)

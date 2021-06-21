from andersoncd import estimators
from itertools import product
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

dict_estimators_sk = {}
dict_estimators_sk["Lasso"] = Lasso(alpha=alpha, fit_intercept=False, tol=tol)
dict_estimators_sk["ElasticNet"] = ElasticNet(
    alpha=2 * alpha, l1_ratio=0.5, fit_intercept=False, tol=tol)

dict_penalties = {}
dict_penalties["Lasso"] = L1(alpha)
dict_penalties["ElasticNet"] = L1_plus_L2(alpha, alpha)


@pytest.mark.parametrize(
    "estimator_name, use_acc", product(["Lasso", "ElasticNet"], [True, False]))
def test_solver(estimator_name, use_acc):
    # lasso from sklearn
    estimator_sk = dict_estimators_sk[estimator_name]
    estimator_sk.fit(X, y)
    coef_sk = estimator_sk.coef_

    penalty = dict_penalties[estimator_name]
    w = np.zeros(n_features)
    R = y - X @ w
    norms_X_col = norm(X, axis=0)
    coef_ours = solver(
        X, y, penalty, w, R, norms_X_col, verbose=2, max_iter=10,
        max_epochs=1_000, tol=1e-10)[0]

    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)


# @pytest.mark.parametrize("use_acc", [True, False])
# def test_solver2(use_acc=True):
#     # elastic net from sklearn
#     estimator = ElasticNet(
#         alpha=2 * alpha, l1_ratio=0.5, fit_intercept=False, tol=tol)
#     estimator.fit(X, y)
#     coef_sk = estimator.coef_

#     penalty = L1_plus_L2(alpha, alpha)
#     w = np.zeros(n_features)
#     R = y - X @ w
#     norms_X_col = norm(X, axis=0)
#     coef_ours = solver(
#         X, y, penalty, w, R, norms_X_col, verbose=2, max_iter=20,
#         max_epochs=1_000, tol=1e-10)[0]

#     np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)


if __name__ == '__main__':
    test_solver2(True)

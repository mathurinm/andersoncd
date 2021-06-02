import pytest
import numpy as np
from scipy import sparse
from numpy.linalg import norm

from sklearn.linear_model import ElasticNet, LinearRegression, Lasso

from andersoncd import WeightedLasso
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


def test_wlasso():
    X, y = simu_linreg(n_samples=100, n_features=40)
    # X /= norm(X, axis=0)
    y /= norm(y) / np.sqrt(len(y))
    alpha_max = np.max(np.abs(X.T @ y)) / len(y)
    alpha = alpha_max / 20

    # Lasso:
    clf = WeightedLasso(
        alpha=alpha, weights=np.ones(X.shape[1]),
        fit_intercept=False, tol=1e-10).fit(X, y)
    lasso = Lasso(alpha=alpha, tol=1e-10, fit_intercept=False).fit(X, y)
    np.testing.assert_allclose(clf.coef_, lasso.coef_, rtol=1e-5)

    clf = WeightedLasso(
        alpha=alpha,
        weights=np.zeros(X.shape[1]),
        max_epochs=200,
        max_iter=20, verbose=1, fit_intercept=False).fit(X, y)

    linreg = LinearRegression(fit_intercept=False).fit(X, y)
    np.testing.assert_allclose(clf.coef_, linreg.coef_)


if __name__ == '__main__':
    X, y = simu_linreg(n_samples=100, n_features=40)
    # X /= norm(X, axis=0)
    y /= norm(y) / np.sqrt(len(y))
    alpha_max = np.max(np.abs(X.T @ y)) / len(y)
    alpha = alpha_max / 20

    # Lasso:
    clf = WeightedLasso(
        alpha=alpha, weights=np.ones(X.shape[1]),
        fit_intercept=False, tol=1e-10).fit(X, y)
    lasso = Lasso(alpha=alpha, tol=1e-10, fit_intercept=False).fit(X, y)
    np.testing.assert_allclose(clf.coef_, lasso.coef_, rtol=1e-5)

    clf = WeightedLasso(
        alpha=alpha,
        weights=np.zeros(X.shape[1]),
        max_epochs=200,
        max_iter=20, verbose=1, fit_intercept=False).fit(X, y)

    linreg = LinearRegression(fit_intercept=False).fit(X, y)
    np.testing.assert_allclose(clf.coef_, linreg.coef_)

    # weights[:10] = 0

    # clf = WeightedLasso(
    #     alpha=alpha,
    #     weights=weights,
    #     max_epochs=50,
    #     max_iter=10, verbose=2, fit_intercept=False).fit(X, y)

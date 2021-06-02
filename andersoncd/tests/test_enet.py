import pytest
import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import ElasticNet, Lasso

from andersoncd import WeightedLasso
from andersoncd.lasso import solver_enet
from andersoncd.data.synthetic import simu_linreg


l1_ratios = [0.9, 0.7, 0.6]


@pytest.mark.parametrize(
    "algo, use_acc", [
        ("cd", True), ("pgd", True), ("fista", False)])
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

    coef_ours = solver_enet(
        X, y, alpha * l1_ratio * n_samples,
        rho=alpha * (1 - l1_ratio) * n_samples, tol=tol, algo=algo,
        use_acc=use_acc, max_iter=10_000)[0]

    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)


def test_wlasso():
    X, y = simu_linreg(n_samples=50, n_features=200)
    y /= norm(y) / np.sqrt(len(y))
    alpha_max = np.max(np.abs(X.T @ y)) / len(y)
    alpha = alpha_max / 10

    # Compare Lasso to sklearn:
    clf = WeightedLasso(
        alpha=alpha, weights=np.ones(X.shape[1]),
        fit_intercept=False, tol=1e-14, verbose=2).fit(X, y)
    lasso = Lasso(alpha=alpha, tol=1e-14, fit_intercept=False).fit(X, y)
    np.testing.assert_allclose(clf.coef_, lasso.coef_, rtol=1e-5, atol=1e-5)

    # Linear regression, check that residuals vanish
    clf = WeightedLasso(
        alpha=alpha,
        weights=np.zeros(X.shape[1]),
        max_epochs=200,
        max_iter=20, verbose=2, tol=1e-12, fit_intercept=False).fit(X, y)

    np.testing.assert_allclose(0, norm(y - clf.predict(X)), atol=1e-8)

    # test mixture of both zero and non zero weights
    weights = np.abs(np.random.randn(X.shape[1]))
    alpha_max = np.max(np.abs(X[:, weights != 0].T @ y)) / len(y)
    alpha = alpha_max / 10
    weights[:10] = 0
    clf = WeightedLasso(
        alpha=alpha, weights=weights, fit_intercept=False,
        verbose=2, tol=1e-12).fit(X, y)
    # TODO design test, with KKT?


if __name__ == '__main__':
    pass

import pytest
import numpy as np
from numpy.linalg import norm

from celer import LogisticRegression

from andersoncd.logreg import solver_logreg, apcg_logreg


pCmins = [2, 5, 10]
algos = [("cd", True), ("pgd", True), ("fista", False), ("apcg", False)]


@pytest.mark.parametrize("algo, use_acc", algos)
@pytest.mark.parametrize("pCmin", pCmins)
def test_logreg_solver(algo, use_acc, pCmin):
    # data generation
    np.random.seed(0)
    n_samples = 11
    n_features = 3
    X = np.random.randn(n_samples, n_features)
    y = np.sign(X @ np.random.randn(n_features))

    C_min = 2 / norm(X.T @ y, ord=np.inf)
    C = pCmin * C_min
    tol = 1e-14
    estimator = LogisticRegression(
        C=C, verbose=1, solver="celer-pn", fit_intercept=False, tol=tol)
    estimator.fit(X, y)
    coef_celer = estimator.coef_.ravel()

    if algo == "apcg":
        coef_ours, _, _ = apcg_logreg(
            X, y, alpha=1/C, tol=tol, verbose=True, max_iter=2_00_000)
    else:
        coef_ours = solver_logreg(
            X, y, alpha=1/C,
            tol=tol, algo=algo, use_acc=use_acc, max_iter=20000)[0]
    np.testing.assert_allclose(coef_ours, coef_celer, atol=1e-10)


def test_apcg():
    np.random.seed(0)
    n_samples = 30
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    y = np.sign(X @ np.random.randn(n_features))
    alpha = np.max(np.abs(X.T @ y)) / 100
    tol = 1e-7
    w, E, gaps = apcg_logreg(X, y, alpha, tol=tol,
                             verbose=True, max_iter=1_000_000)
    np.testing.assert_array_less(gaps[-1], tol)


if __name__ == '__main__':
    for pCmin in pCmins:
        for algo, use_acc in algos:
            test_logreg_solver(algo, use_acc, pCmin)

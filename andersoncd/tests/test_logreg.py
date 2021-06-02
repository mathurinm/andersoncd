import pytest
import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import LogisticRegression

from andersoncd.logreg import solver_logreg


pCmins = [2, 5, 10]
algos = [("cd", True), ("pgd", True), ("fista", False)]


@pytest.mark.parametrize("algo, use_acc", algos)
@pytest.mark.parametrize("pCmin", pCmins)
def test_logreg_solver(algo, use_acc, pCmin):
    # data generation
    np.random.seed(0)
    n_samples = 30
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    y = np.sign(X @ np.random.randn(n_features))

    C_min = 2 / norm(X.T @ y, ord=np.inf)
    C = pCmin * C_min
    tol = 1e-14
    estimator = LogisticRegression(
        C=C, verbose=1, solver='liblinear', penalty='l1', fit_intercept=False,
        tol=tol)
    estimator.fit(X, y)
    coef_sklearn = estimator.coef_.ravel()

    coef_ours = solver_logreg(
        X, y, alpha=1/C,
        tol=tol, algo=algo, use_acc=use_acc, max_iter=20000)[0]
    np.testing.assert_allclose(coef_ours, coef_sklearn, atol=1e-7)


if __name__ == '__main__':
    pass

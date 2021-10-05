import pytest
import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import Lasso as Lasso_sklearn

from scipy.sparse import csc_matrix

from andersoncd.data import make_correlated_data
from andersoncd.solver_lasso import numba_celer_primal


X, y, _ = make_correlated_data(
    n_samples=200, n_features=500, density=0.1, random_state=0)

np.random.seed(0)
X_sparse = csc_matrix(X * np.random.binomial(1, 0.1, X.shape))

n_samples, n_features = X.shape
# Lasso will fit with binary values, but else logreg's alpha_max is wrong:
y = np.sign(y)
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = 0.1 * alpha_max
tol = 1e-10
l1_ratio = 0.3


@pytest.mark.parametrize('X', [X, X_sparse])
def test_estimator(X):
    estimator_sk = Lasso_sklearn(
        alpha=alpha, fit_intercept=False, tol=tol, max_iter=3000)

    estimator_sk.fit(X, y)
    coef_sk = estimator_sk.coef_

    coef_cl_numba = numba_celer_primal(
        X, y, alpha, 100, p0=10, tol=1e-12, prune=True,
        gap_freq=10, max_epochs=100_000)

    np.testing.assert_allclose(coef_sk, coef_cl_numba, rtol=1e-6)


if __name__ == '__main__':
    test_estimator(X)

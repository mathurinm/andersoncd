import pytest
import numpy as np
from numpy.linalg import norm

from celer import GroupLasso

from andersoncd.group import solver_group


p_alphas = np.geomspace(1, 1 / 100, num=5)


@pytest.mark.parametrize(
    "algo, use_acc", [("bcd", True), ("pgd", True), ("fista", False)])
@pytest.mark.parametrize("p_alpha", p_alphas)
def test_group_solver(algo, use_acc, p_alpha):

    np.random.seed(0)
    X = np.random.randn(30, 50)
    grp_size = 5
    y = X @ np.random.randn(X.shape[1])

    alpha_max = np.max(norm((X.T @ y).reshape(-1, grp_size), axis=1))
    alpha = alpha_max * p_alpha

    tol = 1e-12

    # our solver
    w = solver_group(
        X, y, alpha, grp_size, algo=algo, max_iter=20000, tol=tol,
        f_gap=100, use_acc=use_acc)[0]

    # alternative solver
    estimator = GroupLasso(
        groups=grp_size, alpha=alpha/len(y), fit_intercept=False, tol=tol)
    estimator.fit(X, y)

    np.testing.assert_allclose(w, estimator.coef_, rtol=1e-6)

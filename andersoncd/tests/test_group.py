import pytest
import numpy as np
from numpy.linalg import norm

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
    w, E, gaps = solver_group(
        X, y, alpha, grp_size, algo=algo, max_iter=20000, tol=tol,
        f_gap=100, use_acc=use_acc)

    np.testing.assert_array_less(gaps[-1], tol)

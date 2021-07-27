import pytest
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa E402
from numpy.linalg import norm  # noqa E402

from andersoncd import Lasso  # noqa E402
from andersoncd.data import make_correlated_data  # noqa E402


@pytest.mark.parametrize("alpha_div", [10, 50])
def test_Lasso(benchmark, alpha_div, n_reps=10):
    X, y, _ = make_correlated_data(
        n_samples=500, n_features=1000, density=0.1)

    # alpha_div = 50
    alpha = norm(X.T @ y, np.inf) / len(y) / alpha_div

    # cache numba compil
    us = Lasso(alpha=alpha, fit_intercept=False,
               verbose=0,
               warm_start=False).fit(X, y)

    @benchmark
    def fit():
        for _ in range(n_reps):
            us.fit(X, y)

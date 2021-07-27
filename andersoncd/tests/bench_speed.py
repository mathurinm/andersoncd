import time
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa E402
from numpy.linalg import norm  # noqa E402

from andersoncd import Lasso  # noqa E402
from andersoncd.data import make_correlated_data  # noqa E402


def test_speed(benchmark):
    # X, y, _ = make_correlated_data(
    #     n_samples=500, n_features=500, density=0.1)

    # alpha_div = 50
    # alpha = norm(X.T @ y, np.inf) / len(y) / alpha_div

    # cache numba compil
    # us = Lasso(alpha=alpha, fit_intercept=False,
    #            verbose=0,
    #            warm_start=False).fit(X, y)

    @benchmark
    def fit():
        for _ in range(1):
            time.sleep(0.2)
        # us.fit(X, y)


def test_other_speed(benchmark):
    # X, y, _ = make_correlated_data(
    #     n_samples=500, n_features=500, density=0.1)

    # alpha_div = 50
    # alpha = norm(X.T @ y, np.inf) / len(y) / alpha_div

    # cache numba compil
    # us = Lasso(alpha=alpha, fit_intercept=False,
    #            verbose=0,
    #            warm_start=False).fit(X, y)

    @benchmark
    def fit():
        for _ in range(1):
            time.sleep(0.5)

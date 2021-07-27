import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import time  # noqa E402
import numpy as np  # noqa E402
from numpy.linalg import norm  # noqa E402

from andersoncd import Lasso  # noqa E402
from andersoncd.data import make_correlated_data  # noqa E402


X, y, w_true = make_correlated_data(
    n_samples=500, n_features=2000, density=0.1)


alpha_div = 50
alpha = norm(X.T @ y, np.inf) / len(y) / alpha_div


us = Lasso(alpha=alpha, fit_intercept=False,
           verbose=0,
           warm_start=False)

all_times = []
for _ in range(20):
    t0 = time.time()
    us.fit(X, y)
    all_times.append(time.time() - t0)


print("Config:")
print(f"    (n, p) = :{X.shape}, alpha_max/{alpha_div}")
print("Time:")
print(f"{np.mean(all_times[1:]):.3f} +- {np.std(all_times[1:]):.3f} s")

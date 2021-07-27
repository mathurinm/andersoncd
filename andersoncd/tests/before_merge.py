import time
import numpy as np
from numpy.linalg import norm

from andersoncd import Lasso
from andersoncd.data import make_correlated_data


X, y, w_true = make_correlated_data(
    n_samples=100, n_features=500, density=0.1)


alpha_div = 50
alpha = norm(X.T @ y, np.inf) / len(y) / alpha_div


us = Lasso(alpha=alpha, fit_intercept=False,
           verbose=0,
           warm_start=False)

all_times = []
for _ in range(10):
    t0 = time.time()
    us.fit(X, y)
    all_times.append(time.time() - t0)


print("Config:")
print(f"    (n, p) = :{X.shape}, alpha_max/{alpha_div}")
print("Time:")
print(f"{np.mean(all_times[1:]):.3f} +- {np.std(all_times[1:]):.3f} s")

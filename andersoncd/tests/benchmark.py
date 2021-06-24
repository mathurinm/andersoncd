import numpy as np
import time
from sklearn.linear_model import Lasso

from andersoncd import WeightedLasso
from andersoncd.data import make_correlated_data


X, y, _ = make_correlated_data(n_samples=1000, n_features=2000)


weights = np.ones(X.shape[1])
alpha = np.max(np.abs(X.T @ y)) / len(y) / 50

us = WeightedLasso(alpha=alpha, weights=weights, fit_intercept=False,
                   max_epochs=20, max_iter=1, verbose=2,
                   warm_start=False).fit(X, y)

us.max_epochs = 10000
us.max_iter = 20

t0 = time.time()
us.fit(X, y)
t_us = time.time() - t0


sk = Lasso(alpha=alpha, fit_intercept=False)

t0 = time.time()
sk.fit(X, y)
t_sk = time.time() - t0

print(f'Us: {t_us:.4f} s')
print(f'sk: {t_sk:.4f} s')

import numpy as np
import time

from andersoncd import WeightedLasso
from andersoncd.data import make_correlated_data


X, y, _ = make_correlated_data(n_samples=100, n_features=200)


weights = np.ones(X.shape[1])
alpha = np.max(np.abs(X.T @ y)) / len(y) / 50

clf = WeightedLasso(alpha=alpha, fit_intercept=False,
                    max_epochs=20, max_iter=1, verbose=2).fit(X, y)

t0 = time.time()
clf.fit(X, y)
t_us = time.time() - t0

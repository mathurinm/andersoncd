"""Importance of feature normalization when penalizing."""
# Author: Mathurin Massias
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from andersoncd import Lasso, WeightedLasso
from andersoncd.data import make_correlated_data

n_features = 50
X, _, _ = make_correlated_data(n_samples=50, n_features=n_features)
w_true = np.zeros(n_features)
w_true[:10] = 1

# important features have a smaller norm than the other
X[:, :10] *= 0.1

y = X @ w_true + 3. * np.random.randn(X.shape[0])

alpha_max = np.max(np.abs(X.T @ y)) / len(y)
alpha = alpha_max / 10

las = Lasso(alpha=alpha, fit_intercept=False).fit(X, y)
wei = WeightedLasso(alpha=alpha, weights=norm(X, axis=0)).fit(X, y)

fig, axarr = plt.subplots(1, 2)
axarr[0].stem(las.coef_)
axarr[1].stem(wei.coef_)
plt.show(block=False)

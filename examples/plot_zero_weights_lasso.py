"""
=====================================
Weighted Lasso with some zero weights
=====================================

This example demonstrates how to use a weighted lasso with some vanishing
weights. The fast Celer solver is adapted to use primal Anderson acceleration,
allowing it to not compute the dual and handle 0 weights.
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import fetch_openml

from andersoncd import WeightedLasso
from andersoncd.plot_utils import configure_plt
from andersoncd.data import make_correlated_data

configure_plt(fontsize=14, poster=False)

dataset = "simu"

if dataset == "simu":
    n_features = 100
    X, y, _ = make_correlated_data(
        n_samples=100, n_features=n_features, corr=0.7)
else:
    X, y = fetch_openml("leukemia", return_X_y=True)
    X, y = X.to_numpy(), y.to_numpy()
    n_features = 100
    X /= norm(X, axis=0)
    X = X[:, :n_features]
    y = LabelBinarizer().fit_transform(y)[:, 0].astype(float)

weights = np.empty(n_features)
# unpenalize the first 10 features:
weights[:10] = 0
# put large penalty on the 10-50 features
weights[10:50] = 5
# put small penalty on last 50 features
weights[50:] = 1

alpha_max = np.max(np.abs(
    X[:, weights != 0].T @ y / weights[weights != 0])) / len(y)
clf = WeightedLasso(alpha=alpha_max/10, weights=weights,
                    fit_intercept=False, verbose=1, tol=1e-10).fit(X, y)


plt.figure(figsize=(5, 4))
plt.axvline(50, color='k', linestyle='--')
plt.axvline(10, color='k', linestyle='--')
plt.stem(clf.coef_)
plt.title("Estimated coefficients")
plt.show(block=False)

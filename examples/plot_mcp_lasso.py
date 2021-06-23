"""
=========================================
Value and proximal operators of penalties
=========================================

Illustrate the value and proximal operators of some sparse penalties.
"""
# Author: Mathurin Massias <mathurin.massias@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
from celer.datasets import make_correlated_data

from andersoncd import WeightedLasso, MCP


X, y, w_true = make_correlated_data(n_samples=40, n_features=20)

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / len(y)


lasso = WeightedLasso(weights=np.ones(
    X.shape[1]), fit_intercept=False, warm_start=True)
mcp = MCP(gamma=3, fit_intercept=False, verbose=1)
# mcp2 = MCP(gamma=2, fit_intercept=False, verbose=1)

coef_mcp = np.zeros([10, X.shape[1]])
coef_lasso = np.zeros([10, X.shape[1]])

for i, alpha in enumerate(np.geomspace(alpha_max, alpha_max/1000, num=10)):
    mcp.alpha = alpha
    mcp.fit(X, y)
    coef_mcp[i] = mcp.coef_

    lasso.alpha = alpha
    lasso.fit(X, y)
    coef_lasso[i] = lasso.coef_


fig, axarr = plt.subplots(2, 1, constrained_layout=True)
axarr[0].plot(coef_mcp)
# for i in range(n_feature)
axarr[0].plot(np.vstack([w_true] * 10), linestyle='--')
axarr[0].set_title("MCP path")

axarr[1].plot(coef_lasso)
axarr[1].plot(np.vstack([w_true] * 10), linestyle='--')
axarr[1].set_title("Lasso path")
plt.show(block=False)

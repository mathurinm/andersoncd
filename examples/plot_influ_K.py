"""
Influence of number of extrapolated points K
============================================

How many points must be extrapolated for optimal performance?
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import sparse
from numpy.linalg import norm
from celer.datasets import fetch_libsvm
from celer.plot_utils import configure_plt

from andersoncd.lasso import solver_enet


configure_plt()

###############################################################################
# Load the data:

dataset = "rcv1_train"
X, y = fetch_libsvm(dataset)
X = X[:, :1000]

X.multiply(1 / sparse.linalg.norm(X, axis=0))
y -= y.mean()
y /= norm(y)


###############################################################################
# Solve the problem for various values of K

alpha = 0
tol = 1e-15
max_iter = 1000
f_gap = 10

K_list = [0, 2, 3, 4, 5, 10, 20]

dict_Es = {}

for K in K_list:
    print("Running CD Anderson with K=%d" % K)
    use_acc = K != 0
    w, E, _ = solver_enet(
        X, y, alpha=alpha, f_gap=f_gap, max_iter=max_iter, tol=tol,
        algo="cd", use_acc=use_acc, K=K)
    dict_Es[K] = E


###############################################################################
# Plot results

palette = sns.color_palette("colorblind")
p_star = np.inf
for E in dict_Es.values():
    p_star = min(p_star, min(E))


fig, ax = plt.subplots(figsize=[9.3, 5.6])
for i, K in enumerate(K_list):
    E = dict_Es[K]
    if K == 0:
        label = "CD, no acc"
        linestyle = 'solid'
        color = palette[1]
    else:
        label = "CD, K=%i" % K
        linestyle = 'dashed'
        color = plt.cm.viridis(i / len(K_list))
    ax.semilogy(
        f_gap * np.arange(len(E)), E - p_star,
        label=label, color=color, linestyle=linestyle)


ax.set_xlabel(r"iteration $k$")
ax.set_yticks((1e-15, 1e-10, 1e-5, 1))
ax.set_ylabel(r"$f(x^{(k)}) - f(x^*)$")
plt.tight_layout()

plt.legend()
plt.show(block=False)

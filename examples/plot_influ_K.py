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
from libsvmdata import fetch_libsvm

from andersoncd.plot_utils import configure_plt, _plot_legend_apart
from andersoncd.lasso import solver_enet

save_fig = False
# save_fig = True

configure_plt()

###############################################################################
# Load the data:

dataset = "rcv1.binary"
X, y = fetch_libsvm(dataset)
X = X[:, :1000]

X.multiply(1 / sparse.linalg.norm(X, axis=0))
y -= y.mean()
y /= norm(y)


###############################################################################
# Solve the problem for various values of K

alpha = 0
tol = 1e-15
max_iter = 5_000
f_gap = 10

K_list = [0, 2, 3, 4, 5, 10, 20]

dict_Es = {}
dict_times = {}

for K in K_list:
    print("Running CD Anderson with K=%d" % K)
    use_acc = K != 0
    _, E, _, times = solver_enet(
        X, y, alpha=alpha, f_gap=f_gap, max_iter=max_iter, tol=tol,
        algo="cd", use_acc=use_acc, K=K, compute_time=True, verbose=True)
    dict_Es[K] = E
    dict_times[K] = times

###############################################################################
# Plot results

palette = sns.color_palette("colorblind")
p_star = np.inf
for E in dict_Es.values():
    p_star = min(p_star, min(E))


fig, ax = plt.subplots(figsize=[9.3, 5.6])
for i, K in enumerate(K_list):
    times = dict_times[K]
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
        times, E - p_star,
        # f_gap * np.arange(len(E)), E - p_star,
        label=label, color=color, linestyle=linestyle)


ax.set_xlabel(r"Times (s)")
ax.set_yticks((1e-15, 1e-10, 1e-5, 1))
ax.set_ylabel(r"Suboptimality")
plt.xlim((0, 6))
plt.ylim(1e-15, 1)
plt.tight_layout()

if save_fig:
    fig_dir = ""
    fig_dir_svg = ""
    fig.savefig(
        "%sinflu_n_acc_K_time.pdf" % fig_dir, bbox_inches="tight")
    fig.savefig(
        "%sinflu_n_acc_K_time.svg" % fig_dir_svg, bbox_inches="tight")
    _plot_legend_apart(
        ax, "%sinflu_n_acc_K_time_legend.pdf" % fig_dir, ncol=3)

plt.legend()
plt.show(block=False)

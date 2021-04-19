"""
Comparison of CD, GD, inertial and Anderson acceleration
========================================================

Coordinate descent outperforms gradient descent, and Anderson acceleration
outperforms inertial acceleration.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from numpy.linalg import norm
from scipy.sparse.linalg import cg
from libsvmdata import fetch_libsvm

from andersoncd.lasso import solver_enet, primal_enet, apcg_enet
from andersoncd.plot_utils import (
    configure_plt, _plot_legend_apart, dict_color)

dict_algo_name = {}
dict_algo_name["pgd", False] = "GD (Gradient Descent)"
dict_algo_name["cd", False] = "CD (Coordinate Descent)"
dict_algo_name["bcd", False] = "BCD"
dict_algo_name["pgd", True] = "GD - Anderson"
dict_algo_name["cd", True] = "CD - Anderson"
dict_algo_name["bcd", True] = "BCD - Anderson"
dict_algo_name["rcd", False] = "RCD (Randomized)"
dict_algo_name["rbcd", False] = "RBCD"
dict_algo_name["fista", False] = "GD - inertial"
dict_algo_name["apcg", False] = "CD - inertial"

save_fig = False

configure_plt()

###############################################################################
# Load the data:

n_features = 1000
X, y = fetch_libsvm('rcv1.binary', normalize=True)
X = X[:, :n_features]

X.multiply(1 / sparse.linalg.norm(X, axis=0))
y -= y.mean()
y /= norm(y)

# conjugate gradient competitor:
E_cg = []
E_cg.append(norm(y) ** 2 / 2)
times_cg = []
times_cg.append(0)

t_start = time.time()


def callback(x):
    pobj = primal_enet(y - X @ x, x, 0)
    E_cg.append(pobj)
    print(pobj)
    times_cg.append(time.time() - t_start)


w_star = cg(
    X.T @ X, X.T @ y, callback=callback, maxiter=1_000, tol=1e-32)[0]
E_cg = np.array(E_cg)


###############################################################################
# Run algorithms:

# solvers parameters:
alpha = 0  # Least Squares
tol = 1e-13
max_iter = 100_000
f_gap = 1

all_algos = [
    ('pgd', False),
    ('pgd', True),
    ('fista', False),
    ('cd', False),
    ('cd', True),
    ('apcg', False),
    ('rcd', False)
]

dict_Es = {}
dict_times = {}

tmax = 2

for algo in all_algos:
    print("Running ", dict_algo_name[algo])
    if algo[0] == 'apcg':
        _, E, _, times = apcg_enet(
            X, y, alpha, max_iter=max_iter, tol=tol, f_gap=f_gap,
            verbose=True, compute_time=True, tmax=tmax)
    else:
        _, E, _, times = solver_enet(
            X, y, alpha=alpha, f_gap=f_gap, max_iter=max_iter, tol=tol,
            algo=algo[0], use_acc=algo[1], verbose=True, compute_time=True,
            tmax=tmax)
    dict_Es[algo] = E.copy()
    dict_times[algo] = times.copy()


p_star = primal_enet(y - X @ w_star, w_star, alpha)
for E in dict_Es.values():
    p_star = min(p_star, min(E))

###############################################################################
# Plot convergence curves:
plt.close('all')
fig, ax = plt.subplots(figsize=(9, 6))
fig_times, ax_times = plt.subplots(figsize=(9, 4))


for algo in all_algos:
    E = dict_Es[algo]
    times = dict_times[algo]
    use_acc = algo[1]
    if use_acc:
        linestyle = 'dashed'
    elif algo[0].startswith(('fista', 'apcg')):
        linestyle = 'dotted'
    elif algo[0].startswith('rcd'):
        linestyle = '-'
    else:
        linestyle = 'solid'

    ax.semilogy(
        f_gap * np.arange(len(E)), E - p_star,
        label=dict_algo_name[algo],
        color=dict_color[algo[0]], linestyle=linestyle)

    ax_times.semilogy(
        times, E - p_star,
        label=dict_algo_name[algo],
        color=dict_color[algo[0]], linestyle=linestyle)

ax.semilogy(
    np.arange(len(E_cg)), E_cg - p_star, label="Conjugate Gradient",
    color='black', linestyle='dashdot')

ax_times.semilogy(
    times_cg, E_cg - p_star, label="Conjugate Gradient",
    color='black', linestyle='dashdot')


ax.set_yticks((1e-15, 1e-10, 1e-5, 1e0))
ax_times.set_yticks((1e-15, 1e-10, 1e-5, 1e0))

ax.set_title("Convergence on Least Squares")

fontsize = 25
# ax_times.set_ylabel("OLS \n rcv1", fontsize=fontsize)
ax_times.set_ylabel(r"Suboptimality", fontsize=fontsize)
ax_times.set_xlabel("Time (s)", fontsize=fontsize)
ax_times.set_xlim((0, tmax))
ax_times.set_ylim((1e-16, 1))
ax_times.tick_params(axis='x', labelsize=35)
ax_times.tick_params(axis='y', labelsize=35)

fig.tight_layout()
fig_times.tight_layout()

if save_fig:
    all_fig_dir = [
        "",
        ""
    ]
    for fig_dir in all_fig_dir:
        _plot_legend_apart(
            ax_times, "%senergies_time_ols_legend.pdf" % fig_dir, ncol=3)
        _plot_legend_apart(
            ax_times, "%senergies_time_ols_legend.svg" % fig_dir, ncol=3)


fig.show()
fig_times.show()

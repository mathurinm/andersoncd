"""
====================================
Plot Anderson CD for the group Lasso
====================================

This example shows the performance of Anderson acceleration
of coordinate descent for the group Lasso.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from andersoncd.group import solver_group

from collections import defaultdict

from andersoncd.plot_utils import (
    configure_plt, _plot_legend_apart, dict_algo_name, dict_color)
from extra.utils import fetch_leukemia
from scipy import sparse

save_fig = True
# save_fig = False

configure_plt()

X, y, = fetch_leukemia()
X = X[:, :7120]
# np.random.seed(0)
# X = np.random.randn(30, 50)
if not sparse.issparse(X):
    y -= y.mean()
    X -= np.mean(X, axis=0)[None, :]
    X /= norm(X, axis=0)[None, :]
else:
    # pass
    X.multiply(1 / sparse.linalg.norm(X, axis=0))
    y -= y.mean()
    y /= norm(y)

grp_size = 5
y = X @ np.random.randn(X.shape[1])


alpha_max = np.max(norm((X.T @ y).reshape(-1, grp_size), axis=1))
alpha = alpha_max / 10


tol = 1e-32

max_iter = 500
# solver_group(
#     X, y, alpha, grp_size, max_iter=max_iter, tol=tol, algo="pgd", f_gap=100,
#     use_acc=True)


E = defaultdict(lambda: dict())
dict_times = defaultdict(lambda: dict())

all_algos = [
    ('bcd', True),
    ('pgd', True),
    ('bcd', False),
    ('pgd', False),
    ('rbcd', False),
    ('fista', False),
]

for algo in all_algos:
    acc = algo[1]
    _, E[algo[0]][algo[1]], _, dict_times[algo[0]][algo[1]] = solver_group(
        X, y, alpha, grp_size, max_iter=max_iter, tol=tol, algo=algo[0],
        f_gap=5, use_acc=algo[1], compute_time=True)


p_star = E["bcd"][False][-1]
E0 = E["bcd"][True][0]

plt.close('all')
plt.figure()
fig, ax = plt.subplots(figsize=(9, 4))
# for algo in algos:
for algo in all_algos:
    acc = algo[1]
    if acc:
        linestyle = 'dashed'
    elif algo[0].startswith(('fista', 'apcg')):
        linestyle = 'dotted'
    elif algo[0].startswith('rbcd'):
        linestyle = '-'
    else:
        linestyle = 'solid'
    ax.semilogy(
        (E[algo[0]][acc] - p_star) / E0, label=dict_algo_name[algo],
        color=dict_color[algo[0]], linestyle=linestyle)

ax.set_yticks((1e-15, 1e-10, 1e-5, 1e0))

fontsize = 30
# ax_times.set_ylabel("OLS \n rcv1", fontsize=fontsize)
ax.set_ylabel(r"Suboptimality")
ax.set_xlabel(r"Time (s)", fontsize=fontsize)
ax.set_xlim((0, 100))
ax.set_ylim((1e-15, 1))
ax.tick_params(axis='x', labelsize=35)
ax.tick_params(axis='y', labelsize=35)
fig.tight_layout()

if save_fig:
    fig_dir = "../../extrapol_cd/tex/aistats20/prebuiltimages/"
    fig_dir_svg = "../../extrapol_cd/tex/aistats20/images/"
    fig.savefig(
        "%senergies_group_time.pdf" % fig_dir, bbox_inches="tight")
    fig.savefig(
        "%senergies_group_time.svg" % fig_dir_svg, bbox_inches="tight")
    _plot_legend_apart(
        ax, "%senergies_group_time_legend.pdf" % fig_dir, ncol=3)


plt.legend()
plt.show(block=False)

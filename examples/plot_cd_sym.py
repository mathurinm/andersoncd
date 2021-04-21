
"""
Convergence of CD, pseudo-symmetric CD, and their Anderson versions
==================================================================

On least squares and logistic regression, performance of pseudo-symmetric
coordinate descent.
"""
from collections import defaultdict
import time

import numpy as np
import seaborn as sns
from scipy import sparse
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from scipy.optimize import fmin_l_bfgs_b
from libsvmdata import fetch_libsvm

from andersoncd.plot_utils import configure_plt, _plot_legend_apart
from andersoncd.utils import power_method
from andersoncd.lasso import solver_enet, primal_enet
from andersoncd.logreg import solver_logreg, primal_logreg


save_fig = False
# save_fig = True

configure_plt()

###############################################################################
# Load the data:
dataset = "real-sim"
n_features = 1000
X, y = fetch_libsvm(dataset)

X = X[:, :n_features]
X.multiply(1 / sparse.linalg.norm(X, axis=0))


###############################################################################
# Generate figures for both Least Squares and Logistic regression
for pb in ("ols", "logreg"):
    if pb == 'lasso':
        y -= y.mean()
        y /= norm(y)

    f_gap = 10
    tol = 1e-15
    max_iter = 1000

    t_start = time.time()
    t_optimal = []
    # run "best algorithm": conj. grad. for LS, LBFGS for logreg:
    E_optimal = []
    if pb == "logreg":
        rho = power_method(X) ** 2 / 100_000  # a bit of enet regularization
        E_optimal.append(np.log(2) * len(y))
        t_optimal.append(0)
        label_opt = "L-BFGS"
        tmax = 20
        t_start = time.time()

        def callback(x):
            pobj = primal_logreg(X @ x, y, x, 0, rho)
            E_optimal.append(pobj)
            t_optimal.append(time.time() - t_start)

        def obj(x):
            return np.log(1 + np.exp(- y * (X @ x))).sum() + rho * x @ x / 2

        def fprime(x):
            return - X.T @ (y / (1 + np.exp(y * (X @ x)))) + rho * x

        fmin_l_bfgs_b(obj, np.zeros(
            X.shape[1]), fprime=fprime, callback=callback, factr=0.01, pgtol=0,
            maxiter=max_iter)
    else:
        alpha = 0
        rho = 0  # no elastic net
        E_optimal.append(norm(y) ** 2 / 2)
        t_optimal.append(0)
        label_opt = "conjugate grad."
        t_start = time.time()
        tmax = 3

        def callback(x):
            pobj = primal_enet(y - X @ x, x, alpha)
            E_optimal.append(pobj)
            t_optimal.append(time.time() - t_start)

        cg(X.T @ X, X.T @ y, callback=callback, maxiter=max_iter, tol=1e-32)
    E_optimal = np.array(E_optimal)

    all_algos = [
        ('cd', False),
        ('cd', True),
        ('cdsym', False),
        ('cdsym', True),
        ('rcd', False),
    ]

    dict_coef = defaultdict(lambda: 1)
    dict_coef['cdsym'] = 2
    algo_names = {}
    algo_names["cd", False] = "CD"
    algo_names["rcd", False] = "RCD"
    algo_names["cdsym", False] = "CD sym"
    algo_names["cd", True] = "CD - Anderson"
    algo_names["cdsym", True] = "CD sym - Anderson"

    dict_Es = {}
    dict_times = {}

    for algo in all_algos:
        print("Running %s" % algo_names[algo])
        if pb == "ols":
            _, E, _, times = solver_enet(
                X, y, alpha=alpha, f_gap=f_gap,
                max_iter=int(max_iter/dict_coef[algo[0]]), tol=tol,
                algo=algo[0], use_acc=algo[1], compute_time=True, tmax=tmax)
        elif pb == "logreg":
            _, E, _, times = solver_logreg(
                X, y, alpha=alpha, rho=rho, f_gap=f_gap,
                max_iter=max_iter//dict_coef[algo[0]], tol=tol,
                algo=algo[0], use_acc=algo[1], compute_time=True, tmax=tmax)
        dict_Es[algo] = E
        dict_times[algo] = times

    current_palette = sns.color_palette("colorblind")
    dict_color = {}
    dict_color["cd"] = current_palette[1]
    dict_color['cdsym'] = current_palette[2]
    dict_color["rcd"] = current_palette[3]

    p_star = E_optimal[-1]
    for E in dict_Es.values():
        p_star = min(p_star, min(E))

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 5))
    for algo in all_algos:
        E = dict_Es[algo]
        use_acc = algo[1]
        linestyle = 'dashed' if use_acc else 'solid'
        ax.semilogy(
            dict_times[algo],
            # dict_coef[algo[0]] * f_gap * np.arange(len(E)),
            E - p_star,
            label=algo_names[algo],
            color=dict_color[algo[0]], linestyle=linestyle)

    ax.semilogy(
        t_optimal,
        # np.arange(len(E_optimal)),
        E_optimal - p_star,
        label=label_opt, color='black', linestyle='dashdot')

    dict_dataset = {}
    dict_dataset["rcv1.binary"] = "rcv1"
    dict_dataset["real-sim"] = "real_sim"  # use _ not - for latex
    dict_dataset["leukemia"] = "leukemia"

    str_info = "%s (%i st columns)" % (dataset, n_features)
    title = pb + str_info

    plt.ylabel(r"Suboptimality")
    plt.xlabel(r"Time (s)")
    plt.xlim((0, tmax))
    plt.ylim((1e-10, None))
    plt.tight_layout()

    if save_fig:
        fig_dir = ""
        fig_dir_svg = ""
        fig.savefig(
            "%senergies_cdsym_%s_time.pdf" % (fig_dir, pb),
            bbox_inches="tight")
        fig.savefig(
            "%senergies_cdsym_%s_time.svg" % (fig_dir, pb),
            bbox_inches="tight")
        _plot_legend_apart(
            ax, "%senergies_cdsym_%s_legend.pdf" % (fig_dir, pb), ncol=4)

    plt.legend()
    plt.title(title.replace('_', ' '))
    plt.show(block=False)

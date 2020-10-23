
"""
Convergence of CD, pseudo-symmetric CD, and their Anderson versions
==================================================================

On least squares and logistic regression, performance of pseudo-symmetric
coordinate descent.
"""
from collections import defaultdict

import numpy as np
import seaborn as sns
from scipy import sparse
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from scipy.optimize import fmin_l_bfgs_b
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import label_binarize
from celer.datasets import fetch_libsvm
from celer.plot_utils import configure_plt

from extracd.utils import power_method
from extracd.lasso import solver_enet, primal_enet
from extracd.logreg import solver_logreg, primal_logreg


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
    max_iter = 2000

    # run "best algorithm": conj. grad. for LS, LBFGS for logreg:
    E_optimal = []
    if pb == "logreg":
        rho = power_method(X) ** 2 / 100_000  # a bit of enet regularization
        E_optimal.append(np.log(2) * len(y))
        label_opt = "L-BFGS"

        def callback(x):
            pobj = primal_logreg(X @ x, y, x, 0, rho)
            E_optimal.append(pobj)

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
        label_opt = "conjugate grad."

        def callback(x):
            pobj = primal_enet(y - X @ x, x, alpha)
            E_optimal.append(pobj)
        cg(X.T @ X, X.T @ y, callback=callback, maxiter=max_iter, tol=1e-32)
    E_optimal = np.array(E_optimal)

    all_algos = [
        ('cd', False),
        ('cd', True),
        ('cdsym', False),
        ('cdsym', True),
    ]

    dict_coef = defaultdict(lambda: 1)
    dict_coef['cdsym'] = 2
    algo_names = {}
    algo_names["cd", False] = "CD"
    algo_names["cdsym", False] = "CD sym"
    algo_names["cd", True] = "CD - Anderson"
    algo_names["cdsym", True] = "CD sym - Anderson"

    dict_Es = {}

    for algo in all_algos:
        print("Running %s" % algo_names[algo])
        if pb == "ols":
            w, E, _ = solver_enet(
                X, y, alpha=alpha, f_gap=f_gap,
                max_iter=int(max_iter/dict_coef[algo[0]]), tol=tol,
                algo=algo[0], use_acc=algo[1])
        elif pb == "logreg":
            w, E, _ = solver_logreg(
                X, y, alpha=alpha, rho=rho, f_gap=f_gap,
                max_iter=max_iter//dict_coef[algo[0]], tol=tol,
                algo=algo[0], use_acc=algo[1])
        dict_Es[algo] = E

    current_palette = sns.color_palette("colorblind")
    dict_color = {}
    dict_color["cd"] = current_palette[1]
    dict_color['cdsym'] = current_palette[2]

    p_star = E_optimal[-1]
    for E in dict_Es.values():
        p_star = min(p_star, min(E))

    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, algo in enumerate(all_algos):
        E = dict_Es[algo]
        use_acc = algo[1]
        linestyle = 'dashed' if use_acc else 'solid'

        ax.semilogy(
            dict_coef[algo[0]] * f_gap * np.arange(len(E)), E - p_star,
            label=algo_names[algo],
            color=dict_color[algo[0]], linestyle=linestyle)

    ax.semilogy(
        np.arange(len(E_optimal)), E_optimal - p_star,
        label=label_opt, color='black', linestyle='dashdot')

    dict_dataset = {}
    dict_dataset["rcv1_train"] = "rcv1"
    dict_dataset["real-sim"] = "real_sim"  # use _ not - for latex
    dict_dataset["leukemia"] = "leukemia"

    str_info = "%s (%i st columns)" % (dataset, n_features)
    title = pb + str_info

    plt.ylabel(r"$f(x^{(k)}) - f(x^{*})$")
    plt.xlabel("nb gradient calls")
    plt.ylim((1e-10, None))
    plt.tight_layout()

    plt.legend()
    plt.title(title.replace('_', ' '))
    plt.show(block=False)

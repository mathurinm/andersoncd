
"""
Plot convergence of CD, pseudo-symmetric CD, and Anderson versions
==================================================================

TODO desc
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

########################################################################
pb = "logreg"
# pb = "lasso"

########################################################################
dataset = "rcv1_train"
# dataset = "real-sim"
# dataset = "leukemia"
########################################################################

if dataset == "real-sim":
    n_features = 2000
elif dataset == "rcv1_train":
    # n_features = 1000
    n_features = 5000
elif dataset == 'leukemia':
    n_features = 1000

# load and center data
if dataset == "leukemia":
    data = fetch_openml("leukemia")
    X = np.asfortranarray(data.data)
    y = 2 * label_binarize(data.target, classes=np.unique(data.target)) - 1
    y = y.squeeze().astype(float)
else:
    X, y = fetch_libsvm(dataset)

X = X[:, :n_features]
if not sparse.issparse(X):
    X -= np.mean(X, axis=0)[None, :]
    X /= norm(X, axis=0)[None, :]
else:
    X.multiply(1 / sparse.linalg.norm(X, axis=0))

if pb == 'lasso':
    y -= y.mean()
    y /= norm(y)

if dataset == 'real-sim':
    max_iter = 1_000
else:
    max_iter = 1_000

f_gap = 10
div_alpha = np.inf  # ols
tol = 1e-15


E_optimal = []
if pb == "logreg":
    rho = power_method(X) ** 2 / 100_000  # a bit of enet
    alpha_max = np.max(np.abs(X.T @ y)) / 2
    E_optimal.append(np.log(2) * len(y))
    label_opt = "L-BFGS"

    def callback(x):
        pobj = primal_logreg(X @ x, y, x, 0, rho)
        E_optimal.append(pobj)

    def obj(x):
        return np.log(1 + np.exp(- y * (X @ x))).sum() + rho * x @ x / 2

    def fprime(x):
        return - X.T @ (y / (1 + np.exp(y * (X @ x)))) + rho * x

    res = fmin_l_bfgs_b(obj, np.zeros(
        X.shape[1]), fprime=fprime, callback=callback, factr=0.01, pgtol=0,
        maxiter=max_iter)
else:
    alpha_max = np.max(np.abs(X.T @ y))
    alpha = alpha_max / div_alpha
    rho = 0  # no elastic net
    E_optimal.append(norm(y) ** 2 / 2)
    label_opt = "conjugate grad."

    def callback(x):
        pobj = primal_enet(y - X @ x, x, alpha)
        E_optimal.append(pobj)
    w_star = cg(
        X.T @ X, X.T @ y, callback=callback, maxiter=max_iter, tol=1e-32)[0]
E_optimal = np.array(E_optimal)

alpha = alpha_max / div_alpha


all_algos = [
    ('cd', False, None),
    ('cd', True, 5),
    ('cdsym', False, None),
    ('cdsym', True, 5),
]


dict_coef = defaultdict(lambda: 1)
dict_coef['cdsym'] = 2

dict_Es = {}


for algo in all_algos:
    if pb == "lasso":
        w, E, _ = solver_enet(
            X, y, alpha=alpha, f_gap=f_gap,
            max_iter=int(max_iter/dict_coef[algo[0]]), tol=tol, algo=algo[0],
            use_acc=algo[1], K=algo[2])
    elif pb == "logreg":
        w, E, _ = solver_logreg(
            X, y, alpha=alpha, rho=rho, f_gap=f_gap,
            max_iter=max_iter//dict_coef[algo[0]], tol=tol,
            algo=algo[0], use_acc=algo[1], K=algo[2])
    dict_Es[algo] = E


current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["cd"] = current_palette[1]
dict_color['cdsym'] = current_palette[2]

dict_algo_name = {}
dict_algo_name[False, "cd"] = "CD"
dict_algo_name[False, "cdsym"] = "CD sym"
dict_algo_name[True, "cd"] = "CD - Anderson"
dict_algo_name[True, "cdsym"] = "CD sym - Anderson"


p_star = E_optimal[-1]
for E in dict_Es.values():
    p_star = min(p_star, min(E))

plt.close('all')
fig, ax = plt.subplots(figsize=(10, 5))
for i, algo in enumerate(all_algos):
    E = dict_Es[algo]
    use_acc = algo[1]
    K = algo[2]
    if use_acc:
        linestyle = 'dashed'
    elif algo[0] == 'fista':
        linestyle = 'dotted'
    else:
        linestyle = 'solid'

    ax.semilogy(
        dict_coef[algo[0]] * f_gap * np.arange(len(E)), E - p_star,
        label=dict_algo_name[use_acc, algo[0]],
        color=dict_color[algo[0]], linestyle=linestyle)

ax.semilogy(
    np.arange(len(E_optimal)), E_optimal - p_star,
    label=label_opt, color='black', linestyle='dashdot')

dict_dataset = {}
dict_dataset["rcv1_train"] = "rcv1"
dict_dataset["real-sim"] = "real_sim"  # use _ not - for latex
dict_dataset["leukemia"] = "leukemia"

if div_alpha == np.inf:
    str_info = "%s %i st columns" % (dict_dataset[dataset], n_features)
    title = "OLS " + str_info if pb == "lasso" else "logreg " + str_info
else:
    title = r'%s, $\lambda = \lambda_{\mathrm{max}} / %s $' % (
        pb, div_alpha)

plt.ylabel(r"$f(x^{(k)}) - f(x^{*})$")
plt.xlabel("nb gradient calls")
plt.ylim((1e-10, None))
plt.tight_layout()

plt.legend()
plt.title(title.replace('_', ' '))
plt.show(block=False)

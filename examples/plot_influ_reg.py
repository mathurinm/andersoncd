"""
Plot influence of linear system regularization
==============================================

Why regularizing the linear system seems to hurt Anderson performance.
"""
from collections import defaultdict

import numpy as np
import seaborn as sns
from scipy import sparse
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from celer.datasets import fetch_libsvm
from celer.plot_utils import configure_plt

from extracd.lasso import solver_enet, primal_enet, apcg


configure_plt()

n_features = 1000
corr = 0.5
X, y = fetch_libsvm('rcv1_train')
X = X[:, :n_features]

if not sparse.issparse(X):
    y -= y.mean()
    X -= np.mean(X, axis=0)[None, :]
    X /= norm(X, axis=0)[None, :]
else:
    X.multiply(1 / sparse.linalg.norm(X, axis=0))
    y -= y.mean()
    y /= norm(y)

alpha_max = np.max(np.abs(X.T @ y))

alpha = 0
tol = 1e-15
E_cg = []
E_cg.append(norm(y) ** 2 / 2)

max_iter = 1500


def my_func(x):
    pobj = primal_enet(y - X @ x, x, alpha)
    E_cg.append(pobj)


w_star = cg(X.T @ X, X.T @ y, callback=my_func, maxiter=max_iter, tol=1e-32)[0]

E_cg = np.array(E_cg)


f_gap = 10
return_all = False


all_algos = [
    ('cd', False, None, None),
    ('cd', True, 6, 1e-3),
    ('cd', True, 6, 1e-4),
    ('cd', True, 6, 1e-5),
    ('cd', True, 6, 1e-10),
    ('cd', True, 6, None),
]

dict_coef = defaultdict(lambda: 1)
dict_coef['cdsym'] = 2
dict_coef['cd2'] = 2


dict_Es = {}

for algo in all_algos:
    if algo[0] == 'apcg':
        w, E, gaps = apcg(
            X, y, alpha, max_iter=max_iter, tol=tol, f_gap=f_gap)
    else:
        w, E, _ = solver_enet(
            X, y, alpha=alpha, f_gap=f_gap, max_iter=max_iter, tol=tol,
            return_all=return_all, algo=algo[0], use_acc=algo[1],
            K=algo[2], reg_amount=algo[3])
    dict_Es[algo] = E.copy()


current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["pgd"] = current_palette[0]
dict_color["fista"] = current_palette[0]
dict_color["cd"] = current_palette[1]
dict_color["apcg"] = current_palette[1]
dict_color['cdsym'] = current_palette[2]
dict_color["cd2"] = current_palette[3]

dict_algo_name = {}
dict_algo_name[False, "pgd"] = "GD"
dict_algo_name[False, "cd"] = "CD"
dict_algo_name[False, "cdsym"] = "CD SYM"
dict_algo_name[True, "pgd"] = "GD - Anderson"
dict_algo_name[True, "cd"] = "CD - Anderson"
dict_algo_name[True, "cdsym"] = "CD SYM - Anderson"
dict_algo_name[False, "fista"] = "GD - inertial"
dict_algo_name[False, "apcg"] = "CD - inertial"


p_star = primal_enet(y - X @ w_star, w_star, alpha)
for E in dict_Es.values():
    p_star = min(p_star, min(E))

###############################################################
# starts plot
current_palette = sns.color_palette("colorblind")

plt.close('all')
fig, ax = plt.subplots(figsize=(8, 5))

for i, algo in enumerate(all_algos):
    E = dict_Es[algo]
    use_acc = algo[1]
    K = algo[2]
    if use_acc:
        linestyle = 'dashed'
    elif algo[0].startswith(('fista', 'apcg')):
        linestyle = 'dotted'
    else:
        linestyle = 'solid'

    if not use_acc:
        label = "CD, no acc"
        color = current_palette[1]
    else:
        if algo[3] is None:
            label = r"$\lambda_{\mathrm{reg}} = 0$"
            color = current_palette[1]
        else:
            label = r"$\lambda_{\mathrm{reg}} = 10^{%d}$" % np.log10(algo[3])
            color = plt.cm.viridis((i + 1) / len(all_algos))
    ax.semilogy(
        dict_coef[algo[0]] * f_gap * np.arange(len(E)), E - p_star,
        label=label, color=color, linestyle=linestyle)


ax.set_yticks((1e-15, 1e-10, 1e-5, 1))
plt.ylabel(r"$f(x^{(k)}) - f(x^{*})$")
plt.xlabel(r"iteration $k$")
plt.tight_layout()
plt.legend()
plt.show(block=False)

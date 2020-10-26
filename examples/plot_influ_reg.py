"""
Influence of linear system regularization
=========================================

Why regularizing the linear system seems to hurt Anderson performance.
"""
import numpy as np
import seaborn as sns
from scipy import sparse
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from celer.datasets import fetch_libsvm
from celer.plot_utils import configure_plt

from andersoncd.lasso import solver_enet, primal_enet


configure_plt()

###############################################################################
# Load the data:
n_features = 1000
X, y = fetch_libsvm('rcv1_train')
X = X[:, :n_features]

X.multiply(1 / sparse.linalg.norm(X, axis=0))
y -= y.mean()
y /= norm(y)


###############################################################################
# Run solver with various regularization strengths:
alpha = 0
tol = 1e-15
f_gap = 10
max_iter = 1000

reg_amount_list = [1e-3, 1e-4, 1e-5, 1e-10, None]

dict_Es = {}

for reg_amount in reg_amount_list:
    w, E, _ = solver_enet(
        X, y, alpha=alpha, f_gap=f_gap, max_iter=max_iter, tol=tol,
        algo="cd", use_acc=True, reg_amount=reg_amount)
    dict_Es[reg_amount] = E.copy()

E_noacc = solver_enet(X, y, alpha=alpha, f_gap=f_gap, max_iter=max_iter,
                      algo="cd", use_acc=False)[1]

palette = sns.color_palette("colorblind")


p_star = np.inf
for E in dict_Es.values():
    p_star = min(p_star, min(E))

###############################################################################
# Plot results

plt.close('all')
fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(
    f_gap * np.arange(len(E_noacc)), E_noacc - p_star,
    label="CD, no acc.", color=palette[1])


for i, reg_amount in enumerate(reg_amount_list):
    E = dict_Es[reg_amount]

    if reg_amount is None:
        label = r"$\lambda_{\mathrm{reg}} = 0$"
        color = palette[1]
    else:
        label = r"$\lambda_{\mathrm{reg}} = 10^{%d}$" % np.log10(
            reg_amount)
        color = plt.cm.viridis((i + 1) / len(reg_amount_list))
    ax.semilogy(
        f_gap * np.arange(len(E)), E - p_star,
        label=label, color=color, linestyle="dashed")


ax.set_yticks((1e-15, 1e-10, 1e-5, 1))
plt.ylabel(r"$f(x^{(k)}) - f(x^{*})$")
plt.xlabel(r"iteration $k$")
plt.tight_layout()
plt.legend()
plt.show(block=False)

"""
Influence of linear system regularization
=========================================

Why regularizing the linear system seems to hurt Anderson performance.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from libsvmdata import fetch_libsvm

from andersoncd.plot_utils import configure_plt, _plot_legend_apart
from andersoncd.logreg import solver_logreg


configure_plt()

###############################################################################
# Load the data:
# n_features = 1000
X, y = fetch_libsvm('rcv1_train', normalize=True)


###############################################################################
# Run solver with various regularization strengths:
# alpha = 0
div_alpha = 30
alpha_max = np.max(np.abs(X.T @ y)) / 2
alpha = alpha_max / div_alpha

tol = 1e-10
f_gap = 10
max_iter = 600

reg_amount_list = [1e-3, 1e-4, 1e-5, 1e-7, None]

dict_Es = {}

for reg_amount in reg_amount_list:
    E = solver_logreg(
        X, y, alpha=alpha, f_gap=f_gap, max_iter=max_iter, tol=tol,
        algo="cd", use_acc=True, reg_amount=reg_amount, verbose=True)[1]
    dict_Es[reg_amount] = E.copy()

E_noacc = solver_logreg(
    X, y, alpha=alpha, f_gap=f_gap, max_iter=max_iter,
    algo="cd", use_acc=False, tol=tol, verbose=True)[1]

palette = sns.color_palette("colorblind")


p_star = np.inf
for E in dict_Es.values():
    p_star = min(p_star, min(E))

###############################################################################
# Plot results

plt.close('all')
fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogy(
    f_gap * np.arange(len(E_noacc)), (E_noacc - p_star) / E_noacc[0],
    label="PCD, no acc.", color=palette[1])


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
        f_gap * np.arange(len(E)), (E - p_star) / E[0],
        label=label, color=color, linestyle="dashed")


ax.set_yticks((1e-15, 1e-10, 1e-5, 1))
plt.ylabel(r"$f(x^{(k)}) - f(x^{*})$")
plt.xlabel(r"iteration $k$")
plt.xlim(0, 600)
plt.ylim(1e-15, 1)
plt.tight_layout()


save_fig = False
fig_dir = "../"
fig_dir_svg = "../"

if save_fig:
    fig.savefig(
        "%sinflu_reg_amount_logreg.pdf" % fig_dir, bbox_inches="tight")
    fig.savefig(
        "%sinflu_reg_amount_logreg.svg" % fig_dir_svg, bbox_inches="tight")
    fig = _plot_legend_apart(
        ax, "%sinflu_reg_amount_logreg_legend.pdf" % fig_dir, ncol=3)


plt.legend()
plt.show(block=False)
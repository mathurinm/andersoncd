import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from numpy.linalg import norm
from libsvmdata import fetch_libsvm
from andersoncd.data.real import load_openml
from andersoncd.plot_utils import configure_plt

from andersoncd.lasso import solver_enet, apcg_enet


configure_plt()

###############################################################################
# Load the data:

# n_features = 1000
# X, y = fetch_libsvm('rcv1_train', normalize=True)
# # X = X[:, :n_features]

# y -= y.mean()
# y /= norm(y)

X, y = load_openml("leukemia")

###############################################################################
# Run algorithms:

# solvers parameters:
div_alpha = 10
div_rho = 10
alpha_max = np.max(np.abs(X.T @ y))
alpha = alpha_max / div_alpha
rho = alpha / div_rho
# rho = 0

tol = 1e-20
max_iter = 20_000
f_gap = 1
all_algos = [
    # ('apcg', False),
    # ('pgd', False),
    # ('pgd', True),
    # ('fista', False),
    # ('cd', False),
    # ('rcd', False),
    ('cd', True),
]

dict_algo_name = {}
dict_algo_name["pgd", False] = "GD"
dict_algo_name["cd", False] = "CD"
dict_algo_name["rcd", False] = "RCD"
dict_algo_name["pgd", True] = "GD - Anderson"
dict_algo_name["cd", True] = "CD - Anderson"
dict_algo_name["fista", False] = "GD - inertial"
dict_algo_name["apcg", False] = "CD - inertial"

tmax = 2
# tmax =
dict_Es = {}
dict_times = {}

for algo in all_algos:
    print("Running ", dict_algo_name[algo])
    if algo[0] == 'apcg':
        _, E, _, times = apcg_enet(
            X, y, alpha, rho=rho, max_iter=max_iter, tol=tol,
            f_gap=f_gap, verbose=True, compute_time=True, tmax=tmax)
    else:
        _, E, gaps, times = solver_enet(
            X, y, alpha=alpha, rho=rho,
            f_gap=f_gap, max_iter=max_iter, tol=tol,
            algo=algo[0], use_acc=algo[1], verbose=True, compute_time=True,
            tmax=tmax, seed=0)
    dict_Es[algo] = E.copy()
    dict_times[algo] = times


current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["pgd"] = current_palette[0]
dict_color["fista"] = current_palette[0]
dict_color["cd"] = current_palette[1]
dict_color["rcd"] = current_palette[1]
dict_color["apcg"] = current_palette[1]


p_star = np.infty
for E in dict_Es.values():
    p_star = min(p_star, min(E))

###############################################################################
# Plot convergence curves:

# plt.figure()

fig, ax = plt.subplots(figsize=(9, 6))
fig2, ax2 = plt.subplots(figsize=(9, 6))

for algo in all_algos:
    E = dict_Es[algo]
    times = dict_times[algo]
    use_acc = algo[1]
    if use_acc:
        linestyle = 'dashed'
    elif algo[0].startswith(('fista', 'apcg')):
        linestyle = 'dotted'
    elif algo[0].startswith('rcd'):
        linestyle = (0, (3, 5, 1, 5, 1, 5))
    else:
        linestyle = 'solid'

    # if algo[0] == 'rcd':
    #     marker = "^"
    # else:
    #     marker = None
    ax.semilogy(
        f_gap * np.arange(len(E)), E - p_star, label=dict_algo_name[algo],
        color=dict_color[algo[0]], linestyle=linestyle)

    ax2.semilogy(
        times, E - p_star, label=dict_algo_name[algo],
        color=dict_color[algo[0]], linestyle=linestyle)


ax.set_ylabel(r"$f(x^{(k)}) - f(x^{*})$")
ax.set_xlabel(r"iteration $k$")

ax2.set_ylabel(r"$f(x^{(k)}) - f(x^{*})$")
ax2.set_xlabel("Time (s)")
ax.set_yticks((1e-15, 1e-10, 1e-5, 1e0))
ax2.set_yticks((1e-15, 1e-10, 1e-5, 1e0))
ax.set_title(r"Lasso $\lambda_{\max} / %i$" % div_alpha)
ax2.set_title(r"Lasso $\lambda_{\max} / %i$" % div_alpha)
fig.tight_layout()
fig2.tight_layout()
ax.legend()
fig.show()
fig2.show()

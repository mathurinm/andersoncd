"""
Comparison of CD, GD, inertial and Anderson acceleration
========================================================

Coordinate descent outperforms gradient descent, and Anderson acceleration
outperforms inertial acceleration.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from numpy.linalg import norm
from scipy.sparse.linalg import cg
from libsvmdata import fetch_libsvm
from andersoncd.plot_utils import configure_plt

from andersoncd.lasso import solver_enet, primal_enet, apcg


configure_plt()

###############################################################################
# Load the data:

n_features = 1000
X, y = fetch_libsvm('rcv1_train')
X = X[:, :n_features]

X.multiply(1 / sparse.linalg.norm(X, axis=0))
y -= y.mean()
y /= norm(y)

# conjugate gradient competitor:
E_cg = []
E_cg.append(norm(y) ** 2 / 2)


def callback(x):
    pobj = primal_enet(y - X @ x, x, 0)
    E_cg.append(pobj)


w_star = cg(
    X.T @ X, X.T @ y, callback=callback, maxiter=500, tol=1e-32)[0]
E_cg = np.array(E_cg)


###############################################################################
# Run algorithms:

# solvers parameters:
alpha = 0  # Least Squares
tol = 1e-13
max_iter = 2000
f_gap = 10

all_algos = [
    ('pgd', False),
    ('pgd', True),
    ('fista', False),
    ('cd', False),
    ('cd', True),
    ('apcg', False),
]

dict_algo_name = {}
dict_algo_name["pgd", False] = "GD"
dict_algo_name["cd", False] = "CD"
dict_algo_name["pgd", True] = "GD - Anderson"
dict_algo_name["cd", True] = "CD - Anderson"
dict_algo_name["fista", False] = "GD - inertial"
dict_algo_name["apcg", False] = "CD - inertial"


dict_Es = {}

for algo in all_algos:
    print("Running ", dict_algo_name[algo])
    if algo[0] == 'apcg':
        w, E, gaps = apcg(
            X, y, alpha, max_iter=max_iter, tol=tol, f_gap=f_gap,
            verbose=False)
    else:
        w, E, _ = solver_enet(
            X, y, alpha=alpha, f_gap=f_gap, max_iter=max_iter, tol=tol,
            algo=algo[0], use_acc=algo[1], verbose=False)
    dict_Es[algo] = E.copy()


current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["pgd"] = current_palette[0]
dict_color["fista"] = current_palette[0]
dict_color["cd"] = current_palette[1]
dict_color["apcg"] = current_palette[1]


p_star = primal_enet(y - X @ w_star, w_star, alpha)
for E in dict_Es.values():
    p_star = min(p_star, min(E))

###############################################################################
# Plot convergence curves:

plt.close('all')
fig, ax = plt.subplots(figsize=(9, 6))


for algo in all_algos:
    E = dict_Es[algo]
    use_acc = algo[1]
    if use_acc:
        linestyle = 'dashed'
    elif algo[0].startswith(('fista', 'apcg')):
        linestyle = 'dotted'
    else:
        linestyle = 'solid'

    ax.semilogy(
        f_gap * np.arange(len(E)), E - p_star,
        label=dict_algo_name[algo],
        color=dict_color[algo[0]], linestyle=linestyle)

ax.semilogy(
    np.arange(len(E_cg)), E_cg - p_star, label="conjugate grad.",
    color='black', linestyle='dashdot')

plt.ylabel(r"$f(x^{(k)}) - f(x^{*})$")
plt.xlabel(r"iteration $k$")
ax.set_yticks((1e-15, 1e-10, 1e-5, 1e0))
plt.title("Convergence on Least Squares")
plt.tight_layout()


plt.legend()
plt.show(block=False)

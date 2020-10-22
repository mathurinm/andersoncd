import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import sparse
from numpy.linalg import norm
from celer.datasets import fetch_libsvm
from celer.plot_utils import configure_plt

from extracd.lasso import solver_enet


configure_plt()

dataset = "rcv1_train"
X, y = fetch_libsvm(dataset)
X = X[:, :1000]

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
max_iter = 1500
f_gap = 10

dict_algo = {}

dict_algo = [
    ('cd', False, 0),
    ('cd', True, 2),
    ('cd', True, 3),
    ('cd', True, 4),
    ('cd', True, 5),
    ('cd', True, 10),
    ('cd', True, 20),
]

dict_coef = {}
dict_coef['gd'] = 1
dict_coef['cd'] = 1
dict_coef['cdsym'] = 2
dict_coef['cd2'] = 2


dict_Es = {}

for algo in dict_algo:
    w, E, _ = solver_enet(
        X, y, alpha=alpha, f_gap=f_gap, max_iter=max_iter, tol=tol,
        return_all=False, algo=algo[0], use_acc=algo[1],
        K=algo[2])
    dict_Es[algo] = E.copy()


current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["gd"] = current_palette[0]
dict_color["cd"] = current_palette[1]
dict_color['cdsym'] = current_palette[2]
dict_color["cd2"] = current_palette[3]


p_star = np.inf
for E in dict_Es.values():
    p_star = min(p_star, min(E))


fig, ax = plt.subplots(figsize=[9.3, 5.6])
for i, algo in enumerate(dict_algo):
    E = dict_Es[algo]
    K = algo[2]
    if K == 0:
        label = "CD, no acc"
        linestyle = 'solid'
        color = dict_color["cd"]
    else:
        label = "CD, K=%i" % (algo[2])
        linestyle = 'dashed'
        color = plt.cm.viridis(i / len(dict_algo))
    ax.semilogy(
        dict_coef[algo[0]] * f_gap * np.arange(len(E)), E - p_star,
        label=label, color=color, linestyle=linestyle)


ax.set_xlabel(r"iteration $k$")
ax.set_xlim(0, 1500)
ax.set_yticks((1e-15, 1e-10, 1e-5, 1))
ax.set_ylabel(r"$f(x^{(k)}) - f(x^*)$")
plt.tight_layout()

plt.legend()
plt.show(block=False)

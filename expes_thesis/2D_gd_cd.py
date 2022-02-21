from itertools import product
from operator import le
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import seaborn as sns

from andersoncd.plot_utils import configure_plt
from sparse_ho.utils_plot import discrete_color

# save_fig = True
save_fig = False

configure_plt(fontsize=18)
current_palette = sns.color_palette("colorblind")

dict_fig_dir = {}
dict_fig_dir["manuscript"] = "../manuscript/figures/intro/"
dict_fig_dir["slides"] = "../extrapol_cd/tex/slides_jds/prebuiltimages/"

n_features = 2
H = np.zeros((n_features, n_features))
H[0, 0] = 7
H[1, 1] = 8
H[0, 1] = 3
H[1, 0] = 3


_range = np.arange(n_features)
fixed_perm = np.random.choice(n_features, n_features, replace=False)
feats = {
    "CD - cyclic": lambda: _range,
    "CD - fixed permutation": lambda: fixed_perm,
    "CD - shuffle": lambda: np.random.choice(
        n_features, n_features, replace=False),
    "CD - random": lambda: np.random.choice(
        n_features, n_features, replace=True)
}

n_iter = 5
L = np.diag(H)
LA = np.linalg.norm(H, ord=2)

algos = ["GD",
         "CD - cyclic"]
dict_obj = {}
dict_iterates = {}
dict_iterates["GD"] = []
dict_iterates["CD - cyclic"] = []

current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["GD"] = current_palette[0]
dict_color["CD - cyclic"] = current_palette[1]
dict_color["CD - random"] = current_palette[2]

for algo in algos:
    w = np.array([8., -6.])
    dict_iterates[algo].append(w.copy())
    list_objs = []
    list_objs.append(w.T @ H @ w / 2)

    for _ in range(n_iter):
        if algo == "GD":
            w -= H @ w / LA
            dict_iterates[algo].append(w.copy())
            list_objs.append(w.T @ H @ w / 2)
        else:
            for j in feats[algo]():
                w[j] -= H[j, :] @ w / L[j]
                dict_iterates[algo].append(w.copy())
                list_objs.append(w.T @ H @ w / 2)
    dict_obj[algo] = np.array(list_objs.copy())

for algo in algos:
    dict_iterates[algo] = np.array(dict_iterates[algo])


def f(x, y):
    res = H[0, 0] * x ** 2
    res += H[1, 1] * y ** 2
    res += (H[1, 0] + H[0, 1]) * x * y
    return res / 2


all_x = np.linspace(-8, 8, num=100)
all_y = np.linspace(-8, 8, num=100)

Z = np.zeros((len(all_x), len(all_y)))
for i, j in product(np.arange(len(all_x)), np.arange(len(all_y))):
    Z[i, j] = f(all_x[i], all_y[j])

##############################################
dict_xlabel = {}
dict_xlabel["manuscript"] = r'$\beta_1$'
dict_xlabel["slides"] = r'$x_1$'
##############################################
dict_ylabel = {}
dict_ylabel["manuscript"] = r'$\beta_2$'
dict_ylabel["slides"] = r'$x_2$'
##############################################
dict_fontsize = {}
dict_fontsize["manuscript"] = 40
dict_fontsize["slides"] = 50

dict_title = {}
dict_title["manuscript", "CD"] = None
dict_title["manuscript", "GD"] = None
dict_title["slides", "CD"] = "Coordinate descent"
dict_title["slides", "GD"] = "Gradient descent"

xticks = [-3, 0, 4, 8]
yticks = [-8, -4, 0, 3]
xlim = (-3, 8)
ylim = (-8, 3)

plt.rcParams['figure.constrained_layout.use'] = True
# for fig in ["manuscript", "slides"]:
for fig in ["slides"]:
    fontsize = dict_fontsize[fig]
    fig_dir = dict_fig_dir[fig]
    plt.figure(figsize=(8, 8), constrained_layout=True)
    levels = np.flip(dict_obj["CD - cyclic"][:6])
    X, Y = np.meshgrid(all_x, all_y)
    plt.contour(
        X, Y, Z.T, levels=levels, linewidths=2, colors='blue')
    colors = discrete_color(len(dict_iterates["GD"][:, 0]), 'Blues')
    plt.scatter(
        dict_iterates["GD"][:, 0], dict_iterates["GD"][:, 1],
        color=colors, s=400)
    delta = linspace(0, 1)
    iterates = dict_iterates["GD"]
    for i in range(n_iter - 1):
        x = delta * iterates[i, 0] + (1 - delta) * iterates[i + 1, 0]
        y = delta * iterates[i, 1] + (1 - delta) * iterates[i + 1, 1]
        plt.plot(x, y, linestyle="--", color='blue')

    plt.xlabel(dict_xlabel[fig], fontsize=fontsize)
    plt.ylabel(dict_ylabel[fig], fontsize=fontsize)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(xticks, fontsize=fontsize)
    plt.yticks(yticks, fontsize=fontsize)
    plt.title(dict_title[fig, "GD"], fontsize=fontsize)
    # plt.tight_layout()

    if save_fig:
        plt.savefig(fig_dir + "2d_gd.pdf")
    plt.show()

    plt.figure(figsize=(8, 8), constrained_layout=True)
    levels = np.flip(dict_obj["CD - cyclic"][:6])
    X, Y = np.meshgrid(all_x, all_y)
    plt.contour(
        X, Y, Z.T, levels=levels, linewidths=2, colors='blue')
    # plt.scatter(dict_iterates["GD"][:, 0], dict_iterates["GD"][:, 1])
    colors = discrete_color(len(dict_iterates["CD - cyclic"][:, 0]), 'Oranges')
    plt.scatter(
        dict_iterates["CD - cyclic"][:, 0], dict_iterates["CD - cyclic"][:, 1],
        marker="X", color=colors, s=400, zorder=2)
    iterates = dict_iterates["CD - cyclic"]
    plt.xlim(xlim)
    plt.ylim(ylim)
    for i in range(n_iter - 1):
        plt.hlines(
            iterates[2 * i, 1], iterates[2 * (i + 1), 0], iterates[2 * i, 0],
            colors='red', linestyles='--', zorder=2)
        plt.vlines(
            iterates[2 * i + 1, 0], iterates[2 * (i + 1) + 1, 1],
            iterates[2 * i + 1, 1],
            colors='red', linestyles='--', zorder=2)

    plt.xticks(xticks, fontsize=fontsize)
    plt.yticks(yticks, fontsize=fontsize)
    plt.xlabel(dict_xlabel[fig], fontsize=fontsize)
    plt.ylabel(dict_ylabel[fig], fontsize=fontsize)
    plt.title(dict_title[fig, "CD"], fontsize=fontsize)
    # plt.tight_layout()
    # save_fig = False
    # save_fig = True
    if save_fig:
        plt.savefig(fig_dir + "2d_cd.pdf")
    plt.show()

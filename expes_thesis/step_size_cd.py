import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.sparse.linalg import svds

from libsvmdata import fetch_libsvm

from andersoncd.lasso import solver_enet
from andersoncd.plot_utils import configure_plt, _plot_legend_apart

configure_plt(fontsize=12)

save_fig = False
# save_fig = True
fig_dir = "../manuscript/figures/intro/"

current_palette = sns.color_palette("colorblind")

# you need to pass leukemia in sparse mat to make it work
datasets = ['real-sim', 'rcv1.binary']
# datasets = ['rcv1.binary']
# p_step_sizes = np.linspace(0, 2, num=11)
p_step_sizes = np.linspace(0, 1, num=5)
algos = [('1Lcd', p_step_size) for p_step_size in p_step_sizes]
algos.insert(0, ('pgd', 1))

dict_E = {}

for dataset in datasets:
    X, y = fetch_libsvm(dataset, normalize=True, min_nnz=3)
    y -= y.mean()
    y /= norm(y)
    X = X[:, :1000]

    L = svds(X, k=1)[1][0] ** 2

    for algo in algos:
        p_step_size = algo[1]
        step_size = 1 / (1 / L + (1 - 1 / L) * p_step_size)
        _, E, _ = solver_enet(
            X, y, alpha=0, rho=0, max_iter=2000, tmax=1000, f_gap=1,
            algo=algo[0], verbose=True, use_acc=False, step_size=step_size)
        dict_E[dataset, algo] = E.copy()

dict_xlim = {}
dict_xlim['leukemia'] = 20
dict_xlim['rcv1.binary'] = 2000
dict_xlim['real-sim'] = 2000
dict_xlim['news20.binary'] = 2000

dict_dataset_name = {}
dict_dataset_name['leukemia'] = 'leukemia'
dict_dataset_name['rcv1.binary'] = 'rcv1'
dict_dataset_name['real-sim'] = 'real-sim'
dict_dataset_name['news20.binary'] = 'news20'

dict_markevery = {}
dict_markevery['leukemia'] = 5
dict_markevery['rcv1.binary'] = 500
dict_markevery['real-sim'] = 500
dict_markevery['news20.binary'] = 100


fig, ax = plt.subplots(
    1, 3, sharex=False, sharey=True,
    figsize=[14, 5])

for i, dataset in enumerate(datasets):
    E_min = np.infty
    for algo in algos:
        E_min = min(E_min, dict_E[dataset, algo].min())

    for algo in algos:
        E = dict_E[dataset, algo]
        if algo[0] == '1Lcd':
            if algo[1] == 1:
                color = current_palette[1]
            else:
                color = plt.cm.viridis((algo[1] + 1) / 2)
            label = r"CD $\delta = %.2f$" % (algo[1])
            linestyle = '-'
            marker = None
            markevery = None
        else:
            color = 'black'
            label = 'GD'
            linestyle = '--'
            marker = "^"
            markevery = dict_markevery[dataset]
        ax[i].semilogy(
            E - E_min, label=label, color=color, linestyle=linestyle,
            marker=marker, markevery=markevery)
    ax[i].set_ylim(1e-15, 1)
    ax[i].set_xlim(-0.5, dict_xlim[dataset])
    ax[i].set_xlabel(r'$\#$ epochs')
    ax[i].set_title('%s' % dict_dataset_name[dataset])
plt.tight_layout()

ax[0].set_ylabel('Suboptimality')


if save_fig:
    fig.savefig(
        fig_dir + "stepsize_cd2.pdf", bbox_inches="tight")
    _plot_legend_apart(
        ax[0], fig_dir + "stepsize_cd2_legend.pdf", ncol=6)
plt.legend()
plt.show()

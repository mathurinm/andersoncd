import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from andersoncd.plot_utils import configure_plt, _plot_legend_apart


configure_plt()

current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["pgd"] = current_palette[0]
dict_color["fista"] = current_palette[0]
dict_color["cd"] = current_palette[1]
dict_color["apcg"] = current_palette[1]

dict_linestyle = {}
dict_linestyle[False, "pgd"] = "-"
dict_linestyle[False, "cd"] = "-"
dict_linestyle[True, "pgd"] = "--"
dict_linestyle[True, "cd"] = "--"
dict_linestyle[False, "fista"] = 'dotted'
dict_linestyle[False, "apcg"] = 'dotted'

dict_algo_name = {}
dict_algo_name[False, "pgd"] = "PGD"
dict_algo_name[False, "cd"] = "CD"
dict_algo_name[True, "pgd"] = "PGD - Anderson"
dict_algo_name[True, "cd"] = "CD - Anderson"
dict_algo_name[False, "fista"] = "PGD - inertial"
dict_algo_name[False, "apcg"] = "CD - inertial"


dataset_title = {}
dataset_title["leukemia"] = "leukemia"
dataset_title["gina_agnostic"] = "gina agnostic"
dataset_title["hiva_agnostic"] = "hiva agnostic"
dataset_title["upselling"] = "upselling"
dataset_title["rcv1.binary"] = "rcv1"
dataset_title["news20.binary"] = "news20"
dataset_title["kdda_train"] = "kdd"
dataset_title["finance"] = "finance"

dict_xlim = {}

dict_xlim["rcv1.binary", 10] = 200
dict_xlim["rcv1.binary", 100] = 1000
dict_xlim["rcv1.binary", 1000] = 10000
dict_xlim["rcv1.binary", 5000] = 100_000

dict_xlim["gina_agnostic", 10] = 1000
dict_xlim["gina_agnostic", 100] = 1000
dict_xlim["gina_agnostic", 1000] = 10_000
dict_xlim["gina_agnostic", 5000] = 10_000

dict_xlim["hiva_agnostic", 10] = 200
dict_xlim["hiva_agnostic", 100] = 5000
dict_xlim["hiva_agnostic", 1000] = 100_000
dict_xlim["hiva_agnostic", 5000] = 300_000

dict_xlim["leukemia", 10] = 500
dict_xlim["leukemia", 100] = 5000
dict_xlim["leukemia", 1000] = 100_000
dict_xlim["leukemia", 5000] = 100_000

dict_xlim["news20.binary", 10] = 400
dict_xlim["news20.binary", 100] = 5000
dict_xlim["news20.binary", 1000] = 500_000

dict_xlim["finance", 10] = 1000
dict_xlim["news20.binary", 100] = 5000
dict_xlim["news20.binary", 1000] = 500_000


####################################
div_alphas = [10, 100, 1000, 5000]
# use div_alphas = [10] if you want to be fast
########################################

div_rhos = [10, 100]

fig, axarr = plt.subplots(
    len(div_rhos), len(div_alphas), sharex=False, sharey=True,
    figsize=[14, 5], constrained_layout=True)

fig_E, axarr_E = plt.subplots(
    len(div_rhos), len(div_alphas), sharex=False, sharey=True,
    figsize=[14, 5], constrained_layout=True)


# choose your dataset, first generate the data for this dataset with main.py
dataset = "leukemia"
for idx1, div_rho in enumerate(div_rhos):
    for idx2, div_alpha in enumerate(div_alphas):
        df_data_all = pandas.read_pickle(
            "results/%s_%i_%i.pkl" % (dataset, div_alpha, div_rho))

        df_data = df_data_all[df_data_all['div_alpha'] == div_alpha]
        gaps = df_data['gaps']
        f_gaps = df_data['f_gaps']
        use_accs = df_data['use_acc']
        algo_names = df_data['algo_name']

        for gap, f_gap, use_acc, algo_name in zip(
                gaps, f_gaps, use_accs, algo_names):
            axarr.flat[idx1 * len(div_alphas) + idx2].semilogy(
                f_gap * np.arange(len(gap)), gap / gap[0],
                label=dict_algo_name[use_acc, algo_name],
                linestyle=dict_linestyle[use_acc, algo_name],
                color=dict_color[algo_name])
            try:
                axarr.flat[idx1 * len(div_alphas) + idx2].set_xlim(
                    0, dict_xlim[dataset, div_alpha])
            except Exception:
                print("no xlim")
        if idx1 == len(div_rhos) - 1:
            axarr.flat[
                (len(div_rhos) - 1) * len(div_alphas) + idx2].set_xlabel(
                    r"iteration $k$")
            axarr_E.flat[
                (len(div_rhos) - 1) * len(div_alphas) + idx2].set_xlabel(
                    r"iteration $k$")
        if idx1 == 0:
            axarr.flat[idx2].set_title(
                r"$\lambda =\lambda_{\max} / %i  $ " % div_alpha)
            axarr_E.flat[idx2].set_title(
                r"$\lambda =\lambda_{\max} / %i  $ " % div_alpha)
        Es = df_data['E']
        pobj_star = min(E.min() for E in Es)
        for E, f_gap, use_acc, algo_name in zip(
                Es, f_gaps, use_accs, algo_names):
            axarr_E.flat[idx1 * len(div_alphas) + idx2].semilogy(
                f_gap * np.arange(len(E)), (E - pobj_star) / E[0],
                label=dict_algo_name[use_acc, algo_name],
                linestyle=dict_linestyle[use_acc, algo_name],
                color=dict_color[algo_name])
        axarr.flat[idx1 * len(div_alphas) + idx2].set_yticks(
            (1, 1e-4, 1e-8, 1e-12, 1e-16))
        axarr.flat[idx1 * len(div_alphas) + idx2].set_ylim(
            (1e-19, 1))
        axarr_E.flat[idx1 * len(div_alphas) + idx2].set_yticks(
            (1, 1e-4, 1e-8, 1e-12, 1e-16))
        axarr_E.flat[idx1 * len(div_alphas) + idx2].set_ylim(
            (1e-16, 1))
    axarr.flat[idx1 * len(div_alphas)].set_ylabel(
        r"$\rho = \lambda / %i$" % div_rho)

    axarr_E.flat[idx1 * len(div_alphas)].set_ylabel(
        r"$\rho = \lambda / %i$" % div_rho)

save_fig = False

if save_fig:
    fig_dir = "../"
    fig_dir_svg = "../"
    fig.savefig(
        "%sgaps_real_enet.pdf" % fig_dir, bbox_inches="tight")
    fig.savefig(
        "%sgaps_real_enet.svg" % fig_dir_svg, bbox_inches="tight")
    fig_E.savefig(
        "%senergies_real_enet.pdf" % fig_dir, bbox_inches="tight")
    fig_E.savefig(
        "%senergies_real_enet.svg" % fig_dir_svg, bbox_inches="tight")
    _plot_legend_apart(
        axarr[0][0], "%senergies_real_enet_legend.pdf" % fig_dir, ncol=6)


fig.show()
fig_E.show()

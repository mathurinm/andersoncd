from collections import defaultdict
import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from andersoncd.plot_utils import configure_plt, _plot_legend_apart


# to generate the exact fig_gaps of the paper:
# dataset_names = [
#     "leukemia", "gina_agnostic", "hiva_agnostic", 'rcv1_train']
# div_alphas = [10, 100, 1000, 5000]

# save_fig = False
save_fig = True

# if you want to run the expe quickly choose instead:
# dataset_names = ["rcv1_train", "rcv1_train"]
# dataset_names = ["leukemia"]
dataset_name = "leukemia"
div_alphas = [10, 100, 1000, 5000]
div_rhos = [10, 100]


######################################################################
# config
configure_plt()

algos = [
    ['cd', True, 5],
    ['pgd', True, 5],
    ['cd', False, 5],
    ['pgd', False, 5],
    ['apcg', False, 5],
    ['fista', False, 5],
    ['rcd', False, 5],
]

current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["pgd"] = current_palette[0]
dict_color["fista"] = current_palette[0]
dict_color["cd"] = current_palette[1]
dict_color["rcd"] = current_palette[4]
dict_color["apcg"] = current_palette[1]

dict_linestyle = {}
dict_linestyle[False, "pgd"] = "-"
dict_linestyle[False, "cd"] = "-"
dict_linestyle[True, "pgd"] = "--"
dict_linestyle[True, "cd"] = "--"
dict_linestyle[False, "rcd"] = "-"
dict_linestyle[False, "fista"] = 'dotted'
dict_linestyle[False, "apcg"] = 'dotted'

dict_algo_name = {}
dict_algo_name[False, "pgd"] = "PGD"
dict_algo_name[False, "cd"] = "PCD"
dict_algo_name[False, "rcd"] = "PRCD"
dict_algo_name[True, "pgd"] = "PGD - Anderson"
dict_algo_name[True, "cd"] = "PCD - Anderson"
dict_algo_name[False, "fista"] = "PGD - inertial"
dict_algo_name[False, "apcg"] = "PCD - inertial"


dataset_title = {}
dataset_title["leukemia"] = "leukemia"
dataset_title["gina_agnostic"] = "gina agnostic"
dataset_title["hiva_agnostic"] = "hiva agnostic"
dataset_title["upselling"] = "upselling"
dataset_title["rcv1_train"] = "rcv1"
dataset_title["news20"] = "news20"
dataset_title["kdda_train"] = "kdd"
dataset_title["finance"] = "finance"

dict_xlim = defaultdict(lambda: None, key=None)

dict_xlim["leukemia", 10, 10] = 0.5
dict_xlim["leukemia", 100, 10] = 5
dict_xlim["leukemia", 1000, 10] = 50
dict_xlim["leukemia", 5000, 10] = 190

dict_xlim["leukemia", 10, 100] = 0.75
dict_xlim["leukemia", 100, 100] = 13
dict_xlim["leukemia", 1000, 100] = 43
dict_xlim["leukemia", 5000, 100] = 132

###############################################################################

fig_times_E, axarr_times_E = plt.subplots(
    len(div_rhos), len(div_alphas), sharex=False, sharey=True,
    figsize=[14, 5], constrained_layout=True)

fig_times_gaps, axarr_times_gaps = plt.subplots(
    len(div_rhos), len(div_alphas), sharex=False, sharey=False,
    figsize=[14, 5], constrained_layout=True)

all_axarr = [axarr_times_E, axarr_times_gaps]

fontsize = 22


for idx1, div_rho in enumerate(div_rhos):
    for idx2, div_alpha in enumerate(div_alphas):
        pobj_star = np.infty
        for algo in algos:
            algo_name, use_acc = algo[0], algo[1]
            df_data_all = pandas.read_pickle(
                "results/enet_%s_%s_%s%i_%i.pkl" % (
                    dataset_name, algo_name, str(use_acc), div_alpha, div_rho))

            Es = df_data_all['E']
            pobj_star = min(min(E.min() for E in Es), pobj_star)

        df_data_all = pandas.read_pickle(
            "results/enet_%s_%s_%s%i_%i.pkl" % (
                dataset_name, 'cd', str(True), div_alpha, div_rho))
        E0 = df_data_all['E'][0][0]

        for algo in algos:
            algo_name, use_acc = algo[0], algo[1]
            df_data_all = pandas.read_pickle(
                "results/enet_%s_%s_%s%i_%i.pkl" % (
                    dataset_name, algo_name, str(use_acc), div_alpha, div_rho))

            use_acc = df_data_all['use_acc'].to_numpy()[0]
            algo_name = df_data_all['algo_name'].to_numpy()[0]
            times = df_data_all['times'].to_numpy()[0]
            Es = df_data_all['E'].to_numpy()[0]
            gaps = df_data_all['gaps'].to_numpy()[0]

            # for E, gap, times, use_acc, algo_name in zip(
            #         Es, gaps, all_times, use_accs, algo_names):
            axarr_times_E[idx1, idx2].semilogy(
                times, (Es - pobj_star),
                label=dict_algo_name[use_acc, algo_name],
                linestyle=dict_linestyle[use_acc, algo_name],
                color=dict_color[algo_name])
            axarr_times_gaps[idx1, idx2].semilogy(
                times, gaps,
                label=dict_algo_name[use_acc, algo_name],
                linestyle=dict_linestyle[use_acc, algo_name],
                color=dict_color[algo_name])
        for axarr in all_axarr:
            axarr[idx1, idx2].tick_params(axis='x', labelsize=25)
            axarr[idx1, idx2].set_xlim(
                0, dict_xlim[dataset_name, div_alpha, div_rho])

            axarr[0, idx2].set_title(
                r"$\lambda = \lambda_{\max} / %i  $ " % div_alpha,
                fontsize=fontsize)
            axarr[-1, idx2].set_xlabel("Time (s)", fontsize=fontsize)

    for axarr in all_axarr:
        axarr[idx1, 0].set_yticks((1e-15, 1e-10, 1e-5, 1))
        axarr[idx1, 0].tick_params(axis='y', labelsize=25)

        axarr[idx1, 0].set_ylabel(
            r"$\rho = \lambda / %i  $ " % div_rho, fontsize=fontsize)


axarr_times_E[0, 0].set_ylim((1e-16, 2))
# axarr_times_gaps[0, 0].set_ylim((1e-16, 100))


if save_fig:
    fig_dir = "../../../extrapol_cd/tex/aistats20/prebuiltimages/"
    fig_dir_svg = "../../../extrapol_cd/tex/aistats20/images/"
    fig_times_E.savefig(
        "%senergies_enet_time.pdf" % fig_dir, bbox_inches="tight")
    fig_times_E.savefig(
        "%senergies_enet_time.svg" % fig_dir_svg, bbox_inches="tight")
    fig_times_gaps.savefig(
        "%sgaps_enet_time.pdf" % fig_dir, bbox_inches="tight")
    fig_times_gaps.savefig(
        "%sgaps_enet_time.svg" % fig_dir_svg, bbox_inches="tight")
    _plot_legend_apart(
        axarr_times_E[0, 0], "%senergies_enet_time_legend.pdf" % fig_dir,
        ncol=4)


fig_times_E.show()
fig_times_gaps.show()

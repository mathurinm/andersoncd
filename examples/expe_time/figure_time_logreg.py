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
# dataset_names = ["gina_agnostic", "rcv1_train"]
dataset_names = ["gina_agnostic", "rcv1_train", "news20"]
# dataset_names = ["leukemia", "gina_agnostic", "hiva_agnostic", "rcv1_train"]
# div_alphas = [10, 100]
div_alphas = [10, 100, 1000]


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
dict_xlim["rcv1_train", 10] = 5.2
dict_xlim["gina_agnostic", 10] = 25
dict_xlim["gina_agnostic", 100] = 72
dict_xlim["gina_agnostic", 1000] = 500
###############################################################################

fig_times_E, axarr_times_E = plt.subplots(
    len(dataset_names), len(div_alphas), sharex=False, sharey=True,
    figsize=[14, 8], constrained_layout=True)

fig_times_gaps, axarr_times_gaps = plt.subplots(
    len(dataset_names), len(div_alphas), sharex=False, sharey=True,
    figsize=[14, 8], constrained_layout=True)

all_axarr = [axarr_times_E, axarr_times_gaps]

fontsize = 22


for idx1, dataset_name in enumerate(dataset_names):
    for idx2, div_alpha in enumerate(div_alphas):
        pobj_star = np.infty
        for algo in algos:
            algo_name, use_acc = algo[0], algo[1]
            df_data_all = pandas.read_pickle(
                "results/logreg_%s_%s_%s%i.pkl" % (
                    dataset_name, algo_name, str(use_acc), div_alpha))

            Es = df_data_all['E']
            pobj_star = min(min(E.min() for E in Es), pobj_star)

        df_data_all = pandas.read_pickle(
            "results/logreg_%s_%s_%s%i.pkl" % (
                dataset_name, 'cd', str(False), div_alpha))
        E0 = df_data_all['E'][0][0]

        for algo in algos:
            algo_name, use_acc = algo[0], algo[1]
            df_data_all = pandas.read_pickle(
                "results/logreg_%s_%s_%s%i.pkl" % (
                    dataset_name, algo_name, str(use_acc), div_alpha))

            use_acc = df_data_all['use_acc'].to_numpy()[0]
            algo_name = df_data_all['algo_name'].to_numpy()[0]
            times = df_data_all['times'].to_numpy()[0]
            E = df_data_all['E'].to_numpy()[0]
            gaps = df_data_all['gaps'].to_numpy()[0]

            axarr_times_E[idx1, idx2].semilogy(
                times, (E - pobj_star) / E0,
                label=dict_algo_name[use_acc, algo_name],
                linestyle=dict_linestyle[use_acc, algo_name],
                color=dict_color[algo_name])

            axarr_times_gaps[idx1, idx2].semilogy(
                times[:-1], gaps[:-1],
                label=dict_algo_name[use_acc, algo_name],
                linestyle=dict_linestyle[use_acc, algo_name],
                color=dict_color[algo_name])

        for axarr in all_axarr:
            axarr[idx1, idx2].tick_params(axis='x', labelsize=25)
            axarr[idx1, idx2].set_xlim(
                0, dict_xlim[dataset_name, div_alpha])

            axarr[0, idx2].set_title(
                r"$\lambda_{\max} / %i  $ " % div_alpha,
                fontsize=fontsize)
            axarr_times_E[-1, idx2].set_xlabel("Time (s)", fontsize=fontsize)
    for axarr in all_axarr:
        axarr[idx1, 0].set_yticks((1e-15, 1e-10, 1e-5, 1))
        axarr[idx1, 0].tick_params(axis='y', labelsize=25)

        axarr[idx1, 0].set_ylabel(
            dataset_title[dataset_name], fontsize=fontsize)

axarr_times_E[0, 0].set_ylim((1e-16, 2))


if save_fig:
    fig_dir = "../../../extrapol_cd/tex/aistats20/prebuiltimages/"
    fig_dir_svg = "../../../extrapol_cd/tex/aistats20/images/"
    fig_times_E.savefig(
        "%senergies_logreg_time.pdf" % fig_dir, bbox_inches="tight")
    fig_times_E.savefig(
        "%senergies_logreg_time.svg" % fig_dir_svg, bbox_inches="tight")
    fig_times_gaps.savefig(
        "%sgaps_logreg_time.pdf" % fig_dir, bbox_inches="tight")
    fig_times_gaps.savefig(
        "%sgaps_logreg_time.svg" % fig_dir_svg, bbox_inches="tight")
    _plot_legend_apart(
        axarr_times_E[0, 0], "%senergies_logreg_time_legend.pdf" % fig_dir,
        ncol=4)


fig_times_E.show()
fig_times_gaps.show()

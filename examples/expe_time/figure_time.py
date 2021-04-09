import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from andersoncd.plot_utils import configure_plt, _plot_legend_apart


# to generate the exact fig_gaps of the paper:
# dataset_names = [
#     "leukemia", "gina_agnostic", "hiva_agnostic", 'rcv1_train']
# div_alphas = [10, 100, 1000, 5000]


# if you want to run the expe quickly choose instead:
dataset_names = ["rcv1.binary", "leukemia", "news20.binary"]
div_alphas = [10, 100, 1000]
# div_alphas = [10, 100, 1000]


######################################################################
# config
configure_plt()

algos = [
    ['cd', True, 5],
    ['pgd', True, 5],
    ['cd', False, 5],
    ['pgd', False, 5],
    ['rcd', False, 5],
    # ['apcg', False, 5],
    ['fista', False, 5]
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
###############################################################################

fig_times_E, axarr_times_E = plt.subplots(
    3, len(div_alphas), sharex=False, sharey=True,
    figsize=[14, 8], constrained_layout=True)

fontsize = 45


for idx1, div_alpha in enumerate(dataset_names):
    for idx2, div_alpha in enumerate(div_alphas):
        pobj_star = np.infty
        for algo in algos:
            algo_name, use_acc = algo[0], algo[1]
            if idx1 == 0:
                df_data_all = pandas.read_pickle(
                    "results/%s_%s_%s%i.pkl" % (
                        "rcv1.binary", algo_name, str(use_acc), div_alpha))
            if idx1 == 1:
                df_data_all = pandas.read_pickle(
                    "results/enet_%s_%s_%s%i.pkl" % (
                        "leukemia", algo_name, str(use_acc), div_alpha))
            if idx1 == 2:
                df_data_all = pandas.read_pickle(
                    "results/logreg_%s_%s_%s%i.pkl" % (
                        "news20.binary", algo_name, str(use_acc), div_alpha))
            Es = df_data_all['E']
            pobj_star = min(min(E.min() for E in Es), pobj_star)

        for algo in algos:
            algo_name, use_acc = algo[0], algo[1]
            if idx1 == 0:
                df_data_all = pandas.read_pickle(
                    "results/%s_%s_%s%i.pkl" % (
                        "rcv1.binary", algo_name, str(use_acc), div_alpha))
            if idx1 == 1:
                df_data_all = pandas.read_pickle(
                    "results/enet_%s_%s_%s%i.pkl" % (
                        "leukemia", algo_name, str(use_acc), div_alpha))
            if idx1 == 2:
                df_data_all = pandas.read_pickle(
                    "results/logreg_%s_%s_%s%i.pkl" % (
                        "news20.binary", algo_name, str(use_acc), div_alpha))
                # if div_alpha == 1000:
                #     import ipdb; ipdb.set_trace()

            use_accs = df_data_all['use_acc']
            algo_names = df_data_all['algo_name']
            all_times = df_data_all['times']
            Es = df_data_all['E']

            for E, times, use_acc, algo_name in zip(
                    Es, all_times, use_accs, algo_names):
                axarr_times_E[idx1, idx2].semilogy(
                    times, (E - pobj_star) / E[0],
                    label=dict_algo_name[use_acc, algo_name],
                    linestyle=dict_linestyle[use_acc, algo_name],
                    color=dict_color[algo_name])
            axarr_times_E[idx1, idx2].tick_params(axis='x', labelsize=35)

        axarr_times_E[0, idx2].set_title(
            r"$\lambda_{\max} / %i  $ " % div_alpha,
            fontsize=fontsize)
        axarr_times_E[-1, idx2].set_xlabel("Time (s)", fontsize=fontsize)
    axarr_times_E[idx1, 0].set_yticks((1e-15, 1e-10, 1e-5, 1))
    axarr_times_E[idx1, 0].tick_params(axis='y', labelsize=35)

axarr_times_E[0, 0].set_ylabel("Lasso \n rcv1", fontsize=fontsize)
axarr_times_E[1, 0].set_ylabel("Enet \n leuk.", fontsize=fontsize)
axarr_times_E[2, 0].set_ylabel("Logreg \n news20.binary", fontsize=fontsize)

axarr_times_E[0, 0].set_ylim((1e-16, 2))

save_fig = False
# save_fig = True

if save_fig:
    fig_dir = "../"
    fig_dir_svg = "../"
    fig_times_E.savefig(
        "%senergies_time.pdf" % fig_dir, bbox_inches="tight")
    fig_times_E.savefig(
        "%senergies_time.svg" % fig_dir_svg, bbox_inches="tight")
    _plot_legend_apart(
        axarr_times_E[0, 0], "%senergies_time_legend.pdf" % fig_dir, ncol=3)


fig_times_E.show()

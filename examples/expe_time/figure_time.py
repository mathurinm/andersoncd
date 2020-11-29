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
dataset_names = ["rcv1_train", "leukemia"]
div_alphas = [10, 100, 1000]


################## config ####################################################
configure_plt()

algos = [
    ['cd', False, 5],
    ['rcd', False, 5],
    ['cd', True, 5],
    # ['apcg', False, 5],
    ['pgd', True, 5],
    ['pgd', False, 5],
    ['fista', False, 5]
]

current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["pgd"] = current_palette[0]
dict_color["fista"] = current_palette[0]
dict_color["cd"] = current_palette[1]
dict_color["rcd"] = current_palette[1]
dict_color["apcg"] = current_palette[1]

dict_linestyle = {}
dict_linestyle[False, "pgd"] = "-"
dict_linestyle[False, "cd"] = "-"
dict_linestyle[True, "pgd"] = "--"
dict_linestyle[True, "cd"] = "--"
dict_linestyle[False, "rcd"] = (0, (3, 5, 1, 5, 1, 5))
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

dict_xlim = {}

dict_xlim["rcv1_train", 10] = 200
dict_xlim["rcv1_train", 100] = 1000
dict_xlim["rcv1_train", 1000] = 10000
dict_xlim["rcv1_train", 5000] = 100_000

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

dict_xlim["news20", 10] = 400
dict_xlim["news20", 100] = 5000
dict_xlim["news20", 1000] = 500_000

dict_xlim["finance", 10] = 1000
dict_xlim["news20", 100] = 5000
dict_xlim["news20", 1000] = 500_000
###############################################################################

fig_times_E, axarr_times_E = plt.subplots(
    2, len(div_alphas), sharex=False, sharey=True,
    figsize=[14, 8], constrained_layout=True)


for idx1, div_alpha in enumerate(dataset_names):
    for idx2, div_alpha in enumerate(div_alphas):
        pobj_star = np.infty
        for algo in algos:
            algo_name, use_acc = algo[0], algo[1]
            if idx1 == 0:
                df_data_all = pandas.read_pickle(
                    "results/%s_%s_%s%i.pkl" % (
                        "rcv1_train", algo_name, str(use_acc), div_alpha))
            if idx1 == 1:
                df_data_all = pandas.read_pickle(
                    "results/enet_%s_%s_%s%i.pkl" % (
                        "leukemia", algo_name, str(use_acc), div_alpha))
            Es = df_data_all['E']
            pobj_star = min(min(E.min() for E in Es), pobj_star)

        for algo in algos:
            algo_name, use_acc = algo[0], algo[1]
            if idx1 == 0:
                df_data_all = pandas.read_pickle(
                    "results/%s_%s_%s%i.pkl" % (
                        "rcv1_train", algo_name, str(use_acc), div_alpha))
            if idx1 == 1:
                # import ipdb; ipdb.set_trace()
                df_data_all = pandas.read_pickle(
                    "results/enet_%s_%s_%s%i.pkl" % (
                        "leukemia", algo_name, str(use_acc), div_alpha))

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

        axarr_times_E[0, idx2].set_title(
            r"Lasso  $\lambda =\lambda_{\max} / %i  $ " % div_alpha)
        axarr_times_E[-1, idx2].set_xlabel("Time (s)")
    axarr_times_E[idx1, 0].set_yticks((1e-15, 1e-10, 1e-5, 1))

axarr_times_E[0, 0].set_ylabel("Lasso \n rcv1")

axarr_times_E[1, 0].set_ylabel("Enet \n leukemia")

save_fig = True

if save_fig:
    fig_dir = "../../../extrapol_cd/tex/aistats20/prebuiltimages/"
    fig_dir_svg = "../../../extrapol_cd/tex/aistats20/images/"
    fig_times_E.savefig(
        "%senergies_time.pdf" % fig_dir, bbox_inches="tight")
    fig_times_E.savefig(
        "%senergies_time.svg" % fig_dir_svg, bbox_inches="tight")
    _plot_legend_apart(
        axarr_times_E[0, 0], "%senergies_time_legend.pdf" % fig_dir, ncol=6)


fig_times_E.show()

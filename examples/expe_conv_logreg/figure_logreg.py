import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from andersoncd.plot_utils import configure_plt, plot_legend_apart


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
dict_algo_name[False, "cd"] = "PCD"
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

dict_xlim["gina_agnostic", 10] = 2500
dict_xlim["gina_agnostic", 100] = 3000
dict_xlim["gina_agnostic", 1000] = 10_000
dict_xlim["rcv1_train", 10] = 1_000
dict_xlim["rcv1_train", 100] = 5_000
dict_xlim["rcv1_train", 1000] = 200_000
dict_xlim["news20", 10] = 3_000
dict_xlim["news20", 100] = 10_000
dict_xlim["news20", 1000] = 150_000

dataset_names = ["gina_agnostic", 'rcv1_train', "news20"]
div_alphas = [10, 100, 1_000]

fig, axarr = plt.subplots(
    len(dataset_names), len(div_alphas), sharex=False, sharey=True,
    figsize=[14, 8], constrained_layout=True)

fig_E, axarr_E = plt.subplots(
    len(dataset_names), len(div_alphas), sharex=False, sharey=True,
    figsize=[14, 8], constrained_layout=True)


for idx1, dataset in enumerate(dataset_names):

    for idx2, div_alpha in enumerate(div_alphas):
        df_data_all = pandas.read_pickle(
            "results/%s_%i.pkl" % (dataset, div_alpha))

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
            axarr.flat[idx1 * len(div_alphas) + idx2].set_ylim((1e-10, 1))
        if idx1 == len(dataset_names) - 1:
            axarr.flat[
                (len(dataset_names) - 1) * len(div_alphas) + idx2].set_xlabel(
                    r"iteration $k$")
            axarr_E.flat[
                (len(dataset_names) - 1) * len(div_alphas) + idx2].set_xlabel(
                    r"iteration $k$")
        if idx1 == 0:
            axarr.flat[idx2].set_title(
                r"$\lambda =\lambda_{\max} / %i  $ " % div_alpha)
            axarr_E.flat[idx2].set_title(
                r"$\lambda =\lambda_{\max} / %i  $ " % div_alpha)
            axarr_E.flat[idx1 * len(div_alphas) + idx2].set_ylim((1e-12, 1))
        Es = df_data['E']
        pobj_star = min(E.min() for E in Es)
        for E, f_gap, use_acc, algo_name in zip(
                Es, f_gaps, use_accs, algo_names):
            axarr_E.flat[idx1 * len(div_alphas) + idx2].semilogy(
                f_gap * np.arange(len(E)), (E - pobj_star) / E[0],
                label=dict_algo_name[use_acc, algo_name],
                linestyle=dict_linestyle[use_acc, algo_name],
                color=dict_color[algo_name])
            try:
                axarr_E.flat[idx1 * len(div_alphas) + idx2].set_xlim(
                    0, dict_xlim[dataset, div_alpha])
            except Exception:
                print("no xlim")
        axarr.flat[idx1 * len(div_alphas) + idx2].set_yticks(
            (1, 1e-4, 1e-8, 1e-12))
        axarr_E.flat[idx1 * len(div_alphas) + idx2].set_yticks(
            (1, 1e-4, 1e-8, 1e-12))
    axarr.flat[idx1 * len(div_alphas)].set_ylabel(
        "%s" % dataset_title[dataset])

    axarr_E.flat[idx1 * len(div_alphas)].set_ylabel(
        "%s" % dataset_title[dataset])

save_fig = False
fig_dir = "../../../extrapol_cd/tex/aistats20/prebuiltimages/"
fig_dir_svg = "../../../extrapol_cd/tex/aistats20/images/"

if save_fig:
    fig.savefig(
        "%sgaps_real_logreg.pdf" % fig_dir, bbox_inches="tight")
    fig.savefig(
        "%sgaps_real_logreg.svg" % fig_dir_svg, bbox_inches="tight")
    fig_E.savefig(
        "%senergies_real_logreg.pdf" % fig_dir, bbox_inches="tight")
    fig_E.savefig(
        "%senergies_real_logreg.svg" % fig_dir_svg, bbox_inches="tight")
    plot_legend_apart(
        axarr[0][0], "%senergies_real_logreg_legend.pdf" % fig_dir, ncol=6)


fig.show()
fig_E.show()

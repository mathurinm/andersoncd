# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

C_LIST = sns.color_palette("colorblind", 8)
C_LIST_DARK = sns.color_palette("dark", 8)


def configure_plt(fontsize=10, poster=True):
    rc('font', **{'family': 'sans-serif',
                  'sans-serif': ['Computer Modern Roman']})
    usetex = matplotlib.checkdep_usetex(True)
    params = {'axes.labelsize': fontsize,
              'font.size': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize - 2,
              'ytick.labelsize': fontsize - 2,
              'text.usetex': usetex,
              'figure.figsize': (8, 6)}
    plt.rcParams.update(params)

    sns.set_palette('colorblind')
    sns.set_style("ticks")
    if poster:
        sns.set_context("poster")


def _plot_legend_apart(ax, figname, ncol=None):
    """Do all your plots with fig, ax = plt.subplots(),
    don't call plt.legend() at the end but this instead"""
    if ncol is None:
        ncol = len(ax.lines)
    fig = plt.figure(figsize=(30, 4), constrained_layout=True)
    fig.legend(ax.lines, [line.get_label() for line in ax.lines], ncol=ncol,
               loc="upper center")
    fig.tight_layout()
    fig.savefig(figname, bbox_inches="tight")
    os.system("pdfcrop %s %s" % (figname, figname))
    return fig


dict_algo_name = {}
dict_algo_name["pgd", False] = "GD"
dict_algo_name["cd", False] = "CD"
dict_algo_name["bcd", False] = "BCD"
dict_algo_name["pgd", True] = "GD - Anderson"
dict_algo_name["cd", True] = "CD - Anderson"
dict_algo_name["bcd", True] = "BCD - Anderson"
dict_algo_name["rcd", False] = "RCD"
dict_algo_name["rbcd", False] = "RBCD"
dict_algo_name["fista", False] = "GD - inertial"
dict_algo_name["apcg", False] = "CD - inertial"


current_palette = sns.color_palette("colorblind")
dict_color = {}
dict_color["pgd"] = current_palette[0]
dict_color["fista"] = current_palette[0]
dict_color["cd"] = current_palette[1]
dict_color["bcd"] = current_palette[1]
dict_color["rcd"] = current_palette[4]
dict_color["rbcd"] = current_palette[4]
dict_color["apcg"] = current_palette[1]

# Author: Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

C_LIST = sns.color_palette("colorblind", 8)
C_LIST_DARK = sns.color_palette("dark", 8)


def configure_plt():
    rc('font', **{'family': 'sans-serif',
                  'sans-serif': ['Computer Modern Roman']})
    usetex = matplotlib.checkdep_usetex(True)
    params = {'axes.labelsize': 12,
              'font.size': 12,
              'legend.fontsize': 12,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': usetex,
              'figure.figsize': (8, 6)}
    plt.rcParams.update(params)

    sns.set_palette('colorblind')
    sns.set_context("poster")
    sns.set_style("ticks")


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

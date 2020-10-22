import pandas
import matplotlib.pyplot as plt
from celer.plot_utils import configure_plt


configure_plt()

datasets = [
    "gina_agnostic", "rcv1_train", "real-sim", "news20"]


dataset_title = {}
dataset_title["leukemia"] = "leukemia"
dataset_title["gina_agnostic"] = "gina agnostic"
dataset_title["hiva_agnostic"] = "hiva agnostic"
dataset_title["upselling"] = "upselling"
dataset_title["rcv1_train"] = "rcv1"
dataset_title["news20"] = "news20"
dataset_title["kdda_train"] = "kdd"
dataset_title["real-sim"] = "real-sim"
dataset_title["finance"] = "finance"

list_k = [0, 7, 8, 9]

fontsize = 50

fig, axarr = plt.subplots(
    2, 2, sharex=False, sharey=False, figsize=[10, 8], constrained_layout=True)

for i, dataset in enumerate(datasets):
    for idxk, k in enumerate(list_k):
        color = plt.cm.viridis(idxk / len(list_k))
        df = pandas.read_pickle("results/%s_%i.pkl" % (dataset, k))
        rayleighs = df['rayleighs'].to_numpy()[0]
        label = r"$W(T^{%d})$" % 2 ** k
        axarr.flat[i].plot(
            rayleighs.real, rayleighs.imag, color=color, label=label)
    axarr.flat[i].plot(1, 0, marker="P", c='k')
    axarr.flat[i].set_title(dataset_title[dataset])
    axarr.flat[i].axis("equal")
    axarr.flat[i].set_xticks((-2, -1, 0, 1, 2))
    axarr.flat[i].set_yticks((-2, -1, 0, 1, 2))


fig.show()

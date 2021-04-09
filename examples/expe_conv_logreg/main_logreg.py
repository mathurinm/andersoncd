from itertools import product

import pandas
import numpy as np
from libsvmdata import fetch_libsvm
from joblib import Parallel, delayed, parallel_backend

from andersoncd.data.real import load_openml
from andersoncd.logreg import solver_logreg, apcg_logreg


# to reproduce the fig of the paper (long)
dataset_names = ["gina_agnostic", 'rcv1.binary', 'news20.binary']
div_alphas = [10, 100, 1000]

# to be fast:
# dataset_names = ["gina_agnostic"]
# div_alphas = [10, 100]


n_jobs = 1

"""Config
"""
algos = [
    ['cd', True, 5],
    ['cd', False, 5],
    ['apcg', False, 5],
    ['pgd', True, 5],
    ['pgd', False, 5],
    ['fista', False, 5]
]


dict_maxiter = {}
dict_maxiter["leukemia", 10] = 10_000
dict_maxiter["mushroom", 10] = 10_000
dict_maxiter["gina_agnostic", 10] = 2_500
dict_maxiter["hiva_agnostic", 10] = 10_000
dict_maxiter["upselling", 10] = 10_000
dict_maxiter["rcv1.binary", 10] = 1_000
dict_maxiter["news20.binary", 10] = 3_000
dict_maxiter["kdda_train", 10] = 1_000
dict_maxiter["finance", 10] = 5_000

dict_maxiter["leukemia", 100] = 10_000
dict_maxiter["mushroom", 100] = 10_000
dict_maxiter["gina_agnostic", 100] = 3_000
dict_maxiter["hiva_agnostic", 100] = 10_000
dict_maxiter["upselling", 100] = 10_000
dict_maxiter["rcv1.binary", 100] = 5_000
dict_maxiter["news20.binary", 100] = 10_000
dict_maxiter["kdda_train", 100] = 5_000
dict_maxiter["finance", 100] = 50_000

dict_maxiter["leukemia", 1000] = 100_000
dict_maxiter["mushroom", 1000] = 100_000
dict_maxiter["gina_agnostic", 1000] = 100_000
dict_maxiter["hiva_agnostic", 1000] = 100_000
dict_maxiter["upselling", 1000] = 100_000
dict_maxiter["rcv1.binary", 1000] = 200_000
dict_maxiter["news20.binary", 1000] = 150_000
dict_maxiter["kdda_train", 1000] = 1_000
dict_maxiter["finance", 1000] = 50_000

dict_maxiter["leukemia", 5_000] = 300_000
dict_maxiter["mushroom", 5_000] = 300_000
dict_maxiter["gina_agnostic", 5_000] = 300_000
dict_maxiter["hiva_agnostic", 5_000] = 300_000
dict_maxiter["upselling", 5000] = 100_000
dict_maxiter["rcv1.binary", 5000] = 100_000
dict_maxiter["news20.binary", 5000] = 1_000_000
dict_maxiter["kdda_train", 5000] = 1_000


dict_f_gap = {}
dict_f_gap["leukemia"] = 10
dict_f_gap["mushroom"] = 10
dict_f_gap["gina_agnostic"] = 10
dict_f_gap["hiva_agnostic"] = 10
dict_f_gap["upselling"] = 10
dict_f_gap["rcv1.binary"] = 10
dict_f_gap["news20.binary"] = 10
dict_f_gap["kdda_train"] = 10
dict_f_gap["finance"] = 50

"""End config
"""


def parallel_function(dataset_name, algo, div_alpha):
    algo_name, use_acc, K = algo
    if dataset_name.startswith((
            'rcv1.binary', 'news20.binary', 'kdda_train', 'finance')):
        X, y = fetch_libsvm(dataset_name, normalize=True)
    else:
        X, y = load_openml(dataset_name, normalize_y=False)

    alpha_max = np.max(np.abs(X.T @ y)) / 2
    alpha = alpha_max / div_alpha
    tol = 1e-12
    f_gap = dict_f_gap[dataset_name]

    max_iter = dict_maxiter[dataset_name, div_alpha]
    if algo_name == 'apcg':
        w, E, gaps = apcg_logreg(
            X, y, alpha, max_iter=max_iter, tol=tol, f_gap=f_gap, verbose=True)
    else:
        w, E, gaps = solver_logreg(
            X, y, alpha=alpha, f_gap=f_gap, max_iter=max_iter,
            tol=tol, algo=algo_name, use_acc=use_acc, K=K, verbose=True)

    return (dataset_name, algo_name, use_acc, K, div_alpha, w, E, gaps, f_gap)


print("enter parallel")
backend = 'loky'

with parallel_backend("loky", inner_max_num_threads=1):
    results = Parallel(
        n_jobs=n_jobs, verbose=100)(
        delayed(parallel_function)(
            dataset_name, algo, div_alpha)
        for dataset_name, algo, div_alpha in product(
            dataset_names, algos, div_alphas))

print('OK finished parallel')


df = pandas.DataFrame(results)
df.columns = [
    'dataset', 'algo_name', 'use_acc', 'K', 'div_alpha', "optimum", "E",
    "gaps", "f_gaps"]

for dataset_name in dataset_names:
    for div_alpha in div_alphas:
        df_temp = df[df['dataset'] == dataset_name]
        df_temp[df_temp['div_alpha'] == div_alpha].to_pickle(
            "results/%s_%i.pkl" % (dataset_name, div_alpha))

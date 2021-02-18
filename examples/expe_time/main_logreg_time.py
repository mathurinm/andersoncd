from itertools import product
from collections import defaultdict

import pandas
import numpy as np
from libsvmdata import fetch_libsvm
from joblib import Parallel, delayed, parallel_backend

from andersoncd.data.real import load_openml
from andersoncd.logreg import solver_logreg, apcg_logreg


# to generate the exact fig of the paper:
# dataset_names = [
#     "leukemia", "gina_agnostic", "hiva_agnostic", 'rcv1_train']
# div_alphas = [10, 100, 1000, 5000]


# if you want to run the file quickly choose instead:
# dataset_names = ["leukemia"]
# dataset_names = ["gina_agnostic"]
dataset_names = ["news20"]
div_alphas = [1000]
# div_alphas = [10, 100, 1000, 5000]


algos = [
    ['apcg', False, 5],
    ['pgd', False, 5],
    ['pgd', True, 5],
    ['fista', False, 5],
    ['cd', False, 5],
    ['rcd', False, 5],
    ['cd', True, 5]
]

dict_maxiter = defaultdict(lambda: 1_000_000, key=None)


dict_f_gap = {}
dict_f_gap["leukemia"] = 10
dict_f_gap["mushroom"] = 10
dict_f_gap["gina_agnostic"] = 10
dict_f_gap["hiva_agnostic"] = 10
dict_f_gap["upselling"] = 10
dict_f_gap["rcv1_train"] = 10
dict_f_gap["news20"] = 10
dict_f_gap["kdda_train"] = 10
dict_f_gap["finance"] = 50

dict_tmax = {}

dict_tmax["gina_agnostic", 10] = 40
dict_tmax["gina_agnostic", 100] = 200
dict_tmax["gina_agnostic", 1000] = 1000
dict_tmax["gina_agnostic", 5000] = 3600

dict_tmax["rcv1_train", 10] = 10
dict_tmax["rcv1_train", 100] = 250
dict_tmax["rcv1_train", 1000] = 6250

dict_tmax["news20", 10] = 200
dict_tmax["news20", 100] = 7000
dict_tmax["news20", 1000] = 30_000


def parallel_function(dataset_name, algo, div_alpha):
    algo_name, use_acc, K = algo
    if dataset_name.startswith((
            'rcv1_train', 'news20', 'kdda_train', 'finance')):
        X, y = fetch_libsvm(dataset_name, normalize=True)
    else:
        X, y = load_openml(dataset_name, normalize_y=False)

    alpha_max = np.max(np.abs(X.T @ y)) / 2
    alpha = alpha_max / div_alpha
    tol = 1e-14
    f_gap = dict_f_gap[dataset_name]

    max_iter = dict_maxiter[dataset_name, div_alpha]

    tmax = dict_tmax[dataset_name, div_alpha]

    # for _ in range(2):

    if algo_name == 'apcg':
        w, E, gaps, times = apcg_logreg(
            X, y, alpha, max_iter=max_iter, tol=tol, f_gap=f_gap,
            compute_time=True, tmax=10, verbose=True)
    else:
        w, E, gaps, times = solver_logreg(
            X, y, alpha, f_gap=f_gap, max_iter=max_iter, tol=tol,
            use_acc=use_acc, K=K, algo=algo_name, compute_time=True,
            tmax=10, verbose=True, seed=42)

    if algo_name == 'apcg':
        w, E, gaps, times = apcg_logreg(
            X, y, alpha, max_iter=max_iter, tol=tol, f_gap=f_gap,
            compute_time=True, tmax=tmax, verbose=True)
    else:
        w, E, gaps, times = solver_logreg(
            X, y, alpha, f_gap=f_gap, max_iter=max_iter, tol=tol,
            use_acc=use_acc, K=K, algo=algo_name, compute_time=True,
            tmax=tmax, verbose=True, seed=42)

    if len(times) == 0:
        return 1 / 0

    my_results = (
        dataset_name, algo_name, use_acc, K, div_alpha, w,
        E, gaps, f_gap, times)
    df = pandas.DataFrame(my_results).transpose()
    df.columns = [
        'dataset', 'algo_name', 'use_acc', 'K', 'div_alpha', "optimum", "E",
        "gaps", "f_gaps", "times"]
    str_results = "results/logreg_%s_%s_%s%i.pkl" % (
                dataset_name, algo_name, str(use_acc), div_alpha)
    df.to_pickle(str_results)


print("enter parallel")
n_jobs = len(dataset_names) * len(div_alphas) * len(algos)

# n_jobs = 1
n_jobs = min(n_jobs, 15)

with parallel_backend("loky", inner_max_num_threads=1):
    Parallel(
        n_jobs=n_jobs, verbose=100)(
        delayed(parallel_function)(
            dataset_name, algo, div_alpha)
        for dataset_name, algo, div_alpha in product(
            dataset_names, algos, div_alphas))

print('OK finished parallel')

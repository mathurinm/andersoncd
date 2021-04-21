from itertools import product
from collections import defaultdict

import pandas
import numpy as np
from libsvmdata import fetch_libsvm
from joblib import Parallel, delayed, parallel_backend

from andersoncd.data.real import load_openml
from andersoncd.lasso import solver_enet, apcg_enet


dataset_names = ["leukemia"]
div_alphas = [100, 1000, 5000]
div_rhos = [100]

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
dict_f_gap["leukemia"] = 1
dict_f_gap["mushroom"] = 10
dict_f_gap["gina_agnostic"] = 10
dict_f_gap["hiva_agnostic"] = 10
dict_f_gap["upselling"] = 10
dict_f_gap["rcv1.binary"] = 10
dict_f_gap["news20.binary"] = 10
dict_f_gap["kdda_train"] = 10
dict_f_gap["finance"] = 50

dict_tmax = {}
dict_tmax["leukemia", 10] = 2
dict_tmax["leukemia", 100] = 30
dict_tmax["leukemia", 1000] = 100
dict_tmax["leukemia", 5000] = 200

dict_tmax["rcv1.binary", 10] = 5
dict_tmax["rcv1.binary", 100] = 10
dict_tmax["rcv1.binary", 1000] = 120
dict_tmax["rcv1.binary", 5000] = 600


def parallel_function(dataset_name, algo, div_alpha, div_rho):
    algo_name, use_acc, K = algo
    if dataset_name.startswith((
            'rcv1.binary', 'news20.binary', 'kdda_train', 'finance')):
        X, y = fetch_libsvm(dataset_name, normalize=True)
        y -= y.mean()
        y /= np.linalg.norm(y)
    else:
        X, y = load_openml(dataset_name)

    alpha_max = np.max(np.abs(X.T @ y))
    alpha = alpha_max / div_alpha
    tol = 1e-20
    f_gap = dict_f_gap[dataset_name]

    max_iter = dict_maxiter[dataset_name, div_alpha]

    tmax = dict_tmax[dataset_name, div_alpha]

    for _ in range(2):
        if algo_name == 'apcg':
            w, E, gaps, times = apcg_enet(
                X, y, alpha, rho=alpha/div_rho, max_iter=max_iter, tol=tol,
                f_gap=f_gap, compute_time=True, tmax=tmax, verbose=True)
        else:
            w, E, gaps, times = solver_enet(
                X, y, alpha, rho=alpha/div_rho, f_gap=f_gap, max_iter=max_iter,
                tol=tol, use_acc=use_acc, K=K, algo=algo_name,
                compute_time=True, tmax=tmax, verbose=True, seed=42)

    my_results = (
        dataset_name, algo_name, use_acc, K, div_alpha, div_rho, w,
        E, gaps, f_gap, times)
    df = pandas.DataFrame(my_results).transpose()
    df.columns = [
        'dataset', 'algo_name', 'use_acc', 'K', 'div_alpha', 'div_rho',
        "optimum", "E", "gaps", "f_gaps", "times"]
    str_results = "results/enet_%s_%s_%s%i_%i.pkl" % (
                dataset_name, algo_name, str(use_acc), div_alpha, div_rho)
    df.to_pickle(str_results)


print("enter parallel")
n_jobs = len(dataset_names) * len(div_alphas) * len(algos)

# n_jobs = 1
n_jobs = min(n_jobs, 15)

with parallel_backend("loky", inner_max_num_threads=1):
    Parallel(
        n_jobs=n_jobs, verbose=100)(
        delayed(parallel_function)(
            dataset_name, algo, div_alpha, div_rho)
        for dataset_name, algo, div_alpha, div_rho in product(
            dataset_names, algos, div_alphas, div_rhos))

print('OK finished parallel')


# print("Enter sequential")
# for dataset_name, algo, div_alpha, div_rho in product(
#         dataset_names, algos, div_alphas, div_rhos):
#     parallel_function(dataset_name, algo, div_alpha, div_rho)

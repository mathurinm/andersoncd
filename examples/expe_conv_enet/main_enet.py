from itertools import product

import pandas
import numpy as np
from celer.datasets import fetch_libsvm
from joblib import Parallel, delayed, parallel_backend

from andersoncd.data.real import load_openml
from andersoncd.lasso import solver_enet, apcg


dataset_names = [
    "leukemia", "gina_agnostic", "hiva_agnostic", 'rcv1_train', 'news20']


algos = [
    ['cd', True, 5],
    ['cd', False, 5],
    ['pgd', True, 5],
    ['pgd', False, 5],
    ['fista', False, 5]
]

div_alphas = [10, 100, 1000, 5_000]
div_rhos = [1, 10, 100]

##################################################
# If you want to run the file quickly, only choose:
# dataset_names = ["leukemia"]
# div_alphas = [10]
################################################

dict_maxiter = {}
dict_maxiter["leukemia", 10] = 1000
dict_maxiter["mushroom", 10] = 10_000
dict_maxiter["gina_agnostic", 10] = 10_000
dict_maxiter["hiva_agnostic", 10] = 10_000
dict_maxiter["upselling", 10] = 10_000
dict_maxiter["rcv1_train", 10] = 3_000
dict_maxiter["news20", 10] = 10_000
dict_maxiter["kdda_train", 10] = 10_000
dict_maxiter["finance", 10] = 5_000

dict_maxiter["leukemia", 100] = 10_000
dict_maxiter["mushroom", 100] = 10_000
dict_maxiter["gina_agnostic", 100] = 10_000
dict_maxiter["hiva_agnostic", 100] = 10_000
dict_maxiter["upselling", 100] = 10_000
dict_maxiter["rcv1_train", 100] = 10_000
dict_maxiter["news20", 100] = 10_000
dict_maxiter["kdda_train", 100] = 1_000
dict_maxiter["finance", 100] = 5_000

dict_maxiter["leukemia", 1000] = 100_000
dict_maxiter["mushroom", 1000] = 100_000
dict_maxiter["gina_agnostic", 1000] = 100_000
dict_maxiter["hiva_agnostic", 1000] = 100_000
dict_maxiter["upselling", 1000] = 100_000
dict_maxiter["rcv1_train", 1000] = 100_000
dict_maxiter["news20", 1000] = 1_000_000
dict_maxiter["kdda_train", 1000] = 1_000
dict_maxiter["finance", 1000] = 5_000

dict_maxiter["leukemia", 5_000] = 300_000
dict_maxiter["mushroom", 5_000] = 300_000
dict_maxiter["gina_agnostic", 5_000] = 300_000
dict_maxiter["hiva_agnostic", 5_000] = 300_000
dict_maxiter["upselling", 5000] = 100_000
dict_maxiter["rcv1_train", 5000] = 100_000
dict_maxiter["news20", 5000] = 1_000_000
dict_maxiter["kdda_train", 5000] = 1_000


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


def parallel_function(dataset_name, algo, div_alpha, div_rho):
    algo_name, use_acc, K = algo
    if dataset_name.startswith((
            'rcv1_train', 'news20', 'kdda_train', 'finance')):
        X, y = fetch_libsvm(dataset_name)
        y /= np.linalg.norm(y)
    else:
        X, y = load_openml(dataset_name)

    alpha_max = np.max(np.abs(X.T @ y))
    alpha = alpha_max / div_alpha
    tol = 1e-16
    f_gap = dict_f_gap[dataset_name]

    max_iter = dict_maxiter[dataset_name, div_alpha]

    if algo_name == 'apcg':
        w, E, gaps = apcg(
            X, y, alpha, max_iter=max_iter, tol=tol, f_gap=f_gap)
    else:
        w, E, gaps = solver_enet(
            X, y, alpha, rho=alpha/div_rho, f_gap=f_gap, max_iter=max_iter,
            tol=tol, use_acc=use_acc, K=K, algo=algo_name)

    return (
        dataset_name, algo_name, use_acc, K, div_alpha, div_rho, w, E, gaps,
        f_gap)


print("enter parallel")
backend = 'loky'
n_jobs = len(dataset_names) * len(div_alphas) * len(algos) * len(div_rhos)
n_jobs = min(n_jobs, 15)

with parallel_backend("loky", inner_max_num_threads=1):
    results = Parallel(
        n_jobs=n_jobs, verbose=100)(
        delayed(parallel_function)(
            dataset_name, algo, div_alpha, div_rho)
        for dataset_name, algo, div_alpha, div_rho in product(
            dataset_names, algos, div_alphas, div_rhos))
df = pandas.DataFrame(results)
df.columns = [
    'dataset', 'algo_name', 'use_acc', 'K', 'div_alpha', 'div_rho',
    "optimum", "E", "gaps", "f_gaps"]

for dataset_name in dataset_names:
    for div_alpha, div_rho in product(div_alphas, div_rhos):
        df_temp = df.loc[
            (df['div_alpha'] == div_alpha) & (df['div_rho'] == div_rho) &
            (df['dataset'] == dataset_name)]
        df_temp[df_temp['div_alpha'] == div_alpha].to_pickle(
            "%s_%i_%i.pkl" % (dataset_name, div_alpha, div_rho))

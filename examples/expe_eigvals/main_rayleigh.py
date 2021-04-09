from itertools import product

import os
import pandas
import scipy
import numpy as np
from numpy.linalg import norm
from libsvmdata import fetch_libsvm
from joblib import Parallel, delayed, parallel_backend

from andersoncd.utils import get_cd_mat
from andersoncd.data.real import load_openml


datasets = [
    "gina_agnostic", "rcv1.binary", "real-sim", "news20.binary"]

list_k = [0, 5, 6, 7, 8, 9, 10]

dict_rayleighs = {}


def parallel_function(dataset, k):
    print(dataset, k)
    if dataset.startswith((
            'real-sim', 'rcv1_train', 'news20.binary', 'kdda_train', 'finance')):
        X, y = fetch_libsvm(dataset)
        X = X[:, :1000]
    else:
        X, y = load_openml(dataset)

    A = get_cd_mat(X, rho=1e-3)
    A = A
    for i in range(k):
        A = A @ A
    d = A.shape[0]

    n_points = 100

    rayleighs = np.zeros(n_points + 1, dtype=complex)

    for i, theta in enumerate(
            np.linspace(0, 2 * np.pi, n_points, endpoint=False)):
        print(i)
        H = (np.exp(theta * 1j) * A + np.exp(-theta * 1j) * A.T.conj()) / 2
        _, eigvec = scipy.linalg.eigh(H, eigvals=(d-1, d-1))
        eigvec /= norm(eigvec)
        rayleighs[i] = eigvec.T.conj() @ A @ eigvec
    rayleighs[-1] = rayleighs[0]
    return dataset, k, rayleighs


n_jobs = 40

with parallel_backend("loky", inner_max_num_threads=1):
    results = Parallel(
        n_jobs=n_jobs, verbose=100)(
        delayed(parallel_function)(dataset, k)
        for dataset, k in product(datasets, list_k))


df = pandas.DataFrame(results)
df.columns = [
    'dataset', 'k', 'rayleighs']

for dataset in datasets:
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    for k in list_k:
        df_temp = df[df['dataset'] == dataset]
        df_temp[df_temp['k'] == k].to_pickle(
            "results/%s_%i.pkl" % (dataset, k))

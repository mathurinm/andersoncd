"""File to download and load real data from openml."""

import numpy as np
from numpy.linalg import norm
from sklearn.datasets import fetch_openml

PATH_openml = '~/scikit_learn_data'


def get_mushroom():
    """n_samples, n_feature = (8124, 20)
    """
    dataset = fetch_openml("mushroom", data_home=PATH_openml)
    X = np.asfortranarray(dataset.data.astype(float))
    X = clean_X(X)
    n_samples = X.shape[0]
    y = np.ones(n_samples)
    y[dataset.target == 'e'] = -1
    y = clean_y(y)
    return X, y


def get_madelon():
    """n_samples, n_feature = (2600, 500)
    """
    dataset = fetch_openml("madelon", data_home=PATH_openml)
    X = np.asfortranarray(dataset.data.astype(float))
    X = clean_X(X)
    n_samples = X.shape[0]
    y = np.ones(n_samples)
    y[dataset.target == '2'] = -1
    y = clean_y(y)
    return X, y


def get_gina_agnostic(normalize_y=False):
    """n_samples, n_feature = (3468, 970)
    """
    dataset = fetch_openml("gina_agnostic", data_home=PATH_openml)
    X = np.asfortranarray(dataset.data.astype(float))
    X = clean_X(X)
    n_samples = X.shape[0]
    y = np.ones(n_samples)
    y[dataset.target == '-1'] = -1
    if normalize_y:
        y = clean_y(y)
    return X, y


def get_upselling():
    """n_samples, n_feature = (50000, 230)
    """
    dataset = fetch_openml("KDDCup09_upselling", data_home=PATH_openml)
    X = np.asfortranarray(dataset.data.astype(float))
    X = clean_X(X)
    n_samples = X.shape[0]
    y = np.ones(n_samples)
    y[dataset.target == '-1'] = -1
    y = clean_y(y)
    return X, y


def get_leukemia(normalize_y=True):
    """n_samples, n_feature = (??)
    """
    dataset = fetch_openml("leukemia", data_home=PATH_openml)
    X = np.asfortranarray(dataset.data.astype(float))
    X = clean_X(X)
    n_samples = X.shape[0]
    y = np.ones(n_samples)
    y[dataset.target == 'ALL'] = -1
    if normalize_y:
        y = clean_y(y)
    return X, y


def get_hiva_agnostic(normalize_y=True):
    """n_samples, n_feature =  (4229, 1617)
    """
    dataset = fetch_openml("hiva_agnostic", data_home=PATH_openml)
    X = np.asfortranarray(dataset.data.astype(float))
    X = clean_X(X)
    n_samples = X.shape[0]
    y = np.ones(n_samples)
    y[dataset.target == '-1'] = -1
    if normalize_y:
        y = clean_y(y)
    return X, y


def clean_X(X, normalize=True):
    X = X[:, norm(X, axis=0) != 0]
    X = X[:, np.logical_not(np.isnan(norm(X, axis=0)))]

    if normalize:
        X -= np.mean(X, axis=0)[None, :]
        X /= norm(X, axis=0)[None, :]
    return X


def clean_y(y):
    y -= y.mean()
    y /= norm(y)
    return y


def load_openml(dataset, normalize_y=True):
    if dataset == 'madelon':
        return get_madelon()
    elif dataset == 'mushroom':
        return get_mushroom()
    elif dataset == 'gina_agnostic':
        return get_gina_agnostic(normalize_y=normalize_y)
    elif dataset == 'hiva_agnostic':
        return get_hiva_agnostic(normalize_y=normalize_y)
    elif dataset == 'upselling':
        return get_upselling()
    elif dataset == 'leukemia':
        return get_leukemia(normalize_y=normalize_y)
    else:
        raise ValueError

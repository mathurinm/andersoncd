import time
import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import Lasso as Lasso_sk
from celer import Lasso as Lasso_cl
from libsvmdata import fetch_libsvm
from scipy import sparse

from andersoncd import Lasso
from andersoncd.data import make_correlated_data
from andersoncd.solver import _kkt_violation, _kkt_violation_sparse


def kkt(X, y, w, datafit, penalty):
    if sparse.issparse(X):
        return norm(_kkt_violation_sparse(
            w, X.data, X.indptr, X.indices, y, X @ w,
            datafit, penalty, np.arange(X.shape[1])),
            ord=np.inf)
    else:
        return norm(_kkt_violation(
            w, X, y, X @ w, datafit, penalty, np.arange(X.shape[1])),
            ord=np.inf)


# dataset = "real-sim"
# dataset = "rcv1.binary"
# dataset = "finance"
dataset = "simu"

if dataset == "simu":
    # X, y, w_true = make_correlated_data(
    #     n_samples=1000, n_features=2000, density=0.1)
    X, y, _ = make_correlated_data(
        n_samples=500, n_features=10_000, random_state=0)
else:
    X, y = fetch_libsvm(dataset)

alpha_div = 10
alpha = norm(X.T @ y, np.inf) / len(y) / alpha_div

# Compute quantites using sklearn
tol_sk = 1e-3
sk = Lasso_sk(
    alpha=alpha, fit_intercept=False, max_iter=10**6, tol=tol_sk)

t0 = time.time()
sk.fit(X, y)
t_sk = time.time() - t0

obj_sk = np.mean((y - X @ sk.coef_) ** 2) / 2. + sk.alpha * norm(sk.coef_, 1)

us = Lasso(alpha=alpha, fit_intercept=False,
           max_epochs=20, max_iter=1, verbose=2,
           warm_start=False).fit(X, y)
kkt_sk = kkt(X, y, sk.coef_, us.datafit, us.penalty)


# Time other implementation
dict_estimators = {}
dict_estimators["us"] = Lasso(
    alpha=alpha, fit_intercept=False, tol=sk.tol, max_iter=1)
dict_estimators["cl"] = Lasso_cl(
    alpha=alpha, fit_intercept=False, tol=sk.tol, max_iter=1)

dict_times = {}
dict_kkt = {}
dict_obj = {}

dict_tols = {}
dict_tols["us"] = kkt_sk
dict_tols["cl"] = tol_sk

for estimator_name in dict_estimators.keys():
    estimator = dict_estimators[estimator_name]
    estimator.max_iter = 1
    estimator.fit(X, y)
    estimator.max_iter = 10_000
    estimator.tol = dict_tols[estimator_name]
    t0 = time.time()
    estimator.fit(X, y)
    t_elapsed = time.time() - t0
    dict_times[estimator_name] = t_elapsed
    dict_kkt[estimator_name] = kkt(
        X, y, estimator.coef_, us.datafit, us.penalty)
    dict_obj[estimator_name] = np.mean(
        (y - X @ estimator.coef_) ** 2) / 2. + estimator.alpha * norm(
            estimator.coef_, 1)

print(f'sk: {t_sk:.4f} s, kkt: {kkt_sk:.2e}, obj: {obj_sk:.10f}')
for estimator_name in dict_estimators.keys():
    t = dict_times[estimator_name]
    kkt = dict_kkt[estimator_name]
    obj = dict_obj[estimator_name]
    print(f'{estimator_name:s}: {t:.4f} s, kkt: {kkt:.2e}, obj: {obj:.10f}')

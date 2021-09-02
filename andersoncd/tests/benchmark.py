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


# X, y, w_true = make_correlated_data(
    # n_samples=1000, n_features=2000, density=0.1)

X, y = fetch_libsvm("rcv1.binary")
# X, y = fetch_libsvm("finance")

alpha_div = 1000
alpha = norm(X.T @ y, np.inf) / len(y) / alpha_div


sk = Lasso_sk(alpha=alpha, fit_intercept=False, max_iter=10**6, tol=1e-5)

t0 = time.time()
sk.fit(X, y)
t_sk = time.time() - t0

us = Lasso(alpha=alpha, fit_intercept=False,
           max_epochs=20, max_iter=1, verbose=2,
           warm_start=False).fit(X, y)


kkt_sk = kkt(X, y, sk.coef_, us.datafit, us.penalty)
us.tol = kkt_sk
us.max_epochs = 10000
us.max_iter = 50

t0 = time.time()
us.fit(X, y)
t_us = time.time() - t0

kkt_us = kkt(X, y, us.coef_, us.datafit, us.penalty)


cl = Lasso_cl(alpha=alpha, fit_intercept=False, tol=sk.tol)

t0 = time.time()
cl.fit(X, y)
t_cl = time.time() - t0
kkt_cl = kkt(X, y, cl.coef_, us.datafit, us.penalty)


obj_us = np.mean((y - X @ us.coef_) ** 2) / 2. + us.alpha * norm(us.coef_, 1)
obj_sk = np.mean((y - X @ sk.coef_) ** 2) / 2. + sk.alpha * norm(sk.coef_, 1)
obj_cl = np.mean((y - X @ cr.coef_) ** 2) / 2. + cr.alpha * norm(cr.coef_, 1)

print("#" * 80)
print("Setup:")
print(f'    X: {X.shape}')
print(f'    alpha_max / {alpha_div}')
print(f'    nnz in sol: {(cr.coef_ != 0).sum()}')
# print(f'    nnz in tru: {(w_true != 0).sum()}')

print(f'us: {t_us:.4f} s, kkt: {kkt_us:.2e}, obj: {obj_us:.10f}')
print(f'sk: {t_sk:.4f} s, kkt: {kkt_sk:.2e}, obj: {obj_sk:.10f}')
print(f'cl: {t_cl:.4f} s, kkt: {kkt_cl:.2e}, obj: {obj_cl:.10f}')

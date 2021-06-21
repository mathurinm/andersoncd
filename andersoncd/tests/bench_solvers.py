import time
import numpy as np
from numpy.linalg import norm

from andersoncd.solver import solver
from andersoncd.penalties import L1, WeightedL1, L1_plus_L2
from celer.datasets import make_correlated_data

X, y, _ = make_correlated_data(
    n_samples=300, n_features=10_000, snr=5, corr=0.99, random_state=1,
    density=0.2)
n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = 0.01 * alpha_max
tol = 1e-14


dict_penalties = {}
dict_penalties["Lasso"] = L1(alpha)
dict_penalties["WeightedLasso"] = WeightedL1(alpha, np.ones(n_features))
dict_penalties["ElasticNet"] = L1_plus_L2(alpha, alpha)


def bench_solver(estimator_name, use_acc):
    penalty = dict_penalties[estimator_name]
    w = np.zeros(n_features)
    R = y - X @ w
    norms_X_col = norm(X, axis=0)
    t_start = time.time()
    solver(
        X, y, penalty, w, R, norms_X_col, verbose=2, max_iter=10,
        max_epochs=1_000, tol=1e-10)
    t_ellapsed = time.time() - t_start
    print(t_ellapsed)


if __name__ == '__main__':
    bench_solver("WeightedLasso", True)

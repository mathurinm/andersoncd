from andersoncd.datafits import Quadratic
from andersoncd.penalties import L1
from andersoncd.solver import solver, solver_path
from andersoncd.celer_numba import numba_celer_primal
import numpy as np
# from kernprof import profile
from celer.datasets import make_correlated_data
from scipy.sparse import csc_matrix, issparse
from libsvmdata import fetch_libsvm


# dataset = "real-sim"
# dataset = "rcv1.binary"
dataset = "finance"
# dataset = "simu"

if dataset == "simu":
    X, y, _ = make_correlated_data(
        n_samples=500, n_features=10_000, random_state=0)
else:
    X, y = fetch_libsvm(dataset)

# X = csc_matrix(X)

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / len(y)
alpha = alpha_max / 10
pen = L1(alpha_max / 10)
datafit = Quadratic()

if issparse(X):
    datafit.initialize_sparse(X.data, X.indptr, X.indices, y)
else:
    datafit.initialize(X, y)

# numba_celer_primal(X, y, alpha, n_iter=30, tol=1e-2)
# profile(numba_celer_primal)(
#     X, y, alpha, n_iter=30, tol=1e-2)

w = np.zeros(X.shape[1])
solver(X, y, datafit, pen, w, np.zeros_like(y), verbose=2, tol=1e-2)
w = np.zeros(X.shape[1])
profile(solver)(
    X, y, datafit, pen, w, np.zeros(X.shape[0]), verbose=2, tol=1e-2)

# solver_path(X, y, datafit, pen, alphas=[alpha], tol=1e-2)
# profile(solver_path)(X, y, datafit, pen, alphas=[alpha], tol=1e-2)

# def solver(
#         X, y, datafit, penalty, w, Xw, max_iter=50,
#         max_epochs=50_000, p0=10, tol=1e-4, use_acc=True, K=5, verbose=0):

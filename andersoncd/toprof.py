from andersoncd.datafits import Quadratic
from andersoncd.penalties import L1
from andersoncd.solver import solver
import numpy as np
# from kernprof import profile
from celer.datasets import make_correlated_data

X, y, _ = make_correlated_data(n_samples=1000, n_features=1000, random_state=0)

alpha_max = np.linalg.norm(X.T @ y, ord=np.inf) / len(y)
pen = L1(alpha_max/10)
datafit = Quadratic()

w = np.zeros(X.shape[1])
datafit.initialize(X, y)

solver(X, y, datafit, pen, w, np.zeros_like(y), verbose=2)


w = np.zeros(X.shape[1])

print("#" * 80)


profile(solver)(X, y, datafit, pen, w, np.zeros(X.shape[0]), verbose=2)


# def solver(
#         X, y, datafit, penalty, w, Xw, max_iter=50,
#         max_epochs=50_000, p0=10, tol=1e-4, use_acc=True, K=5, verbose=0):

import numpy as np
from numpy.linalg import norm

from celer import LogisticRegression

from andersoncd.logreg import apcg_logreg

# data generation
np.random.seed(0)
n_samples = 11
n_features = 3
X = np.random.randn(n_samples, n_features)
y = np.sign(X @ np.random.randn(n_features))

# param algo
algo = "apcg"
use_acc = False
pCmin = 10

C_min = 2 / norm(X.T @ y, ord=np.inf)
C = pCmin * C_min
tol = 1e-14
estimator = LogisticRegression(
    C=C, verbose=1, solver="celer-pn", fit_intercept=False, tol=tol)
estimator.fit(X, y)
coef_celer = estimator.coef_.ravel()

coef_apcg, _, _ = apcg_logreg(
    X, y, alpha=1/C, tol=tol, verbose=True, max_iter=100_000)

print("coeff celer:", coef_celer)
print("coeff apcg:", coef_apcg)
# even with n_features = 3 and 100_000 iteration apcg does not identify
# the support!

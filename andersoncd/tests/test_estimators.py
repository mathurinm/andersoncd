import pytest
import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import ElasticNet as ElasticNet_sklearn
from sklearn.linear_model import LogisticRegression as LogReg_sklearn

from scipy.sparse import csc_matrix

from andersoncd.data import make_correlated_data
from andersoncd.estimators import (
    Lasso, WeightedLasso, ElasticNet, MCP, LogisticRegression)

X, y, _ = make_correlated_data(
    n_samples=200, n_features=500, density=0.1, random_state=0)

np.random.seed(0)
X_sparse = csc_matrix(X * np.random.binomial(1, 0.1, X.shape))

n_samples, n_features = X.shape
# Lasso will fit with binary values, but else logreg's alpha_max is wrong:
y = np.sign(y)
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = 0.05 * alpha_max
tol = 1e-10
l1_ratio = 0.3

dict_estimators_sk = {}
dict_estimators_ours = {}

dict_estimators_sk["Lasso"] = Lasso_sklearn(
    alpha=alpha, fit_intercept=False, tol=tol)
dict_estimators_ours["Lasso"] = Lasso(
    alpha=alpha, fit_intercept=False, tol=tol)

dict_estimators_sk["wLasso"] = Lasso_sklearn(
    alpha=alpha, fit_intercept=False, tol=tol)
dict_estimators_ours["wLasso"] = WeightedLasso(
    alpha=alpha, fit_intercept=False, tol=tol, weights=np.ones(n_features))

dict_estimators_sk["ElasticNet"] = ElasticNet_sklearn(
    alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, tol=tol)
dict_estimators_ours["ElasticNet"] = ElasticNet(
    alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, tol=tol)

dict_estimators_sk["MCP"] = Lasso_sklearn(
    alpha=alpha, fit_intercept=False, tol=tol)
dict_estimators_ours["MCP"] = MCP(
    alpha=alpha, gamma=np.inf, fit_intercept=False, tol=tol)

dict_estimators_sk["LogisticRegression"] = LogReg_sklearn(
    C=1/(alpha * n_samples), fit_intercept=False, tol=tol, penalty='l1',
    solver='liblinear', max_iter=100)
dict_estimators_ours["LogisticRegression"] = LogisticRegression(
    C=1/(alpha * n_samples), fit_intercept=False, tol=tol,
    penalty='l1', verbose=True)


@pytest.mark.parametrize(
    "estimator_name",
    ["Lasso", "wLasso", "ElasticNet", "MCP", "LogisticRegression"])
@pytest.mark.parametrize('X', [X, X_sparse])
def test_estimator(estimator_name, X):
    # lasso from sklearn
    estimator_sk = dict_estimators_sk[estimator_name]
    estimator_ours = dict_estimators_ours[estimator_name]
    if estimator_name == "LogisticRegression":
        estimator_sk.fit(X, np.sign(y))
        estimator_ours.fit(X, np.sign(y))
    else:
        estimator_sk.fit(X, y)
        estimator_ours.fit(X, y)
    coef_sk = estimator_sk.coef_
    coef_ours = estimator_ours.coef_

    # assert that something was fitted:
    np.testing.assert_array_less(1e-5, norm(coef_ours))
    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)


if __name__ == '__main__':
    test_estimator("LogisticRegression", X)

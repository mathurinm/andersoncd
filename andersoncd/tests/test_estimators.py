import pytest
import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import ElasticNet as ElasticNet_sklearn

from scipy.sparse import csc_matrix

from andersoncd.data import make_correlated_data
from andersoncd.estimators import Lasso, WeightedLasso, ElasticNet, MCP

X, y, _ = make_correlated_data(
    n_samples=500, n_features=1000, density=0.1, random_state=0)

X_sparse = csc_matrix(X)

n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = 0.05 * alpha_max
tol = 1e-14
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


@pytest.mark.parametrize(
    "estimator_name", ["Lasso", "wLasso", "ElasticNet", "MCP"])
@pytest.mark.parametrize('X', [X, X_sparse])
def test_estimator(estimator_name, X):
    # lasso from sklearn
    estimator_sk = dict_estimators_sk[estimator_name]
    estimator_sk.fit(X, y)
    coef_sk = estimator_sk.coef_

    estimator_ours = dict_estimators_ours[estimator_name]
    estimator_ours.fit(X, y)
    coef_ours = estimator_ours.coef_

    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)


if __name__ == '__main__':
    test_estimator("ElasticNet", X_sparse)

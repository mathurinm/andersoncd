import pytest
import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import ElasticNet as ElasticNet_sklearn

from andersoncd.data.synthetic import simu_linreg
from andersoncd.penalties import L1, L1_plus_L2
from andersoncd.estimators import WeightedLasso, ElasticNet

X, y = simu_linreg(n_samples=30, n_features=40)
n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = 0.05 * alpha_max
tol = 1e-14
l1_ratio = 0.3

dict_estimators_sk = {}
dict_estimators_sk["Lasso"] = Lasso_sklearn(
    alpha=alpha, fit_intercept=False, tol=tol)
dict_estimators_sk["ElasticNet"] = ElasticNet_sklearn(
    alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, tol=tol)

dict_estimators_ours = {}
dict_estimators_ours["Lasso"] = WeightedLasso(
    alpha=alpha, fit_intercept=False, tol=tol, weights=np.ones(n_features))
dict_estimators_ours["ElasticNet"] = ElasticNet(
    alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, tol=tol)

dict_penalties = {}
dict_penalties["Lasso"] = L1(alpha)
dict_penalties["ElasticNet"] = L1_plus_L2(alpha, alpha)


@pytest.mark.parametrize("estimator_name", ["Lasso", "ElasticNet"])
def test_estimator(estimator_name):
    # lasso from sklearn
    estimator_sk = dict_estimators_sk[estimator_name]
    estimator_sk.fit(X, y)
    coef_sk = estimator_sk.coef_

    estimator_ours = dict_estimators_ours[estimator_name]
    estimator_ours.fit(X, y)
    coef_ours = estimator_ours.coef_

    np.testing.assert_allclose(coef_ours, coef_sk, atol=1e-6)


if __name__ == '__main__':
    test_estimator("ElasticNet")

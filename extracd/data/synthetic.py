import numpy as np
from scipy.linalg.special_matrices import toeplitz
from sklearn.utils import check_random_state


def simu_linreg(
        coefs=None, n_samples=1000, n_features=1000, corr=0.5,
        random_state=42):
    """Simulation of a linear regression model

    Parameters
    ----------
    coefs : `numpy.array`, shape (n_features,)
        Coefficients of the model

    n_samples : `int`, default=1000
        Number of samples to simulate

    corr : `float`, default=0.5
         Correlation between features i and j is corr^|i - j|


    Returns
    -------
    X : `numpy.ndarray`, shape (n_samples, n_features)
        Simulated features matrix. It samples of a centered Gaussian
        vector with covariance given by the Toeplitz matrix

    y : `numpy.array`, shape (n_samples,)
        Simulated targets
    """
    rng = check_random_state(random_state)
    if coefs is None:
        coefs = rng.randn(n_features)
    cov = toeplitz(corr ** np.arange(0, n_features))
    X = rng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    y = X.dot(coefs) + rng.randn(n_samples)
    return X, y

# Author: Quentin Bertrand <quentin.bertrand@inria.fr>
#         Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import ElasticNet as ElasticNet_sklearn

from andersoncd.penalties import WeightedL1, L1_plus_L2
from andersoncd.solver import solver_path


class WeightedLasso(Lasso_sklearn):
    r"""
    WeightedLasso estimator based on Celer solver and primal extrapolation.
    Supports weights equal to 0, i.e. unpenalized features.

    The optimization objective for WeightedLasso is::

    (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * \sum_j weights_j |w_j|

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square.
        For numerical reasons, using ``alpha = 0`` with the
        ``WeightedLasso`` object is not advised.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    prune : 0 | 1, optional
        Whether or not to use pruning when growing working sets.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    weights : array, shape (n_features,), optional (default=None)
        Positive weights used in the L1 penalty part of the Lasso
        objective. If None, weights equal to 1 are used.

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.


    See also
    --------
    Lasso
    WeightedLassoCV

    References
    ----------
    .. [1] Q. Bertrand, M. Massias
      "Anderson acceleration of coordinate descent", AISTATS 2021,
      http://proceedings.mlr.press/v130/bertrand21a.html

    .. [2] M. Massias, A. Gramfort, J. Salmon
      "Celer: a Fast Solver for the Lasso with Dual Extrapolation", ICML 2018,
      http://proceedings.mlr.press/v80/massias18a.html


    .. [3] M. Massias, S. Vaiter, A. Gramfort, J. Salmon
      "Dual extrapolation for sparse GLMs", JMLR 2020,
      https://jmlr.org/papers/v21/19-587.html
    """

    def __init__(self, alpha=1., max_iter=100, max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, prune=True, fit_intercept=True,
                 weights=None, normalize=False, warm_start=False):
        super(WeightedLasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, normalize=normalize,
            warm_start=warm_start)
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.penalty = WeightedL1(alpha, weights)

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **kwargs):
        """Compute weighted Lasso path with Celer + primal extrapolation."""
        return solver_path(
            X, y, self.penalty, alphas=alphas, coef_init=coef_init,
            max_iter=self.max_iter, return_n_iter=return_n_iter,
            max_epochs=self.max_epochs, p0=self.p0, tol=self.tol,
            prune=self.prune, verbose=self.verbose,)


class ElasticNet(ElasticNet_sklearn):
    r"""
    Elastic net estimator based on Celer solver and primal extrapolation.

    The optimization objective for Elastic net is::

    (1 / (2 * n_samples)) * ||y - X w||^2_2 + l1_ratio * alpha * \sum_j |w_j|
    + (1 - l1_ratio) * alpha / 2 \sum_j w_j ** 2

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square.
        For numerical reasons, using ``alpha = 0`` with the
        ``ElasticNet`` object is not advised.

    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.

    max_iter : int, optional
        Maximum number of iterations (subproblem definitions)

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    verbose : bool or integer
        Amount of verbosity.

    tol : float, optional
        Stopping criterion for the optimization.

    prune : 0 | 1, optional
        Whether or not to use pruning when growing working sets.

    fit_intercept : bool, optional (default=True)
        Whether or not to fit an intercept.

    normalize : bool, optional (default=False)
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True,  the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        constant term in decision function.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.


    See also
    --------
    ElasticNetCV

    References
    ----------
    .. [1] Q. Bertrand, M. Massias
      "Anderson acceleration of coordinate descent", AISTATS 2021,
      http://proceedings.mlr.press/v130/bertrand21a.html

    .. [2] M. Massias, A. Gramfort, J. Salmon
      "Celer: a Fast Solver for the Lasso with Dual Extrapolation", ICML 2018,
      http://proceedings.mlr.press/v80/massias18a.html


    .. [3] M. Massias, S. Vaiter, A. Gramfort, J. Salmon
      "Dual extrapolation for sparse GLMs", JMLR 2020,
      https://jmlr.org/papers/v21/19-587.html
    """

    # def __init__(self, alpha=1., rho=1, max_iter=100,
    def __init__(self, alpha=1., l1_ratio=0.5, max_iter=100,
                 max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, prune=True, fit_intercept=True,
                 normalize=False, warm_start=False):
        super(ElasticNet, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, normalize=normalize,
            warm_start=warm_start)
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.penalty = L1_plus_L2(
            # alpha, rho)
            self.l1_ratio * alpha, (1 - self.l1_ratio) * alpha)

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **kwargs):
        """Compute weighted Lasso path with Celer + primal extrapolation."""
        return solver_path(
            X, y, self.penalty, alphas=alphas, coef_init=coef_init,
            max_iter=self.max_iter, return_n_iter=return_n_iter,
            max_epochs=self.max_epochs, p0=self.p0, tol=self.tol,
            prune=self.prune, verbose=self.verbose)

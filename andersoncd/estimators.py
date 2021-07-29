# Author: Quentin Bertrand <quentin.bertrand@inria.fr>
#         Mathurin Massias <mathurin.massias@gmail.com>
# License: BSD 3 clause

import numpy as np
import numbers

from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import check_classification_targets

from sklearn.linear_model import Lasso as Lasso_sklearn
from sklearn.linear_model import ElasticNet as ElasticNet_sklearn
from sklearn.linear_model import LogisticRegression as LogReg_sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

from andersoncd.penalties import L1, WeightedL1, L1_plus_L2, MCP_pen
from andersoncd.datafits import Quadratic, Logistic
from andersoncd.solver import solver_path


class Lasso(Lasso_sklearn):
    r"""
    Lasso estimator based on Celer solver and primal extrapolation.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * \sum_j |w_j|

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

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
    LassoCV

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
                 normalize=False, warm_start=False):
        super(Lasso, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, normalize=normalize,
            warm_start=warm_start)
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.datafit = Quadratic()
        self.penalty = L1(alpha)

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **kwargs):
        """Compute Lasso path with Celer + primal extrapolation."""
        return solver_path(
            X, y, self.datafit, self.penalty, alphas=alphas,
            coef_init=coef_init, max_iter=self.max_iter,
            return_n_iter=return_n_iter, max_epochs=self.max_epochs,
            p0=self.p0, tol=self.tol, prune=self.prune, verbose=self.verbose)


class WeightedLasso(Lasso_sklearn):
    r"""
    WeightedLasso estimator based on Celer solver and primal extrapolation.
    Supports weights equal to 0, i.e. unpenalized features.

    The optimization objective for WeightedLasso is::

    (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * \sum_j weights_j |w_j|

    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

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
        self.datafit = Quadratic()
        self.penalty = WeightedL1(alpha, weights)

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **kwargs):
        """Compute weighted Lasso path with Celer + primal extrapolation."""
        return solver_path(
            X, y, self.datafit, self.penalty, alphas=alphas,
            coef_init=coef_init,
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
        Penalty strength.

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

    def __init__(self, alpha=1., l1_ratio=0.5, max_iter=100,
                 max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, prune=True, fit_intercept=True,
                 normalize=False, warm_start=False):
        super(ElasticNet, self).__init__(
            alpha=alpha, l1_ratio=l1_ratio, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, normalize=normalize,
            warm_start=warm_start)
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.datafit = Quadratic()
        self.penalty = L1_plus_L2(alpha, l1_ratio)

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **kwargs):
        """Compute Elastic Net path with Celer + primal extrapolation."""
        return solver_path(
            X, y, self.datafit, self.penalty, alphas=alphas,
            coef_init=coef_init,
            max_iter=self.max_iter, return_n_iter=return_n_iter,
            max_epochs=self.max_epochs, p0=self.p0, tol=self.tol,
            prune=self.prune, verbose=self.verbose)


class MCP(Lasso_sklearn):
    r"""
    MCP estimator based on Celer solver and primal extrapolation.

    The optimization objective for MCP is, with:
    With x >= 0
    pen(x) = alpha * x - x^2 / (2 * gamma) if x =< gamma * alpha
               gamma * alpha ** 2 / 2      if x > gamma * alpha

    obj = (1 / (2 * n_samples)) * ||y - X w||^2_2 + pen(|w_j|)

    For more details see
    Coordinate descent algorithms for nonconvex penalized regression,
    with applications to biological feature selection, Breheny and Huang.


    Parameters
    ----------
    alpha : float, optional
        Penalty strength.

    gamma : float, default=3
        If gamma = 1, the prox of MCP is a hard thresholding.
        If gamma=np.inf it is a soft thresholding.
        Should be larger than (or equal to) 1

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
    MCPCV

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

    def __init__(self, alpha=1., gamma=3, max_iter=100,
                 max_epochs=50_000, p0=10,
                 verbose=0, tol=1e-4, prune=True, fit_intercept=True,
                 normalize=False, warm_start=False):
        super(MCP, self).__init__(
            alpha=alpha, tol=tol, max_iter=max_iter,
            fit_intercept=fit_intercept, normalize=normalize,
            warm_start=warm_start)
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.gamma = gamma
        self.datafit = Quadratic()
        self.penalty = MCP_pen(alpha, gamma)

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **kwargs):
        """Compute MCP path with Celer + primal extrapolation."""
        return solver_path(
            X, y, self.datafit, self.penalty, alphas=alphas,
            coef_init=coef_init,
            max_iter=self.max_iter, return_n_iter=return_n_iter,
            max_epochs=self.max_epochs, p0=self.p0, tol=self.tol,
            prune=self.prune, verbose=self.verbose)


class LogisticRegression(LogReg_sklearn):
    """
    Sparse Logistic regression scikit-learn estimator based on Celer solver.

    The optimization objective for sparse Logistic regression is::

    mean(log(1 + e^{-y_i x_i^T w})) + 1. / C * ||w||_1

    The solvers use a working set strategy. To solve problems restricted to a
    subset of features.

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.

    penalty : 'l1'.
        Other penalties are not supported.

    solver : "andersoncd"
        Other solvers are not supported.

    tol : float, optional
        Stopping criterion for the optimization: the solver runs until the
        duality gap is smaller than ``tol * len(y) * log(2)`` or the
        maximum number of iteration is reached.

    fit_intercept : bool, optional (default=False)
        Whether or not to fit an intercept. Currently True is not supported.

    max_iter : int, optional
        The maximum number of iterations (subproblem definitions)

    verbose : bool or integer
        Amount of verbosity.

    max_epochs : int
        Maximum number of CD epochs on each subproblem.

    p0 : int
        First working set size.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Only False is supported so far.


    Attributes
    ----------

    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.

        `coef_` is of shape (1, n_features) when the given problem is binary.

    intercept_ :  ndarray of shape (1,) or (n_classes,)
        constant term in decision function. Not handled yet.

    n_iter_ : int
        Number of subproblems solved by Celer to reach the specified tolerance.

    Examples
    --------
    >>> from andersoncd import LogisticRegression
    >>> clf = LogisticRegression(C=1.)
    >>> clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 1])
    LogisticRegression(C=1.0, penalty='l1', tol=0.0001, fit_intercept=False,
    max_iter=50, verbose=False, max_epochs=50000, p0=10, warm_start=False)

    >>> print(clf.coef_)
    [[0.4001237  0.01949392]]

    See also
    --------
    celer_path

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

    def __init__(
            self, C=1., penalty='l1', solver="andersoncd", tol=1e-4,
            prune=True, fit_intercept=False, max_iter=50, verbose=False,
            max_epochs=50000, p0=10, warm_start=False):
        super(LogisticRegression, self).__init__(
            tol=tol, C=C)

        self.verbose = verbose
        self.max_epochs = max_epochs
        self.p0 = p0
        self.prune = prune
        self.max_iter = max_iter
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.datafit = Logistic()
        self.penalty_ours = L1(1 / C)  # what about the n_samples ?
        # how can we fix this ?
        # # + penalty attr is already taken for sparselogreg

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """
        # TODO handle normalization, centering
        # TODO intercept
        if self.fit_intercept:
            raise NotImplementedError(
                "Fitting an intercept is not implement yet")
        # TODO support warm start
        if self.penalty != 'l1':
            raise NotImplementedError(
                'Only L1 penalty is supported, got %s' % self.penalty)

        if not isinstance(self.C, numbers.Number) or self.C <= 0:
            raise ValueError("Penalty term must be positive; got (C=%r)"
                             % self.C)
        # below are copy pasted excerpts from sklearn.linear_model._logistic
        X, y = check_X_y(X, y, accept_sparse='csr', order="C")
        check_classification_targets(y)
        enc = LabelEncoder()
        y_ind = enc.fit_transform(y)
        self.classes_ = enc.classes_
        n_classes = len(enc.classes_)

        if n_classes <= 2:
            coefs = self.path(
                X, 2 * y_ind - 1, np.array([self.C]), solver=self.solver)[1]
            self.coef_ = coefs.T  # must be [1, n_features]
            self.intercept_ = 0
            # import ipdb; ipdb.set_trace()
        else:
            self.coef_ = np.empty([n_classes, X.shape[1]])
            self.intercept_ = 0.
            multiclass = OneVsRestClassifier(self).fit(X, y)
            self.coef_ = multiclass.coef_

        return self

    def path(self, X, y, Cs, coef_init=None, return_n_iter=True, **kwargs):
        """
        Compute sparse Logistic Regression path.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Cs : ndarray
            Values of regularization strenghts for which solutions are
            computed

        coef_init : array, shape (n_features,), optional
            Initial value of the coefficients.
        """
        return solver_path(
            # TODO the choice of parametrization in the class is different
            # from the one in the penalty, this could be confusing
            X, y, self.datafit, self.penalty_ours, alphas=1 / (Cs * len(y)),
            coef_init=coef_init, max_iter=self.max_iter,
            return_n_iter=return_n_iter, max_epochs=self.max_epochs,
            p0=self.p0, tol=self.tol, prune=self.prune, verbose=self.verbose)

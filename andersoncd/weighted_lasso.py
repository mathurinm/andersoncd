import numpy as np
from numba import njit
from scipy import sparse
from numpy.linalg import norm

from sklearn.utils import check_array
from sklearn.linear_model import Lasso as Lasso_sklearn
from andersoncd.lasso import ST


class WeightedLasso(Lasso_sklearn):
    r"""
    WeightedLasso estimator based on Celer solver and primal extrapolation.
    Supports weights equal to 0, i.e. unpenalized features.

    The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - X w||^2_2 + alpha * \sum_j weights_j |w_j|

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square.
        For numerical reasons, using ``alpha = 0`` with the
        ``Lasso`` object is not advised.

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
    TODO

    References
    ----------
    .. [1] M. Massias, A. Gramfort, J. Salmon
      "Celer: a Fast Solver for the Lasso wit Dual Extrapolation", ICML 2018,
      http://proceedings.mlr.press/v80/massias18a.html

    .. [2] Q. Bertrand, M. Massias
      "Anderson acceleration of coordinate descent", AISTATS 2021,
      http://proceedings.mlr.press/v130/bertrand21a.html

    .. [3] M. Massias, S. Vaiter, A. Gramfort, J. Salmon
      "Dual extrapolation for sparse GLMs", JMLR 2020,
      https://jmlr.org/papers/v21/19-587.html
    """

    def __init__(self, alpha=1., max_iter=100, max_epochs=50000, p0=10,
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
        self.weights = weights

    def path(self, X, y, alphas, coef_init=None, return_n_iter=True, **kwargs):
        """Compute Lasso path with Celer and primal extrapolation."""
        return celer_primal_path(
            X, y, alphas=alphas, coef_init=coef_init,
            max_iter=self.max_iter, return_n_iter=return_n_iter,
            max_epochs=self.max_epochs, p0=self.p0, verbose=self.verbose,
            tol=self.tol, prune=self.prune, weights=self.weights)


def celer_primal_path(X, y, eps=1e-3, n_alphas=100, alphas=None,
                      coef_init=None, max_iter=20, max_epochs=50_000,
                      p0=10, verbose=0, tol=1e-4, prune=0, weights=None,
                      return_n_iter=False):
    r"""Compute optimization path with Celer primal as inner solver.

    With `n = len(y)` and `p = len(w)` the number of samples and features,
    the loss is:

    .. math::

        \frac{||y - X w||_2^2}{2 n} + \alpha \sum_1^p weights_j |w_j|


    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Target values

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min = 1e-3 * alpha_max``

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : ndarray, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    coef_init : ndarray, shape (n_features,) | None, optional, (default=None)
        Initial value of coefficients. If None, np.zeros(n_features) is used.

    max_iter : int, optional
        The maximum number of iterations (definition of working set and
        resolution of problem restricted to features in working set)

    max_epochs : int, optional
        Maximum number of (block) CD epochs on each subproblem.

    p0 : int, optional
        First working set size.

    verbose : bool or integer, optional
        Amount of verbosity. 0/False is silent

    tol : float, optional
        The tolerance for the optimization.

    prune : 0 | 1, optional
        Whether or not to use pruning when growing working sets.

    weights : ndarray, shape (n_features,) or (n_groups,), optional
        Feature/group weights used in the penalty. Default to array of ones.
        Features with weights equal to np.inf are ignored.

    X_offset : np.array, shape (n_features,), optional
        Used to center sparse X without breaking sparsity. Mean of each column.
        See sklearn.linear_model.base._preprocess_data().

    X_scale : np.array, shape (n_features,), optional
        Used to scale centered sparse X without breaking sparsity. Norm of each
        centered column. See sklearn.linear_model.base._preprocess_data().

    return_n_iter : bool, optional
        If True, number of iterations along the path are returned.


    Returns
    -------
    alphas : array, shape (n_alphas,)
        The alphas along the path where models are computed.

    coefs : array, shape (n_features, n_alphas)
        Coefficients along the path.

    dual_gaps : array, shape (n_alphas,)
        Duality gaps returned by the solver along the path.  TODO stop crit
    """

    if sparse.issparse(X):
        raise ValueError("Spare design matrices are not supported yet.")

    X = check_array(X, 'csc', dtype=[np.float64, np.float32],
                    order='F', copy=False, accept_large_sparse=False)
    y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                    ensure_2d=False)

    n_samples, n_features = X.shape

    # if X_offset is not None:
    #     X_sparse_scaling = X_offset / X_scale
    #     X_sparse_scaling = np.asarray(X_sparse_scaling, dtype=X.dtype)
    # else:
    #     X_sparse_scaling = np.zeros(n_features, dtype=X.dtype)

    # X_dense, X_data, X_indices, X_indptr = _sparse_and_dense(X)

    if weights is None:
        weights = np.ones(n_features, dtype=X.dtype)
    elif (weights < 0).any():
        raise ValueError("Strictly negative weights are not supported.")

    penalized = weights > 0
    if alphas is None:
        alpha_max = np.max(np.abs(
            X[:, penalized] @ y / weights[penalized])) / n_samples

        alphas = alpha_max * np.geomspace(1, eps, n_alphas, dtype=X.dtype)
    else:
        alphas = np.sort(alphas)[::-1]

    n_alphas = len(alphas)

    coefs = np.zeros((n_features, n_alphas), order='F', dtype=X.dtype)
    kkt_maxs = np.zeros(n_alphas)

    if return_n_iter:
        n_iters = np.zeros(n_alphas, dtype=int)

    norms_X_col = norm(X, axis=0)

    for t in range(n_alphas):
        alpha = alphas[t]
        if verbose:
            to_print = "##### Computing alpha %d/%d" % (t + 1, n_alphas)
            print("#" * len(to_print))
            print(to_print)
            print("#" * len(to_print))
        if t > 0:
            w = coefs[:, t - 1].copy()
            p0 = max(len(np.where(w != 0)[0]), 1)
        else:
            if coef_init is not None:
                w = coef_init.copy()
                p0 = max((w != 0.).sum(), p0)
                R = y - X @ w
            else:
                w = np.zeros(n_features, dtype=X.dtype)
                R = y.copy()

        sol = celer_primal(
            X, y, alpha, w, R, norms_X_col, weights,
            max_iter=max_iter, max_epochs=max_epochs, p0=p0, tol=tol,
            verbose=verbose)

        coefs[:, t] = w.copy()
        kkt_maxs[t] = sol[-1]
        # TODO sol[0], sol[1], sol[2][-1]
        if return_n_iter:
            n_iters[t] = len(sol[1])

    results = alphas, coefs, kkt_maxs
    if return_n_iter:
        results += (n_iters,)

    return results


# def _kkt_violation(XtR, weights, alpha):
#     for j in range(n_features):
#         if weights[j] > 0:
#             kkt[j] = max(0, np.abs(XtR[j]) / n_samples - alpha * weights[j]))
#         else:
#                 kkt_max=max(kkt_max, np.abs(XtR[j]))


def celer_primal(
        X, y, alpha, w, R, norms_X_col, weights,
        max_iter=50, max_epochs=50_000, p0=10, tol=1e-4,
        verbose=0):
    n_samples, n_features = X.shape
    pen = weights > 0
    unpen = ~pen
    n_unpen = unpen.sum()
    print(n_unpen)
    p0 = max(p0, n_unpen)
    obj_out = []
    lc = norms_X_col ** 2

    for t in range(max_iter):
        kkt = _kkt_violation(w, X, R, weights, alpha, np.arange(n_features))
        kkt_max = np.max(kkt)
        if verbose:
            print(f"KKT max violation: {kkt_max:.2e}")
        if kkt_max <= tol:
            break
        # 1) select features
        ws_size = max(p0 + n_unpen,
                      min(2 * (w != 0).sum() - n_unpen, n_features))
        kkt[unpen] = np.inf  # always include unpenalized features
        ws = np.argsort(kkt)[-ws_size:]

        if verbose:
            print(f'Iteration {t}, {ws_size} feats in subpb.')

        # 2) do iterations on smaller problem
        for epoch in range(max_epochs):
            _cd_wlasso(X, w, R, alpha, weights, lc, ws)

            if epoch % 10 == 0:
                # todo maybe we can improve here by restricting to ws
                p_obj = primal_wlasso(R, w, alpha, weights)

                kkt_ws = _kkt_violation(w, X, R, weights, alpha, ws)
                kkt_ws_max = np.max(kkt_ws)
                if max(verbose - 1, 0):
                    print(f"    Epoch {epoch}, objective {p_obj:.10f}, "
                          f"kkt {kkt_ws_max:.2e}")
                if kkt_ws_max < tol:
                    if max(verbose - 1, 0):
                        print("    Early exit")
                    break
        obj_out.append(p_obj)

    return w, np.array(obj_out), kkt_max


@njit
def _kkt_violation(w, X, R, weights, alpha, ws):
    n_samples = X.shape[0]
    kkt = np.zeros(ws.shape[0])
    for idx in range(ws.shape[0]):
        j = ws[idx]
        grad_j = X[:, j].T @ R / n_samples
        if w[j] == 0:
            # distance of grad_j to alpha * weight * [-1, 1]
            kkt[idx] = max(0, np.abs(grad_j) - alpha * weights[j])
        else:
            # distance of grad_j to alpha * weight * sign(w[j])
            kkt[idx] = np.abs(np.abs(grad_j) - alpha * weights[j])
    return kkt


@njit
def _cd_wlasso(X, w, R, alpha, weights, lc, feats):
    # we apply ST even when weights[j] == 0
    n_samples = len(R)
    for j in feats:
        Xj = X[:, j]
        old_w_j = w[j]
        w[j] = ST(old_w_j + np.dot(Xj, R) / lc[j],
                  n_samples * alpha * weights[j] / lc[j])
        if w[j] != old_w_j:
            R += (old_w_j - w[j]) * Xj


def primal_wlasso(R, w, alpha, weights):
    return 0.5 * norm(R) ** 2 / len(R) + alpha * norm(weights * w, ord=1)

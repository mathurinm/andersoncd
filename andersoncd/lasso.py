import time
import numpy as np

from scipy import sparse
from numba import njit

from numpy.linalg import norm
from scipy.sparse.linalg import svds


def primal_enet(R, w, alpha, rho=0):
    """Primal objective of the Elastic Net

    Parameters
    ----------
    R : array, shape (n_samples,)
        Residuals (y - Xw)
    w : array, shape (n_features,)
        Coefficient vector
    alpha : float
        L1 regularization strength.
    rho : float (default=0)
        L2 regularization strength.

    Returns
    -------
    p_obj : float
        Value of the objective at w.
    """
    p_obj = 0.5 * norm(R) ** 2 + alpha * norm(w, ord=1)
    if rho != 0.:
        p_obj += rho / 2. * w @ w
    return p_obj


@njit
def dual_lasso(y, theta, alpha):
    d_obj = norm(y) ** 2 / 2.
    d_obj -= norm(y / alpha - theta) ** 2 * alpha ** 2 / 2.
    return d_obj


@njit
def dual_enet(XTR, Xw, y, alpha, rho):
    d_obj = np.sum(y ** 2) / 2. - np.sum(Xw ** 2) / 2
    for j in range(XTR.shape[0]):
        d_obj -= 1. / (2 * rho) * ST(XTR[j], alpha) ** 2
    return d_obj


@njit
def ST(x, u):
    """Soft-thresholding of scalar x at level u."""
    if x > u:
        return x - u
    elif x < - u:
        return x + u
    else:
        return 0.


@njit
def ST_vec(x, u):
    return np.sign(x) * np.maximum(0., np.abs(x) - u)


@njit
def _cd_enet(X, w, R, alpha, rho, lc, feats):
    for j in feats:
        Xj = X[:, j]
        old_w_j = w[j]
        w[j] = ST(old_w_j + np.dot(Xj, R) / lc[j], alpha / lc[j])
        if rho != 0:
            w[j] /= 1 + rho / lc[j]
        if w[j] != old_w_j:
            R += (old_w_j - w[j]) * Xj


@njit
def _cd_enet_sparse(
        X_data, X_indices, X_indptr, w, R, alpha, rho, lc, feats):
    for j in feats:
        tmp = 0.
        for ix in range(X_indptr[j], X_indptr[j + 1]):
            tmp += X_data[ix] * R[X_indices[ix]]
        old_w_j = w[j]
        w[j] = ST(old_w_j + tmp / lc[j], alpha / lc[j])
        if rho != 0:
            w[j] /= 1 + rho / lc[j]
        if w[j] != old_w_j:
            for ix in range(X_indptr[j], X_indptr[j + 1]):
                R[X_indices[ix]] += (old_w_j - w[j]) * X_data[ix]


def solver_enet(
        X, y, alpha, rho=0, max_iter=10000, tol=1e-4, f_gap=10, K=5,
        use_acc=True, algo='cd', reg_amount=None, seed=0, verbose=False,
        compute_time=False, tmax=1000, w0=None):
    """Solve the Lasso/Enet with CD/ISTA/FISTA, eventually with extrapolation.

    The minimized objective is:
    norm(y - Xw, ord=2)**2 / 2 + alpha * norm(x, ord=1) + rho/2 * norm(w) ** 2

    Parameters
    ----------
    X : {array_like, sparse matrix}, shape (n_samples, n_features)
        Design matrix.

    y : ndarray, shape (n_samples,)
        Observation vector.

    alpha : float
        L1 regularization strength.

    rho : float (optional, default=0)
        L2 regularization strength.

    max_iter : int, default=1000
        Maximum number of iterations.

    tol : float, default=1e-4
        The algorithm early stops if the duality gap is less than tol.

    f_gap: int, default=10
        The gap is computed every f_gap iterations.

    K : int, default=5
        Number of points used for Anderson extrapolation.

    use_acc : bool, default=False TODO change to True?
        Whether or not to use Anderson acceleration.

    algo : {'cd', 'pgd', 'fista'}
        Algorithm used to solve the Elastic net problem.

    reg_amount : float or None (default=None)
        Amount of regularization used when solving for the extrapolation
        coefficients. None means 0 and should be preferred.

    seed : int (default=0)
        Seed for randomness.

    verbose : bool, default=False
        Verbosity.

    compute_time : bool, default=False
        If you want to compute time or not

    tmax : float, default=1000
        Maximum time (in seconds) the algorithm is allowed to run

    Returns
    -------
    W : array, shape (n_features,)
        Estimated coefficients.

    E : ndarray
        Objectives every gap_freq iterations.

    gaps : ndarray
        Duality gaps every gap_freq iterations.
    """

    is_sparse = sparse.issparse(X)
    n_samples, n_features = X.shape

    if not is_sparse and not np.isfortran(X):
        X = np.asfortranarray(X)

    _range = np.arange(n_features)
    # make them callable for shuffle:
    np.random.seed(seed)
    feats = dict(cd=lambda: _range,
                 cd2=lambda: np.hstack((_range, _range)),
                 cdsym=lambda: np.hstack((_range, _range[::-1])),
                 cdshuf=lambda: np.random.choice(
                     n_features, n_features, replace=False),
                 rcd=lambda: np.random.choice(
                     n_features, n_features, replace=True),
                 )

    if use_acc:
        last_K_w = np.zeros([K + 1, n_features])
        U = np.zeros([K, n_features])

    if algo == 'pgd' or algo == 'fista':
        if is_sparse:
            L = svds(X, k=1)[1][0] ** 2
        else:
            L = norm(X, ord=2) ** 2

    if is_sparse:
        lc = sparse.linalg.norm(X, axis=0) ** 2
    else:
        lc = (X ** 2).sum(axis=0)

    if w0 is None:
        w = np.zeros(n_features)
        R = y.copy()
    else:
        w = w0.copy()
        R = y - X @ w
    if algo == 'fista':
        z = np.zeros(n_features)
        t_new = 1


    E = []
    gaps = np.zeros(max_iter // f_gap)

    if compute_time:
        times = []
        t_start = time.time()

    for it in range(max_iter):
        if it % f_gap == 0:
            if algo == 'fista':
                R = y - X @ w
            p_obj = primal_enet(R, w, alpha, rho)
            E.append(p_obj)
            if compute_time:
                times.append(time.time() - t_start)
                if time.time() - t_start > tmax:
                    break

            if alpha != 0:
                theta = R / alpha

                if rho == 0:
                    d_norm_theta = np.max(np.abs(X.T @ theta))
                    if d_norm_theta > 1.:
                        theta /= d_norm_theta
                    d_obj = dual_lasso(y, theta, alpha)
                else:
                    XTR = X.T @ R
                    d_obj = dual_enet(XTR, y - R, y, alpha, rho)

                gap = p_obj - d_obj

                if verbose:
                    print("Iteration %d, p_obj::%.5f, d_obj::%.5f, gap::%.2e" %
                          (it, p_obj, d_obj, gap))
                gaps[it // f_gap] = gap
                if gap < tol:
                    print("Early exit")
                    break
            else:
                if verbose:
                    print("Iteration %d, p_obj::%.10f" % (it, p_obj))

        if algo.startswith(("cd", "rcd")):
            if is_sparse:
                _cd_enet_sparse(X.data, X.indices, X.indptr, w,
                                R, alpha, rho, lc, feats[algo]())
            else:
                _cd_enet(X, w, R, alpha, rho, lc, feats[algo]())

        elif algo == 'pgd':
            w[:] = ST_vec(w + X.T @ R / L, alpha / L)
            if rho != 0:
                w /= (1. + rho / L)
            R[:] = y - X @ w

        elif algo == 'fista':
            w_old = w.copy()
            w[:] = ST_vec(z - X.T @ (X @ z - y) / L, alpha / L)
            if rho != 0:
                w /= 1. + rho / L
            t_old = t_new
            t_new = (1. + np.sqrt(1 + 4 * t_old ** 2)) / 2.
            z[:] = w + (t_old - 1.) / t_new * (w - w_old)
        else:
            raise ValueError("Unknown algo %s" % algo)

        if use_acc:
            last_K_w[it % (K + 1)] = w

            if it % (K + 1) == K:
                for k in range(K):
                    U[k] = last_K_w[k + 1] - last_K_w[k]
                C = np.dot(U, U.T)
                if reg_amount is not None:
                    C += reg_amount * norm(C, ord=2) * np.eye(C.shape[0])

                try:
                    z = np.linalg.solve(C, np.ones(K))
                    c = z / z.sum()
                    w_acc = np.sum(last_K_w[:-1] * c[:, None], axis=0)
                    p_obj = primal_enet(R, w, alpha, rho)
                    R_acc = y - X @ w_acc
                    p_obj_acc = primal_enet(R_acc, w_acc, alpha, rho)
                    if p_obj_acc < p_obj:
                        w = w_acc
                        R = R_acc
                except np.linalg.LinAlgError:
                    if verbose:
                        print("----------Linalg error")
    if compute_time:
        return w, np.array(E), gaps[:it // f_gap + 1], times
    else:
        return w, np.array(E), gaps[:it // f_gap + 1]


@ njit
def _apcg(X, z, u, tau, Xu, Xz, y, alpha, rho, lc):
    n_features = X.shape[1]
    for j in np.random.choice(n_features, n_features):
        z_j_old = z[j]
        step = 1. / (lc[j] * tau * n_features)
        z[j] = ST(z[j] - X[:, j] @ (tau ** 2 * Xu + Xz - y) * step,
                  alpha * step)
        if rho != 0:
            z[j] /= 1 + rho * step
        dz = z[j] - z_j_old
        u[j] -= (1 - n_features * tau) / tau ** 2 * dz
        Xu -= (1 - n_features * tau) / tau ** 2 * dz * X[:, j]
        Xz += dz * X[:, j]
        tau_old = tau
        tau = ((tau ** 4 + 4 * tau ** 2) ** 0.5 - tau ** 2) / 2
    return tau, tau_old


@ njit
def _apcg_sparse(
        data, indices, indptr, z, u, tau, Xu, Xz, y, alpha, rho, lc,
        n_features):
    for j in np.random.choice(n_features, n_features):
        Xjs = data[indptr[j]:indptr[j+1]]
        idx_nz = indices[indptr[j]:indptr[j+1]]
        z_j_old = z[j]
        step = 1. / (lc[j] * tau * n_features)
        z[j] = ST(z[j] - Xjs @ ((tau ** 2 * Xu[idx_nz] + Xz[idx_nz] -
                                 y[idx_nz])) * step,
                  alpha * step)
        if rho != 0:
            z[j] /= 1 + rho * step
        dz = z[j] - z_j_old
        u[j] -= (1 - n_features * tau) / tau ** 2 * dz
        Xu[idx_nz] -= (1 - n_features * tau) / tau ** 2 * dz * Xjs
        Xz[idx_nz] += dz * Xjs
        tau_old = tau
        tau = ((tau ** 4 + 4 * tau ** 2) ** 0.5 - tau ** 2) / 2
    return tau, tau_old


def apcg_enet(
        X, y, alpha, rho=0, max_iter=10000, tol=1e-4, f_gap=10, verbose=False,
        compute_time=False, tmax=1000):
    """Solve the Lasso/Enet with CD/ISTA/FISTA, eventually with extrapolation.

    The minimized objective is:
    norm(y - Xw, ord=2)**2 / 2 + alpha * norm(x, ord=1) + rho/2 * norm(w) ** 2

    Parameters
    ----------
    X : {array_like, sparse matrix}, shape (n_samples, n_features)
        Design matrix

    y : ndarray, shape (n_samples,)
        Observation vector

    alpha : float
        Regularization strength for the l1 norm

    rho : float, default=0
        Regularization strength for the l2 (squared) norm

    max_iter : int, default=1000
        Maximum number of iterations

    tol : float, default=1e-4
        The algorithm early stops if the duality gap is less than tol.

    f_gap : int, default=10
        The gap is computed every f_gap iterations.

    verbose : bool, default=False
        Verbosity.

    compute_time : bool, default=False
        Whether or not to time the algorithm

    tmax : float, default=1000
        Maximum time (in seconds) the algorithm is run

    Returns
    -------
    W : array, shape (n_features,)
        Estimated coefficients.

    E : ndarray
        Objectives every gap_freq iterations.

    gaps : ndarray
        Duality gaps every gap_freq iterations.
    """

    np.random.seed(0)
    n_samples, n_features = X.shape
    is_sparse = sparse.issparse(X)
    if not is_sparse and not np.isfortran(X):
        X = np.asfortranarray(X)

    if is_sparse:
        lc = sparse.linalg.norm(X, axis=0) ** 2
    else:
        lc = (X ** 2).sum(axis=0)

    if compute_time:
        times = []
        t_start = time.time()
    # Algo 2 in Li, Lu, Xiao 2014
    w = np.zeros(n_features)
    u = np.zeros(n_features)
    z = w.copy()
    tau = 1. / n_features
    Xu = np.zeros(n_samples)
    Xz = np.zeros(n_samples)
    E = []
    gaps = np.zeros(max_iter // f_gap)

    for it in range(max_iter):
        if sparse.issparse(X):
            tau, tau_old = _apcg_sparse(
                X.data, X.indices, X.indptr, z, u, tau, Xu, Xz, y, alpha, rho,
                lc, n_features)
        else:
            tau, tau_old = _apcg(X, z, u, tau, Xu, Xz, y, alpha, rho, lc)

        if it % f_gap == 0:
            w = tau_old ** 2 * u + z
            R = y - X @ w  # MM: todo this is brutal if f_gap = 1

            p_obj = primal_enet(R, w, alpha, rho)
            E.append(p_obj)
            if compute_time:
                times.append(time.time() - t_start)
                if time.time() - t_start > tmax:
                    break

            if np.abs(p_obj) > np.abs(E[0] * 1e3):
                break

            if alpha != 0:
                theta = R / alpha
                if rho == 0:
                    d_norm_theta = np.max(np.abs(X.T @ theta))
                    if d_norm_theta > 1.:
                        theta /= d_norm_theta
                    d_obj = dual_lasso(y, theta, alpha)
                else:
                    XTR = X.T @ R
                    d_obj = dual_enet(XTR, y - R, y, alpha, rho)

                gap = p_obj - d_obj

                if verbose:
                    print("Iteration %d, p_obj::%.5f, d_obj::%.5f, gap::%.2e" %
                          (it, p_obj, d_obj, gap))
                gaps[it // f_gap] = gap
                if gap < tol:
                    print("Early exit")
                    break
            else:
                if verbose:
                    print("Iteration %d, p_obj::%.10f" % (it, p_obj))

    if compute_time:
        return w, np.array(E), gaps[:it // f_gap + 1], times
    else:
        return w, np.array(E), gaps[:it // f_gap + 1]

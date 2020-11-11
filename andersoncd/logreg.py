import numpy as np

from scipy import sparse
from numba import njit

from numpy.linalg import norm

from andersoncd.utils import power_method
from andersoncd.lasso import ST, ST_vec


def primal_logreg(Xw, y, w, alpha, rho=0):
    return (np.log(1 + np.exp(-y * Xw)).sum() + alpha * norm(w, ord=1) +
            rho * w @ w / 2.)


@njit
def xlogx(x):
    if x < 1e-10:
        return 0.
    else:
        return x * np.log(x)


@njit
def negative_ent(x):
    if 0. <= x <= 1.:
        return xlogx(x) + xlogx(1. - x)
    else:
        return np.inf


@njit
def dual_logreg(y, theta, alpha):
    d_obj = 0
    for i in range(y.shape[0]):
        d_obj -= negative_ent(alpha * y[i] * theta[i])
    return d_obj


@njit
def sigmoid(x):
    return 1 / (1. + np.exp(- x))


@njit
def _cd_logreg(X, w, Xw, y, alpha, rho, lc, feats):
    for j in feats:
        Xj = X[:, j]
        old_w_j = w[j]
        w[j] = ST(old_w_j + y * Xj @ sigmoid(-y * Xw) / lc[j], alpha / lc[j])
        if rho != 0:
            w[j] /= 1 + rho / lc[j]

        if w[j] != old_w_j:
            Xw += (w[j] - old_w_j) * Xj


@njit
def _cd_logreg_sparse(
        X_data, X_indices, X_indptr, w, Xw, y, alpha, rho, lc, feats):
    for j in feats:
        grad = 0.
        for i in range(X_indptr[j], X_indptr[j + 1]):
            idx = X_indices[i]
            grad -= X_data[i] * y[idx] * sigmoid(-y[idx] * Xw[idx])
        old_w_j = w[j]
        w[j] = ST(old_w_j - grad / lc[j], alpha / lc[j])
        if rho != 0:
            w[j] /= 1 + rho / lc[j]

        if w[j] != old_w_j:
            for ix in range(X_indptr[j], X_indptr[j + 1]):
                Xw[X_indices[ix]] += (w[j] - old_w_j) * X_data[ix]


def solver_logreg(
        X, y, alpha, rho=0, max_iter=10000, tol=1e-4, f_gap=10, K=5,
        use_acc=True, algo='cd', seed=0, reg_amount=None, verbose=False):
    """Solve the sparse logistic regression with CD/ISTA/FISTA,
    eventually with extrapolation.

    Objective:
    sum_1^n_samples log(1 + e^{-y_i x_i^T w}) + alpha * norm(x, ord=1)
    + rho/2 * norm(w) ** 2

    Parameters:
    algo: string
        'cd', 'pgd', 'fista'

    alpha: strength of the l1 penalty

    rho: strength of the squared l2 penalty
    """
    np.random.seed(seed)

    is_sparse = sparse.issparse(X)
    n_features = X.shape[1]
    _range = np.arange(n_features)
    feats = dict(cd=lambda: _range,
                 cd2=lambda: np.hstack((_range, _range)),
                 cdsym=lambda: np.hstack((_range, _range[::-1])),
                 cdshuf=lambda: np.random.choice(n_features, n_features,
                                                 replace=False))

    if not is_sparse and not np.isfortran(X):
        X = np.asfortranarray(X)

    if use_acc:
        last_K_w = np.zeros([K + 1, n_features])
        U = np.zeros([K, n_features])

    if algo == 'pgd' or algo == 'fista':
        if is_sparse:
            L = power_method(X, max_iter=1000) ** 2 / 4
        else:
            L = norm(X, ord=2) ** 2 / 4

    if is_sparse:
        lc = sparse.linalg.norm(X, axis=0) ** 2 / 4
    else:
        lc = (X ** 2).sum(axis=0) / 4

    w = np.zeros(n_features)
    if algo == 'fista':
        z = np.zeros(n_features)
        t_new = 1

    Xw = np.zeros(len(y))
    E = []
    gaps = np.zeros(max_iter // f_gap)

    for it in range(max_iter):
        if it % f_gap == 0:
            if algo == 'fista':
                Xw = X @ w
            p_obj = primal_logreg(Xw, y, w, alpha, rho)
            E.append(p_obj)

            if alpha != 0:
                theta = y * sigmoid(-y * Xw) / alpha

                d_norm_theta = np.max(np.abs(X.T @ theta))
                if d_norm_theta > 1.:
                    theta /= d_norm_theta
                d_obj = dual_logreg(y, theta, alpha)

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

        if algo.startswith('cd'):
            if is_sparse:
                _cd_logreg_sparse(X.data, X.indices, X.indptr, w, Xw, y,
                                  alpha, rho, lc, feats[algo]())
            else:
                _cd_logreg(X, w, Xw, y, alpha, rho, lc, feats[algo]())

        elif algo == 'pgd':
            w[:] = ST_vec(
                w + 1. / L * X.T @ (y / (1. + np.exp(y * Xw))), alpha / L)
            if rho != 0:
                w /= 1. + rho / L
            Xw[:] = X @ w

        elif algo == 'fista':
            w_old = w.copy()
            w[:] = ST_vec(
                z + X.T @ (y / (1. + np.exp(y * (X @ z)))) / L, alpha / L)
            if rho != 0:
                w /= 1. + rho / L
            t_old = t_new
            t_new = (1. + np.sqrt(1 + 4 * t_old ** 2)) / 2.
            z[:] = w + (t_old - 1.) / t_new * (w - w_old)
        else:
            raise ValueError("Unknown algo %s" % algo)

        if use_acc:
            if it < K + 1:
                last_K_w[it] = w
            else:
                for k in range(K):
                    last_K_w[k] = last_K_w[k + 1]
                last_K_w[K] = w

                for k in range(K):
                    U[k] = last_K_w[k + 1] - last_K_w[k]

                C = np.dot(U, U.T)
                if reg_amount is not None:
                    C += reg_amount * norm(C, ord=2) * np.eye(C.shape[0])
                try:
                    z = np.linalg.solve(C, np.ones(K))
                    c = z / z.sum()
                    w_acc = np.sum(last_K_w[:-1] * c[:, None],
                                   axis=0)
                    p_obj = primal_logreg(Xw, y, w, alpha, rho)
                    Xw_acc = X @ w_acc
                    p_obj_acc = primal_logreg(Xw_acc, y, w_acc, alpha, rho)
                    if p_obj_acc < p_obj:
                        w = w_acc
                        Xw = Xw_acc
                except np.linalg.LinAlgError:
                    if verbose:
                        print("----------Linalg error")

    return w, np.array(E), gaps[:it // f_gap + 1]


@ njit
def _apcg(X, z, u, tau, Xu, Xz, y, alpha, lc):
    n_features = X.shape[1]
    for j in np.random.choice(n_features, n_features):
        z_j_old = z[j]
        step = 1. / (lc[j] * tau * n_features)
        z[j] = ST(z[j] +
                  y * X[:, j] @ sigmoid(- y * (tau ** 2 * Xu + Xz)) * step,
                  alpha * step)
        dz = z[j] - z_j_old
        u[j] -= (1 - n_features * tau) / tau ** 2 * dz
        Xu -= (1 - n_features * tau) / tau ** 2 * dz * X[:, j]
        Xz += dz * X[:, j]
        tau_old = tau
        tau = ((tau ** 4 + 4 * tau ** 2) ** 0.5 - tau ** 2) / 2
    return tau, tau_old


@ njit
def _apcg_sparse(
        data, indices, indptr, z, u, tau, Xu, Xz, y, alpha, lc, n_features):
    for j in np.random.choice(n_features, n_features):
        Xj = data[indptr[j]:indptr[j+1]]
        idx_nz = indices[indptr[j]:indptr[j+1]]
        z_j_old = z[j]
        step = 1. / (lc[j] * tau * n_features)
        grad = 0
        for i in range(indptr[j], indptr[j + 1]):
            idx = indices[i]
            grad -= data[i] * y[idx] * \
                sigmoid(-y[idx] * (tau ** 2 * Xu[idx] + Xz[idx]))

        z[j] = ST(z[j] - grad * step, alpha * step)
        dz = z[j] - z_j_old
        u[j] -= (1 - n_features * tau) / tau ** 2 * dz
        Xu[idx_nz] -= (1 - n_features * tau) / tau ** 2 * dz * Xj
        Xz[idx_nz] += dz * Xj
        tau_old = tau
        tau = ((tau ** 4 + 4 * tau ** 2) ** 0.5 - tau ** 2) / 2
    return tau, tau_old


def apcg_logreg(X, y, alpha, max_iter=10000, tol=1e-4, f_gap=10,
                verbose=False, seed=42):
    """Solve Logistic regression with accelerated proximal coordinate gradient.
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape
    is_sparse = sparse.issparse(X)
    if not is_sparse and not np.isfortran(X):
        X = np.asfortranarray(X)

    if is_sparse:
        lc = sparse.linalg.norm(X, axis=0) ** 2 / 4.
    else:
        lc = (X ** 2).sum(axis=0) / 4.

    w = np.zeros(n_features)
    u = np.zeros(n_features)
    z = w.copy()
    tau = 1. / n_features
    Xu = np.zeros(n_samples)
    Xz = np.zeros(n_samples)
    E = []
    gaps = np.zeros(max_iter // f_gap)

    for it in range(max_iter):
        if is_sparse:
            tau, tau_old = _apcg_sparse(
                X.data, X.indices, X.indptr, z, u, tau, Xu, Xz, y, alpha, lc,
                n_features)
        else:
            tau, tau_old = _apcg(X, z, u, tau, Xu, Xz, y, alpha, lc)

        if it % f_gap == 0:
            w = tau_old ** 2 * u + z
            Xw = X @ w
            p_obj = primal_logreg(Xw, y, w, alpha)
            E.append(p_obj)

            if np.abs(p_obj) > np.abs(E[0] * 1e3):
                break

            if alpha != 0:
                theta = y * sigmoid(-y * Xw) / alpha

                d_norm_theta = np.max(np.abs(X.T @ theta))
                if d_norm_theta > 1.:
                    theta /= d_norm_theta
                d_obj = dual_logreg(y, theta, alpha)

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

    return w, np.array(E), gaps[:it // f_gap + 1]

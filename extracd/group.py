import numpy as np

from scipy import sparse
from numba import njit

from numpy.linalg import norm

from extracd.utils import power_method
from extracd.lasso import dual_lasso


def primal_grp(R, w, alpha, grp_size):
    return (0.5 * norm(R) ** 2 + alpha *
            norm(w.reshape(-1, grp_size), axis=1).sum())


@njit
def BST(x, u):
    """Block soft-thresholding of vector x at level u."""
    norm_x = norm(x)
    if norm_x < u:
        return np.zeros_like(x)
    else:
        return (1 - u / norm_x) * x


# @njit
def BST_vec(x, u, grp_size):
    norm_grp = norm(x.reshape(-1, grp_size), axis=1)
    scaling = np.maximum(1 - u / norm_grp, 0)
    return (x.reshape(-1, grp_size) * scaling[:, None]).reshape(x.shape[0])


# @njit
def _bcd(X, w, R, alpha, lc):
    grp_size = w.shape[0] // lc.shape[0]
    for g in range(lc.shape[0]):
        grp = slice(g * grp_size, (g + 1) * grp_size)
        Xg = X[:, grp]
        old_w_g = w[grp].copy()
        w[grp] = BST(old_w_g + Xg.T @ R / lc[g], alpha / lc[g])
        if norm(w[grp] - old_w_g) != 0:
            R += ((old_w_g - w[grp])[None, :] * Xg).sum(axis=1)


@njit
def _bcd_sparse(
        X_data, X_indices, X_indptr, w, R, alpha, lc):
    grp_size = w.shape[0] // lc.shape[0]
    grad = np.zeros(grp_size)

    for g in range(lc.shape[0]):
        grad.fill(0)
        grp = slice(g * grp_size, (g + 1) * grp_size)

        for j in range(grp_size * g, grp_size * (g + 1)):
            for ix in range(X_indptr[j], X_indptr[j + 1]):
                grad[j % g] += X_data[ix] * R[X_indices[ix]]
        old_w_g = w[grp].copy()
        w[grp] = BST(old_w_g + grad / lc[g], alpha / lc[g])
        if norm(w[grp] - old_w_g) != 0:
            for j in range(g * grp_size, (g + 1) * grp_size):
                for ix in range(X_indptr[j], X_indptr[j + 1]):
                    R[X_indices[ix]] += (old_w_g[j % grp_size] -
                                         w[j % grp_size]) * X_data[ix]


def solver_group(
        X, y, alpha, grp_size, max_iter=10000, tol=1e-4, f_gap=10, K=5,
        use_acc=False, algo='bcd', return_all=False):
    """Solve the GroupLasso with BCD/ISTA/FISTA, eventually with extrapolation.

    Groups are contiguous, of size grp_size.
    Objective:
    norm(y - Xw, ord=2)**2 / 2 + alpha * sum_g ||w_{[g]}||_2

    Parameters:
    algo: string
        'bcd', 'pgd', 'fista'

    alpha: strength of the group penalty
    """
    if return_all:
        iterates = []

    is_sparse = sparse.issparse(X)
    n_features = X.shape[1]
    if n_features % grp_size != 0:
        raise ValueError("n_features is not a multiple of group size")
    n_groups = n_features // grp_size

    if not is_sparse and not np.isfortran(X):
        X = np.asfortranarray(X)

    last_K_w = np.zeros([K + 1, n_features])
    U = np.zeros([K, n_features])

    if algo in ('pgd', 'fista'):
        if is_sparse:
            L = power_method(X, max_iter=1000) ** 2
        else:
            L = norm(X, ord=2) ** 2

    lc = np.zeros(n_groups)
    for g in range(n_groups):
        X_g = X[:, g * grp_size: (g + 1) * grp_size]
        if is_sparse:
            gram = (X_g.T @ X_g).todense()
            lc[g] = norm(gram, ord=2)
        else:
            lc[g] = norm(X_g, ord=2) ** 2

    w = np.zeros(n_features)
    if algo == 'fista':
        z = np.zeros(n_features)
        t_new = 1

    R = y.copy()
    E = []
    gaps = np.zeros(max_iter // f_gap)

    for it in range(max_iter):
        if return_all:
            iterates.append(w.copy())
        if it % f_gap == 0:
            if algo == 'fista':
                R = y - X @ w
            p_obj = primal_grp(R, w, alpha, grp_size)
            E.append(p_obj)
            theta = R / alpha

            d_norm_theta = np.max(
                norm((X.T @ theta).reshape(-1, grp_size), axis=1))
            if d_norm_theta > 1.:
                theta /= d_norm_theta
            d_obj = dual_lasso(y, theta, alpha)

            gap = p_obj - d_obj

            print("Iteration %d, p_obj::%.5f, d_obj::%.5f, gap::%.2e" %
                  (it, p_obj, d_obj, gap))
            gaps[it // f_gap] = gap
            if gap < tol:
                print("Early exit")
                break

        if algo == 'bcd':
            if is_sparse:
                _bcd_sparse(
                    X.data, X.indices, X.indptr, w, R, alpha, lc)
            else:
                _bcd(X, w, R, alpha, lc)
        elif algo == 'pgd':
            w[:] = BST_vec(w + 1. / L * X.T @ R, alpha / L, grp_size)
            R[:] = y - X @ w
        elif algo == 'fista':
            w_old = w.copy()
            w[:] = BST_vec(z - X.T @ (X @ z - y) / L, alpha / L, grp_size)
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
                last_K_w[K - 1] = w

                for k in range(K):
                    U[k] = last_K_w[k + 1] - last_K_w[k]
                C = np.dot(U, U.T)

                try:
                    z = np.linalg.solve(C, np.ones(K))
                    c = z / z.sum()
                    w_acc = np.sum(last_K_w[:-1] * c[:, None],
                                   axis=0)
                    p_obj = primal_grp(R, w, alpha, grp_size)
                    R_acc = y - X @ w_acc
                    p_obj_acc = primal_grp(R_acc, w_acc, alpha, grp_size)
                    if p_obj_acc < p_obj:
                        w = w_acc
                        R = R_acc
                except np.linalg.LinAlgError:
                    print("----------Linalg error")

    if return_all:
        return w, np.array(E), gaps[:it // f_gap + 1], np.array(iterates)
    else:
        return w, np.array(E), gaps[:it // f_gap + 1]

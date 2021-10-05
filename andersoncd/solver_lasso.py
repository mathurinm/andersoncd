
import numpy as np
from scipy import sparse
from numpy.linalg import norm
from numba import njit


@njit
def dual_lasso(alpha, norm_y2, theta, y):
    d_obj = 0
    n_samples = theta.shape[0]
    for i in range(n_samples):
        d_obj -= (y[i] / (alpha * n_samples) - theta[i]) ** 2
    d_obj *= 0.5 * alpha ** 2 * n_samples
    d_obj += norm_y2 / (2. * n_samples)
    return d_obj


@njit
def primal_lasso(alpha, R, w):
    return (R @ R) / (2 * len(R)) + alpha * np.sum(np.abs(w))


@njit
def ST(x, tau):
    if x > tau:
        return x - tau
    elif x < -tau:
        return x + tau
    return 0


@njit
def create_dual_pt(alpha, out, R):
    """Copied from cython code hence the unpythonic way to do it."""
    n_samples = R.shape[0]
    scal = 1 / (alpha * n_samples)
    out[:] = R
    out *= scal


@njit
def dnorm_l1(theta, X, skip):
    dnorm = 0
    XT_theta = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        if not skip[j]:
            Xj_theta = X[:, j] @ theta
            dnorm = max(dnorm, np.abs(Xj_theta))
            XT_theta[j] = Xj_theta
    return dnorm, XT_theta


@njit
def dnorm_l1_sparse(theta, X_data, X_indices, X_indptr, skip):
    dnorm = 0
    XT_theta = np.zeros(X_indptr.shape[0] - 1)
    for j in range(X_indptr.shape[0] - 1):
        if not skip[j]:
            Xj_theta = 0
            for ix in range(X_indptr[j], X_indptr[j + 1]):
                Xj_theta += X_data[ix] * theta[X_indices[ix]]
            dnorm = max(dnorm, np.abs(Xj_theta))
            XT_theta[j] = Xj_theta
    return dnorm, XT_theta


@njit
def set_prios(X, norms_X_col, prios, screened, radius, n_screened, XT_theta):
    n_features = X.shape[1]
    for j in range(n_features):
        if screened[j] or norms_X_col[j] == 0:
            prios[j] = np.inf
            continue
        prios[j] = (1. - np.abs(XT_theta[j])) / norms_X_col[j]
        if prios[j] > radius:
            screened[j] = True
            n_screened += 1
    return n_screened


@njit
def set_prios_sparse(X_indptr, norms_X_col, prios, screened, radius,
                     n_screened, XT_theta):
    for j in range(X_indptr.shape[0] - 1):
        if screened[j] or norms_X_col[j] == 0:
            prios[j] = np.inf
            continue
        prios[j] = (1 - np.abs(XT_theta[j])) / norms_X_col[j]
        if prios[j] > radius:
            screened[j] = True
            n_screened += 1
    return n_screened


@njit
def create_accel_pt(
        epoch, gap_freq, alpha, R, out, last_K_R, U, UtU, verbose):
    K = U.shape[0] + 1
    n_samples = R.shape[0]
    tmp = 1. / (n_samples * alpha)

    if epoch // gap_freq < K:
        last_K_R[(epoch // gap_freq), :] = R
    else:
        for k in range(K - 1):
            last_K_R[k, :] = last_K_R[k + 1, :]
        last_K_R[K - 1, :] = R
        for k in range(K - 1):
            U[k] = last_K_R[(k + 1), :] - last_K_R[k, :]

        # double for loop but small : K**2/2
        for k in range(K - 1):
            for j in range(k, K - 1):
                UtU[k, j] = U[k] @ U[j]
                UtU[j, k] = UtU[k, j]

        try:
            anderson = np.linalg.solve(UtU, np.ones(UtU.shape[0]))
        except Exception:
            # np.linalg.LinAlgError
            # Numba only accepts Error/Exception inheriting from the generic
            # Exception class
            if verbose:
                print("Singular matrix when computing accelerated point.")
        else:
            anderson /= np.sum(anderson)

            out[:] = 0
            for k in range(K - 1):
                out += anderson[k] * last_K_R[k, :]

            out *= tmp
            # out now holds the extrapolated dual point:
            # LASSO: R_acc / (alpha * n_samples)


@njit
def create_accel_primal_pt(epoch, gap_freq, w, out, last_K_w, U, UtU, verbose):
    K = U.shape[0] + 1

    if epoch // gap_freq < K:
        last_K_w[(epoch // gap_freq), :] = w
    else:
        for k in range(K - 1):
            last_K_w[k, :] = last_K_w[k + 1, :]
        last_K_w[K - 1, :] = w
        for k in range(K - 1):
            U[k] = last_K_w[(k + 1), :] - last_K_w[k, :]

        # double for loop but small : K**2/2
        for k in range(K - 1):
            for j in range(k, K - 1):
                UtU[k, j] = U[k] @ U[j]
                UtU[j, k] = UtU[k, j]

        try:
            anderson = np.linalg.solve(UtU, np.ones(UtU.shape[0]))
        except Exception:
            # np.linalg.LinAlgError
            # Numba only accepts Error/Exception inheriting from the generic
            # Exception class
            if verbose:
                print("Singular matrix when computing accelerated point. "
                      "Skipped.")
        else:
            anderson /= np.sum(anderson)
            out[:] = 0
            for k in range(K - 1):
                out += anderson[k] * last_K_w[k, :]


@njit
def create_ws(prune, w, prios, p0, t, screened, C, n_screened, prev_ws_size):
    n_features = w.shape[0]
    if prune:
        nnz = 0
        for j in range(n_features):
            if w[j] != 0:
                prios[j] = -1.
                nnz += 1

        if t == 0:
            ws_size = p0 if nnz == 0 else nnz
        else:
            ws_size = 2 * nnz

    else:
        for j in range(n_features):
            if w[j] != 0:
                prios[j] = -1  # include active features
        if t == 0:
            ws_size = p0
        else:
            for j in range(prev_ws_size):
                if not screened[C[j]]:
                    # include previous features, if not screened
                    prios[C[j]] = -1
            ws_size = 2 * prev_ws_size
    if ws_size > n_features - n_screened:
        ws_size = n_features - n_screened

    return ws_size


@njit
def cd_epoch(C, norms_X_col, X, R, alpha, w, inv_lc, n_samples):
    for j in C:
        if norms_X_col[j] == 0.:
            continue
        old_w_j = w[j]

        w[j] += X[:, j] @ R * inv_lc[j]

        w[j] = ST(w[j], alpha * inv_lc[j] * n_samples)

        # R -= (w_j - old_w_j) (X[:, j]
        diff = old_w_j - w[j]
        if diff != 0.:
            R += diff * X[:, j]


@njit
def cd_epoch_sparse(C, norms_X_col, X_data, X_indices, X_indptr, R, alpha, w,
                    inv_lc, n_samples):
    for j in C:
        if norms_X_col[j] == 0.:
            continue
        old_w_j = w[j]

        grad = 0.
        for ix in range(X_indptr[j], X_indptr[j + 1]):
            grad += X_data[ix] * R[X_indices[ix]]

        w[j] = ST(w[j] + grad * inv_lc[j], alpha * inv_lc[j] * n_samples)

        diff = old_w_j - w[j]
        if diff != 0.:
            for ix in range(X_indptr[j], X_indptr[j + 1]):
                R[X_indices[ix]] += diff * X_data[ix]


@njit
def compute_residual_sparse(X_data, X_indices, X_indptr, y, w):
    R = y.copy()
    for j in range(X_indptr.shape[0] - 1):
        if w[j] != 0:
            for ix in range(X_indptr[j], X_indptr[j + 1]):
                R[X_indices[ix]] -= w[j] * X_data[ix]
    return R


@njit
def compute_residual(X, y, w):
    R = y.copy()
    n_features = X.shape[1]
    for j in range(n_features):
        if w[j] != 0:
            R[:] -= w[j] * X[:, j]
    return R


@njit
def sparse_norm(X_data, X_indptr):
    n_features = X_indptr.shape[0] - 1
    norms = np.zeros(n_features)
    for j in range(n_features):
        sum_sq = 0
        for ix in range(X_indptr[j], X_indptr[j + 1]):
            sum_sq += X_data[ix] ** 2
        norms[j] = np.sqrt(sum_sq)
    return norms


def numba_celer_primal(
        X, y, alpha, n_iter, p0=10, tol=1e-12, prune=True,
        gap_freq=10, max_epochs=100_000, verbose=0):
    is_sparse = sparse.issparse(X)

    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    R = y.copy()
    verbose_in = max(verbose - 1, 0)
    n_screened = 0

    # tol *= norm(y) ** 2 / n_samples
    if p0 > n_features:
        p0 = n_features

    prios = np.empty(n_features, dtype=X.dtype)
    screened = np.zeros(n_features, dtype=np.int32)
    notin_WS = np.zeros(n_features, dtype=np.int32)

    # acceleration variables:
    K = 6
    last_K_w = np.empty((K, n_features), dtype=X.dtype)
    U = np.empty((K - 1, n_features), dtype=X.dtype)
    UtU = np.empty((K - 1, K - 1), dtype=X.dtype)

    if is_sparse:
        norms_X_col = sparse_norm(X.data, X.indptr)
    else:
        norms_X_col = norm(X, axis=0)

    inv_lc = 1 / norms_X_col ** 2
    norm_y2 = norm(y) ** 2

    gaps = np.zeros(n_iter, dtype=X.dtype)

    theta = np.zeros(n_samples, dtype=X.dtype)
    theta_in = np.zeros(n_samples, dtype=X.dtype)

    wacc = np.zeros(n_features, dtype=X.dtype)
    d_obj_from_inner = 0.0

    all_features = np.arange(n_features, dtype=np.int32)
    C = all_features.copy()  # weird init needed by numba ?
    ws_size = p0  # just to pass something to create_ws at iter 0

    for t in range(n_iter):
        # if t != 0:
        create_dual_pt(alpha, theta, R)

        if is_sparse:
            scal, XT_theta = dnorm_l1_sparse(
                theta, X.data, X.indices, X.indptr, screened)
        else:
            scal, XT_theta = dnorm_l1(theta, X, screened)

        if scal > 1.0:
            theta /= scal
            XT_theta /= scal
        d_obj = dual_lasso(alpha, norm_y2, theta, y)

        # also test dual point returned by inner solver after 1st iter:
        if is_sparse:
            scal, XT_theta_in = dnorm_l1_sparse(
                theta_in, X.data, X.indices, X.indptr, screened)
        else:
            scal, XT_theta_in = dnorm_l1(theta_in, X, screened)

        if scal > 1.:
            theta_in /= scal
            XT_theta_in /= scal

        d_obj_from_inner = dual_lasso(alpha, norm_y2, theta_in, y)

        if d_obj_from_inner > d_obj:
            d_obj = d_obj_from_inner
            theta[:] = theta_in
            XT_theta[:] = XT_theta_in

        highest_d_obj = d_obj

        p_obj = primal_lasso(alpha, R, w)
        gap = p_obj - highest_d_obj
        gaps[t] = gap
        if verbose:
            print("Iter {:d}: primal {:.10f}, gap {:.2e}".format(
                  t, p_obj, gap), end="")

        if gap <= tol:
            if verbose:
                print("\nEarly exit, gap: {:.2e} < {:.2e}".format(gap, tol))
            break

        radius = np.sqrt(2 * gap / n_samples) / alpha

        if is_sparse:
            n_screened = set_prios_sparse(X.indptr, norms_X_col, prios,
                                          screened, radius, n_screened,
                                          XT_theta)
        else:
            n_screened = set_prios(X, norms_X_col, prios, screened, radius,
                                   n_screened, XT_theta)

        ws_size = create_ws(prune, w, prios, p0, t, screened, C, n_screened,
                            ws_size)
        # if ws_size === n_features then argpartition will break:
        if ws_size == n_features:
            C = all_features
        else:
            C = np.argpartition(np.asarray(prios), ws_size)[
                :ws_size].astype(np.int32)

        notin_WS.fill(1)
        notin_WS[C] = 0

        if prune:
            tol_in = 0.3 * gap
        else:
            tol_in = tol

        if verbose:
            print(", {:d} feats in subpb ({:d} left)".format(
                  len(C), n_features - n_screened))

        # calling inner solver which will modify w and R inplace
        highest_d_obj_in = 0
        for epoch in range(max_epochs):
            if epoch != 0 and epoch % gap_freq == 0:
                create_dual_pt(alpha, theta_in, R)

                if is_sparse:
                    scal = dnorm_l1_sparse(theta_in, X.data, X.indices,
                                           X.indptr, notin_WS)[0]
                else:
                    scal = dnorm_l1(theta_in, X, notin_WS)[0]

                if scal > 1.:
                    theta_in /= scal

                d_obj_in = dual_lasso(alpha, norm_y2, theta_in, y)
                p_obj_in = primal_lasso(alpha, R, w)

                if True:  # also compute accelerated primal point
                    create_accel_primal_pt(epoch, gap_freq, w, wacc, last_K_w,
                                           U, UtU, verbose)

                    if epoch // gap_freq >= K:
                        p_obj_accel = primal_lasso(alpha, R, wacc)

                        if p_obj_accel < p_obj_in:
                            p_obj_in = p_obj_accel
                            w[:] = wacc
                            if is_sparse:
                                R = compute_residual_sparse(X.data, X.indices,
                                                            X.indptr, y, w)
                            else:
                                R = compute_residual(X, y, w)

                if d_obj_in > highest_d_obj_in:
                    highest_d_obj_in = d_obj_in

                # CAUTION: code does not yet  include a best_theta.
                # Can be an issue in screening: dgap and theta might disagree.

                p_obj_in = primal_lasso(alpha, R, w)  # TODO maybe small
                # improvement here
                gap_in = p_obj_in - highest_d_obj_in

                if verbose_in:
                    print("Epoch {:d}, primal {:.10f}, gap: {:.2e}".format(
                          epoch, p_obj_in, gap_in))
                if gap_in < tol_in:
                    if verbose_in:
                        print("Exit epoch {:d}, gap: {:.2e} < {:.2e}".format(
                              epoch, gap_in, tol_in))
                    break

            if is_sparse:
                cd_epoch_sparse(C, norms_X_col, X.data, X.indices, X.indptr, R,
                                alpha, w, inv_lc, n_samples)
            else:
                cd_epoch(C, norms_X_col, X, R, alpha, w, inv_lc,
                         n_samples)

        else:
            print("!!! Inner solver did not converge at epoch "
                  "{:d}, gap: {:.2e} > {:.2e}".format(epoch, gap_in, tol_in))
    return w

import numpy as np
from numba import njit


def solver(
        X, y, penalty, w, R, norms_X_col, max_iter=50,
        max_epochs=50_000, p0=10, tol=1e-4, use_acc=False, K=5, verbose=0):
    """
    penalty: Penalty object
    p0: first size of working set.
    """
    n_samples, n_features = X.shape[1]
    pen = penalty.is_penalized(n_features)
    unpen = ~pen
    n_unpen = unpen.sum()
    obj_out = []
    lc = norms_X_col ** 2

    for t in range(max_iter):

        kkt = _kkt_violation(w, X, R, penalty, np.arange(n_features))
        kkt_max = np.max(kkt)
        if verbose:
            print(f"KKT max violation: {kkt_max:.2e}")
        if kkt_max <= tol:
            break
        # 1) select features : all unpenalized, + 2 * (nnz and penalized)
        ws_size = max(p0 + n_unpen,
                      min(2 * (w != 0).sum() - n_unpen, n_features))
        kkt[unpen] = np.inf  # always include unpenalized features
        ws = np.argsort(kkt)[-ws_size:]

        if use_acc:
            last_K_w = np.zeros([K + 1, n_features])
            U = np.zeros([K, n_features])

        if verbose:
            print(f'Iteration {t}, {ws_size} feats in subpb.')

        # 2) do iterations on smaller problem
        for epoch in range(max_epochs):
            _cd_epoch(X, w, R, penalty, lc, ws)

            # TODO optimize computation using ws
            if use_acc:
                last_K_w[epoch % (K + 1)] = w

                if epoch % (K + 1) == K:
                    for k in range(K):
                        U[k] = last_K_w[k + 1] - last_K_w[k]
                    C = np.dot(U, U.T)

                    try:
                        z = np.linalg.solve(C, np.ones(K))
                        c = z / z.sum()
                        w_acc = w.copy()
                        w_acc = np.sum(
                            last_K_w[:-1] * c[:, None], axis=0)
                        datafit = (R ** 2).sum() / (2 * n_samples)
                        p_obj = datafit + penalty.value(w)
                        R_acc = y - X @ w_acc
                        datafit_acc = (R_acc ** 2).sum() / (2 * n_samples)
                        p_obj_acc = datafit_acc + penalty.value(w_acc)
                        if p_obj_acc < p_obj:
                            w[:] = w_acc
                            R[:] = R_acc
                    except np.linalg.LinAlgError:
                        if max(verbose - 1, 0):
                            print("----------Linalg error")

            if epoch % 10 == 0:
                # todo maybe we can improve here by restricting to ws
                p_obj = (R ** 2).sum() / (2 * n_samples) + penalty.value(w)

                kkt_ws = _kkt_violation(w, X, R, penalty, ws)
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
def _kkt_violation(w, X, R, penalty, ws):
    n_samples = X.shape[0]
    grad = np.zeros(ws.shape[0])
    for idx, j in enumerate(ws):
        grad[j] = X[:, j] @ R / n_samples
    return penalty.subdiff_distance(grad, w)


@njit
def _cd_epoch(X, w, R, penalty, lc, feats):
    n_samples = R.shape[0]
    for j in feats:
        Xj = X[:, j]
        old_w_j = w[j]
        w[j] = penalty.prox_1d(
            old_w_j + np.dot(Xj, R) / lc[j], n_samples / lc[j], j)
        if w[j] != old_w_j:
            R += (old_w_j - w[j]) * Xj

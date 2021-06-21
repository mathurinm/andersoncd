from numba.experimental import jitclass


@jitclass
class Quadratic():
    def __init__():
        pass

    @staticmethod
    def objective(R):
        return R @ R / (2 * len(R))

    @staticmethod
    def gradient(X, y, w, R):
        return - X.T @ R

    @staticmethod
    def gradient_scalar(X, y, w, R, j):
        return - X[:, j] @ R

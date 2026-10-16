import numpy as np


class SqEuclid:
    def c(self, x, y):
        return np.sum((x - y) ** 2, axis=-1)

    def psi(self, xA, xs, y):
        """(5.1)"""
        return self.c(xA, y) - self.c(xs, y)

    def psi_hat(self, xA, xs, rep_y, rad_y):
        """(5.7)"""
        return self.psi(xA, xs, rep_y) - 2 * np.linalg.norm(xs - xA, axis=-1) * rad_y

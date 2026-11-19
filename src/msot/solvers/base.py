import numpy as np


class LocalSolver:
    def solve_local(self, rows, columns, c, mu, nu, state=None, start=None):
        """OT restricted to N = (rows, columns) with entry costs c.

        Returns (pi, alpha, beta, state): pi a scipy COO matrix, (alpha, beta) local dual optimizers,
        state opaque warm-start data. start: optional coupling on N to begin from when there is no state;
        a hint only, the result does not depend on it.
        """
        raise NotImplementedError


def solve_dense(X, Y, mu, nu, cost, solver):
    """Dense reference: solver.solve_local on all of X x Y."""
    rows, columns = np.divmod(np.arange(len(X) * len(Y)), len(Y))
    return solver.solve_local(rows, columns, cost.c(X[rows], Y[columns]), mu, nu)

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from .base import LocalSolver


class ScipyHighs(LocalSolver):
    """HiGHS dual simplex via scipy.optimize.linprog, solved from scratch each call."""

    def solve_local(self, rows, columns, c, mu, nu, state=None):
        size_x, size_y, size_N = len(mu), len(nu), len(rows)
        constraints = csr_matrix(
            (np.ones(2 * size_N), (np.concatenate([rows, size_x + columns]), np.tile(np.arange(size_N), 2))),
            shape=(size_x + size_y, size_N))
        result = linprog(c, A_eq=constraints, b_eq=np.concatenate([mu, nu]), bounds=(0, None), method="highs-ds")
        if result.status != 0:
            raise RuntimeError(result.message)
        keep = result.x > 0
        pi = coo_matrix((result.x[keep], (rows[keep], columns[keep])), shape=(size_x, size_y))
        duals = result.eqlin.marginals
        return pi, duals[:size_x], duals[size_x:], None


def solve_dense(X, Y, mu, nu, cost, solver=None):
    rows, columns = np.divmod(np.arange(len(X) * len(Y)), len(Y))
    return (solver or ScipyHighs()).solve_local(rows, columns, cost.c(X[rows], Y[columns]), mu, nu)

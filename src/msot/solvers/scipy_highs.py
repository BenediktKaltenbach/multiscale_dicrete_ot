import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from .base import LocalSolver


class ScipyHighs(LocalSolver):
    """HiGHS dual simplex via scipy.optimize.linprog, solved from scratch each call.

    HiGHS tolerances are absolute, so the marginals are scaled to mean mass 1; the duals are unaffected.
    The primal tolerance stays loose: a refined N1 is feasible only up to the rounding of the coarse solution.
    """

    def __init__(self, primal_tolerance=1e-7, dual_tolerance=1e-9):
        self.options = {"primal_feasibility_tolerance": primal_tolerance,
                        "dual_feasibility_tolerance": dual_tolerance}

    def solve_local(self, rows, columns, c, mu, nu, state=None):
        size_x, size_y, size_N = len(mu), len(nu), len(rows)
        scale = size_x / mu.sum()
        constraints = csr_matrix(
            (np.ones(2 * size_N), (np.concatenate([rows, size_x + columns]), np.tile(np.arange(size_N), 2))),
            shape=(size_x + size_y, size_N))
        result = linprog(c, A_eq=constraints, b_eq=scale * np.concatenate([mu, nu]), bounds=(0, None),
                         method="highs-ds", options=self.options)
        if result.status != 0:
            raise RuntimeError(result.message)
        keep = result.x > 0
        pi = coo_matrix((result.x[keep] / scale, (rows[keep], columns[keep])), shape=(size_x, size_y))
        duals = result.eqlin.marginals
        return pi, duals[:size_x], duals[size_x:], None


def solve_dense(X, Y, mu, nu, cost, solver=None):
    rows, columns = np.divmod(np.arange(len(X) * len(Y)), len(Y))
    return (solver or ScipyHighs()).solve_local(rows, columns, cost.c(X[rows], Y[columns]), mu, nu)

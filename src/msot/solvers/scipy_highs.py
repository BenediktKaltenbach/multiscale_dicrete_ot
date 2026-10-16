import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix

from .base import LocalSolver


class ScipyHighs(LocalSolver):
    """HiGHS dual simplex via scipy.optimize.linprog, solved from scratch each call."""

    def solve_local(self, rows, cols, c, mu, nu, state=None):
        m, n, nnz = len(mu), len(nu), len(rows)
        A = csr_matrix((np.ones(2 * nnz), (np.concatenate([rows, m + cols]), np.tile(np.arange(nnz), 2))),
                       shape=(m + n, nnz))
        res = linprog(c, A_eq=A, b_eq=np.concatenate([mu, nu]), bounds=(0, None), method="highs-ds")
        if res.status != 0:
            raise RuntimeError(res.message)
        keep = res.x > 0
        pi = coo_matrix((res.x[keep], (rows[keep], cols[keep])), shape=(m, n))
        y = res.eqlin.marginals
        return pi, y[:m], y[m:], None


def solve_dense(X, Y, mu, nu, cost, solver=None):
    rows, cols = np.divmod(np.arange(len(X) * len(Y)), len(Y))
    return (solver or ScipyHighs()).solve_local(rows, cols, cost.c(X[rows], Y[cols]), mu, nu)

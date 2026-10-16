import numpy as np

from .shielding import shield


def north_west_corner(mu, nu, order_x, order_y):
    """Feasible initial N1 with |X| + |Y| - 1 entries: NW corner rule along the given orderings."""
    cx, cy = np.cumsum(mu[order_x]), np.cumsum(nu[order_y])
    breaks = np.unique(np.concatenate([cx, cy]))[:-1]
    i = np.minimum(np.searchsorted(cx, breaks, side="right"), len(mu) - 1)
    j = np.minimum(np.searchsorted(cy, breaks, side="right"), len(nu) - 1)
    rows = np.concatenate([[order_x[0]], order_x[i]])
    cols = np.concatenate([[order_y[0]], order_y[j]])
    return rows, cols


def solve_sparse(X, Y, mu, nu, cost, S, N1, solver, tol=1e-12, max_iter=100):
    """Alg. 4.1. Stops when N no longer grows or C(pi) no longer decreases."""
    rows, cols = N1
    state, C_old, history = None, np.inf, []
    for _ in range(max_iter):
        c = cost.c(X[rows], Y[cols])
        pi, alpha, beta, state = solver.solve_local(rows, cols, c, mu, nu, state)
        C = float(pi.data @ cost.c(X[pi.row], Y[pi.col]))
        history.append({"C": C, "N": len(rows)})
        new_rows, new_cols = shield(pi, S, X, Y, cost, tol)
        old = rows.astype(np.int64) * len(Y) + cols
        new = new_rows.astype(np.int64) * len(Y) + new_cols
        if np.isin(new, old).all() or C >= C_old - tol * abs(C_old):
            return pi, alpha, beta, (new_rows, new_cols), history
        rows, cols, C_old = new_rows, new_cols, C
    raise RuntimeError("solve_sparse did not converge")

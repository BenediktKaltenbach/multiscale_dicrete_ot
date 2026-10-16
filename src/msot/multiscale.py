from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix

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


def solve_sparse(X, mu, Yh, k, cost, S, N1, solver, tol=1e-12, max_iter=100):
    """Alg. 4.1 at scale k, X the points of X_k. Stops when N no longer grows or C(pi) no longer decreases."""
    Y, nu = Yh.rep_points(k), Yh.mass[k]
    rows, cols = N1
    state, C_old, history = None, np.inf, []
    for _ in range(max_iter):
        t0 = perf_counter()
        c = cost.c(X[rows], Y[cols])
        pi, alpha, beta, state = solver.solve_local(rows, cols, c, mu, nu, state)
        t1 = perf_counter()
        new_rows, new_cols = shield(pi, S, X, Yh, k, cost, tol)
        C = float(pi.data @ cost.c(X[pi.row], Y[pi.col]))
        history.append({"C": C, "N": len(rows), "t_solve": t1 - t0, "t_shield": perf_counter() - t1})
        old = rows.astype(np.int64) * len(Y) + cols
        new = new_rows.astype(np.int64) * len(Y) + new_cols
        if np.isin(new, old).all() or C >= C_old - tol * abs(C_old):
            return pi, alpha, beta, (new_rows, new_cols), history
        rows, cols, C_old = new_rows, new_cols, C
    raise RuntimeError("solve_sparse did not converge")


def refine(pi, HX, HY, k):
    """N1 at scale k: children(x) x children(y) for all (x, y) in spt pi, pi a coupling at scale k + 1."""
    ix, cx = HX.children(k + 1)
    iy, cy = HY.children(k + 1)
    a, b = pi.row, pi.col
    la, lb = np.diff(ix)[a], np.diff(iy)[b]
    count = la * lb
    e = np.repeat(np.arange(len(a)), count)
    i = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    return cx[ix[a][e] + i // lb[e]], cy[iy[b][e] + i % lb[e]]


def solve_multiscale(HX, HY, cost, solver, tol=1e-12, max_iter=100):
    """Alg. 4.6, starting from the trivial problem at level K."""
    K = HX.K
    pi = coo_matrix((HX.mass[K], ([0], [0])), shape=(1, 1))
    info = []
    for k in range(K - 1, -1, -1):
        N1 = refine(pi, HX, HY, k)
        pi, alpha, beta, N, history = solve_sparse(HX.rep_points(k), HX.mass[k], HY, k, cost, HX.nbrs[k], N1,
                                                   solver, tol, max_iter)
        info.append({"k": k, "cells": (HX.n_cells(k), HY.n_cells(k)), "N1": len(N1[0]), "history": history})
    return pi, alpha, beta, info

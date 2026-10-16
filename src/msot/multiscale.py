from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix

from .shielding import shield


def solve_sparse(X, mu, Yh, k, cost, S, N1, solver, tolerance=1e-12, max_iterations=100):
    """Alg. 4.1 at scale k, X the points of X_k. Stops when N no longer grows or C(pi) no longer decreases."""
    Y, nu = Yh.rep_points(k), Yh.mass[k]
    rows, columns = N1
    state, C_old, history = None, np.inf, []
    for _ in range(max_iterations):
        start = perf_counter()
        c = cost.c(X[rows], Y[columns])
        pi, alpha, beta, state = solver.solve_local(rows, columns, c, mu, nu, state)
        solved = perf_counter()
        new_rows, new_columns = shield(pi, S, X, Yh, k, cost, tolerance)
        C = float(pi.data @ cost.c(X[pi.row], Y[pi.col]))
        history.append({"C": C, "N": len(rows), "time_solve": solved - start, "time_shield": perf_counter() - solved})
        old_keys = rows.astype(np.int64) * len(Y) + columns
        new_keys = new_rows.astype(np.int64) * len(Y) + new_columns
        if np.isin(new_keys, old_keys).all() or C >= C_old - tolerance * abs(C_old):
            return pi, alpha, beta, (new_rows, new_columns), history
        rows, columns, C_old = new_rows, new_columns, C
    raise RuntimeError("solve_sparse did not converge")


def refine(pi, HX, HY, k):
    """N1 at scale k: children(x) x children(y) for all (x, y) in spt pi, pi a coupling at scale k + 1."""
    index_pointer_x, children_x = HX.children(k + 1)
    index_pointer_y, children_y = HY.children(k + 1)
    cells_x, cells_y = pi.row, pi.col
    lengths_x, lengths_y = np.diff(index_pointer_x)[cells_x], np.diff(index_pointer_y)[cells_y]
    counts = lengths_x * lengths_y
    entry = np.repeat(np.arange(len(cells_x)), counts)
    position = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    rows = children_x[index_pointer_x[cells_x][entry] + position // lengths_y[entry]]
    columns = children_y[index_pointer_y[cells_y][entry] + position % lengths_y[entry]]
    return rows, columns


def solve_multiscale(HX, HY, cost, solver, tolerance=1e-12, max_iterations=100, candidates=None):
    """Alg. 4.6, starting from the trivial problem at level K.

    candidates(HX, k) gives S on X_k; by default the 1-ring of the level's mesh, HX.neighbours[k].
    """
    K = HX.K
    pi = coo_matrix((HX.mass[K], ([0], [0])), shape=(1, 1))
    info = []
    for k in range(K - 1, -1, -1):
        N1 = refine(pi, HX, HY, k)
        S = candidates(HX, k) if candidates else HX.neighbours[k]
        pi, alpha, beta, N, history = solve_sparse(HX.rep_points(k), HX.mass[k], HY, k, cost, S, N1,
                                                   solver, tolerance, max_iterations)
        info.append({"k": k, "cells": (HX.number_of_cells(k), HY.number_of_cells(k)), "N1": len(N1[0]),
                     "history": history})
    return pi, alpha, beta, info

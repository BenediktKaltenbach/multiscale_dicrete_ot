from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix

from .shielding import shield, unique_entries


def solve_sparse(X, mu, Yh, k, cost, S, N1, solver, tolerance=1e-12, max_iterations=100, start=None):
    """
    Alg. 4.1 at scale k, X the points of X_k. Stops when N no longer grows or C(pi) no longer decreases.
    Alternates solving the local problem and shielding until convergence. Returns the coupling pi, dual potentials alpha, beta, the support N, and a history of costs and times.
    start: optional coupling on N1 handed to the first solve (solve_local).
    """
    Y, nu = Yh.rep_points(k), Yh.mass[k]
    rows, columns = N1
    state, C_old, history = None, np.inf, []
    for _ in range(max_iterations):
        began = perf_counter()
        c = cost.c(X[rows], Y[columns])
        pi, alpha, beta, state = solver.solve_local(rows, columns, c, mu, nu, state, start)
        start = None
        solved = perf_counter()
        new_rows, new_columns = shield(pi, S, X, Yh, k, cost, tolerance)
        C = float(pi.data @ cost.c(X[pi.row], Y[pi.col]))
        history.append({"C": C, "N": len(rows), "time_solve": solved - began, "time_shield": perf_counter() - solved})
        old_keys = rows.astype(np.int64) * len(Y) + columns
        new_keys = new_rows.astype(np.int64) * len(Y) + new_columns
        if np.isin(new_keys, old_keys).all() or C >= C_old - tolerance * abs(C_old):
            return pi, alpha, beta, (new_rows, new_columns), history
        rows, columns, C_old = new_rows, new_columns, C
    raise RuntimeError("solve_sparse did not converge")


def refine(pi, HX, HY, k):
    """N1 at scale k: return children(x) x children(y) for all (x, y) in spt pi, where pi is a coupling at scale k + 1."""
    index_pointer_x, children_x = HX.children(k + 1)
    index_pointer_y, children_y = HY.children(k + 1)
    supp_x, supp_y = pi.row, pi.col #positions of the support of pi in X_k and Y_k
    
    lengths_x, lengths_y = np.diff(index_pointer_x)[supp_x], np.diff(index_pointer_y)[supp_y] #lengths_x[i], lengths_y[i] = |children(xᵢ)|, |children(yᵢ)|
    counts = lengths_x * lengths_y #counts how many children pairs (x', y') there are for each (x, y) in the support of pi
    entry = np.repeat(np.arange(len(supp_x)), counts) #x entries repeated according to how many children pairs they have
    position = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts) #position of each child pair (x', y') in the list of children pairs for its parent (x, y)
    rows = children_x[index_pointer_x[supp_x][entry] + position // lengths_y[entry]] #unflatten the children pairs (x', y') into a single list of rows and columns
    columns = children_y[index_pointer_y[supp_y][entry] + position % lengths_y[entry]]
    return rows, columns


def dilate(N, neighbours, size_y):
    """N plus (x, y') for every (x, y) in N and y' in the 1-ring of y; neighbours is CSR (index_pointer, indices)."""
    rows, columns = N
    index_pointer, indices = neighbours
    counts = np.diff(index_pointer)[columns]
    position = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    ring = indices[np.repeat(index_pointer[columns], counts) + position]
    return unique_entries(np.concatenate([rows, np.repeat(rows, counts)]), np.concatenate([columns, ring]), size_y)


def north_west(first, amount_first, second, amount_second):
    """North-west corner rule: the monotone coupling of two lists with equal total amount.

    Returns (index into first, index into second, amount) for each piece of the coupling. With float amounts the
    totals may differ by rounding; the last pieces then go to the last element.
    """
    ends_first, ends_second = np.cumsum(amount_first), np.cumsum(amount_second)
    ends = np.union1d(ends_first, ends_second)
    amount = np.diff(ends, prepend=0)
    in_first = np.minimum(np.searchsorted(ends_first, ends), len(first) - 1)
    in_second = np.minimum(np.searchsorted(ends_second, ends), len(second) - 1)
    return first[in_first], second[in_second], amount


def north_west_start(pi, HX, HY, k, order="index"):
    """A coupling at scale k inside refine(spt pi), pi the coupling at scale k + 1, by the north-west rule: the
    children of each cell of X share the entries of pi in its row, likewise in Y for the columns, and within each
    entry of pi the pieces of both sides are coupled. Exact for integer masses.

    order: "index" matches children and entries in index order; "transport" sorts the children of each cell and the
    entries of its row (column) along the cell's mean transport direction (mass-weighted rep(y) - rep(x)).
    """
    pi = pi.tocoo()
    entries = np.arange(len(pi.data))
    split = []
    for H, coarse, partner in ((HX, pi.row, HY.rep_points(k + 1)[pi.col]), (HY, pi.col, HX.rep_points(k + 1)[pi.row])):
        if order == "transport":
            children, by_cell = _along_transport(pi, HX, HY, k, H, coarse, partner)
        else:
            children, by_cell = np.argsort(H.parent[k], kind="stable"), np.argsort(coarse, kind="stable")
        child, entry, amount = north_west(children, H.mass[k][children], entries[by_cell], pi.data[by_cell])
        split.append((child[amount > 0], entry[amount > 0], amount[amount > 0]))
    (x, entry_x, amount_x), (y, entry_y, amount_y) = split
    order_x, order_y = np.argsort(entry_x, kind="stable"), np.argsort(entry_y, kind="stable")
    rows, columns, amount = north_west(x[order_x], amount_x[order_x], y[order_y], amount_y[order_y])
    keep = amount > 0
    return coo_matrix((amount[keep], (rows[keep], columns[keep])),
                      shape=(HX.number_of_cells(k), HY.number_of_cells(k)))


def _along_transport(pi, HX, HY, k, H, coarse, partner):
    """Children of the cells of H at scale k + 1 and the entries of pi, both grouped by cell and sorted along the
    cell's mean transport direction (children by position, entries by the rep of the partner cell)."""
    displacement = HY.rep_points(k + 1)[pi.col] - HX.rep_points(k + 1)[pi.row]
    cells = H.number_of_cells(k + 1)
    direction = np.stack([np.bincount(coarse, pi.data * displacement[:, i], minlength=cells)
                          for i in range(displacement.shape[1])], axis=1)
    parent = H.parent[k]
    children = np.lexsort((np.einsum("ij,ij->i", H.rep_points(k), direction[parent]), parent))
    by_cell = np.lexsort((np.einsum("ij,ij->i", partner, direction[coarse]), coarse))
    return children, by_cell


def solve_multiscale(HX, HY, cost, solver, tolerance=1e-12, max_iterations=100, candidates=None, dilation=True,
                     north_west="index"):
    """Alg. 4.6, starting from the trivial problem at level K.

    candidates(HX, k) gives S on X_k; by default the 1-ring of the level's mesh, HX.neighbours[k].
    dilation: N1 = refine(spt pi) dilated by the 1-ring of the fine Y mesh (dilate), so the first coupling of a
    level can already move mass across the borders of the refined blocks; False gives Schmitzer's N1.
    north_west: the first solve of each level starts from north_west_start of the coarser coupling (lies in N1),
    with this order ("index" or "transport"); None starts cold.
    """
    K = HX.K
    pi = coo_matrix((HX.mass[K], ([0], [0])), shape=(1, 1)) #initial trivial coupling
    info = []
    for k in range(K - 1, -1, -1):
        N1 = refine(pi, HX, HY, k) #N1 is tuple (rows, columns) of indices of the support of pi
        if dilation:
            N1 = dilate(N1, HY.neighbours[k], HY.number_of_cells(k))
        S = candidates(HX, k) if candidates else HX.neighbours[k] #default is 1-ring, but candidates can be provided, e. g. bracketed candidates
        start = north_west_start(pi, HX, HY, k, north_west) if north_west else None
        pi, alpha, beta, N, history = solve_sparse(HX.rep_points(k), HX.mass[k], HY, k, cost, S, N1,
                                                   solver, tolerance, max_iterations, start)
        info.append({"k": k, "cells": (HX.number_of_cells(k), HY.number_of_cells(k)), "N1": len(N1[0]),
                     "history": history})
    return pi, alpha, beta, info

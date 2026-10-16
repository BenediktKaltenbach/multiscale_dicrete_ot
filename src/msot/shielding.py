import numpy as np


def extract_map(pi):
    """step-i: t(x) = argmax_y pi(x, y)."""
    return np.asarray(pi.tocsr().argmax(axis=1)).ravel()


def missed_brute(xA, xs, ys, Y, cost, tolerance):
    """step-iii by checking (5.2) for every yB in Y."""
    psi_B = cost.psi(xA, xs[:, None], Y[None])
    psi_s = cost.psi(xA, xs, ys)
    return np.flatnonzero(~(psi_B > psi_s[:, None] + tolerance).any(axis=0))


def search_tree(xA, xs, ys, Yh, k, cost, tolerance):
    """Alg. 5.2, level-synchronous from Y_K down to Y_k. Returns the missed cells of Y_k."""
    psi_s = cost.psi(xA, xs, ys)[:, None]
    cells = np.zeros(1, dtype=np.int64)
    for level in range(Yh.K, k - 1, -1):
        rad = Yh.rad[level][cells] if level > k else np.zeros(len(cells))
        psi_hat = cost.psi_hat(xA, xs[:, None], Yh.rep_points(level)[cells][None], rad[None])
        cells = cells[~(psi_hat > psi_s + tolerance).any(axis=0)]
        if level > k:
            cells = _expand(*Yh.children(level), cells)
    return cells


def _ranges(starts, lengths):
    """Concatenation of the index ranges [start, start + length)."""
    return np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(lengths.sum())


def _expand(index_pointer, indices, cells):
    return indices[_ranges(index_pointer[cells], np.diff(index_pointer)[cells])]


def search_all(X, S, ys, Yh, k, cost, tolerance, sources=None):
    """Alg. 5.2 for all x_A in sources (default all of X) at once, level by level. S is CSR (index_pointer, indices)
    of candidates x_s on X, ys their partners t(x_s) in the same order. Returns the pairs (x_A, missed cell of Y_k)."""
    index_pointer, indices = S
    counts = np.diff(index_pointer)
    owner = np.repeat(np.arange(len(X)), counts)
    psi_s = cost.psi(X[owner], X[indices], ys)
    pair_x = np.arange(len(X)) if sources is None else sources
    pair_cell = np.zeros(len(pair_x), dtype=np.int64)
    for level in range(Yh.K, k - 1, -1):
        pair_of = np.repeat(np.arange(len(pair_x)), counts[pair_x])
        entry = _ranges(index_pointer[pair_x], counts[pair_x])
        cell = pair_cell[pair_of]
        rad = Yh.rad[level][cell] if level > k else 0.0
        psi_hat = cost.psi_hat(X[owner[entry]], X[indices[entry]], Yh.rep_points(level)[cell], rad)
        shielded = np.bincount(pair_of[psi_hat > psi_s[entry] + tolerance], minlength=len(pair_x)) > 0
        pair_x, pair_cell = pair_x[~shielded], pair_cell[~shielded]
        if level > k:
            children_pointer, children = Yh.children(level)
            pair_x = np.repeat(pair_x, np.diff(children_pointer)[pair_cell])
            pair_cell = _expand(children_pointer, children, pair_cell)
    return pair_x, pair_cell


def shield(pi, S, X, Yh, k, cost, tolerance=1e-12, chunk=4096):
    """Alg. 5.1 at scale k, vectorised over all x_A. pi is COO; all stored entries count as spt pi.

    X are the points of X_k, S a CSR adjacency (index_pointer, indices) on X_k. If the cost provides
    needed_candidates, redundant candidates are skipped in the search (step-iii, Y_hat unchanged); step-ii
    still adds (xA, t(xs)) for all candidates, since these entries speed up the convergence of Alg. 4.1.
    The search runs on chunks of x_A to bound memory (about 75 kB per x_A at peak).
    """
    t = extract_map(pi)
    Y = Yh.rep_points(k)
    index_pointer, indices = S
    owner = np.repeat(np.arange(len(X)), np.diff(index_pointer))
    search = S
    prune = getattr(cost, "needed_candidates_all", None)
    if prune is not None:
        needed = prune(X, S, Y[t[indices]], tolerance)
        if not needed.all():
            search = np.concatenate([[0], np.cumsum(np.bincount(owner[needed], minlength=len(X)))]), indices[needed]
    rows, columns = [pi.row, owner], [pi.col, t[indices]]
    for start in range(0, len(X), chunk):
        sources = np.arange(start, min(start + chunk, len(X)))
        missed_x, missed_y = search_all(X, search, Y[t[search[1]]], Yh, k, cost, tolerance, sources)
        rows.append(missed_x)
        columns.append(missed_y)
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    return unique_entries(rows, columns, len(Y))


def unique_entries(rows, columns, n):
    keys = np.unique(rows.astype(np.int64) * n + columns)
    return keys // n, keys % n

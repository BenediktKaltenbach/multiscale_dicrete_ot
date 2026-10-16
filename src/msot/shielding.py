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


def _expand(index_pointer, indices, cells):
    starts, lengths = index_pointer[cells], np.diff(index_pointer)[cells]
    offsets = np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(lengths.sum())
    return indices[offsets]


def shield(pi, S, X, Yh, k, cost, tolerance=1e-12):
    """Alg. 5.1 at scale k. pi is COO; all stored entries count as spt pi.

    X are the points of X_k, S a CSR adjacency (index_pointer, indices) on X_k. If the cost provides
    needed_candidates, redundant candidates are skipped in the search (step-iii, Y_hat unchanged); step-ii
    still adds (xA, t(xs)) for all candidates, since these entries speed up the convergence of Alg. 4.1.
    """
    t = extract_map(pi)
    Y = Yh.rep_points(k)
    index_pointer, indices = S
    prune = getattr(cost, "needed_candidates", None)
    rows, columns = [pi.row], [pi.col]
    for a in range(len(X)):
        candidates = indices[index_pointer[a]:index_pointer[a + 1]]
        needed = candidates if prune is None else candidates[prune(X[a], X[candidates], Y[t[candidates]], tolerance)]
        Y_hat = search_tree(X[a], X[needed], Y[t[needed]], Yh, k, cost, tolerance)
        new = np.concatenate([t[candidates], Y_hat])
        rows.append(np.full(len(new), a))
        columns.append(new)
    return unique_entries(np.concatenate(rows), np.concatenate(columns), len(Y))


def unique_entries(rows, columns, n):
    keys = np.unique(rows.astype(np.int64) * n + columns)
    return keys // n, keys % n

import numpy as np


def extract_map(pi):
    """step-i: t(x) = argmax_y pi(x, y)."""
    return np.asarray(pi.tocsr().argmax(axis=1)).ravel()


def missed_brute(xA, xs, ys, Y, cost, tol):
    """step-iii by checking (5.2) for every yB in Y."""
    psi_B = cost.psi(xA, xs[:, None], Y[None])
    psi_s = cost.psi(xA, xs, ys)
    return np.flatnonzero(~(psi_B > psi_s[:, None] + tol).any(axis=0))


def search_tree(xA, xs, ys, Yh, k, cost, tol):
    """Alg. 5.2, level-synchronous from Y_K down to Y_k. Returns the missed cells of Y_k."""
    psi_s = cost.psi(xA, xs, ys)[:, None]
    cells = np.zeros(1, dtype=np.int64)
    for l in range(Yh.K, k - 1, -1):
        rad = Yh.rad[l][cells] if l > k else np.zeros(len(cells))
        psi_hat = cost.psi_hat(xA, xs[:, None], Yh.rep_points(l)[cells][None], rad[None])
        cells = cells[~(psi_hat > psi_s + tol).any(axis=0)]
        if l > k:
            cells = _expand(*Yh.children(l), cells)
    return cells


def _expand(indptr, indices, cells):
    starts, lens = indptr[cells], np.diff(indptr)[cells]
    offsets = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())
    return indices[offsets]


def shield(pi, S, X, Yh, k, cost, tol=1e-12):
    """Alg. 5.1 at scale k. X are the points of X_k, S a CSR adjacency (indptr, indices) on X_k."""
    t = extract_map(pi)
    Y = Yh.rep_points(k)
    indptr, indices = S
    spt_rows, spt_cols = pi.tocsr().nonzero()
    rows, cols = [spt_rows], [spt_cols]
    for a in range(len(X)):
        s = indices[indptr[a]:indptr[a + 1]]
        Y_hat = search_tree(X[a], X[s], Y[t[s]], Yh, k, cost, tol)
        new = np.concatenate([t[s], Y_hat])
        rows.append(np.full(len(new), a))
        cols.append(new)
    return unique_entries(np.concatenate(rows), np.concatenate(cols), len(Y))


def unique_entries(rows, cols, n):
    keys = np.unique(rows.astype(np.int64) * n + cols)
    return keys // n, keys % n

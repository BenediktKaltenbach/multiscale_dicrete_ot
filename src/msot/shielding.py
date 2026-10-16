import numpy as np


def extract_map(pi):
    """step-i: t(x) = argmax_y pi(x, y)."""
    return np.asarray(pi.tocsr().argmax(axis=1)).ravel()


def missed_brute(xA, xs, ys, Y, cost, tol):
    """step-iii by checking (5.2) for every yB in Y."""
    psi_B = cost.psi(xA, xs[:, None], Y[None])
    psi_s = cost.psi(xA, xs, ys)
    return np.flatnonzero(~(psi_B > psi_s[:, None] + tol).any(axis=0))


def shield(pi, S, X, Y, cost, tol=1e-12):
    """Alg. 5.1 with brute-force Y_hat. S is a CSR adjacency (indptr, indices)."""
    t = extract_map(pi)
    indptr, indices = S
    spt_rows, spt_cols = pi.tocsr().nonzero()
    rows, cols = [spt_rows], [spt_cols]
    for a in range(len(X)):
        s = indices[indptr[a]:indptr[a + 1]]
        Y_hat = missed_brute(X[a], X[s], Y[t[s]], Y, cost, tol)
        new = np.concatenate([t[s], Y_hat])
        rows.append(np.full(len(new), a))
        cols.append(new)
    return unique_entries(np.concatenate(rows), np.concatenate(cols), len(Y))


def unique_entries(rows, cols, n):
    keys = np.unique(rows.astype(np.int64) * n + cols)
    return keys // n, keys % n

import numpy as np


def dual_violation(alpha, beta, X, Y, cost):
    """max over X x Y of alpha(x) + beta(y) - c(x, y); <= 0 means globally dual feasible."""
    return max((alpha[a] + beta - cost.c(X[a], Y)).max() for a in range(len(X)))


def is_shielding(pi, rows, cols, X, Y, cost, tol=1e-12):
    """Brute-force check of Def. 3.5 for the neighbourhood N = (rows, cols)."""
    pi = pi.tocsr()
    xs_all, ys_all = pi.nonzero()
    for a in range(len(X)):
        in_row = np.zeros(len(Y), bool)
        in_row[cols[rows == a]] = True
        k = in_row[ys_all]
        xs, ys = X[xs_all[k]], Y[ys_all[k]]
        psi_B = cost.psi(X[a], xs[:, None], Y[None])
        psi_s = cost.psi(X[a], xs, ys)
        shielded = (psi_B > psi_s[:, None] + tol).any(axis=0)
        if not np.all(shielded | in_row):
            return False
    return True

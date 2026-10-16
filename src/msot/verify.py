import numpy as np


def dual_violation(alpha, beta, X, Y, cost):
    """max over X x Y of alpha(x) + beta(y) - c(x, y); <= 0 means globally dual feasible."""
    return max((alpha[a] + beta - cost.c(X[a], Y)).max() for a in range(len(X)))


def is_shielding(pi, rows, columns, X, Y, cost, tolerance=1e-12):
    """Brute-force check of Def. 3.5 for the neighbourhood N = (rows, columns)."""
    support_x, support_y = pi.tocsr().nonzero()
    for a in range(len(X)):
        in_row = np.zeros(len(Y), bool)
        in_row[columns[rows == a]] = True
        usable = in_row[support_y]
        xs, ys = X[support_x[usable]], Y[support_y[usable]]
        psi_B = cost.psi(X[a], xs[:, None], Y[None])
        psi_s = cost.psi(X[a], xs, ys)
        shielded = (psi_B > psi_s[:, None] + tolerance).any(axis=0)
        if not np.all(shielded | in_row):
            return False
    return True

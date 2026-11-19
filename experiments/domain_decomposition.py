"""Prototype for docs/domain_decomposition.md, step 1: the start coupling at level 0 and the certificate on N.

north_west_start refines the coarse optimum by north-west corner rules (integral, inside refine(spt pi_1)).
certificate builds potentials with equality on spt pi, fixes the constants between the components of the support
graph by Bellman-Ford and checks alpha + beta <= c + tolerance on all of N.
The check run solves levels K - 1 to 1 as usual and tests both on N1 of level 0.

python experiments/domain_decomposition.py [kind n seed, default uniform 4096 0]
"""
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.multiscale import dilate, refine, solve_sparse
from msot.problems import make_problem
from msot.solvers.cplex import Cplex

TOLERANCE = 1e-9


def north_west(first, amount_first, second, amount_second):
    """North-west corner rule within groups: both lists sorted by group, equal totals per group.

    Returns (index into first, index into second, amount) of the monotone coupling of the two lists.
    """
    ends_first, ends_second = np.cumsum(amount_first), np.cumsum(amount_second)
    ends = np.union1d(ends_first, ends_second)
    amount = np.diff(ends, prepend=0)
    return (first[np.searchsorted(ends_first, ends)], second[np.searchsorted(ends_second, ends)], amount)


def _parent(H, k):
    """Parent at level k + 1 of every cell of level k."""
    return H.parent[k]


def north_west_start(pi, HX, HY, k):
    """Integral coupling at level k inside refine(spt pi), pi the coupling at level k + 1 (design note, steps 1-3)."""
    pi = pi.tocoo()
    entries = np.arange(len(pi.data))
    split = []
    for H, coarse in ((HX, pi.row), (HY, pi.col)):
        children = np.argsort(_parent(H, k), kind="stable")
        by_cell = np.argsort(coarse, kind="stable")
        child, entry, amount = north_west(children, H.mass[k][children], entries[by_cell], pi.data[by_cell])
        split.append((child[amount > 0], entry[amount > 0], amount[amount > 0]))
    (x, entry_x, amount_x), (y, entry_y, amount_y) = split
    order_x, order_y = np.argsort(entry_x, kind="stable"), np.argsort(entry_y, kind="stable")
    rows, columns, amount = north_west(x[order_x], amount_x[order_x], y[order_y], amount_y[order_y])
    keep = amount > 0
    shape = (HX.number_of_cells(k), HY.number_of_cells(k))
    return coo_matrix((amount[keep], (rows[keep], columns[keep])), shape=shape)


def certificate(pi, N, c, tolerance=TOLERANCE):
    """Is pi optimal on N? c are the costs of the entries of N. Returns alpha, beta and a report."""
    rows, columns = N
    size_x, size_y = pi.shape
    positive = pi.data > 0
    support_x, support_y = pi.row[positive], pi.col[positive]
    graph = coo_matrix((np.ones(len(support_x)), (support_x, size_x + support_y)), shape=(size_x + size_y,) * 2)
    count, component = connected_components(graph, directed=False)
    roots = np.unique(component, return_index=True)[1]
    root = size_x + size_y  # virtual root joined to one node of every component
    graph = coo_matrix((np.ones(len(support_x) + count), (np.r_[support_x, np.full(count, root)],
                                                          np.r_[size_x + support_y, roots])), shape=(root + 1,) * 2)
    order, predecessor = breadth_first_order(graph.tocsr(), root, directed=False)
    support_cost = dict(zip(zip(support_x.tolist(), support_y.tolist()), _support_costs(pi, positive, N, c)))
    potential = np.zeros(root + 1)
    for node in order[1:].tolist():
        previous = predecessor[node]
        if previous != root:
            x, y = (node, previous - size_x) if node < size_x else (previous, node - size_x)
            potential[node] = support_cost[x, y] - potential[previous]
    alpha, beta = potential[:size_x], potential[size_x:root]

    reduced = c - alpha[rows] - beta[columns]
    source, target = component[size_x + columns], component[rows]  # s_P - s_Q <= reduced for x in P, y in Q
    shift, converged = np.zeros(count), False
    for _ in range(count):
        candidate = shift.copy()
        np.minimum.at(candidate, target, shift[source] + reduced + tolerance / 2)
        if np.array_equal(candidate, shift):
            converged = True
            break
        shift = candidate
    alpha, beta = alpha + shift[component[:size_x]], beta - shift[component[size_x:]]
    violation = alpha[rows] + beta[columns] - c
    support_error = np.abs(alpha[support_x] + beta[support_y] - _support_costs(pi, positive, N, c)).max()
    report = {"components": count, "negative cycle": not converged, "violated entries": int(np.sum(violation > tolerance)),
              "largest violation": float(violation.max()), "support error": float(support_error)}
    report["optimal"] = converged and report["violated entries"] == 0
    return alpha, beta, report


def _support_costs(pi, positive, N, c):
    """Costs of the positive entries of pi, looked up in N."""
    size_y = pi.shape[1]
    keys = N[0].astype(np.int64) * size_y + N[1]
    order = np.argsort(keys)
    wanted = pi.row[positive].astype(np.int64) * size_y + pi.col[positive]
    position = order[np.searchsorted(keys[order], wanted)]
    if not np.array_equal(keys[position], wanted):
        raise ValueError("spt pi is not contained in N")
    return c[position]


def coarse_levels(HX, HY, cost, solver):
    """Levels K - 1 to 1 as in solve_multiscale; returns the coupling at level 1."""
    pi = coo_matrix((HX.mass[HX.K], ([0], [0])), shape=(1, 1))
    for k in range(HX.K - 1, 0, -1):
        N1 = dilate(refine(pi, HX, HY, k), HY.neighbours[k], HY.number_of_cells(k))
        pi = solve_sparse(HX.rep_points(k), HX.mass[k], HY, k, cost, HX.neighbours[k], N1, solver)[0]
    return pi


def main():
    kind, n, seed = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else ("uniform", 4096, 0)
    X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed))
    HX, HY = build_pair(X, mu, Y, nu)
    cost, solver = SquaredEuclidean(), Cplex(warm_start=False)
    coarse = coarse_levels(HX, HY, cost, solver)
    N1 = dilate(refine(coarse, HX, HY, 0), HY.neighbours[0], HY.number_of_cells(0))
    c = cost.c(HX.rep_points(0)[N1[0]], HY.rep_points(0)[N1[1]])

    start = north_west_start(coarse, HX, HY, 0)
    inside = np.isin(start.row.astype(np.int64) * len(nu) + start.col, N1[0].astype(np.int64) * len(nu) + N1[1])
    print(f"{kind} n = {n} seed {seed}: |N1| = {len(c)}, north-west start with {start.nnz} entries; "
          f"marginals exact: {np.array_equal(np.bincount(start.row, start.data, len(mu)), HX.mass[0])} and "
          f"{np.array_equal(np.bincount(start.col, start.data, len(nu)), HY.mass[0])}, inside N1: {inside.all()}, "
          f"integral: {np.array_equal(start.data, np.round(start.data))}")
    optimum = solver.solve_local(*N1, c, HX.mass[0], HY.mass[0])[0]
    for name, pi in (("north-west start", start), ("CPLEX optimum", optimum)):
        C = float(pi.data @ cost.c(HX.rep_points(0)[pi.row], HY.rep_points(0)[pi.col]))
        print(f"  {name}: C = {C:.12e},", certificate(pi, N1, c)[2])


if __name__ == "__main__":
    main()

"""Prototype for docs/domain_decomposition.md, Stage A: domain decomposition for one solve on N1 of level 0.

north_west_start refines the coarse optimum by north-west corner rules (integral, inside refine(spt pi_1)).
certificate builds potentials with equality on spt pi, fixes the constants between the components of the support
graph by Bellman-Ford and checks alpha + beta <= c + tolerance on all of N.
partitions groups the cells of level l of the X hierarchy twice: A by their parents at level l + 1, B by greedy
matching along the level-l mesh edges that cross the borders of A. A sweep solves every composite cell of A, then of
B (cell problem: mu on X_J against its current outflow pi(X_J, .), on the entries of N in these rows and columns).
Per problem: global CPLEX on N1 as reference, then sweeps from the north-west start for l = 1, 2, 3 until the
certificate passes, the sweeps stall or MAX_SWEEPS. Writes results/domain_decomposition/runs.json;
plot_domain_decomposition.py makes the figure dd_sweeps.

python experiments/domain_decomposition.py
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

from msot.cost import SquaredEuclidean
from msot.hierarchy import _attach, _match, _passes_cap, _sorted_edges, build_pair
from msot.multiscale import dilate, refine, solve_sparse
from msot.problems import make_problem
from msot.shielding import _ranges
from msot.solvers.cplex import Cplex
from candidate_sets import environment

PROBLEMS = (("uniform", 16384, 0), ("clustered", 16384, 0), ("grid", 16384, 2))
BASIC_LEVELS = (1, 2, 3)
MAX_SWEEPS = 50
STALL = 1e-12
TOLERANCE = 1e-9
RESULTS = Path(__file__).parent / "results" / "domain_decomposition"


def north_west(first, amount_first, second, amount_second):
    """North-west corner rule within groups: both lists sorted by group, equal totals per group.

    Returns (index into first, index into second, amount) of the monotone coupling of the two lists.
    """
    ends_first, ends_second = np.cumsum(amount_first), np.cumsum(amount_second)
    ends = np.union1d(ends_first, ends_second)
    amount = np.diff(ends, prepend=0)
    return (first[np.searchsorted(ends_first, ends)], second[np.searchsorted(ends_second, ends)], amount)


def north_west_start(pi, HX, HY, k):
    """Integral coupling at level k inside refine(spt pi), pi the coupling at level k + 1 (design note, steps 1-3)."""
    pi = pi.tocoo()
    entries = np.arange(len(pi.data))
    split = []
    for H, coarse in ((HX, pi.row), (HY, pi.col)):
        children = np.argsort(H.parent[k], kind="stable")
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
    support_cost = _support_costs(pi, positive, N, c)
    graph = coo_matrix((np.ones(len(support_x)), (support_x, size_x + support_y)), shape=(size_x + size_y,) * 2)
    count, component = connected_components(graph, directed=False)
    roots = np.unique(component, return_index=True)[1]
    root = size_x + size_y  # virtual root joined to one node of every component
    graph = coo_matrix((np.ones(len(support_x) + count), (np.r_[support_x, np.full(count, root)],
                                                          np.r_[size_x + support_y, roots])), shape=(root + 1,) * 2)
    order, predecessor = breadth_first_order(graph.tocsr(), root, directed=False)
    edge_cost = dict(zip(zip(support_x.tolist(), support_y.tolist()), support_cost.tolist()))
    potential = np.zeros(root + 1)
    for node in order[1:].tolist():
        previous = predecessor[node]
        if previous != root:
            x, y = (node, previous - size_x) if node < size_x else (previous, node - size_x)
            potential[node] = edge_cost[x, y] - potential[previous]
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
    support_error = np.abs(alpha[support_x] + beta[support_y] - support_cost).max()
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


def partitions(HX, l, length_cap=3.0):
    """Composite cell of every point of X for partitions A and B; the basic cells are the cells of level l."""
    parent = HX.parent[l]
    edge_start, edge_end, length = _sorted_edges(HX, l)
    crossing = _passes_cap(edge_start, edge_end, length, HX.number_of_cells(l), length_cap)
    crossing &= parent[edge_start] != parent[edge_end]
    edge_start, edge_end = edge_start[crossing], edge_end[crossing]
    labels = np.arange(HX.number_of_cells(l))
    shifted = _attach(_match(_match(labels, edge_start, edge_end), edge_start, edge_end), edge_start, edge_end)
    return HX.cell_of[l + 1], shifted[HX.cell_of[l]]


def solve_partition(pi, labels, N, c, mu, solver):
    """Solves the cell problem of every composite cell (independent of each other); returns the new pi and times."""
    size_x, size_y = pi.shape
    rows, columns = N
    order = np.argsort(rows, kind="stable")
    index_pointer = np.r_[0, np.cumsum(np.bincount(rows, minlength=size_x))]
    flow = pi.tocsr()
    cells = np.argsort(labels, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(labels))]
    local_x, local_y = np.full(size_x, -1), np.full(size_y, -1)
    new_rows, new_columns, new_values, times = [], [], [], []
    for J in range(len(bounds) - 1):
        start = perf_counter()
        points = cells[bounds[J]:bounds[J + 1]]
        outflow = flow[points].tocoo()
        targets, target_of = np.unique(outflow.col, return_inverse=True)
        nu_J = np.bincount(target_of, outflow.data, len(targets))
        entries = order[_ranges(index_pointer[points], np.diff(index_pointer)[points])]
        local_x[points], local_y[targets] = np.arange(len(points)), np.arange(len(targets))
        entries = entries[local_y[columns[entries]] >= 0]
        cell_rows, cell_columns = local_x[rows[entries]], local_y[columns[entries]]
        local_x[points], local_y[targets] = -1, -1
        built = perf_counter()
        solution = solver.solve_local(cell_rows, cell_columns, c[entries], mu[points], nu_J)[0]
        solved = perf_counter()
        positive = solution.data > 0
        new_rows.append(points[solution.row[positive]])
        new_columns.append(targets[solution.col[positive]])
        new_values.append(solution.data[positive])
        times.append((built - start, solved - built, perf_counter() - solved))
    pi = coo_matrix((np.concatenate(new_values), (np.concatenate(new_rows), np.concatenate(new_columns))),
                    shape=(size_x, size_y))
    return pi, np.array(times)


def sweeps(start, partition_labels, N, c, mu, C_star, cost_of, solver):
    """Sweeps over A and B from start; one record per sweep."""
    pi, C_old, records = start, cost_of(start), []
    for sweep in range(1, MAX_SWEEPS + 1):
        record = {"sweep": sweep}
        for name, labels in zip("AB", partition_labels):
            pi, times = solve_partition(pi, labels, N, c, mu, solver)
            record[f"time {name}"] = {"build": float(times[:, 0].sum()), "cplex": float(times[:, 1].sum()),
                                      "rest": float(times[:, 2].sum()), "largest cell": float(times.sum(axis=1).max())}
        C = cost_of(pi)
        checked = perf_counter()
        report = certificate(pi, N, c)[2]
        record.update({"C": C, "gap": (C - C_star) / C_star, "time certificate": perf_counter() - checked, **report})
        records.append(record)
        print(f"    sweep {sweep:2d}: gap {record['gap']:.2e}, violated {report['violated entries']}, "
              f"components {report['components']}, cplex {record['time A']['cplex'] + record['time B']['cplex']:.2f} s, "
              f"build {record['time A']['build'] + record['time B']['build']:.2f} s", flush=True)
        if report["optimal"]:
            return pi, records, "certificate"
        if C >= C_old - STALL * abs(C_old):
            return pi, records, "stalled"
        C_old = C
    return pi, records, "max sweeps"


def coarse_levels(HX, HY, cost, solver):
    """Levels K - 1 to 1 as in solve_multiscale; returns the coupling at level 1."""
    pi = coo_matrix((HX.mass[HX.K], ([0], [0])), shape=(1, 1))
    for k in range(HX.K - 1, 0, -1):
        N1 = dilate(refine(pi, HX, HY, k), HY.neighbours[k], HY.number_of_cells(k))
        pi = solve_sparse(HX.rep_points(k), HX.mass[k], HY, k, cost, HX.neighbours[k], N1, solver)[0]
    return pi


def main(problems=PROBLEMS, path=RESULTS / "runs.json"):
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"environment": environment(), "runs": {}}
    for kind, n, seed in problems:
        name = f"{kind}_{n}_{seed}"
        if name in stored["runs"]:
            continue
        print(name, flush=True)
        X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed))
        HX, HY = build_pair(X, mu, Y, nu)
        cost, solver = SquaredEuclidean(), Cplex(warm_start=False)
        coarse = coarse_levels(HX, HY, cost, Cplex())
        N1 = dilate(refine(coarse, HX, HY, 0), HY.neighbours[0], HY.number_of_cells(0))
        X0, Y0, mu0, nu0 = HX.rep_points(0), HY.rep_points(0), HX.mass[0], HY.mass[0]
        c = cost.c(X0[N1[0]], Y0[N1[1]])

        def cost_of(pi):
            return float(pi.data @ cost.c(X0[pi.row], Y0[pi.col]))

        start = perf_counter()
        optimum = solver.solve_local(*N1, c, mu0, nu0)[0]
        reference = {"time": perf_counter() - start, "C": cost_of(optimum)}
        north_west_coupling = north_west_start(coarse, HX, HY, 0)
        assert np.array_equal(np.bincount(north_west_coupling.row, north_west_coupling.data, n), mu0)
        assert np.array_equal(np.bincount(north_west_coupling.col, north_west_coupling.data, len(nu0)), nu0)
        run = {"kind": kind, "n": n, "seed": seed, "N1": len(c), "reference CPLEX on N1": reference,
               "start gap": (cost_of(north_west_coupling) - reference["C"]) / reference["C"], "levels": {}}
        print(f"  |N1| {len(c)}, CPLEX {reference['time']:.2f} s, start gap {run['start gap']:.2e}", flush=True)
        for l in BASIC_LEVELS:
            labels = partitions(HX, l)
            sizes = [np.bincount(labels[0]), np.bincount(labels[1])]
            print(f"  l = {l}: {len(sizes[0])} cells in A (mean {sizes[0].mean():.1f}, max {sizes[0].max()} points), "
                  f"{len(sizes[1])} in B (mean {sizes[1].mean():.1f}, max {sizes[1].max()})", flush=True)
            _, records, stop = sweeps(north_west_coupling, labels, N1, c, mu0, reference["C"], cost_of, solver)
            print(f"    stop: {stop}", flush=True)
            run["levels"][l] = {"cells A": len(sizes[0]), "cells B": len(sizes[1]),
                                "points per cell A (mean, max)": [float(sizes[0].mean()), int(sizes[0].max())],
                                "points per cell B (mean, max)": [float(sizes[1].mean()), int(sizes[1].max())],
                                "stop": stop, "sweeps": records}
        stored["runs"][name] = run
        path.write_text(json.dumps(stored, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Hybrid test for docs/domain_decomposition.md: global CPLEX on N1 of level 0, started from the coupling after
0, 1 or 3 sweeps (0: the north-west start), against CPLEX from scratch.

The starting basis is a spanning tree of the bipartite graph of N1 that prefers the support of the coupling
(minimum spanning tree with weight 1 on spt pi and 2 elsewhere), plus one basic row; all other columns at 0.
If spt pi is a forest, the basis reproduces pi exactly; otherwise CPLEX starts from a nearby basis.
Writes results/domain_decomposition/warm.json.

python experiments/domain_decomposition_warm.py [n, default 16384] [sweep counts, default 0,1,3]
"""
import json
import sys
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.multiscale import dilate, refine
from msot.problems import make_problem
from msot.solvers.cplex import Cplex
from domain_decomposition import RESULTS, coarse_levels, north_west_start, partitions, solve_partition

KINDS = (("uniform", 0), ("clustered", 0), ("grid", 2))
BASIC_LEVEL = 3
SWEEPS = (0, 1, 3)


def start_basis(pi, N, size_x, size_y):
    """Column and row status of a basis whose tree prefers spt pi; also whether spt pi is a forest."""
    rows, columns = N
    keys = rows.astype(np.int64) * size_y + columns
    positive = pi.data > 0
    in_support = np.isin(keys, pi.row[positive].astype(np.int64) * size_y + pi.col[positive])
    nodes = size_x + size_y
    graph = coo_matrix((np.where(in_support, 1.0, 2.0), (rows, size_x + columns)), shape=(nodes, nodes)).tocsr()
    tree = minimum_spanning_tree(graph).tocoo()
    tree_keys = tree.row.astype(np.int64) * size_y + (tree.col - size_x)
    support_graph = coo_matrix((np.ones(positive.sum()), (pi.row[positive], size_x + pi.col[positive])),
                               shape=(nodes, nodes))
    forest = positive.sum() == nodes - connected_components(support_graph, directed=False)[0]
    return np.isin(keys, tree_keys), len(tree_keys), bool(forest)


def solve(model):
    start = perf_counter()
    model.solve()
    return perf_counter() - start, model.solution.get_objective_value()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    sweep_counts = tuple(int(count) for count in sys.argv[2].split(",")) if len(sys.argv) > 2 else SWEEPS
    path = RESULTS / "warm.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for kind, seed in KINDS:
        name = f"{kind}_{n}_{seed}"
        if name in stored:
            continue
        print(name, flush=True)
        X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed))
        HX, HY = build_pair(X, mu, Y, nu)
        cost, solver = SquaredEuclidean(), Cplex(warm_start=False)
        coarse = coarse_levels(HX, HY, cost, Cplex())
        N1 = dilate(refine(coarse, HX, HY, 0), HY.neighbours[0], HY.number_of_cells(0))
        mu0, nu0 = HX.mass[0], HY.mass[0]
        c = cost.c(HX.rep_points(0)[N1[0]], HY.rep_points(0)[N1[1]])
        size_x, size_y = len(mu0), len(nu0)

        model = solver.build(*N1, c, mu0, nu0, 1.0, True)
        time, C_star = solve(model)
        run = {"kind": kind, "n": n, "seed": seed, "N1": len(c), "from scratch": {"time": time}}
        print(f"  from scratch: {time:.2f} s", flush=True)

        labels = partitions(HX, BASIC_LEVEL)
        pi, done, sweep_time = north_west_start(coarse, HX, HY, 0), 0, 0.0
        for sweeps in sweep_counts:
            while done < sweeps:
                start = perf_counter()
                for partition in labels:
                    pi = solve_partition(pi, partition, N1, c, mu0, solver)[0]
                sweep_time += perf_counter() - start
                done += 1
            start = perf_counter()
            basic, tree_size, forest = start_basis(pi, N1, size_x, size_y)
            model = solver.build(*N1, c, mu0, nu0, 1.0, True)
            status = model.start.status
            row_status = [status.basic] + [status.at_lower_bound] * (size_x + size_y - 1)
            model.start.set_start(np.where(basic, status.basic, status.at_lower_bound).tolist(), row_status, [], [], [], [])
            basis_time = perf_counter() - start
            time, C = solve(model)
            gap = (float(pi.data @ c[np.searchsorted(N1[0].astype(np.int64) * size_y + N1[1],
                                                       pi.row.astype(np.int64) * size_y + pi.col)]) - C_star) / C_star
            run[f"after {sweeps} sweeps"] = {"sweep time": sweep_time, "basis time": basis_time, "time": time,
                                             "start gap": gap, "forest": forest,
                                             "tree columns": tree_size, "same optimum": abs(C - C_star) <= 1e-12 * C_star}
            print(f"  after {sweeps} sweeps (gap {gap:.1e}, forest {forest}): sweeps {sweep_time:.2f} s, basis "
                  f"{basis_time:.2f} s, CPLEX {time:.2f} s", flush=True)
        stored[name] = run
        path.write_text(json.dumps(stored, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

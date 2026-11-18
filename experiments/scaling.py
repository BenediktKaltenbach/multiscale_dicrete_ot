"""Thesis experiment: scaling of the multiscale solver in n and sparsity of N (baseline, docs/experiment_scaling.md).

Delaunay 1-ring, no pruning, setup as in candidate_sets.py. Dense references on the same problems, run directly
after the multiscale run: POT ot.emd for n <= 8192 (n = 16384 with seed 0 only), CPLEX network simplex on all of
X x Y for n <= CPLEX_LIMIT. Resumable: every run is written to experiments/results/scaling/runs.json at once and
finished (n, kind, seed) are skipped; if POT could not be imported, a later run adds only the missing POT references. plot_scaling.py makes the figures.

python experiments/scaling.py [max n, default 65536]
"""
import json
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

try:
    import ot
except ImportError:  # e.g. blocked by Windows application control; POT references are added by a later run
    ot = None

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.problems import make_problem
from msot.solvers.base import solve_dense
from msot.solvers.cplex import Cplex
from candidate_sets import environment, solve

SIZES = [2**power for power in range(10, 17)]
GRID_SIZES = [4**power for power in range(5, 9)]
SEEDS = (0, 1, 2)
CPLEX_LIMIT = 4096
RESULTS = Path(__file__).parent / "results" / "scaling"


def needs_pot(n, seed):
    return n <= 8192 or (n == 16384 and seed == 0)


def per_level(result):
    """Per level k (fine level last): |N1|, rounds of Alg. 4.1, |N| of the last round and max |N|, per cell of X."""
    levels = []
    for level in result["levels"]:
        cells, history = level["cells"][0], level["history"]
        levels.append({"k": level["k"], "N1 per cell": level["N1"] / cells, "rounds": len(history),
                       "last N per cell": history[-1]["N"] / cells,
                       "max N per cell": max(step["N"] for step in history) / cells})
    return levels


def pot_reference(X, Y, mu, nu, cost):
    start = perf_counter()
    matrix = cost.c(X[:, None], Y[None])
    matrix_time = perf_counter() - start
    start = perf_counter()
    value = ot.emd2(mu / 1e9, nu / 1e9, matrix, numItermax=10**10)
    return {"time": perf_counter() - start, "cost matrix time": matrix_time, "cost per unit mass": float(value)}


def cplex_reference(X, Y, mu, nu, cost):
    start = perf_counter()
    pi, *_ = solve_dense(X, Y, mu, nu, cost, Cplex(warm_start=False))
    total = perf_counter() - start
    return {"time": total, "cost per unit mass": float(pi.data @ cost.c(X[pi.row], Y[pi.col]) / mu.sum())}


def add_pot(result, X, Y, mu, nu, cost, path, stored):
    """POT reference for a finished run that lacks it (POT was not importable then)."""
    if ot is None or "pot" in result or not needs_pot(result["n"], result["seed"]):
        return
    dense = pot_reference(X, Y, mu, nu, cost)
    dense["relative cost difference"] = abs(result["cost per unit mass"] - dense["cost per unit mass"]) / dense["cost per unit mass"]
    dense["added later"] = date.today().isoformat()
    result["pot"] = dense
    path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
    print(f"n = {result['n']:5d} {result['kind']:9s} seed {result['seed']}: pot added, {dense['time']:.1f} s", flush=True)


def main():
    largest = int(sys.argv[1]) if len(sys.argv) > 1 else SIZES[-1]
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "runs.json"
    stored = (json.loads(path.read_text(encoding="utf-8")) if path.exists()
              else {"environment": environment() | {"pot": version("pot") if ot else "not importable"}, "runs": []})
    done = {(run["n"], run["kind"], run["seed"]): run for run in stored["runs"]}
    commit = environment()["commit"]
    cost = SquaredEuclidean(prune=False)
    for seed in SEEDS:
        for n in [n for n in SIZES if n <= largest]:
            for kind in ("uniform", "clustered") + (("grid",) if n in GRID_SIZES else ()):
                X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed))
                if (n, kind, seed) in done:
                    add_pot(done[(n, kind, seed)], X, Y, mu, nu, cost, path, stored)
                    continue
                start = perf_counter()
                HX, HY = build_pair(X, mu, Y, nu)
                hierarchy_time = perf_counter() - start
                result = solve(HX, HY, X, Y, mu, None)
                result.update({"n": n, "kind": kind, "seed": seed, "hierarchy time": hierarchy_time, "K": HX.K,
                               "cells per level": [HX.number_of_cells(k) for k in range(HX.K + 1)],
                               "cells per level of Y": [HY.number_of_cells(k) for k in range(HY.K + 1)],
                               "commit": commit, "date": date.today().isoformat()})
                result["other time"] = result["total time"] - result["solve time"] - result["shield time"]
                result["per level"] = per_level(result)
                line = (f"n = {n:5d} {kind:9s} seed {seed}: total {result['total time']:7.1f} s, "
                        f"|N| per point {result['per level'][-1]['last N per cell']:5.2f}, "
                        f"dual violation {result['dual violation']:.1e}")
                for name, reference, applies in (("pot", pot_reference, ot is not None and needs_pot(n, seed)),
                                                 ("cplex dense", cplex_reference, n <= CPLEX_LIMIT)):
                    if applies:
                        dense = reference(X, Y, mu, nu, cost)
                        dense["relative cost difference"] = (abs(result["cost per unit mass"] - dense["cost per unit mass"])
                                                             / dense["cost per unit mass"])
                        result[name] = dense
                        line += f", {name} {dense['time']:.1f} s (difference {dense['relative cost difference']:.0e})"
                stored["runs"].append(result)
                path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
                print(line, flush=True)


if __name__ == "__main__":
    main()

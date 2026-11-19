"""Thesis experiment: candidate pruning in 4 and 5 dimensions, Delaunay 1-ring, n = 4096 (branch higher-dimensions).

Setup in docs/experiment_pruning_dimensions.md. For each dimension, point kind and seed one problem and one hierarchy
pair is solved without pruning, with the exact face test and with the closest heuristic (order rotated per seed).
With pruning, every call of the face test and of filter_missed is counted.
Writes experiments/results/pruning_dimensions/runs.json; plot_pruning_dimensions.py makes the figures.

python experiments/pruning_dimensions.py [dimensions, default 4 5]
"""
import json
import sys
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np

import msot.shielding
from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.higher_dimensions import SquaredEuclideanPruning, make_problem
from msot.multiscale import solve_multiscale
from msot.solvers.cplex import Cplex
from candidate_sets import dual_violation, environment

N, KINDS, SEEDS = 4096, ("uniform", "clustered"), (0, 1, 2)
CHUNKS = {4: 1024, 5: 256}  # x_A per chunk of the shield search; bounds the memory (5D at 1024 needs more than 15 GB)
VARIANTS = ("off", "exact", "closest")
RESULTS = Path(__file__).parent / "results" / "pruning_dimensions"

filter_calls = []
_filter_missed = msot.shielding.filter_missed


def counted_filter_missed(pair_x, pair_cell, *arguments):
    kept_x, kept_cell = _filter_missed(pair_x, pair_cell, *arguments)
    filter_calls.append({"missed": len(pair_x), "removed": len(pair_x) - len(kept_x)})
    return kept_x, kept_cell


msot.shielding.filter_missed = counted_filter_missed


class Counted:
    """Records per call of the face test the level size, the candidates and the kept ones."""

    def __init__(self, cost):
        self.cost, self.calls = cost, []

    def __getattr__(self, name):
        return getattr(self.cost, name)

    def needed_candidates_all(self, X, S, ys, tolerance):
        needed = self.cost.needed_candidates_all(X, S, ys, tolerance)
        if self.cost.prune:
            self.calls.append({"points": len(X), "candidates": len(needed), "kept": int(needed.sum())})
        return needed


def cost_of(name):
    return Counted(SquaredEuclidean() if name == "off" else SquaredEuclideanPruning(name))


def solve(HX, HY, X, Y, mu, cost):
    """Solves with a Counted cost and returns times, result and the recorded pruning and filter calls."""
    filter_calls.clear()
    start = perf_counter()
    pi, alpha, beta, info = solve_multiscale(HX, HY, cost, Cplex())
    total = perf_counter() - start
    return {"total time": total,
            "solve time": sum(step["time_solve"] for level in info for step in level["history"]),
            "shield time": sum(step["time_shield"] for level in info for step in level["history"]),
            "cost per unit mass": float(pi.data @ cost.c(X[pi.row], Y[pi.col]) / mu.sum()),
            "support of pi": int(pi.nnz),
            "dual violation": dual_violation(alpha, beta, X, Y, cost),
            "levels": [{"k": level["k"], "cells": level["cells"], "N1": level["N1"], "history": level["history"]}
                       for level in info],
            "pruning calls": cost.calls, "filter calls": list(filter_calls)}


def main():
    dimensions = [int(argument) for argument in sys.argv[1:]] or [4, 5]
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "runs.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"environment": environment(), "runs": []}
    runs = [run for run in stored["runs"] if run["dimension"] not in dimensions]
    commit = environment()["commit"]
    for dimension in dimensions:
        msot.shielding.shield.__defaults__ = (1e-12, CHUNKS[dimension])  # (tolerance, chunk)
        for kind in KINDS:
            for seed in SEEDS:
                X, mu, Y, nu = make_problem(N, kind, np.random.default_rng(seed), dimension)
                start = perf_counter()
                HX, HY = build_pair(X, mu, Y, nu)
                hierarchy_time = perf_counter() - start
                for name in VARIANTS[seed:] + VARIANTS[:seed]:
                    result = solve(HX, HY, X, Y, mu, cost_of(name))
                    result.update({"dimension": dimension, "kind": kind, "seed": seed, "pruning": name,
                                   "hierarchy time": hierarchy_time, "K": HX.K, "chunk": CHUNKS[dimension],
                                   "cells per level": [HX.number_of_cells(k) for k in range(HX.K + 1)],
                                   "1-ring at level 0": float(np.diff(HX.neighbours[0][0]).mean()),
                                   "commit": commit, "date": date.today().isoformat()})
                    runs.append(result)
                    dropped = sum(call["candidates"] - call["kept"] for call in result["pruning calls"])
                    candidates = sum(call["candidates"] for call in result["pruning calls"]) or 1
                    print(f"d = {dimension} {kind:9s} seed {seed} {name:7s}: total {result['total time']:6.1f} s, "
                          f"shield {result['shield time']:6.1f} s, rounds {sum(len(level['history']) for level in result['levels'])}, "
                          f"dropped {100 * dropped / candidates:4.1f} %, "
                          f"filter removed {sum(call['removed'] for call in result['filter calls'])}, "
                          f"dual violation {result['dual violation']:.1e}", flush=True)
                    stored["runs"] = runs
                    path.write_text(json.dumps(stored, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

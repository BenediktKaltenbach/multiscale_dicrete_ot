"""Thesis experiment: effect of candidate pruning on the run time, Delaunay 1-ring, at 128 x 128.

Setup in docs/experiment_pruning.md (shared with docs/experiment_candidate_sets.md except for the variable).
For each point kind and seed one problem and one hierarchy pair is solved with pruning off and on (order
alternated per seed). With pruning on, every call of needed_candidates_all and filter_missed is counted.
Writes experiments/results/pruning/runs.json; plot_pruning_runtime.py makes the figures.

python experiments/pruning.py
"""
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import msot.shielding
from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.multiscale import solve_multiscale
from msot.problems import make_problem
from msot.solvers.cplex import Cplex
from candidate_sets import dual_violation, environment

N, KINDS, SEEDS = 128 * 128, ("uniform", "clustered", "grid"), (0, 1, 2)
RESULTS = Path(__file__).parent / "results" / "pruning"


class CountedSquaredEuclidean(SquaredEuclidean):
    """SquaredEuclidean that records, per call of the pruning, the level size, the candidates and the kept ones."""

    def __init__(self, prune):
        super().__init__(prune)
        self.calls = []

    def needed_candidates_all(self, X, S, ys, tolerance, xs=None, chunk=1024):
        needed = super().needed_candidates_all(X, S, ys, tolerance, xs, chunk)
        if self.prune:
            self.calls.append({"points": len(X), "candidates": len(needed), "kept": int(needed.sum())})
        return needed


filter_calls = []
_filter_missed = msot.shielding.filter_missed


def counted_filter_missed(pair_x, pair_cell, *arguments):
    kept_x, kept_cell = _filter_missed(pair_x, pair_cell, *arguments)
    filter_calls.append({"missed": len(pair_x), "removed": len(pair_x) - len(kept_x)})
    return kept_x, kept_cell


msot.shielding.filter_missed = counted_filter_missed


def solve(HX, HY, X, Y, mu, prune):
    cost = CountedSquaredEuclidean(prune)
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
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = []
    for kind in KINDS:
        for seed in SEEDS:
            X, mu, Y, nu = make_problem(N, kind, np.random.default_rng(seed))
            start = perf_counter()
            HX, HY = build_pair(X, mu, Y, nu)
            hierarchy_time = perf_counter() - start
            for prune in ((False, True) if seed % 2 == 0 else (True, False)):
                result = solve(HX, HY, X, Y, mu, prune)
                result.update({"kind": kind, "seed": seed, "prune": prune, "hierarchy time": hierarchy_time,
                               "K": HX.K, "cells per level": [HX.number_of_cells(k) for k in range(HX.K + 1)]})
                runs.append(result)
                dropped = sum(call["candidates"] - call["kept"] for call in result["pruning calls"])
                candidates = sum(call["candidates"] for call in result["pruning calls"]) or 1
                print(f"{kind:9s} seed {seed} prune {prune!s:5s}: total {result['total time']:6.1f} s, "
                      f"shield {result['shield time']:6.1f} s, iterations {sum(len(level['history']) for level in result['levels'])}, "
                      f"dropped {100 * dropped / candidates:4.1f} %, "
                      f"filter removed {sum(call['removed'] for call in result['filter calls'])}, "
                      f"dual violation {result['dual violation']:.1e}", flush=True)
    (RESULTS / "runs.json").write_text(json.dumps({"environment": environment(), "runs": runs}, indent=1),
                                       encoding="utf-8")


if __name__ == "__main__":
    main()

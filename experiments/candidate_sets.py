"""Thesis experiment: candidate sets S(x_A), Delaunay 1-ring vs. 8 sector neighbours, at 128 x 128.

Setup in docs/experiment_candidate_sets.md. For each point kind and seed one problem and one hierarchy pair is built
and solved with both candidate sets (order alternated per seed); every run is checked for global dual feasibility.
Writes experiments/results/candidate_sets/runs.json and q_level0.npz; plot_candidate_sets.py makes the figures.

python experiments/candidate_sets.py
"""
import json
import platform
from importlib.metadata import version
from pathlib import Path
from subprocess import run
from time import perf_counter

import numpy as np

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.mesh import candidate_statistics, sector_neighbours
from msot.multiscale import solve_multiscale
from msot.problems import make_problem
from msot.solvers.cplex import Cplex

N, KINDS, SEEDS, SECTORS = 128 * 128, ("uniform", "clustered", "grid"), (0, 1, 2), 8
RESULTS = Path(__file__).parent / "results" / "candidate_sets"


def environment():
    commit = run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = run(["git", "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip() != ""
    return {"commit": commit, "uncommitted changes in src": dirty, "python": platform.python_version(),
            "platform": platform.platform(), "processor": platform.processor(),
            **{package: version(package) for package in ("numpy", "scipy", "cplex")}}


def dual_violation(alpha, beta, X, Y, cost, block=512):
    """max over X x Y of alpha + beta - c, in blocks of rows of X (Cor. 3.10)."""
    return max(float((alpha[start:start + block, None] + beta[None] - cost.c(X[start:start + block, None], Y[None])).max())
               for start in range(0, len(X), block))


def sectors(HX, k):
    return sector_neighbours(HX.rep_points(k), SECTORS)


def solve(HX, HY, X, Y, mu, candidates):
    cost = SquaredEuclidean(prune=False)
    start = perf_counter()
    pi, alpha, beta, info = solve_multiscale(HX, HY, cost, Cplex(), candidates=candidates)
    total = perf_counter() - start
    levels = [{"k": level["k"], "cells": level["cells"], "N1": level["N1"], "history": level["history"]}
              for level in info]
    return {"total time": total,
            "solve time": sum(step["time_solve"] for level in info for step in level["history"]),
            "shield time": sum(step["time_shield"] for level in info for step in level["history"]),
            "cost per unit mass": float(pi.data @ cost.c(X[pi.row], Y[pi.col]) / mu.sum()),
            "support of pi": int(pi.nnz),
            "dual violation": dual_violation(alpha, beta, X, Y, cost),
            "levels": levels}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs, q_level0 = [], {}
    for kind in KINDS:
        for seed in SEEDS:
            X, mu, Y, nu = make_problem(N, kind, np.random.default_rng(seed))
            start = perf_counter()
            HX, HY = build_pair(X, mu, Y, nu)
            hierarchy_time = perf_counter() - start
            variants = {"1-ring": None, f"{SECTORS} sectors": sectors}
            order = list(variants) if seed % 2 == 0 else list(variants)[::-1]
            for name in order:
                result = solve(HX, HY, X, Y, mu, variants[name])
                result.update({"kind": kind, "seed": seed, "S": name, "hierarchy time": hierarchy_time,
                               "K": HX.K, "cells per level": [HX.number_of_cells(k) for k in range(HX.K + 1)]})
                runs.append(result)
                level0 = result["levels"][-1]
                print(f"{kind:9s} seed {seed} {name:9s}: total {result['total time']:6.1f} s, "
                      f"level-0 iterations {len(level0['history'])}, |N| per point {level0['history'][-1]['N'] / N:.2f}, "
                      f"dual violation {result['dual violation']:.1e}", flush=True)
            for name, S in (("1-ring", HX.neighbours[0]), (f"{SECTORS} sectors", sectors(HX, 0))):
                q_level0[f"{kind}/{seed}/{name}"] = candidate_statistics(HX.rep_points(0), S)[0]
    (RESULTS / "runs.json").write_text(json.dumps({"environment": environment(), "runs": runs}, indent=1),
                                       encoding="utf-8")
    np.savez(RESULTS / "q_level0.npz", **q_level0)


if __name__ == "__main__":
    main()

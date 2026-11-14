"""Thesis experiment: pruning off / exact facets / closest 60 % (centre t(x_A)) on clustered points in d = 2, 4, 5.

One session for all three dimensions, so the times of a dimension are comparable. d = 2: the problems of
pruning.py (msot.problems, n = 128^2); d = 4, 5: those of pruning_dimensions.py (n = 4096). Order rotated per seed.
Writes experiments/results/pruning_clustered/runs.json; plot_pruning_summary.py makes the figure.

python experiments/pruning_clustered.py [dimensions, default 2 4 5]
"""
import json
import sys
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np

import msot.problems
import msot.shielding
from msot.hierarchy import build_pair
from msot.higher_dimensions import make_problem
from candidate_sets import environment
from pruning_dimensions import CHUNKS, SEEDS, VARIANTS, cost_of, solve

SIZES = {2: 128 * 128, 4: 4096, 5: 4096}
CHUNKS = {2: 4096, **CHUNKS}
RESULTS = Path(__file__).parent / "results" / "pruning_clustered"


def problem(dimension, seed):
    rng = np.random.default_rng(seed)
    if dimension == 2:
        return msot.problems.make_problem(SIZES[2], "clustered", rng)
    return make_problem(SIZES[dimension], "clustered", rng, dimension)


def main():
    dimensions = [int(argument) for argument in sys.argv[1:]] or [2, 4, 5]
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "runs.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"environment": environment(), "runs": []}
    runs = [run for run in stored["runs"] if run["dimension"] not in dimensions]
    commit = environment()["commit"]
    for dimension in dimensions:
        msot.shielding.shield.__defaults__ = (1e-12, CHUNKS[dimension])  # (tolerance, chunk)
        for seed in SEEDS:
            X, mu, Y, nu = problem(dimension, seed)
            start = perf_counter()
            HX, HY = build_pair(X, mu, Y, nu)
            hierarchy_time = perf_counter() - start
            for name in VARIANTS[seed:] + VARIANTS[:seed]:
                result = solve(HX, HY, X, Y, mu, cost_of(name))
                result.update({"dimension": dimension, "kind": "clustered", "seed": seed, "pruning": name, "n": len(X),
                               "hierarchy time": hierarchy_time, "chunk": CHUNKS[dimension],
                               "cells per level": [HX.number_of_cells(k) for k in range(HX.K + 1)],
                               "commit": commit, "date": date.today().isoformat()})
                runs.append(result)
                dropped = sum(call["candidates"] - call["kept"] for call in result["pruning calls"])
                candidates = sum(call["candidates"] for call in result["pruning calls"]) or 1
                print(f"d = {dimension} seed {seed} {name:7s}: total {result['total time']:6.1f} s, "
                      f"shield {result['shield time']:6.1f} s, solve {result['solve time']:5.1f} s, "
                      f"dropped {100 * dropped / candidates:4.1f} %, "
                      f"filter removed {sum(call['removed'] for call in result['filter calls'])}, "
                      f"dual violation {result['dual violation']:.1e}", flush=True)
                stored["runs"] = runs
                path.write_text(json.dumps(stored, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

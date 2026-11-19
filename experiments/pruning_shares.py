"""Thesis experiment: keep share of the closest rule in 4D. The stored runs used the centre t(x_A) (commit 883ce1e);
the rule now ranks by the mean of the t(x_s).

Problems and settings as in pruning_dimensions.py (docs/experiment_pruning_dimensions.md). Per problem one run
without pruning and one per keep share of SquaredEuclideanPruning("closest"); the order is rotated per seed.
Writes experiments/results/pruning_shares/runs.json.

python experiments/pruning_shares.py [dimension, default 4]
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
from candidate_sets import environment
from pruning_dimensions import CHUNKS, KINDS, N, SEEDS, Counted, solve

VARIANTS = ("off", 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
RESULTS = Path(__file__).parent / "results" / "pruning_shares"


def cost_of(variant):
    return Counted(SquaredEuclidean() if variant == "off" else SquaredEuclideanPruning("closest", keep_share=variant))


def main():
    dimension = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "runs.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"environment": environment(), "runs": []}
    runs = [run for run in stored["runs"] if run["dimension"] != dimension]
    commit = environment()["commit"]
    msot.shielding.shield.__defaults__ = (1e-12, CHUNKS[dimension])  # (tolerance, chunk)
    for kind in KINDS:
        for seed in SEEDS:
            X, mu, Y, nu = make_problem(N, kind, np.random.default_rng(seed), dimension)
            start = perf_counter()
            HX, HY = build_pair(X, mu, Y, nu)
            hierarchy_time = perf_counter() - start
            shift = 2 * seed
            for variant in VARIANTS[shift:] + VARIANTS[:shift]:
                result = solve(HX, HY, X, Y, mu, cost_of(variant))
                result.update({"dimension": dimension, "kind": kind, "seed": seed, "keep share": variant,
                               "hierarchy time": hierarchy_time, "chunk": CHUNKS[dimension],
                               "cells per level": [HX.number_of_cells(k) for k in range(HX.K + 1)],
                               "commit": commit, "date": date.today().isoformat()})
                runs.append(result)
                print(f"d = {dimension} {kind:9s} seed {seed} share {str(variant):4s}: total {result['total time']:6.1f} s, "
                      f"shield {result['shield time']:6.1f} s, solve {result['solve time']:5.1f} s, "
                      f"filter removed {sum(call['removed'] for call in result['filter calls'])}, "
                      f"dual violation {result['dual violation']:.1e}", flush=True)
                stored["runs"] = runs
                path.write_text(json.dumps(stored, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

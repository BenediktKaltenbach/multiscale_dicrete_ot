"""Mesh hierarchy (option B) vs. quadtree (option A): partition quality and solve_multiscale behaviour."""
from pathlib import Path
from time import perf_counter

import numpy as np

from hierarchy_quality import level_table
from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair, build_pair_quadtree
from msot.multiscale import solve_multiscale
from msot.problems import make_problem
from msot.solvers.highs import Highs

BUILDERS = {
    "B mesh": lambda X, mu, Y, nu: build_pair(X, mu, Y, nu, n_dense=256),
    "A quadtree": build_pair_quadtree,
}


def run(X, mu, Y, nu, builder):
    start = perf_counter()
    HX, HY = builder(X, mu, Y, nu)
    built = perf_counter()
    pi, _, _, info = solve_multiscale(HX, HY, SquaredEuclidean(), Highs())
    histories = [level["history"] for level in info]
    return HX, {
        "C": pi.data @ SquaredEuclidean().c(X[pi.row], Y[pi.col]),
        "K": HX.K,
        "iterations": sum(len(history) for history in histories),
        "iterations at level 0": len(histories[-1]),
        "max |N| / cells": max(step["N"] / level["cells"][0] for level in info for step in level["history"]),
        "final |N| / n": histories[-1][-1]["N"] / len(X),
        "time build": built - start,
        "time solve": sum(step["time_solve"] for history in histories for step in history),
        "time shield": sum(step["time_shield"] for history in histories for step in history),
    }


def main():
    report = []
    columns = ["K", "iterations", "iterations at level 0", "max |N| / cells", "final |N| / n",
               "time build", "time solve", "time shield"]
    for n in [1024, 4096]:
        for kind, mask in [("uniform", None), ("clustered", None), ("grid", "disc")]:
            X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(0), mask=mask)
            report.append(f"== {kind}, mask {mask}, n = {n}")
            report.append(f"{'':12s}" + "".join(f"{name:>23s}" for name in columns))
            costs, tables = [], []
            for name, builder in BUILDERS.items():
                HX, result = run(X, mu, Y, nu, builder)
                costs.append(result["C"])
                report.append(f"{name:12s}" + "".join(f"{result[c]:23.3g}" for c in columns))
                tables.append(f"-- {name}, X hierarchy\n{level_table(HX)}")
            report.append(f"relative cost difference A vs B: {abs(costs[0] - costs[1]) / costs[0]:.1e}")
            report += tables + [""]
            print("\n".join(report[-len(tables) - 5:]), flush=True)

    results = Path(__file__).parent / "results"
    results.mkdir(exist_ok=True)
    (results / "compare_hierarchies.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()

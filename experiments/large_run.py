"""One large multiscale run, e.g. 256 x 256: per-level statistics and a global dual feasibility check (Cor. 3.10).

python experiments/large_run.py n kind hierarchy candidates [check]
  kind: uniform | clustered | anisotropic | grid-disc; hierarchy: mesh | quadtree; candidates: 1-ring | 8-sectors
"""
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair, build_pair_quadtree
from msot.mesh import sector_neighbours
from msot.multiscale import solve_multiscale
from msot.problems import gaussian_masses, make_problem, quantize, sample_points
from msot.solvers.highs import Highs

RESULTS = Path(__file__).parent / "results"


def problem(n, kind, rng):
    if kind == "anisotropic":
        X, Y = (sample_points(n, "uniform", rng) * [1.0, 0.1] for _ in range(2))
        return X, quantize(gaussian_masses(X, rng), 10**9), Y, quantize(gaussian_masses(Y, rng), 10**9)
    if kind == "grid-disc":
        return make_problem(n, "grid", rng, mask="disc")
    return make_problem(n, kind, rng)


def dual_violation_blocks(alpha, beta, X, Y, cost, block=512):
    """max over X x Y of alpha + beta - c, in blocks of rows of X."""
    return max((alpha[start:start + block, None] + beta[None] - cost.c(X[start:start + block, None], Y[None])).max()
               for start in range(0, len(X), block))


def main():
    n, kind, hierarchy, candidates = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    check = len(sys.argv) > 5 and sys.argv[5] == "check"
    name = f"large_{n}_{kind}_{hierarchy}_{candidates}"
    lines = [f"== n = {n}, {kind}, {hierarchy} hierarchy, S = {candidates}, integer masses, Highs warm start"]

    def log(text):
        print(text, flush=True)
        lines.append(text)

    start = perf_counter()
    X, mu, Y, nu = problem(n, kind, np.random.default_rng(0))
    HX, HY = build_pair(X, mu, Y, nu) if hierarchy == "mesh" else build_pair_quadtree(X, mu, Y, nu)
    built = perf_counter()
    log(f"problem and hierarchy: {built - start:.1f} s, K = {HX.K}, cells of X per level {[HX.number_of_cells(k) for k in range(HX.K + 1)]}")
    choose = None if candidates == "1-ring" else lambda H, k: sector_neighbours(H.rep_points(k), 8)
    cost, solver = SquaredEuclidean(), Highs()
    pi, alpha, beta, info = solve_multiscale(HX, HY, cost, solver, candidates=choose)
    solved = perf_counter()
    log(" k   cells X   |N1|/cell  iterations  |N| last/cell  solve s  shield s")
    for level in info:
        history = level["history"]
        log(f"{level['k']:2d} {level['cells'][0]:9d} {level['N1'] / level['cells'][0]:10.2f} {len(history):11d} "
            f"{history[-1]['N'] / level['cells'][0]:14.2f} {sum(h['time_solve'] for h in history):8.1f} "
            f"{sum(h['time_shield'] for h in history):9.1f}")
    for level in info[-3:]:
        log(f"level {level['k']} per iteration: C (relative to the first) / |N| / solve s")
        first = level["history"][0]["C"]
        for step in level["history"]:
            log(f"   {step['C'] / first - 1:+.3e}  {step['N']:9d}  {step['time_solve']:7.1f}")
    log(f"multiscale total: {solved - built:.1f} s, cost {pi.data @ cost.c(X[pi.row], Y[pi.col]) / mu.sum():.10g} per unit mass, "
        f"support of pi {pi.nnz} entries")
    if check:
        violation = dual_violation_blocks(alpha, beta, X, Y, cost)
        log(f"global dual feasibility: max(alpha + beta - c) = {violation:.2e} ({perf_counter() - solved:.1f} s)")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

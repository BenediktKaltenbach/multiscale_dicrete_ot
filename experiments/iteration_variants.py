"""Variants of Alg. 4.1 against the long iteration tail; implemented here only, msot is unchanged.

A baseline: N <- shield(pi).  B cumulative: N <- shield(pi) | N.  C two-phase: first N <- spt pi | step-ii entries
(no Y_hat search) until the cost stalls, then the baseline loop; only the last N has to be shielding (Cor. 3.10).
D two-phase, short: as C, but phase 1 stops once a round improves the cost by less than 1e-6 (relative).
E1 dilated N1: refine(spt pi) plus, for each entry (x, y), the entries (x, y') for y' in the 1-ring of y at the fine level.
E2 refined coarse N: N1 = children x children of the coarse level's final shielding N instead of spt pi.

python experiments/iteration_variants.py n kind [variant ...]

(kind: grid-disc | uniform | clustered)
"""
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

import msot.multiscale as multiscale
from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.problems import make_problem
from msot.solvers.cplex import Cplex
from msot.shielding import extract_map, shield, unique_entries

RESULTS = Path(__file__).parent / "results"
original_solve_sparse = multiscale.solve_sparse
original_refine = multiscale.refine
final_neighbourhoods = {}


def make_solver():
    return Cplex()


def local_neighbourhood(pi, S, size_y):
    """spt pi and the step-ii entries (x_A, t(x_s)), x_s in S(x_A), without the Y_hat search."""
    t = extract_map(pi)
    index_pointer, indices = S
    owner = np.repeat(np.arange(len(index_pointer) - 1), np.diff(index_pointer))
    return unique_entries(np.concatenate([pi.row, owner]), np.concatenate([pi.col, t[indices]]), size_y)


def keys(rows, columns, size_y):
    return rows.astype(np.int64) * size_y + columns


def loop(X, mu, Yh, k, cost, S, N, solver, next_neighbourhood, tolerance=1e-12, max_iterations=100, state=None):
    Y, nu = Yh.rep_points(k), Yh.mass[k]
    rows, columns = N
    C_old, history = np.inf, []
    for _ in range(max_iterations):
        start = perf_counter()
        pi, alpha, beta, state = solver.solve_local(rows, columns, cost.c(X[rows], Y[columns]), mu, nu, state)
        solved = perf_counter()
        new_rows, new_columns = next_neighbourhood(pi, rows, columns)
        C = float(pi.data @ cost.c(X[pi.row], Y[pi.col]))
        history.append({"C": C, "N": len(rows), "time_solve": solved - start, "time_shield": perf_counter() - solved})
        if np.isin(keys(new_rows, new_columns, len(Y)), keys(rows, columns, len(Y))).all() or C >= C_old - tolerance * abs(C_old):
            return pi, alpha, beta, (new_rows, new_columns), history, state
        rows, columns, C_old = new_rows, new_columns, C
    raise RuntimeError("did not converge")


def variant(name):
    def solve_sparse(X, mu, Yh, k, cost, S, N1, solver, tolerance=1e-12, max_iterations=100):
        size_y = len(Yh.mass[k])
        full = lambda pi, rows, columns: shield(pi, S, X, Yh, k, cost, tolerance)
        if name == "B cumulative":
            def grow(pi, rows, columns):
                new_rows, new_columns = shield(pi, S, X, Yh, k, cost, tolerance)
                return unique_entries(np.concatenate([new_rows, rows]), np.concatenate([new_columns, columns]), size_y)
            return loop(X, mu, Yh, k, cost, S, N1, solver, grow, tolerance, max_iterations)[:5]
        if name in ("C two-phase", "D two-phase, short"):
            local = lambda pi, rows, columns: local_neighbourhood(pi, S, size_y)
            phase_one_tolerance = tolerance if name == "C two-phase" else 1e-6
            pi, _, _, N, first, state = loop(X, mu, Yh, k, cost, S, N1, solver, local, phase_one_tolerance, max_iterations)
            pi, alpha, beta, N, second, _ = loop(X, mu, Yh, k, cost, S, shield(pi, S, X, Yh, k, cost, tolerance),
                                                 solver, full, tolerance, max_iterations, state)
            for step in first:
                step["phase"] = 1
            return pi, alpha, beta, N, first + second
        result = loop(X, mu, Yh, k, cost, S, N1, solver, full, tolerance, max_iterations)[:5]
        final_neighbourhoods[k] = result[3]
        return result
    return solve_sparse


class Entries:
    def __init__(self, rows, columns):
        self.row, self.col = rows, columns


def dilated_refine(pi, HX, HY, k):
    rows, columns = original_refine(pi, HX, HY, k)
    index_pointer, indices = HY.neighbours[k]
    counts = np.diff(index_pointer)[columns]
    extra_rows = np.repeat(rows, counts)
    extra_columns = indices[np.repeat(index_pointer[columns], counts) + np.arange(counts.sum())
                            - np.repeat(np.cumsum(counts) - counts, counts)]
    return unique_entries(np.concatenate([rows, extra_rows]), np.concatenate([columns, extra_columns]),
                          HY.number_of_cells(k))


def coarse_neighbourhood_refine(pi, HX, HY, k):
    if k + 1 not in final_neighbourhoods:
        return original_refine(pi, HX, HY, k)
    return original_refine(Entries(*final_neighbourhoods[k + 1]), HX, HY, k)


def main():
    n, kind = int(sys.argv[1]), sys.argv[2]
    X, mu, Y, nu = make_problem(n, "grid" if kind == "grid-disc" else kind, np.random.default_rng(0),
                                mask="disc" if kind == "grid-disc" else None)
    HX, HY = build_pair(X, mu, Y, nu)
    cost = SquaredEuclidean()
    lines = [f"== n = {n}, {kind}, mesh hierarchy, 1-ring; per level: iterations (phase 1 + phase 2), max |N|/cell, "
             f"solve s, shield s"]
    names = sys.argv[3:] or ["A baseline", "D two-phase, short", "E1 dilated N1", "E2 refined coarse N"]
    for name in names:
        final_neighbourhoods.clear()
        multiscale.solve_sparse = variant(name)
        multiscale.refine = {"E1 dilated N1": dilated_refine, "E2 refined coarse N": coarse_neighbourhood_refine}.get(
            name, original_refine)
        start = perf_counter()
        pi, _, _, info = multiscale.solve_multiscale(HX, HY, cost, make_solver())
        total = perf_counter() - start
        multiscale.solve_sparse, multiscale.refine = original_solve_sparse, original_refine
        lines.append(f"-- {name}: total {total:.1f} s, cost {pi.data @ cost.c(X[pi.row], Y[pi.col]):.12e}")
        for level in info[-3:]:
            history = level["history"]
            phase_one = sum(step.get("phase") == 1 for step in history)
            lines.append(f"   level {level['k']}: iterations {phase_one} + {len(history) - phase_one}, "
                         f"|N1|/cell {history[0]['N'] / level['cells'][0]:.2f}, "
                         f"max |N|/cell {max(step['N'] for step in history) / level['cells'][0]:.2f}, "
                         f"solve {sum(step['time_solve'] for step in history):.1f} s, "
                         f"shield {sum(step['time_shield'] for step in history):.1f} s")
        print("\n".join(lines[-4:]), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"iteration_variants_{n}_{kind}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

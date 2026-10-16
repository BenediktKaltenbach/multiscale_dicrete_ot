"""HiGHS cold (fresh LP per call) vs warm start (one model per solve_sparse, primal simplex from the old basis)."""
import shutil
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.multiscale import solve_multiscale
from msot.problems import make_problem
from msot.solvers.highs import Highs


def run(X, mu, Y, nu, HX, HY, warm_start, primal_tolerance):
    solver = Highs(primal_tolerance, warm_start=warm_start)
    start = perf_counter()
    try:
        pi, _, _, info = solve_multiscale(HX, HY, SquaredEuclidean(), solver)
    except RuntimeError as error:
        return {"failed": f"{error} after {len(solver.simplex_iterations)} successful local solves"}
    histories = [level["history"] for level in info]
    return {
        "C": pi.data @ SquaredEuclidean().c(X[pi.row], Y[pi.col]),
        "time total": perf_counter() - start,
        "time solve": sum(step["time_solve"] for history in histories for step in history),
        "time shield": sum(step["time_shield"] for history in histories for step in history),
        "iterations": sum(len(history) for history in histories),
        "iterations at level 0": len(histories[-1]),
        "simplex iterations": sum(solver.simplex_iterations),
        "simplex at level 0": sum(solver.simplex_iterations[-len(histories[-1]):]),
    }


def main(n=4096):
    columns = ["time total", "time solve", "time shield", "iterations", "iterations at level 0",
               "simplex iterations", "simplex at level 0"]
    report = []
    cases = [("uniform", None, 1e-10), ("clustered", None, 1e-10), ("grid", "disc", 1e-10), ("grid", "disc", 1e-9)]
    for kind, mask, primal_tolerance in cases:
        X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(0), mask=mask)
        HX, HY = build_pair(X, mu, Y, nu)
        results = {name: run(X, mu, Y, nu, HX, HY, warm_start, primal_tolerance)
                   for name, warm_start in [("cold", False), ("warm", True)]}
        report.append(f"== {kind}, mask {mask}, n = {n}, K = {HX.K}, primal tolerance {primal_tolerance:g}")
        report.append(f"{'':6s}" + "".join(f"{name:>22s}" for name in columns))
        for name, result in results.items():
            report.append(f"{name:6s}" + (result.get("failed") or "".join(f"{result[c]:22.4g}" for c in columns)))
        if all("C" in result for result in results.values()):
            C_cold, C_warm = results["cold"]["C"], results["warm"]["C"]
            report.append(f"relative cost difference warm vs cold: {abs(C_warm - C_cold) / C_cold:.1e}")
        report.append("")
        print("\n".join(report[-6:]), flush=True)

    text = "\n".join(report) + "\n"
    results = Path(__file__).parent / "results"
    try:
        results.mkdir(exist_ok=True)
        (results / "warm_start.txt").write_text(text)
    except OSError:
        temporary = Path(tempfile.mkdtemp()) / "warm_start.txt"
        temporary.write_text(text)
        shutil.copy(temporary, results / "warm_start.txt")


if __name__ == "__main__":
    main()

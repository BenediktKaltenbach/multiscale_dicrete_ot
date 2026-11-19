"""Repeat of the n = 65536 warm-start timing: CPLEX on N1 of level 0 from scratch (A) and from the north-west
start (B), alternated A B B A A B B A per problem so that drift hits both alike. Records the power status
before each solve (Win32_Battery, 2 = mains). Writes results/domain_decomposition/warm_repeat.json after every
solve; a restart continues where the stored solves end.

python experiments/domain_decomposition_warm_repeat.py [n, default 65536]
"""
import json
import subprocess
import sys
from time import perf_counter

import numpy as np

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.multiscale import dilate, refine
from msot.problems import make_problem
from msot.solvers.cplex import Cplex
from domain_decomposition import RESULTS, coarse_levels, north_west_start
from domain_decomposition_warm import KINDS, solve, start_basis

ORDER = "ABBAABBA"


def on_mains():
    status = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                            capture_output=True, text=True).stdout.strip()
    return status == "2"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    path = RESULTS / "warm_repeat.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for kind, seed in KINDS:
        name = f"{kind}_{n}_{seed}"
        solves = stored.get(name, {}).get("solves", [])
        if len(solves) == len(ORDER):
            continue
        print(name, flush=True)
        X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed))
        HX, HY = build_pair(X, mu, Y, nu)
        cost, solver = SquaredEuclidean(), Cplex(warm_start=False)
        coarse = coarse_levels(HX, HY, cost, Cplex())
        N1 = dilate(refine(coarse, HX, HY, 0), HY.neighbours[0], HY.number_of_cells(0))
        mu0, nu0 = HX.mass[0], HY.mass[0]
        c = cost.c(HX.rep_points(0)[N1[0]], HY.rep_points(0)[N1[1]])
        pi = north_west_start(coarse, HX, HY, 0)
        for variant in ORDER[len(solves):]:
            mains = on_mains()
            start = perf_counter()
            model = solver.build(*N1, c, mu0, nu0, 1.0, True)
            if variant == "B":
                basic = start_basis(pi, N1, len(mu0), len(nu0))[0]
                status = model.start.status
                model.start.set_start(np.where(basic, status.basic, status.at_lower_bound).tolist(),
                                      [status.basic] + [status.at_lower_bound] * (len(mu0) + len(nu0) - 1),
                                      [], [], [], [])
            setup = perf_counter() - start
            time, objective = solve(model)
            solves.append({"variant": variant, "setup": setup, "time": time, "objective": objective, "mains": mains})
            stored[name] = {"kind": kind, "n": n, "seed": seed, "N1": len(c), "solves": solves}
            path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
            print(f"  {variant}: setup {setup:.2f} s, CPLEX {time:.2f} s, mains {mains}", flush=True)


if __name__ == "__main__":
    main()

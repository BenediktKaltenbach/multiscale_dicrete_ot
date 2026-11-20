"""Whole solve_multiscale with and without the north-west start, alternated A B B A per problem
(A = north_west=None, B = north_west="index"), power status recorded before each run (Win32_Battery, 2 = mains).
Writes results/north_west/runs.json after every run; a restart continues where the stored runs end.

python experiments/north_west_timing.py [n, default 65536]
"""
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.multiscale import solve_multiscale
from msot.problems import make_problem
from msot.solvers.cplex import Cplex

KINDS = (("uniform", 0), ("clustered", 0), ("grid", 2))
ORDER = "ABBA"
RESULTS = Path(__file__).parent / "results" / "north_west"


def on_mains():
    status = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                            capture_output=True, text=True).stdout.strip()
    return status == "2"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 65536
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "runs.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cost = SquaredEuclidean()
    for kind, seed in KINDS:
        name = f"{kind}_{n}_{seed}"
        runs = stored.get(name, [])
        if len(runs) == len(ORDER):
            continue
        print(name, flush=True)
        X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed))
        HX, HY = build_pair(X, mu, Y, nu)
        for variant in ORDER[len(runs):]:
            mains = on_mains()
            start = perf_counter()
            pi, alpha, beta, info = solve_multiscale(HX, HY, cost, Cplex(), north_west="index" if variant == "B" else None)
            total = perf_counter() - start
            levels = [{"k": level["k"], "rounds": len(level["history"]),
                       "first solve": level["history"][0]["time_solve"],
                       "solve": sum(round_["time_solve"] for round_ in level["history"]),
                       "shield": sum(round_["time_shield"] for round_ in level["history"])} for level in info]
            runs.append({"variant": variant, "north_west": variant == "B", "mains": mains, "total": total,
                         "C": float(pi.data @ cost.c(X[pi.row], Y[pi.col])), "levels": levels})
            stored[name] = runs
            path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
            print(f"  {variant}: total {total:.1f} s, level 0 first solve {levels[-1]['first solve']:.1f} s, "
                  f"rounds {levels[-1]['rounds']}, mains {mains}", flush=True)


if __name__ == "__main__":
    main()

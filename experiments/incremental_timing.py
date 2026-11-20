"""Whole solve_multiscale with full and with incremental shielding, alternated A B B A per problem
(A = incremental=False, B = incremental=True), power status recorded before each run (Win32_Battery, 2 = mains).
Per level: rounds, solve and shield time; per round of level 0: shield time. Writes results/incremental/runs.json
after every run; a restart continues where the stored runs end.

python experiments/incremental_timing.py [kind:n:seed ...] (default: the problems in PROBLEMS)
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

PROBLEMS = (("grid", 16384, 2), ("uniform", 16384, 0), ("clustered", 32768, 0),
            ("grid", 65536, 2), ("uniform", 65536, 0), ("clustered", 65536, 0))
ORDER = "ABBA"
RESULTS = Path(__file__).parent / "results" / "incremental"


def on_mains():
    status = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                            capture_output=True, text=True).stdout.strip()
    return status == "2"


def main():
    problems = [(kind, int(n), int(seed)) for kind, n, seed in (argument.split(":") for argument in sys.argv[1:])]
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "runs.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cost = SquaredEuclidean()
    for kind, n, seed in problems or PROBLEMS:
        name = f"{kind}_{n}_{seed}"
        runs = stored.get(name, [])
        if len(runs) == len(ORDER):
            continue
        print(name, flush=True)
        X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed))
        HX, HY = build_pair(X, mu, Y, nu)
        for variant in ORDER[len(runs):]:
            mains = on_mains()
            began = perf_counter()
            pi, alpha, beta, info = solve_multiscale(HX, HY, cost, Cplex(), incremental=variant == "B")
            total = perf_counter() - began
            levels = [{"k": level["k"], "rounds": len(level["history"]),
                       "solve": sum(round_["time_solve"] for round_ in level["history"]),
                       "shield": sum(round_["time_shield"] for round_ in level["history"])} for level in info]
            runs.append({"variant": variant, "incremental": variant == "B", "mains": mains, "total": total,
                         "C": float(pi.data @ cost.c(X[pi.row], Y[pi.col])), "levels": levels,
                         "level 0 shield per round": [round_["time_shield"] for round_ in info[-1]["history"]]})
            stored[name] = runs
            path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
            print(f"  {variant}: total {total:.1f} s, level 0 shield {levels[-1]['shield']:.1f} s, "
                  f"solve {levels[-1]['solve']:.1f} s, rounds {levels[-1]['rounds']}, mains {mains}", flush=True)


if __name__ == "__main__":
    main()

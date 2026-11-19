"""Stage 1 of the north-west variant check: the first CPLEX solve of level 0 from scratch and from three versions
of the north-west start, which differ only in the order in which children and entries are matched:

    index      the order of the indices (msot.multiscale.north_west_start)
    cell       in each coarse cell, children and the entries of its row (column) sorted along the cell's mean
               transport direction (mass-weighted rep(y) - rep(x) over the cell's entries of pi)
    entry      as cell, and within each entry (x, y) of pi the pieces of both sides sorted along rep(y) - rep(x)

Variants alternated S I C E E C I S per repeat (S = from scratch). Per solve: start gap (C(start) - C*) / C* on N1,
time to build the start, time of solve_local (model, basis, CPLEX), power status before the solve.
Writes results/north_west/variants.json after every solve; a restart continues where the stored solves end.

python experiments/north_west_variants.py [n, ...] (default 16384 65536)
"""
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.multiscale import dilate, north_west, north_west_start, refine, solve_sparse
from msot.problems import make_problem
from msot.solvers.cplex import Cplex

KINDS = (("uniform", 0), ("clustered", 0), ("grid", 2))
ORDER = "SICEECIS"
NAMES = {"S": "scratch", "I": "index", "C": "cell", "E": "entry"}
RESULTS = Path(__file__).parent / "results" / "north_west"


def on_mains():
    status = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                            capture_output=True, text=True).stdout.strip()
    return status == "2"


def directed_start(pi, HX, HY, k, per_entry):
    """north_west_start with children and entries sorted along transport directions (variants cell and entry)."""
    pi = pi.tocoo()
    entries = np.arange(len(pi.data))
    displacement = HY.rep_points(k + 1)[pi.col] - HX.rep_points(k + 1)[pi.row]
    split = []
    for H, coarse, partner in ((HX, pi.row, HY.rep_points(k + 1)[pi.col]), (HY, pi.col, HX.rep_points(k + 1)[pi.row])):
        cells = H.number_of_cells(k + 1)
        direction = np.stack([np.bincount(coarse, pi.data * displacement[:, i], minlength=cells)
                              for i in range(displacement.shape[1])], axis=1)
        parent, points = H.parent[k], H.rep_points(k)
        children = np.lexsort((np.einsum("ij,ij->i", points, direction[parent]), parent))
        by_cell = np.lexsort((np.einsum("ij,ij->i", partner, direction[coarse]), coarse))
        child, entry, amount = north_west(children, H.mass[k][children], entries[by_cell], pi.data[by_cell])
        keep = amount > 0
        split.append((child[keep], entry[keep], amount[keep]))
    (x, entry_x, amount_x), (y, entry_y, amount_y) = split
    if per_entry:
        order_x = np.lexsort((np.einsum("ij,ij->i", HX.rep_points(k)[x], displacement[entry_x]), entry_x))
        order_y = np.lexsort((np.einsum("ij,ij->i", HY.rep_points(k)[y], displacement[entry_y]), entry_y))
    else:
        order_x, order_y = np.argsort(entry_x, kind="stable"), np.argsort(entry_y, kind="stable")
    rows, columns, amount = north_west(x[order_x], amount_x[order_x], y[order_y], amount_y[order_y])
    keep = amount > 0
    return coo_matrix((amount[keep], (rows[keep], columns[keep])),
                      shape=(HX.number_of_cells(k), HY.number_of_cells(k)))


START = {"I": north_west_start,
         "C": lambda pi, HX, HY, k: directed_start(pi, HX, HY, k, False),
         "E": lambda pi, HX, HY, k: directed_start(pi, HX, HY, k, True)}


def coarse_coupling(HX, HY, cost):
    """The optimal coupling at level 1, by solve_multiscale's loop down to level 1."""
    pi = coo_matrix((HX.mass[HX.K], ([0], [0])), shape=(1, 1))
    for k in range(HX.K - 1, 0, -1):
        N1 = dilate(refine(pi, HX, HY, k), HY.neighbours[k], HY.number_of_cells(k))
        pi = solve_sparse(HX.rep_points(k), HX.mass[k], HY, k, cost, HX.neighbours[k], N1, Cplex(),
                          start=north_west_start(pi, HX, HY, k))[0]
    return pi


def main():
    sizes = [int(n) for n in sys.argv[1:]] or [16384, 65536]
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "variants.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cost = SquaredEuclidean()
    for n in sizes:
        for kind, seed in KINDS:
            name = f"{kind}_{n}_{seed}"
            run = stored.get(name, {"kind": kind, "n": n, "seed": seed, "solves": []})
            if len(run["solves"]) == len(ORDER):
                continue
            print(name, flush=True)
            X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed))
            HX, HY = build_pair(X, mu, Y, nu)
            coarse = coarse_coupling(HX, HY, cost)
            rows, columns = dilate(refine(coarse, HX, HY, 0), HY.neighbours[0], HY.number_of_cells(0))
            c = cost.c(HX.rep_points(0)[rows], HY.rep_points(0)[columns])
            keys = rows.astype(np.int64) * len(nu) + columns
            run["N1"] = len(c)
            for variant in ORDER[len(run["solves"]):]:
                mains = on_mains()
                began = perf_counter()
                start = START[variant](coarse, HX, HY, 0) if variant in START else None
                start_time = perf_counter() - began
                began = perf_counter()
                pi = Cplex(warm_start=False).solve_local(rows, columns, c, mu, nu, None, start)[0]
                solve_time = perf_counter() - began
                C = float(pi.data @ c[np.searchsorted(keys, pi.row.astype(np.int64) * len(nu) + pi.col)])
                solve = {"variant": NAMES[variant], "mains": mains, "start time": start_time,
                         "solve time": solve_time, "C": C}
                if start is not None:
                    assert np.array_equal(np.bincount(start.row, start.data, len(mu)), mu)
                    assert np.array_equal(np.bincount(start.col, start.data, len(nu)), nu)
                    start_keys = start.row.astype(np.int64) * len(nu) + start.col
                    solve["start C"] = float(start.data @ c[np.searchsorted(keys, start_keys)])
                    solve["start support"] = len(start.data)
                run["solves"].append(solve)
                stored[name] = run
                path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
                gap = f", start gap {(solve['start C'] - C) / C:.1e}" if start is not None else ""
                print(f"  {NAMES[variant]}: start {start_time:.2f} s, solve {solve_time:.1f} s{gap}, mains {mains}",
                      flush=True)


if __name__ == "__main__":
    main()

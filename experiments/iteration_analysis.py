"""Why does solve_sparse need 20-27 iterations per level? Per-iteration analysis of the finest level.

python experiments/iteration_analysis.py n kind   (kind: grid-disc | uniform | clustered)
"""
import sys
from pathlib import Path

import numpy as np

import msot.multiscale as multiscale
from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.problems import make_problem
from msot.shielding import extract_map

RESULTS = Path(__file__).parent / "results"


def solver():
    try:
        from msot.solvers.cplex import Cplex
        return Cplex()
    except ImportError:
        from msot.solvers.highs import Highs
        return Highs()


def main():
    n, kind = int(sys.argv[1]), sys.argv[2]
    X, mu, Y, nu = make_problem(n, "grid" if kind == "grid-disc" else kind, np.random.default_rng(0),
                                mask="disc" if kind == "grid-disc" else None)
    HX, HY = build_pair(X, mu, Y, nu)
    records = []
    original_shield = multiscale.shield

    def recording_shield(pi, S, Xk, Yh, k, cost, tolerance=1e-12):
        rows, columns = original_shield(pi, S, Xk, Yh, k, cost, tolerance)
        records.append({"k": k, "pi": pi.tocoo(), "t": extract_map(pi), "N": rows.astype(np.int64) * len(Yh.mass[k]) + columns,
                        "C": float(pi.data @ cost.c(Xk[pi.row], Yh.rep_points(k)[pi.col]))})
        return rows, columns

    multiscale.shield = recording_shield
    multiscale.solve_multiscale(HX, HY, SquaredEuclidean(), solver())
    multiscale.shield = original_shield

    lines = [f"== n = {n}, {kind}, mesh hierarchy, 1-ring",
             "per iteration at level 0: relative gap to the final cost, |N| per point, entries added / dropped vs. previous N "
             "(per point), points whose t changed, points whose support changed, mass moved (fraction of total)"]
    level = [r for r in records if r["k"] == 0]
    final = level[-1]["C"]
    for i, r in enumerate(level):
        line = f"{i + 1:3d}  gap {(r['C'] - final) / final:9.2e}  |N|/n {len(r['N']) / n:6.2f}"
        if i:
            previous = level[i - 1]
            added = np.setdiff1d(r["N"], previous["N"], assume_unique=True).size
            dropped = np.setdiff1d(previous["N"], r["N"], assume_unique=True).size
            changed_t = np.count_nonzero(r["t"] != previous["t"])
            support = lambda p: set(zip(p.row[p.data > 0].tolist(), p.col[p.data > 0].tolist()))
            now, before = support(r["pi"]), support(previous["pi"])
            changed_rows = len({x for x, _ in now ^ before})
            moved = abs(r["pi"].tocsr() - previous["pi"].tocsr()).sum() / 2 / mu.sum()
            line += (f"  added {added / n:5.2f}  dropped {dropped / n:5.2f}  t changed {changed_t:6d}"
                     f"  support changed {changed_rows:6d}  mass moved {moved:8.2e}")
        lines.append(line)
        print(line, flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"iteration_analysis_{n}_{kind}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

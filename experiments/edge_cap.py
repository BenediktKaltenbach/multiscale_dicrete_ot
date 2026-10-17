"""Edge length cap of the mesh hierarchy: length_cap in {None, 2, 3, 4}, hierarchy quality and solve_multiscale."""
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.multiscale import solve_multiscale
from msot.problems import gaussian_masses, make_problem, sample_points
from msot.solvers.cplex import Cplex

KINDS = ("uniform", "clustered", "anisotropic")
LENGTH_CAPS = (None, 2.0, 3.0, 4.0)
N = 4096
N_DENSE = 256
SEED = 0
RESULTS = Path(__file__).parent / "results"


def problem(kind, rng):
    if kind != "anisotropic":
        return make_problem(N, kind, rng)
    X, Y = sample_points(N, "uniform", rng) * [1.0, 0.1], sample_points(N, "uniform", rng) * [1.0, 0.1]
    mu, nu = gaussian_masses(X, rng), gaussian_masses(Y, rng)
    mu /= mu.sum()
    nu /= nu.sum()
    for _ in range(10):
        nu[np.argmax(nu)] += mu.sum() - nu.sum()
    return X, mu, Y, nu


def compactness(H, k):
    """rad / distance from rep to the nearest other rep at level k."""
    reps = H.rep_points(k)
    return H.rad[k] / cKDTree(reps).query(reps, 2)[0][:, 1]


def coarsening_levels(H):
    """Levels k >= 1 made by a coarsening step (not the top level K = {X})."""
    return range(1, H.K)


def children_histogram(H, k):
    return np.bincount(np.bincount(H.parent[k - 1]), minlength=6)[1:6]


def copy_fraction(H):
    copies = sum(children_histogram(H, k)[0] for k in coarsening_levels(H))
    return copies / sum(H.number_of_cells(k) for k in coarsening_levels(H))


def hierarchy_table(H):
    lines = ["  k  cells   children 1/2/3/4/5        copy %   compactness median    q90     max  cells > 2"]
    for k in coarsening_levels(H):
        histogram, ratio = children_histogram(H, k), compactness(H, k)
        lines.append(f"{k:3d} {H.number_of_cells(k):6d}   " + " ".join(f"{c:4d}" for c in histogram)
                     + f"   {100 * histogram[0] / H.number_of_cells(k):6.1f}   {np.median(ratio):18.2f} "
                       f"{np.quantile(ratio, 0.9):6.2f} {ratio.max():7.2f} {np.sum(ratio > 2):10d}")
    return "\n".join(lines)


def solve_table(info):
    lines = ["  k  cells X/Y      iterations  final |N|  final |N| / cells X  time solve  time shield"]
    for level in info:
        history = level["history"]
        cells_x, cells_y = level["cells"]
        lines.append(f"{level['k']:3d} {cells_x:6d}/{cells_y:<6d} {len(history):11d} {history[-1]['N']:10d} "
                     f"{history[-1]['N'] / cells_x:20.2f} {sum(step['time_solve'] for step in history):11.2f} "
                     f"{sum(step['time_shield'] for step in history):12.2f}")
    return "\n".join(lines)


def run(X, mu, Y, nu, length_cap):
    HX, HY = build_pair(X, mu, Y, nu, n_dense=N_DENSE, length_cap=length_cap)
    pi, _, _, info = solve_multiscale(HX, HY, SquaredEuclidean(), Cplex())
    steps = [step for level in info for step in level["history"]]
    return HX, HY, info, {
        "K": HX.K,
        "copy % X": 100 * copy_fraction(HX),
        "max compactness X k=1": compactness(HX, 1).max(),
        "max compactness X k=2": compactness(HX, 2).max() if HX.K > 2 else np.nan,
        "iterations": len(steps),
        "iterations k=0": len(info[-1]["history"]),
        "final |N| / n": info[-1]["history"][-1]["N"] / len(X),
        "time solve": sum(step["time_solve"] for step in steps),
        "time shield": sum(step["time_shield"] for step in steps),
        "C": float(pi.data @ SquaredEuclidean().c(X[pi.row], Y[pi.col])),
    }


def main():
    start = perf_counter()
    RESULTS.mkdir(exist_ok=True)
    header = [f"Edge length cap: length_cap in {LENGTH_CAPS}, n = {N}, n_dense = {N_DENSE}, seed {SEED} per kind, "
              f"solve_multiscale with SquaredEuclidean and Cplex",
              "copy % = fraction of cells with a single child over the coarsening levels 1..K-1; "
              "compactness = rad / distance from rep to the nearest other rep at level k",
              "relative difference = |C - C(length_cap None)| / C(length_cap None)", ""]
    columns = ["K", "copy % X", "max compactness X k=1", "max compactness X k=2", "iterations", "iterations k=0",
               "final |N| / n", "time solve", "time shield"]
    summary, details = [], []
    for kind in KINDS:
        X, mu, Y, nu = problem(kind, np.random.default_rng(SEED))
        summary += [f"== {kind}", f"{'length_cap':>10s}" + "".join(f"{name:>{len(name) + 2}s}" for name in columns)
                    + f"{'C':>20s}{'relative difference':>21s}"]
        C_reference = None
        for length_cap in LENGTH_CAPS:
            HX, HY, info, result = run(X, mu, Y, nu, length_cap)
            C_reference = result["C"] if C_reference is None else C_reference
            difference = abs(result["C"] - C_reference) / C_reference
            summary.append(f"{str(length_cap):>10s}" + "".join(f"{result[name]:{len(name) + 2}.3g}" for name in columns)
                           + f"{result['C']:20.12g}{difference:21.1e}")
            details += [f"== {kind}, length_cap {length_cap}, K = {HX.K}",
                        "-- X hierarchy", hierarchy_table(HX), "-- Y hierarchy", hierarchy_table(HY),
                        "-- solve_multiscale", solve_table(info), ""]
            print(summary[-1], flush=True)
        summary.append("")
        text = "\n".join(header + summary + details + [f"runtime so far {perf_counter() - start:.0f} s"])
        (RESULTS / "edge_cap.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

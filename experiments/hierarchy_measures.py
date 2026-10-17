"""Mesh hierarchy (build_pair) vs. quadtree (build_pair_quadtree): compactness of the cells and size of N1."""
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair, build_pair_quadtree
from msot.multiscale import solve_multiscale
from msot.problems import gaussian_masses, make_problem, sample_points
from msot.solvers.base import solve_dense
from msot.solvers.cplex import Cplex

KINDS = ("uniform", "clustered", "anisotropic")
N_DENSE = 256
HIERARCHIES = {"mesh": lambda X, mu, Y, nu: build_pair(X, mu, Y, nu, n_dense=N_DENSE),
               "quadtree": build_pair_quadtree}
COLOURS = {"mesh": "#2a78d6", "quadtree": "#eb6834"}
RESULTS = Path(__file__).parent / "results"


def sample(n, kind, rng):
    if kind == "anisotropic":
        return sample_points(n, "uniform", rng) * [1.0, 0.1]
    return sample_points(n, kind, rng)


def problem(n, kind, rng):
    if kind != "anisotropic":
        return make_problem(n, kind, rng)
    X, Y = sample(n, kind, rng), sample(n, kind, rng)
    mu, nu = gaussian_masses(X, rng), gaussian_masses(Y, rng)
    mu /= mu.sum()
    nu /= nu.sum()
    for _ in range(10):
        nu[np.argmax(nu)] += mu.sum() - nu.sum()
    return X, mu, Y, nu


def compactness(H):
    """rad / distance from rep to the nearest other rep, per cell, for each level k >= 1 with more than one cell."""
    ratios = {}
    for k in range(1, H.K + 1):
        if H.number_of_cells(k) > 1:
            points = H.rep_points(k)
            nearest_rep_distance = cKDTree(points).query(points, 2)[0][:, 1]
            ratios[k] = H.rad[k] / nearest_rep_distance
    return ratios


def compactness_table(ratios, H):
    lines = ["  k  cells  median    q90      max  cells > 2"]
    for k, ratio in ratios.items():
        lines.append(f"{k:3d} {H.number_of_cells(k):6d}  {np.median(ratio):6.2f} {np.quantile(ratio, 0.9):6.2f} "
                     f"{ratio.max():8.2f} {np.sum(ratio > 2):10d}")
    return "\n".join(lines)


def neighbourhood_table(info):
    lines = ["  k  cells X/Y      |N1|  |N1|/cells X  |N| last solve  |N|/cells X  iterations"]
    for level in info:
        cells_x, cells_y = level["cells"]
        N = level["history"][-1]["N"]
        lines.append(f"{level['k']:3d} {cells_x:6d}/{cells_y:<6d} {level['N1']:8d} {level['N1'] / cells_x:12.2f} "
                     f"{N:15d} {N / cells_x:11.2f} {len(level['history']):11d}")
    return "\n".join(lines)


def plot_compactness(kind, ratios_by_hierarchy, n):
    figure, axes = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    for offset, (name, ratios) in zip((-width / 2, width / 2), ratios_by_hierarchy.items()):
        levels = list(ratios)
        boxes = axes.boxplot([ratios[k] for k in levels], positions=np.array(levels) + offset, widths=width * 0.85,
                             patch_artist=True, flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})
        for box in boxes["boxes"]:
            box.set(facecolor=COLOURS[name], alpha=0.6, edgecolor=COLOURS[name])
        for element in ("whiskers", "caps", "medians"):
            for line in boxes[element]:
                line.set_color("#333333")
        axes.plot([], [], "s", color=COLOURS[name], alpha=0.6, label=name)
    levels = sorted(set().union(*ratios_by_hierarchy.values()))
    axes.set_xticks(levels, [str(k) for k in levels])
    axes.axhline(2, color="#999999", linestyle="--", linewidth=1)
    axes.set_xlabel("level k")
    axes.set_ylabel("rad / distance to nearest other rep")
    axes.set_title(f"Compactness per cell, {kind}, n = {n}, mesh n_dense = {N_DENSE}")
    axes.grid(axis="y", color="#e5e5e5")
    axes.set_axisbelow(True)
    axes.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(RESULTS / f"compactness_{kind}.png", dpi=150)
    plt.close(figure)


def main():
    start = perf_counter()
    RESULTS.mkdir(exist_ok=True)
    n = 4096
    rng = np.random.default_rng(0)
    report = [f"Measure 1: compactness rad / (distance from rep to nearest other rep at level k), n = {n}, "
              f"mesh n_dense = {N_DENSE}, unit masses, seed 0", ""]
    for kind in KINDS:
        points, mass = sample(n, kind, rng), np.ones(n)
        ratios_by_hierarchy = {}
        for name, build in HIERARCHIES.items():
            H = build(points, mass, points, mass)[0]
            ratios_by_hierarchy[name] = compactness(H)
            report += [f"== {kind}, {name}, K = {H.K}", compactness_table(ratios_by_hierarchy[name], H), ""]
        plot_compactness(kind, ratios_by_hierarchy, n)

    n = 1000
    cost, solver = SquaredEuclidean(), Cplex()
    report += [f"Measure 2: N1 = refine output and |N| in solve_multiscale, n = {n}, mesh n_dense = {N_DENSE}, seed 1",
               "|N| last solve = size of the N solved on in the last solve_sparse iteration of that level "
               "(the final shielding N is not returned)", ""]
    rng = np.random.default_rng(1)
    for kind in KINDS:
        X, mu, Y, nu = problem(n, kind, rng)
        time = perf_counter()
        pi, *_ = solve_dense(X, Y, mu, nu, cost, Cplex(warm_start=False))
        C_dense = float(pi.data @ cost.c(X[pi.row], Y[pi.col]))
        time_dense = perf_counter() - time
        for name, build in HIERARCHIES.items():
            HX, HY = build(X, mu, Y, nu)
            time = perf_counter()
            try:
                pi, _, _, info = solve_multiscale(HX, HY, cost, solver)
            except RuntimeError as error:
                report += [f"== {kind}, {name}, K = {HX.K}: FAILED, RuntimeError: {error}", ""]
                continue
            time_multiscale = perf_counter() - time
            C = float(pi.data @ cost.c(X[pi.row], Y[pi.col]))
            difference = abs(C - C_dense) / abs(C_dense)
            report += [f"== {kind}, {name}, K = {HX.K}: C = {C:.10g}, dense {C_dense:.10g}, "
                       f"relative difference {difference:.1e} ({'match' if difference < 1e-7 else 'MISMATCH'}), "
                       f"time {time_multiscale:.1f} s (dense {time_dense:.1f} s)",
                       neighbourhood_table(info), ""]

    report.append(f"total runtime {perf_counter() - start:.0f} s")
    text = "\n".join(report)
    print(text)
    (RESULTS / "hierarchy_measures.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()

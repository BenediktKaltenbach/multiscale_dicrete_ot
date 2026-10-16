"""Step 6: shielding candidate sets S. Sector neighbours (6, 8) with and without pruning vs. the Delaunay 1-ring."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.mesh import candidate_statistics, sector_neighbours
from msot.multiscale import solve_multiscale
from msot.problems import gaussian_masses, make_problem, sample_points
from msot.shielding import extract_map, search_tree
from msot.solvers.highs import Highs

RESULTS = Path(__file__).parent / "results"
TOLERANCE = 1e-12
VARIANTS = {
    "1-ring, pruned": (None, True),
    "6 sectors, pruned": (6, True),
    "8 sectors, pruned": (8, True),
    "8 sectors, unpruned": (8, False),
}


def anisotropic_problem(n, rng):
    X = sample_points(n, "uniform", rng) * [1.0, 0.1]
    Y = sample_points(n, "uniform", rng) * [1.0, 0.1]
    mu, nu = gaussian_masses(X, rng), gaussian_masses(Y, rng)
    mu /= mu.sum()
    nu /= nu.sum()
    for _ in range(10):
        nu[np.argmax(nu)] += mu.sum() - nu.sum()
    return X, mu, Y, nu


def problems(n):
    yield "uniform", make_problem(n, "uniform", np.random.default_rng(0))
    yield "clustered", make_problem(n, "clustered", np.random.default_rng(0))
    yield "anisotropic", anisotropic_problem(n, np.random.default_rng(0))
    yield "grid, disc mask", make_problem(n, "grid", np.random.default_rng(0), mask="disc")


def per_vertex(X, Y, pi, S, HY, cost):
    """|Y_hat|, number of needed candidates and the Lipschitz estimate L of t per x_A at the final coupling."""
    t = extract_map(pi)
    index_pointer, indices = S
    size_Y_hat, needed, lipschitz = (np.zeros(len(X)) for _ in range(3))
    for a in range(len(X)):
        candidates = indices[index_pointer[a]:index_pointer[a + 1]]
        if cost.prune:
            candidates = candidates[cost.needed_candidates(X[a], X[candidates], Y[t[candidates]], TOLERANCE)]
        size_Y_hat[a] = len(search_tree(X[a], X[candidates], Y[t[candidates]], HY, 0, cost, TOLERANCE))
        needed[a] = len(candidates)
        all_candidates = indices[index_pointer[a]:index_pointer[a + 1]]
        lipschitz[a] = (np.linalg.norm(Y[t[all_candidates]] - Y[t[a]], axis=1)
                        / np.linalg.norm(X[all_candidates] - X[a], axis=1)).max()
    return size_Y_hat, needed, lipschitz


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    RESULTS.mkdir(exist_ok=True)
    report = [f"n = {n}, mesh hierarchy (build_pair defaults), spacing = distance to the nearest other point of X",
              "S statistics at level 0; |Y_hat|, needed candidates and L at the final coupling; times summed over all levels", ""]
    columns = ["|S|", "needed", "q median", "q 5%", "q<=0 %", "D/sp med", "D/sp 99%", "D/sp max",
               "|Y_hat| mean", "|Y_hat| 99%", "it. k=0", "it. total", "|N|/n", "solve s", "shield s", "rel. cost"]
    for kind, (X, mu, Y, nu) in problems(n):
        HX, HY = build_pair(X, mu, Y, nu)
        spacing = cKDTree(X).query(X, 2)[0][:, 1]
        report += [f"== {kind}", f"{'':22s}" + "".join(f"{c:>13s}" for c in columns)]
        figure, axes = plt.subplots(1, len(VARIANTS), figsize=(4 * len(VARIANTS), 4), sharex=True, sharey=True)
        reference_cost = None
        for axis, (name, (sectors, prune)) in zip(axes, VARIANTS.items()):
            candidates = None if sectors is None else lambda H, k: sector_neighbours(H.rep_points(k), sectors)
            cost = SquaredEuclidean(prune=prune)
            pi, _, _, info = solve_multiscale(HX, HY, cost, Highs(), candidates=candidates)
            C = pi.data @ cost.c(X[pi.row], Y[pi.col])
            reference_cost = reference_cost or C
            S = candidates(HX, 0) if candidates else HX.neighbours[0]
            q, D = candidate_statistics(X, S)
            size_Y_hat, needed, lipschitz = per_vertex(X, Y, pi, S, HY, cost)
            histories = [level["history"] for level in info]
            values = [np.diff(S[0]).mean(), needed.mean(), np.median(q), np.quantile(q, 0.05), 100 * np.mean(q <= 0),
                      np.median(D / spacing), np.quantile(D / spacing, 0.99), (D / spacing).max(),
                      size_Y_hat.mean(), np.quantile(size_Y_hat, 0.99), len(histories[-1]),
                      sum(len(history) for history in histories), histories[-1][-1]["N"] / n,
                      sum(step["time_solve"] for history in histories for step in history),
                      sum(step["time_shield"] for history in histories for step in history),
                      abs(C - reference_cost) / reference_cost]
            report.append(f"{name:22s}" + "".join(f"{v:13.3g}" for v in values))
            print(report[-1], flush=True)
            bounded = q > 0
            radius = lipschitz[bounded] * D[bounded] / q[bounded]
            axis.loglog((radius / spacing[bounded]) ** 2, np.maximum(size_Y_hat[bounded], 0.5), ".", markersize=2, alpha=0.4)
            axis.set_title(name)
            axis.set_xlabel("(L D / q)^2 in spacings^2 (Prop. 5.3)")
        axes[0].set_ylabel("|Y_hat| per x_A (0 shown at 0.5)")
        figure.suptitle(f"{kind}, n = {n}: |Y_hat| against the radius of Prop. 5.3, interior vertices (q > 0)")
        figure.tight_layout()
        figure.savefig(RESULTS / f"candidate_sets_{kind.split(',')[0]}.png", dpi=150)
        plt.close(figure)
        report.append("")
    (RESULTS / "candidate_sets.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()

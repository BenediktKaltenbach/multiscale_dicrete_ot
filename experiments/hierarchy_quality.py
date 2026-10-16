"""Quality measures of the mesh hierarchy (docs/hierarchy.md) and cost of search_tree vs. brute force."""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from msot.cost import SquaredEuclidean
from msot.hierarchy import agglomerate, build_pair
from msot.multiscale import solve_multiscale
from msot.problems import make_problem, sample_points
from msot.shielding import shield
from msot.solvers.highs import Highs


class CountingSquaredEuclidean(SquaredEuclidean):
    evaluations = 0

    def psi_hat(self, xA, xs, rep_y, rad_y):
        values = super().psi_hat(xA, xs, rep_y, rad_y)
        self.evaluations += values.size
        return values


def level_table(H):
    lines = ["  k  cells  children 1/2/3/4/>4 (%)        rad ratio  rad / nearest neighbour distance"]
    for k in range(1, H.K + 1):
        number_of_children = np.bincount(H.parent[k - 1])
        fractions = [np.mean(number_of_children == c) for c in (1, 2, 3, 4)] + [np.mean(number_of_children > 4)]
        rad_below = H.rad[k - 1]
        positive = rad_below > 0
        ratio = np.median(H.rad[k][H.parent[k - 1]][positive] / rad_below[positive]) if positive.any() else np.nan
        points = H.rep_points(k)
        nearest_neighbour_distance = cKDTree(points).query(points, 2)[0][:, 1].mean() if len(points) > 1 else np.nan
        lines.append(f"{k:3d} {H.number_of_cells(k):6d}  " + " ".join(f"{100 * f:5.1f}" for f in fractions)
                     + f"   {ratio:9.2f}  {np.median(H.rad[k]) / nearest_neighbour_distance:6.2f}")
    return "\n".join(lines)


def main():
    rng = np.random.default_rng(0)
    n = 4096
    samples = {
        "uniform": sample_points(n, "uniform", rng),
        "clustered": sample_points(n, "clustered", rng),
        "anisotropic": sample_points(n, "uniform", rng) * [1.0, 0.1],
    }
    report = []
    for name, points in samples.items():
        report += [f"== {name}, n = {n}", level_table(agglomerate(points, np.ones(n), n_dense=16)), ""]

    n = 2000
    X, mu, Y, nu = make_problem(n, "uniform", np.random.default_rng(1))
    HX, HY = build_pair(X, mu, Y, nu, n_dense=16)
    S = HX.neighbours[0]
    pi = solve_multiscale(HX, HY, SquaredEuclidean(), Highs())[0]
    cost = CountingSquaredEuclidean()
    rows, _ = shield(pi, S, X, HY, 0, cost)
    brute_force = np.diff(S[0]).sum() * n
    report += [f"== search_tree on optimal coupling, uniform, n = {n}, K = {HY.K}",
               f"psi_hat evaluations: tree {cost.evaluations}, brute force {brute_force}, "
               f"ratio {brute_force / cost.evaluations:.1f}",
               f"|N| / n = {len(rows) / n:.1f}"]

    text = "\n".join(report)
    print(text)
    results = Path(__file__).parent / "results"
    results.mkdir(exist_ok=True)
    (results / "hierarchy_quality.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()

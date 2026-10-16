"""Quality measures of the mesh hierarchy (docs/hierarchy.md) and cost of search_tree vs. brute force."""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from msot.cost import SqEuclid
from msot.hierarchy import agglomerate, build_pair
from msot.multiscale import north_west_corner, solve_sparse
from msot.problems import make_problem, sample_points
from msot.shielding import shield
from msot.solvers.scipy_highs import ScipyHighs


class CountingSqEuclid(SqEuclid):
    evaluations = 0

    def psi_hat(self, xA, xs, rep_y, rad_y):
        out = super().psi_hat(xA, xs, rep_y, rad_y)
        self.evaluations += out.size
        return out


def level_table(H):
    lines = ["  k  cells  children 1/2/3/4/>4 (%)        rad ratio  rad/nn"]
    for k in range(1, H.K + 1):
        n_child = np.bincount(H.parent[k - 1])
        hist = [np.mean(n_child == c) for c in (1, 2, 3, 4)] + [np.mean(n_child > 4)]
        prev = H.rad[k - 1]
        ratio = np.median(H.rad[k][H.parent[k - 1]][prev > 0] / prev[prev > 0]) if (prev > 0).any() else np.nan
        P = H.rep_points(k)
        nn = cKDTree(P).query(P, 2)[0][:, 1].mean() if len(P) > 1 else np.nan
        lines.append(f"{k:3d} {H.n_cells(k):6d}  " + " ".join(f"{100 * h:5.1f}" for h in hist)
                     + f"   {ratio:9.2f}  {np.median(H.rad[k]) / nn:6.2f}")
    return "\n".join(lines)


def main():
    rng = np.random.default_rng(0)
    n = 4096
    samples = {
        "uniform": sample_points(n, "uniform", rng),
        "clustered": sample_points(n, "clustered", rng),
        "anisotropic": sample_points(n, "uniform", rng) * [1.0, 0.1],
    }
    out = []
    for name, P in samples.items():
        out += [f"== {name}, n = {n}", level_table(agglomerate(P, np.ones(n), n_dense=16)), ""]

    n = 2000
    X, mu, Y, nu = make_problem(n, "uniform", np.random.default_rng(1))
    HX, HY = build_pair(X, mu, Y, nu, n_dense=16)
    S = HX.nbrs[0]
    N1 = north_west_corner(mu, nu, np.argsort(X[:, 0]), np.argsort(Y[:, 0]))
    pi = solve_sparse(X, mu, HY, 0, SqEuclid(), S, N1, ScipyHighs())[0]
    cost = CountingSqEuclid()
    rows, _ = shield(pi, S, X, HY, 0, cost)
    brute = np.diff(S[0]).sum() * n
    out += [f"== search_tree on optimal coupling, uniform, n = {n}, K = {HY.K}",
            f"psi_hat evaluations: tree {cost.evaluations}, brute force {brute}, ratio {brute / cost.evaluations:.1f}",
            f"|N| / n = {len(rows) / n:.1f}"]

    text = "\n".join(out)
    print(text)
    results = Path(__file__).parent / "results"
    results.mkdir(exist_ok=True)
    (results / "hierarchy_quality.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()

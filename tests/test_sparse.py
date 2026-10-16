import unittest

import numpy as np

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.mesh import one_ring, sector_neighbours
from msot.multiscale import solve_sparse
from msot.problems import make_problem
from msot.shielding import extract_map, missed_brute, search_tree
from msot.solvers.highs import Highs
from msot.solvers.scipy_highs import solve_dense
from msot.verify import dual_violation, is_shielding


def north_west_corner(mu, nu, order_x, order_y):
    """Feasible initial N1 with at most |X| + |Y| - 1 entries: north-west corner rule along the given orderings."""
    cumulative_x, cumulative_y = np.cumsum(mu[order_x]), np.cumsum(nu[order_y])
    breaks = np.unique(np.concatenate([cumulative_x, cumulative_y]))[:-1]
    i = np.minimum(np.searchsorted(cumulative_x, breaks, side="right"), len(mu) - 1)
    j = np.minimum(np.searchsorted(cumulative_y, breaks, side="right"), len(nu) - 1)
    rows = np.concatenate([[order_x[0]], order_x[i]])
    columns = np.concatenate([[order_y[0]], order_y[j]])
    return rows, columns


class TestSearchTree(unittest.TestCase):
    def test_matches_brute_force(self):
        cost, solver = SquaredEuclidean(), Highs()
        for kind in ["uniform", "clustered", "grid"]:
            X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0))
            HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
            for k in [0, 1]:
                Xk, Yk = HX.rep_points(k), HY.rep_points(k)
                N1 = north_west_corner(HX.mass[k], HY.mass[k], np.argsort(Xk[:, 0]), np.argsort(Yk[:, 1]))
                pi = solver.solve_local(*N1, cost.c(Xk[N1[0]], Yk[N1[1]]), HX.mass[k], HY.mass[k])[0]
                t = extract_map(pi)
                index_pointer, indices = HX.neighbours[k]
                for a in range(len(Xk)):
                    candidates = indices[index_pointer[a]:index_pointer[a + 1]]
                    triple = Xk[a], Xk[candidates], Yk[t[candidates]]
                    np.testing.assert_array_equal(np.sort(search_tree(*triple, HY, k, cost, 1e-12)),
                                                  missed_brute(*triple, Yk, cost, 1e-12))


class TestPruning(unittest.TestCase):
    def test_pruning_keeps_Y_hat(self):
        cost, solver = SquaredEuclidean(prune=True), Highs()
        for kind in ["uniform", "clustered", "grid"]:
            X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0))
            _, HY = build_pair(X, mu, Y, nu, n_dense=8)
            N1 = north_west_corner(mu, nu, np.argsort(X[:, 0]), np.argsort(Y[:, 1]))
            t = extract_map(solver.solve_local(*N1, cost.c(X[N1[0]], Y[N1[1]]), mu, nu)[0])
            for S in (one_ring(X), sector_neighbours(X, 8)):
                index_pointer, indices = S
                for a in range(len(X)):
                    candidates = indices[index_pointer[a]:index_pointer[a + 1]]
                    needed = candidates[cost.needed_candidates(X[a], X[candidates], Y[t[candidates]], 1e-12)]
                    np.testing.assert_array_equal(
                        np.sort(search_tree(X[a], X[needed], Y[t[needed]], HY, 0, cost, 1e-12)),
                        np.sort(search_tree(X[a], X[candidates], Y[t[candidates]], HY, 0, cost, 1e-12)))


class TestSolveSparse(unittest.TestCase):
    def check(self, n, kind, seed, mask=None):
        X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed), mask=mask)
        cost, solver = SquaredEuclidean(), Highs()
        _, HY = build_pair(X, mu, Y, nu, n_dense=8)
        pi_dense, _, _, _ = solve_dense(X, Y, mu, nu, cost)
        C_dense = pi_dense.data @ cost.c(X[pi_dense.row], Y[pi_dense.col])

        N1 = north_west_corner(mu, nu, np.argsort(X[:, 0]), np.argsort(Y[:, 0]))
        pi, alpha, beta, N, history = solve_sparse(X, mu, HY, 0, cost, one_ring(X), N1, solver)
        C = pi.data @ cost.c(X[pi.row], Y[pi.col])

        self.assertAlmostEqual(C, C_dense, delta=1e-9 * C_dense)
        self.assertLess(dual_violation(alpha, beta, X, Y, cost), 1e-9)
        self.assertTrue(is_shielding(pi, *N, X, Y, cost))
        self.assertLess(len(N[0]), len(X) * len(Y))
        return history

    def test_uniform(self):
        for seed in range(3):
            self.check(100, "uniform", seed)

    def test_clustered(self):
        self.check(100, "clustered", 0)

    def test_grid_with_mask(self):
        self.check(144, "grid", 0, mask="disc")


if __name__ == "__main__":
    unittest.main()

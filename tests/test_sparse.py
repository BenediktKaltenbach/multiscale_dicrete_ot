import unittest

import numpy as np

from msot.cost import SqEuclid
from msot.hierarchy import build_pair
from msot.mesh import one_ring
from msot.multiscale import north_west_corner, solve_sparse
from msot.problems import make_problem
from msot.shielding import extract_map, missed_brute, search_tree
from msot.solvers.scipy_highs import ScipyHighs, solve_dense
from msot.verify import dual_violation, is_shielding


class TestSearchTree(unittest.TestCase):
    def test_matches_brute_force(self):
        cost, solver = SqEuclid(), ScipyHighs()
        for kind in ["uniform", "clustered", "grid"]:
            X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0))
            HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
            for k in [0, 1]:
                Xk, Yk = HX.rep_points(k), HY.rep_points(k)
                N1 = north_west_corner(HX.mass[k], HY.mass[k], np.argsort(Xk[:, 0]), np.argsort(Yk[:, 1]))
                pi = solver.solve_local(*N1, cost.c(Xk[N1[0]], Yk[N1[1]]), HX.mass[k], HY.mass[k])[0]
                t = extract_map(pi)
                indptr, indices = HX.nbrs[k]
                for a in range(len(Xk)):
                    s = indices[indptr[a]:indptr[a + 1]]
                    args = Xk[a], Xk[s], Yk[t[s]]
                    np.testing.assert_array_equal(np.sort(search_tree(*args, HY, k, cost, 1e-12)),
                                                  missed_brute(*args, Yk, cost, 1e-12))


class TestSolveSparse(unittest.TestCase):
    def check(self, n, kind, seed, mask=None):
        X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed), mask=mask)
        cost, solver = SqEuclid(), ScipyHighs()
        _, HY = build_pair(X, mu, Y, nu, n_dense=8)
        pi_d, _, _, _ = solve_dense(X, Y, mu, nu, cost)
        C_dense = pi_d.data @ cost.c(X[pi_d.row], Y[pi_d.col])

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

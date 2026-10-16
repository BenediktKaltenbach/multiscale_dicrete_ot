import unittest

import numpy as np

from msot.cost import SqEuclid
from msot.hierarchy import build_pair
from msot.multiscale import solve_multiscale
from msot.problems import make_problem
from msot.shielding import shield
from msot.solvers.scipy_highs import ScipyHighs, solve_dense
from msot.verify import dual_violation, is_shielding


class TestSolveMultiscale(unittest.TestCase):
    def check(self, X, mu, Y, nu, n_dense=8):
        cost, solver = SqEuclid(), ScipyHighs()
        HX, HY = build_pair(X, mu, Y, nu, n_dense)
        pi, alpha, beta, info = solve_multiscale(HX, HY, cost, solver)
        pi_d = solve_dense(X, Y, mu, nu, cost)[0]
        C = pi.data @ cost.c(X[pi.row], Y[pi.col])
        C_dense = pi_d.data @ cost.c(X[pi_d.row], Y[pi_d.col])

        self.assertEqual(len(info), HX.K)
        self.assertAlmostEqual(C, C_dense, delta=1e-9 * C_dense)
        self.assertLess(dual_violation(alpha, beta, X, Y, cost), 1e-9)
        self.assertTrue(is_shielding(pi, *shield(pi, HX.nbrs[0], X, HY, 0, cost), X, Y, cost))
        return info

    def test_kinds_and_masks(self):
        for kind in ["uniform", "clustered", "grid"]:
            for mask in [None, "halfplane"]:
                with self.subTest(kind=kind, mask=mask):
                    self.check(*make_problem(400, kind, np.random.default_rng(0), mask=mask))

    def test_unequal_sizes(self):
        rng = np.random.default_rng(1)
        X, Y = rng.random((30, 2)), rng.random((300, 2))
        mu, nu = np.full(30, 1 / 30), np.full(300, 1 / 300)
        self.check(X, mu, Y, nu * mu.sum() / nu.sum(), n_dense=4)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.multiscale import solve_multiscale
from msot.problems import make_problem
from msot.solvers.highs import Highs
from msot.verify import dual_violation

try:
    from msot.solvers.cplex import Cplex
except ImportError:
    Cplex = None


@unittest.skipIf(Cplex is None, "cplex not installed")
class TestCplex(unittest.TestCase):
    def test_matches_highs(self):
        cost = SquaredEuclidean()
        for kind, mask, atoms in [("uniform", None, 10**9), ("clustered", "disc", 10**9), ("grid", "disc", 10**9),
                                  ("uniform", None, None)]:
            with self.subTest(kind=kind, mask=mask, atoms=atoms):
                X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0), mask=mask, atoms=atoms)
                HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
                costs = []
                for solver in (Highs(), Cplex(), Cplex(warm_start=False)):
                    pi, alpha, beta, _ = solve_multiscale(HX, HY, cost, solver)
                    costs.append(pi.data @ cost.c(X[pi.row], Y[pi.col]))
                    self.assertLess(dual_violation(alpha, beta, X, Y, cost), 1e-9)
                self.assertAlmostEqual(costs[1], costs[0], delta=1e-9 * costs[0])
                self.assertAlmostEqual(costs[2], costs[0], delta=1e-9 * costs[0])


if __name__ == "__main__":
    unittest.main()

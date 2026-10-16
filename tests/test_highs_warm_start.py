import unittest

import numpy as np

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.multiscale import solve_multiscale
from msot.problems import make_problem
from msot.solvers.highs import Highs
from msot.verify import dual_violation
from tests.test_sparse import north_west_corner


class RecordingHighs(Highs):
    """Checks after every warm call that the model holds N and spt pi, and that the key mapping matches the model."""

    def __init__(self, test):
        super().__init__(warm_start=True)
        self.test, self.deleted = test, 0

    def solve_local(self, rows, columns, c, mu, nu, state=None):
        previous_keys = np.empty(0) if state is None else state["keys"]
        pi, alpha, beta, state = super().solve_local(rows, columns, c, mu, nu, state)
        size_y, keys = len(nu), state["keys"]
        self.deleted += np.isin(previous_keys, keys, invert=True).sum()
        self.test.assertTrue(np.isin(rows.astype(np.int64) * size_y + columns, keys).all())
        self.test.assertTrue(np.isin(pi.row.astype(np.int64) * size_y + pi.col, keys).all())
        lp = state["highs"].getLp()
        self.test.assertEqual(lp.num_col_, len(keys))
        index = np.asarray(lp.a_matrix_.index_).reshape(-1, 2)
        np.testing.assert_array_equal(index[:, 0], keys // size_y)
        np.testing.assert_array_equal(index[:, 1] - len(mu), keys % size_y)
        return pi, alpha, beta, state


def solve(X, mu, Y, nu, solver):
    HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
    pi, alpha, beta, _ = solve_multiscale(HX, HY, SquaredEuclidean(), solver)
    return pi.data @ SquaredEuclidean().c(X[pi.row], Y[pi.col]), alpha, beta


class TestWarmStart(unittest.TestCase):
    def test_warm_matches_cold(self):
        for kind in ["uniform", "clustered", "grid"]:
            with self.subTest(kind=kind):
                X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0), mask="disc")
                C_cold = solve(X, mu, Y, nu, Highs(warm_start=False))[0]
                warm = Highs()
                C, alpha, beta = solve(X, mu, Y, nu, warm)
                self.assertAlmostEqual(C, C_cold, delta=1e-9 * C_cold)
                self.assertLess(dual_violation(alpha, beta, X, Y, SquaredEuclidean()), 1e-9)

    def test_model_contains_N_and_support(self):
        solver = RecordingHighs(self)
        solve(*make_problem(400, "clustered", np.random.default_rng(0)), solver)
        self.assertGreater(solver.deleted, 0)

    def test_superset_matches_cold(self):
        rng = np.random.default_rng(2)
        X, Y = rng.random((40, 2)), rng.random((50, 2))
        mu, nu = rng.random(40) + 0.5, rng.random(50) + 0.5
        nu *= mu.sum() / nu.sum()
        cost = SquaredEuclidean()
        rows, columns = north_west_corner(mu, nu, np.argsort(X[:, 0]), np.argsort(Y[:, 1]))
        extra = np.unique(rng.integers(0, 40 * 50, 600))
        keys = np.union1d(rows.astype(np.int64) * 50 + columns, extra)
        rows_superset, columns_superset = keys // 50, keys % 50
        c, c_superset = cost.c(X[rows], Y[columns]), cost.c(X[rows_superset], Y[columns_superset])

        state = Highs().solve_local(rows, columns, c, mu, nu)[3]
        warm = Highs().solve_local(rows_superset, columns_superset, c_superset, mu, nu, state)
        cold = Highs(warm_start=False).solve_local(rows_superset, columns_superset, c_superset, mu, nu)
        for pi, alpha, beta, _ in [warm, cold]:
            np.testing.assert_allclose(pi.sum(axis=1).A1, mu, atol=1e-10)
            np.testing.assert_allclose(pi.sum(axis=0).A1, nu, atol=1e-10)
            self.assertLess((alpha[rows_superset] + beta[columns_superset] - c_superset).max(), 1e-9)
        C_warm, C_cold = (pi.data @ cost.c(X[pi.row], Y[pi.col]) for pi, *_ in [warm, cold])
        self.assertAlmostEqual(C_warm, C_cold, delta=1e-9 * C_cold)
        self.assertAlmostEqual(warm[1] @ mu + warm[2] @ nu, C_cold, delta=1e-9 * C_cold)


if __name__ == "__main__":
    unittest.main()

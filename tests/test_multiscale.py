import unittest

import numpy as np
from scipy.sparse import coo_matrix

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair, build_pair_quadtree
from msot.mesh import sector_neighbours
from msot.multiscale import dilate, north_west_start, refine, solve_multiscale
from msot.problems import make_problem
from msot.shielding import shield
from msot.solvers.base import solve_dense
from msot.solvers.cplex import Cplex
from msot.verify import dual_violation, is_shielding


class TestSolveMultiscale(unittest.TestCase):
    def check(self, X, mu, Y, nu, n_dense=8, quadtree=False, candidates=None, dilation=True, north_west="index"):
        cost, solver = SquaredEuclidean(), Cplex()
        HX, HY = build_pair_quadtree(X, mu, Y, nu) if quadtree else build_pair(X, mu, Y, nu, n_dense)
        pi, alpha, beta, info = solve_multiscale(HX, HY, cost, solver, candidates=candidates, dilation=dilation,
                                                north_west=north_west)
        pi_dense = solve_dense(X, Y, mu, nu, cost, Cplex(warm_start=False))[0]
        C = pi.data @ cost.c(X[pi.row], Y[pi.col])
        C_dense = pi_dense.data @ cost.c(X[pi_dense.row], Y[pi_dense.col])

        self.assertEqual(len(info), HX.K)
        self.assertAlmostEqual(C, C_dense, delta=1e-9 * C_dense)
        self.assertLess(dual_violation(alpha, beta, X, Y, cost), 1e-9)
        S = candidates(HX, 0) if candidates else HX.neighbours[0]
        self.assertTrue(is_shielding(pi, *shield(pi, S, X, HY, 0, cost), X, Y, cost))
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

    def test_sector_neighbours(self):
        for kind, sectors in [("uniform", 6), ("clustered", 8)]:
            with self.subTest(kind=kind, sectors=sectors):
                self.check(*make_problem(400, kind, np.random.default_rng(0)),
                           candidates=lambda H, k: sector_neighbours(H.rep_points(k), sectors))

    def test_without_dilation(self):
        self.check(*make_problem(400, "uniform", np.random.default_rng(0)), dilation=False)

    def test_without_north_west(self):
        self.check(*make_problem(400, "clustered", np.random.default_rng(0)), north_west=None)

    def test_transport_order(self):
        self.check(*make_problem(400, "uniform", np.random.default_rng(0)), north_west="transport")

    def test_dilate(self):
        X, mu, Y, nu = make_problem(400, "uniform", np.random.default_rng(1))
        HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
        coarse = coo_matrix((HX.mass[1][:1], ([0], [0])), shape=(HX.number_of_cells(1), HY.number_of_cells(1)))
        rows, columns = refine(coarse, HX, HY, 0)
        dilated = set(zip(*dilate((rows, columns), HY.neighbours[0], HY.number_of_cells(0))))
        index_pointer, indices = HY.neighbours[0]
        expected = set(zip(rows.tolist(), columns.tolist()))
        for x, y in zip(rows.tolist(), columns.tolist()):
            expected |= {(x, int(z)) for z in indices[index_pointer[y]:index_pointer[y + 1]]}
        self.assertEqual({(int(a), int(b)) for a, b in dilated}, expected)

    def test_north_west_start(self):
        cost = SquaredEuclidean()
        for kind in ["uniform", "clustered", "grid"]:
            X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0))
            HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
            coarse = solve_dense(HX.rep_points(1), HY.rep_points(1), HX.mass[1], HY.mass[1], cost,
                                 Cplex(warm_start=False))[0]
            refined = set(zip(*(entries.tolist() for entries in refine(coarse, HX, HY, 0))))
            for order in ["index", "transport"]:
                with self.subTest(kind=kind, order=order):
                    start = north_west_start(coarse, HX, HY, 0, order)
                    self.assertTrue(np.array_equal(np.bincount(start.row, start.data, len(mu)), mu))
                    self.assertTrue(np.array_equal(np.bincount(start.col, start.data, len(nu)), nu))
                    self.assertLessEqual(set(zip(start.row.tolist(), start.col.tolist())), refined)

    def test_quadtree(self):
        for kind in ["clustered", "grid"]:
            with self.subTest(kind=kind):
                self.check(*make_problem(400, kind, np.random.default_rng(0), mask="disc"), quadtree=True)


if __name__ == "__main__":
    unittest.main()

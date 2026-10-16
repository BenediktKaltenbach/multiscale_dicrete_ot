import unittest

import numpy as np

from msot.hierarchy import agglomerate, build_pair
from msot.problems import make_problem


class TestHierarchy(unittest.TestCase):
    def check_invariants(self, H):
        n = len(H.points)
        self.assertEqual(H.n_cells(H.K), 1)
        for k in range(H.K + 1):
            self.assertEqual(len(H.cell_of[k]), n)
            self.assertAlmostEqual(H.mass[k].sum(), H.mass[0].sum(), delta=1e-14)
            self.assertTrue((H.cell_of[k][H.rep[k]] == np.arange(H.n_cells(k))).all())
            dist = np.linalg.norm(H.points - H.points[H.rep[k][H.cell_of[k]]], axis=1)
            self.assertTrue((dist <= H.rad[k][H.cell_of[k]]).all())
            if k < H.K:
                np.testing.assert_array_equal(H.parent[k][H.cell_of[k]], H.cell_of[k + 1])
                self.assertTrue(np.isin(H.rep[k + 1], H.rep[k]).all())

    def test_invariants(self):
        for kind in ["uniform", "clustered", "grid"]:
            X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0))
            HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
            self.assertEqual(HX.K, HY.K)
            self.check_invariants(HX)
            self.check_invariants(HY)

    def test_unequal_sizes(self):
        rng = np.random.default_rng(1)
        HX, HY = build_pair(rng.random((20, 2)), np.ones(20), rng.random((500, 2)), np.ones(500), n_dense=4)
        self.assertEqual(HX.K, HY.K)
        self.check_invariants(HX)
        self.check_invariants(HY)

    def test_about_four_children(self):
        P = np.random.default_rng(2).random((2000, 2))
        H = agglomerate(P, np.ones(2000), n_dense=32)
        self.assertGreater(H.n_cells(0) / H.n_cells(1), 3)


if __name__ == "__main__":
    unittest.main()

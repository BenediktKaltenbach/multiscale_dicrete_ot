import unittest

import numpy as np
from scipy.spatial import cKDTree

from msot.hierarchy import agglomerate, build_pair, build_pair_quadtree
from msot.problems import make_problem


class TestHierarchy(unittest.TestCase):
    def check_invariants(self, H):
        n = len(H.points)
        self.assertEqual(H.number_of_cells(H.K), 1)
        for k in range(H.K + 1):
            self.assertEqual(len(H.cell_of[k]), n)
            self.assertAlmostEqual(H.mass[k].sum(), H.mass[0].sum(), delta=1e-14)
            self.assertTrue((H.cell_of[k][H.rep[k]] == np.arange(H.number_of_cells(k))).all())
            distance = np.linalg.norm(H.points - H.points[H.rep[k][H.cell_of[k]]], axis=1)
            self.assertTrue((distance <= H.rad[k][H.cell_of[k]]).all())
            if k < H.K:
                np.testing.assert_array_equal(H.parent[k][H.cell_of[k]], H.cell_of[k + 1])
                self.assertTrue(np.isin(H.rep[k + 1], H.rep[k]).all())

    def check_mesh(self, H):
        self.check_invariants(H)
        for k in range(H.K - 1):
            self.assertLessEqual(np.bincount(H.parent[k]).max(), 5)

    def test_invariants(self):
        for kind in ["uniform", "clustered", "grid"]:
            for length_cap in [3.0, None]:
                with self.subTest(kind=kind, length_cap=length_cap):
                    X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0))
                    HX, HY = build_pair(X, mu, Y, nu, n_dense=8, length_cap=length_cap)
                    self.assertEqual(HX.K, HY.K)
                    self.check_mesh(HX)
                    self.check_mesh(HY)

    def test_unequal_sizes(self):
        for length_cap in [3.0, None]:
            with self.subTest(length_cap=length_cap):
                rng = np.random.default_rng(1)
                HX, HY = build_pair(rng.random((20, 2)), np.ones(20), rng.random((500, 2)), np.ones(500), n_dense=4,
                                    length_cap=length_cap)
                self.assertEqual(HX.K, HY.K)
                self.check_mesh(HX)
                self.check_mesh(HY)

    def test_length_cap_compactness(self):
        n = 4096
        for scale in ([1.0, 1.0], [1.0, 0.1]):
            points = np.random.default_rng(0).random((n, 2)) * scale
            maximum = {}
            for length_cap in [3.0, None]:
                H = agglomerate(points, np.ones(n), n_dense=256, length_cap=length_cap)
                reps = H.rep_points(1)
                maximum[length_cap] = (H.rad[1] / cKDTree(reps).query(reps, 2)[0][:, 1]).max()
            with self.subTest(scale=scale):
                self.assertLess(maximum[3.0], 5)
                self.assertLess(maximum[3.0], maximum[None])

    def test_quadtree(self):
        for kind in ["uniform", "clustered", "grid"]:
            X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0))
            HX, HY = build_pair_quadtree(X, mu, Y, nu)
            self.assertEqual(HX.K, HY.K)
            self.assertEqual(HX.K, 5)
            for H in (HX, HY):
                self.check_invariants(H)
                side = np.ptp(np.vstack([X, Y]), axis=0).max()
                for k in range(1, H.K + 1):
                    self.assertTrue((H.rad[k] <= np.sqrt(2) * side * 2.0 ** (k - H.K) + 1e-12).all())

    def test_about_four_children(self):
        points = np.random.default_rng(2).random((2000, 2))
        H = agglomerate(points, np.ones(2000), n_dense=32)
        self.check_mesh(H)
        self.assertGreater(H.number_of_cells(0) / H.number_of_cells(1), 3)


if __name__ == "__main__":
    unittest.main()

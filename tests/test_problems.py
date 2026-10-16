import unittest

import numpy as np

from msot.problems import make_problem


class TestProblems(unittest.TestCase):
    def test_masses_and_points(self):
        for kind in ["uniform", "clustered", "grid"]:
            for mask in [None, "halfplane", "disc"]:
                X, mu, Y, nu = make_problem(100, kind, np.random.default_rng(0), mask=mask)
                self.assertTrue((mu > 0).all() and (nu > 0).all())
                self.assertEqual(mu.sum(), nu.sum())
                for points in (X, Y):
                    self.assertTrue(((points >= 0) & (points <= 1)).all())
                    self.assertEqual(len(np.unique(points, axis=0)), len(points))

    def test_integer_atoms(self):
        X, mu, Y, nu = make_problem(500, "clustered", np.random.default_rng(1), mask="disc")
        for mass in (mu, nu):
            np.testing.assert_array_equal(mass, np.round(mass))
            self.assertEqual(mass.sum(), 10**9)
            self.assertGreaterEqual(mass.min(), 1)
        X, mu, Y, nu = make_problem(500, "clustered", np.random.default_rng(1), atoms=None)
        self.assertEqual(mu.sum(), nu.sum())
        self.assertAlmostEqual(mu.sum(), 1.0, delta=1e-12)

    def test_reproducible(self):
        a = make_problem(50, "clustered", np.random.default_rng(3))
        b = make_problem(50, "clustered", np.random.default_rng(3))
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)


if __name__ == "__main__":
    unittest.main()

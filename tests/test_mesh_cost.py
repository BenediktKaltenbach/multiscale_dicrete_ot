import unittest

import numpy as np
from scipy.spatial import ConvexHull

from msot.cost import SqEuclid
from msot.mesh import cone_q, one_ring


class TestMesh(unittest.TestCase):
    def test_one_ring_symmetric(self):
        P = np.random.default_rng(0).random((200, 2))
        indptr, indices = one_ring(P)
        edges = {(a, b) for a in range(len(P)) for b in indices[indptr[a]:indptr[a + 1]]}
        self.assertTrue(all((b, a) in edges for a, b in edges))

    def test_q_sign(self):
        P = np.random.default_rng(1).random((200, 2))
        q = cone_q(P, one_ring(P))
        hull = np.zeros(len(P), bool)
        hull[ConvexHull(P).vertices] = True
        self.assertTrue((q[hull] <= 1e-12).all())
        self.assertTrue((q[~hull] > 0).all())


class TestSqEuclid(unittest.TestCase):
    def test_psi_hat_is_lower_bound(self):
        rng = np.random.default_rng(2)
        cost = SqEuclid()
        for _ in range(100):
            xA, xs, rep = rng.random((3, 2))
            rad = rng.random() * 0.3
            ys = rep + rad * rng.uniform(-1, 1, (50, 2)) / np.sqrt(2)
            self.assertTrue((cost.psi(xA, xs, ys) >= cost.psi_hat(xA, xs, rep, rad) - 1e-12).all())


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
from scipy.spatial import ConvexHull

from msot.cost import SquaredEuclidean
from msot.mesh import cone_q, one_ring, sector_neighbours


class TestMesh(unittest.TestCase):
    def test_one_ring_symmetric(self):
        points = np.random.default_rng(0).random((200, 2))
        index_pointer, indices = one_ring(points)
        edges = {(a, b) for a in range(len(points)) for b in indices[index_pointer[a]:index_pointer[a + 1]]}
        self.assertTrue(all((b, a) in edges for a, b in edges))

    def test_q_sign(self):
        points = np.random.default_rng(1).random((200, 2))
        q = cone_q(points, one_ring(points))
        hull = np.zeros(len(points), bool)
        hull[ConvexHull(points).vertices] = True
        self.assertTrue((q[hull] <= 1e-12).all())
        self.assertTrue((q[~hull] > 0).all())

    def test_sectors_on_grid_give_four_neighbourhood(self):
        side = 10
        coordinates = np.arange(side, dtype=float)
        points = np.stack(np.meshgrid(coordinates, coordinates, indexing="ij"), axis=-1).reshape(-1, 2)
        index_pointer, indices = sector_neighbours(points, 4, offset=-np.pi / 4)
        for i in range(1, side - 1):
            for j in range(1, side - 1):
                vertex = i * side + j
                expected = {vertex - side, vertex + side, vertex - 1, vertex + 1}
                self.assertEqual(set(indices[index_pointer[vertex]:index_pointer[vertex + 1]].tolist()), expected)

    def test_sectors_nearest_and_q_floor(self):
        points = np.random.default_rng(3).random((2000, 2))
        for sectors in (6, 8):
            index_pointer, indices = sector_neighbours(points, sectors)
            counts = np.diff(index_pointer)
            self.assertLessEqual(counts.max(), sectors)
            q = cone_q(points, (index_pointer, indices))
            self.assertTrue((q[counts == sectors] >= np.cos(2 * np.pi / sectors) - 1e-12).all())
            for vertex in range(0, 2000, 97):
                others = np.argsort(np.linalg.norm(points - points[vertex], axis=1))[1:33]
                directions = points[others] - points[vertex]
                sector = (np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2 * np.pi) // (2 * np.pi / sectors)).astype(int)
                distance = np.linalg.norm(directions, axis=1)
                expected = {others[sector == s][np.argmin(distance[sector == s])] for s in np.unique(sector)}
                self.assertEqual(set(indices[index_pointer[vertex]:index_pointer[vertex + 1]].tolist()), expected)


class TestSquaredEuclidean(unittest.TestCase):
    def test_psi_hat_is_lower_bound(self):
        rng = np.random.default_rng(2)
        cost = SquaredEuclidean()
        for _ in range(100):
            xA, xs, rep = rng.random((3, 2))
            rad = rng.random() * 0.3
            ys = rep + rad * rng.uniform(-1, 1, (50, 2)) / np.sqrt(2)
            self.assertTrue((cost.psi(xA, xs, ys) >= cost.psi_hat(xA, xs, rep, rad) - 1e-12).all())


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from msot.cost import SqrtQuadratic, SquaredEuclidean
from msot.hierarchy import build_pair
from msot.multiscale import solve_multiscale
from msot.problems import make_problem
from msot.shielding import extract_map, missed_brute, search_tree, shield
from msot.solvers.base import solve_dense
from msot.solvers.cplex import Cplex
from msot.verify import dual_violation


def coupling(kind, n=400, seed=0, cost=None):
    """A problem with its hierarchies and the optimal coupling at level 0."""
    X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed))
    HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
    pi = solve_multiscale(HX, HY, cost or SqrtQuadratic(), Cplex())[0]
    return X, Y, HX, HY, pi


class TestSqrtQuadratic(unittest.TestCase):
    def test_psi_hat_is_lower_bound(self):
        rng = np.random.default_rng(2)
        cost = SqrtQuadratic()
        for _ in range(200):
            xA, xs, rep = rng.random((3, 2))
            rad = rng.random() * 0.3
            ys = rep + rad * rng.uniform(-1, 1, (50, 2)) / np.sqrt(2)
            self.assertTrue((cost.psi(xA, xs, ys) >= cost.psi_hat(xA, xs, rep, rad) - 1e-12).all())

    def test_chart_normal_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        cost = SqrtQuadratic()
        xA, xs, ys = rng.random((3, 20, 2))
        normals, offsets = cost._face_data(xA, xs, ys, 0.0)
        p_s = cost.chart(xA, ys)

        def f(a, p):
            y = xA[a] + p / np.sqrt(1 - np.sum(p ** 2))
            return cost.psi(xA[a], xs[a], y)

        for a in range(20):
            step = 1e-6
            gradient = np.array([(f(a, p_s[a] + step * e) - f(a, p_s[a] - step * e)) / (2 * step) for e in np.eye(2)])
            np.testing.assert_allclose(normals[a], gradient, rtol=1e-5, atol=1e-8)
            self.assertAlmostEqual(offsets[a], normals[a] @ p_s[a], delta=1e-12)

    def test_unshielded_set_inside_tangent_half_plane(self):
        """Loeper (MTW): {y : psi(y) <= psi(y_s)} is convex in the chart, so it lies in the tangent half-plane."""
        rng = np.random.default_rng(4)
        cost = SqrtQuadratic()
        Y = rng.random((4000, 2))
        for _ in range(50):
            xA, xs, ys = rng.random((3, 2))
            normal, offset = cost._face_data(xA[None], xs[None], ys[None], 0.0)
            unshielded = cost.psi(xA, xs, Y) <= cost.psi(xA, xs, ys)
            side = cost.chart(xA, Y[unshielded]) @ normal[0] - offset[0]
            self.assertTrue((side <= 1e-12).all())

    def test_search_matches_brute_force(self):
        cost = SqrtQuadratic()
        for kind in ["uniform", "clustered"]:
            X, Y, HX, HY, pi = coupling(kind, cost=cost)
            t = extract_map(pi)
            for k in (0, 1):
                Xk, Yk = HX.rep_points(k), HY.rep_points(k)
                tk = t if k == 0 else HY.cell_of[1][t[HX.rep[1]]]
                index_pointer, indices = HX.neighbours[k]
                for a in range(0, len(Xk), 3):
                    candidates = indices[index_pointer[a]:index_pointer[a + 1]]
                    triple = Xk[a], Xk[candidates], Yk[tk[candidates]]
                    np.testing.assert_array_equal(np.sort(search_tree(*triple, HY, k, cost, 1e-12)),
                                                  missed_brute(*triple, Yk, cost, 1e-12))

    def test_pruning_with_filter_is_exact(self):
        for cost_class in (SqrtQuadratic, SquaredEuclidean):
            for kind in ["uniform", "clustered", "grid"]:
                with self.subTest(cost=cost_class.__name__, kind=kind):
                    X, Y, HX, HY, pi = coupling(kind, cost=cost_class())
                    plain = shield(pi, HX.neighbours[0], X, HY, 0, cost_class(prune=False))
                    pruned = shield(pi, HX.neighbours[0], X, HY, 0, cost_class(prune=True))
                    np.testing.assert_array_equal(plain[0], pruned[0])
                    np.testing.assert_array_equal(plain[1], pruned[1])

    def test_multiscale_matches_dense(self):
        for kind, mask in [("uniform", None), ("clustered", "disc"), ("grid", "disc")]:
            with self.subTest(kind=kind):
                X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0), mask=mask)
                HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
                dense = solve_dense(X, Y, mu, nu, SqrtQuadratic(), Cplex(warm_start=False))[0]
                C_dense = dense.data @ SqrtQuadratic().c(X[dense.row], Y[dense.col])
                for prune in (False, True):
                    cost = SqrtQuadratic(prune=prune)
                    pi, alpha, beta, _ = solve_multiscale(HX, HY, cost, Cplex())
                    self.assertAlmostEqual(pi.data @ cost.c(X[pi.row], Y[pi.col]), C_dense, delta=1e-9 * C_dense)
                    self.assertLess(dual_violation(alpha, beta, X, Y, cost), 1e-6)


if __name__ == "__main__":
    unittest.main()

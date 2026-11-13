import unittest

import numpy as np
from scipy.optimize import linprog

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.higher_dimensions import SquaredEuclideanPruning, _facets, make_problem
from msot.multiscale import solve_multiscale
from msot.shielding import extract_map, missed_brute, search_tree, shield
from msot.solvers.base import solve_dense
from msot.solvers.cplex import Cplex
from msot.verify import dual_violation


def coupling(kind, dimension, n=300, seed=0):
    X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed), dimension)
    HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
    pi = solve_multiscale(HX, HY, SquaredEuclidean(), Cplex())[0]
    return X, Y, HX, HY, pi


class TestHigherDimensions(unittest.TestCase):
    def test_facets_match_one_linear_program_per_half_space(self):
        rng = np.random.default_rng(5)
        for dimension in (3, 4, 5):
            for _ in range(20):
                normals = rng.normal(size=(30, dimension))
                offsets = rng.uniform(0.5, 1.5, 30)
                expected = []
                for s in range(30):
                    relaxed = offsets.copy()
                    relaxed[s] += 1.0
                    result = linprog(-normals[s], A_ub=normals, b_ub=relaxed, bounds=(None, None), method="highs")
                    expected.append(result.status == 3 or -result.fun > offsets[s] + 1e-9)
                np.testing.assert_array_equal(_facets(normals, offsets), expected)

    def test_search_matches_brute_force(self):
        cost = SquaredEuclidean()
        for dimension in (3, 4):
            X, Y, HX, HY, pi = coupling("uniform", dimension)
            t = extract_map(pi)
            index_pointer, indices = HX.neighbours[0]
            for a in range(0, len(X), 5):
                candidates = indices[index_pointer[a]:index_pointer[a + 1]]
                triple = X[a], X[candidates], Y[t[candidates]]
                np.testing.assert_array_equal(np.sort(search_tree(*triple, HY, 0, cost, 1e-12)),
                                              missed_brute(*triple, Y, cost, 1e-12))

    def test_pruning_with_filter_is_exact(self):
        for dimension in (4, 5):
            for kind in ("uniform", "clustered"):
                X, Y, HX, HY, pi = coupling(kind, dimension)
                plain = shield(pi, HX.neighbours[0], X, HY, 0, SquaredEuclidean())
                for face_test in ("exact", "closest"):
                    with self.subTest(dimension=dimension, kind=kind, face_test=face_test):
                        cost = SquaredEuclideanPruning(face_test)
                        self.assertFalse(cost.needed_candidates_all(X, HX.neighbours[0], Y[extract_map(pi)][
                            HX.neighbours[0][1]], 1e-12).all())
                        pruned = shield(pi, HX.neighbours[0], X, HY, 0, cost)
                        np.testing.assert_array_equal(plain[0], pruned[0])
                        np.testing.assert_array_equal(plain[1], pruned[1])

    def test_multiscale_matches_dense(self):
        for dimension in (3, 4):
            with self.subTest(dimension=dimension):
                X, mu, Y, nu = make_problem(300, "clustered", np.random.default_rng(1), dimension)
                HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
                dense = solve_dense(X, Y, mu, nu, SquaredEuclidean(), Cplex(warm_start=False))[0]
                C_dense = dense.data @ SquaredEuclidean().c(X[dense.row], Y[dense.col])
                for cost in (SquaredEuclidean(), SquaredEuclideanPruning("exact"), SquaredEuclideanPruning("closest")):
                    pi, alpha, beta, _ = solve_multiscale(HX, HY, cost, Cplex())
                    self.assertAlmostEqual(pi.data @ cost.c(X[pi.row], Y[pi.col]), C_dense, delta=1e-9 * C_dense)
                    self.assertLess(dual_violation(alpha, beta, X, Y, cost), 1e-6)


if __name__ == "__main__":
    unittest.main()

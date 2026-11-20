import unittest

import numpy as np

from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.mesh import one_ring, sector_neighbours
from msot.multiscale import solve_multiscale, solve_sparse
from msot.problems import make_problem
from msot.shielding import extract_map, missed_brute, search_tree, shield, shield_changed, unique_entries
from msot.solvers.base import solve_dense
from msot.solvers.cplex import Cplex
from msot.verify import dual_violation, is_shielding


def north_west_corner(mu, nu, order_x, order_y):
    """Feasible initial N1 with at most |X| + |Y| - 1 entries: north-west corner rule along the given orderings."""
    cumulative_x, cumulative_y = np.cumsum(mu[order_x]), np.cumsum(nu[order_y])
    breaks = np.unique(np.concatenate([cumulative_x, cumulative_y]))[:-1]
    i = np.minimum(np.searchsorted(cumulative_x, breaks, side="right"), len(mu) - 1)
    j = np.minimum(np.searchsorted(cumulative_y, breaks, side="right"), len(nu) - 1)
    rows = np.concatenate([[order_x[0]], order_x[i]])
    columns = np.concatenate([[order_y[0]], order_y[j]])
    return rows, columns


class TestSearchTree(unittest.TestCase):
    def test_matches_brute_force(self):
        cost, solver = SquaredEuclidean(), Cplex()
        for kind in ["uniform", "clustered", "grid"]:
            X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0))
            HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
            for k in [0, 1]:
                Xk, Yk = HX.rep_points(k), HY.rep_points(k)
                N1 = north_west_corner(HX.mass[k], HY.mass[k], np.argsort(Xk[:, 0]), np.argsort(Yk[:, 1]))
                pi = solver.solve_local(*N1, cost.c(Xk[N1[0]], Yk[N1[1]]), HX.mass[k], HY.mass[k])[0]
                t = extract_map(pi)
                index_pointer, indices = HX.neighbours[k]
                for a in range(len(Xk)):
                    candidates = indices[index_pointer[a]:index_pointer[a + 1]]
                    triple = Xk[a], Xk[candidates], Yk[t[candidates]]
                    np.testing.assert_array_equal(np.sort(search_tree(*triple, HY, k, cost, 1e-12)),
                                                  missed_brute(*triple, Yk, cost, 1e-12))


class TestPruning(unittest.TestCase):
    def test_pruning_keeps_Y_hat(self):
        cost, solver = SquaredEuclidean(prune=True), Cplex()
        for kind in ["uniform", "clustered", "grid"]:
            X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0))
            _, HY = build_pair(X, mu, Y, nu, n_dense=8)
            N1 = north_west_corner(mu, nu, np.argsort(X[:, 0]), np.argsort(Y[:, 1]))
            t = extract_map(solver.solve_local(*N1, cost.c(X[N1[0]], Y[N1[1]]), mu, nu)[0])
            for S in (one_ring(X), sector_neighbours(X, 8)):
                index_pointer, indices = S
                for a in range(len(X)):
                    candidates = indices[index_pointer[a]:index_pointer[a + 1]]
                    needed = candidates[cost.needed_candidates(X[a], X[candidates], Y[t[candidates]], 1e-12)]
                    np.testing.assert_array_equal(
                        np.sort(search_tree(X[a], X[needed], Y[t[needed]], HY, 0, cost, 1e-12)),
                        np.sort(search_tree(X[a], X[candidates], Y[t[candidates]], HY, 0, cost, 1e-12)))


class TestVectorisedShield(unittest.TestCase):
    def test_equals_loop(self):
        solver = Cplex()
        for kind in ["uniform", "clustered"]:
            X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0))
            HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
            for k in (0, 1):
                Xk, Yk = HX.rep_points(k), HY.rep_points(k)
                N1 = north_west_corner(HX.mass[k], HY.mass[k], np.argsort(Xk[:, 0]), np.argsort(Yk[:, 1]))
                pi = solver.solve_local(*N1, SquaredEuclidean().c(Xk[N1[0]], Yk[N1[1]]), HX.mass[k], HY.mass[k])[0]
                t = extract_map(pi)
                for prune in (False, True):
                    cost = SquaredEuclidean(prune=prune)
                    index_pointer, indices = HX.neighbours[k]
                    rows, columns = [pi.row], [pi.col]
                    for a in range(len(Xk)):
                        candidates = indices[index_pointer[a]:index_pointer[a + 1]]
                        needed = candidates[cost.needed_candidates(Xk[a], Xk[candidates], Yk[t[candidates]], 1e-12)]
                        new = np.concatenate([t[candidates], search_tree(Xk[a], Xk[needed], Yk[t[needed]], HY, k, cost, 1e-12)])
                        rows.append(np.full(len(new), a))
                        columns.append(new)
                    expected = unique_entries(np.concatenate(rows), np.concatenate(columns), len(Yk))
                    result = shield(pi, HX.neighbours[k], Xk, HY, k, cost)
                    np.testing.assert_array_equal(result[0], expected[0])
                    np.testing.assert_array_equal(result[1], expected[1])


class TestShieldChanged(unittest.TestCase):
    def test_equals_shield_in_every_round(self):
        for kind in ["uniform", "clustered", "grid"]:
            for prune in (False, True):
                with self.subTest(kind=kind, prune=prune):
                    X, mu, Y, nu = make_problem(400, kind, np.random.default_rng(0))
                    _, HY = build_pair(X, mu, Y, nu, n_dense=8)
                    cost, solver, S = SquaredEuclidean(prune=prune), Cplex(), one_ring(X)
                    N = north_west_corner(mu, nu, np.argsort(X[:, 0]), np.argsort(Y[:, 1]))
                    state, previous, rounds = None, None, 0
                    while True:
                        pi, _, _, state = solver.solve_local(*N, cost.c(X[N[0]], Y[N[1]]), mu, nu, state)
                        full = shield(pi, S, X, HY, 0, cost)
                        if previous is not None:
                            changed = shield_changed(pi, previous, S, X, HY, 0, cost)
                            np.testing.assert_array_equal(changed[0], full[0])
                            np.testing.assert_array_equal(changed[1], full[1])
                        rounds += 1
                        keys = lambda entries: entries[0].astype(np.int64) * len(Y) + entries[1]
                        if np.isin(keys(full), keys(N)).all():
                            break
                        previous, N = (pi, full), full
                    self.assertGreater(rounds, 2)

    def test_same_history_as_full_shield(self):
        X, mu, Y, nu = make_problem(400, "clustered", np.random.default_rng(1))
        HX, HY = build_pair(X, mu, Y, nu, n_dense=8)
        histories = []
        for incremental in (True, False):
            info = solve_multiscale(HX, HY, SquaredEuclidean(), Cplex(), incremental=incremental)[3]
            histories.append([(round_["C"], round_["N"]) for level in info for round_ in level["history"]])
        self.assertEqual(histories[0], histories[1])


class TestSolveSparse(unittest.TestCase):
    def check(self, n, kind, seed, mask=None):
        X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed), mask=mask)
        cost, solver = SquaredEuclidean(), Cplex()
        _, HY = build_pair(X, mu, Y, nu, n_dense=8)
        pi_dense, _, _, _ = solve_dense(X, Y, mu, nu, cost, Cplex(warm_start=False))
        C_dense = pi_dense.data @ cost.c(X[pi_dense.row], Y[pi_dense.col])

        N1 = north_west_corner(mu, nu, np.argsort(X[:, 0]), np.argsort(Y[:, 0]))
        pi, alpha, beta, N, history = solve_sparse(X, mu, HY, 0, cost, one_ring(X), N1, solver)
        C = pi.data @ cost.c(X[pi.row], Y[pi.col])

        self.assertAlmostEqual(C, C_dense, delta=1e-9 * C_dense)
        self.assertLess(dual_violation(alpha, beta, X, Y, cost), 1e-9)
        self.assertTrue(is_shielding(pi, *N, X, Y, cost))
        self.assertLess(len(N[0]), len(X) * len(Y))
        return history

    def test_uniform(self):
        for seed in range(3):
            self.check(100, "uniform", seed)

    def test_clustered(self):
        self.check(100, "clustered", 0)

    def test_grid_with_mask(self):
        self.check(144, "grid", 0, mask="disc")


if __name__ == "__main__":
    unittest.main()

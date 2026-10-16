import highspy
import numpy as np
from scipy.sparse import coo_matrix

from .base import LocalSolver


class Highs(LocalSolver):
    """HiGHS dual simplex via highspy, solved from scratch each call.

    pi holds every basic entry, including degenerate zeros, so its support is connected like a network simplex
    basis tree; basic entries satisfy alpha + beta = c (2.4), which is all shielding needs. Marginals are scaled
    to mean mass 1 because HiGHS tolerances are absolute; the duals are unaffected. The primal tolerance is
    tight because basic values down to -tolerance are clipped to 0, and a refined N1 inherits that error.
    """

    def __init__(self, primal_tolerance=1e-10, dual_tolerance=1e-9):
        self.options = {"output_flag": False, "solver": "simplex", "simplex_strategy": 1,
                        "primal_feasibility_tolerance": primal_tolerance,
                        "dual_feasibility_tolerance": dual_tolerance}

    def solve_local(self, rows, columns, c, mu, nu, state=None):
        size_x, size_y, size_N = len(mu), len(nu), len(rows)
        scale = size_x / mu.sum()
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = size_N, size_x + size_y
        lp.col_cost_ = np.asarray(c, dtype=float)
        lp.col_lower_, lp.col_upper_ = np.zeros(size_N), np.full(size_N, highspy.kHighsInf)
        lp.row_lower_ = lp.row_upper_ = scale * np.concatenate([mu, nu])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.arange(0, 2 * size_N + 1, 2, dtype=np.int32)
        lp.a_matrix_.index_ = np.stack([rows, size_x + columns], axis=1).ravel().astype(np.int32)
        lp.a_matrix_.value_ = np.ones(2 * size_N)
        h = highspy.Highs()
        for name, value in self.options.items():
            h.setOptionValue(name, value)
        h.passModel(lp)
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(h.modelStatusToString(h.getModelStatus()))
        solution, basis = h.getSolution(), h.getBasis()
        basic = np.array([status == highspy.HighsBasisStatus.kBasic for status in basis.col_status])
        values = np.maximum(np.asarray(solution.col_value)[basic], 0) / scale
        pi = coo_matrix((values, (rows[basic], columns[basic])), shape=(size_x, size_y))
        duals = np.asarray(solution.row_dual)
        return pi, duals[:size_x], duals[size_x:], None

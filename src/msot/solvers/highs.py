import highspy
import numpy as np
from scipy.sparse import coo_matrix

from .base import LocalSolver


class Highs(LocalSolver):
    """HiGHS simplex via highspy.

    pi holds every basic entry, including degenerate zeros, so its support is connected like a network simplex
    basis tree; basic entries satisfy alpha + beta = c (2.4), which is all shielding needs. Marginals are scaled
    to mean mass 1 because HiGHS tolerances are absolute; the duals are unaffected. The primal tolerance is
    tight because basic values down to -tolerance are clipped to 0, and a refined N1 inherits that error.

    With warm_start, state keeps the model and the entry key (row * size_y + column) of each model column. The
    next call adds the new entries of N as columns and deletes only nonbasic columns outside N, so the old basis
    stays primal feasible and the primal simplex continues from it. The model's columns are then a superset of N,
    which is still shielding, and pi lives on the model's basic columns.
    """

    def __init__(self, primal_tolerance=1e-10, dual_tolerance=1e-9, warm_start=True):
        self.options = {"output_flag": False, "solver": "simplex", "simplex_strategy": 1,
                        "primal_feasibility_tolerance": primal_tolerance,
                        "dual_feasibility_tolerance": dual_tolerance}
        self.warm_start = warm_start
        self.simplex_iterations = []

    def solve_local(self, rows, columns, c, mu, nu, state=None):
        size_x, size_y = len(mu), len(nu)
        scale = size_x / mu.sum()
        keys = rows.astype(np.int64) * size_y + columns
        if self.warm_start and state is not None and state["shape"] == (size_x, size_y):
            h = state["highs"]
            keys = update_columns(h, state["keys"], keys, c, size_x, size_y)
            h.setOptionValue("simplex_strategy", 4)
        else:
            h = self.build(rows, columns, c, mu, nu, scale)
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(h.modelStatusToString(h.getModelStatus()))
        self.simplex_iterations.append(h.getInfo().simplex_iteration_count)
        solution, basic = h.getSolution(), basic_columns(h)
        values = np.maximum(np.asarray(solution.col_value)[basic], 0) / scale
        pi = coo_matrix((values, (keys[basic] // size_y, keys[basic] % size_y)), shape=(size_x, size_y))
        duals = np.asarray(solution.row_dual)
        state = {"highs": h, "keys": keys, "shape": (size_x, size_y)} if self.warm_start else None
        return pi, duals[:size_x], duals[size_x:], state

    def build(self, rows, columns, c, mu, nu, scale):
        size_x, size_y, size_N = len(mu), len(nu), len(rows)
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
        return h


def basic_columns(h):
    status = h.getBasis().col_status
    return np.fromiter((s == highspy.HighsBasisStatus.kBasic for s in status), dtype=bool, count=len(status))


def update_columns(h, model_keys, keys, c, size_x, size_y):
    """Delete the nonbasic model columns outside N, add the entries of N the model lacks; returns the new keys."""
    delete = ~basic_columns(h) & ~np.isin(model_keys, keys)
    if delete.any():
        h.deleteCols(int(delete.sum()), np.flatnonzero(delete).astype(np.int32))
        model_keys = model_keys[~delete]
    new = ~np.isin(keys, model_keys)
    size_new = int(new.sum())
    if size_new:
        rows, columns = keys[new] // size_y, keys[new] % size_y
        h.addCols(size_new, np.asarray(c, dtype=float)[new], np.zeros(size_new), np.full(size_new, highspy.kHighsInf),
                  2 * size_new, np.arange(0, 2 * size_new, 2, dtype=np.int32),
                  np.stack([rows, size_x + columns], axis=1).ravel().astype(np.int32), np.ones(2 * size_new))
    return np.concatenate([model_keys, keys[new]])

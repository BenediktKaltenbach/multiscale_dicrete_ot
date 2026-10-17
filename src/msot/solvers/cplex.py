import cplex
import numpy as np
from scipy.sparse import coo_matrix

from .base import LocalSolver


class Cplex(LocalSolver):
    """CPLEX network optimizer (network simplex), as in Schmitzer 2016, Sect. 6.

    Each entry (x, y) of N is an arc x -> y: coefficient +1 in the row of x (= mu(x)) and -1 in the row of y
    (= -nu(y)), a pure network matrix, so CPLEX solves it with its network simplex. alpha = dual of the x rows,
    beta = -(dual of the y rows). As in Highs, pi holds every basic column including degenerate zeros, integer
    masses (mass atoms) are rounded to the exact integral basic solution and checked, float masses are scaled
    to mean mass 1. With warm_start, state keeps the model and the entry key (row * size_y + column) of each
    model column; the next call adds the new entries of N, deletes only nonbasic columns outside N and CPLEX
    continues from the current basis.
    """

    def __init__(self, optimality_tolerance=1e-9, warm_start=True, threads=1):
        self.optimality_tolerance = optimality_tolerance
        self.warm_start = warm_start
        self.threads = threads

    def solve_local(self, rows, columns, c, mu, nu, state=None):
        size_x, size_y = len(mu), len(nu)
        integral = is_integral(mu) and is_integral(nu)
        scale = 1.0 if integral else size_x / mu.sum()
        keys = rows.astype(np.int64) * size_y + columns
        if self.warm_start and state is not None and state["shape"] == (size_x, size_y):
            model = state["cplex"]
            keys = update_columns(model, state["keys"], keys, c, size_x, size_y)
        else:
            model = self.build(rows, columns, c, mu, nu, scale, integral)
        model.solve()
        if model.solution.get_status() != model.solution.status.optimal:
            raise RuntimeError(model.solution.get_status_string())
        column_status, _ = model.solution.basis.get_basis()
        basic = np.asarray(column_status) == model.solution.basis.status.basic
        values = np.asarray(model.solution.get_values())[basic]
        values = np.maximum(np.round(values) if integral else values, 0) / scale
        pi = coo_matrix((values, (keys[basic] // size_y, keys[basic] % size_y)), shape=(size_x, size_y))
        if integral and not (np.array_equal(np.bincount(pi.row, pi.data, size_x), mu)
                             and np.array_equal(np.bincount(pi.col, pi.data, size_y), nu)):
            raise RuntimeError("rounded basic solution does not meet the integer marginals")
        duals = np.asarray(model.solution.get_dual_values())
        state = {"cplex": model, "keys": keys, "shape": (size_x, size_y)} if self.warm_start else None
        return pi, duals[:size_x], -duals[size_x:], state

    def build(self, rows, columns, c, mu, nu, scale, integral):
        size_x = len(mu)
        model = cplex.Cplex()
        for stream in (model.set_log_stream, model.set_results_stream, model.set_warning_stream):
            stream(None)
        parameters = model.parameters
        parameters.lpmethod.set(parameters.lpmethod.values.network)
        parameters.threads.set(self.threads)
        parameters.network.tolerances.optimality.set(self.optimality_tolerance)
        parameters.simplex.tolerances.optimality.set(self.optimality_tolerance)
        if not integral:
            parameters.network.tolerances.feasibility.set(1e-10)
            parameters.simplex.tolerances.feasibility.set(1e-9)
        model.linear_constraints.add(rhs=list(scale * np.concatenate([mu, -nu])), senses="E" * (size_x + len(nu)))
        add_columns(model, rows, columns, c, size_x)
        return model


def is_integral(mass):
    return bool(np.all(mass == np.round(mass)))


def add_columns(model, rows, columns, c, size_x):
    arcs = [[[int(x), size_x + int(y)], [1.0, -1.0]] for x, y in zip(rows, columns)]
    model.variables.add(obj=list(np.asarray(c, dtype=float)), lb=[0.0] * len(arcs), columns=arcs)


def update_columns(model, model_keys, keys, c, size_x, size_y):
    """Delete the nonbasic model columns outside N, add the entries of N the model lacks; returns the new keys."""
    column_status, _ = model.solution.basis.get_basis()
    basic = np.asarray(column_status) == model.solution.basis.status.basic
    delete = ~basic & ~np.isin(model_keys, keys)
    if delete.any():
        model.variables.delete(np.flatnonzero(delete).tolist())
        model_keys = model_keys[~delete]
    new = ~np.isin(keys, model_keys)
    if new.any():
        add_columns(model, keys[new] // size_y, keys[new] % size_y, np.asarray(c)[new], size_x)
    return np.concatenate([model_keys, keys[new]])

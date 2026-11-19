"""Phase 2 of docs/analysis_rounds.md: instrumented rerun of three scaling problems, per round of Alg. 4.1 at level 0.

No change of the algorithm: a recording solver keeps N, alpha and beta of each local solve, and a wrapper around
multiscale.shield records each level-0 round after the real shield has run (so the stored shield times include the
instrumentation). Per round: cost, changed t(x), mass moved, ties and tie switches, entries added to and removed from
N by source (spt pi, step ii, step iii), both stopping rules, global dual violation and the violating x, hops in the
level-0 mesh to the changes and violations of the previous round (H1), and the relative mass of the changed x (H3).
Writes results/rounds/instrumented.json and instrumented_<kind>_<n>_<seed>.npz; plot with rounds_where.py.

python experiments/rounds_instrumented.py
"""
import json
from pathlib import Path

import numpy as np

import msot.multiscale
from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.problems import make_problem
from msot.shielding import extract_map
from msot.solvers.cplex import Cplex

PROBLEMS = (("grid", 16384, 2), ("clustered", 32768, 0), ("uniform", 16384, 0))
RESULTS = Path(__file__).parent / "results" / "rounds"
VIOLATION = 1e-9
FLOOR_SHARE = 0.01  # relative mass mu(x) n / sum(mu) below this: a point of the 1e-3 density floor


class RecordingCplex(Cplex):
    def solve_local(self, rows, columns, c, mu, nu, state=None):
        pi, alpha, beta, state = super().solve_local(rows, columns, c, mu, nu, state)
        self.last = rows, columns, alpha, beta
        return pi, alpha, beta, state


def keys(rows, columns, size):
    return rows.astype(np.int64) * size + columns


def lookup(coo, rows, columns, size):
    """Values of the COO matrix at (rows, columns), 0 where it stores nothing."""
    stored = keys(coo.row, coo.col, size)
    order = np.argsort(stored)
    wanted = keys(rows, columns, size)
    position = np.minimum(np.searchsorted(stored[order], wanted), len(stored) - 1)
    found = stored[order][position] == wanted
    return np.where(found, coo.data[order][position], 0.0)


def hops(sources, neighbours, size, limit=200):
    """Hops in the mesh from every point to the nearest source (inf if none within limit)."""
    index_pointer, indices = neighbours
    distance = np.full(size, np.inf)
    distance[sources] = 0
    frontier = np.asarray(sources)
    for step in range(1, limit + 1):
        if len(frontier) == 0:
            break
        reached = indices[_ranges(index_pointer[frontier], np.diff(index_pointer)[frontier])]
        reached = np.unique(reached[np.isinf(distance[reached])])
        distance[reached] = step
        frontier = reached
    return distance


def _ranges(starts, lengths):
    return np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(lengths.sum())


def violations(alpha, beta, X, Y, cost, block=256):
    """Maximum of alpha + beta - c over X x Y, the number of pairs above VIOLATION and the x that have one."""
    largest, pairs, violating = -np.inf, 0, []
    for start in range(0, len(X), block):
        excess = alpha[start:start + block, None] + beta[None] - cost.c(X[start:start + block, None], Y[None])
        largest = max(largest, float(excess.max()))
        above = excess > VIOLATION
        pairs += int(above.sum())
        violating.append(start + np.flatnonzero(above.any(axis=1)))
    return largest, pairs, np.concatenate(violating)


def summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return {"count": len(values), "median": float(np.median(finite)) if len(finite) else None,
            "share <= 1": float(np.mean(values <= 1)), "share <= 2": float(np.mean(values <= 2)),
            "share <= 5": float(np.mean(values <= 5)), "max finite": float(finite.max()) if len(finite) else None}


class Recorder:
    def __init__(self, HX, HY, solver, cost):
        self.HX, self.HY, self.solver, self.cost = HX, HY, solver, cost
        self.X, self.Y = HX.rep_points(0), HY.rep_points(0)
        self.mu = HX.mass[0]
        self.relative_mass = self.mu * len(self.mu) / self.mu.sum()
        self.rounds, self.previous, self.previous_changed = [], None, None
        self.last_change = np.zeros(len(self.X), dtype=np.int32)
        self.last_violation = np.zeros(len(self.X), dtype=np.int32)

    def shield(self, pi, S, X, Yh, k, cost, tolerance=1e-12, **keywords):
        new_rows, new_columns = original_shield(pi, S, X, Yh, k, cost, tolerance, **keywords)
        if k == 0:
            self.record(pi, S, new_rows, new_columns, tolerance)
        return new_rows, new_columns

    def record(self, pi, S, new_rows, new_columns, tolerance):
        n, size = len(self.X), len(self.Y)
        r = len(self.rounds) + 1
        rows, columns, alpha, beta = self.solver.last
        coo = pi.tocoo()
        C = float(coo.data @ self.cost.c(self.X[coo.row], self.Y[coo.col]))
        pi = pi.tocsr()
        t = extract_map(pi)
        row_maximum = pi.max(axis=1).toarray().ravel()
        at_maximum = np.bincount(coo.row[coo.data == row_maximum[coo.row]], minlength=n)
        old, new = keys(rows, columns, size), keys(new_rows, new_columns, size)
        added, removed = np.setdiff1d(new, old), np.setdiff1d(old, new)
        support = keys(coo.row, coo.col, size)
        index_pointer, indices = S
        step_two = keys(np.repeat(np.arange(n), np.diff(index_pointer)), t[indices], size)
        from_support = np.isin(added, support)
        from_step_two = ~from_support & np.isin(added, step_two)
        largest, pairs, violating = violations(alpha, beta, self.X, self.Y, self.cost)
        self.last_violation[violating] = r
        record = {"round": r, "C": C, "N": len(rows), "added": len(added), "removed": len(removed),
                  "added from spt pi": int(from_support.sum()), "added from step ii": int(from_step_two.sum()),
                  "added from step iii": int((~from_support & ~from_step_two).sum()),
                  "containment": len(added) == 0, "tie rows": int(np.sum(at_maximum > 1)),
                  "dual violation": largest, "violating pairs": pairs, "violating x": len(violating),
                  "violating x at the floor": float(np.mean(self.relative_mass[violating] < FLOOR_SHARE)) if len(violating) else None}
        if self.previous is not None:
            previous_pi, previous_t, previous_violating, previous_C = self.previous
            changed = np.flatnonzero(t != previous_t)
            self.last_change[changed] = r
            tie_switch = (lookup(coo, changed, previous_t[changed], size) == lookup(coo, changed, t[changed], size))
            spacing = 1 / np.sqrt(n)
            jump = np.linalg.norm(self.Y[t[changed]] - self.Y[previous_t[changed]], axis=1) / spacing
            record.update({"cost decrease": (previous_C - C) / C, "cost rule": C >= previous_C - tolerance * abs(previous_C),
                           "mass moved": float(abs(pi - previous_pi).sum() / 2 / self.mu.sum()),
                           "changed x": len(changed), "tie switches": int(tie_switch.sum()),
                           "changed x at the floor": float(np.mean(self.relative_mass[changed] < FLOOR_SHARE)) if len(changed) else None,
                           "relative mass of changed x, median": float(np.median(self.relative_mass[changed])) if len(changed) else None,
                           "jump of t in spacings": summary(jump)})
            if len(changed) and self.previous_changed is not None and len(self.previous_changed):
                record["hops from changed to previous changed"] = summary(
                    hops(self.previous_changed, self.HX.neighbours[0], n)[changed])
            if len(violating) and len(previous_violating):
                record["hops from violating to previous violating"] = summary(
                    hops(previous_violating, self.HX.neighbours[0], n)[violating])
            self.previous_changed = changed
        else:
            self.previous_changed = None
        self.previous = pi, t, violating, C
        self.rounds.append(record)
        print(f"  round {r:2d}: C {C:.12e}, |N| {len(rows)}, added {len(added)}, changed {record.get('changed x', '-')}, "
              f"violation {largest:.1e} at {len(violating)} x", flush=True)


original_shield = msot.multiscale.shield


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "instrumented.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    floor_share_of_all = {}
    for kind, n, seed in PROBLEMS:
        name = f"{kind}_{n}_{seed}"
        if name in stored:
            continue
        print(name, flush=True)
        X, mu, Y, nu = make_problem(n, kind, np.random.default_rng(seed))
        HX, HY = build_pair(X, mu, Y, nu)
        solver, cost = RecordingCplex(), SquaredEuclidean(prune=False)
        recorder = Recorder(HX, HY, solver, cost)
        msot.multiscale.shield = recorder.shield
        try:
            pi, alpha, beta, info = msot.multiscale.solve_multiscale(HX, HY, cost, solver)
        finally:
            msot.multiscale.shield = original_shield
        floor_share_of_all[name] = float(np.mean(recorder.relative_mass < FLOOR_SHARE))
        stored[name] = {"kind": kind, "n": n, "seed": seed, "rounds at level 0": len(info[-1]["history"]),
                        "share of all x at the floor": floor_share_of_all[name], "rounds": recorder.rounds}
        np.savez(RESULTS / f"instrumented_{name}.npz", X=recorder.X, relative_mass=recorder.relative_mass,
                 last_change=recorder.last_change, last_violation=recorder.last_violation)
        path.write_text(json.dumps(stored, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

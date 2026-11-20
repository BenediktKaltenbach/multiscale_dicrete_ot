"""Recall of the closest rule: how many exact facets it keeps per share and level (branch higher-dimensions).

Solves each problem of pruning_dimensions.py once without pruning. In every shield round, for every x_A, the
half-spaces (5.5) of the 1-ring are ranked by their signed distance to a centre c, once with c = mean of the t(x_s)
(as in SquaredEuclideanPruning("closest")) and once with c = t(x_A) = argmax_y pi(x_A, y), and compared with the
exact facets. No timing. Writes experiments/results/pruning_recall/recall.json.

python experiments/pruning_recall.py [dimensions, default 4 5] [--seeds 0 1 2]
"""
import json
import sys
from pathlib import Path

import numpy as np

import msot.multiscale
import msot.shielding
from msot.cost import SquaredEuclidean
from msot.hierarchy import build_pair
from msot.higher_dimensions import _facets, make_problem
from msot.multiscale import solve_multiscale
from msot.solvers.cplex import Cplex
from candidate_sets import environment
from pruning_dimensions import CHUNKS, KINDS, N

SHARES = np.round(np.arange(0.1, 1.01, 0.1), 1)
CENTRES = ("mean of t(x_s)", "t(x_A)")
RESULTS = Path(__file__).parent / "results" / "pruning_recall"


def rank_within(owner, index_pointer, distance):
    """Rank of every entry among the entries of its x_A, by increasing distance."""
    order = np.lexsort((distance, owner))
    rank = np.empty(len(owner), int)
    rank[order] = np.arange(len(owner)) - np.repeat(index_pointer[:-1], np.diff(index_pointer))
    return rank


class Recorder:
    """Wraps shield; per round compares the closest ranking with the exact facets and accumulates per level."""

    def __init__(self, cost):
        self.cost, self.levels = cost, {}

    def __call__(self, pi, S, X, Yh, k, cost, tolerance=1e-12, chunk=4096):
        index_pointer, indices = S
        counts = np.diff(index_pointer)
        owner = np.repeat(np.arange(len(X)), counts)
        t = msot.shielding.extract_map(pi)
        Y = Yh.rep_points(k)
        normals, offsets = self.cost._face_data(X[owner], X[indices], Y[t[indices]], tolerance)
        facet = np.concatenate([_facets(normals[index_pointer[a]:index_pointer[a + 1]],
                                        offsets[index_pointer[a]:index_pointer[a + 1]]) for a in range(len(X))])
        level = self.levels.setdefault(k, {"rounds": 0, "candidates": 0, "facets": 0, "all kept": 0,
                                           **{centre: {"kept facets": np.zeros(len(SHARES), int), "inside": 0,
                                                       "full recall share": []} for centre in CENTRES}})
        level["rounds"] += 1
        level["candidates"] += len(indices)
        level["facets"] += int(facet.sum())
        level["all kept"] += int((np.bincount(owner, facet, len(X)) == counts).sum())
        mean = np.stack([np.bincount(owner, Y[t[indices]][:, i], len(X)) for i in range(X.shape[1])], axis=1)
        mean /= np.maximum(counts, 1)[:, None]
        for centre_name, centre in zip(CENTRES, (mean, Y[t])):
            distance = (offsets - np.einsum("ij,ij->i", normals, centre[owner])) / np.linalg.norm(normals, axis=1)
            rank = rank_within(owner, index_pointer, distance)
            record = level[centre_name]
            for i, share in enumerate(SHARES):
                record["kept facets"][i] += int((facet & (rank < np.ceil(share * counts[owner]))).sum())
            record["inside"] += int((np.bincount(owner, distance < -tolerance, len(X)) == 0).sum())
            farthest = np.zeros(len(X))
            np.maximum.at(farthest, owner[facet], rank[facet] + 1)
            record["full recall share"].append(farthest / np.maximum(counts, 1))
        return shield(pi, S, X, Yh, k, cost, tolerance, chunk)


shield = msot.shielding.shield


def summary(levels):
    result = []
    for k in sorted(levels, reverse=True):
        level = levels[k]
        points = sum(len(part) for part in level[CENTRES[0]]["full recall share"])
        entry = {"k": k, "rounds": level["rounds"], "candidates": level["candidates"], "facets": level["facets"],
                 "x_A with all candidates facets": level["all kept"], "x_A rounds": points}
        for centre in CENTRES:
            record = level[centre]
            full = np.concatenate(record["full recall share"])
            entry[centre] = {"recall per share": dict(zip(SHARES.tolist(), (record["kept facets"] / level["facets"]).tolist())),
                             "centre inside U(x_A)": record["inside"] / points,
                             "share for full recall": {"median": float(np.median(full)),
                                                       "90 %": float(np.quantile(full, 0.9)),
                                                       "mean": float(full.mean())}}
        result.append(entry)
    return result


def main():
    arguments = sys.argv[1:] + ["--seeds"]
    split = arguments.index("--seeds")
    dimensions = [int(argument) for argument in arguments[:split]] or [4, 5]
    seeds = [int(argument) for argument in arguments[split + 1:] if argument != "--seeds"] or [0]
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "recall.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"environment": environment(), "runs": []}
    runs = [run for run in stored["runs"] if (run["dimension"], run["seed"]) not in
            {(dimension, seed) for dimension in dimensions for seed in seeds}]
    cost = SquaredEuclidean()
    for dimension in dimensions:
        for kind in KINDS:
            for seed in seeds:
                X, mu, Y, nu = make_problem(N, kind, np.random.default_rng(seed), dimension)
                HX, HY = build_pair(X, mu, Y, nu)
                recorder = Recorder(cost)
                msot.multiscale.shield = lambda *a, **b: recorder(*a, **b, chunk=CHUNKS[dimension])
                solve_multiscale(HX, HY, cost, Cplex(), incremental=False)
                run = {"dimension": dimension, "kind": kind, "seed": seed, "levels": summary(recorder.levels),
                       "commit": environment()["commit"]}
                runs.append(run)
                stored["runs"] = runs
                path.write_text(json.dumps(stored, indent=1), encoding="utf-8")
                for level in run["levels"]:
                    print(f"d = {dimension} {kind:9s} seed {seed} k = {level['k']}: facets "
                          f"{100 * level['facets'] / level['candidates']:4.1f} %  " + "  ".join(
                              f"{centre}: recall at 0.6 {100 * level[centre]['recall per share'][0.6]:5.1f} %, "
                              f"inside {100 * level[centre]['centre inside U(x_A)']:5.1f} %, "
                              f"full recall median {level[centre]['share for full recall']['median']:.2f}"
                              for centre in CENTRES), flush=True)
    msot.multiscale.shield = shield


if __name__ == "__main__":
    main()

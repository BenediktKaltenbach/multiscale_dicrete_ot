"""Thesis experiment: how much harder discontinuous masks make the problems, Delaunay 1-ring, at 128 x 128.

Setup in docs/experiment_masks.md. For each point kind and seed the problem of candidate_sets.py is solved as it is
and with masks: mu and nu each lose their mass (down to the floor) on an edge strip and a disc. The masks are drawn
once (MASK_SEED), differ between mu and nu and are the same in all 9 problems. Order alternated per seed.
Writes experiments/results/masks/runs.json; plot_masks.py makes the figures.

python experiments/masks.py
"""
import json
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np

from msot.hierarchy import build_pair
from msot.problems import gaussian_masses, quantize, sample_points
from candidate_sets import environment, solve

N, KINDS, SEEDS = 128 * 128, ("uniform", "clustered", "grid"), (0, 1, 2)
MASK_SEED, STRIP_DEPTH, MAXIMAL_TILT, DISC_RADIUS, FLOOR, ATOMS = 2026, 0.1, np.radians(20), 0.06, 1e-3, 10**9
RESULTS = Path(__file__).parent / "results" / "masks"


def draw_mask(rng):
    """An edge strip {p : <normal, p> > offset} of depth STRIP_DEPTH at the middle of a random edge, tilted by up to
    MAXIMAL_TILT, and a disc of radius DISC_RADIUS with its centre uniform in [0.15, 0.85]^2."""
    edge = int(rng.integers(4))
    angle = edge * np.pi / 2 + rng.uniform(-MAXIMAL_TILT, MAXIMAL_TILT)  # outward normal of the strip
    normal = np.array([np.cos(angle), np.sin(angle)])
    middle = 0.5 + 0.5 * np.array([np.cos(edge * np.pi / 2), np.sin(edge * np.pi / 2)])
    inner = middle - STRIP_DEPTH * np.array([np.cos(edge * np.pi / 2), np.sin(edge * np.pi / 2)])
    return {"edge": edge, "normal": normal.tolist(), "offset": float(normal @ inner),
            "disc centre": rng.uniform(0.15, 0.85, 2).tolist(), "disc radius": DISC_RADIUS}


def inside(points, mask):
    strip = points @ np.array(mask["normal"]) > mask["offset"]
    disc = np.linalg.norm(points - mask["disc centre"], axis=1) < mask["disc radius"]
    return strip | disc


def apply_mask(mass, points, mask):
    """mass = density + FLOOR * mean(density) (gaussian_masses); inside the mask only the floor is kept."""
    floor = FLOOR * mass.mean() / (1 + FLOOR)
    return np.where(inside(points, mask), floor, mass)


MASKS = dict(zip(("mu", "nu"), (draw_mask(rng) for rng in [np.random.default_rng(MASK_SEED)] * 2)))


def problems(kind, seed):
    """The problem of make_problem(N, kind, default_rng(seed)) and its masked version."""
    rng = np.random.default_rng(seed)
    X, Y = sample_points(N, kind, rng), sample_points(N, kind, rng)
    mu, nu = gaussian_masses(X, rng), gaussian_masses(Y, rng)
    plain = (X, quantize(mu, ATOMS), Y, quantize(nu, ATOMS))
    masked = (X, quantize(apply_mask(mu, X, MASKS["mu"]), ATOMS), Y, quantize(apply_mask(nu, Y, MASKS["nu"]), ATOMS))
    return {"unmasked": plain, "masked": masked}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs, commit = [], environment()["commit"]
    for kind in KINDS:
        for seed in SEEDS:
            variants = problems(kind, seed)
            for name in (("unmasked", "masked") if seed % 2 == 0 else ("masked", "unmasked")):
                X, mu, Y, nu = variants[name]
                start = perf_counter()
                HX, HY = build_pair(X, mu, Y, nu)
                hierarchy_time = perf_counter() - start
                result = solve(HX, HY, X, Y, mu, None)
                masked_points = [int(inside(X, MASKS["mu"]).sum()), int(inside(Y, MASKS["nu"]).sum())]
                _, plain_mu, _, plain_nu = variants["unmasked"]
                removed_mass = [float(plain_mu[inside(X, MASKS["mu"])].sum() / plain_mu.sum()),
                                float(plain_nu[inside(Y, MASKS["nu"])].sum() / plain_nu.sum())]
                result.update({"kind": kind, "seed": seed, "problem": name, "hierarchy time": hierarchy_time,
                               "K": HX.K, "cells per level": [HX.number_of_cells(k) for k in range(HX.K + 1)],
                               "points in the masks of X and Y": masked_points,
                               "share of the unmasked mass in the masks of X and Y": removed_mass, "commit": commit,
                               "date": date.today().isoformat()})
                runs.append(result)
                level0 = result["levels"][-1]
                print(f"{kind:9s} seed {seed} {name:8s}: total {result['total time']:6.1f} s, "
                      f"level-0 iterations {len(level0['history'])}, |N| / |X| {level0['history'][-1]['N'] / N:.2f}, "
                      f"dual violation {result['dual violation']:.1e}", flush=True)
    (RESULTS / "runs.json").write_text(json.dumps({"environment": environment(), "masks": MASKS, "runs": runs},
                                                  indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

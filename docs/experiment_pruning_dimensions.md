# Experiment: candidate pruning in 4 and 5 dimensions

Thesis experiment on branch `higher-dimensions`. Script `experiments/pruning_dimensions.py`, figures `experiments/plot_pruning_dimensions.py`.
Raw results in `experiments/results/pruning_dimensions/` (`runs.json`; not under version control).

## Question
In 2D the Delaunay 1-ring has 6 points and only 5–16 % of them are redundant, so pruning saves nothing (`docs/experiment_pruning.md`). A standalone check found 38 % (4D) and 42 % (5D) of the 1-ring redundant, with 36 and 73 candidates per point (`docs/rationale.md` §4b). Does pruning pay off in the full multiscale solver there, with an exact face test and with a cheap heuristic?

## Code
- d-dimensional code is contained in `src/msot/higher_dimensions.py`; the planar modules are unchanged except the mass centroid in `hierarchy._add_level`, which now uses all coordinates (identical in 2D). Tests: `tests/test_higher_dimensions.py` (facet test against one LP per half-space in 3D–5D; search equals brute force in 3D and 4D; pruned and unpruned shield give identical N in 4D and 5D for both face tests; multiscale equals the dense optimum in 3D and 4D with all three variants).
- Everything else as in `docs/experiment_candidate_sets.md` (machine, solver CPLEX network simplex with warm start and 1 thread, integer masses with 10^9 atoms, mesh hierarchy with n_dense = 256 and length cap 3, dilated N1, tolerance 1e-12, verification by the global dual violation).

## Problems (`higher_dimensions.make_problem(4096, kind, default_rng(seed), dimension)`)
- n = 4096 points in X and in Y in [0, 1]^d, d = 4 and 5; kinds `uniform` and `clustered` (no grid: 4096 is no 5th power); seeds 0, 1, 2: 12 problems. Generator order as in 2D: points of X, points of Y, masses mu, masses nu.
- `clustered`: 4 Gaussians with centres uniform in [0.15, 0.85]^d, standard deviations uniform in [0.03, 0.15] along the d axes of a random orthonormal frame (QR of a d x d Gaussian matrix, signs fixed by the diagonal of R), truncated to [0, 1]^d by rejection.
- Masses: 4 Gaussian bumps with eigenvalues exp(uniform(log 1.8, log 100)) times spacing^2, spacing = n^(-1/d) (0.125 in 4D, 0.189 in 5D), random orthonormal frame, centre uniform in [0, 1]^d, weight uniform in [0, 1], normalised by sqrt((2 pi)^d det); plus 1e-3 times the mean density. No masks. Quantised to 10^9 atoms.
- Hierarchy (Delaunay in d dimensions, Qhull): 4096, about 1050, 270, 70, 1 cells (K = 4); 1-ring at level 0 about 35 points (4D) and 73 points (5D).

## Variable: pruning
- `off`: `SquaredEuclidean()`, all candidates in the search.
- `exact`: `SquaredEuclideanPruning("exact")`: per x_A and round, the half-spaces {y : <x_s - x_A, y - t(x_s)> <= tolerance / 2} of all candidates (5.5); their Chebyshev centre c by one LP (scipy `linprog`, HiGHS); the candidates whose dual point (x_s - x_A) / (b_s - <x_s - x_A, c>) is a vertex of the convex hull of the dual points and the origin (Qhull) are the facets and are kept. If P has no finite inscribed ball or the hull is degenerate, all candidates are kept. One Python call per x_A.
- `closest`: `SquaredEuclideanPruning("closest", keep_share=0.6)`: per x_A, keep the ceil(0.6 |S(x_A)|) half-spaces with the smallest signed distance to the mean of the partners t(x_s), an estimate of t(x_A); vectorised over all x_A. 0.6 is about the facet share measured in 4D. May drop needed candidates.
- In both pruned variants the search uses only the kept candidates, step ii keeps all, and `filter_missed` checks the missed cells exactly against the dropped candidates, so N, the rounds and the result equal the run without pruning.
- Shield chunk: 1024 points x_A per chunk in 4D, 256 in 5D (set in the script; 1024 in 5D needs more than 15 GB). Same for all variants of a dimension.
- Order per problem rotated over the seeds: seed 0 off, exact, closest; seed 1 exact, closest, off; seed 2 closest, off, exact.
- Recorded additionally: per face-test call the level size, candidates and kept ones; per `filter_missed` call the missed pairs and how many it removed.

## Pilot (2026-11-13, uniform seed 0, mains power; not part of the results)
| d | off | closest | exact | rounds per level | \|N\| / \|X\| at level 0 | peak memory (off) |
|---|---|---|---|---|---|---|
| 4 (chunk 4096) | 86.9 s (shield 70.0) | 59.5 s (42.7) | 105.9 s (89.3) | 1, 3, 4, 4 | 33.6 | 4.9 GB |
| 5 (chunk 256) | 312.0 s (shield 253.9) | 220.2 s (152.9) | 423.7 s (296.4) | 1, 3, 4, 4 | 68.8 | 1.9 GB |

Identical cost and rounds in all variants. The exact run in 5D also had twice the solve time (125.5 s vs. 56.5 s) for the identical sequence of local problems, so timing noise in long runs can be large.

## Results (2026-11-13, commit 810a373, mains power, 36 runs)
Figures in `experiments/results/pruning_dimensions/`: `pruning_dimensions_time`, `pruning_dimensions_ratio`, `pruning_dimensions_dropped`.

**Correctness.** In every problem the three variants give the same cost, the same support of pi (8191 = 2n - 1), the same rounds per level and the same |N| in every round. Dual violation at most 1.1e-16. With the exact face test `filter_missed` never removes a pair (no needed candidate is dropped); with the closest heuristic it removes the pairs the reduced search wrongly reported as missed.

**Problems.** Hierarchy 4096, about 1050, 270, 70, 1 cells in 4D and 4096, about 1035, 263, 67, 1 in 5D. 1-ring at level 0: 34.7 (4D uniform), 34.1 (4D clustered), 72.9 (5D uniform), 67.8 (5D clustered). |N| / |X| at level 0: 32–35 (4D), 66–70 (5D). Rounds per level from coarse to fine: 1, 3, 3–4, 4 (uniform), 1, 3–4, 4–5, 5–7 (clustered).

**Times** (mean of 3 seeds, range in brackets; shield and solve summed over all rounds):

| d | kind | pruning | total [s] | shield [s] | solve [s] | shield / off | total / off |
|---|---|---|---|---|---|---|---|
| 4 | uniform | off | 107.1 (104.0–111.7) | 79.0 | 27.3 | 1 | 1 |
| 4 | uniform | exact | 161.0 (151.1–172.8) | 131.7 | 28.5 | 1.67 (1.61–1.69) | 1.50 |
| 4 | uniform | closest | 84.2 (78.5–88.8) | 54.5 | 28.8 | 0.69 (0.67–0.72) | 0.79 |
| 4 | clustered | off | 126.4 (115.4–133.9) | 101.2 | 24.5 | 1 | 1 |
| 4 | clustered | exact | 194.3 (177.6–213.0) | 169.1 | 24.5 | 1.67 (1.58–1.73) | 1.54 |
| 4 | clustered | closest | 99.0 (84.5–116.3) | 71.1 | 27.1 | 0.70 (0.66–0.78) | 0.78 |
| 5 | uniform | off | 428.9 (402.8–457.4) | 322.6 | 104.6 | 1 | 1 |
| 5 | uniform | exact | 369.7 (286.8–414.9) | 289.6 | 78.4 | 0.89 (0.76–0.97) | 0.86 |
| 5 | uniform | closest | 294.2 (283.0–306.3) | 202.0 | 90.4 | 0.63 (0.62–0.64) | 0.69 |
| 5 | clustered | off | 322.1 (308.4–333.3) | 267.8 | 52.8 | 1 | 1 |
| 5 | clustered | exact | 286.7 (265.7–305.9) | 235.6 | 49.4 | 0.88 (0.85–0.92) | 0.89 |
| 5 | clustered | closest | 216.2 (206.4–223.6) | 165.9 | 48.7 | 0.62 (0.62–0.62) | 0.67 |

Hierarchy (not in the totals): 2.0 s / 2.2 s in 4D, 16.6 s / 10.3 s in 5D (uniform / clustered).

**Dropped candidates and filter.**

| d | kind | exact: no facet | closest: dropped | closest: missed pairs removed by the filter |
|---|---|---|---|---|
| 4 | uniform | 30.8 % | 38.8 % | 22 102 of 625 806 (3.5 %) |
| 4 | clustered | 39.2 % | 38.8 % | 172 052 of 1 108 682 (15.5 %) |
| 5 | uniform | 29.4 % | 39.4 % | 5 299 of 757 006 (0.7 %) |
| 5 | clustered | 37.9 % | 39.4 % | 72 029 of 1 106 726 (6.5 %) |

The share without facet grows towards the fine levels: 4–12 % at level 3, 13–25 % at level 2, 23–34 % at level 1, 31–41 % at level 0 (figure `pruning_dimensions_dropped`). Clustered points have more redundant half-spaces than uniform ones.

**Reading.**
- Pruning only pays off when the 1-ring is large. The exact face test (one LP and one Qhull call per x_A in Python) costs more than it saves at about 35 candidates in 4D (shield time 1.67x) and pays off a little at about 70 candidates in 5D (0.88x).
- The vectorised closest heuristic is faster in all 12 problems: shield time 0.62–0.70x, total time 0.67–0.79x, a little more in 5D than in 4D. Its wrong drops cost only extra filter work (filter_missed stays exact); they are more frequent for clustered points (6.5–15.5 % of the missed pairs) than for uniform ones (0.7–3.5 %).
- Solve times differ between variants for identical sequences of local problems (5D uniform: 60–119 s), so solve-time differences are timing noise. In 5D uniform seed 2 the exact run was fast in both solve (60.5 s) and shield (224.7 s vs. 318–326 s in the other seeds); the run order cannot explain this (exact was third there), so the 0.76 looks like an outlier and the mean ratio 0.89 is slightly optimistic.

## Addendum: recall of the closest rule (2026-11-13, seed 0, no timing)
Script `experiments/pruning_recall.py`, raw results `experiments/results/pruning_recall/recall.json`. Each problem is solved once without pruning; in every shield round the 1-ring half-spaces of every x_A are ranked by signed distance to a centre c and compared with the exact facets. Two centres: c = mean of the t(x_s) (the implemented rule) and c = t(x_A) = argmax_y pi(x_A, y). Recall = kept facets / all facets. A random choice of the same size has recall about equal to the keep share; a perfect ranking has min(1, share / facet share).

Recall at keep share 0.6, level 0 (finest):

| d | kind | facets | random | closest, mean of t(x_s) | closest, t(x_A) | perfect | c inside P: mean / t(x_A) | share for all facets (median) |
|---|---|---|---|---|---|---|---|---|
| 4 | uniform | 67.9 % | 60 % | 76.6 % | 74.3 % | 88 % | 85 % / 99 % | 0.89 / 0.91 |
| 4 | clustered | 61.1 % | 60 % | 79.3 % | 78.5 % | 98 % | 75 % / 98 % | 0.86 / 0.86 |
| 5 | uniform | 68.4 % | 60 % | 74.4 % | 72.4 % | 88 % | 83 % / 100 % | 0.94 / 0.95 |
| 5 | clustered | 61.4 % | 60 % | 76.5 % | 77.3 % | 98 % | 69 % / 98 % | 0.91 / 0.89 |

- The ranking gets about half of the way from a random choice to a perfect one at level 0. At the coarse levels nearly all candidates are facets (90–97 % at k = 3), so any choice of 60 % drops about a third of the facets there (recall 62–66 %).
- t(x_A) lies inside P in 96–100 % of the cases (as cyclical monotonicity suggests), the mean of the t(x_s) only in 27–85 %. The recall is nevertheless the same (within 2 points): the distance to the boundary carries little information about whether a half-space is a facet. In d dimensions all hyperplanes pass through partners t(x_s) close to t(x_A); whether one is a facet depends mainly on its direction relative to the others.
- Recall rises smoothly with the share (at level 0 from 41–47 % at 0.3 to 95–98 % at 0.9), with no knee. For at least 10 % of the x_A the last-ranked half-space is a facet, so no share below 1 keeps all facets; exactness always comes from `filter_missed`.
- Consequence: 0.6 is not singled out by the geometry. It is a tuning parameter that trades search work against filter work; only timing can choose it. Using t(x_A) instead of the mean would not change the result, so the rule stays as implemented.

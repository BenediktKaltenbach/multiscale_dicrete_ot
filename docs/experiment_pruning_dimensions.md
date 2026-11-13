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

## Results
(to be filled in after the run)

# Experiment: candidate sets S(x_A), Delaunay 1-ring vs. 8 sector neighbours

Thesis experiment, run 2026-11-12. Script `experiments/candidate_sets.py`, figures `experiments/plot_candidate_sets.py`.
Raw results in `experiments/results/candidate_sets/` (`runs.json`, `q_level0.npz`, figures as PDF and 600 ppi PNG; the results folder is not under version control).

## Question
Does the choice of the candidate set S(x_A) for the shielding search (Schmitzer 2016, Alg. 5.1, Prop. 5.3) change the run time or the work of the multiscale solver on unstructured point sets? Compared: the Delaunay 1-ring of the level's mesh (default) and the nearest point in each of 8 angular sectors.

## Code and machine
- Repository `multiscale_dicrete_ot`, branch `mtw-pruning`, commit `a5628c8` (recorded automatically in `runs.json` together with a flag for uncommitted changes in `src/`). The squared Euclidean code path is identical to `main` at `c1428bf`.
- Laptop: AMD Ryzen 7 PRO 6850U (8 cores, 16 threads, 2.7 GHz base), 14.8 GB RAM, Windows 11 Pro 10.0.26200, mains power, idle sleep blocked during the runs.
- Python 3.14.2, numpy 2.4.1, scipy 1.17.0, CPLEX 22.2.0.1 (Python API, IBM academic licence).
- All runs sequential in one process; CPLEX restricted to 1 thread; numpy/scipy with their default threading.

## Problems (`problems.make_problem`)
- n = 16384 points in X and in Y (128 x 128), point kinds `uniform`, `clustered`, `grid`, seeds 0, 1, 2: 9 problems.
- One generator per problem, `numpy.random.default_rng(seed)`, used in this order: points of X, points of Y, masses mu, masses nu. X and Y are independent samples of the same kind.
- Point kinds (`problems.sample_points`), all in the unit square:
  - `uniform`: i.i.d. uniform on [0, 1]^2.
  - `clustered`: mixture of 4 Gaussians, equal weights; centres uniform in [0.15, 0.85]^2; per Gaussian the two standard deviations are uniform in [0.03, 0.15] and the axes are rotated by a uniform angle in [0, pi); samples outside [0, 1]^2 are rejected and redrawn.
  - `grid`: the 128 x 128 cell centres (i + 0.5) / 128, i = 0..127, identical for X and Y (only the masses differ); no mask.
- Masses (`problems.gaussian_masses`, same for all kinds): sum of 4 Gaussian bumps, each with a centre uniform in [0, 1]^2, a covariance with eigenvalues exp(uniform(log 1.8, log 100)) times spacing^2 (spacing = 1 / sqrt(n) = 1/128) in a uniformly rotated frame, and a weight uniform in [0, 1]; plus a floor of 1e-3 times the mean density, so every point has positive mass.
- Integer masses (Schmitzer, Remark 6.2; `problems.quantize`): mu and nu are rounded to integer numbers of mass atoms with exactly 10^9 atoms in total each, at least one atom per point (the surplus or deficit from rounding is put on the largest mass).

## Hierarchy (`hierarchy.build_pair`, mesh hierarchy, option B)
- Built once per problem and used by both candidate sets.
- Level 0: every point is a cell, rep = point, rad = 0, mesh = Delaunay triangulation (`scipy.spatial.Delaunay`, Qhull; on the grid Qhull chooses one diagonal per square).
- Coarsening step: mesh edges sorted by rep distance; edge length cap 3 (an edge (a, b) is usable only if its length is at most 3 times the larger of the two nearest-rep distances); two rounds of greedy matching (pairs, then groups of up to 4); every cell left single is attached to the adjacent group with the shortest edge that has at most 4 cells (so at most 5 children). X and Y are coarsened in lockstep until both have at most n_dense = 256 cells; then one top cell is added, so both have the same depth K.
- New cell: mass = sum of the children's masses; rep = the child rep nearest to the mass centroid of the cell's points (reps are existing points); rad = exact maximum distance from the rep to the cell's points; mesh = Delaunay of the new reps.
- K and the number of cells per level are recorded per problem in `runs.json`.

## Solver (`multiscale.solve_multiscale`, Alg. 4.6)
- Cost squared Euclidean c(x, y) = |x - y|^2 (`cost.SquaredEuclidean(prune=False)`): **pruning off**, every candidate is used in the search.
- Start: trivial coupling at level K (one cell times one cell).
- Initial neighbourhood of level k: N1 = children(x) x children(y) for all (x, y) in spt pi of level k+1, dilated by the 1-ring of the fine Y mesh (`dilation=True`, `multiscale.dilate`).
- Alg. 4.1 (`solve_sparse`) per level: solve the local problem on N, then shield (Alg. 5.1), repeat. Stop when the new N is contained in the old N or the cost no longer decreases (C >= C_old - 1e-12 |C_old|); at most 100 rounds (never reached).
- Shielding (`shielding.shield`): t(x) = argmax_y pi(x, y) (step i); entries spt pi and (x_A, t(x_s)) for every candidate x_s in S(x_A) (step ii); the missed cells of the level-synchronous search (Alg. 5.2 over the Y hierarchy, step iii), with psi_hat(x_A, x_s, cell) = psi(x_A, x_s, rep) - 2 |x_s - x_A| rad and the strict test psi_hat > psi(x_A, x_s, t(x_s)) + 1e-12; search in chunks of 4096 points x_A.
- Local solver (`solvers.cplex.Cplex`): CPLEX network optimizer (network simplex) on the pure network LP (one arc x -> y per entry of N); optimality tolerance 1e-9 (network and simplex); 1 thread; warm start within a level (the next round adds the new entries of N as columns and deletes only the nonbasic columns outside N, and CPLEX continues from the current basis); a new model per level. Basic solutions are integral (total unimodularity), so the values are rounded and the marginals checked exactly; spt pi = all basic columns, including degenerate zeros. Duals: alpha = row duals of X, beta = minus the row duals of Y.

## Candidate sets (the variable of the experiment)
- `1-ring`: S(x_A) = the neighbours of x_A in the Delaunay mesh of the level (`HX.neighbours[k]`, computed while building the hierarchy).
- `8 sectors`: S(x_A) = in each of 8 equal angular sectors around x_A (first sector starting at angle 0), the nearest point among the 32 nearest points of x_A (`mesh.sector_neighbours(points, 8)`, k-d tree); sectors with none of the 32 nearest stay empty (at the boundary). Computed per level inside the timed run.
- Both are computed on the reps of X_k at every level k.

## Measurements
- Total time: wall clock (`time.perf_counter`) of `solve_multiscale`, from the trivial coupling to the final pi at level 0; includes computing the sector neighbours, excludes problem generation, hierarchy construction (recorded separately as "hierarchy time") and the verification below.
- Solve time and shield time: sums over all rounds of all levels of the time in `solver.solve_local` and in `shield`.
- Per level and round (`levels[...]["history"]`): cost C of pi, |N| (number of entries of the local problem's N; the CPLEX model can additionally hold basic columns from the previous round), solve and shield time; per level: number of cells, |N1|.
- "Iterations at level 0": rounds of Alg. 4.1 at the finest level. "|N| / |X| at level 0": size of the shielding neighbourhood N of the last round at level 0 (the number of variables of the local problem) divided by n; not to be confused with the candidate set S(x_A), which has 6.0 (1-ring) or 7.8 (8 sectors) points on average at level 0.
- Cone constant q(x_A) (5.6) at level 0 for both candidate sets: cos of half the largest angular gap between the directions x_s - x_A (`mesh.cone_q`); q <= 0 means P(x_A) is unbounded (boundary points).

## Run protocol and verification
- Per problem both candidate sets run back to back on the same hierarchy; order 1-ring first for seeds 0 and 2, 8 sectors first for seed 1, to spread drift (thermal state, background load) over both variants.
- Every run: global dual feasibility max over all of X x Y of alpha + beta - c (Cor. 3.10, blockwise over 512 rows of X) is recorded; it must be <= 1e-9 for an optimal pi. Both candidate sets must reach the same optimal cost per unit mass.

## Results
Run 2026-11-12 at commit `a5628c8`, no uncommitted changes in `src/`. All 9 problems have K = 5 (e.g. uniform seed 0: 16384, 4586, 1272, 361, 97, 1 cells of X per level); building the hierarchy pair takes 0.3 s (0.5 s on the grid).

**Correctness.** In all 18 runs the dual violation max(alpha + beta - c) over X x Y is at most 8.9e-10 (exactly 0 on the grid). Per problem both candidate sets give the same optimal cost (relative difference at most 8e-16).

Means over the 3 seeds (single seeds in brackets; times in s):

| kind | S | total time | solve | shield | iterations at level 0 | all iterations | \|N\| / \|X\| at level 0 |
|---|---|---|---|---|---|---|---|
| grid | 1-ring | 64.0 (46.2, 48.2, 97.5) | 19.7 | 43.1 | 10, 9, 23 | 32, 31, 47 | 9.8, 11.8, 13.2 |
| grid | 8 sectors | 79.2 (52.7, 51.7, 133.3) | 17.7 | 60.5 | 10, 8, 32 | 31, 31, 56 | 8.3, 8.8, 8.8 |
| uniform | 1-ring | 56.9 (63.2, 45.7, 61.9) | 21.8 | 34.2 | 8, 8, 9 | 35, 28, 38 | 14.8, 10.8, 9.7 |
| uniform | 8 sectors | 57.7 (62.2, 51.4, 59.7) | 19.9 | 36.8 | 8, 8, 8 | 37, 28, 30 | 12.6, 9.6, 9.3 |
| clustered | 1-ring | 59.4 (66.2, 51.0, 61.1) | 23.4 | 35.3 | 12, 11, 9 | 38, 33, 31 | 11.2, 6.9, 10.7 |
| clustered | 8 sectors | 60.5 (58.0, 57.2, 66.2) | 22.2 | 37.4 | 9, 9, 10 | 33, 32, 31 | 12.0, 7.6, 10.6 |

Cone constant q at level 0, all seeds pooled:

| kind | S | 5 % quantile | median | share with q <= 0 |
|---|---|---|---|---|
| grid | 1-ring | 0.71 | 0.71 | 0.02 % |
| grid | 8 sectors | 0.92 | 0.92 | 0.02 % |
| uniform | 1-ring | 0.35 | 0.64 | 0.15 % |
| uniform | 8 sectors | 0.57 | 0.80 | 0.93 % |
| clustered | 1-ring | 0.38 | 0.64 | 0.12 % |
| clustered | 8 sectors | 0.61 | 0.80 | 0.39 % |

On the grid the boundary points (about 3 %) have q = 0 up to rounding (a 180° gap), which is why the share with q <= 0 is so small there.

**Findings.**
- Run time is a tie on unstructured points: the 8 sectors take +1.4 % (uniform) and +1.9 % (clustered) on average, well within the spread between seeds (up to 38 %). On the grid the 8 sectors are slower in every seed (+14 %, +7 %, +37 %).
- The sector sets have the clearly better cone constant (median 0.80 vs. 0.64 on random points, 0.92 vs. 0.71 on the grid), and accordingly mostly the smaller N at level 0 (uniform -11 %, grid -25 %, clustered +5 %) and a slightly shorter solve time (-5 to -10 %). But the search costs more: S has 7.8 candidates per point instead of 6.0 at level 0 (7.9 vs. 5.9 on the grid), and every visited cell is tested against every candidate, so the shield time grows by 6–8 % on random points and by 40 % on the grid. The two effects cancel on random points, and the search cost wins on the grid.
- The empty outward sectors at the boundary give the sector sets more points with q <= 0 (P unbounded) on random points (0.9 % vs. 0.15 % uniform).
- Grid seed 2 is the hardest problem for both candidate sets (23 and 32 rounds at level 0); it dominates the grid means. The other two grid seeds behave like the random point sets.
- Conclusion: the better cone constant of the sector sets does not translate into a faster solver; the Delaunay 1-ring stays the default (it comes with the hierarchy for free and is never slower on average).

Figures (in `experiments/results/candidate_sets/`): `candidate_sets_runtime` (total time), `candidate_sets_level0` (iterations and |N| / |X| at level 0), `candidate_sets_q` (cumulative distribution of q at level 0).

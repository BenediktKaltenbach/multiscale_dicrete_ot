# Architecture

Sparse multiscale discrete OT with shielding neighbourhoods (Schmitzer 2016) on unstructured planar meshes. First the squared Euclidean cost, later MTW-type costs.

## Conventions
- Python, numpy/scipy, CPLEX (network optimizer) as the LP solver. Nimble, readable code, few comments.
- Notation and naming follow Schmitzer 2016 (equation and algorithm numbers given in brackets below).
- Only simplex-type methods (interior-point solutions are not sparse). Masses are integer mass atoms (Remark 6.2), so every local solution meets its marginals exactly; costs and the solver stay floating point.
- Every design step is agreed with Benedikt before implementation.

## Modules
| File | Contents |
|---|---|
| `hierarchy.py` | Hierarchical partition: per level k arrays `rep[k]`, `rad[k]`, `parent[k]`, `mass[k]`, `cell_of[k]`, `neighbours[k]`; `children` derived. Builders: `agglomerate` / `build_pair` (mesh hierarchy, option B, default) and `quadtree` / `build_pair_quadtree` (Def. 4.4, option A, baseline); see `docs/hierarchy.md` |
| `mesh.py` | Delaunay 1-ring (`one_ring`), sector neighbours (`sector_neighbours`), cone constant q (5.6) and D per vertex (`cone_q`, `candidate_statistics`) |
| `cost.py` | Cost classes with `c`, `psi` (5.1), `psi_hat` (5.3); `SquaredEuclidean` uses (5.7) and has the optional candidate pruning (`needed_candidates_all`). Coarse cost c_k(x, y) = c(rep x, rep y) (4.2) |
| `shielding.py` | map extraction t (step-i), candidate sets S(x_A), `shield` (Alg. 5.1, vectorised over all x_A in chunks of 4096), `search_all` (Alg. 5.2 for many x_A at once, on (x_A, cell) pairs level by level), `search_tree` (Alg. 5.2 for one x_A, reference for tests) |
| `solvers/` | `base.py` interface and `solve_dense` (dense reference with any solver), `cplex.py` (CPLEX network optimizer, the only backend) |
| `multiscale.py` | `solve_sparse` (Alg. 4.1), `solve_multiscale` (Alg. 4.6, with `dilate` of the initial N1 by default), `refine`, `dilate` |
| `verify.py` | global dual feasibility alpha(x) + beta(y) <= c(x, y) on X x Y; brute-force check of Def. 3.5 for small problems |
| `problems.py` | reproducible test problems (X, mu, Y, nu), see `docs/problems.md` |

## Project layout
```
pyproject.toml        # pip install -e .
src/msot/             # the modules above
tests/                # unittest (python -m unittest), invariants and comparison with the dense optimum (solve_dense)
experiments/          # fixed seeds, results and plots in experiments/results/ (not in git): plot_hierarchy.py and
                      # hierarchy_measures.py (hierarchy checks), iteration_analysis.py (convergence of Alg. 4.1),
                      # large_run.py (full run at any size with the global dual certificate)
notebooks/            # optional, visual exploration only; imports msot, no algorithm code
```

## Interfaces
- `solve_local(rows, columns, c, mu, nu, state=None) -> (pi, alpha, beta, state)`. Here (rows, columns) are the entries of N and c their costs, pi is a scipy COO matrix, and state is opaque, backend-specific warm-start data.
- `shield(pi, S, X, Yh, k, cost, tolerance) -> (rows, columns)`, X the points of X_k.
- `solve_multiscale(HX, HY, cost, solver, tolerance, max_iterations, candidates=None, dilation=True) -> (pi, alpha, beta, info)`; `info` holds per level the cells, |N1| and the history of `solve_sparse` (C, |N|, solve and shield time per iteration).
- `solve_dense(X, Y, mu, nu, cost, solver)`: the dense problem with the same solver, as reference.

## Invariants and tolerances
- Def. 4.3: the coarsest level is a single cell, and X and Y have the same depth K (pad with copy levels if needed).
- (4.1): |rep(x) - x| <= rad(x) for all leaves x of a cell; rad is computed over the leaves.
- rep is an existing point: the child rep closest to the cell's mass centroid. This keeps reps nested and always inside the domain (needed for MTW costs, which are only defined on Omega x Lambda).
- Shielded only if psi_hat > psi(y_s) + tol. spt pi = all basic entries of the local solution, including degenerate zeros (they satisfy (2.4)); this keeps spt pi a spanning tree. Alg. 4.1 stops when N no longer grows or the cost no longer decreases (relative 1e-12); max_iterations = 100 is only a safety net (at most 47 rounds observed).
- mu and nu are integer mass atoms with exactly equal totals (default 10^9, `problems.make_problem`); float masses (`atoms=None`) still work, but the solver then only meets the marginals up to its feasibility tolerance, and refined N1 can become infeasible.
- The same Y hierarchy is used for multiscale solving and for the search (`search_all`): at scale k the leaves of the search are the cells of Y_k.

## Initial neighbourhood per level
- N1 = refine(spt pi), i.e. children(x) x children(y) over spt pi of the coarser level (Alg. 4.6), then dilated by one ring of the fine Y mesh: N1 <- N1 plus (x, y') for every (x, y) in N1 and y' in `HY.neighbours[k][y]` (`multiscale.dilate`, `solve_multiscale(..., dilation=True)`, default). `dilation=False` gives Schmitzer's N1. Only the start of Alg. 4.1 changes; its loop and the certificate (Cor. 3.10) are unchanged. Cost-independent (uses only the Y mesh), so it carries over to MTW costs.

## Map t and candidate sets
- t(x) = argmax_y pi(x, y).
- S(x_A): Delaunay 1-ring of each level's mesh (`H.neighbours[k]`, default of `solve_multiscale`). Option: sector neighbours, `mesh.sector_neighbours(points, sectors)`: the nearest point in each of `sectors` equal angular sectors among the 32 nearest points (filled sectors give q >= cos(2 pi / sectors)). Passed as `solve_multiscale(..., candidates=lambda H, k: ...)`. Comparison in `candidate_sets.py` (script removed, numbers kept in rationale.md), reasons in `rationale.md`. Possible expansion: adaptive 2-ring where 0 < q < q_min (q_min about 0.5). Boundary vertices have q <= 0 (P is unbounded); this is accepted.
- Pruning (`SquaredEuclidean(prune=True)`, off by default): `needed_candidates` keeps only the candidates whose half-plane (5.5) is a face of the polytope P (Fig. 3; tight at two vertices); unbounded or empty P keeps all. `shield` skips the others in the search (Y_hat unchanged) but still adds the step-ii entries (x_A, t(xs)) of all candidates, since dropping them costs solve_sparse iterations.
- At coarse levels: the same construction on that level's reps (sector neighbours of `H.rep_points(k)`, or the 1-ring of that level's mesh).

## Solver notes
- POT `ot.emd` (optional dependency `reference`): independent dense check (network simplex, floats); the tests use `solve_dense` with CPLEX.
- CPLEX (`solvers/cplex.py`, dependency `cplex` + `docplex`; the full version needs the academic CPLEX Studio and `docplex config --upgrade <CPLEX_Studio path>`, the PyPI package alone is the Community Edition, limited to 1000 variables): network optimizer (network simplex, as Schmitzer). Each entry of N is an arc x -> y with +1 in the row of x and -1 in the row of y, so the whole LP is a pure network; alpha = dual of the x rows, beta = -(dual of the y rows). spt pi = all basic columns including degenerate zeros. One thread, optimality tolerance 1e-9.
- Integer masses are solved exactly: the transport matrix is totally unimodular, so every basic solution is integral; the values are rounded and the marginals checked exactly, and refined N1 stay exactly feasible on every level. Float masses are scaled to mean mass 1 (tolerances are absolute).
- Warm start (`Cplex(warm_start=True)`, default): one persistent model per `solve_sparse`; new entries of N are added as columns, only nonbasic columns outside N are deleted, and CPLEX continues from the current basis. The first solve of each level is cold.
- HiGHS (highspy) was used until 2026-10-17 and removed to keep the code small; its measurements are kept in `rationale.md`.

## Order of implementation (status 2026-10-17)
1. Done: point clouds, Delaunay, S, dense reference, dual-feasibility check.
2. Done: single-scale `solve_sparse` with a brute-force Y_hat, matching the dense optimum.
3. Done: mesh hierarchy (B) with length cap and leftover attachment, level-synchronous search, checked against the brute-force Y_hat.
4. Done: `solve_multiscale`, with the dilated initial neighbourhood.
5. Done: quadtree (A) as a baseline, compared with B (`compare_hierarchies.py` (script removed, numbers kept in rationale.md), `hierarchy_measures.py`, 256 x 256 runs).
6. Done: variants of S with q/D statistics (`candidate_sets.py` (script removed, numbers kept in rationale.md)): 1-ring default, 8 sector neighbours as an option, pruning off by default; possible expansion: adaptive 2-ring.
7. Done: warm start (HiGHS first, now CPLEX).
8. Next: MTW costs.

## Hierarchy (option B)
- Bottom-up from the given points: level 0 = points with their Delaunay mesh.
- One coarsening step: an edge length cap (length <= 3 x the larger nearest-rep spacing of its two ends), two rounds of greedy matching along the shortest allowed mesh edges, then leftover single cells join a neighbouring group (at most 5 children; on average 3.6 on random points, 4 on grids). Then compute rep, rad, mass and the Delaunay mesh of the new reps.
- Coarsen X and Y with the same number of rounds until both have at most n_dense = 256 cells; above that, add the trivial top level. Details in `docs/hierarchy.md`.

## MTW readiness
- Correctness of shield and the search (Alg. 5.2) holds for any cost; MTW only affects sparsity. The core therefore uses only `c`, `psi`, `psi_hat`.
- Everything Euclidean-specific (bound (5.7), half-plane fences, polygon geometry) lives in `SquaredEuclidean` or in experiments, never in shielding.py.
- MTW costs need their own `psi_hat`; the generic fallback is psi(rep y) - L * rad(y), with L bounding the y-Lipschitz constant of psi on the cell (e.g. via sup |D_xy c| * |x_A - x_s|).
- The c-exponential (charts) is an optional cost method, used for analysis and plots, not by the solver.

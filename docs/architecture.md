# Architecture

Sparse multiscale discrete OT with shielding neighbourhoods (Schmitzer 2016) on unstructured planar meshes. First the squared Euclidean cost, later MTW-type costs.

## Conventions
- Python, numpy/scipy. Nimble, readable code, few comments.
- Notation and naming follow Schmitzer 2016 (equation and algorithm numbers given in brackets below).
- Only floating-point solvers, and only simplex-type methods (interior-point solutions are not sparse).
- Every design step is agreed with Benedikt before implementation.

## Modules
| File | Contents |
|---|---|
| `hierarchy.py` | Hierarchical partition: per level k arrays `rep[k]`, `rad[k]`, `parent[k]`, `mass[k]`; `children` derived. Builders: `agglomerate` (mesh hierarchy, option B, first) and `quadtree` (Def. 4.4, option A, baseline) |
| `mesh.py` | Delaunay mesh per level, 1-ring adjacency, cone constant q (5.6) per vertex |
| `cost.py` | Cost classes with `c`, `psi` (5.1), `psi_hat` (5.3); `SquaredEuclidean` uses (5.7). Coarse cost c_k(x, y) = c(rep x, rep y) (4.2) |
| `shielding.py` | map extraction t (step-i), candidate sets S(x_A), `shield` (Alg. 5.1), `search_tree` (Alg. 5.2, level-synchronous) |
| `solvers/` | `base.py` interface, `highs.py` (highspy, default), `scipy_highs.py` (linprog, positive support only), later `pot_dense.py`, `cplex.py` |
| `multiscale.py` | `solve_sparse` (Alg. 4.1), `solve_multiscale` (Alg. 4.6) |
| `verify.py` | global dual feasibility alpha(x) + beta(y) <= c(x, y) on X x Y; brute-force check of Def. 3.5 for small problems |
| `problems.py` | reproducible test problems (X, mu, Y, nu), see `docs/problems.md` |

## Project layout
```
pyproject.toml        # pip install -e .
src/msot/             # the modules above
tests/                # unittest (python -m unittest), invariants and comparison with POT
experiments/          # scripts with fixed seeds; save results and plots
notebooks/            # optional, visual exploration only; imports msot, no algorithm code
```

## Interfaces
- `solve_local(rows, columns, c, mu, nu, state=None) -> (pi, alpha, beta, state)`. Here (rows, columns) are the entries of N and c their costs, pi is a scipy COO matrix, and state is opaque, backend-specific warm-start data.
- `shield(pi, S, X, Yh, k, cost, tolerance) -> (rows, columns)`, X the points of X_k.
## Invariants and tolerances
- Def. 4.3: the coarsest level is a single cell, and X and Y have the same depth K (pad with copy levels if needed).
- (4.1): |rep(x) - x| <= rad(x) for all leaves x of a cell; rad is computed over the leaves.
- rep is an existing point: the child rep closest to the cell's mass centroid. This keeps reps nested and always inside the domain (needed for MTW costs, which are only defined on Omega x Lambda).
- Shielded only if psi_hat > psi(y_s) + tol. spt pi = all basic entries of the local solution, including degenerate zeros (they satisfy (2.4)); this keeps spt pi a spanning tree, so refined N1 stay feasible in floating point. Alg. 4.1 stops when N no longer grows (or on a relative cost tolerance).
- mu and nu are integer mass atoms with exactly equal totals (default 10^9, `problems.make_problem`); float masses normalised to equal totals still work, but HiGHS then only meets the marginals up to its tolerance.
- The same Y hierarchy is used for multiscale solving and for search_tree: at scale k the leaves of the search are the cells of Y_k.

## Map t and candidate sets
- t(x) = argmax_y pi(x, y).
- S(x_A): Delaunay 1-ring of each level's mesh (`H.neighbours[k]`, default of `solve_multiscale`). Option: sector neighbours, `mesh.sector_neighbours(points, sectors)`: the nearest point in each of `sectors` equal angular sectors among the 32 nearest points (filled sectors give q >= cos(2 pi / sectors)). Passed as `solve_multiscale(..., candidates=lambda H, k: ...)`. Comparison in `experiments/candidate_sets.py`, reasons in `rationale.md`. Possible expansion: adaptive 2-ring where 0 < q < q_min (q_min about 0.5). Boundary vertices have q <= 0 (P is unbounded); this is accepted.
- Pruning (`SquaredEuclidean(prune=True)`, off by default): `needed_candidates` keeps only the candidates whose half-plane (5.5) is a face of the polytope P (Fig. 3; tight at two vertices); unbounded or empty P keeps all. `shield` skips the others in the search (Y_hat unchanged) but still adds the step-ii entries (x_A, t(xs)) of all candidates, since dropping them costs solve_sparse iterations.
- At coarse levels: the same construction on that level's reps (sector neighbours of `H.rep_points(k)`, or the 1-ring of that level's mesh).

## Solver notes
- POT `ot.emd`: dense reference (network simplex, floats).
- HiGHS (`highspy`, `solvers/highs.py`): integer masses are solved exactly. The transport matrix is totally unimodular, so every basic solution is integral; HiGHS values are rounded to integers (primal tolerance 1e-3 atoms) and the marginals are checked exactly. Refined N1 are then exactly feasible on every level. Float masses: scaled to mean mass 1 (HiGHS tolerances are absolute), primal tolerance 1e-10, since clipped negative basic values carry over to a refined N1; this failed on masked grids at n = 4096, hence the atoms. Dual tolerance 1e-9 in both cases.
- Warm start (step 7, `Highs(warm_start=True)`, default): one persistent model per `solve_sparse`; new entries of N are added as columns (`addCols`), only nonbasic columns outside N are deleted, then the primal simplex continues from the old basis. About 10x fewer simplex iterations, solve time at n = 4096 from 68-128 s to about 6-7 s. The first solve of each level is cold (dual simplex).

## Order of implementation (open for discussion)
1. Point clouds, Delaunay, S, dense POT reference, dual-feasibility check.
2. Single-scale `solve_sparse` with a brute-force Y_hat; must match POT.
3. Mesh hierarchy (B) + `search_tree`, checked against the brute-force Y_hat.
4. `solve_multiscale`.
5. Quadtree (A) as a baseline; compare A vs B.
6. Variants of S with q/D statistics: sector neighbours (6, 8) with pruning of redundant candidates; 1-ring as the reference; later possibly an adaptive 2-ring.
7. HiGHS warm start.
8. MTW costs.

## Hierarchy (option B)
- Bottom-up from the given points: level 0 = points with their Delaunay mesh.
- One coarsening step: two rounds of greedy matching along the shortest mesh edges (about 4 children per cell); unmatched cells become copy cells. Then compute rep, rad, mass and the Delaunay mesh of the new reps.
- Coarsen X and Y with the same number of rounds until a level is small enough for a dense solve; above that, add the trivial top level.

## MTW readiness
- Correctness of shield / search_tree holds for any cost; MTW only affects sparsity. The core therefore uses only `c`, `psi`, `psi_hat`.
- Everything Euclidean-specific (bound (5.7), half-plane fences, polygon geometry) lives in `SquaredEuclidean` or in experiments, never in shielding.py.
- MTW costs need their own `psi_hat`; the generic fallback is psi(rep y) - L * rad(y), with L bounding the y-Lipschitz constant of psi on the cell (e.g. via sup |D_xy c| * |x_A - x_s|).
- The c-exponential (charts) is an optional cost method, used for analysis and plots, not by the solver.
